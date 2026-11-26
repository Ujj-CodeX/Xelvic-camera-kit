package com.example.engine.render

import android.graphics.Bitmap
import android.opengl.EGL14
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.opengl.GLUtils
import com.example.engine.capture.CameraInputSurface
import com.example.engine.capture.FrameRing
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.photo.OffscreenRenderer
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * CompositionMode (README Section 11.5 & 21 Step 7):
 * - TRANSPARENT_OVERLAY: Initial architecture (`CameraX PreviewView + Transparent GL Surface`).
 *   PreviewView renders the camera feed; the transparent GL surface renders effects over PreviewView.
 * - SINGLE_SCENE_GL: Single-scene composition path in which GL imports the camera texture and
 *   renders camera background (Pass 1) and effects (Passes 2–9) together. Used when post effects,
 *   camera-pixel blend modes, eye recolor, or video recording require reading the camera frame in GL,
 *   or when selected by the user/offset test.
 */
enum class CompositionMode {
    TRANSPARENT_OVERLAY,
    SINGLE_SCENE_GL
}

/**
 * Measurement status for the Mandatory Preview-to-Overlay Offset Test (README Section 11.5 & 21 Step 7).
 * Never invents measurements when physical device recording has not been performed.
 */
enum class OffsetMeasurementStatus {
    NOT_MEASURED,
    MEASURED_ALIGNED,
    MEASURED_DETACHED
}

/**
 * CompositionSurface separation interface (README Section 11.5 & 21 Step 7).
 *
 * Separates scene evaluation and render graph execution from the output composition surface
 * so that `CompiledEffect`, scene model, package format, and rendering semantics remain
 * 100% identical whether rendering to `CameraX PreviewView + Transparent GL Surface` or
 * `Single-Scene GL Composition`.
 */
interface CompositionSurface {
    val mode: CompositionMode
    val renderGraph: RenderGraph
    val sceneEvaluator: SceneEvaluator
    val isSurfaceReady: Boolean
    val surfaceWidth: Int
    val surfaceHeight: Int
    val displayRotationDegrees: Int
    val isFrontCamera: Boolean
    val contextLossRecoveryCount: Int
    val areGlResourcesAllocated: Boolean
    val glCreateCount: Int
    val glReleaseCount: Int

    var activeEffect: CompiledEffect?

    /**
     * Called on the GL thread when the surface/EGL context is created or recreated after context loss.
     */
    fun onSurfaceCreated(width: Int = 0, height: Int = 0)

    /**
     * Called on the GL thread when surface dimensions or display rotation change.
     */
    fun onSurfaceChanged(width: Int, height: Int, rotationDegrees: Int = displayRotationDegrees)

    /**
     * Called when the surface is destroyed.
     */
    fun onSurfaceDestroyed()

    /**
     * Explicitly notifies the composition surface of an EGL/OpenGL context loss and restores
     * all GPU resources (shaders, VBOs/IBOs, textures, GLB models) on the GL thread.
     */
    fun onContextLostAndRecreated()

    /**
     * Lifecycle hooks:
     * - `onPause()`: stops frame processing only (no GL teardown on short pause).
     * - `onStop()`: releases GL resources on the GL thread (guarded against double-free).
     * - `onStart()` / `onResume()`: resumes frame processing and resets active effect behavior state;
     *   GL resources are restored once via the single idempotent entry point (`onSurfaceCreated` or
     *   lazy first draw in `renderFrame`).
     */
    fun onPause()
    fun onStop()
    fun onStart()
    fun onResume()

    /**
     * Updates camera orientation and lens configuration and re-derives transforms via FacePoseAdapter.
     */
    fun onCameraConfigChanged(isFrontCamera: Boolean, rotationDegrees: Int)

    /**
     * Executes Pass 1 (composition surface clear or camera background draw) followed by
     * SignalBus, BehaviorRunner, SceneEvaluator, and Passes 2–9 in `RenderGraph` using
     * the identical `CompiledEffect` and `FaceState`.
     */
    fun renderFrame(
        faceState: FaceState,
        dtSec: Float,
        renderTimeSec: Float = 0f,
        nowMs: Long = 0L,
        governor: PerformanceGovernor? = null
    )

    /**
     * Releases all GL resources owned by this composition surface on the GL thread.
     */
    fun release()
}

/**
 * Abstract base implementation sharing lifecycle, context recovery, and FacePoseAdapter transform
 * derivation across both `TransparentOverlayCompositionSurface` and `SingleSceneCompositionSurface`.
 */
abstract class BaseCompositionSurface(
    override val mode: CompositionMode,
    override val renderGraph: RenderGraph,
    override val sceneEvaluator: SceneEvaluator = SceneEvaluator(maxNodes = 64)
) : CompositionSurface {

    @Volatile
    override var isSurfaceReady: Boolean = false
        protected set

    @Volatile
    override var surfaceWidth: Int = 1080
        protected set

    @Volatile
    override var surfaceHeight: Int = 1920
        protected set

    @Volatile
    override var displayRotationDegrees: Int = 0
        protected set

    @Volatile
    override var isFrontCamera: Boolean = true
        protected set

    @Volatile
    override var contextLossRecoveryCount: Int = 0
        protected set

    @Volatile
    override var areGlResourcesAllocated: Boolean = false
        protected set

    @Volatile
    var allocatedContextGeneration: Int = -1
        protected set

    @Volatile
    override var glCreateCount: Int = 0
        protected set

    @Volatile
    override var glReleaseCount: Int = 0
        protected set

    @Volatile
    var isPaused: Boolean = false
        protected set

    @Volatile
    var isStopped: Boolean = false
        protected set

    override var activeEffect: CompiledEffect? = null

    protected var hasInitializedOnce: Boolean = false

    /**
     * Single idempotent entry point for GL resource creation/restoration on the GL thread.
     * Invoked from [onSurfaceCreated] and lazily on the first draw in [renderFrame]:
     * - If GL resources already exist for the current `resourceOwner.contextGeneration`, skips immediately
     *   (preventing duplicate creation across `onResume`, `onSurfaceCreated`, and `renderFrame`).
     * - Otherwise creates or restores `renderGraph` and surface GL resources once and records the
     *   current `contextGeneration`.
     */
    fun ensureGlResourcesOnGlThread(): Boolean {
        renderGraph.resourceOwner.bindGlThread()
        val currentGen = renderGraph.resourceOwner.contextGeneration
        if (areGlResourcesAllocated &&
            allocatedContextGeneration == currentGen &&
            renderGraph.areGlResourcesAllocated &&
            renderGraph.allocatedContextGeneration == currentGen
        ) {
            return false
        }

        if (hasInitializedOnce) {
            contextLossRecoveryCount++
            val isActualContextLoss = !renderGraph.areGlResourcesAllocated ||
                renderGraph.allocatedContextGeneration != currentGen
            if (isActualContextLoss) {
                renderGraph.onContextRecreated(activeEffect)
            } else {
                onReleaseSurfaceGlResources()
            }
            onInitSurfaceGlResources(isContextRecovery = true)
        } else {
            renderGraph.init()
            onInitSurfaceGlResources(isContextRecovery = false)
            hasInitializedOnce = true
        }

        val finalGen = renderGraph.resourceOwner.contextGeneration
        areGlResourcesAllocated = true
        allocatedContextGeneration = finalGen
        glCreateCount++
        return true
    }

    /**
     * Idempotent GL resource release on the GL thread (used on `ON_STOP` and `release()`).
     * Guaranteed never to double-free if called multiple times.
     */
    fun releaseGlResourcesOnGlThread(): Boolean {
        if (!areGlResourcesAllocated && !renderGraph.areGlResourcesAllocated) {
            return false
        }
        renderGraph.resourceOwner.bindGlThread()
        onReleaseSurfaceGlResources()
        renderGraph.release()
        areGlResourcesAllocated = false
        allocatedContextGeneration = -1
        glReleaseCount++
        return true
    }

    override fun onSurfaceCreated(width: Int, height: Int) {
        renderGraph.resourceOwner.bindGlThread()
        ensureGlResourcesOnGlThread()
        if (width > 0 && height > 0) {
            surfaceWidth = width
            surfaceHeight = height
            renderGraph.resize(width, height)
        }
        isSurfaceReady = true
    }

    override fun onSurfaceChanged(width: Int, height: Int, rotationDegrees: Int) {
        renderGraph.resourceOwner.bindGlThread()
        surfaceWidth = width.coerceAtLeast(1)
        surfaceHeight = height.coerceAtLeast(1)
        displayRotationDegrees = FacePoseAdapter.normalizeRotationDegrees(rotationDegrees)
        renderGraph.resize(surfaceWidth, surfaceHeight)
        isSurfaceReady = true
    }

    override fun onSurfaceDestroyed() {
        isSurfaceReady = false
        areGlResourcesAllocated = false
        allocatedContextGeneration = -1
    }

    override fun onContextLostAndRecreated() {
        renderGraph.resourceOwner.bindGlThread()
        areGlResourcesAllocated = false
        allocatedContextGeneration = -1
        hasInitializedOnce = true
        contextLossRecoveryCount++
        renderGraph.onContextRecreated(activeEffect)
        onInitSurfaceGlResources(isContextRecovery = true)
        if (surfaceWidth > 0 && surfaceHeight > 0) {
            renderGraph.resize(surfaceWidth, surfaceHeight)
        }
        areGlResourcesAllocated = true
        allocatedContextGeneration = renderGraph.resourceOwner.contextGeneration
        glCreateCount++
        isSurfaceReady = true
    }

    override fun onPause() {
        // Short pause (ON_PAUSE): stop frame processing only; do NOT release GL resources
        isPaused = true
    }

    override fun onStop() {
        // Long/background stop (ON_STOP): release GL resources on the GL thread
        isPaused = true
        isStopped = true
        releaseGlResourcesOnGlThread()
    }

    override fun onStart() {
        isStopped = false
        if (surfaceWidth > 0 && surfaceHeight > 0 && hasInitializedOnce) {
            isSurfaceReady = true
        }
    }

    override fun onResume() {
        val wasPausedOrStopped = isPaused || isStopped
        isStopped = false
        isPaused = false
        if (wasPausedOrStopped) {
            // Reset behavior/clip/trigger/spring state on resume after a pause
            activeEffect?.reset()
        }
        if (surfaceWidth > 0 && surfaceHeight > 0 && hasInitializedOnce) {
            isSurfaceReady = true
        }
    }

    override fun onCameraConfigChanged(isFrontCamera: Boolean, rotationDegrees: Int) {
        this.isFrontCamera = isFrontCamera
        this.displayRotationDegrees = FacePoseAdapter.normalizeRotationDegrees(rotationDegrees)
    }

    override fun renderFrame(
        faceState: FaceState,
        dtSec: Float,
        renderTimeSec: Float,
        nowMs: Long,
        governor: PerformanceGovernor?
    ) {
        if (isPaused || isStopped || !isSurfaceReady) return
        renderGraph.resourceOwner.bindGlThread()

        // Lazy first draw / post-stop restoration via the single idempotent entry point
        ensureGlResourcesOnGlThread()

        // Preserve per-frame isFrontCamera and projectionParams carried on FaceState so a frame
        // captured before a lens switch never uses the new camera's values (only fall back to
        // surface's isFrontCamera when rendering FaceState.EMPTY).
        val effectiveFaceState = if (faceState === FaceState.EMPTY && faceState.isFrontCamera != isFrontCamera) {
            faceState.copy(isFrontCamera = isFrontCamera)
        } else {
            faceState
        }

        // Pass 1: Surface-specific background handling (transparent clear vs single-scene camera pass)
        executePass1Background(effectiveFaceState)

        // Scene evaluation & Passes 2–9 using identical CompiledEffect and FaceState
        val effect = activeEffect
        if (effect != null) {
            if (governor != null) {
                governor.packageMinTier = effect.minTier
            }
            val requestedTier = governor?.currentTier ?: when (effect.activeTier) {
                "B" -> PerformanceGovernor.Tier.B
                "C" -> PerformanceGovernor.Tier.C
                else -> PerformanceGovernor.Tier.A
            }
            val effectiveTier = TierPolicy.resolveEffectiveTier(
                effect = effect,
                requestedTier = requestedTier,
                governorMaxTextureSize = governor?.maxTextureSize ?: 1024
            )
            effect.activeTier = effectiveTier.tierId

            // 1. Evaluate SignalBus in topological order into float slots
            effect.signalBus.evaluate(effectiveFaceState, renderTimeSec)

            // 2. Step Behaviors (triggers -> clips -> maps -> springs -> lifecycle fade)
            effect.behaviorRunner.step(
                dt = dtSec,
                signalBus = effect.signalBus,
                faceState = effectiveFaceState,
                slots = effect.propertySlots,
                nowMs = nowMs,
                disabledBehaviors = effectiveTier.disabledBehaviors,
                hiddenNodes = effectiveTier.hideNodes
            )

            // 3. Evaluate Scene Transforms & Anchors
            sceneEvaluator.evaluate(effect, effectiveFaceState, effect.propertySlots)

            // In TRANSPARENT_OVERLAY mode (Path A), fullscreen post nodes cannot read the underlying
            // PreviewView hardware surface and must be skipped in live preview only.
            if (mode == CompositionMode.TRANSPARENT_OVERLAY) {
                val nodes = effect.nodes
                for (i in nodes.indices) {
                    if (nodes[i].type == "post" && i < sceneEvaluator.isNodeVisible.size) {
                        sceneEvaluator.isNodeVisible[i] = false
                    }
                }
            } else if (mode == CompositionMode.SINGLE_SCENE_GL && !effectiveFaceState.isVisible) {
                // In SINGLE_SCENE_GL mode (Path B), global camera post-processing nodes (colorGrade,
                // colorMatrix, vignette, blur) and screen-space overlays do not depend on face landmarks
                // and remain active on the live camera scene even when no face is currently tracked.
                val nodes = effect.nodes
                val tierOverride = effect.tierOverrides[effect.activeTier]
                for (i in nodes.indices) {
                    val node = nodes[i]
                    val isGlobalPost = node.type == "post" && (
                        node.postEffect == "colorGrade" ||
                            node.postEffect == "colorMatrix" ||
                            node.postEffect == "vignette" ||
                            node.postEffect == "blur" ||
                            node.postEffect == "softFocus" ||
                            node.postEffect == "lut" ||
                            node.postEffect == "grain" ||
                            node.postEffect == "sharpen"
                        )
                    if ((isGlobalPost || node.type == "overlay") &&
                        TierPolicy.isNodeVisibleInTier(i, node.baseVisible, tierOverride) &&
                        i < sceneEvaluator.isNodeVisible.size
                    ) {
                        effect.propertySlots.visible[i] = 1f
                        effect.propertySlots.opacity[i] = node.baseOpacity
                        sceneEvaluator.isNodeVisible[i] = node.baseOpacity > 0.001f
                    }
                }
            }

            // 4. Execute multi-pass rendering (Passes 2–9) with DeterministicFrameContext
            val frameCtx = DeterministicFrameContext.forPreview(
                packageId = effect.packageId,
                renderTimeSec = renderTimeSec,
                frameIndex = nowMs,
                explicitSeed = effect.photoPolicy.seed
            )
            renderGraph.render(
                effect = effect,
                faceState = effectiveFaceState,
                evaluator = sceneEvaluator,
                viewportWidth = surfaceWidth,
                viewportHeight = surfaceHeight,
                governor = governor,
                frameContext = frameCtx
            )
            onAfterRenderGraph(effect, effectiveFaceState)
        } else {
            // Even when no effect is active, drain pending GL deletions and staged preloads
            renderGraph.resourceOwner.drainPendingDeletions()
            renderGraph.textures.stepFrameUploads(maxUploadsPerFrame = 1)
            onAfterRenderGraph(null, effectiveFaceState)
        }
    }

    override fun release() {
        releaseGlResourcesOnGlThread()
        isSurfaceReady = false
    }

    protected abstract fun onInitSurfaceGlResources(isContextRecovery: Boolean)
    protected abstract fun executePass1Background(faceState: FaceState)
    protected open fun onAfterRenderGraph(effect: CompiledEffect?, faceState: FaceState) {}
    protected abstract fun onReleaseSurfaceGlResources()
}

/**
 * TransparentOverlayCompositionSurface (README Section 11.5 & 21 Step 7):
 * Initial architecture: `CameraX PreviewView + Transparent GL Surface`.
 * Pass 1 clears the color buffer to transparent `(0, 0, 0, 0)` so the underlying `PreviewView`
 * remains visible while Passes 2–8 composite 3D/2D scene nodes with premultiplied alpha on top.
 */
class TransparentOverlayCompositionSurface(
    renderGraph: RenderGraph = RenderGraph(),
    sceneEvaluator: SceneEvaluator = SceneEvaluator(maxNodes = 64)
) : BaseCompositionSurface(CompositionMode.TRANSPARENT_OVERLAY, renderGraph, sceneEvaluator) {

    override fun onInitSurfaceGlResources(isContextRecovery: Boolean) {
        if (hasActiveGlContext()) {
            GLES30.glClearColor(0f, 0f, 0f, 0f)
        }
    }

    override fun executePass1Background(faceState: FaceState) {
        if (hasActiveGlContext()) {
            GLES30.glColorMask(true, true, true, true)
            GLES30.glClearColor(0f, 0f, 0f, 0f)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        }
    }

    override fun onReleaseSurfaceGlResources() {
        // No extra camera background textures owned in transparent overlay mode
    }

    private fun hasActiveGlContext(): Boolean {
        return try {
            val ctx = EGL14.eglGetCurrentContext()
            ctx != null && ctx != EGL14.EGL_NO_CONTEXT
        } catch (t: Throwable) {
            false
        }
    }
}

/**
 * SingleSceneCompositionSurface (README Section 11.5 & 21 Step 7/10, Chunk B0):
 * Single-scene composition path (Path B) in which GL imports the camera texture (2D RGBA or EXTERNAL_OES
 * via [CameraInputSurface] and [FrameRing]) and renders:
 * `CAMERA_FRAME -> CAMERA_PIXEL_PASSES -> CLEAR_DEPTH -> OCCLUDERS -> OPAQUE_NODES -> TRANSPARENT_NODES -> POST_NODES -> OVERLAY_NODES -> FINAL_OUTPUT`.
 *
 * Single-Owner Input Policy (Chunk B0):
 * When [cameraInputSurface] is bound, [CameraInputSurface] is the single owner of `mirrorH`, `cropCenter`,
 * and `rotationDegrees` in `uTexMatrix`. Therefore `uMirrorX` in the background shader is `0` so the front
 * camera is never double-mirrored.
 */
class SingleSceneCompositionSurface(
    renderGraph: RenderGraph = RenderGraph(),
    sceneEvaluator: SceneEvaluator = SceneEvaluator(maxNodes = 64)
) : BaseCompositionSurface(CompositionMode.SINGLE_SCENE_GL, renderGraph, sceneEvaluator) {

    companion object {
        private const val VS_CAMERA_BG = """#version 300 es
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec2 aTexCoord;
            uniform mat4 uTexMatrix;
            uniform int uMirrorX;
            uniform int uIsExternalOes;
            out vec2 vTexCoord;
            void main() {
                vec2 uv = aTexCoord;
                if (uIsExternalOes == 0) {
                    uv.y = 1.0 - uv.y;
                }
                if (uMirrorX == 1) {
                    uv.x = 1.0 - uv.x;
                }
                vTexCoord = (uTexMatrix * vec4(uv, 0.0, 1.0)).xy;
                gl_Position = vec4(aPosition, 0.0, 1.0);
            }
        """

        private const val FS_CAMERA_BG_2D = """#version 300 es
            precision mediump float;
            in vec2 vTexCoord;
            uniform sampler2D uCameraTex;
            out vec4 fragColor;
            void main() {
                fragColor = texture(uCameraTex, vTexCoord);
            }
        """

        private const val FS_CAMERA_BG_OES = """#version 300 es
            #extension GL_OES_EGL_image_external_essl3 : require
            precision mediump float;
            in vec2 vTexCoord;
            uniform samplerExternalOES uCameraTex;
            out vec4 fragColor;
            void main() {
                fragColor = texture(uCameraTex, vTexCoord);
            }
        """
    }

    var cameraTextureId: Int = 0
        private set
    var isExternalOesTexture: Boolean = false
        private set
    var pass1ExecutionCount: Int = 0
        private set

    var cameraInputSurface: CameraInputSurface? = null
        private set

    private var ownedUploadedCameraTexId: Int = 0
    private var ownedOesCameraTexId: Int = 0
    private var cameraFrameBitmap: Bitmap? = null
    private var cameraFrameAlreadyMirrored: Boolean = true

    private var sceneFbo: Int = 0
    private var sceneColorTex: Int = 0
    private var sceneDepthRbo: Int = 0
    private var sceneFboWidth: Int = 0
    private var sceneFboHeight: Int = 0

    val currentSceneFboId: Int
        get() = sceneFbo

    val currentSceneColorTexId: Int
        get() = sceneColorTex

    val currentSceneFboWidth: Int
        get() = sceneFboWidth

    val currentSceneFboHeight: Int
        get() = sceneFboHeight

    var videoRecorder: com.example.engine.capture.VideoRecorderPipeline? = null

    private val identityMatrix16 = FloatArray(16).apply {
        this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
    }

    /**
     * Last uniform `uMirrorX` value (`0` or `1`) applied in `executePass1Background` (exposed for unit tests).
     */
    var lastShaderMirrorXUniform: Int = 0
        private set

    /**
     * Last camera frame slot read from [CameraInputSurface] / [FrameRing] during Pass 1.
     */
    var lastReadCameraFrameEntry: FrameRing.FrameSlot? = null
        private set

    val hasCameraSceneInput: Boolean
        get() = cameraTextureId != 0 ||
            cameraFrameBitmap != null ||
            (cameraInputSurface != null && (cameraInputSurface!!.hasValidFrame || cameraInputSurface!!.isAttachedToGl))

    val isCameraTextureOk: Boolean
        get() = (cameraInputSurface != null && cameraInputSurface!!.hasValidFrame) ||
            cameraTextureId != 0 ||
            cameraFrameBitmap != null

    val latestCameraSensorTimestampNs: Long
        get() = cameraInputSurface?.lastSensorTimestampNs ?: 0L

    var lastComposedSceneBitmap: Bitmap? = null
        private set

    private val cpuSceneCompositor by lazy { OffscreenRenderer() }

    private var bgProgram2D: Int = 0
    private var bgProgramOes: Int = 0
    private var bgQuadVao: Int = 0
    private var bgQuadVbo: Int = 0
    private val texMatrix = FloatArray(16).apply {
        this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
    }

    fun getCurrentTexMatrix(): FloatArray = texMatrix.clone()

    /**
     * Binds a [CameraInputSurface] (Path B single-owner camera input with [FrameRing]).
     */
    fun bindCameraInputSurface(inputSurface: CameraInputSurface?) {
        this.cameraInputSurface = inputSurface
        if (inputSurface != null) {
            // CameraInputSurface is the single owner of mirrorH, cropCenter, and rotationDegrees
            cameraFrameAlreadyMirrored = true
            inputSurface.updateOptions(
                inputSurface.options.copy(
                    mirrorH = isFrontCamera,
                    viewportWidth = surfaceWidth,
                    viewportHeight = surfaceHeight
                )
            )
            if (inputSurface.boundTextureId != 0) {
                cameraTextureId = inputSurface.boundTextureId
                isExternalOesTexture = true
            } else if (hasActiveGlContext()) {
                ensureOesCameraTextureAttachedOnGlThread(inputSurface)
            }
        }
    }

    fun bindCameraTexture(textureId: Int, isExternalOes: Boolean = false, transformMatrix: FloatArray? = null) {
        cameraTextureId = textureId
        isExternalOesTexture = isExternalOes
        if (transformMatrix != null && transformMatrix.size == 16) {
            System.arraycopy(transformMatrix, 0, texMatrix, 0, 16)
        }
    }

    /**
     * Imports a camera frame Bitmap into the single-scene composition surface.
     * On an active GL context, uploads to a `GL_TEXTURE_2D` camera texture for Pass 1;
     * also updates `lastComposedSceneBitmap` combining Pass 1 camera + Passes 2–8 scene & post nodes.
     */
    fun bindCameraFrameBitmap(bitmap: Bitmap?, alreadyMirroredForFrontCamera: Boolean = true) {
        cameraFrameBitmap = bitmap
        cameraFrameAlreadyMirrored = alreadyMirroredForFrontCamera
        if (bitmap != null && hasActiveGlContext()) {
            renderGraph.resourceOwner.bindGlThread()
            if (ownedUploadedCameraTexId == 0) {
                val texs = IntArray(1)
                GLES30.glGenTextures(1, texs, 0)
                ownedUploadedCameraTexId = texs[0]
                if (ownedUploadedCameraTexId != 0) {
                    renderGraph.resourceOwner.trackTexture(ownedUploadedCameraTexId)
                }
            }
            if (ownedUploadedCameraTexId != 0) {
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, ownedUploadedCameraTexId)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
                GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
                cameraTextureId = ownedUploadedCameraTexId
                isExternalOesTexture = false
            }
        }
    }

    override fun onSurfaceChanged(width: Int, height: Int, rotationDegrees: Int) {
        super.onSurfaceChanged(width, height, rotationDegrees)
        cameraInputSurface?.let { input ->
            input.updateOptions(
                input.options.copy(
                    mirrorH = isFrontCamera,
                    viewportWidth = surfaceWidth,
                    viewportHeight = surfaceHeight
                )
            )
        }
    }

    override fun onCameraConfigChanged(isFrontCamera: Boolean, rotationDegrees: Int) {
        super.onCameraConfigChanged(isFrontCamera, rotationDegrees)
        cameraInputSurface?.let { input ->
            input.updateOptions(
                input.options.copy(
                    mirrorH = isFrontCamera,
                    viewportWidth = surfaceWidth,
                    viewportHeight = surfaceHeight
                )
            )
        }
    }

    override fun onPause() {
        super.onPause()
        cameraInputSurface?.let { input ->
            if (input.options.pausable) {
                input.setPaused(true)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        cameraInputSurface?.setPaused(false)
    }

    private fun ensureOesCameraTextureAttachedOnGlThread(inputSurface: CameraInputSurface) {
        if (!hasActiveGlContext()) return
        renderGraph.resourceOwner.bindGlThread()
        if (ownedOesCameraTexId == 0) {
            val texs = IntArray(1)
            GLES30.glGenTextures(1, texs, 0)
            ownedOesCameraTexId = texs[0]
            if (ownedOesCameraTexId != 0) {
                renderGraph.resourceOwner.trackTexture(ownedOesCameraTexId)
                GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, ownedOesCameraTexId)
                GLES30.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES30.GL_TEXTURE_MIN_FILTER,
                    GLES30.GL_LINEAR
                )
                GLES30.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES30.GL_TEXTURE_MAG_FILTER,
                    GLES30.GL_LINEAR
                )
                GLES30.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES30.GL_TEXTURE_WRAP_S,
                    GLES30.GL_CLAMP_TO_EDGE
                )
                GLES30.glTexParameteri(
                    GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                    GLES30.GL_TEXTURE_WRAP_T,
                    GLES30.GL_CLAMP_TO_EDGE
                )
                GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0)
            }
        }
        if (ownedOesCameraTexId != 0 && (!inputSurface.isAttachedToGl || inputSurface.boundTextureId != ownedOesCameraTexId)) {
            inputSurface.attachToGlContext(ownedOesCameraTexId)
        }
        if (ownedOesCameraTexId != 0) {
            cameraTextureId = ownedOesCameraTexId
            isExternalOesTexture = true
        }
    }

    private fun ensureSceneFbo(width: Int, height: Int) {
        if (!hasActiveGlContext()) return
        val w = width.coerceAtLeast(1)
        val h = height.coerceAtLeast(1)
        if (sceneFbo != 0 && sceneColorTex != 0 && sceneFboWidth == w && sceneFboHeight == h) {
            return
        }
        releaseSceneFbo()
        try {
            val fbos = IntArray(1)
            val texs = IntArray(1)
            val rbos = IntArray(1)
            GLES30.glGenFramebuffers(1, fbos, 0)
            GLES30.glGenTextures(1, texs, 0)
            GLES30.glGenRenderbuffers(1, rbos, 0)
            sceneFbo = fbos[0]
            sceneColorTex = texs[0]
            sceneDepthRbo = rbos[0]
            if (sceneFbo == 0 || sceneColorTex == 0) {
                releaseSceneFbo()
                return
            }
            renderGraph.resourceOwner.trackFramebuffer(sceneFbo)
            renderGraph.resourceOwner.trackTexture(sceneColorTex)
            if (sceneDepthRbo != 0) {
                renderGraph.resourceOwner.trackRenderbuffer(sceneDepthRbo)
            }

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sceneColorTex)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8,
                w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, sceneFbo)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D,
                sceneColorTex,
                0
            )
            if (sceneDepthRbo != 0) {
                GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, sceneDepthRbo)
                GLES30.glRenderbufferStorage(GLES30.GL_RENDERBUFFER, GLES30.GL_DEPTH_COMPONENT24, w, h)
                GLES30.glFramebufferRenderbuffer(
                    GLES30.GL_FRAMEBUFFER,
                    GLES30.GL_DEPTH_ATTACHMENT,
                    GLES30.GL_RENDERBUFFER,
                    sceneDepthRbo
                )
                GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, 0)
            }
            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                releaseSceneFbo()
                return
            }
            sceneFboWidth = w
            sceneFboHeight = h
        } catch (_: Throwable) {
            releaseSceneFbo()
        }
    }

    private fun releaseSceneFbo() {
        if (sceneFbo != 0) {
            renderGraph.resourceOwner.queueDeleteFramebuffer(sceneFbo)
            sceneFbo = 0
        }
        if (sceneColorTex != 0) {
            renderGraph.resourceOwner.queueDeleteTexture(sceneColorTex)
            sceneColorTex = 0
        }
        if (sceneDepthRbo != 0) {
            if (hasActiveGlContext()) {
                try {
                    GLES30.glDeleteRenderbuffers(1, intArrayOf(sceneDepthRbo), 0)
                } catch (_: Throwable) {
                }
            }
            sceneDepthRbo = 0
        }
        sceneFboWidth = 0
        sceneFboHeight = 0
    }

    override fun onInitSurfaceGlResources(isContextRecovery: Boolean) {
        if (isContextRecovery) {
            if (cameraTextureId == ownedOesCameraTexId || cameraTextureId == ownedUploadedCameraTexId) {
                cameraTextureId = 0
            }
            sceneFbo = 0
            sceneColorTex = 0
            sceneDepthRbo = 0
            sceneFboWidth = 0
            sceneFboHeight = 0
            bgProgram2D = 0
            bgProgramOes = 0
            bgQuadVao = 0
            bgQuadVbo = 0
            ownedOesCameraTexId = 0
            ownedUploadedCameraTexId = 0
            cameraInputSurface?.detachFromGlContext()
        }
        if (!hasActiveGlContext()) return
        buildCameraBackgroundResources()
        cameraInputSurface?.let { input ->
            ensureOesCameraTextureAttachedOnGlThread(input)
        }
        cameraFrameBitmap?.let { bmp ->
            bindCameraFrameBitmap(bmp, cameraFrameAlreadyMirrored)
        }
    }

    override fun executePass1Background(faceState: FaceState) {
        pass1ExecutionCount++
        val inputSurface = cameraInputSurface
        if (inputSurface != null) {
            if (!inputSurface.isAttachedToGl && hasActiveGlContext()) {
                ensureOesCameraTextureAttachedOnGlThread(inputSurface)
            }
            if (surfaceWidth > 0 && surfaceHeight > 0 &&
                (inputSurface.options.viewportWidth != surfaceWidth ||
                    inputSurface.options.viewportHeight != surfaceHeight ||
                    inputSurface.options.mirrorH != faceState.isFrontCamera)
            ) {
                inputSurface.updateOptions(
                    inputSurface.options.copy(
                        mirrorH = faceState.isFrontCamera,
                        viewportWidth = surfaceWidth,
                        viewportHeight = surfaceHeight
                    )
                )
            }
            val targetTrackerTsNs = if (inputSurface is com.example.engine.capture.SurfaceTextureCameraInputSurface &&
                inputSurface.syncFrameRingToTrackerTimestamp &&
                faceState.poseTimestampNs > 0L
            ) {
                faceState.poseTimestampNs
            } else {
                0L
            }
            val entry = inputSurface.readFrame(
                outTexMatrix = texMatrix,
                targetTimestampNs = targetTrackerTsNs,
                useAlignedClock = true
            )
            lastReadCameraFrameEntry = entry
            if (entry != null && entry.textureId != 0) {
                cameraTextureId = entry.textureId
                isExternalOesTexture = entry.isExternalOes
                // CameraInputSurface is the single owner of mirrorH, cropCenter, and rotation
                cameraFrameAlreadyMirrored = true
            }
        }

        val shouldMirrorInShader = if (inputSurface != null) {
            false
        } else {
            faceState.isFrontCamera && !cameraFrameAlreadyMirrored
        }
        lastShaderMirrorXUniform = if (shouldMirrorInShader) 1 else 0

        if (!hasActiveGlContext()) return
        ensureSceneFbo(surfaceWidth, surfaceHeight)
        if (sceneFbo != 0) {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, sceneFbo)
            GLES30.glViewport(0, 0, surfaceWidth.coerceAtLeast(1), surfaceHeight.coerceAtLeast(1))
        }

        GLES30.glColorMask(true, true, true, true)
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        renderCameraBackgroundToCurrentFbo()
    }

    /**
     * Draws the active camera texture quad into the currently bound framebuffer using
     * the latest `texMatrix` and `lastShaderMirrorXUniform`.
     */
    fun renderCameraBackgroundToCurrentFbo() {
        if (!hasActiveGlContext()) return
        val activeProgram = if (isExternalOesTexture) bgProgramOes else bgProgram2D
        if (cameraTextureId != 0 && activeProgram != 0 && bgQuadVao != 0) {
            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            GLES30.glDepthMask(false)
            GLES30.glDisable(GLES30.GL_BLEND)

            GLES30.glUseProgram(activeProgram)
            val uTexMatLoc = GLES30.glGetUniformLocation(activeProgram, "uTexMatrix")
            val uMirrorLoc = GLES30.glGetUniformLocation(activeProgram, "uMirrorX")
            val uIsOesLoc = GLES30.glGetUniformLocation(activeProgram, "uIsExternalOes")
            val uCamTexLoc = GLES30.glGetUniformLocation(activeProgram, "uCameraTex")

            GLES30.glUniformMatrix4fv(uTexMatLoc, 1, false, texMatrix, 0)
            GLES30.glUniform1i(uMirrorLoc, lastShaderMirrorXUniform)
            GLES30.glUniform1i(uIsOesLoc, if (isExternalOesTexture) 1 else 0)

            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            val target = if (isExternalOesTexture) GLES11Ext.GL_TEXTURE_EXTERNAL_OES else GLES30.GL_TEXTURE_2D
            GLES30.glBindTexture(target, cameraTextureId)
            GLES30.glUniform1i(uCamTexLoc, 0)

            GLES30.glBindVertexArray(bgQuadVao)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 6)
            GLES30.glBindVertexArray(0)
            GLES30.glBindTexture(target, 0)
        }
    }

    override fun onAfterRenderGraph(effect: CompiledEffect?, faceState: FaceState) {
        if (hasActiveGlContext() && sceneFbo != 0 && sceneColorTex != 0 && bgProgram2D != 0 && bgQuadVao != 0) {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, surfaceWidth.coerceAtLeast(1), surfaceHeight.coerceAtLeast(1))
            GLES30.glColorMask(true, true, true, true)
            val waitingForFirstOesFrame =
                cameraInputSurface != null && !cameraInputSurface!!.hasValidFrame && cameraFrameBitmap == null
            if (waitingForFirstOesFrame) {
                // Keep FBO 0 transparent until the first OES frame arrives so PreviewView underneath stays visible
                GLES30.glClearColor(0f, 0f, 0f, 0f)
                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
            } else {
                GLES30.glDisable(GLES30.GL_DEPTH_TEST)
                GLES30.glDepthMask(false)
                GLES30.glDisable(GLES30.GL_BLEND)

                GLES30.glUseProgram(bgProgram2D)
                val uTexMatLoc = GLES30.glGetUniformLocation(bgProgram2D, "uTexMatrix")
                val uMirrorLoc = GLES30.glGetUniformLocation(bgProgram2D, "uMirrorX")
                val uIsOesLoc = GLES30.glGetUniformLocation(bgProgram2D, "uIsExternalOes")
                val uCamTexLoc = GLES30.glGetUniformLocation(bgProgram2D, "uCameraTex")

                GLES30.glUniformMatrix4fv(uTexMatLoc, 1, false, identityMatrix16, 0)
                GLES30.glUniform1i(uMirrorLoc, 0)
                // sceneColorTex is an OpenGL FBO texture with bottom-left origin (no Y-flip needed)
                GLES30.glUniform1i(uIsOesLoc, 1)

                GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sceneColorTex)
                GLES30.glUniform1i(uCamTexLoc, 0)

                GLES30.glBindVertexArray(bgQuadVao)
                GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 6)
                GLES30.glBindVertexArray(0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            }
        }

        val camBmp = cameraFrameBitmap
        if (camBmp != null) {
            if (effect != null && faceState.isVisible && faceState.fadeWeight >= 0.001f) {
                val out = cpuSceneCompositor.renderCpuFallback(
                    baseBitmap = camBmp,
                    effect = effect,
                    faceState = faceState,
                    evaluator = sceneEvaluator,
                    tierMaxTexSize = 1024
                )
                lastComposedSceneBitmap = out.bitmap
            } else {
                lastComposedSceneBitmap = camBmp.copy(Bitmap.Config.ARGB_8888, true)
            }
        }

        val recorder = videoRecorder
        if (recorder != null && recorder.state == com.example.engine.capture.VideoRecorderPipeline.State.RECORDING) {
            val sensorTsNs = lastReadCameraFrameEntry?.sensorTimestampNs?.takeIf { it > 0L }
                ?: latestCameraSensorTimestampNs.takeIf { it > 0L }
                ?: faceState.timestampNs.takeIf { it > 0L }
                ?: System.nanoTime()
            val fboW = if (sceneFboWidth > 0) sceneFboWidth else surfaceWidth.coerceAtLeast(1)
            val fboH = if (sceneFboHeight > 0) sceneFboHeight else surfaceHeight.coerceAtLeast(1)
            recorder.recordGlSceneFboFrame(
                sceneFboId = sceneFbo,
                sceneColorTexId = sceneColorTex,
                srcWidth = fboW,
                srcHeight = fboH,
                sensorTimestampNs = sensorTsNs
            )
        }
    }

    override fun onReleaseSurfaceGlResources() {
        cameraInputSurface?.detachFromGlContext()
        releaseSceneFbo()
        if (bgQuadVao != 0) {
            renderGraph.resourceOwner.queueDeleteVao(bgQuadVao)
            bgQuadVao = 0
        }
        if (bgQuadVbo != 0) {
            renderGraph.resourceOwner.queueDeleteBuffer(bgQuadVbo)
            bgQuadVbo = 0
        }
        if (bgProgram2D != 0) {
            renderGraph.resourceOwner.queueDeleteProgram(bgProgram2D)
            bgProgram2D = 0
        }
        if (bgProgramOes != 0) {
            renderGraph.resourceOwner.queueDeleteProgram(bgProgramOes)
            bgProgramOes = 0
        }
        if (ownedUploadedCameraTexId != 0) {
            renderGraph.resourceOwner.queueDeleteTexture(ownedUploadedCameraTexId)
            ownedUploadedCameraTexId = 0
        }
        if (ownedOesCameraTexId != 0) {
            renderGraph.resourceOwner.queueDeleteTexture(ownedOesCameraTexId)
            ownedOesCameraTexId = 0
        }
        if (renderGraph.resourceOwner.isOnGlThread()) {
            renderGraph.resourceOwner.drainPendingDeletions()
        }
    }

    private fun buildCameraBackgroundResources() {
        // Standard OpenGL bottom-left UV coordinates: (-1, -1) -> (0, 0), (+1, +1) -> (1, 1)
        val quad = floatArrayOf(
            -1f, -1f, 0f, 0f,
             1f, -1f, 1f, 0f,
             1f,  1f, 1f, 1f,
            -1f, -1f, 0f, 0f,
             1f,  1f, 1f, 1f,
            -1f,  1f, 0f, 1f
        )
        val vaos = IntArray(1); GLES30.glGenVertexArrays(1, vaos, 0)
        val vbos = IntArray(1); GLES30.glGenBuffers(1, vbos, 0)
        bgQuadVao = vaos[0]
        bgQuadVbo = vbos[0]

        renderGraph.resourceOwner.trackVao(bgQuadVao)
        renderGraph.resourceOwner.trackBuffer(bgQuadVbo)

        GLES30.glBindVertexArray(bgQuadVao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, bgQuadVbo)
        val fb = ByteBuffer.allocateDirect(quad.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        fb.put(quad).position(0)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, quad.size * 4, fb, GLES30.GL_STATIC_DRAW)

        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 4 * 4, 0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 4 * 4, 2 * 4)
        GLES30.glBindVertexArray(0)

        bgProgram2D = compileProgram(VS_CAMERA_BG, FS_CAMERA_BG_2D)
        if (bgProgram2D != 0) {
            renderGraph.resourceOwner.trackProgram(bgProgram2D)
        }
        bgProgramOes = compileProgram(VS_CAMERA_BG, FS_CAMERA_BG_OES)
        if (bgProgramOes != 0) {
            renderGraph.resourceOwner.trackProgram(bgProgramOes)
        }
    }

    private fun compileProgram(vsSource: String, fsSource: String): Int {
        val vs = compileShader(GLES30.GL_VERTEX_SHADER, vsSource)
        val fs = compileShader(GLES30.GL_FRAGMENT_SHADER, fsSource)
        if (vs == 0 || fs == 0) return 0
        val prog = GLES30.glCreateProgram()
        GLES30.glAttachShader(prog, vs)
        GLES30.glAttachShader(prog, fs)
        GLES30.glLinkProgram(prog)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, linked, 0)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        if (linked[0] == 0) {
            GLES30.glDeleteProgram(prog)
            return 0
        }
        return prog
    }

    private fun compileShader(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src)
        GLES30.glCompileShader(s)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            GLES30.glDeleteShader(s)
            return 0
        }
        return s
    }

    private fun hasActiveGlContext(): Boolean {
        return try {
            val ctx = EGL14.eglGetCurrentContext()
            ctx != null && ctx != EGL14.EGL_NO_CONTEXT
        } catch (t: Throwable) {
            false
        }
    }
}

/**
 * Mandatory Preview-to-Overlay Offset Test Harness & Surface Selector (README Section 11.5 & 21 Step 7).
 *
 * Implements:
 * 1. Face-attached calibration marker projection (`projectFaceMarkerNormalized`) so a marker attached to
 *    the tracked face can be rendered on the GL overlay during fast head movement.
 * 2. Simultaneous timestamp & screen-space alignment sample recording (`recordSample`).
 * 3. Evaluation of visible detachment and lag (`evaluate`), producing an explicit `CompositionDecision`
 *    without inventing measurements when physical-device recording has not yet been executed.
 * 4. Policy-based `CompositionSurface` selection (`selectCompositionMode`) that preserves
 *    `CameraX PreviewView + Transparent GL Surface` unless post effects, video recording, or a
 *    measured offset test requires migrating to `SINGLE_SCENE_GL`.
 */
class OffsetTestHarness {

    data class OffsetSample(
        val frameTimestampNs: Long,
        val overlayRenderTimestampNs: Long,
        val previewMarkerXNorm: Float,
        val previewMarkerYNorm: Float,
        val overlayMarkerXNorm: Float,
        val overlayMarkerYNorm: Float,
        val headSpeedNormalizedPerSec: Float
    ) {
        val lagMs: Float
            get() = ((overlayRenderTimestampNs - frameTimestampNs).coerceAtLeast(0L)) / 1_000_000f

        val spatialOffsetNorm: Float
            get() {
                val dx = overlayMarkerXNorm - previewMarkerXNorm
                val dy = overlayMarkerYNorm - previewMarkerYNorm
                return sqrt(dx * dx + dy * dy)
            }
    }

    data class CompositionDecision(
        val status: OffsetMeasurementStatus,
        val sampleCount: Int,
        val meanLagMs: Float?,
        val p95LagMs: Float?,
        val maxSpatialOffsetNorm: Float?,
        val visibleDetachmentDetected: Boolean,
        val recommendedMode: CompositionMode,
        val rationale: String
    )

    @Volatile
    var isMarkerEnabled: Boolean = false

    private val samples = CopyOnWriteArrayList<OffsetSample>()

    @Volatile
    var lastDecision: CompositionDecision = CompositionDecision(
        status = OffsetMeasurementStatus.NOT_MEASURED,
        sampleCount = 0,
        meanLagMs = null,
        p95LagMs = null,
        maxSpatialOffsetNorm = null,
        visibleDetachmentDetected = false,
        recommendedMode = CompositionMode.TRANSPARENT_OVERLAY,
        rationale = "NOT MEASURED on physical device: preserving default CameraX PreviewView + Transparent GL Surface per README Section 11.5."
    )
        private set

    fun clearSamples() {
        samples.clear()
        lastDecision = CompositionDecision(
            status = OffsetMeasurementStatus.NOT_MEASURED,
            sampleCount = 0,
            meanLagMs = null,
            p95LagMs = null,
            maxSpatialOffsetNorm = null,
            visibleDetachmentDetected = false,
            recommendedMode = CompositionMode.TRANSPARENT_OVERLAY,
            rationale = "NOT MEASURED on physical device: preserving default CameraX PreviewView + Transparent GL Surface per README Section 11.5."
        )
    }

    fun getSampleCount(): Int = samples.size

    /**
     * Projects a marker attached to the tracked face (`nose_tip` or `head_center`) into normalized
     * screen coordinates `[0..1, 0..1]` using `FacePoseAdapter` so both Preview and GL Overlay
     * alignment can be compared directly.
     */
    fun projectFaceMarkerNormalized(
        faceState: FaceState,
        viewportAspect: Float,
        localOffsetFwX: Float = 0f,
        localOffsetFwY: Float = 0f,
        localOffsetFwZ: Float = 0.35f
    ): FloatArray {
        val camPos = FacePoseAdapter.faceLocalToCamera(
            localX = localOffsetFwX,
            localY = localOffsetFwY,
            localZ = localOffsetFwZ,
            matrix = faceState.poseMatrix,
            fw = faceState.faceWidth
        )
        val viewAspect = viewportAspect.coerceAtLeast(0.1f)
        val fovYViewDeg = faceState.projectionParams.visibleViewFovYDeg(viewAspect)
        val proj = FacePoseAdapter.createCameraProjection(
            fovYDeg = fovYViewDeg,
            aspectRatio = viewAspect,
            near = faceState.projectionParams.near,
            far = faceState.projectionParams.far,
            frontFacing = faceState.isFrontCamera
        )
        val clipX = proj[0] * camPos[0] + proj[4] * camPos[1] + proj[8] * camPos[2] + proj[12]
        val clipY = proj[1] * camPos[0] + proj[5] * camPos[1] + proj[9] * camPos[2] + proj[13]
        val clipW = (proj[3] * camPos[0] + proj[7] * camPos[1] + proj[11] * camPos[2] + proj[15]).let {
            if (abs(it) < 1e-5f) 1f else it
        }
        val ndcX = (clipX / clipW).coerceIn(-1f, 1f)
        val ndcY = (clipY / clipW).coerceIn(-1f, 1f)
        val normX = (ndcX + 1f) * 0.5f
        val normY = (1f - ndcY) * 0.5f
        return floatArrayOf(normX, normY)
    }

    /**
     * Records a measured alignment sample from simultaneous preview and GL overlay tracking.
     */
    fun recordSample(sample: OffsetSample) {
        samples.add(sample)
    }

    /**
     * Evaluates recorded samples without inventing thresholds when no samples exist.
     * Caller provides explicit detachment criteria when evaluating measured recordings.
     */
    fun evaluate(detachmentSpatialThresholdNorm: Float, detachmentLagThresholdMs: Float): CompositionDecision {
        val snapshot = samples.toList()
        if (snapshot.isEmpty()) {
            val unmeasured = CompositionDecision(
                status = OffsetMeasurementStatus.NOT_MEASURED,
                sampleCount = 0,
                meanLagMs = null,
                p95LagMs = null,
                maxSpatialOffsetNorm = null,
                visibleDetachmentDetected = false,
                recommendedMode = CompositionMode.TRANSPARENT_OVERLAY,
                rationale = "NOT MEASURED: 0 physical device recording samples available; preserving CameraX PreviewView + Transparent GL Surface."
            )
            lastDecision = unmeasured
            return unmeasured
        }

        val lags = snapshot.map { it.lagMs }.sorted()
        val meanLag = lags.average().toFloat()
        val p95Index = ((lags.size - 1) * 0.95f).toInt().coerceIn(0, lags.size - 1)
        val p95Lag = lags[p95Index]
        val maxOffset = snapshot.maxOf { it.spatialOffsetNorm }

        val detached = maxOffset > detachmentSpatialThresholdNorm || p95Lag > detachmentLagThresholdMs
        val status = if (detached) {
            OffsetMeasurementStatus.MEASURED_DETACHED
        } else {
            OffsetMeasurementStatus.MEASURED_ALIGNED
        }
        val recommended = if (detached) {
            CompositionMode.SINGLE_SCENE_GL
        } else {
            CompositionMode.TRANSPARENT_OVERLAY
        }
        val decision = CompositionDecision(
            status = status,
            sampleCount = snapshot.size,
            meanLagMs = meanLag,
            p95LagMs = p95Lag,
            maxSpatialOffsetNorm = maxOffset,
            visibleDetachmentDetected = detached,
            recommendedMode = recommended,
            rationale = if (detached) {
                "Measured visible detachment (maxOffset=$maxOffset, p95LagMs=$p95Lag): migrate to SINGLE_SCENE_GL."
            } else {
                "Measured alignment within tolerance (maxOffset=$maxOffset, p95LagMs=$p95Lag): keep TRANSPARENT_OVERLAY."
            }
        )
        lastDecision = decision
        return decision
    }

    companion object {
        /**
         * Determines the active CompositionMode per README Section 11.5 & 21 Step 7:
         * - Preserves `TRANSPARENT_OVERLAY` (`CameraX PreviewView + Transparent GL Surface`) by default,
         *   and whenever `hasCameraSceneInput == false` (Path B camera texture not wired to live preview).
         * - Moves to `SINGLE_SCENE_GL` when `hasCameraSceneInput == true` and post effects, video
         *   recording, or the measured offset test decision specifies migration.
         */
        @JvmOverloads
        fun selectCompositionMode(
            effect: CompiledEffect?,
            requiresVideoRecording: Boolean = false,
            offsetDecision: CompositionDecision? = null,
            hasCameraSceneInput: Boolean = true
        ): CompositionMode {
            if (!hasCameraSceneInput) {
                return CompositionMode.TRANSPARENT_OVERLAY
            }
            if (requiresVideoRecording) {
                return CompositionMode.SINGLE_SCENE_GL
            }
            if (effect != null && effect.nodes.any { it.type == "post" || it.type == "recolor" || it.type == "eyeColor" }) {
                return CompositionMode.SINGLE_SCENE_GL
            }
            if (offsetDecision != null &&
                offsetDecision.status == OffsetMeasurementStatus.MEASURED_DETACHED &&
                offsetDecision.visibleDetachmentDetected
            ) {
                return CompositionMode.SINGLE_SCENE_GL
            }
            return CompositionMode.TRANSPARENT_OVERLAY
        }
    }
}
