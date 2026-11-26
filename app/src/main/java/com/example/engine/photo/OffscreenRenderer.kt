package com.example.engine.photo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix as AndroidMatrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLUtils
import android.opengl.Matrix
import android.util.Log
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.GlbLoader
import com.example.engine.render.RenderGraph
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.CompiledMaterial
import com.example.engine.scene.CompiledNode
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * OffscreenRenderer (README Section 14, Section 16 & Section 21 Step 9):
 *
 * - Owns an explicit EGL14 context and Pbuffer surface on the background photo worker thread.
 * - Defines and tests EGL context sharing (`sharedContext: EGLContext`), falling back cleanly to an
 *   isolated owned EGLContext if the driver rejects sharing or if no preview context is supplied.
 * - Preserves strict GL-thread/resource ownership rules via `renderGraph.resourceOwner.bindGlThread()`.
 * - Renders the evaluated `CompiledEffect` offscreen into a bound FBO (`GL_RGBA` + `GL_DEPTH_COMPONENT16`)
 *   and composites the result over the full-resolution photograph.
 * - Provides a documented, deterministic CPU fallback (`renderCpuFallback`) for simple sprite effects
 *   or incompatible/headless devices that uses the exact same `SceneEvaluator` world matrices,
 *   `FaceMeshRenderer` 3D vertices, and `FacePoseAdapter.createCameraProjection` matrix so photo
 *   placement matches preview intent without any filter-specific code paths.
 */
class OffscreenRenderer {

    companion object {
        private const val TAG = "OffscreenRenderer"
        private const val EGL_OPENGL_ES3_BIT_KHR = 0x0040
        private const val DEFAULT_MAX_OFFSCREEN_DIMENSION = 2560
    }

    enum class RenderBackend {
        EGL_OFFSCREEN,
        CPU_FALLBACK
    }

    data class SharingReport(
        val sharingRequested: Boolean,
        val isSharingActive: Boolean,
        val isEglInitialized: Boolean,
        val sharedContextHandle: Long,
        val ownedContextHandle: Long,
        val fallbackToUnsharedContext: Boolean
    )

    data class OffscreenRenderOutput(
        val bitmap: Bitmap,
        val backendUsed: RenderBackend,
        val renderedNodeCount: Int,
        val sharingReport: SharingReport,
        val allocatedOverlayBytes: Long
    )

    val renderGraph = RenderGraph()

    var eglDisplay: EGLDisplay? = null
        private set
    var eglConfig: EGLConfig? = null
        private set
    var eglContext: EGLContext? = null
        private set
    var eglSurface: EGLSurface? = null
        private set

    var isEglInitialized: Boolean = false
        private set
    var sharingRequested: Boolean = false
        private set
    var isSharingActive: Boolean = false
        private set
    var fallbackToUnsharedContext: Boolean = false
        private set
    var sharedContextHandle: Long = 0L
        private set
    var eglPbufferWidth: Int = 0
        private set
    var eglPbufferHeight: Int = 0
        private set

    private var fboId: Int = 0
    private var fboColorTexId: Int = 0
    private var fboDepthRboId: Int = 0
    private var fboWidth: Int = 0
    private var fboHeight: Int = 0

    // Pre-allocated matrix scratch arrays for background worker thread
    private val viewProjMatrix = FloatArray(16)
    private val nodeMvp = FloatArray(16)
    private val clipVec = FloatArray(4)
    private val worldVec = FloatArray(4)

    var lastPostApplicationCount: Int = 0
        private set
    var cpuPostApplicationCount: Int = 0
        private set

    private val photoStageOrderScratch = ArrayList<String>(8)
    val lastPhotoStageOrder: List<String>
        get() = photoStageOrderScratch

    private val renderedNodeIdsScratch = ArrayList<String>(64)
    val lastRenderedNodeIds: List<String>
        get() = renderedNodeIdsScratch

    /**
     * Initializes an explicitly owned EGL14 context and Pbuffer surface on the current worker thread.
     * Supports optional `sharedContext` for GPU texture/buffer sharing with the preview context,
     * and falls back cleanly to an isolated owned EGLContext if sharing is unsupported by the driver.
     */
    @Synchronized
    fun initEglContext(
        width: Int,
        height: Int,
        sharedContext: EGLContext? = null
    ): Boolean {
        val clampedW = width.coerceIn(1, DEFAULT_MAX_OFFSCREEN_DIMENSION)
        val clampedH = height.coerceIn(1, DEFAULT_MAX_OFFSCREEN_DIMENSION)
        val wantShare = sharedContext != null && sharedContext != EGL14.EGL_NO_CONTEXT && sharedContext.nativeHandle != 0L

        if (isEglInitialized &&
            eglPbufferWidth == clampedW &&
            eglPbufferHeight == clampedH &&
            sharingRequested == wantShare
        ) {
            return makeCurrent()
        }

        releaseEglOnly()
        sharingRequested = wantShare
        sharedContextHandle = if (wantShare && sharedContext != null) sharedContext.nativeHandle else 0L
        isSharingActive = false
        fallbackToUnsharedContext = false

        return try {
            val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            if (display == null || display == EGL14.EGL_NO_DISPLAY) {
                isEglInitialized = false
                return false
            }

            val version = IntArray(2)
            if (!EGL14.eglInitialize(display, version, 0, version, 1)) {
                isEglInitialized = false
                return false
            }

            val configAttribs24 = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_DEPTH_SIZE, 24,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            var foundConfig = EGL14.eglChooseConfig(display, configAttribs24, 0, configs, 0, 1, numConfigs, 0) &&
                numConfigs[0] > 0 && configs[0] != null
            if (!foundConfig) {
                val configAttribs16 = intArrayOf(
                    EGL14.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                    EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                    EGL14.EGL_RED_SIZE, 8,
                    EGL14.EGL_GREEN_SIZE, 8,
                    EGL14.EGL_BLUE_SIZE, 8,
                    EGL14.EGL_ALPHA_SIZE, 8,
                    EGL14.EGL_DEPTH_SIZE, 16,
                    EGL14.EGL_NONE
                )
                foundConfig = EGL14.eglChooseConfig(display, configAttribs16, 0, configs, 0, 1, numConfigs, 0) &&
                    numConfigs[0] > 0 && configs[0] != null
            }
            if (!foundConfig) {
                isEglInitialized = false
                return false
            }

            val chosenConfig = configs[0]!!
            val contextAttribs = intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
            )

            var ctx: EGLContext? = null
            if (wantShare && sharedContext != null) {
                ctx = EGL14.eglCreateContext(display, chosenConfig, sharedContext, contextAttribs, 0)
                if (ctx != null && ctx != EGL14.EGL_NO_CONTEXT && ctx.nativeHandle != 0L) {
                    isSharingActive = true
                } else {
                    fallbackToUnsharedContext = true
                    ctx = EGL14.eglCreateContext(display, chosenConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
                }
            } else {
                ctx = EGL14.eglCreateContext(display, chosenConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            }

            if (ctx == null || ctx == EGL14.EGL_NO_CONTEXT) {
                isEglInitialized = false
                return false
            }

            val pbufferAttribs = intArrayOf(
                EGL14.EGL_WIDTH, clampedW,
                EGL14.EGL_HEIGHT, clampedH,
                EGL14.EGL_NONE
            )
            val surface = EGL14.eglCreatePbufferSurface(display, chosenConfig, pbufferAttribs, 0)
            if (surface == null || surface == EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroyContext(display, ctx)
                isEglInitialized = false
                return false
            }

            if (!EGL14.eglMakeCurrent(display, surface, surface, ctx)) {
                EGL14.eglDestroySurface(display, surface)
                EGL14.eglDestroyContext(display, ctx)
                isEglInitialized = false
                return false
            }

            eglDisplay = display
            eglConfig = chosenConfig
            eglContext = ctx
            eglSurface = surface
            eglPbufferWidth = clampedW
            eglPbufferHeight = clampedH
            isEglInitialized = true

            // Bind GlResourceOwner to this background photo worker thread and initialize RenderGraph
            renderGraph.resourceOwner.bindGlThread()
            renderGraph.init()
            renderGraph.resize(clampedW, clampedH)
            true
        } catch (t: Throwable) {
            Log.w(TAG, "EGL offscreen initialization unavailable (${t.message}), will use documented CPU fallback")
            renderGraph.resourceOwner.bindGlThread()
            isEglInitialized = false
            false
        }
    }

    /**
     * Tests and reports EGL context sharing state for diagnostics and unit verification.
     */
    @Synchronized
    fun verifyContextSharing(
        sharedContext: EGLContext? = null,
        testWidth: Int = 64,
        testHeight: Int = 64
    ): SharingReport {
        renderGraph.resourceOwner.bindGlThread()
        initEglContext(testWidth, testHeight, sharedContext)
        return getSharingReport()
    }

    fun getSharingReport(): SharingReport {
        val ctx = eglContext
        val ownedHandle = try {
            if (ctx != null && ctx != EGL14.EGL_NO_CONTEXT) ctx.nativeHandle else 0L
        } catch (_: Throwable) {
            0L
        }
        return SharingReport(
            sharingRequested = sharingRequested,
            isSharingActive = isSharingActive,
            isEglInitialized = isEglInitialized,
            sharedContextHandle = sharedContextHandle,
            ownedContextHandle = ownedHandle,
            fallbackToUnsharedContext = fallbackToUnsharedContext
        )
    }

    private fun makeCurrent(): Boolean {
        val disp = eglDisplay ?: return false
        val surf = eglSurface ?: return false
        val ctx = eglContext ?: return false
        if (!isEglInitialized || disp == EGL14.EGL_NO_DISPLAY || surf == EGL14.EGL_NO_SURFACE || ctx == EGL14.EGL_NO_CONTEXT) {
            return false
        }
        return try {
            val ok = EGL14.eglMakeCurrent(disp, surf, surf, ctx)
            if (ok) {
                renderGraph.resourceOwner.bindGlThread()
            }
            ok
        } catch (_: Throwable) {
            false
        }
    }

    private fun ensureOffscreenFbo(width: Int, height: Int): Boolean {
        renderGraph.resourceOwner.bindGlThread()
        if (fboId != 0 && fboWidth == width && fboHeight == height) {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            return status == GLES30.GL_FRAMEBUFFER_COMPLETE
        }

        releaseFbo()

        return try {
            val fbos = IntArray(1)
            val texs = IntArray(1)
            val rbos = IntArray(1)
            GLES30.glGenFramebuffers(1, fbos, 0)
            GLES30.glGenTextures(1, texs, 0)
            GLES30.glGenRenderbuffers(1, rbos, 0)

            fboId = fbos[0]
            fboColorTexId = texs[0]
            fboDepthRboId = rbos[0]

            // On JVM Robolectric stubs, glGenFramebuffers returns 0
            if (fboId == 0 || fboColorTexId == 0 || fboDepthRboId == 0) {
                releaseFbo()
                return false
            }

            renderGraph.resourceOwner.trackFramebuffer(fboId)
            renderGraph.resourceOwner.trackTexture(fboColorTexId)
            renderGraph.resourceOwner.trackRenderbuffer(fboDepthRboId)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboColorTexId)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D,
                0,
                GLES30.GL_RGBA,
                width,
                height,
                0,
                GLES30.GL_RGBA,
                GLES30.GL_UNSIGNED_BYTE,
                null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D,
                fboColorTexId,
                0
            )

            GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, fboDepthRboId)
            GLES30.glRenderbufferStorage(GLES30.GL_RENDERBUFFER, GLES30.GL_DEPTH_COMPONENT16, width, height)
            GLES30.glFramebufferRenderbuffer(
                GLES30.GL_FRAMEBUFFER,
                GLES30.GL_DEPTH_ATTACHMENT,
                GLES30.GL_RENDERBUFFER,
                fboDepthRboId
            )

            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                releaseFbo()
                return false
            }

            fboWidth = width
            fboHeight = height
            true
        } catch (_: Throwable) {
            releaseFbo()
            false
        }
    }

    private fun releaseFbo() {
        try {
            if (fboId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
                renderGraph.resourceOwner.untrackFramebuffer(fboId)
                fboId = 0
            }
            if (fboColorTexId != 0) {
                GLES30.glDeleteTextures(1, intArrayOf(fboColorTexId), 0)
                renderGraph.resourceOwner.untrackTexture(fboColorTexId)
                fboColorTexId = 0
            }
            if (fboDepthRboId != 0) {
                GLES30.glDeleteRenderbuffers(1, intArrayOf(fboDepthRboId), 0)
                renderGraph.resourceOwner.untrackRenderbuffer(fboDepthRboId)
                fboDepthRboId = 0
            }
        } catch (_: Throwable) {
            fboId = 0
            fboColorTexId = 0
            fboDepthRboId = 0
        }
        fboWidth = 0
        fboHeight = 0
    }

    /**
     * Renders the evaluated `CompiledEffect` onto a copy of `baseBitmap` using either the owned
     * EGL offscreen FBO pipeline or the documented CPU fallback when EGL/FBO hardware is unavailable
     * or `forceCpuFallback` is requested.
     */
    @Synchronized
    fun renderToPhoto(
        baseBitmap: Bitmap,
        effect: CompiledEffect,
        faceState: FaceState,
        evaluator: SceneEvaluator,
        sharedContext: EGLContext? = null,
        forceCpuFallback: Boolean = false,
        tierMaxTexSize: Int = 1024
    ): OffscreenRenderOutput {
        val width = baseBitmap.width.coerceAtLeast(1)
        val height = baseBitmap.height.coerceAtLeast(1)

        // Ensure thread ownership is bound to the current photo worker thread
        renderGraph.resourceOwner.bindGlThread()

        if (!forceCpuFallback) {
            val eglReady = initEglContext(width, height, sharedContext)
            if (eglReady && ensureOffscreenFbo(width, height)) {
                val eglResult = renderViaEglOffscreen(baseBitmap, effect, faceState, evaluator, width, height, tierMaxTexSize)
                if (eglResult != null) {
                    return eglResult
                }
            }
        } else {
            // Even when forcing CPU fallback, record sharing state cleanly
            sharingRequested = sharedContext != null && sharedContext != EGL14.EGL_NO_CONTEXT
            sharedContextHandle = if (sharingRequested && sharedContext != null) sharedContext.nativeHandle else 0L
        }

        // Documented CPU fallback for sprite/faceMesh/model effects or headless/incompatible devices
        return renderCpuFallback(baseBitmap, effect, faceState, evaluator, tierMaxTexSize)
    }

    private fun renderViaEglOffscreen(
        baseBitmap: Bitmap,
        effect: CompiledEffect,
        faceState: FaceState,
        evaluator: SceneEvaluator,
        width: Int,
        height: Int,
        tierMaxTexSize: Int
    ): OffscreenRenderOutput? {
        return try {
            photoStageOrderScratch.clear()
            renderedNodeIdsScratch.clear()
            lastPostApplicationCount = 0
            cpuPostApplicationCount = 0

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            GLES30.glViewport(0, 0, width, height)

            // 1. Draw the base photo into the FBO first (uploaded to fboColorTexId in GL bottom-left orientation)
            val flipMatrix = AndroidMatrix().apply {
                postScale(1f, -1f, width / 2f, height / 2f)
            }
            val flippedBaseBmp = Bitmap.createBitmap(baseBitmap, 0, 0, width, height, flipMatrix, false)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboColorTexId)
            GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, flippedBaseBmp, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            if (flippedBaseBmp !== baseBitmap && !flippedBaseBmp.isRecycled) {
                flippedBaseBmp.recycle()
            }
            GLES30.glDepthMask(true)
            GLES30.glClear(GLES30.GL_DEPTH_BUFFER_BIT)
            photoStageOrderScratch.add("BASE_PHOTO_FBO")

            // 2. Render scene effects & GPU PostRenderer (once) directly inside the FBO over the base photo
            val photoTimeSec = effect.photoPolicy.timeSec ?: renderGraph.lastFrameContext.renderTimeSec
            val photoFrameCtx = com.example.engine.render.DeterministicFrameContext.forPhoto(
                packageId = effect.packageId,
                photoTimeSec = photoTimeSec,
                explicitSeed = effect.photoPolicy.seed
            )
            lastPhotoFrameContext = photoFrameCtx
            renderGraph.restoreEffectResourcesOnGlThread(effect, tierMaxTexSize)
            renderGraph.render(
                effect = effect,
                faceState = faceState,
                evaluator = evaluator,
                viewportWidth = width,
                viewportHeight = height,
                frameContext = photoFrameCtx
            )
            GLES30.glFinish()
            photoStageOrderScratch.add("EFFECTS")
            renderedNodeIdsScratch.addAll(renderGraph.lastRenderedNodeIds)
            if (renderGraph.lastRenderedPostCount > 0) {
                lastPostApplicationCount = 1
                photoStageOrderScratch.add("POST_ONCE")
            }

            // 3. Read back the finished FBO (no CPU double application of PostRenderer!)
            val pixelBuf = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
            GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, pixelBuf)
            pixelBuf.rewind()

            val rawFboBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            rawFboBmp.copyPixelsFromBuffer(pixelBuf)
            val overlayBytes = rawFboBmp.allocationByteCount.toLong()

            val outputBmp = Bitmap.createBitmap(rawFboBmp, 0, 0, width, height, flipMatrix, false)
            if (outputBmp !== rawFboBmp && !rawFboBmp.isRecycled) {
                rawFboBmp.recycle()
            }

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

            val visibleRenderNodes = countVisibleRenderNodes(effect, evaluator)
            OffscreenRenderOutput(
                bitmap = outputBmp,
                backendUsed = RenderBackend.EGL_OFFSCREEN,
                renderedNodeCount = visibleRenderNodes,
                sharingReport = getSharingReport(),
                allocatedOverlayBytes = overlayBytes
            )
        } catch (t: Throwable) {
            Log.w(TAG, "EGL FBO render failed (${t.message}), switching to documented CPU fallback")
            null
        }
    }

    var lastPhotoFrameContext: com.example.engine.render.DeterministicFrameContext =
        com.example.engine.render.DeterministicFrameContext.DEFAULT
        private set

    /**
     * Documented CPU Fallback (README Section 16 point 4):
     * Renders the evaluated `CompiledEffect` onto `baseBitmap` using the exact same `SceneEvaluator`
     * world matrices, `FaceMeshRenderer` 3D vertices, and `FacePoseAdapter.createCameraProjection`
     * camera projection matrix as the live OpenGL ES 3.0 pipeline.
     *
     * Enforces the Chunk B0 pass order:
     * 1. Base photo (`BASE_PHOTO`)
     * 2. Camera-pixel passes (`CAMERA_PIXEL_PASSES`: `faceMesh`, `eyeColor`) — before 3D props
     * 3. Opaque 3D/2D nodes (`OPAQUE_NODES`)
     * 4. Alpha lenses / transparent 3D/2D nodes (`TRANSPARENT_NODES`)
     * 5. Post-processing nodes (`POST_ONCE`, applied exactly once)
     * 6. Screen-space overlay nodes (`OVERLAY_NODES`)
     */
    fun renderCpuFallback(
        baseBitmap: Bitmap,
        effect: CompiledEffect,
        faceState: FaceState,
        evaluator: SceneEvaluator,
        tierMaxTexSize: Int = 1024
    ): OffscreenRenderOutput {
        val width = baseBitmap.width.coerceAtLeast(1)
        val height = baseBitmap.height.coerceAtLeast(1)
        val outputBmp = cloneBitmapWithPixels(baseBitmap)

        val photoTimeSec = effect.photoPolicy.timeSec ?: renderGraph.lastFrameContext.renderTimeSec
        lastPhotoFrameContext = com.example.engine.render.DeterministicFrameContext.forPhoto(
            packageId = effect.packageId,
            photoTimeSec = photoTimeSec,
            explicitSeed = effect.photoPolicy.seed
        )

        photoStageOrderScratch.clear()
        renderedNodeIdsScratch.clear()
        lastPostApplicationCount = 0
        cpuPostApplicationCount = 0
        photoStageOrderScratch.add("BASE_PHOTO")

        if (!faceState.isVisible || faceState.fadeWeight < 0.001f) {
            return OffscreenRenderOutput(
                bitmap = outputBmp,
                backendUsed = RenderBackend.CPU_FALLBACK,
                renderedNodeCount = 0,
                sharingReport = getSharingReport(),
                allocatedOverlayBytes = 0L
            )
        }

        val canvas = Canvas(outputBmp)
        val aspect = width.toFloat() / height.toFloat().coerceAtLeast(1f)
        val fovYViewDeg = faceState.projectionParams.visibleViewFovYDeg(aspect)
        FacePoseAdapter.createCameraProjection(
            fovYDeg = fovYViewDeg,
            aspectRatio = aspect,
            near = faceState.projectionParams.near,
            far = faceState.projectionParams.far,
            frontFacing = faceState.isFrontCamera,
            out = viewProjMatrix
        )

        val requestedTier = when (effect.activeTier) {
            "B" -> PerformanceGovernor.Tier.B
            "C" -> PerformanceGovernor.Tier.C
            else -> PerformanceGovernor.Tier.A
        }
        val effectiveTier = TierPolicy.resolveEffectiveTier(effect, requestedTier, tierMaxTexSize)
        val tierOverride = effect.tierOverrides[effectiveTier.tierId]
        val maxTex = effectiveTier.textureMaxSize

        // Partition and sort visible renderable nodes identically to RenderGraph:
        // 1. cameraPixelList (faceMesh, eyeColor) -> 2. opaqueList -> 3. transparentList -> 4. post -> 5. overlayList
        val nodes = effect.nodes
        val cameraPixelList = ArrayList<Int>()
        val opaqueList = ArrayList<Int>()
        val transparentList = ArrayList<Int>()
        val overlayList = ArrayList<Int>()
        for (i in nodes.indices) {
            val node = nodes[i]
            if (!evaluator.isNodeVisible[i]) continue
            if (!TierPolicy.isNodeVisibleInTier(i, node.baseVisible, tierOverride)) continue
            when (node.type) {
                "occluder", "post", "group" -> Unit
                "faceMesh", "eyeColor", "recolor" -> cameraPixelList.add(i)
                "overlay" -> overlayList.add(i)
                else -> {
                    val mat = com.example.engine.render.ModelRenderer.resolvePrimaryModelMaterial(node, effect.materials)
                    if (mat?.blend == "opaque") {
                        opaqueList.add(i)
                    } else {
                        transparentList.add(i)
                    }
                }
            }
        }

        cameraPixelList.sortBy { nodes[it].renderOrder }
        transparentList.sortWith { a, b ->
            val orderCmp = nodes[a].renderOrder.compareTo(nodes[b].renderOrder)
            if (orderCmp != 0) {
                orderCmp
            } else {
                val zA = evaluator.worldMatrices[a * 16 + 14]
                val zB = evaluator.worldMatrices[b * 16 + 14]
                zA.compareTo(zB)
            }
        }
        overlayList.sortBy { nodes[it].renderOrder }

        photoStageOrderScratch.add("CAMERA_PIXEL_PASSES")
        photoStageOrderScratch.add("OPAQUE_NODES")
        photoStageOrderScratch.add("TRANSPARENT_NODES")

        val prePostIndices = ArrayList<Int>(cameraPixelList.size + opaqueList.size + transparentList.size)
        prePostIndices.addAll(cameraPixelList)
        prePostIndices.addAll(opaqueList)
        prePostIndices.addAll(transparentList)

        var renderedNodes = 0
        var transientAllocatedBytes = 0L

        for (nodeIdx in prePostIndices) {
            val node = nodes[nodeIdx]
            val opacity = (effect.propertySlots.opacity[nodeIdx] * effect.propertySlots.visible[nodeIdx]).coerceIn(0f, 1f)
            if (opacity <= 0.001f) continue

            val mat = com.example.engine.render.ModelRenderer.resolvePrimaryModelMaterial(node, effect.materials)
            val worldOffset = nodeIdx * 16
            Matrix.multiplyMM(nodeMvp, 0, viewProjMatrix, 0, evaluator.worldMatrices, worldOffset)

            when (node.type) {
                "sprite" -> {
                    val bytes = drawCpuSpriteNode(
                        outputBmp = outputBmp,
                        canvas = canvas,
                        node = node,
                        effect = effect,
                        evaluator = evaluator,
                        worldOffset = worldOffset,
                        opacity = opacity,
                        mat = mat,
                        width = width,
                        height = height,
                        isFrontCamera = faceState.isFrontCamera,
                        maxTexSize = maxTex
                    )
                    if (bytes >= 0L) {
                        renderedNodes++
                        transientAllocatedBytes += bytes
                        renderedNodeIdsScratch.add(node.id)
                    }
                }
                "faceMesh" -> {
                    val bytes = drawCpuFaceMeshNode(
                        outputBmp = outputBmp,
                        canvas = canvas,
                        node = node,
                        effect = effect,
                        faceState = faceState,
                        opacity = opacity,
                        mat = mat,
                        width = width,
                        height = height,
                        aspect = aspect,
                        maxTexSize = maxTex
                    )
                    if (bytes >= 0L) {
                        renderedNodes++
                        transientAllocatedBytes += bytes
                        renderedNodeIdsScratch.add(node.id)
                    }
                }
                "eyeColor", "recolor" -> {
                    val modified = renderGraph.recolor.applyNodeToBitmap(
                        bitmap = outputBmp,
                        node = node,
                        effect = effect,
                        faceState = faceState,
                        opacity = opacity
                    )
                    if (modified > 0) {
                        renderedNodes++
                        renderedNodeIdsScratch.add(node.id)
                    }
                }
                "model" -> {
                    val bytes = drawCpuModelNode(
                        outputBmp = outputBmp,
                        canvas = canvas,
                        node = node,
                        effect = effect,
                        evaluator = evaluator,
                        worldOffset = worldOffset,
                        opacity = opacity,
                        mat = mat,
                        width = width,
                        height = height,
                        isFrontCamera = faceState.isFrontCamera,
                        maxTexSize = maxTex
                    )
                    if (bytes >= 0L) {
                        renderedNodes++
                        transientAllocatedBytes += bytes
                        renderedNodeIdsScratch.add(node.id)
                    }
                }
            }
        }

        if (effect.passGraph.isCustomManifestGraph && faceState.isVisible && faceState.fadeWeight >= 0.001f) {
            val passGraph = effect.passGraph
            for ((pIdx, passNode) in passGraph.passes.withIndex()) {
                if (!passGraph.isPassEnabledInTier(passNode, effectiveTier)) continue
                val isStandaloneRecolor = (passNode.templateId == "recolor@1" ||
                    passNode.templateId == "pass.recolor@1" ||
                    passNode.templateId == "pixel.recolor@1" ||
                    passNode.templateId == "eyeColor@1") &&
                    nodes.none { it.id == passNode.id }
                if (isStandaloneRecolor) {
                    val resolvedParams = passGraph.resolvePassParams(passNode, effect.propertySlots, effectiveTier)
                    val maskRefId = passNode.parsedReads.firstOrNull {
                        it.kind == com.example.engine.render.RenderTargetRef.TargetKind.MASK
                    }?.identifier ?: "iris"
                    val cleanMaskId = maskRefId.removePrefix("\$mask:").trim().lowercase().replace('-', '_')
                    val synthNode = com.example.engine.scene.CompiledNode(
                        index = nodes.size + pIdx,
                        id = passNode.id,
                        parentIndex = -1,
                        tags = emptyList(),
                        type = "recolor",
                        anchor = null,
                        basePosition = floatArrayOf(0f, 0f, 0f),
                        baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
                        baseScale = floatArrayOf(1f, 1f, 1f),
                        rotationFollow = "none",
                        materialId = null,
                        baseVisible = true,
                        baseOpacity = 1f,
                        renderOrder = pIdx,
                        faceMeshRegion = "\$mask:$cleanMaskId",
                        postEffect = "recolor@1",
                        postParams = resolvedParams
                    )
                    val modified = renderGraph.recolor.applyNodeToBitmap(
                        bitmap = outputBmp,
                        node = synthNode,
                        effect = effect,
                        faceState = faceState,
                        opacity = 1.0f
                    )
                    if (modified > 0) {
                        renderedNodes++
                        renderedNodeIdsScratch.add(passNode.id)
                    }
                }
            }
        }

        // Execute PostRenderer on the composed scene (base photograph + virtual nodes) EXACTLY ONCE
        val appliedPostCount = renderGraph.post.applyPostNodesToBitmap(
            composedBitmap = outputBmp,
            effect = effect,
            isNodeVisible = evaluator.isNodeVisible,
            faceState = faceState,
            regionMaskPass = renderGraph.regionMask,
            frameContext = lastPhotoFrameContext
        )
        if (appliedPostCount > 0) {
            lastPostApplicationCount = 1
            cpuPostApplicationCount = 1
            photoStageOrderScratch.add("POST_ONCE")
            for (i in nodes.indices) {
                if (nodes[i].type == "post" && evaluator.isNodeVisible[i] &&
                    TierPolicy.isNodeVisibleInTier(i, nodes[i].baseVisible, tierOverride)
                ) {
                    renderedNodeIdsScratch.add(nodes[i].id)
                }
            }
        }
        renderedNodes += appliedPostCount

        photoStageOrderScratch.add("OVERLAY_NODES")
        for (nodeIdx in overlayList) {
            val node = nodes[nodeIdx]
            val mat = node.materialId?.let { effect.materials[it] }
                ?: node.tags.firstNotNullOfOrNull { effect.materials[it] }
            val opacity = (effect.propertySlots.opacity[nodeIdx] * effect.propertySlots.visible[nodeIdx] * (mat?.opacity ?: 1f)).coerceIn(0f, 1f)
            if (opacity <= 0.001f) continue
            val dynFrame = effect.propertySlots.getDynamicParam(
                "pass:${node.id}.frameIndex",
                effect.propertySlots.getDynamicParam("sprite:${node.id}.frameIndex", Float.NaN)
            )
            val explicitFrameIndex = if (dynFrame.isFinite() && dynFrame >= 0f) {
                kotlin.math.round(dynFrame).toInt()
            } else {
                node.spriteFrameIndex
            }
            val assetId = node.spriteAssetId ?: mat?.mapAssetId
            val asset = assetId?.let { effect.assets[it] }
            if (asset != null) {
                val bmpFile = File(asset.localPath)
                if (bmpFile.exists()) {
                    val overlayBmp = BitmapFactory.decodeFile(bmpFile.absolutePath)
                    if (overlayBmp != null) {
                        transientAllocatedBytes += overlayBmp.allocationByteCount.toLong()
                        val instanceCount = node.particleCount.coerceIn(1, 32)
                        val overlayParticleState = FloatArray(5)
                        for (pIdx in 0 until instanceCount) {
                            com.example.engine.render.SpriteRenderer.evaluateParticleInstance(
                                node = node,
                                particleIndex = pIdx,
                                renderTimeSec = lastPhotoFrameContext.renderTimeSec,
                                outState5 = overlayParticleState
                            )
                            val pDx = overlayParticleState[0]
                            val pDy = overlayParticleState[1]
                            val pScale = overlayParticleState[2]
                            val pTwinkle = overlayParticleState[3]
                            val pFrameOffset = overlayParticleState[4].toInt()
                            val instOpacity = (opacity * pTwinkle).coerceIn(0f, 1f)
                            if (instOpacity <= 1e-4f) continue
                            com.example.engine.render.SpriteRenderer.applyOverlayToBitmap(
                                targetBitmap = outputBmp,
                                overlayBitmap = overlayBmp,
                                fit = node.overlayFit,
                                safeArea = node.overlaySafeArea,
                                opacity = instOpacity,
                                blendMode = mat?.blend ?: "alpha",
                                isFrontCamera = faceState.isFrontCamera,
                                rows = node.spriteRows,
                                cols = node.spriteCols,
                                fps = node.spriteFps,
                                loop = node.spriteLoop,
                                renderTimeSec = lastPhotoFrameContext.renderTimeSec,
                                frameCount = node.spriteFrameCount,
                                explicitFrameIndex = explicitFrameIndex,
                                startFrame = node.spriteStartFrame + pFrameOffset,
                                colorTint = mat?.color,
                                offsetNormX = pDx,
                                offsetNormY = pDy,
                                scaleFactor = pScale
                            )
                        }
                        overlayBmp.recycle()
                    }
                }
            }
            renderedNodeIdsScratch.add(node.id)
            renderedNodes++
        }

        return OffscreenRenderOutput(
            bitmap = outputBmp,
            backendUsed = RenderBackend.CPU_FALLBACK,
            renderedNodeCount = renderedNodes,
            sharingReport = getSharingReport(),
            allocatedOverlayBytes = transientAllocatedBytes
        )
    }

    private fun drawCpuSpriteNode(
        outputBmp: Bitmap,
        canvas: Canvas,
        node: CompiledNode,
        effect: CompiledEffect,
        evaluator: SceneEvaluator,
        worldOffset: Int,
        opacity: Float,
        mat: CompiledMaterial?,
        width: Int,
        height: Int,
        isFrontCamera: Boolean,
        maxTexSize: Int
    ): Long {
        // Project local quad center (0, 0, 0, 1) and half-extents (-0.5..+0.5 in local quad space)
        val centerScreen = projectLocalPointToScreen(nodeMvp, 0f, 0f, 0f, width, height) ?: return -1L
        val rightEdge = projectLocalPointToScreen(nodeMvp, 0.5f, 0f, 0f, width, height) ?: centerScreen
        val topEdge = projectLocalPointToScreen(nodeMvp, 0f, 0.5f, 0f, width, height) ?: centerScreen

        val halfW = max(12f, hypot(rightEdge[0] - centerScreen[0], rightEdge[1] - centerScreen[1]))
        val halfH = max(12f, hypot(topEdge[0] - centerScreen[0], topEdge[1] - centerScreen[1]))

        val dx = rightEdge[0] - centerScreen[0]
        val dy = rightEdge[1] - centerScreen[1]
        val m00 = evaluator.worldMatrices[worldOffset + 0]
        val m01 = evaluator.worldMatrices[worldOffset + 1]
        val worldRollDeg = Math.toDegrees(atan2(m01.toDouble(), m00.toDouble())).toFloat()
        val screenRollDeg = if (isFrontCamera) worldRollDeg else -worldRollDeg

        val baseAlphaFloat = (opacity * (mat?.opacity ?: 1f)).coerceIn(0f, 1f)
        val hasExplicitColor = mat != null
        val tint = mat?.color ?: floatArrayOf(1f, 1f, 1f)
        val tintR = (tint.getOrElse(0) { 1f }.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val tintG = (tint.getOrElse(1) { 1f }.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val tintB = (tint.getOrElse(2) { 1f }.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val blendMode = mat?.blend ?: "alpha"

        val asset = (node.spriteAssetId ?: mat?.mapAssetId)?.let { effect.assets[it] }
        val spriteBmp = asset?.let { decodeSampledBitmap(it.localPath, maxTexSize) }
        val allocated = spriteBmp?.allocationByteCount?.toLong() ?: 0L

        val renderTimeSec = lastPhotoFrameContext.renderTimeSec
        val dynFrame = effect.propertySlots.getDynamicParam(
            "pass:${node.id}.frameIndex",
            effect.propertySlots.getDynamicParam("sprite:${node.id}.frameIndex", Float.NaN)
        )
        val explicitFrameIndex = if (dynFrame.isFinite() && dynFrame >= 0f) {
            kotlin.math.round(dynFrame).toInt()
        } else {
            node.spriteFrameIndex
        }

        val instanceCount = node.particleCount.coerceIn(1, 32)
        val pState = FloatArray(5)
        val uvRect = FloatArray(4)

        for (pIdx in 0 until instanceCount) {
            com.example.engine.render.SpriteRenderer.evaluateParticleInstance(node, pIdx, renderTimeSec, pState)
            val pDx = pState[0]
            val pDy = pState[1]
            val pScale = pState[2]
            val pTwinkle = pState[3]
            val pFrameOffset = pState[4].toInt()
            val effectiveAlpha = (baseAlphaFloat * pTwinkle * 255f).roundToInt().coerceIn(0, 255)
            if (effectiveAlpha <= 1) continue

            com.example.engine.render.SpriteRenderer.computeFlipbookUvRect(
                rows = node.spriteRows,
                cols = node.spriteCols,
                fps = node.spriteFps,
                loop = node.spriteLoop,
                renderTimeSec = renderTimeSec,
                outRect4 = uvRect,
                frameCount = node.spriteFrameCount,
                explicitFrameIndex = explicitFrameIndex,
                startFrame = node.spriteStartFrame + pFrameOffset,
                texWidth = spriteBmp?.width ?: 0,
                texHeight = spriteBmp?.height ?: 0
            )

            val instCx = centerScreen[0] + pDx * halfW * 2f
            val instCy = centerScreen[1] - pDy * halfH * 2f
            val instHalfW = (halfW * pScale).coerceAtLeast(4f)
            val instHalfH = (halfH * pScale).coerceAtLeast(4f)
            val dstRect = RectF(
                instCx - instHalfW,
                instCy - instHalfH,
                instCx + instHalfW,
                instCy + instHalfH
            )

            val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                alpha = effectiveAlpha
                if (tintR < 250 || tintG < 250 || tintB < 250) {
                    colorFilter = PorterDuffColorFilter(Color.rgb(tintR, tintG, tintB), PorterDuff.Mode.MULTIPLY)
                }
                when (blendMode.lowercase()) {
                    "additive", "add" -> xfermode = PorterDuffXfermode(PorterDuff.Mode.ADD)
                    "screen" -> xfermode = PorterDuffXfermode(PorterDuff.Mode.SCREEN)
                    "multiply" -> xfermode = PorterDuffXfermode(PorterDuff.Mode.MULTIPLY)
                }
            }

            val probeX = instCx.roundToInt().coerceIn(0, width - 1)
            val probeY = instCy.roundToInt().coerceIn(0, height - 1)
            val beforePixel = outputBmp.getPixel(probeX, probeY)

            val srcRect = if (spriteBmp != null && spriteBmp.width > 0 && spriteBmp.height > 0) {
                val sLeft = (uvRect[0] * spriteBmp.width).roundToInt().coerceIn(0, spriteBmp.width - 1)
                val sTop = (uvRect[1] * spriteBmp.height).roundToInt().coerceIn(0, spriteBmp.height - 1)
                val sRight = ((uvRect[0] + uvRect[2]) * spriteBmp.width).roundToInt().coerceIn(sLeft + 1, spriteBmp.width)
                val sBottom = ((uvRect[1] + uvRect[3]) * spriteBmp.height).roundToInt().coerceIn(sTop + 1, spriteBmp.height)
                Rect(sLeft, sTop, sRight, sBottom)
            } else {
                null
            }

            canvas.save()
            if (abs(screenRollDeg) > 0.05f && dx.isFinite() && dy.isFinite()) {
                canvas.rotate(screenRollDeg, instCx, instCy)
            }
            if (spriteBmp != null) {
                canvas.drawBitmap(spriteBmp, srcRect, dstRect, paint)
            } else {
                paint.style = Paint.Style.FILL
                val fillR = if (!hasExplicitColor && tintR == 255) 245 else tintR
                val fillG = if (!hasExplicitColor && tintG == 255) 205 else tintG
                val fillB = if (!hasExplicitColor && tintB == 255) 65 else tintB
                paint.color = Color.argb(effectiveAlpha, fillR, fillG, fillB)
                canvas.drawOval(dstRect, paint)
            }
            canvas.restore()

            // If running on a headless JVM stub where Canvas is a no-op, rasterize into outputBmp directly
            if (outputBmp.getPixel(probeX, probeY) == beforePixel && effectiveAlpha > 4) {
                val fallbackColor = Color.argb(
                    effectiveAlpha,
                    if (!hasExplicitColor && tintR == 255 && tintG == 255 && tintB == 255) 245 else tintR,
                    if (!hasExplicitColor && tintR == 255 && tintG == 255 && tintB == 255) 205 else tintG,
                    if (!hasExplicitColor && tintR == 255 && tintG == 255 && tintB == 255) 65 else tintB
                )
                rasterizeRectIntoBitmap(
                    target = outputBmp,
                    rect = dstRect,
                    spriteBmp = spriteBmp,
                    fallbackColor = fallbackColor,
                    effectiveAlpha = effectiveAlpha,
                    uvRect4 = uvRect,
                    tintR = tintR,
                    tintG = tintG,
                    tintB = tintB,
                    blendMode = blendMode
                )
            }
        }

        spriteBmp?.recycle()
        return allocated
    }

    private fun drawCpuFaceMeshNode(
        outputBmp: Bitmap,
        canvas: Canvas,
        node: CompiledNode,
        effect: CompiledEffect,
        faceState: FaceState,
        opacity: Float,
        mat: CompiledMaterial?,
        width: Int,
        height: Int,
        aspect: Float,
        maxTexSize: Int
    ): Long {
        renderGraph.faceMesh.updateVertices(
            faceState = faceState,
            viewportAspect = aspect
        )

        val faceMeshMvp = FloatArray(16)
        val scratchInv = FloatArray(16)
        val scratchDelta = FloatArray(16)
        com.example.engine.render.FaceMeshRenderer.computePoseCorrectedMvp(
            viewProjectionMatrix = viewProjMatrix,
            faceState = faceState,
            outMvp = faceMeshMvp,
            scratchInvLandmark = scratchInv,
            scratchRigidDelta = scratchDelta
        )

        val livePos = renderGraph.faceMesh.livePositions
        val vertCount = UvRegistry.VERTEX_COUNT
        val screenVerts = FloatArray(vertCount * 2)
        for (v in 0 until vertCount) {
            val vx = livePos[v * 3]
            val vy = livePos[v * 3 + 1]
            val vz = livePos[v * 3 + 2]
            val p = projectLocalPointToScreen(faceMeshMvp, vx, vy, vz, width, height)
            if (p != null) {
                screenVerts[v * 2] = p[0]
                screenVerts[v * 2 + 1] = p[1]
            } else {
                return -1L
            }
        }

        val mapAsset = mat?.mapAssetId?.let { effect.assets[it] }
        val maskAsset = mat?.maskAssetId?.let { effect.assets[it] }
        val mapBmp = mapAsset?.let { decodeSampledBitmap(it.localPath, maxTexSize) }
        val maskBmp = maskAsset?.let { decodeSampledBitmap(it.localPath, maxTexSize) }

        val effectiveOpacity = (opacity * (mat?.opacity ?: 1f)).coerceIn(0f, 1f)
        val effectiveAlpha = (effectiveOpacity * 255f).roundToInt().coerceIn(0, 255)

        val protectEyes = mat?.protectEyes ?: true
        val protectLips = mat?.protectLips ?: true
        val protectMouthInterior = mat?.protectMouthInterior ?: true
        val protectAny = protectEyes || protectLips || protectMouthInterior
        val blendMode = com.example.engine.render.FaceMeshRenderer.BlendMode.fromString(mat?.blend)
        val mouthOpenWeight = com.example.engine.render.FaceMeshRenderer.resolveMouthOpenWeight(faceState)
        val eyeBlinkWeight = com.example.engine.render.FaceMeshRenderer.resolveEyeBlinkWeight(faceState)

        val tris = UvRegistry.CANONICAL_TRIANGLES
        val uvs = UvRegistry.CANONICAL_UVS
        val protWeights = UvRegistry.VERTEX_PROTECTION_WEIGHTS

        var allocated = 0L
        if (mapBmp != null) allocated += mapBmp.allocationByteCount.toLong()
        if (maskBmp != null) allocated += maskBmp.allocationByteCount.toLong()

        val tint = mat?.color ?: floatArrayOf(0.9f, 0.35f, 0.55f, 1f)
        val tintR = tint.getOrElse(0) { 0.9f }.coerceIn(0f, 1f)
        val tintG = tint.getOrElse(1) { 0.35f }.coerceIn(0f, 1f)
        val tintB = tint.getOrElse(2) { 0.55f }.coerceIn(0f, 1f)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }
        val triPath = Path()
        val scratchBaseRgb = FloatArray(3)
        val scratchSrcRgb = FloatArray(3)
        val scratchBlendedRgb = FloatArray(3)

        val triCount = tris.size / 3
        for (t in 0 until triCount) {
            val i0 = tris[t * 3].toInt() and 0xFFFF
            val i1 = tris[t * 3 + 1].toInt() and 0xFFFF
            val i2 = tris[t * 3 + 2].toInt() and 0xFFFF

            val pAvg = ((protWeights[i0] + protWeights[i1] + protWeights[i2]) / 3f).coerceIn(0f, 1f)
            if (protectAny && pAvg < 0.25f) continue

            // Compute UV centroid to sample map and mask textures deterministically
            val uMid = ((uvs[i0 * 2] + uvs[i1 * 2] + uvs[i2 * 2]) / 3f).coerceIn(0f, 1f)
            val vMid = ((uvs[i0 * 2 + 1] + uvs[i1 * 2 + 1] + uvs[i2 * 2 + 1]) / 3f).coerceIn(0f, 1f)

            var sampleA = 1f
            var sampleR = tintR
            var sampleG = tintG
            var sampleB = tintB

            if (mapBmp != null) {
                val px = (uMid * (mapBmp.width - 1)).roundToInt().coerceIn(0, mapBmp.width - 1)
                val py = (vMid * (mapBmp.height - 1)).roundToInt().coerceIn(0, mapBmp.height - 1)
                val c = mapBmp.getPixel(px, py)
                if (Color.alpha(c) > 0) {
                    sampleA = Color.alpha(c) / 255f
                    sampleR = (Color.red(c) / 255f) * tintR
                    sampleG = (Color.green(c) / 255f) * tintG
                    sampleB = (Color.blue(c) / 255f) * tintB
                }
            }

            if (maskBmp != null) {
                val mx = (uMid * (maskBmp.width - 1)).roundToInt().coerceIn(0, maskBmp.width - 1)
                val my = (vMid * (maskBmp.height - 1)).roundToInt().coerceIn(0, maskBmp.height - 1)
                val mc = maskBmp.getPixel(mx, my)
                val maskWeight = max(Color.red(mc), Color.alpha(mc)) / 255f
                if (maskWeight > 0f) {
                    sampleA *= maskWeight
                }
            }

            if (protectAny) {
                var dynamicProtection = pAvg
                if (protectMouthInterior && mouthOpenWeight > 0.01f && vMid >= 0.62f && vMid <= 0.88f && uMid >= 0.34f && uMid <= 0.66f) {
                    dynamicProtection *= (1f - mouthOpenWeight * 0.90f).coerceIn(0f, 1f)
                }
                if (protectEyes && eyeBlinkWeight > 0.01f && vMid >= 0.28f && vMid <= 0.50f &&
                    ((uMid >= 0.16f && uMid <= 0.44f) || (uMid >= 0.56f && uMid <= 0.84f))
                ) {
                    dynamicProtection *= (1f - eyeBlinkWeight * 0.85f).coerceIn(0f, 1f)
                }
                sampleA *= dynamicProtection
            }

            val finalAlpha = (sampleA * effectiveOpacity).coerceIn(0f, 1f)
            val triAlpha = (finalAlpha * 255f).roundToInt().coerceIn(0, 255)
            if (triAlpha <= 2) continue

            val x0 = screenVerts[i0 * 2]
            val y0 = screenVerts[i0 * 2 + 1]
            val x1 = screenVerts[i1 * 2]
            val y1 = screenVerts[i1 * 2 + 1]
            val x2 = screenVerts[i2 * 2]
            val y2 = screenVerts[i2 * 2 + 1]

            val cx = ((x0 + x1 + x2) / 3f).roundToInt().coerceIn(0, width - 1)
            val cy = ((y0 + y1 + y2) / 3f).roundToInt().coerceIn(0, height - 1)
            val beforeC = outputBmp.getPixel(cx, cy)

            if (blendMode != com.example.engine.render.FaceMeshRenderer.BlendMode.ALPHA &&
                blendMode != com.example.engine.render.FaceMeshRenderer.BlendMode.OPAQUE
            ) {
                scratchBaseRgb[0] = Color.red(beforeC) / 255f
                scratchBaseRgb[1] = Color.green(beforeC) / 255f
                scratchBaseRgb[2] = Color.blue(beforeC) / 255f
                scratchSrcRgb[0] = sampleR
                scratchSrcRgb[1] = sampleG
                scratchSrcRgb[2] = sampleB
                com.example.engine.render.FaceMeshRenderer.blendRgbWithBase(
                    baseRgb = scratchBaseRgb,
                    srcRgb = scratchSrcRgb,
                    alpha = 1f,
                    blendMode = blendMode,
                    outRgb = scratchBlendedRgb
                )
                sampleR = scratchBlendedRgb[0]
                sampleG = scratchBlendedRgb[1]
                sampleB = scratchBlendedRgb[2]
            }

            val triColor = Color.argb(
                triAlpha,
                (sampleR * 255f).roundToInt().coerceIn(0, 255),
                (sampleG * 255f).roundToInt().coerceIn(0, 255),
                (sampleB * 255f).roundToInt().coerceIn(0, 255)
            )
            paint.color = triColor

            triPath.reset()
            triPath.moveTo(x0, y0)
            triPath.lineTo(x1, y1)
            triPath.lineTo(x2, y2)
            triPath.close()
            canvas.drawPath(triPath, paint)

            if (outputBmp.getPixel(cx, cy) == beforeC) {
                blendPixelPatchIntoBitmap(outputBmp, cx, cy, 2, triColor)
            }
        }

        mapBmp?.recycle()
        maskBmp?.recycle()
        return allocated
    }

    private fun drawCpuModelNode(
        outputBmp: Bitmap,
        canvas: Canvas,
        node: CompiledNode,
        effect: CompiledEffect,
        evaluator: SceneEvaluator,
        worldOffset: Int,
        opacity: Float,
        mat: CompiledMaterial?,
        width: Int,
        height: Int,
        isFrontCamera: Boolean = false,
        maxTexSize: Int
    ): Long {
        val asset = node.modelAssetId?.let { effect.assets[it] } ?: return -1L
        val file = File(asset.localPath)
        val glbModel = if (file.exists()) renderGraph.glbLoader.load(file) else null

        val effectiveMat = mat ?: com.example.engine.render.ModelRenderer.resolvePrimaryModelMaterial(node, effect.materials)
        val effectiveAlpha = ((opacity * (effectiveMat?.opacity ?: 1f)).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val baseColor = effectiveMat?.color ?: floatArrayOf(0.88f, 0.72f, 0.22f, 1f)
        val r = (baseColor.getOrElse(0) { 0.88f }.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val g = (baseColor.getOrElse(1) { 0.72f }.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val b = (baseColor.getOrElse(2) { 0.22f }.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
        val modelColor = Color.argb(effectiveAlpha, r, g, b)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = modelColor
        }

        val triPath = Path()
        var drewMesh = false
        val lighting = effect.lighting ?: com.example.engine.scene.CompiledLighting()
        val scratchAlbedo = FloatArray(3)
        val scratchNormal = FloatArray(3)
        val scratchLitRgb = FloatArray(3)
        val scratchEmissiveMap = FloatArray(3)

        if (glbModel != null && glbModel.meshes.isNotEmpty()) {
            val meshMvp = FloatArray(16)
            val nodeWorld = FloatArray(16)
            System.arraycopy(evaluator.worldMatrices, worldOffset, nodeWorld, 0, 16)

            // Evaluate GLB node hierarchy world matrices
            for (glbNode in glbModel.nodes) {
                val t = glbNode.baseTranslation
                val q = glbNode.baseRotation
                val s = glbNode.baseScale
                val m = glbNode.localMatrix
                val qx = q[0]; val qy = q[1]; val qz = q[2]; val qw = q[3]
                val sx = s[0].coerceAtLeast(0.001f)
                val sy = s[1].coerceAtLeast(0.001f)
                val sz = s[2].coerceAtLeast(0.001f)
                val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
                val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
                val wx = qw * qx; val wy = qw * qy; val wz = qw * qz
                m[0] = (1f - 2f * (yy + zz)) * sx; m[1] = (2f * (xy + wz)) * sx; m[2] = (2f * (xz - wy)) * sx; m[3] = 0f
                m[4] = (2f * (xy - wz)) * sy; m[5] = (1f - 2f * (xx + zz)) * sy; m[6] = (2f * (yz + wx)) * sy; m[7] = 0f
                m[8] = (2f * (xz + wy)) * sz; m[9] = (2f * (yz - wx)) * sz; m[10] = (1f - 2f * (xx + yy)) * sz; m[11] = 0f
                m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1f

                val pIdx = glbNode.parentIndex
                if (pIdx in glbModel.nodes.indices) {
                    Matrix.multiplyMM(glbNode.worldMatrix, 0, glbModel.nodes[pIdx].worldMatrix, 0, glbNode.localMatrix, 0)
                } else {
                    Matrix.multiplyMM(glbNode.worldMatrix, 0, nodeWorld, 0, glbNode.localMatrix, 0)
                }
            }

            val activeTier = effect.activeTier.uppercase()

            fun isTierVisible(nodeName: String): Boolean {
                val lower = nodeName.lowercase()
                return when (activeTier) {
                    "A" -> !lower.contains("lod_b") && !lower.contains("lod_c")
                    "B" -> !lower.contains("lod_a") && !lower.contains("lod_c")
                    "C" -> !lower.contains("lod_a") && !lower.contains("lod_b")
                    else -> true
                }
            }

            fun isMeshTranslucent(mesh: com.example.engine.render.GlbLoader.GlbMesh, subnodeName: String): Boolean {
                val glbMat = if (mesh.materialIndex in glbModel.materials.indices) glbModel.materials[mesh.materialIndex] else null
                val subMat = com.example.engine.render.ModelRenderer.resolveSubnodeMaterial(
                    node = node,
                    subnodeName = subnodeName,
                    glbMatName = glbMat?.name ?: "",
                    primaryMaterial = effectiveMat,
                    materials = effect.materials
                )
                val alphaMode = (glbMat?.alphaMode ?: "OPAQUE").uppercase()
                val blendMode = (subMat?.blend ?: "opaque").lowercase()
                val matOp = (subMat?.opacity ?: 1f) * (glbMat?.baseColorFactor?.getOrElse(3) { 1f } ?: 1f)
                return blendMode != "opaque" || alphaMode == "BLEND" || matOp < 0.999f || opacity < 0.999f
            }

            fun drawSingleMesh(mesh: com.example.engine.render.GlbLoader.GlbMesh, subnodeName: String, worldMat: FloatArray, mvpMat: FloatArray) {
                val pos = mesh.cpuPositions
                val norms = mesh.cpuNormals
                val uvs = mesh.cpuTexCoords
                val idx = mesh.cpuIndices
                val vertCount = pos.size / 3
                if (vertCount < 3) return

                val glbMat = if (mesh.materialIndex in glbModel.materials.indices) glbModel.materials[mesh.materialIndex] else null
                val subMat = com.example.engine.render.ModelRenderer.resolveSubnodeMaterial(
                    node = node,
                    subnodeName = subnodeName,
                    glbMatName = glbMat?.name ?: "",
                    primaryMaterial = effectiveMat,
                    materials = effect.materials
                )
                val usePbr = (subMat?.template == "pbr@1") ||
                    (subMat?.template != "unlit@1" && com.example.engine.render.ModelRenderer.hasActiveGlbPbr(glbModel, subMat, glbMat))
                val useLambert = !usePbr && (subMat?.template == "lambert@1")

                val matCol = subMat?.color
                val glbCol = glbMat?.baseColorFactor
                val factorR = (matCol?.getOrElse(0) { 1f } ?: 1f) * (glbCol?.getOrElse(0) { 1f } ?: 1f)
                val factorG = (matCol?.getOrElse(1) { 1f } ?: 1f) * (glbCol?.getOrElse(1) { 1f } ?: 1f)
                val factorB = (matCol?.getOrElse(2) { 1f } ?: 1f) * (glbCol?.getOrElse(2) { 1f } ?: 1f)
                val factorA = (glbCol?.getOrElse(3) { 1f } ?: 1f)
                val subOpacity = (opacity * (subMat?.opacity ?: 1f) * factorA).coerceIn(0f, 1f)
                val meshAlpha = (subOpacity * 255f).roundToInt().coerceIn(0, 255)
                if (meshAlpha <= 2) return

                val alphaMode = (glbMat?.alphaMode ?: "OPAQUE").uppercase()
                val alphaCutoff = glbMat?.alphaCutoff ?: 0.5f

                val texIdx = when {
                    glbMat != null && glbMat.baseColorTextureIndex in glbModel.cpuTextureBitmaps.indices -> glbMat.baseColorTextureIndex
                    mesh.materialIndex in glbModel.cpuTextureBitmaps.indices -> mesh.materialIndex
                    glbModel.cpuTextureBitmaps.isNotEmpty() -> 0
                    else -> -1
                }
                val texBmp = if (texIdx >= 0) glbModel.cpuTextureBitmaps[texIdx] else null
                val mrBmp = if (glbMat != null && glbMat.metallicRoughnessTextureIndex in glbModel.cpuTextureBitmaps.indices) {
                    glbModel.cpuTextureBitmaps[glbMat.metallicRoughnessTextureIndex]
                } else null
                val occBmp = if (glbMat != null && glbMat.occlusionTextureIndex in glbModel.cpuTextureBitmaps.indices) {
                    glbModel.cpuTextureBitmaps[glbMat.occlusionTextureIndex]
                } else null
                val emBmp = if (glbMat != null && glbMat.emissiveTextureIndex in glbModel.cpuTextureBitmaps.indices) {
                    glbModel.cpuTextureBitmaps[glbMat.emissiveTextureIndex]
                } else null

                val met = when {
                    subMat != null && subMat.hasExplicitMetallic -> subMat.metallic
                    glbMat != null -> glbMat.metallicFactor
                    subMat != null -> subMat.metallic
                    else -> 0.0f
                }
                val rough = when {
                    subMat != null && subMat.hasExplicitRoughness -> subMat.roughness
                    glbMat != null -> glbMat.roughnessFactor
                    subMat != null -> subMat.roughness
                    else -> 0.45f
                }
                val em = when {
                    subMat != null && subMat.hasExplicitEmissive -> subMat.emissive
                    glbMat != null -> glbMat.emissiveFactor
                    subMat != null -> subMat.emissive
                    else -> floatArrayOf(0f, 0f, 0f)
                }

                fun sampleBitmapAtUv(bmp: Bitmap?, u: Float, v: Float): Int? {
                    if (bmp == null || bmp.isRecycled || bmp.width <= 0 || bmp.height <= 0) return null
                    val px = (u * (bmp.width - 1)).roundToInt().coerceIn(0, bmp.width - 1)
                    val py = (v * (bmp.height - 1)).roundToInt().coerceIn(0, bmp.height - 1)
                    return bmp.getPixel(px, py)
                }

                fun shadeAndDrawTriangle(i0: Int, i1: Int, i2: Int) {
                    val p0 = projectLocalPointToScreen(mvpMat, pos[i0 * 3], pos[i0 * 3 + 1], pos[i0 * 3 + 2], width, height) ?: return
                    val p1 = projectLocalPointToScreen(mvpMat, pos[i1 * 3], pos[i1 * 3 + 1], pos[i1 * 3 + 2], width, height) ?: return
                    val p2 = projectLocalPointToScreen(mvpMat, pos[i2 * 3], pos[i2 * 3 + 1], pos[i2 * 3 + 2], width, height) ?: return

                    var sR = factorR
                    var sG = factorG
                    var sB = factorB
                    var sA = subOpacity
                    var hasUv = false
                    var uMid = 0.5f
                    var vMid = 0.5f
                    if (uvs.size >= (maxOf(i0, i1, i2) + 1) * 2) {
                        hasUv = true
                        uMid = (uvs[i0 * 2] + uvs[i1 * 2] + uvs[i2 * 2]) / 3f
                        vMid = (uvs[i0 * 2 + 1] + uvs[i1 * 2 + 1] + uvs[i2 * 2 + 1]) / 3f
                        uMid -= kotlin.math.floor(uMid)
                        vMid -= kotlin.math.floor(vMid)
                    }
                    if (hasUv) {
                        val c = sampleBitmapAtUv(texBmp, uMid, vMid)
                        if (c != null) {
                            sR *= (Color.red(c) / 255f)
                            sG *= (Color.green(c) / 255f)
                            sB *= (Color.blue(c) / 255f)
                            sA *= (Color.alpha(c) / 255f)
                        }
                    }
                    if (alphaMode == "MASK" && sA < alphaCutoff) return
                    val triAlpha = if (alphaMode == "MASK") 255 else (sA.coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                    if (triAlpha <= 2) return

                    var triMet = met
                    var triRough = rough
                    var triAo = 1f
                    var hasEmMap = false
                    if (hasUv) {
                        val mrC = sampleBitmapAtUv(mrBmp, uMid, vMid)
                        if (mrC != null) {
                            triMet = (triMet * (Color.blue(mrC) / 255f)).coerceIn(0f, 1f)
                            triRough = (triRough * (Color.green(mrC) / 255f)).coerceIn(0.04f, 1f)
                        }
                        val occC = sampleBitmapAtUv(occBmp, uMid, vMid)
                        if (occC != null) {
                            triAo = (Color.red(occC) / 255f).coerceIn(0f, 1f)
                        }
                        val emC = sampleBitmapAtUv(emBmp, uMid, vMid)
                        if (emC != null) {
                            scratchEmissiveMap[0] = Color.red(emC) / 255f
                            scratchEmissiveMap[1] = Color.green(emC) / 255f
                            scratchEmissiveMap[2] = Color.blue(emC) / 255f
                            hasEmMap = true
                        }
                    }

                    if (usePbr || useLambert) {
                        var lnx = 0f; var lny = 0.25f; var lnz = 0.968f
                        if (norms.size >= (maxOf(i0, i1, i2) + 1) * 3) {
                            lnx = (norms[i0 * 3] + norms[i1 * 3] + norms[i2 * 3]) / 3f
                            lny = (norms[i0 * 3 + 1] + norms[i1 * 3 + 1] + norms[i2 * 3 + 1]) / 3f
                            lnz = (norms[i0 * 3 + 2] + norms[i1 * 3 + 2] + norms[i2 * 3 + 2]) / 3f
                        }
                        scratchNormal[0] = worldMat[0] * lnx + worldMat[4] * lny + worldMat[8] * lnz
                        scratchNormal[1] = worldMat[1] * lnx + worldMat[5] * lny + worldMat[9] * lnz
                        scratchNormal[2] = worldMat[2] * lnx + worldMat[6] * lny + worldMat[10] * lnz
                        if (isFrontCamera) {
                            scratchNormal[0] = -scratchNormal[0]
                        }
                        if (scratchNormal[2] < 0f && ((subMat?.doubleSided ?: true) || (glbMat?.doubleSided ?: true))) {
                            scratchNormal[0] = -scratchNormal[0]
                            scratchNormal[1] = -scratchNormal[1]
                            scratchNormal[2] = -scratchNormal[2]
                        }
                        scratchAlbedo[0] = sR
                        scratchAlbedo[1] = sG
                        scratchAlbedo[2] = sB
                        val sheen = subMat?.fresnelSheen ?: 0.15f
                        if (usePbr) {
                            com.example.engine.render.MaterialRegistry.evaluatePbrColor(
                                albedoRgb = scratchAlbedo,
                                normal = scratchNormal,
                                metallic = triMet,
                                roughness = triRough,
                                emissiveRgb = em,
                                lighting = lighting,
                                outRgb = scratchLitRgb,
                                occlusion = triAo,
                                fresnelSheen = sheen,
                                emissiveMapRgb = if (hasEmMap) scratchEmissiveMap else null
                            )
                        } else {
                            com.example.engine.render.MaterialRegistry.evaluateLambertColor(
                                texRgb = scratchAlbedo,
                                normal = scratchNormal,
                                lighting = lighting,
                                outRgb = scratchLitRgb,
                                fresnelSheen = sheen,
                                includeSpecular = true
                            )
                        }
                        sR = scratchLitRgb[0]
                        sG = scratchLitRgb[1]
                        sB = scratchLitRgb[2]
                    }

                    paint.color = Color.argb(
                        triAlpha,
                        (sR * 255f).roundToInt().coerceIn(0, 255),
                        (sG * 255f).roundToInt().coerceIn(0, 255),
                        (sB * 255f).roundToInt().coerceIn(0, 255)
                    )
                    triPath.reset()
                    triPath.moveTo(p0[0], p0[1])
                    triPath.lineTo(p1[0], p1[1])
                    triPath.lineTo(p2[0], p2[1])
                    triPath.close()
                    canvas.drawPath(triPath, paint)
                    drewMesh = true
                }

                if (idx != null && idx.size >= 3) {
                    val triCount = idx.size / 3
                    for (t in 0 until triCount) {
                        val i0 = (idx[t * 3].toInt() and 0xFFFF).coerceIn(0, vertCount - 1)
                        val i1 = (idx[t * 3 + 1].toInt() and 0xFFFF).coerceIn(0, vertCount - 1)
                        val i2 = (idx[t * 3 + 2].toInt() and 0xFFFF).coerceIn(0, vertCount - 1)
                        shadeAndDrawTriangle(i0, i1, i2)
                    }
                } else {
                    val triCount = vertCount / 3
                    for (t in 0 until triCount) {
                        shadeAndDrawTriangle(t * 3, t * 3 + 1, t * 3 + 2)
                    }
                }
            }

            if (glbModel.nodes.isNotEmpty()) {
                for (translucentPass in 0..1) {
                    val wantTranslucent = translucentPass == 1
                    for (glbNode in glbModel.nodes) {
                        if (!isTierVisible(glbNode.name)) continue
                        val primitives = glbModel.meshesForGltfMeshIndex(glbNode.meshIndex)
                        if (primitives.isNotEmpty()) {
                            Matrix.multiplyMM(meshMvp, 0, viewProjMatrix, 0, glbNode.worldMatrix, 0)
                            for (mesh in primitives) {
                                if (isMeshTranslucent(mesh, glbNode.name) == wantTranslucent) {
                                    drawSingleMesh(mesh, glbNode.name, glbNode.worldMatrix, meshMvp)
                                }
                            }
                        }
                    }
                }
            } else {
                System.arraycopy(nodeMvp, 0, meshMvp, 0, 16)
                for (translucentPass in 0..1) {
                    val wantTranslucent = translucentPass == 1
                    for (mesh in glbModel.meshes) {
                        if (isMeshTranslucent(mesh, "") == wantTranslucent) {
                            drawSingleMesh(mesh, "", nodeWorld, meshMvp)
                        }
                    }
                }
            }
            glbModel.release()
        }

        // Also ensure a visible projected core at the model's 3D anchor
        val center = projectLocalPointToScreen(nodeMvp, 0f, 0f, 0f, width, height) ?: return -1L
        val edge = projectLocalPointToScreen(nodeMvp, 0.4f, 0f, 0f, width, height)
        val radiusPx = if (edge != null) {
            hypot(edge[0] - center[0], edge[1] - center[1]).coerceIn(12f, width * 0.35f)
        } else {
            (width * 0.06f).coerceAtLeast(12f)
        }
        val probeX = center[0].roundToInt().coerceIn(0, width - 1)
        val probeY = center[1].roundToInt().coerceIn(0, height - 1)
        val beforeCenter = outputBmp.getPixel(probeX, probeY)
        if (!drewMesh) {
            canvas.drawCircle(center[0], center[1], radiusPx, paint)
        }
        if (outputBmp.getPixel(probeX, probeY) == beforeCenter && effectiveAlpha > 4) {
            blendPixelPatchIntoBitmap(outputBmp, probeX, probeY, radiusPx.roundToInt().coerceIn(10, 48), paint.color)
        }
        return 0L
    }

    private fun cloneBitmapWithPixels(source: Bitmap): Bitmap {
        val w = source.width.coerceAtLeast(1)
        val h = source.height.coerceAtLeast(1)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        return try {
            val pixels = IntArray(w * h)
            source.getPixels(pixels, 0, w, 0, 0, w, h)
            out.setPixels(pixels, 0, w, 0, 0, w, h)
            out
        } catch (_: Throwable) {
            out
        }
    }

    private fun rasterizeRectIntoBitmap(
        target: Bitmap,
        rect: RectF,
        spriteBmp: Bitmap?,
        fallbackColor: Int,
        effectiveAlpha: Int,
        uvRect4: FloatArray? = null,
        tintR: Int = 255,
        tintG: Int = 255,
        tintB: Int = 255,
        blendMode: String = "alpha"
    ) {
        val w = target.width
        val h = target.height
        val xStart = rect.left.roundToInt().coerceIn(0, w - 1)
        val xEnd = rect.right.roundToInt().coerceIn(0, w - 1)
        val yStart = rect.top.roundToInt().coerceIn(0, h - 1)
        val yEnd = rect.bottom.roundToInt().coerceIn(0, h - 1)
        val rw = (xEnd - xStart).coerceAtLeast(1)
        val rh = (yEnd - yStart).coerceAtLeast(1)
        val fbU0 = uvRect4?.getOrElse(0) { 0f } ?: 0f
        val fbV0 = uvRect4?.getOrElse(1) { 0f } ?: 0f
        val fbUScale = uvRect4?.getOrElse(2) { 1f } ?: 1f
        val fbVScale = uvRect4?.getOrElse(3) { 1f } ?: 1f

        for (y in yStart..yEnd) {
            val localV = (y - yStart).toFloat() / rh.toFloat()
            val v = (fbV0 + localV * fbVScale).coerceIn(0f, 1f)
            for (x in xStart..xEnd) {
                val localU = (x - xStart).toFloat() / rw.toFloat()
                val u = (fbU0 + localU * fbUScale).coerceIn(0f, 1f)
                val srcColor: Int
                if (spriteBmp != null && spriteBmp.width > 0 && spriteBmp.height > 0) {
                    val sx = (u * spriteBmp.width.toFloat()).toInt().coerceIn(0, spriteBmp.width - 1)
                    val sy = (v * spriteBmp.height.toFloat()).toInt().coerceIn(0, spriteBmp.height - 1)
                    val c = spriteBmp.getPixel(sx, sy)
                    val texA = Color.alpha(c)
                    if (texA <= 0) continue
                    val a = ((texA / 255f) * effectiveAlpha).roundToInt().coerceIn(0, 255)
                    val r = ((Color.red(c) * tintR) / 255).coerceIn(0, 255)
                    val g = ((Color.green(c) * tintG) / 255).coerceIn(0, 255)
                    val b = ((Color.blue(c) * tintB) / 255).coerceIn(0, 255)
                    srcColor = Color.argb(a, r, g, b)
                } else {
                    srcColor = fallbackColor
                }
                blendSinglePixel(target, x, y, srcColor, blendMode)
            }
        }
    }

    private fun blendPixelPatchIntoBitmap(target: Bitmap, cx: Int, cy: Int, radius: Int, srcColor: Int) {
        val w = target.width
        val h = target.height
        val x0 = (cx - radius).coerceIn(0, w - 1)
        val x1 = (cx + radius).coerceIn(0, w - 1)
        val y0 = (cy - radius).coerceIn(0, h - 1)
        val y1 = (cy + radius).coerceIn(0, h - 1)
        for (y in y0..y1) {
            for (x in x0..x1) {
                blendSinglePixel(target, x, y, srcColor)
            }
        }
    }

    private fun blendSinglePixel(target: Bitmap, x: Int, y: Int, srcColor: Int, blendMode: String = "alpha") {
        val srcA = Color.alpha(srcColor)
        if (srcA <= 0) return
        val dstColor = target.getPixel(x, y)
        val dstR = Color.red(dstColor)
        val dstG = Color.green(dstColor)
        val dstB = Color.blue(dstColor)
        val srcR = Color.red(srcColor)
        val srcG = Color.green(srcColor)
        val srcB = Color.blue(srcColor)

        val (blendR, blendG, blendB) = when (blendMode.lowercase()) {
            "additive", "add" -> Triple(
                (dstR + srcR).coerceAtMost(255),
                (dstG + srcG).coerceAtMost(255),
                (dstB + srcB).coerceAtMost(255)
            )
            "screen" -> Triple(
                255 - ((255 - dstR) * (255 - srcR)) / 255,
                255 - ((255 - dstG) * (255 - srcG)) / 255,
                255 - ((255 - dstB) * (255 - srcB)) / 255
            )
            "multiply" -> Triple(
                (dstR * srcR) / 255,
                (dstG * srcG) / 255,
                (dstB * srcB) / 255
            )
            else -> Triple(srcR, srcG, srcB)
        }

        val invA = 255 - srcA
        val outR = (blendR * srcA + dstR * invA) / 255
        val outG = (blendG * srcA + dstG * invA) / 255
        val outB = (blendB * srcA + dstB * invA) / 255
        val outA = max(srcA, Color.alpha(dstColor))
        target.setPixel(x, y, Color.argb(outA.coerceIn(0, 255), outR.coerceIn(0, 255), outG.coerceIn(0, 255), outB.coerceIn(0, 255)))
    }

    /**
     * Projects a 3D point `(lx, ly, lz, 1)` through `mvpMatrix` into 2D photo pixel coordinates `(px, py)`.
     */
    fun projectLocalPointToScreen(
        mvpMatrix: FloatArray,
        lx: Float,
        ly: Float,
        lz: Float,
        viewportWidth: Int,
        viewportHeight: Int
    ): FloatArray? {
        worldVec[0] = lx
        worldVec[1] = ly
        worldVec[2] = lz
        worldVec[3] = 1.0f
        Matrix.multiplyMV(clipVec, 0, mvpMatrix, 0, worldVec, 0)
        val w = clipVec[3]
        if (w <= 0.0001f || !w.isFinite()) return null
        val ndcX = clipVec[0] / w
        val ndcY = clipVec[1] / w
        val screenX = (ndcX * 0.5f + 0.5f) * viewportWidth.toFloat()
        val screenY = (0.5f - ndcY * 0.5f) * viewportHeight.toFloat()
        return floatArrayOf(screenX, screenY, clipVec[2] / w)
    }

    private fun countVisibleRenderNodes(effect: CompiledEffect, evaluator: SceneEvaluator): Int {
        var count = 0
        for (i in effect.nodes.indices) {
            val n = effect.nodes[i]
            if (evaluator.isNodeVisible[i] && n.type != "occluder" && n.type != "group") {
                val op = effect.propertySlots.opacity[i] * effect.propertySlots.visible[i]
                if (op > 0.001f) count++
            }
        }
        return count
    }

    private fun decodeSampledBitmap(path: String, maxDimension: Int): Bitmap? {
        val file = File(path)
        if (!file.exists()) return null
        return try {
            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(file.absolutePath, boundsOpts)
            var sampleSize = 1
            val maxEdge = max(boundsOpts.outWidth, boundsOpts.outHeight)
            val targetMax = maxDimension.coerceAtLeast(64)
            while (maxEdge / sampleSize > targetMax) {
                sampleSize *= 2
            }
            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            BitmapFactory.decodeFile(file.absolutePath, decodeOpts)
        } catch (_: Throwable) {
            null
        }
    }

    private fun releaseEglOnly() {
        try {
            val disp = eglDisplay
            if (disp != null && disp != EGL14.EGL_NO_DISPLAY) {
                makeCurrent()
                releaseFbo()
                renderGraph.release()
                EGL14.eglMakeCurrent(
                    disp,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_CONTEXT
                )
                val surf = eglSurface
                if (surf != null && surf != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglDestroySurface(disp, surf)
                }
                val ctx = eglContext
                if (ctx != null && ctx != EGL14.EGL_NO_CONTEXT) {
                    EGL14.eglDestroyContext(disp, ctx)
                }
                EGL14.eglTerminate(disp)
            }
        } catch (_: Throwable) {
            // Safe cleanup even on stub/headless JVM
        } finally {
            eglDisplay = null
            eglSurface = null
            eglContext = null
            eglConfig = null
            isEglInitialized = false
            isSharingActive = false
            eglPbufferWidth = 0
            eglPbufferHeight = 0
        }
    }

    @Synchronized
    fun release() {
        releaseEglOnly()
    }
}
