package com.example.engine.render

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.AttributeSet
import android.util.Log
import android.view.Choreographer
import com.example.engine.capture.CameraInputSurface
import com.example.engine.capture.CameraSession
import com.example.engine.capture.FrameClock
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.pkg.PackagePreloader
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.coroutines.resume

/**
 * Transparent GLSurfaceView overlay executing the Xelvic filter runtime.
 */
class XelvicGlSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs), GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "XelvicGlSurfaceView"
    }

    var signalPipeline: SignalPipeline? = null
        set(value) {
            field = value
            value?.isPredictionEnabled = isPredictionEnabled
        }
    var cameraSession: CameraSession? = null
    var preloader: PackagePreloader? = null
    var latencyProbe: LatencyProbe = LatencyProbe.shared
    @Volatile
    var isPredictionEnabled: Boolean = true
        set(value) {
            field = value
            signalPipeline?.isPredictionEnabled = value
        }
    @Volatile
    var useHighRenderThreadPriority: Boolean = false
        private set
    @Volatile
    var lastAppliedRenderThreadPriority: Int = android.os.Process.THREAD_PRIORITY_DEFAULT
        private set

    /**
     * Toggles the GL render thread priority between `NORM` (`Process.THREAD_PRIORITY_DEFAULT`, `0`)
     * and `HIGH` (`Process.THREAD_PRIORITY_DISPLAY`, `-4`).
     */
    fun setUseHighRenderThreadPriority(highPriority: Boolean) {
        useHighRenderThreadPriority = highPriority
        val targetPrio = if (highPriority) {
            android.os.Process.THREAD_PRIORITY_DISPLAY
        } else {
            android.os.Process.THREAD_PRIORITY_DEFAULT
        }
        lastAppliedRenderThreadPriority = targetPrio
        (cameraSession?.latencyProbe ?: latencyProbe).renderThreadPriorityLabel =
            if (highPriority) "HIGH" else "NORM"
        try {
            queueEvent {
                applyRenderThreadPriorityOnGlThread()
            }
        } catch (_: Throwable) {
        }
    }

    @Volatile
    private var lastOsAppliedGlThreadId: Long = -1L
    @Volatile
    private var lastOsAppliedGlThreadPrio: Int = Int.MIN_VALUE

    private fun applyRenderThreadPriorityOnGlThread() {
        val targetPrio = if (useHighRenderThreadPriority) {
            android.os.Process.THREAD_PRIORITY_DISPLAY
        } else {
            android.os.Process.THREAD_PRIORITY_DEFAULT
        }
        lastAppliedRenderThreadPriority = targetPrio
        val tid = Thread.currentThread().id
        if (tid == lastOsAppliedGlThreadId && targetPrio == lastOsAppliedGlThreadPrio) {
            return
        }
        lastOsAppliedGlThreadId = tid
        lastOsAppliedGlThreadPrio = targetPrio
        try {
            android.os.Process.setThreadPriority(targetPrio)
        } catch (_: Throwable) {
        }
    }
    var activeEffect: CompiledEffect? = null
        set(value) {
            field = value
            transparentOverlaySurface.activeEffect = value
            singleSceneSurface.activeEffect = value
            updateCompositionModeForCurrentState()
        }
    var governor: PerformanceGovernor = PerformanceGovernor(context)
        set(value) {
            field = value
            field.onMemoryTrimRequested = {
                renderGraph.textures.trimToFraction(0.5f)
            }
        }

    val renderGraph = RenderGraph()
    private val evaluator = SceneEvaluator(maxNodes = 64)
    val offsetTestHarness = OffsetTestHarness()
    val videoRecorder = com.example.engine.capture.VideoRecorderPipeline()

    private val transparentOverlaySurface = TransparentOverlayCompositionSurface(renderGraph, evaluator)
    private val singleSceneSurface = SingleSceneCompositionSurface(renderGraph, evaluator).apply {
        this.videoRecorder = this@XelvicGlSurfaceView.videoRecorder
    }

    @Volatile
    var isVideoRecordingActive: Boolean = false
        private set

    var onVideoRecordingAutoStopped: ((com.example.engine.capture.VideoRecorderPipeline.RecordingResult) -> Unit)? = null

    /**
     * Explicit PATH:A (`TRANSPARENT_OVERLAY`) vs PATH:B (`SINGLE_SCENE_GL`) preference.
     * Defaults to `TRANSPARENT_OVERLAY` (`PATH:A`) per Chunk B0 requirement #7.
     */
    @Volatile
    var requestedPathMode: CompositionMode = CompositionMode.TRANSPARENT_OVERLAY
        private set

    @Volatile
    var compositionSurface: CompositionSurface = transparentOverlaySurface
        private set

    val hasLiveCameraSceneInput: Boolean
        get() = singleSceneSurface.hasCameraSceneInput

    val isCameraTextureOk: Boolean
        get() = singleSceneSurface.isCameraTextureOk

    val latestCameraSensorTimestampNs: Long
        get() = singleSceneSurface.latestCameraSensorTimestampNs

    fun isPostSkippedInLivePreview(effect: CompiledEffect? = activeEffect): Boolean {
        return effect != null &&
            effect.nodes.any { it.type == "post" } &&
            compositionSurface.mode == CompositionMode.TRANSPARENT_OVERLAY
    }

    fun setRequestedPathMode(mode: CompositionMode) {
        requestedPathMode = mode
        (cameraSession?.latencyProbe ?: latencyProbe).pathModeLabel =
            if (mode == CompositionMode.SINGLE_SCENE_GL) "B" else "A"
        updateCompositionModeForCurrentState()
    }

    /**
     * Starts real-time GPU + AAC audio video recording from `SingleSceneCompositionSurface.sceneFbo`,
     * automatically engaging `SINGLE_SCENE_GL` (`Path B`) so the camera background + 3D AR + Beauty
     * passes are composed into `sceneFbo`.
     */
    fun startVideoRecording(config: com.example.engine.capture.VideoRecorderPipeline.RecordingConfig): Boolean {
        val started = videoRecorder.startRecording(config)
        if (started) {
            isVideoRecordingActive = true
            updateCompositionModeForCurrentState()
        }
        return started
    }

    /**
     * Stops active video recording, finalizes the MP4 container and audio track, and restores the
     * composition mode for the current state.
     */
    fun stopVideoRecording(): com.example.engine.capture.VideoRecorderPipeline.RecordingResult {
        isVideoRecordingActive = false
        val result = videoRecorder.stopRecording()
        updateCompositionModeForCurrentState()
        return result
    }

    private fun finalizeActiveRecordingOnLifecycleInterruption() {
        if (isVideoRecordingActive || videoRecorder.state == com.example.engine.capture.VideoRecorderPipeline.State.RECORDING) {
            isVideoRecordingActive = false
            val res = videoRecorder.stopRecording()
            onVideoRecordingAutoStopped?.invoke(res)
        }
    }

    private fun updateCompositionModeForCurrentState() {
        val recordingNow = isVideoRecordingActive ||
            videoRecorder.state == com.example.engine.capture.VideoRecorderPipeline.State.RECORDING
        val targetMode = if (requestedPathMode == CompositionMode.SINGLE_SCENE_GL || recordingNow) {
            if (singleSceneSurface.hasCameraSceneInput || singleSceneSurface.cameraInputSurface != null) {
                CompositionMode.SINGLE_SCENE_GL
            } else {
                CompositionMode.TRANSPARENT_OVERLAY
            }
        } else {
            val hasExplicitDirectInput =
                singleSceneSurface.cameraTextureId != 0 && singleSceneSurface.cameraInputSurface == null
            OffsetTestHarness.selectCompositionMode(
                effect = activeEffect,
                requiresVideoRecording = recordingNow,
                offsetDecision = offsetTestHarness.lastDecision,
                hasCameraSceneInput = hasExplicitDirectInput || recordingNow
            )
        }
        if (compositionSurface.mode != targetMode) {
            setCompositionMode(targetMode)
        }
    }

    private var viewportWidth = 1080
    private var viewportHeight = 1920
    private var lastFrameTimeNs = 0L
    private var renderStartTimeNs = FrameClock.elapsedRealtimeNs()

    // Main-thread Choreographer vsync tracking for frame-to-frame interval
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile
    var isChoreographerRegistered: Boolean = false
        private set
    @Volatile
    private var latestChoreographerFrameTimeNs: Long = 0L
    private var lastConsumedVsyncNs: Long = 0L
    private var glFramesSinceLastVsync: Int = 0

    @Volatile
    var startupRenderModeLogged: Boolean = false
        private set

    private val choreographerCallback = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            if (!isChoreographerRegistered) return
            latestChoreographerFrameTimeNs = frameTimeNanos
            try {
                Choreographer.getInstance().postFrameCallback(this)
            } catch (_: Throwable) {
            }
        }
    }

    var onFpsCalculated: ((Float, String) -> Unit)? = null
    private var frameCount = 0
    private var lastFpsTimeMs = SystemClock.uptimeMillis()

    private val pendingSnapshotCallback = AtomicReference<((Bitmap?) -> Unit)?>(null)
    @Volatile
    private var lastValidFaceState: FaceState? = null

    @Volatile
    var actualDepthBits: Int = 24
        private set

    init {
        governor.onMemoryTrimRequested = {
            renderGraph.textures.trimToFraction(0.5f)
        }
        setEGLContextClientVersion(3)
        // Request 8-bit RGBA with 24-bit depth (falling back to 16-bit depth if 24-bit is unavailable)
        setEGLConfigChooser(Depth24Fallback16ConfigChooser { chosenDepth ->
            actualDepthBits = chosenDepth
            (cameraSession?.latencyProbe ?: latencyProbe).recordDepthBits(chosenDepth)
        })
        holder.setFormat(PixelFormat.TRANSLUCENT)
        // Use ZOrderMediaOverlay so XelvicGlSurfaceView (-1) composites above PreviewView's
        // hardware SurfaceView (-2) while remaining strictly below the Main App Window (0)
        // where all Jetpack Compose UI controls and diagnostic overlays are rendered.
        setZOrderMediaOverlay(true)
        setWillNotDraw(false)
        preserveEGLContextOnPause = true
        setRenderer(this)
        renderMode = RENDERMODE_CONTINUOUSLY
        logRenderModeAtStartup()
        registerChoreographerOnMainThread()
    }

    override fun draw(canvas: android.graphics.Canvas) {
        // Explicitly punch a transparent hole in the Activity Window (Layer 0) so XelvicGlSurfaceView's
        // hardware surface at ZOrderMediaOverlay (Layer -1) is always visible on all Android ROMs,
        // while Jetpack Compose UI and diagnostic overlays rendered later in Layer 0 composite cleanly on top.
        canvas.drawColor(android.graphics.Color.TRANSPARENT, android.graphics.PorterDuff.Mode.CLEAR)
        super.draw(canvas)
    }

    /**
     * Requests a 24-bit depth buffer for RGBA8888 ES 3.0 rendering, falling back cleanly to 16-bit
     * depth on GPUs that do not expose a 24-bit translucent window config.
     */
    internal class Depth24Fallback16ConfigChooser(
        private val onDepthChosen: ((Int) -> Unit)? = null
    ) : GLSurfaceView.EGLConfigChooser {
        companion object {
            private const val EGL_OPENGL_ES2_BIT = 4
            private const val EGL_OPENGL_ES3_BIT_KHR = 0x0040
        }

        var chosenDepthBits: Int = 24
            private set

        override fun chooseConfig(
            egl: javax.microedition.khronos.egl.EGL10,
            display: javax.microedition.khronos.egl.EGLDisplay
        ): EGLConfig {
            val picked24 = tryChoose(egl, display, minDepthBits = 24, renderableType = EGL_OPENGL_ES3_BIT_KHR)
                ?: tryChoose(egl, display, minDepthBits = 24, renderableType = EGL_OPENGL_ES2_BIT)
            if (picked24 != null) {
                chosenDepthBits = picked24.second
                onDepthChosen?.invoke(chosenDepthBits)
                return picked24.first
            }

            val picked16 = tryChoose(egl, display, minDepthBits = 16, renderableType = EGL_OPENGL_ES3_BIT_KHR)
                ?: tryChoose(egl, display, minDepthBits = 16, renderableType = EGL_OPENGL_ES2_BIT)
                ?: throw IllegalArgumentException("No matching RGBA8888 EGLConfig with 24-bit or 16-bit depth")
            chosenDepthBits = picked16.second
            onDepthChosen?.invoke(chosenDepthBits)
            return picked16.first
        }

        private fun tryChoose(
            egl: javax.microedition.khronos.egl.EGL10,
            display: javax.microedition.khronos.egl.EGLDisplay,
            minDepthBits: Int,
            renderableType: Int
        ): Pair<EGLConfig, Int>? {
            val spec = intArrayOf(
                javax.microedition.khronos.egl.EGL10.EGL_RED_SIZE, 8,
                javax.microedition.khronos.egl.EGL10.EGL_GREEN_SIZE, 8,
                javax.microedition.khronos.egl.EGL10.EGL_BLUE_SIZE, 8,
                javax.microedition.khronos.egl.EGL10.EGL_ALPHA_SIZE, 8,
                javax.microedition.khronos.egl.EGL10.EGL_DEPTH_SIZE, minDepthBits,
                javax.microedition.khronos.egl.EGL10.EGL_RENDERABLE_TYPE, renderableType,
                javax.microedition.khronos.egl.EGL10.EGL_NONE
            )
            val numConfig = IntArray(1)
            if (!egl.eglChooseConfig(display, spec, null, 0, numConfig)) return null
            val count = numConfig[0]
            if (count <= 0) return null

            val configs = arrayOfNulls<EGLConfig>(count)
            if (!egl.eglChooseConfig(display, spec, configs, count, numConfig)) return null

            val valScratch = IntArray(1)
            for (cfg in configs) {
                if (cfg == null) continue
                val d = findAttr(egl, display, cfg, javax.microedition.khronos.egl.EGL10.EGL_DEPTH_SIZE, valScratch)
                val r = findAttr(egl, display, cfg, javax.microedition.khronos.egl.EGL10.EGL_RED_SIZE, valScratch)
                val g = findAttr(egl, display, cfg, javax.microedition.khronos.egl.EGL10.EGL_GREEN_SIZE, valScratch)
                val b = findAttr(egl, display, cfg, javax.microedition.khronos.egl.EGL10.EGL_BLUE_SIZE, valScratch)
                val a = findAttr(egl, display, cfg, javax.microedition.khronos.egl.EGL10.EGL_ALPHA_SIZE, valScratch)
                if (d >= minDepthBits && r == 8 && g == 8 && b == 8 && a == 8) {
                    return cfg to d
                }
            }
            val first = configs.firstOrNull { it != null } ?: return null
            val d = findAttr(egl, display, first, javax.microedition.khronos.egl.EGL10.EGL_DEPTH_SIZE, valScratch)
                .let { if (it > 0) it else minDepthBits }
            return first to d
        }

        private fun findAttr(
            egl: javax.microedition.khronos.egl.EGL10,
            display: javax.microedition.khronos.egl.EGLDisplay,
            config: EGLConfig,
            attribute: Int,
            scratch: IntArray
        ): Int {
            return if (egl.eglGetConfigAttrib(display, config, attribute, scratch)) scratch[0] else 0
        }
    }

    private fun logRenderModeAtStartup() {
        val modeName = if (renderMode == RENDERMODE_CONTINUOUSLY) {
            "RENDERMODE_CONTINUOUSLY"
        } else {
            "RENDERMODE_WHEN_DIRTY"
        }
        Log.i(TAG, "Startup GLSurfaceView renderMode=$modeName ($renderMode)")
        startupRenderModeLogged = true
    }

    private fun registerChoreographerOnMainThread() {
        val task = Runnable {
            if (!isChoreographerRegistered) {
                isChoreographerRegistered = true
                try {
                    Choreographer.getInstance().postFrameCallback(choreographerCallback)
                } catch (_: Throwable) {
                }
            }
        }
        if (Looper.myLooper() == Looper.getMainLooper()) {
            task.run()
        } else {
            mainHandler.post(task)
        }
    }

    private fun unregisterChoreographerOnMainThread() {
        isChoreographerRegistered = false
        latestChoreographerFrameTimeNs = 0L
        lastConsumedVsyncNs = 0L
        glFramesSinceLastVsync = 0
        val task = Runnable {
            try {
                Choreographer.getInstance().removeFrameCallback(choreographerCallback)
            } catch (_: Throwable) {
            }
        }
        if (Looper.myLooper() == Looper.getMainLooper()) {
            task.run()
        } else {
            mainHandler.post(task)
        }
    }

    /**
     * Switches between TRANSPARENT_OVERLAY and SINGLE_SCENE_GL composition without changing
     * CompiledEffect, scene model, package format, or rendering semantics.
     * While Path B is not wired to a live camera texture/bitmap, stays on TRANSPARENT_OVERLAY
     * so the ZOrderOnTop overlay never clears to opaque black over PreviewView.
     */
    fun setCompositionMode(mode: CompositionMode) {
        val recordingNow = isVideoRecordingActive ||
            videoRecorder.state == com.example.engine.capture.VideoRecorderPipeline.State.RECORDING
        val canUseSingleScene = singleSceneSurface.hasCameraSceneInput ||
            ((requestedPathMode == CompositionMode.SINGLE_SCENE_GL || recordingNow) && singleSceneSurface.cameraInputSurface != null)
        val effectiveMode = if (mode == CompositionMode.SINGLE_SCENE_GL && !canUseSingleScene) {
            CompositionMode.TRANSPARENT_OVERLAY
        } else {
            mode
        }
        val prevReady = compositionSurface.isSurfaceReady
        val next = when (effectiveMode) {
            CompositionMode.TRANSPARENT_OVERLAY -> transparentOverlaySurface
            CompositionMode.SINGLE_SCENE_GL -> singleSceneSurface
        }
        next.activeEffect = activeEffect
        next.onCameraConfigChanged(
            isFrontCamera = compositionSurface.isFrontCamera,
            rotationDegrees = compositionSurface.displayRotationDegrees
        )
        if (prevReady && !next.isSurfaceReady) {
            next.onSurfaceChanged(viewportWidth, viewportHeight, compositionSurface.displayRotationDegrees)
        }
        compositionSurface = next
        (cameraSession?.latencyProbe ?: latencyProbe).pathModeLabel =
            if (effectiveMode == CompositionMode.SINGLE_SCENE_GL) "B" else "A"
    }

    fun bindCameraInputSurface(inputSurface: CameraInputSurface?) {
        singleSceneSurface.bindCameraInputSurface(inputSurface)
        (cameraSession?.latencyProbe ?: latencyProbe).cameraTextureOk = singleSceneSurface.isCameraTextureOk
        updateCompositionModeForCurrentState()
    }

    fun bindCameraTexture(textureId: Int, isExternalOes: Boolean = false, transformMatrix: FloatArray? = null) {
        singleSceneSurface.bindCameraTexture(textureId, isExternalOes, transformMatrix)
        (cameraSession?.latencyProbe ?: latencyProbe).cameraTextureOk = singleSceneSurface.isCameraTextureOk
        updateCompositionModeForCurrentState()
    }

    fun bindCameraFrameBitmap(bitmap: Bitmap?, alreadyMirroredForFrontCamera: Boolean = true) {
        singleSceneSurface.bindCameraFrameBitmap(bitmap, alreadyMirroredForFrontCamera)
        (cameraSession?.latencyProbe ?: latencyProbe).cameraTextureOk = singleSceneSurface.isCameraTextureOk
        updateCompositionModeForCurrentState()
    }

    /**
     * Updates camera orientation and lens configuration and re-derives transforms through FacePoseAdapter.
     */
    fun onCameraConfigChanged(isFrontCamera: Boolean, rotationDegrees: Int) {
        transparentOverlaySurface.onCameraConfigChanged(isFrontCamera, rotationDegrees)
        singleSceneSurface.onCameraConfigChanged(isFrontCamera, rotationDegrees)
    }

    @Volatile
    var isGlThreadStopped: Boolean = false
        private set

    @Volatile
    var glReleaseQueuedBeforeSuperOnPauseCount: Int = 0
        private set

    /**
     * Short pause (`ON_PAUSE`):
     * - Finalizes any active video recording so MP4 moov trailer and AudioRecord are never leaked.
     * - Stops the main-thread `Choreographer` callback and frame processing only.
     * - Suspends `PackagePreloader` and `TextureCache` staging.
     * - Does NOT tear down GL resources or pause the `GLSurfaceView` thread, avoiding
     *   GL teardown or black frames on dialogs, multi-window focus changes, or notification shade.
     */
    override fun onPause() {
        finalizeActiveRecordingOnLifecycleInterruption()
        unregisterChoreographerOnMainThread()
        preloader?.onPause(renderGraph.textures)
        renderGraph.textures.onPause()
        transparentOverlaySurface.onPause()
        singleSceneSurface.onPause()
    }

    /**
     * Background stop (`ON_STOP`):
     * - Ensures `ON_PAUSE` state has been applied.
     * - Queues GL resource release onto the GL thread via `queueEvent` BEFORE calling
     *   `super.onPause()` (`GLSurfaceView.onPause()`), because the EGL context may be destroyed
     *   after `super.onPause()`.
     * - Guarded by `BaseCompositionSurface`'s generation/allocation flag so double-free is impossible.
     */
    fun onStop() {
        finalizeActiveRecordingOnLifecycleInterruption()
        if (!transparentOverlaySurface.isPaused) {
            onPause()
        }
        val releaseTask = Runnable {
            transparentOverlaySurface.onStop()
            singleSceneSurface.onStop()
        }
        glReleaseQueuedBeforeSuperOnPauseCount++
        queueEvent(releaseTask)
        if (renderGraph.resourceOwner.isOnGlThread()) {
            releaseTask.run()
        }
        isGlThreadStopped = true
        super.onPause()
    }

    /**
     * Foreground start (`ON_START`):
     * - Resumes `GLSurfaceView` if it was stopped, without eagerly restoring GL resources on the
     *   main thread (restoration happens once on the GL thread via `onSurfaceCreated` or lazy
     *   first draw in `renderFrame`).
     */
    fun onStart() {
        if (isGlThreadStopped) {
            isGlThreadStopped = false
            super.onResume()
        }
        transparentOverlaySurface.onStart()
        singleSceneSurface.onStart()
    }

    /**
     * Active resume (`ON_RESUME`):
     * - Wakes `GLSurfaceView` if stopped, re-registers the `Choreographer` callback on the main
     *   thread, resumes `PackagePreloader` and `TextureCache`, and resets active effect behavior state.
     * - Does NOT duplicate GL resource restoration (single entry point in `onSurfaceCreated` or
     *   lazy first draw in `renderFrame`).
     */
    override fun onResume() {
        if (isGlThreadStopped) {
            isGlThreadStopped = false
            super.onResume()
        }
        lastFrameTimeNs = 0L
        lastConsumedVsyncNs = 0L
        glFramesSinceLastVsync = 0
        preloader?.onResume(renderGraph.textures)
        renderGraph.textures.onResume()
        transparentOverlaySurface.onResume()
        singleSceneSurface.onResume()
        registerChoreographerOnMainThread()
    }

    fun stopFrameDelivery() {
        finalizeActiveRecordingOnLifecycleInterruption()
        unregisterChoreographerOnMainThread()
        preloader?.onPause(renderGraph.textures)
        renderGraph.textures.onPause()
        transparentOverlaySurface.onPause()
        singleSceneSurface.onPause()
    }

    override fun surfaceDestroyed(holder: android.view.SurfaceHolder) {
        finalizeActiveRecordingOnLifecycleInterruption()
        unregisterChoreographerOnMainThread()
        transparentOverlaySurface.onSurfaceDestroyed()
        singleSceneSurface.onSurfaceDestroyed()
        super.surfaceDestroyed(holder)
    }

    override fun onDetachedFromWindow() {
        finalizeActiveRecordingOnLifecycleInterruption()
        unregisterChoreographerOnMainThread()
        super.onDetachedFromWindow()
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        try {
            val depthScratch = IntArray(1)
            GLES30.glGetIntegerv(GLES30.GL_DEPTH_BITS, depthScratch, 0)
            if (depthScratch[0] > 0) {
                actualDepthBits = depthScratch[0]
            }
        } catch (_: Throwable) {
        }
        Log.i(TAG, "GLSurfaceView depthBits=$actualDepthBits")
        (cameraSession?.latencyProbe ?: latencyProbe).recordDepthBits(actualDepthBits)
        transparentOverlaySurface.activeEffect = activeEffect
        singleSceneSurface.activeEffect = activeEffect
        // Single idempotent entry point for GL resource creation/restoration.
        // In Android's GLSurfaceView, Renderer.onSurfaceCreated is ONLY invoked when a brand-new
        // EGLContext has been created. If resources were still marked allocated from an older
        // EGLContext (e.g. driver context loss without onStop), trigger full context-loss recovery.
        if (renderGraph.areGlResourcesAllocated || compositionSurface.areGlResourcesAllocated) {
            compositionSurface.onContextLostAndRecreated()
        } else {
            compositionSurface.onSurfaceCreated(viewportWidth, viewportHeight)
        }
        if (compositionSurface === transparentOverlaySurface) {
            singleSceneSurface.onSurfaceChanged(viewportWidth, viewportHeight, transparentOverlaySurface.displayRotationDegrees)
        } else {
            transparentOverlaySurface.onSurfaceChanged(viewportWidth, viewportHeight, singleSceneSurface.displayRotationDegrees)
        }
        lastFrameTimeNs = FrameClock.elapsedRealtimeNs()
        renderStartTimeNs = lastFrameTimeNs
        lastConsumedVsyncNs = latestChoreographerFrameTimeNs
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width.coerceAtLeast(1)
        viewportHeight = height.coerceAtLeast(1)
        val rot = display?.rotation?.let { r ->
            when (r) {
                android.view.Surface.ROTATION_90 -> 90
                android.view.Surface.ROTATION_180 -> 180
                android.view.Surface.ROTATION_270 -> 270
                else -> 0
            }
        } ?: 0
        transparentOverlaySurface.onSurfaceChanged(viewportWidth, viewportHeight, rot)
        singleSceneSurface.onSurfaceChanged(viewportWidth, viewportHeight, rot)
    }

    override fun onDrawFrame(gl: GL10?) {
        applyRenderThreadPriorityOnGlThread()
        val drawStartNano = System.nanoTime()
        val nowElapsedNs = FrameClock.elapsedRealtimeNs()

        // Compute frame-to-frame interval from Choreographer vsync timestamp (normalized by GL frames
        // drawn since the last vsync so a delayed Main-Thread Choreographer callback never double-counts
        // already-rendered 60fps GL frames and falsely demotes tracker cadence to 15 Hz)
        val vsyncNs = latestChoreographerFrameTimeNs
        val glDeltaMs = if (lastFrameTimeNs > 0L && nowElapsedNs > lastFrameTimeNs) {
            (nowElapsedNs - lastFrameTimeNs).toFloat() / 1_000_000f
        } else {
            16.6f
        }
        val frameIntervalMs = when {
            vsyncNs > 0L && lastConsumedVsyncNs > 0L && vsyncNs > lastConsumedVsyncNs -> {
                val rawVsyncDeltaMs = (vsyncNs - lastConsumedVsyncNs).toFloat() / 1_000_000f
                val spanFrames = (glFramesSinceLastVsync + 1).coerceAtLeast(1)
                val perFrameVsyncMs = rawVsyncDeltaMs / spanFrames.toFloat()
                lastConsumedVsyncNs = vsyncNs
                glFramesSinceLastVsync = 0
                if (lastFrameTimeNs > 0L) minOf(perFrameVsyncMs, glDeltaMs) else perFrameVsyncMs
            }
            else -> {
                if (vsyncNs > 0L && lastConsumedVsyncNs == 0L) {
                    lastConsumedVsyncNs = vsyncNs
                    glFramesSinceLastVsync = 0
                } else {
                    glFramesSinceLastVsync++
                }
                glDeltaMs
            }
        }

        val dt = if (lastFrameTimeNs > 0L && nowElapsedNs >= lastFrameTimeNs) {
            (nowElapsedNs - lastFrameTimeNs).toFloat() / 1_000_000_000f
        } else 0.016f
        lastFrameTimeNs = nowElapsedNs

        // Calculate FPS
        frameCount++
        val nowMs = SystemClock.uptimeMillis()
        if (nowMs - lastFpsTimeMs >= 500) {
            val fps = (frameCount * 1000f) / (nowMs - lastFpsTimeMs)
            frameCount = 0
            lastFpsTimeMs = nowMs
            onFpsCalculated?.invoke(fps, governor.currentTier.name)
        }

        // Clear depth buffer before Pass 1 + Passes 2–7
        GLES30.glClear(GLES30.GL_DEPTH_BUFFER_BIT)

        // Spread any queued background preload GPU uploads across frames (1 upload per frame)
        preloader?.stepGlUploads(renderGraph, renderGraph.textures, maxUploadsPerFrame = 1)

        val pipeline = signalPipeline
        val effect = activeEffect
        val predOn = isPredictionEnabled && (pipeline?.isPredictionEnabled != false)
        var currentFaceState: FaceState? = null
        var resultAgeMs = 0f

        if (pipeline != null && effect != null) {
            // 1. Read predicted FaceState at current render time (elapsedRealtimeNanos base) with
            // zero per-frame heap allocation while consuming any latched one-shot lifecycle event
            val faceState = pipeline.consumeLatestPredictedState(
                renderElapsedNs = nowElapsedNs,
                predictionEnabled = predOn
            )
            // Keep a reference to the latest immutable published state for still-photo fallback
            val published = pipeline.getLatestState()
            if (published.isVisible && published.fadeWeight >= 0.001f) {
                lastValidFaceState = published
            }
            // Note: faceState.isFrontCamera and faceState.projectionParams are carried per-frame from
            // AnalysisFrame -> FaceGeometrySnapshot -> FaceState so a frame captured before a camera
            // switch never uses the new camera's values.
            currentFaceState = faceState
            // Record ageAtRender ONLY while lifecycle == TRACKING (always renderElapsedNs - poseTimestampNs)
            if (faceState.lifecycle == FaceLifecycle.State.TRACKING && faceState.poseTimestampNs > 0L) {
                resultAgeMs = ((nowElapsedNs - faceState.poseTimestampNs).toFloat() / 1_000_000f).coerceAtLeast(0f)
            }

            val renderTimeSec = (nowElapsedNs - renderStartTimeNs).toFloat() / 1_000_000_000f

            // Execute Pass 1 + SignalBus + BehaviorRunner + SceneEvaluator + RenderGraph via CompositionSurface
            compositionSurface.activeEffect = effect
            compositionSurface.renderFrame(
                faceState = faceState,
                dtSec = dt,
                renderTimeSec = renderTimeSec,
                nowMs = nowMs,
                governor = governor
            )
        } else {
            val faceState = pipeline?.consumeLatestPredictedState(
                renderElapsedNs = nowElapsedNs,
                predictionEnabled = predOn
            ) ?: FaceState.EMPTY
            currentFaceState = faceState
            if (faceState.lifecycle == FaceLifecycle.State.TRACKING && faceState.poseTimestampNs > 0L) {
                resultAgeMs = ((nowElapsedNs - faceState.poseTimestampNs).toFloat() / 1_000_000f).coerceAtLeast(0f)
            }
            compositionSurface.activeEffect = null
            compositionSurface.renderFrame(
                faceState = faceState,
                dtSec = dt,
                renderTimeSec = 0f,
                nowMs = nowMs,
                governor = governor
            )
        }

        // Fulfill any pending still-photo GL overlay capture request using a dedicated offscreen FBO
        val snapshotCb = pendingSnapshotCallback.getAndSet(null)
        if (snapshotCb != null) {
            val captureState = when {
                currentFaceState != null && currentFaceState.isVisible && currentFaceState.fadeWeight >= 0.01f -> currentFaceState
                lastValidFaceState != null -> lastValidFaceState!!.copy(isVisible = true, fadeWeight = 1.0f)
                else -> currentFaceState
            }
            val bmp = if (effect != null && captureState != null && viewportWidth > 0 && viewportHeight > 0) {
                renderAndReadOffscreenBitmap(effect, captureState, viewportWidth, viewportHeight)
            } else {
                null
            }
            snapshotCb.invoke(bmp)
        }

        // Old System.nanoTime() duration around onDrawFrame kept ONLY as a debug stat
        val drawCpuDurationMs = (System.nanoTime() - drawStartNano).toFloat() / 1_000_000f

        // Record into zero-allocation LatencyProbe
        val activeProbe = cameraSession?.latencyProbe ?: latencyProbe
        activeProbe.cameraTextureOk = singleSceneSurface.isCameraTextureOk
        activeProbe.pathModeLabel =
            if (compositionSurface.mode == CompositionMode.SINGLE_SCENE_GL) "B" else "A"
        activeProbe.latestCameraSensorTimestampNs = singleSceneSurface.latestCameraSensorTimestampNs
        activeProbe.recordRenderFrame(
            renderTimestampNs = nowElapsedNs,
            faceState = currentFaceState ?: FaceState.EMPTY,
            drawCpuMs = drawCpuDurationMs,
            predictionOn = predOn
        )

        // Record frame-to-frame interval into performance governor (gated on RENDERMODE_CONTINUOUSLY)
        val isContinuous = (renderMode == RENDERMODE_CONTINUOUSLY)
        val trackerHz = cameraSession?.measuredTrackerRateHz ?: governor.targetTrackerCadenceHz.toFloat()
        val convMs = currentFaceState?.conversionMs ?: activeProbe.lastConversionMs
        val infMs = currentFaceState?.inferenceMs ?: activeProbe.lastInferenceMs
        governor.recordFrameDuration(
            durationMs = frameIntervalMs,
            nowMs = nowElapsedNs / 1_000_000L,
            resultAgeAtRenderMs = resultAgeMs,
            currentTrackerHz = trackerHz,
            drawCpuDurationMs = drawCpuDurationMs,
            isContinuousRender = isContinuous,
            conversionMs = convMs,
            inferenceMs = infMs
        )
    }

    /**
     * Captures the currently rendered OpenGL filter overlay as a transparent ARGB_8888 Bitmap.
     */
    suspend fun captureFilterBitmap(): Bitmap? {
        if (activeEffect == null || viewportWidth <= 0 || viewportHeight <= 0) return null
        return withTimeoutOrNull(2000L) {
            suspendCancellableCoroutine { cont ->
                pendingSnapshotCallback.set { bmp ->
                    if (cont.isActive) {
                        cont.resume(bmp)
                    }
                }
                requestRender()
            }
        }
    }

    /**
     * Renders the active effect into a dedicated offscreen RGBA8 + Depth16 FBO and reads it back
     * via native copyPixelsFromBuffer. This guarantees reliable readback across all mobile GPUs
     * regardless of SurfaceFlinger ZOrderOnTop window surface flags, and avoids slow byte loops.
     */
    private fun renderAndReadOffscreenBitmap(
        effect: CompiledEffect,
        faceState: com.example.engine.signal.FaceState,
        width: Int,
        height: Int
    ): Bitmap? {
        val fbos = IntArray(1)
        val texs = IntArray(1)
        val rbos = IntArray(1)
        return try {
            GLES30.glGenFramebuffers(1, fbos, 0)
            GLES30.glGenTextures(1, texs, 0)
            GLES30.glGenRenderbuffers(1, rbos, 0)

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texs[0])
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
                width, height, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

            GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, rbos[0])
            GLES30.glRenderbufferStorage(GLES30.GL_RENDERBUFFER, GLES30.GL_DEPTH_COMPONENT16, width, height)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbos[0])
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, texs[0], 0
            )
            GLES30.glFramebufferRenderbuffer(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_DEPTH_ATTACHMENT,
                GLES30.GL_RENDERBUFFER, rbos[0]
            )

            val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
            if (status == GLES30.GL_FRAMEBUFFER_COMPLETE) {
                GLES30.glViewport(0, 0, width, height)
                GLES30.glColorMask(true, true, true, true)
                GLES30.glDepthMask(true)
                GLES30.glClearColor(0f, 0f, 0f, 0f)
                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)

                if (compositionSurface.mode == CompositionMode.SINGLE_SCENE_GL && singleSceneSurface.isCameraTextureOk) {
                    singleSceneSurface.renderCameraBackgroundToCurrentFbo()
                }
                evaluator.evaluate(effect, faceState, effect.propertySlots)
                val snapshotFrameCtx = DeterministicFrameContext.forPhoto(
                    packageId = effect.packageId,
                    photoTimeSec = effect.photoPolicy.timeSec ?: renderGraph.lastFrameContext.renderTimeSec,
                    explicitSeed = effect.photoPolicy.seed
                )
                renderGraph.render(
                    effect = effect,
                    faceState = faceState,
                    evaluator = evaluator,
                    viewportWidth = width,
                    viewportHeight = height,
                    governor = governor,
                    frameContext = snapshotFrameCtx
                )
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbos[0])
                GLES30.glFinish()
            } else {
                // Fallback to default framebuffer if offscreen FBO incomplete
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
                GLES30.glFinish()
            }

            val buffer = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
            GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer)
            buffer.rewind()

            val rawBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            rawBitmap.copyPixelsFromBuffer(buffer)

            // Flip vertically from OpenGL bottom-left origin to Android top-left Bitmap origin
            val flipMatrix = android.graphics.Matrix().apply { preScale(1f, -1f) }
            val flipped = Bitmap.createBitmap(rawBitmap, 0, 0, width, height, flipMatrix, false)
            if (flipped !== rawBitmap) {
                rawBitmap.recycle()
            }
            flipped
        } catch (e: Exception) {
            null
        } finally {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, 0)
            if (fbos[0] != 0) GLES30.glDeleteFramebuffers(1, fbos, 0)
            if (texs[0] != 0) GLES30.glDeleteTextures(1, texs, 0)
            if (rbos[0] != 0) GLES30.glDeleteRenderbuffers(1, rbos, 0)
        }
    }

    fun release() {
        unregisterChoreographerOnMainThread()
        val task = Runnable {
            transparentOverlaySurface.release()
            singleSceneSurface.release()
        }
        queueEvent(task)
        if (renderGraph.resourceOwner.isOnGlThread()) {
            task.run()
        }
    }
}
