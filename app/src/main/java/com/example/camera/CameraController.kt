package com.example.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Size
import android.view.View
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionFilter
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.camera.view.transform.OutputTransform
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.engine.capture.CameraSession
import com.example.engine.capture.FrameClock
import com.example.engine.capture.SurfaceTextureCameraInputSurface
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.AnalysisResolutionMode
import com.example.engine.perception.ConversionMode
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.photo.StillPipeline
import com.example.engine.render.CompositionMode
import com.example.engine.scene.CompiledEffect
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.SignalPipeline
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * CameraController integrates CameraX, CameraSession, PerformanceGovernor, and MediaPipeFaceGeometryProvider.
 * Crucial rule: Changing active filter never restarts the camera or tracker!
 *
 * Defaults:
 * - `analysisResolutionMode = AnalysisResolutionMode.SMALL` (smallest 16:9 >= 320x180; `LEGACY` still selectable)
 * - `useRgbaOutputFormat = false` (`ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888`)
 * - `useGpuDelegate = false` (MediaPipe `Delegate.CPU`)
 *
 * Aspect Ratio / Crop & Single-Mirror Policy (README Sections 7 & 10):
 * - Both `Preview` and `ImageAnalysis` are bound together in a shared `UseCaseGroup` with
 *   `previewView.viewPort` (`FILL_CENTER`) and a 16:9 target aspect ratio (`AspectRatio.RATIO_16_9`
 *   in `LEGACY`, or smallest 16:9 >= 320x180 in `SMALL`).
 * - `PreviewView.outputTransform` is cached on the MAIN thread (updated on bind, layout changes,
 *   display rotation changes, and lens switches) and composed with the frame's `ImageProxy`
 *   transform via `CoordinateTransform(analysisOutputTransform, previewOutputTransform)` on the
 *   analysis thread.
 * - SINGLE MIRROR POLICY:
 *   - 2D Landmark Path: CameraX `PreviewView.outputTransform` (or the analytical fallback
 *     `FacePoseAdapter.computeManualFillCenterImageToViewTransform`) already mirrors the front
 *     camera (`isFrontCamera == true`). Therefore `FacePoseAdapter.mapLandmarkToView` applies
 *     `imageToViewTransform` directly with NO extra manual flip.
 *   - 3D Pose / Mesh Path: `MediaPipeFaceGeometryProvider.calibratePoseMatrix` and `SignalPipeline`
 *     remain in unmirrored camera space. The single mirror for the 3D path is applied in
 *     `FacePoseAdapter.createCameraProjection(..., frontFacing = isFrontCamera)` via negated
 *     `proj[0] = -f / aspect`.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    companion object {
        private const val TAG = "CameraController"
        const val SMALL_16_9_MIN_LONG_EDGE_PX = 320
        const val SMALL_16_9_MIN_SHORT_EDGE_PX = 180

        /**
         * Selects the smallest 16:9 resolution whose dimensions are >= ~320x180 (preserving the
         * exact same 16:9 aspect ratio as `Preview` and `LEGACY` `ImageAnalysis`).
         */
        fun selectSmall16By9Resolution(
            supportedSizes: List<Size>,
            minLongEdge: Int = SMALL_16_9_MIN_LONG_EDGE_PX,
            minShortEdge: Int = SMALL_16_9_MIN_SHORT_EDGE_PX
        ): List<Size> {
            if (supportedSizes.isEmpty()) return emptyList()

            val targetRatio = 16f / 9f
            val is16By9 = { s: Size ->
                val longE = max(s.width, s.height).toFloat()
                val shortE = min(s.width, s.height).coerceAtLeast(1).toFloat()
                abs((longE / shortE) - targetRatio) <= 0.08f
            }

            val qualifying16By9 = supportedSizes
                .filter { s ->
                    is16By9(s) &&
                        max(s.width, s.height) >= minLongEdge &&
                        min(s.width, s.height) >= minShortEdge
                }
                .sortedBy { it.width * it.height }

            val smaller16By9Fallback = supportedSizes
                .filter { s -> is16By9(s) && s !in qualifying16By9 }
                .sortedByDescending { it.width * it.height }

            val otherQualifying = supportedSizes
                .filter { s ->
                    !is16By9(s) &&
                        max(s.width, s.height) >= minLongEdge &&
                        min(s.width, s.height) >= minShortEdge
                }
                .sortedBy { it.width * it.height }

            val remaining = supportedSizes
                .filter { s ->
                    s !in qualifying16By9 && s !in smaller16By9Fallback && s !in otherQualifying
                }
                .sortedBy { it.width * it.height }

            return qualifying16By9 + smaller16By9Fallback + otherQualifying + remaining
        }

        fun buildSmall16By9ResolutionSelector(): ResolutionSelector {
            return ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(SMALL_16_9_MIN_LONG_EDGE_PX, SMALL_16_9_MIN_SHORT_EDGE_PX),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .setResolutionFilter(
                    ResolutionFilter { supportedSizes, _ ->
                        selectSmall16By9Resolution(supportedSizes)
                    }
                )
                .build()
        }
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var imageCapture: ImageCapture? = null
    private var preview: Preview? = null
    private var camera: Camera? = null
    private var boundPreviewView: PreviewView? = null

    private val analysisExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "CameraAnalysisExecutor").apply { isDaemon = true }
    }
    private val photoWorkerExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "PhotoWorkerExecutor").apply { isDaemon = true }
    }
    private val photoWorkerDispatcher = photoWorkerExecutor.asCoroutineDispatcher()

    val governor = PerformanceGovernor(context)
    val signalPipeline = SignalPipeline()
    val faceProvider = MediaPipeFaceGeometryProvider(
        context = context
    )
    val cameraSession = CameraSession(faceProvider, signalPipeline, governor)
    val stillPipeline = StillPipeline(faceProvider = faceProvider)
    val cameraInputSurface = SurfaceTextureCameraInputSurface(
        clockOffsetCalibrator = cameraSession.clockOffsetCalibrator
    )

    // Debug flags (default resolution mode is SMALL; LEGACY remains selectable; default path is PATH:A)
    @Volatile
    var pathMode: CompositionMode = CompositionMode.TRANSPARENT_OVERLAY
        private set

    @Volatile
    var analysisResolutionMode: AnalysisResolutionMode = AnalysisResolutionMode.SMALL
        private set

    @Volatile
    var useRgbaOutputFormat: Boolean = false
        private set

    @Volatile
    var useGpuDelegate: Boolean = false
        private set

    @Volatile
    var isDebugBlendshapesEnabled: Boolean = true
        private set

    @Volatile
    var useLowAnalysisThreadPriority: Boolean = false
        private set

    @Volatile
    var useHighRenderThreadPriority: Boolean = false
        private set

    @Volatile
    var conversionMode: ConversionMode = ConversionMode.FAST
        private set

    init {
        cameraInputSurface.subscribe {
            cameraSession.latencyProbe.cameraTextureOk = cameraInputSurface.hasValidFrame
            cameraSession.latencyProbe.latestCameraSensorTimestampNs = cameraInputSurface.lastSensorTimestampNs
        }
    }

    // Cached PreviewView outputTransform and dimensions updated ONLY on the MAIN thread
    @OptIn(TransformExperimental::class)
    @Volatile
    private var cachedPreviewOutputTransform: OutputTransform? = null

    @Volatile
    private var cachedPreviewWidth: Int =
        context.resources.displayMetrics.widthPixels.takeIf { it > 0 } ?: 1080

    @Volatile
    private var cachedPreviewHeight: Int =
        context.resources.displayMetrics.heightPixels.takeIf { it > 0 } ?: 1920

    private val previewLayoutListener = View.OnLayoutChangeListener { view, _, _, _, _, _, _, _, _ ->
        val pv = view as? PreviewView ?: return@OnLayoutChangeListener
        refreshCachedPreviewOutputTransformOnMainThread(pv)
    }

    // Reusable transform helpers confined to the single-threaded analysisExecutor
    @OptIn(TransformExperimental::class)
    private val imageProxyTransformFactory = ImageProxyTransformFactory().apply {
        isUsingCropRect = true
        isUsingRotationDegrees = true
    }
    private val reusableCoordinateMatrix = Matrix()
    private val reusableNormalizedMatrix = Matrix()
    private val reusableImageToViewFloats = FloatArray(9)

    private val _isFrontCamera = MutableStateFlow(true)
    val isFrontCamera: StateFlow<Boolean> = _isFrontCamera.asStateFlow()

    private val _isCameraReady = MutableStateFlow(false)
    val isCameraReady: StateFlow<Boolean> = _isCameraReady.asStateFlow()

    @OptIn(TransformExperimental::class)
    private fun refreshCachedPreviewOutputTransformOnMainThread(previewView: PreviewView?) {
        if (previewView == null) return
        val w = previewView.width
        val h = previewView.height
        if (w > 0 && h > 0) {
            cachedPreviewWidth = w
            cachedPreviewHeight = h
            if (cameraInputSurface.options.viewportWidth != w || cameraInputSurface.options.viewportHeight != h) {
                cameraInputSurface.updateOptions(
                    cameraInputSurface.options.copy(
                        viewportWidth = w,
                        viewportHeight = h
                    )
                )
            }
        }
        cachedPreviewOutputTransform = try {
            previewView.outputTransform
        } catch (_: Throwable) {
            null
        }
    }

    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val longStopCloseRunnable = Runnable {
        cameraSession.checkLongStopTimeout()
    }

    fun start(previewView: PreviewView) {
        governor.thermalMonitor.start()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCamera(previewView)
                if (!cameraSession.isProviderStarted) {
                    cameraSession.startProvider(
                        ProviderConfig(
                            outputPose = true,
                            outputBlendshapes = isDebugBlendshapesEnabled,
                            maxFaces = 1,
                            useGpu = useGpuDelegate,
                            analysisLongEdgePx = governor.targetAnalysisSizePx
                        )
                    )
                }
                _isCameraReady.value = true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start CameraProvider: ${e.message}", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun applyEffectRequirements(effect: CompiledEffect?) {
        cameraSession.applyEffectRequirements(effect)
    }

    fun onPause() {
        governor.thermalMonitor.stop()
        cameraInputSurface.setPaused(true)
        cameraSession.onPause()
    }

    @JvmOverloads
    fun onStop(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L) {
        mainHandler.removeCallbacks(longStopCloseRunnable)
        governor.thermalMonitor.stop()
        cameraInputSurface.setPaused(true)
        cameraSession.onStop(nowElapsedMs)
        mainHandler.postDelayed(
            longStopCloseRunnable,
            CameraSession.LONG_STOP_PROVIDER_IDLE_TIMEOUT_MS
        )
    }

    @JvmOverloads
    fun onStart(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L) {
        mainHandler.removeCallbacks(longStopCloseRunnable)
        cameraSession.onStart(nowElapsedMs)
    }

    @JvmOverloads
    fun onResume(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L) {
        mainHandler.removeCallbacks(longStopCloseRunnable)
        governor.thermalMonitor.start()
        cameraInputSurface.setPaused(false)
        cameraSession.onResume(nowElapsedMs)
        refreshCachedPreviewOutputTransformOnMainThread(boundPreviewView)
    }

    fun onDisplayRotationChanged(rotationDegrees: Int) {
        cameraSession.onDisplayRotationChanged(rotationDegrees)
        refreshCachedPreviewOutputTransformOnMainThread(boundPreviewView)
    }

    @OptIn(TransformExperimental::class)
    fun toggleCamera(previewView: PreviewView) {
        val nextFront = !_isFrontCamera.value
        _isFrontCamera.value = nextFront
        // Clear cached PreviewView transform immediately on lens switch so stale lens transform is not reused
        cachedPreviewOutputTransform = null
        cameraInputSurface.updateOptions(
            cameraInputSurface.options.copy(
                mirrorH = nextFront,
                viewportWidth = cachedPreviewWidth,
                viewportHeight = cachedPreviewHeight
            )
        )
        cameraSession.onCameraLensChanged(nextFront)
        bindCamera(previewView)
    }

    /**
     * Switches `analysisResolutionMode` between `LEGACY` and `SMALL` (smallest 16:9 >= 320x180),
     * rebinds CameraX, and resets `ClockOffsetCalibrator` and `LatencyProbe` rolling windows.
     */
    fun setAnalysisResolutionMode(mode: AnalysisResolutionMode, previewView: PreviewView?) {
        if (this.analysisResolutionMode == mode) return
        this.analysisResolutionMode = mode
        faceProvider.setResolutionModeForTelemetry(mode)
        val fmtLabel = if (useRgbaOutputFormat) "RGBA_8888" else "YUV_420_888"
        cameraSession.onAnalysisRebound(
            formatLabel = fmtLabel,
            resolutionModeLabel = mode.name
        )
        if (previewView != null) {
            bindCamera(previewView)
        }
    }

    /**
     * Switches `ImageAnalysis` output format between `YUV_420_888` (`false`) and `RGBA_8888` (`true`),
     * rebinds CameraX, and resets `ClockOffsetCalibrator` and `LatencyProbe` rolling windows.
     */
    fun setUseRgbaOutputFormat(useRgba: Boolean, previewView: PreviewView?) {
        if (this.useRgbaOutputFormat == useRgba) return
        this.useRgbaOutputFormat = useRgba
        val fmtLabel = if (useRgba) "RGBA_8888" else "YUV_420_888"
        cameraSession.onAnalysisRebound(
            formatLabel = fmtLabel,
            resolutionModeLabel = analysisResolutionMode.name
        )
        if (previewView != null) {
            bindCamera(previewView)
        }
    }

    /**
     * Switches MediaPipe delegate between `CPU` (`false`) and `GPU` (`true`), retiring the old
     * landmarker only after any in-flight inference finishes, and resets `LatencyProbe`.
     */
    fun setUseGpuDelegate(useGpu: Boolean) {
        this.useGpuDelegate = useGpu
        cameraSession.setUseGpuDelegate(useGpu)
    }

    /**
     * Enables or disables face blendshape output (`BLEND:ON/OFF`) at runtime on the analysis thread
     * with a 3-second warm-up window.
     */
    fun setDebugBlendshapesEnabled(enabled: Boolean) {
        this.isDebugBlendshapesEnabled = enabled
        cameraSession.setDebugBlendshapesEnabled(enabled)
    }

    /**
     * Toggles the thread priority of the analysis/conversion thread (`PRIO:NORM/LOW`).
     */
    fun setUseLowAnalysisThreadPriority(lowPriority: Boolean) {
        this.useLowAnalysisThreadPriority = lowPriority
        cameraSession.setUseLowAnalysisThreadPriority(lowPriority)
    }

    /**
     * Toggles the GL render thread priority (`RPRIO:NORM/HIGH`) on [glView].
     */
    fun setUseHighRenderThreadPriority(
        highPriority: Boolean,
        glView: com.example.engine.render.XelvicGlSurfaceView?
    ) {
        this.useHighRenderThreadPriority = highPriority
        cameraSession.latencyProbe.renderThreadPriorityLabel = if (highPriority) "HIGH" else "NORM"
        glView?.setUseHighRenderThreadPriority(highPriority)
    }

    /**
     * Toggles the conversion path (`CONV:FAST` single-pass vs `CONV:OLD` two-stage),
     * resetting `LatencyProbe` rolling windows and starting a 3-second warm-up.
     */
    fun setConversionMode(mode: ConversionMode) {
        this.conversionMode = mode
        cameraSession.setConversionMode(mode)
    }

    /**
     * Switches between `PATH:A` (`TRANSPARENT_OVERLAY`, default) and `PATH:B` (`SINGLE_SCENE_GL`)
     * without restarting `MediaPipeFaceGeometryProvider`.
     */
    fun setPathMode(
        mode: CompositionMode,
        previewView: PreviewView?,
        glView: com.example.engine.render.XelvicGlSurfaceView?
    ) {
        this.pathMode = mode
        cameraSession.latencyProbe.pathModeLabel =
            if (mode == CompositionMode.SINGLE_SCENE_GL) "B" else "A"
        cameraInputSurface.updateOptions(
            cameraInputSurface.options.copy(
                cropCenter = true,
                mirrorH = _isFrontCamera.value,
                viewportWidth = cachedPreviewWidth,
                viewportHeight = cachedPreviewHeight
            )
        )
        glView?.bindCameraInputSurface(cameraInputSurface)
        glView?.setRequestedPathMode(mode)
        val activePreview = preview
        if (activePreview != null && previewView != null) {
            previewView.visibility = View.VISIBLE
            if (mode == CompositionMode.SINGLE_SCENE_GL) {
                activePreview.surfaceProvider =
                    cameraInputSurface.createSurfaceProvider(ContextCompat.getMainExecutor(context))
            } else {
                activePreview.surfaceProvider = previewView.surfaceProvider
            }
        }
        cameraSession.latencyProbe.cameraTextureOk =
            (glView?.isCameraTextureOk == true) || cameraInputSurface.hasValidFrame
    }

    @Volatile
    private var pathModeBeforeRecording: CompositionMode? = null

    /**
     * Starts real-time GPU + AAC audio video recording via [glView]'s `VideoRecorderPipeline`.
     * Automatically engages `SINGLE_SCENE_GL` (`Path B`) if the session was in `TRANSPARENT_OVERLAY`
     * so `SingleSceneCompositionSurface.sceneFbo` receives the live camera background + 3D AR + Beauty scene.
     */
    fun startVideoRecording(
        outputFile: java.io.File,
        enableAudio: Boolean,
        previewView: PreviewView?,
        glView: com.example.engine.render.XelvicGlSurfaceView?,
        width: Int = cachedPreviewWidth.coerceAtLeast(720),
        height: Int = cachedPreviewHeight.coerceAtLeast(1280)
    ): Boolean {
        val view = glView ?: return false
        if (pathMode != CompositionMode.SINGLE_SCENE_GL) {
            pathModeBeforeRecording = pathMode
            setPathMode(CompositionMode.SINGLE_SCENE_GL, previewView, view)
        } else {
            pathModeBeforeRecording = null
        }
        val config = com.example.engine.capture.VideoRecorderPipeline.RecordingConfig(
            outputFile = outputFile,
            width = width,
            height = height,
            enableAudio = enableAudio,
            useHardwareCodec = true
        )
        val started = view.startVideoRecording(config)
        if (!started && pathModeBeforeRecording != null) {
            val restore = pathModeBeforeRecording!!
            pathModeBeforeRecording = null
            setPathMode(restore, previewView, view)
        }
        return started
    }

    /**
     * Stops active GPU + AAC video recording and restores the user's previous `pathMode` if it was
     * temporarily switched to `SINGLE_SCENE_GL` for recording.
     */
    fun stopVideoRecording(
        previewView: PreviewView?,
        glView: com.example.engine.render.XelvicGlSurfaceView?
    ): com.example.engine.capture.VideoRecorderPipeline.RecordingResult? {
        val view = glView ?: return null
        val res = view.stopVideoRecording()
        val restore = pathModeBeforeRecording
        pathModeBeforeRecording = null
        if (restore != null && restore != pathMode) {
            setPathMode(restore, previewView, view)
        }
        return res
    }

    @Suppress("DEPRECATION")
    @OptIn(ExperimentalCamera2Interop::class, TransformExperimental::class)
    private fun bindCamera(previewView: PreviewView) {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        if (boundPreviewView !== previewView) {
            boundPreviewView?.removeOnLayoutChangeListener(previewLayoutListener)
            boundPreviewView = previewView
            previewView.addOnLayoutChangeListener(previewLayoutListener)
        }
        refreshCachedPreviewOutputTransformOnMainThread(previewView)

        val isFront = _isFrontCamera.value
        val lensFacing = if (isFront) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        // Capture ViewPort on the Main thread per README Section 10; never read PreviewView from analysisExecutor.
        // Only attach previewView.viewPort when PreviewView has valid measured dimensions; before layout finishes,
        // leaving ViewPort null keeps ImageAnalysis at full 16:9 so computeManualFillCenterImageToViewTransform
        // and CameraInputSurface.computeSingleOwnerInputMatrix both apply identical FILL_CENTER cropping to viewAspect.
        val mainThreadViewPort = if (previewView.width > 0 && previewView.height > 0) {
            previewView.viewPort
        } else {
            null
        }

        cameraInputSurface.updateOptions(
            cameraInputSurface.options.copy(
                cropCenter = true,
                mirrorH = isFront,
                viewportWidth = cachedPreviewWidth,
                viewportHeight = cachedPreviewHeight
            )
        )

        val builtPreview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build().also {
                previewView.visibility = View.VISIBLE
                if (pathMode == CompositionMode.SINGLE_SCENE_GL) {
                    it.surfaceProvider =
                        cameraInputSurface.createSurfaceProvider(ContextCompat.getMainExecutor(context))
                } else {
                    it.surfaceProvider = previewView.surfaceProvider
                }
            }
        preview = builtPreview

        val builtCapture = ImageCapture.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        imageCapture = builtCapture

        val outputFormat = if (useRgbaOutputFormat) {
            ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
        } else {
            ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888
        }

        val analysisBuilder = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(outputFormat)

        if (analysisResolutionMode == AnalysisResolutionMode.SMALL) {
            // Keep the exact same 16:9 aspect ratio as LEGACY & Preview, selecting smallest 16:9 >= 320x180
            analysisBuilder.setResolutionSelector(buildSmall16By9ResolutionSelector())
        } else {
            // LEGACY default: today's exact ImageAnalysis configuration
            analysisBuilder.setTargetAspectRatio(AspectRatio.RATIO_16_9)
        }

        // Attach Camera2Interop capture callback to read LENS_FOCAL_LENGTH and SCALER_CROP_REGION per frame
        try {
            Camera2Interop.Extender(analysisBuilder).setSessionCaptureCallback(
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        session: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        val focalLen = result.get(CaptureResult.LENS_FOCAL_LENGTH)
                        val cropRegion = result.get(CaptureResult.SCALER_CROP_REGION)
                        cameraSession.recordCaptureResultMetadata(focalLen, cropRegion)
                    }
                }
            )
        } catch (e: Exception) {
            Log.w(TAG, "Camera2Interop capture callback unavailable: ${e.message}")
        }

        val builtAnalysis = analysisBuilder.build().also { analysis ->
            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                var submittedToSession = false
                try {
                    val arrivalNs = FrameClock.elapsedRealtimeNs()
                    val rot = imageProxy.imageInfo.rotationDegrees
                    val frameIsFront = _isFrontCamera.value

                    val cropRect = try {
                        imageProxy.cropRect
                    } catch (_: Throwable) {
                        null
                    }
                    val cropW = cropRect?.width()?.takeIf { it > 0 } ?: imageProxy.width.coerceAtLeast(1)
                    val cropH = cropRect?.height()?.takeIf { it > 0 } ?: imageProxy.height.coerceAtLeast(1)
                    val rotNorm = FacePoseAdapter.normalizeRotationDegrees(rot)
                    val uprightCropW = if (rotNorm == 90 || rotNorm == 270) cropH else cropW
                    val uprightCropH = if (rotNorm == 90 || rotNorm == 270) cropW else cropH
                    val cropAspect = uprightCropW.toFloat() / uprightCropH.toFloat().coerceAtLeast(1f)

                    val viewW = cachedPreviewWidth.coerceAtLeast(1)
                    val viewH = cachedPreviewHeight.coerceAtLeast(1)
                    val viewAspect = viewW.toFloat() / viewH.toFloat()

                    val previewOutTransform = cachedPreviewOutputTransform
                    var composedWithCameraX = false
                    if (pathMode == CompositionMode.TRANSPARENT_OVERLAY && previewOutTransform != null && viewW > 0 && viewH > 0) {
                        try {
                            val analysisOutTransform = imageProxyTransformFactory.getOutputTransform(imageProxy)
                            val coordTransform = CoordinateTransform(analysisOutTransform, previewOutTransform)
                            coordTransform.transform(reusableCoordinateMatrix)
                            FacePoseAdapter.composeCameraXNormalizedImageToViewTransform(
                                pixelTransformMatrix = reusableCoordinateMatrix,
                                orientedCropWidth = uprightCropW,
                                orientedCropHeight = uprightCropH,
                                viewWidth = viewW,
                                viewHeight = viewH,
                                outMatrix9 = reusableImageToViewFloats,
                                scratchMatrix = reusableNormalizedMatrix
                            )
                            composedWithCameraX = true
                        } catch (_: Throwable) {
                            composedWithCameraX = false
                        }
                    }

                    if (!composedWithCameraX) {
                        FacePoseAdapter.computeManualFillCenterImageToViewTransform(
                            cropAspectRatio = cropAspect,
                            viewAspectRatio = viewAspect,
                            isFrontCamera = frameIsFront,
                            outMatrix9 = reusableImageToViewFloats
                        )
                    }

                    submittedToSession = true
                    cameraSession.submitAnalysisFrame(
                        imageProxy = imageProxy,
                        timestampNs = arrivalNs,
                        rotationDegrees = rot,
                        isFrontCamera = frameIsFront,
                        imageToViewTransform = reusableImageToViewFloats,
                        arrivalElapsedRealtimeNs = arrivalNs
                    )
                } finally {
                    if (!submittedToSession) {
                        try {
                            imageProxy.close()
                        } catch (_: Throwable) {
                        }
                    }
                }
            }
        }
        imageAnalysis = builtAnalysis

        try {
            if (mainThreadViewPort != null) {
                val useCaseGroup = UseCaseGroup.Builder()
                    .setViewPort(mainThreadViewPort)
                    .addUseCase(builtPreview)
                    .addUseCase(builtCapture)
                    .addUseCase(builtAnalysis)
                    .build()
                camera = provider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    useCaseGroup
                )
            } else {
                camera = provider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    builtPreview,
                    builtCapture,
                    builtAnalysis
                )
            }
            // Refresh PreviewView.outputTransform after binding
            previewView.post {
                refreshCachedPreviewOutputTransformOnMainThread(previewView)
            }
            cameraSession.recordCameraBound(camera?.cameraInfo)
            faceProvider.setResolutionModeForTelemetry(analysisResolutionMode)
            cameraSession.latencyProbe.analysisFormatLabel =
                if (useRgbaOutputFormat) "RGBA_8888" else "YUV_420_888"
            cameraSession.latencyProbe.resolutionModeLabel = analysisResolutionMode.name
            cameraSession.latencyProbe.conversionModeLabel = conversionMode.name
            cameraSession.latencyProbe.delegateLabel = faceProvider.activeDelegateLabel
            Log.i(
                TAG,
                "Camera bound (front=$isFront, resMode=$analysisResolutionMode, format=${cameraSession.latencyProbe.analysisFormatLabel}, delegate=${cameraSession.latencyProbe.delegateLabel})"
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error binding camera lifecycle: ${e.message}", e)
        }
    }

    /**
     * Captures a photograph via CameraX `ImageCapture` and processes it on the dedicated photo
     * background worker via `StillPipeline.processStillPhoto`.
     */
    suspend fun capturePhoto(
        activeEffect: CompiledEffect?,
        previewBitmap: Bitmap? = null,
        glOverlayBitmap: Bitmap? = null,
        targetAspectRatio: Float? = 9f / 16f
    ): StillPipeline.PhotoResult? = withContext(photoWorkerDispatcher) {
        val capture = imageCapture
        if (capture != null) {
            val capturedResult = kotlinx.coroutines.suspendCancellableCoroutine<StillPipeline.PhotoResult?> { continuation ->
                capture.takePicture(
                    photoWorkerExecutor,
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            try {
                                val rawBitmap = imageProxyToRawBitmap(image)
                                val rotation = image.imageInfo.rotationDegrees
                                val photoResult = stillPipeline.processStillPhoto(
                                    rawCaptureBitmap = rawBitmap,
                                    effect = activeEffect,
                                    isFrontCamera = _isFrontCamera.value,
                                    rotationDegrees = rotation,
                                    targetAspectRatio = targetAspectRatio,
                                    sourceAlreadyMirrored = false,
                                    previewFaceStateToIgnore = signalPipeline.getLatestState()
                                )
                                if (rawBitmap !== photoResult.bitmap && !rawBitmap.isRecycled) {
                                    rawBitmap.recycle()
                                }
                                continuation.resume(photoResult) {}
                            } catch (e: Exception) {
                                Log.e(TAG, "ImageCapture photo processing error: ${e.message}", e)
                                continuation.resume(null) {}
                            } finally {
                                image.close()
                            }
                        }

                        override fun onError(exception: ImageCaptureException) {
                            Log.w(TAG, "ImageCapture.takePicture failed (${exception.message}), falling back to still inference on preview frame")
                            continuation.resume(null) {}
                        }
                    }
                )
            }
            if (capturedResult != null) {
                return@withContext capturedResult
            }
        }

        if (previewBitmap != null) {
            return@withContext try {
                stillPipeline.processStillPhoto(
                    rawCaptureBitmap = previewBitmap,
                    effect = activeEffect,
                    isFrontCamera = _isFrontCamera.value,
                    rotationDegrees = 0,
                    targetAspectRatio = targetAspectRatio,
                    sourceAlreadyMirrored = _isFrontCamera.value,
                    previewFaceStateToIgnore = signalPipeline.getLatestState()
                )
            } catch (e: Exception) {
                Log.e(TAG, "Still fallback photo processing error: ${e.message}", e)
                null
            }
        }

        null
    }

    private fun imageProxyToRawBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: Bitmap.createBitmap(image.width.coerceAtLeast(1), image.height.coerceAtLeast(1), Bitmap.Config.ARGB_8888)
    }

    fun stopFrameDelivery() {
        mainHandler.removeCallbacks(longStopCloseRunnable)
        governor.thermalMonitor.stop()
        cameraSession.onPause()
        try {
            imageAnalysis?.clearAnalyzer()
        } catch (_: Throwable) {
        }
    }

    fun unbindCameraX() {
        boundPreviewView?.removeOnLayoutChangeListener(previewLayoutListener)
        boundPreviewView = null
        try {
            cameraProvider?.unbindAll()
        } catch (_: Throwable) {
        }
    }

    fun closeTracker() {
        cameraSession.stop()
        stillPipeline.release()
    }

    fun shutdownExecutors() {
        analysisExecutor.shutdown()
        photoWorkerExecutor.shutdown()
    }

    val areExecutorsShutdown: Boolean
        get() = analysisExecutor.isShutdown && photoWorkerExecutor.isShutdown

    fun stop() {
        stopFrameDelivery()
        unbindCameraX()
        cameraInputSurface.release()
        closeTracker()
        shutdownExecutors()
    }
}
