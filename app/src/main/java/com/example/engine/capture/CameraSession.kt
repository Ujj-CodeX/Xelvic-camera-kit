package com.example.engine.capture

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.ImageProxy
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.AnalysisResolutionMode
import com.example.engine.perception.ConversionMode
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.scene.CompiledEffect
import com.example.engine.signal.CameraIntrinsics
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.ProjectionParams
import com.example.engine.signal.SignalPipeline
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * CameraSession (README Section 4, 7, 8, 10, 13, 14 & 15; AUDIT.md Section 3.2):
 *
 * Coordinates analysis frame cadence, sensor clock-base calibration, camera intrinsics / FOV
 * derivation, dynamic tracker output flags, and single-close ImageProxy ownership between
 * CameraX, PerformanceGovernor, FaceGeometryProvider, SignalPipeline, and LatencyProbe.
 *
 * Rules enforced:
 * - Track once, render many: changing a filter updates ONLY requested tracker output flags;
 *   never restarts the camera or tracker.
 * - Honors governor's `targetTrackerCadenceHz` (15 Hz floor .. 30 Hz ceiling) and `targetAnalysisSizePx`.
 * - Closes every skipped or processed `ImageProxy` exactly once.
 * - Aligns `ImageProxy.imageInfo.timestamp` to `SystemClock.elapsedRealtimeNanos()` via `ClockOffsetCalibrator`.
 * - Carries `isFrontCamera` and `ProjectionParams` per `AnalysisFrame` -> `FaceGeometrySnapshot` -> `FaceState`
 *   so frames captured before a camera switch never use the new camera's values.
 */
class CameraSession @JvmOverloads constructor(
    val provider: FaceGeometryProvider,
    val signalPipeline: SignalPipeline,
    val governor: PerformanceGovernor,
    val clockOffsetCalibrator: ClockOffsetCalibrator = ClockOffsetCalibrator(),
    val latencyProbe: LatencyProbe = LatencyProbe.shared
) {
    companion object {
        /**
         * If the app remains in `ON_STOP` longer than 45 seconds, the `FaceGeometryProvider`
         * is closed to release native MediaPipe resources and lazily restarted on resume.
         * In that long-stop scenario, [trackerStartCount] legitimately increments to 2.
         */
        const val LONG_STOP_PROVIDER_IDLE_TIMEOUT_MS = 45_000L
    }

    private val lastSubmittedTimestampNs = AtomicLong(0L)
    private val lastRawSensorTimestampNs = AtomicLong(0L)
    private val trackerFrameCount = AtomicInteger(0)
    private val lastRateWindowNs = AtomicLong(FrameClock.elapsedRealtimeNs())
    private val sessionCadenceThrottledCounter = AtomicInteger(0)

    val sessionCadenceThrottledCount: Int
        get() = sessionCadenceThrottledCounter.get()

    @Volatile
    var isDebugBlendshapesEnabled: Boolean = true
        private set

    @Volatile
    var useLowAnalysisThreadPriority: Boolean = false
        private set

    @Volatile
    var skipConversionIfPendingFresh: Boolean = false
        private set

    @Volatile
    var conversionMode: ConversionMode = ConversionMode.FAST
        private set

    @Volatile
    var lastHeldBlendshapes: Map<String, Float> = emptyMap()
        private set

    @Volatile
    var measuredTrackerRateHz: Float = 30.0f
        private set

    @Volatile
    var cameraStartCount: Int = 0
        private set

    /**
     * Number of times `FaceGeometryProvider.start` has been invoked.
     * - Remains `1` across filter switches, lens switches, and short pauses (`< 45 s`).
     * - May become `2` (or increment) if the app stays in `ON_STOP` for `>= 45 s`, which closes
     *   the provider and lazily restarts it upon resume.
     */
    @Volatile
    var trackerStartCount: Int = 0
        private set

    @Volatile
    var isProviderStarted: Boolean = false
        private set

    @Volatile
    var hasStartedProviderOnce: Boolean = false
        private set

    @Volatile
    var wasProviderClosedForLongStop: Boolean = false
        private set

    @Volatile
    var lastProviderConfig: ProviderConfig = ProviderConfig()
        private set

    @Volatile
    var activeEffect: CompiledEffect? = null
        private set

    @Volatile
    var droppedWhilePausedCount: Int = 0
        private set

    /**
     * Active camera intrinsics queried from `CameraCharacteristics` on bind / lens change.
     */
    @Volatile
    var activeIntrinsics: CameraIntrinsics = CameraIntrinsics.UNKNOWN
        private set

    /**
     * Latest per-frame `CaptureResult.LENS_FOCAL_LENGTH` (mm) from Camera2Interop capture callback.
     */
    @Volatile
    var lastCaptureFocalLengthMm: Float = 0f
        private set

    /**
     * Latest per-frame `CaptureResult.SCALER_CROP_REGION` from Camera2Interop capture callback.
     */
    @Volatile
    var lastScalerCropRegion: Rect? = null
        private set

    /**
     * Latest derived [ProjectionParams] for the most recently submitted analysis frame.
     */
    @Volatile
    var lastProjectionParams: ProjectionParams = ProjectionParams.DEFAULT
        private set

    init {
        if (provider is com.example.engine.perception.MediaPipeFaceGeometryProvider) {
            provider.onStatusChanged = { delLabel, statusMsg ->
                latencyProbe.delegateLabel = delLabel
                latencyProbe.trackerStatusLabel = statusMsg
                latencyProbe.recordSwapDurationMs(provider.lastSwapMs)
            }
        }
        governor.onTransitionLogged = { reason ->
            latencyProbe.recordGovernorTransition(reason)
        }
        provider.setListener { snapshot ->
            if (isPaused) {
                // Drop any in-flight inference result that arrives while paused
                droppedWhilePausedCount++
                return@setListener
            }
            val count = trackerFrameCount.incrementAndGet()
            val nowNs = FrameClock.elapsedRealtimeNs()
            val elapsedNs = nowNs - lastRateWindowNs.get()
            if (elapsedNs >= 500_000_000L) {
                measuredTrackerRateHz = (count * 1_000_000_000f) / elapsedNs.toFloat()
                trackerFrameCount.set(0)
                lastRateWindowNs.set(nowNs)
            }
            latencyProbe.delegateLabel = provider.activeDelegateLabel
            latencyProbe.trackerStatusLabel = provider.statusMessage
            latencyProbe.recordSwapDurationMs(provider.lastSwapMs)
            latencyProbe.recordTrackerSnapshot(snapshot)
            governor.updateTrackerTelemetry(
                rateHz = measuredTrackerRateHz,
                ageAtRenderMs = latencyProbe.lastAgeAtRenderMs,
                conversionMs = snapshot.conversionMs,
                inferenceMs = snapshot.inferenceMs,
                nowMs = nowNs / 1_000_000L
            )
            if (snapshot.blendshapes.isNotEmpty()) {
                lastHeldBlendshapes = HashMap(snapshot.blendshapes)
            }
            val snapshotForPipeline = if (!isDebugBlendshapesEnabled && snapshot.blendshapes.isEmpty()) {
                val held = if (lastHeldBlendshapes.isNotEmpty()) {
                    lastHeldBlendshapes
                } else {
                    signalPipeline.getLatestState().blendshapes
                }
                if (held.isNotEmpty()) {
                    snapshot.copy(blendshapes = held)
                } else {
                    snapshot
                }
            } else {
                snapshot
            }
            signalPipeline.processSnapshot(snapshotForPipeline, nowNs)
        }
    }

    /**
     * Starts a warm-up window (`5 s` for delegate switch to GPU, `3 s` for delegate switch to CPU,
     * filter switch, rebind, or resume) on both [governor] and [latencyProbe].
     */
    fun triggerWarmup(
        isGpuDelegateSwitch: Boolean,
        reason: String,
        nowElapsedNs: Long = FrameClock.elapsedRealtimeNs()
    ) {
        val govDurationMs = if (isGpuDelegateSwitch) {
            PerformanceGovernor.WARMUP_GPU_DELEGATE_DURATION_MS
        } else {
            PerformanceGovernor.WARMUP_DEFAULT_DURATION_MS
        }
        val probeDurationNs = if (isGpuDelegateSwitch) {
            LatencyProbe.WARMUP_GPU_DELEGATE_DURATION_NS
        } else {
            LatencyProbe.WARMUP_DEFAULT_DURATION_NS
        }
        governor.startWarmup(
            nowMs = nowElapsedNs / 1_000_000L,
            durationMs = govDurationMs,
            reason = reason
        )
        latencyProbe.startWarmup(
            nowNs = nowElapsedNs,
            durationNs = probeDurationNs
        )
    }

    @Volatile
    var useGpuDelegate: Boolean = false
        private set

    fun startProvider(initialConfig: ProviderConfig = ProviderConfig()) {
        lastProviderConfig = initialConfig
        useGpuDelegate = initialConfig.useGpu
        provider.updateTargetCadenceHz(governor.targetTrackerCadenceHz)
        if (isProviderStarted) {
            provider.updateConfig(initialConfig)
            latencyProbe.delegateLabel = provider.activeDelegateLabel
            latencyProbe.trackerStatusLabel = provider.statusMessage
            return
        }
        trackerStartCount++
        isProviderStarted = true
        hasStartedProviderOnce = true
        wasProviderClosedForLongStop = false
        provider.setPaused(isPaused)
        provider.start(initialConfig)
        latencyProbe.delegateLabel = provider.activeDelegateLabel
        latencyProbe.trackerStatusLabel = provider.statusMessage
    }

    private fun ensureProviderStartedOnResume() {
        if (!isProviderStarted && (hasStartedProviderOnce || wasProviderClosedForLongStop)) {
            startProvider(lastProviderConfig)
        }
    }

    /**
     * Switches MediaPipe delegate (CPU vs GPU) without restarting the camera, resets
     * `LatencyProbe` rolling windows for clean comparison, and records the active delegate label
     * (including `"GPU->CPU fallback"` or `"GPU->CPU (watchdog)"`).
     */
    fun setUseGpuDelegate(useGpu: Boolean) {
        useGpuDelegate = useGpu
        provider.switchDelegate(useGpu)
        latencyProbe.delegateLabel = provider.activeDelegateLabel
        latencyProbe.trackerStatusLabel = provider.statusMessage
        latencyProbe.reset()
        triggerWarmup(
            isGpuDelegateSwitch = useGpu,
            reason = if (useGpu) "delegateSwitch:GPU" else "delegateSwitch:CPU"
        )
    }

    /**
     * Enables or disables the face blendshape output of the landmarker at runtime (`BLEND:ON/OFF`),
     * recreating the landmarker on the analysis thread (like the delegate switch) with a 3-second
     * warm-up window. When `OFF`, blendshape-driven behaviors hold their last value.
     */
    fun setDebugBlendshapesEnabled(enabled: Boolean) {
        isDebugBlendshapesEnabled = enabled
        if (!enabled) {
            val currentBlendshapes = signalPipeline.getLatestState().blendshapes
            if (currentBlendshapes.isNotEmpty()) {
                lastHeldBlendshapes = HashMap(currentBlendshapes)
            }
        }
        latencyProbe.blendshapesEnabledLabel = if (enabled) "ON" else "OFF (held)"
        val effectiveOutputBlend = enabled && (activeEffect?.requiresBlendshapes ?: true)
        lastProviderConfig = lastProviderConfig.copy(outputBlendshapes = effectiveOutputBlend)
        provider.setBlendshapesEnabled(effectiveOutputBlend)
        triggerWarmup(
            isGpuDelegateSwitch = false,
            reason = if (enabled) "blendSwitch:ON" else "blendSwitch:OFF"
        )
    }

    /**
     * Toggles the thread priority of the analysis/conversion worker (`PRIO:NORM/LOW`).
     */
    fun setUseLowAnalysisThreadPriority(lowPriority: Boolean) {
        useLowAnalysisThreadPriority = lowPriority
        latencyProbe.analysisThreadPriorityLabel = if (lowPriority) "LOW" else "NORM"
        provider.setUseLowThreadPriority(lowPriority)
    }

    /**
     * Enables or disables skipping conversion when an inference is in flight and the staged
     * converted frame is younger than 1.5 camera intervals (default `false`).
     */
    fun setSkipConversionIfPendingFresh(enabled: Boolean) {
        skipConversionIfPendingFresh = enabled
        provider.setSkipConversionIfPendingFresh(enabled)
    }

    /**
     * Toggles the conversion path (`CONV:FAST` single-pass vs `CONV:OLD` two-stage),
     * resets `LatencyProbe` rolling windows for clean comparison, and triggers a 3-second warm-up.
     */
    fun setConversionMode(mode: ConversionMode) {
        conversionMode = mode
        latencyProbe.conversionModeLabel = mode.name
        provider.setConversionMode(mode)
        latencyProbe.reset()
        triggerWarmup(
            isGpuDelegateSwitch = false,
            reason = "convMode:${mode.name}"
        )
    }

    /**
     * Resets `ClockOffsetCalibrator` and `LatencyProbe` rolling windows whenever the CameraX
     * `ImageAnalysis` use case is rebound (e.g. toggling `YUV_420_888` vs `RGBA_8888` or
     * `LEGACY` vs `SMALL` resolution mode), and starts a 3-second warm-up window.
     */
    fun onAnalysisRebound(
        formatLabel: String = latencyProbe.analysisFormatLabel,
        resolutionModeLabel: String = latencyProbe.resolutionModeLabel
    ) {
        lastSubmittedTimestampNs.set(0L)
        lastRawSensorTimestampNs.set(0L)
        clockOffsetCalibrator.reset()
        latencyProbe.analysisFormatLabel = formatLabel
        latencyProbe.resolutionModeLabel = resolutionModeLabel
        when (resolutionModeLabel) {
            AnalysisResolutionMode.SMALL.name -> provider.setResolutionModeForTelemetry(AnalysisResolutionMode.SMALL)
            AnalysisResolutionMode.LEGACY.name -> provider.setResolutionModeForTelemetry(AnalysisResolutionMode.LEGACY)
        }
        latencyProbe.delegateLabel = provider.activeDelegateLabel
        latencyProbe.reset()
        triggerWarmup(
            isGpuDelegateSwitch = false,
            reason = "rebind:$formatLabel:$resolutionModeLabel"
        )
    }

    /**
     * Extracts `CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE` from a CameraX `CameraInfo`
     * using `Camera2CameraInfo` with the experimental Camera2 interop opt-in.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    fun extractSensorTimestampSource(cameraInfo: CameraInfo): Int {
        return try {
            Camera2CameraInfo.from(cameraInfo)
                .getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
                ?: ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
        } catch (_: Throwable) {
            ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
        }
    }

    /**
     * Extracts [CameraIntrinsics] (`LENS_INFO_AVAILABLE_FOCAL_LENGTHS`, `SENSOR_INFO_PHYSICAL_SIZE`,
     * `SENSOR_INFO_PIXEL_ARRAY_SIZE`, `SENSOR_INFO_ACTIVE_ARRAY_SIZE`) from a CameraX `CameraInfo`
     * using `Camera2CameraInfo`.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    fun extractCameraIntrinsics(cameraInfo: CameraInfo): CameraIntrinsics {
        return try {
            val c2Info = Camera2CameraInfo.from(cameraInfo)
            val focalLengths = c2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val physicalSize = c2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val pixelArraySize = c2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            val activeArrayRect = c2Info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

            val focalMm = focalLengths?.firstOrNull { it > 0f } ?: 0f
            val sensorW = physicalSize?.width ?: 0f
            val sensorH = physicalSize?.height ?: 0f
            val pixelW = pixelArraySize?.width ?: (activeArrayRect?.width() ?: 0)
            val pixelH = pixelArraySize?.height ?: (activeArrayRect?.height() ?: 0)

            CameraIntrinsics(
                focalLengthMm = focalMm,
                sensorWidthMm = sensorW,
                sensorHeightMm = sensorH,
                pixelArrayWidth = pixelW,
                pixelArrayHeight = pixelH,
                activeArrayRect = activeArrayRect
            )
        } catch (_: Throwable) {
            CameraIntrinsics.UNKNOWN
        }
    }

    /**
     * Explicitly configures active camera intrinsics (used on camera bind and in unit tests).
     */
    fun configureCameraIntrinsics(intrinsics: CameraIntrinsics) {
        this.activeIntrinsics = intrinsics
        this.lastCaptureFocalLengthMm = 0f
        this.lastScalerCropRegion = null
    }

    /**
     * Records dynamic per-frame `CaptureResult.LENS_FOCAL_LENGTH` and `CaptureResult.SCALER_CROP_REGION`
     * from a `Camera2Interop` session capture callback (handles multi-focal / logical cameras and zoom).
     */
    fun recordCaptureResultMetadata(
        focalLengthMm: Float?,
        scalerCropRegion: Rect? = null,
        scalerCropLeft: Int = scalerCropRegion?.left ?: 0,
        scalerCropTop: Int = scalerCropRegion?.top ?: 0,
        scalerCropWidth: Int = scalerCropRegion?.let { (it.right - it.left).takeIf { w -> w > 0 } ?: it.width() } ?: 0,
        scalerCropHeight: Int = scalerCropRegion?.let { (it.bottom - it.top).takeIf { h -> h > 0 } ?: it.height() } ?: 0
    ) {
        if (focalLengthMm != null && focalLengthMm.isFinite() && focalLengthMm > 0f) {
            lastCaptureFocalLengthMm = focalLengthMm
        }
        if (scalerCropWidth > 0 && scalerCropHeight > 0) {
            lastScalerCropRegion = (scalerCropRegion ?: Rect()).apply {
                left = scalerCropLeft
                top = scalerCropTop
                right = scalerCropLeft + scalerCropWidth
                bottom = scalerCropTop + scalerCropHeight
            }
        }
    }

    /**
     * Configures the sensor timestamp clock source (`SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME` or
     * `SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN`) and resets warm-up offset calibration.
     */
    fun configureTimestampSource(source: Int) {
        clockOffsetCalibrator.configureSource(source)
    }

    /**
     * Records camera lifecycle binding and configures `ClockOffsetCalibrator` and `activeIntrinsics`
     * from `Camera2CameraInfo` (or falls back to querying `CameraManager` when `cameraInfo` is null).
     */
    @OptIn(ExperimentalCamera2Interop::class)
    @JvmOverloads
    fun recordCameraBound(cameraInfo: CameraInfo? = null) {
        cameraStartCount++
        lastCaptureFocalLengthMm = 0f
        lastScalerCropRegion = null
        if (cameraInfo != null) {
            val source = extractSensorTimestampSource(cameraInfo)
            clockOffsetCalibrator.configureSource(source)
            val extracted = extractCameraIntrinsics(cameraInfo)
            if (extracted.isValid) {
                activeIntrinsics = extracted
            } else {
                val fallback = queryCameraIntrinsicsFromContext(governor.appContext, isFrontCamera)
                if (fallback.isValid) {
                    activeIntrinsics = fallback
                }
            }
        } else {
            val fallbackSource = queryTimestampSourceFromContext(governor.appContext, isFrontCamera)
            clockOffsetCalibrator.configureSource(fallbackSource)
            val fallbackIntrinsics = queryCameraIntrinsicsFromContext(governor.appContext, isFrontCamera)
            if (fallbackIntrinsics.isValid) {
                activeIntrinsics = fallbackIntrinsics
            }
        }
    }

    private fun queryTimestampSourceFromContext(context: Context?, isFront: Boolean): Int {
        if (context == null) return ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
        return try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
                ?: return ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
            val targetFacing = if (isFront) {
                CameraCharacteristics.LENS_FACING_FRONT
            } else {
                CameraCharacteristics.LENS_FACING_BACK
            }
            for (cameraId in manager.cameraIdList) {
                val chars = manager.getCameraCharacteristics(cameraId)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing == targetFacing) {
                    return chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
                        ?: ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
                }
            }
            ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
        } catch (_: Throwable) {
            ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN
        }
    }

    private fun queryCameraIntrinsicsFromContext(context: Context?, isFront: Boolean): CameraIntrinsics {
        if (context == null) return CameraIntrinsics.UNKNOWN
        return try {
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
                ?: return CameraIntrinsics.UNKNOWN
            val targetFacing = if (isFront) {
                CameraCharacteristics.LENS_FACING_FRONT
            } else {
                CameraCharacteristics.LENS_FACING_BACK
            }
            for (cameraId in manager.cameraIdList) {
                val chars = manager.getCameraCharacteristics(cameraId)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing == targetFacing) {
                    val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    val physicalSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    val pixelArraySize = chars.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
                    val activeArrayRect = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

                    val focalMm = focalLengths?.firstOrNull { it > 0f } ?: 0f
                    val sensorW = physicalSize?.width ?: 0f
                    val sensorH = physicalSize?.height ?: 0f
                    val pixelW = pixelArraySize?.width ?: (activeArrayRect?.width() ?: 0)
                    val pixelH = pixelArraySize?.height ?: (activeArrayRect?.height() ?: 0)

                    return CameraIntrinsics(
                        focalLengthMm = focalMm,
                        sensorWidthMm = sensorW,
                        sensorHeightMm = sensorH,
                        pixelArrayWidth = pixelW,
                        pixelArrayHeight = pixelH,
                        activeArrayRect = activeArrayRect
                    )
                }
            }
            CameraIntrinsics.UNKNOWN
        } catch (_: Throwable) {
            CameraIntrinsics.UNKNOWN
        }
    }

    /**
     * Dynamically updates requested tracker outputs when switching filters without restarting
     * the camera or tracker (README Section 3 Rule 1, Section 8 & Section 15).
     */
    fun applyEffectRequirements(effect: CompiledEffect?) {
        val filterChanged = (activeEffect?.packageId != effect?.packageId)
        activeEffect = effect
        governor.packageMinTier = effect?.minTier
        governor.setFilterIrisRequirement(effect?.requiresIris == true)
        latencyProbe.configuredIrisAnalysisSizePx = governor.irisAnalysisSizePx
        if (filterChanged) {
            triggerWarmup(
                isGpuDelegateSwitch = false,
                reason = "filterSwitch:${effect?.packageId ?: "none"}"
            )
        }
        val updatedConfig = if (effect == null) {
            ProviderConfig(
                outputPose = false,
                outputBlendshapes = isDebugBlendshapesEnabled,
                outputLandmarks = false,
                useGpu = useGpuDelegate,
                requestedBlendshapes = emptySet(),
                analysisLongEdgePx = governor.targetAnalysisSizePx
            )
        } else {
            ProviderConfig(
                outputPose = effect.requiresPose,
                outputBlendshapes = isDebugBlendshapesEnabled,
                outputLandmarks = effect.requiredLandmarkCount > 0 || effect.requiresIris,
                useGpu = useGpuDelegate,
                requestedBlendshapes = effect.requestedBlendshapes,
                analysisLongEdgePx = governor.targetAnalysisSizePx
            )
        }
        lastProviderConfig = updatedConfig
        provider.updateAnalysisSize(governor.targetAnalysisSizePx)
        if (isProviderStarted) {
            provider.updateConfig(updatedConfig)
        }
    }

    /**
     * Updates the configured iris analysis size (`320`, `480`, or `640`) on both [governor] and [latencyProbe],
     * and propagates `targetAnalysisSizePx` to [provider].
     */
    fun setConfiguredIrisAnalysisSizePx(sizePx: Int): Int {
        val snapped = governor.setConfiguredIrisAnalysisSizePx(sizePx)
        latencyProbe.configuredIrisAnalysisSizePx = snapped
        provider.updateAnalysisSize(governor.targetAnalysisSizePx)
        lastProviderConfig = lastProviderConfig.copy(analysisLongEdgePx = governor.targetAnalysisSizePx)
        if (isProviderStarted) {
            provider.updateConfig(lastProviderConfig)
        }
        return snapped
    }

    /**
     * Cycles the configured iris analysis size across `320 -> 480 -> 640 -> 320`.
     */
    fun cycleConfiguredIrisAnalysisSizePx(): Int {
        val next = governor.cycleConfiguredIrisAnalysisSizePx()
        latencyProbe.configuredIrisAnalysisSizePx = next
        provider.updateAnalysisSize(governor.targetAnalysisSizePx)
        lastProviderConfig = lastProviderConfig.copy(analysisLongEdgePx = governor.targetAnalysisSizePx)
        if (isProviderStarted) {
            provider.updateConfig(lastProviderConfig)
        }
        return next
    }

    @Volatile
    var isPaused: Boolean = false
        private set

    @Volatile
    var isStopped: Boolean = false
        private set

    @Volatile
    var stoppedAtElapsedMs: Long = -1L
        private set

    @Volatile
    var isFrontCamera: Boolean = true
        private set

    @Volatile
    var displayRotationDegrees: Int = 0
        private set

    /**
     * Transitions the session and `FaceGeometryProvider` to IDLE without closing the provider:
     * - Rejects new frames and closes any pending frame's `ImageProxy` once.
     * - Drops any in-flight inference result that arrives while paused.
     */
    fun onPause() {
        isPaused = true
        provider.setPaused(true)
    }

    /**
     * Marks the session as stopped (`ON_STOP`) and records [nowElapsedMs] so a stop lasting
     * `>= 45 s` closes the `FaceGeometryProvider` and lazily restarts it on resume.
     */
    @JvmOverloads
    fun onStop(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L) {
        if (!isPaused) {
            onPause()
        }
        if (!isStopped) {
            isStopped = true
            stoppedAtElapsedMs = nowElapsedMs
        }
    }

    /**
     * Checks whether the session has been in `ON_STOP` for at least [LONG_STOP_PROVIDER_IDLE_TIMEOUT_MS]
     * (45 s). If so, closes the `FaceGeometryProvider` so it will be lazily restarted on resume.
     */
    @JvmOverloads
    fun checkLongStopTimeout(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L): Boolean {
        if (isStopped && isProviderStarted && stoppedAtElapsedMs >= 0L &&
            (nowElapsedMs - stoppedAtElapsedMs) >= LONG_STOP_PROVIDER_IDLE_TIMEOUT_MS
        ) {
            provider.stop()
            isProviderStarted = false
            wasProviderClosedForLongStop = true
            return true
        }
        return false
    }

    @JvmOverloads
    fun onStart(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L) {
        checkLongStopTimeout(nowElapsedMs)
        isStopped = false
        stoppedAtElapsedMs = -1L
    }

    /**
     * Resumes from pause (`ON_RESUME`):
     * - If the session was in `ON_STOP` for `>= 45 s`, closes the provider (if not already closed)
     *   and lazily restarts it on resume (so [trackerStartCount] becomes 2 in the long-stop case).
     * - Resets `FaceLifecycle` to `NO_FACE`, resets `SignalPipeline` One-Euro filters and
     *   `LatencyCompensator` velocity (so prediction cannot extrapolate stale pre-pause velocity).
     * - Resets active `CompiledEffect` behavior/clip/trigger/spring state.
     * - Resets `ClockOffsetCalibrator` and `LatencyProbe` rolling windows.
     */
    @JvmOverloads
    fun onResume(nowElapsedMs: Long = FrameClock.elapsedRealtimeNs() / 1_000_000L) {
        if (isStopped) {
            checkLongStopTimeout(nowElapsedMs)
            isStopped = false
            stoppedAtElapsedMs = -1L
        }
        val wasPaused = isPaused
        isPaused = false
        provider.setPaused(false)

        if (wasPaused) {
            signalPipeline.reset()
            activeEffect?.reset()
            clockOffsetCalibrator.reset()
            latencyProbe.reset()
        } else {
            clockOffsetCalibrator.reset()
        }
        triggerWarmup(
            isGpuDelegateSwitch = false,
            reason = "resume",
            nowElapsedNs = nowElapsedMs * 1_000_000L
        )
        lastSubmittedTimestampNs.set(0L)
        lastRawSensorTimestampNs.set(0L)

        // Lazily restart the provider if it was closed due to a >45s ON_STOP
        ensureProviderStartedOnResume()
    }

    fun onDisplayRotationChanged(rotationDegrees: Int) {
        displayRotationDegrees = FacePoseAdapter.normalizeRotationDegrees(rotationDegrees)
    }

    fun onCameraLensChanged(isFront: Boolean, rotationDegrees: Int = displayRotationDegrees) {
        val changed = (this.isFrontCamera != isFront)
        this.isFrontCamera = isFront
        this.displayRotationDegrees = FacePoseAdapter.normalizeRotationDegrees(rotationDegrees)
        if (changed) {
            // Reset transient pose filter, velocity history, clock offset calibration, and cached intrinsics on lens switch
            signalPipeline.poseFilter.reset()
            signalPipeline.latencyCompensator.reset()
            lastSubmittedTimestampNs.set(0L)
            lastRawSensorTimestampNs.set(0L)
            lastCaptureFocalLengthMm = 0f
            lastScalerCropRegion = null
            activeIntrinsics = queryCameraIntrinsicsFromContext(governor.appContext, isFront)
            val source = queryTimestampSourceFromContext(governor.appContext, isFront)
            clockOffsetCalibrator.configureSource(source)
            triggerWarmup(
                isGpuDelegateSwitch = false,
                reason = "lensSwitch"
            )
        }
    }

    /**
     * Submits a CameraX analysis frame, calibrating its sensor timestamp to `elapsedRealtimeNanos`,
     * deriving per-frame [ProjectionParams] from camera intrinsics + capture result metadata + crop,
     * throttling to `governor.targetTrackerCadenceHz` (minimum 15 Hz), and guaranteeing that
     * `imageProxy` is closed exactly once if throttled or paused.
     */
    @JvmOverloads
    fun submitAnalysisFrame(
        imageProxy: ImageProxy?,
        timestampNs: Long = FrameClock.elapsedRealtimeNs(),
        rotationDegrees: Int = displayRotationDegrees,
        isFrontCamera: Boolean = this.isFrontCamera,
        imageToViewTransform: FloatArray? = null,
        sensorTimestampNs: Long = -1L,
        arrivalElapsedRealtimeNs: Long = FrameClock.elapsedRealtimeNs(),
        projectionParamsOverride: ProjectionParams? = null
    ): Boolean {
        var handedToProvider = false
        try {
            if (isPaused) {
                return false
            }
            ensureProviderStartedOnResume()
            this.isFrontCamera = isFrontCamera
            val normalizedRotation = FacePoseAdapter.normalizeRotationDegrees(rotationDegrees)
            this.displayRotationDegrees = normalizedRotation

            // Record camera frame arrival in LatencyProbe before cadence throttling
            latencyProbe.recordCameraFrame(arrivalElapsedRealtimeNs)

            val proxySensorTs = if (sensorTimestampNs >= 0L) {
                sensorTimestampNs
            } else {
                try {
                    imageProxy?.imageInfo?.timestamp ?: 0L
                } catch (_: Throwable) {
                    0L
                }
            }

            val hasSensorTs = proxySensorTs > 0L
            val rawSensorTs = if (hasSensorTs) proxySensorTs else timestampNs
            val calibratedCaptureTs = if (hasSensorTs) {
                clockOffsetCalibrator.toElapsedRealtimeNs(rawSensorTs, arrivalElapsedRealtimeNs)
            } else {
                timestampNs
            }

            val cadenceHz = governor.targetTrackerCadenceHz.coerceIn(
                PerformanceGovernor.MIN_TRACKER_CADENCE_HZ,
                PerformanceGovernor.MAX_TRACKER_CADENCE_HZ
            )
            provider.updateTargetCadenceHz(cadenceHz)

            // Cheap session pre-filter with a smaller factor (0.50 of the target period) so
            // high-rate bursts (e.g. 60/120 fps when target is 30 Hz) are dropped before conversion,
            // while any normal cadence frame that passes the session pre-filter (>= 0.85 * period)
            // is never deferred at the provider's 0.85 cadence gate.
            val minIntervalNs = ((1_000_000_000L / cadenceHz) * 50L) / 100L
            val prevNs = lastSubmittedTimestampNs.get()
            val prevRawNs = lastRawSensorTimestampNs.get()
            val cadenceTs = if (hasSensorTs && rawSensorTs != prevRawNs) calibratedCaptureTs else timestampNs

            if (prevNs != 0L && (cadenceTs - prevNs) < minIntervalNs) {
                // Frame throttled by session pre-filter: closed once in finally block
                sessionCadenceThrottledCounter.incrementAndGet()
                return false
            }

            lastSubmittedTimestampNs.set(cadenceTs)
            if (hasSensorTs) {
                lastRawSensorTimestampNs.set(rawSensorTs)
            }
            provider.updateAnalysisSize(governor.targetAnalysisSizePx)

            // Compute per-frame ProjectionParams from activeIntrinsics + dynamic CaptureResult overrides + ImageProxy crop
            val projParams = projectionParamsOverride ?: run {
                val baseIntrinsics = activeIntrinsics
                val capFocal = lastCaptureFocalLengthMm
                val capCrop = lastScalerCropRegion
                val mergedIntrinsics = if (capFocal > 0f || capCrop != null) {
                    val cw = capCrop?.let { (it.right - it.left).takeIf { w -> w > 0 } ?: it.width() } ?: baseIntrinsics.scalerCropWidth
                    val ch = capCrop?.let { (it.bottom - it.top).takeIf { h -> h > 0 } ?: it.height() } ?: baseIntrinsics.scalerCropHeight
                    baseIntrinsics.copy(
                        focalLengthMm = if (capFocal > 0f) capFocal else baseIntrinsics.focalLengthMm,
                        scalerCropRegion = capCrop ?: baseIntrinsics.scalerCropRegion,
                        scalerCropLeft = capCrop?.left ?: baseIntrinsics.scalerCropLeft,
                        scalerCropTop = capCrop?.top ?: baseIntrinsics.scalerCropTop,
                        scalerCropWidth = cw,
                        scalerCropHeight = ch
                    )
                } else {
                    baseIntrinsics
                }

                val imgW = imageProxy?.width ?: 640
                val imgH = imageProxy?.height ?: 360
                val cropRect = try {
                    imageProxy?.cropRect
                } catch (_: Throwable) {
                    null
                }

                FacePoseAdapter.computeProjectionParams(
                    intrinsics = mergedIntrinsics,
                    bufferWidth = imgW,
                    bufferHeight = imgH,
                    cropRect = cropRect,
                    rotationDegrees = normalizedRotation
                )
            }

            lastProjectionParams = projParams
            latencyProbe.recordProjectionMetadata(projParams)

            val frame = AnalysisFrame(
                imageProxy = imageProxy,
                timestampNs = calibratedCaptureTs,
                rotationDegrees = normalizedRotation,
                isFrontCamera = isFrontCamera,
                imageToViewTransform = imageToViewTransform,
                sensorTimestampNs = rawSensorTs,
                submittedAtNs = arrivalElapsedRealtimeNs,
                projectionParams = projParams
            )
            handedToProvider = true
            provider.submit(frame)
            return true
        } finally {
            if (!handedToProvider) {
                try {
                    imageProxy?.close()
                } catch (_: Throwable) {
                }
            }
        }
    }

    fun stop() {
        isPaused = true
        isStopped = true
        isProviderStarted = false
        wasProviderClosedForLongStop = false
        provider.stop()
    }
}

