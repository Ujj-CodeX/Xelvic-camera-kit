package com.example.engine.debug

import android.os.Debug
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import java.util.Arrays
import java.util.Locale

/**
 * LatencyProbe (AUDIT.md Section 3.2 & 3.3, README Sections 7 & 10):
 *
 * Zero-allocation runtime probe measuring the 5-second rolling latency window, stage breakdown,
 * warm-up gating, governor transition reasons, and camera intrinsics / FOV telemetry:
 * - `conversionMs`: Time until the `MPImage` handed to the landmarker is ready (includes pixel
 *   unpacking, rotation, governor second-stage downscale, and `MPImage` construction).
 * - `pendingWaitMs`: Time between conversion completion and `detectAsync` dispatch.
 * - `inferenceMs`: MediaPipe FaceLandmarker inference duration.
 * - `queueMs`: Time spent waiting in the analysis handoff slot before processing starts.
 * - `ageAtRender`: `tRender - tCapture` (`renderTimestampNs - faceState.poseTimestampNs`) in the
 *   unified `SystemClock.elapsedRealtimeNanos()` clock domain.
 * - `predErrPosMm` / `predErrRotDeg`: When a new tracker sample arrives, pose prediction error
 *   at that sample's capture timestamp compared against the newly measured filtered pose.
 * - `gcCountDelta5s`: 5-second rolling delta of `Debug.getRuntimeStat("art.gc.gc-count")`,
 *   sampled at ~2 Hz so steady-state allocation can be verified on device.
 * - `warmup`: Samples during the warm-up window (5 s after GPU delegate switch, 3 s after filter
 *   switch, rebind, or resume) are marked `warmup=true`, excluded from HUD rolling averages and
 *   `age p50/p95`, and kept in the CSV with `warmup=true`.
 * - `govReason`: Logs the reason (`signal name + value`) of the latest governor demote/promote
 *   on `cachedHudLine` and in the CSV (`govReason` column).
 */
class LatencyProbe(
    val windowDurationNs: Long = DEFAULT_WINDOW_DURATION_NS,
    val hudRefreshIntervalNs: Long = DEFAULT_HUD_REFRESH_INTERVAL_NS,
    val windowCapacity: Int = DEFAULT_WINDOW_CAPACITY,
    val csvRingCapacity: Int = DEFAULT_CSV_RING_CAPACITY,
    var gcCountProvider: () -> Long = ::defaultArtGcCount
) {
    companion object {
        const val DEFAULT_WINDOW_DURATION_NS = 5_000_000_000L // 5.0 seconds
        const val DEFAULT_HUD_REFRESH_INTERVAL_NS = 500_000_000L // 500 ms (~2 Hz)
        const val DEFAULT_WINDOW_CAPACITY = 600 // 5s at 120 Hz
        const val DEFAULT_CSV_RING_CAPACITY = 1800 // 30s at 60 Hz
        private const val GC_WINDOW_CAPACITY = 64 // >5s at 2 Hz

        const val WARMUP_DEFAULT_DURATION_NS = 3_000_000_000L // 3.0 seconds
        const val WARMUP_GPU_DELEGATE_DURATION_NS = 5_000_000_000L // 5.0 seconds

        // Chunk R1 fast head-motion thresholds for mask-offset telemetry
        const val FAST_MOTION_LINEAR_SPEED_MPS = 0.12f // 120 mm/s
        const val FAST_MOTION_ANGULAR_SPEED_DEG_S = 45.0f // 45 deg/s

        const val CSV_HEADER =
            "timestamp,ageMs,conversionMs,pendingWaitMs,inferenceMs,queueMs,cameraFps,trackerFps,renderFps,predictionOn,predErrPosMm,predErrRotDeg,fovYDeg,fovSource,focalLengthMm,scalerCropRegion,depthBits,warmup,govReason,idleGapPendingMs,idleGapNoPendingMs,pumpDelayMs,swapMs,convMode,unpackMs,yuvToRgbMs,bitmapCopyMs,rotateScaleMs,mpImageBuildMs,srcWidth,srcHeight,srcRotationDeg"

        fun defaultArtGcCount(): Long {
            return try {
                Debug.getRuntimeStat("art.gc.gc-count")?.toLongOrNull() ?: 0L
            } catch (_: Throwable) {
                0L
            }
        }

        /**
         * Pure-float zero-allocation helper computing the temporal delta (`|cameraFrameTime - landmarkTime|`)
         * and projected screen-space mask offset (in pixels) during head motion (Chunk R1):
         * - `outMetrics[0]` = `absDeltaMs` (`|cameraTsNs - landmarkTsNs|` in ms)
         * - `outMetrics[1]` = `signedDeltaMs` (`(cameraTsNs - landmarkTsNs)` in ms)
         * - `outMetrics[2]` = `maskOffsetPx` (projected screen-space offset in pixels)
         * - `outMetrics[3]` = `headSpeedMps` (effective surface speed in m/s combining linear + angular motion)
         * - `outMetrics[4]` = `headAngularSpeedDegPerSec` (angular speed in deg/s)
         */
        @JvmOverloads
        fun computeMaskMotionOffset(
            cameraFrameSensorTimestampNs: Long,
            renderTimestampNs: Long,
            faceState: FaceState,
            viewportWidth: Int = 1080,
            viewportHeight: Int = 1920,
            outMetrics: FloatArray = FloatArray(5)
        ): FloatArray {
            val landmarkTsNs = faceState.poseTimestampNs
            val refCamTsNs = if (cameraFrameSensorTimestampNs > 0L) {
                cameraFrameSensorTimestampNs
            } else {
                renderTimestampNs
            }
            if (!faceState.isVisible || faceState.lifecycle != FaceLifecycle.State.TRACKING || landmarkTsNs <= 0L || refCamTsNs <= 0L) {
                java.util.Arrays.fill(outMetrics, 0f)
                return outMetrics
            }

            val signedDeltaMs = (refCamTsNs - landmarkTsNs).toFloat() / 1_000_000f
            val absDeltaMs = kotlin.math.abs(signedDeltaMs)
            val dtSec = (absDeltaMs / 1000f).coerceIn(0f, 0.50f)

            val vx = faceState.velocity.getOrElse(0) { 0f }
            val vy = faceState.velocity.getOrElse(1) { 0f }
            val wx = faceState.angularVelocity.getOrElse(0) { 0f }
            val wy = faceState.angularVelocity.getOrElse(1) { 0f }
            val wz = faceState.angularVelocity.getOrElse(2) { 0f }

            val fw = faceState.faceWidth.coerceAtLeast(0.08f)
            val surfaceRadiusMeters = 0.5f * fw

            // Tangential surface velocity from pitch (wx) and yaw (wy) combined with linear translation (vx, vy)
            val surfVx = vx + wy * surfaceRadiusMeters
            val surfVy = vy - wx * surfaceRadiusMeters
            val velocitySpeedMps = kotlin.math.sqrt(surfVx * surfVx + surfVy * surfVy)
            val angularSpeedRadS = kotlin.math.sqrt(wx * wx + wy * wy + wz * wz)
            val angularSpeedDegS = Math.toDegrees(angularSpeedRadS.toDouble()).toFloat()

            // Also account for direct pose translation delta between predicted/current poseMatrix and landmarkPoseMatrix
            val dxPose = faceState.poseMatrix[12] - faceState.landmarkPoseMatrix[12]
            val dyPose = faceState.poseMatrix[13] - faceState.landmarkPoseMatrix[13]
            val poseDeltaMeters = kotlin.math.sqrt(dxPose * dxPose + dyPose * dyPose)
            val dispMeters = maxOf(velocitySpeedMps * dtSec, poseDeltaMeters)
            val effectiveSpeedMps = if (dtSec > 1e-4f) maxOf(velocitySpeedMps, dispMeters / dtSec) else velocitySpeedMps

            val safeW = viewportWidth.coerceAtLeast(1).toFloat()
            val safeH = viewportHeight.coerceAtLeast(1).toFloat()
            val aspect = safeW / safeH
            val fovYViewDeg = faceState.projectionParams.visibleViewFovYDeg(aspect)
            val f = com.example.engine.signal.FacePoseAdapter.focalFactorFromFovY(fovYViewDeg)
            val rawTz = faceState.poseMatrix[14].let { if (kotlin.math.abs(it) > 0.05f) it else -0.50f }
            val absZ = kotlin.math.abs(rawTz).coerceIn(0.25f, 1.50f)
            val pxPerMeter = (f * safeH) / (2.0f * absZ)
            val maskOffsetPx = dispMeters * pxPerMeter

            outMetrics[0] = if (absDeltaMs.isFinite()) absDeltaMs else 0f
            outMetrics[1] = if (signedDeltaMs.isFinite()) signedDeltaMs else 0f
            outMetrics[2] = if (maskOffsetPx.isFinite()) maskOffsetPx else 0f
            outMetrics[3] = if (effectiveSpeedMps.isFinite()) effectiveSpeedMps else 0f
            outMetrics[4] = if (angularSpeedDegS.isFinite()) angularSpeedDegS else 0f
            return outMetrics
        }

        /**
         * Default shared probe instance connecting CameraSession, XelvicGlSurfaceView, and DebugOverlay.
         * Unit tests may instantiate isolated `LatencyProbe()` instances directly.
         */
        val shared = LatencyProbe()
    }

    private val lock = Any()

    // 5-second rolling window for ageAtRender (recorded ONLY while lifecycle == TRACKING and !isWarmup)
    private val ageTimestampsNs = LongArray(windowCapacity)
    private val ageValuesMs = FloatArray(windowCapacity)
    private val sortScratch = FloatArray(windowCapacity)
    private var ageHead = 0
    private var ageSize = 0

    // 5-second rolling window for camera frame arrivals (pre-throttle)
    private val camTimestampsNs = LongArray(windowCapacity)
    private var camHead = 0
    private var camSize = 0

    // 5-second rolling window for completed tracker snapshots
    private val trkTimestampsNs = LongArray(windowCapacity)
    private val trkConversionMs = FloatArray(windowCapacity)
    private val trkUnpackMs = FloatArray(windowCapacity)
    private val trkYuvToRgbMs = FloatArray(windowCapacity)
    private val trkBitmapCopyMs = FloatArray(windowCapacity)
    private val trkRotateScaleMs = FloatArray(windowCapacity)
    private val trkMpImageBuildMs = FloatArray(windowCapacity)
    private val trkPendingWaitMs = FloatArray(windowCapacity)
    private val trkInferenceMs = FloatArray(windowCapacity)
    private val trkIdleGapPendingMs = FloatArray(windowCapacity)
    private val trkIdleGapNoPendingMs = FloatArray(windowCapacity)
    private val trkPumpDelayMs = FloatArray(windowCapacity)
    private val trkBlendshapesEnabled = BooleanArray(windowCapacity)
    private val trkPredErrPosMm = FloatArray(windowCapacity)
    private val trkPredErrRotDeg = FloatArray(windowCapacity)
    private val trkIsWarmup = BooleanArray(windowCapacity)
    private var trkHead = 0
    private var trkSize = 0

    // 5-second rolling window for rendered GL frames
    private val rndTimestampsNs = LongArray(windowCapacity)
    private var rndHead = 0
    private var rndSize = 0

    // 5-second rolling window for ART GC count samples (sampled at ~2 Hz)
    private val gcTimestampsNs = LongArray(GC_WINDOW_CAPACITY)
    private val gcCounts = LongArray(GC_WINDOW_CAPACITY)
    private var gcHead = 0
    private var gcSize = 0

    // 5-second rolling window for Chunk R1 mask-offset telemetry (landmark time vs camera frame time)
    private val maskTimestampsNs = LongArray(windowCapacity)
    private val maskAbsDeltaMs = FloatArray(windowCapacity)
    private val maskOffsetPxRing = FloatArray(windowCapacity)
    private val maskIsFastMotion = BooleanArray(windowCapacity)
    private val maskMetricScratch = FloatArray(5)
    private var maskHead = 0
    private var maskSize = 0

    // In-memory CSV ring buffer (zero per-frame heap allocation)
    private val csvTimestampNs = LongArray(csvRingCapacity)
    private val csvAgeMs = FloatArray(csvRingCapacity)
    private val csvConversionMs = FloatArray(csvRingCapacity)
    private val csvPendingWaitMs = FloatArray(csvRingCapacity)
    private val csvInferenceMs = FloatArray(csvRingCapacity)
    private val csvQueueMs = FloatArray(csvRingCapacity)
    private val csvCameraFps = FloatArray(csvRingCapacity)
    private val csvTrackerFps = FloatArray(csvRingCapacity)
    private val csvRenderFps = FloatArray(csvRingCapacity)
    private val csvPredictionOn = BooleanArray(csvRingCapacity)
    private val csvPredErrPosMm = FloatArray(csvRingCapacity)
    private val csvPredErrRotDeg = FloatArray(csvRingCapacity)
    private val csvFovYDeg = FloatArray(csvRingCapacity)
    private val csvFovIsIntrinsics = BooleanArray(csvRingCapacity)
    private val csvFocalLengthMm = FloatArray(csvRingCapacity)
    private val csvScalerCropRegion = Array(csvRingCapacity) { "0,0,0x0" }
    private val csvDepthBits = IntArray(csvRingCapacity)
    private val csvWarmup = BooleanArray(csvRingCapacity)
    private val csvGovReason = Array(csvRingCapacity) { "none" }
    private val csvIdleGapPendingMs = FloatArray(csvRingCapacity)
    private val csvIdleGapNoPendingMs = FloatArray(csvRingCapacity)
    private val csvPumpDelayMs = FloatArray(csvRingCapacity)
    private val csvSwapMs = FloatArray(csvRingCapacity)
    private val csvConvMode = Array(csvRingCapacity) { "FAST" }
    private val csvUnpackMs = FloatArray(csvRingCapacity)
    private val csvYuvToRgbMs = FloatArray(csvRingCapacity)
    private val csvBitmapCopyMs = FloatArray(csvRingCapacity)
    private val csvRotateScaleMs = FloatArray(csvRingCapacity)
    private val csvMpImageBuildMs = FloatArray(csvRingCapacity)
    private val csvSrcWidth = IntArray(csvRingCapacity)
    private val csvSrcHeight = IntArray(csvRingCapacity)
    private val csvSrcRotationDeg = IntArray(csvRingCapacity)
    private var csvHead = 0
    private var csvSize = 0

    // Warm-up deadline in elapsedRealtimeNanos domain
    @Volatile
    var warmupUntilNs: Long = 0L
        private set

    // Active analysis pipeline configuration labels for A/B comparison
    @Volatile
    var analysisFormatLabel: String = "YUV_420_888"

    @Volatile
    var delegateLabel: String = "CPU"

    @Volatile
    var resolutionModeLabel: String = "SMALL"

    @Volatile
    var conversionModeLabel: String = "FAST"

    @Volatile
    var blendshapesEnabledLabel: String = "ON"

    @Volatile
    var analysisThreadPriorityLabel: String = "NORM"

    @Volatile
    var renderThreadPriorityLabel: String = "NORM"

    @Volatile
    var trackerStatusLabel: String = ""

    @Volatile
    var governorReasonLabel: String = "none"

    // Chunk B0 Path A / Path B & CameraInputSurface telemetry
    @Volatile
    var pathModeLabel: String = "A"

    @Volatile
    var cameraTextureOk: Boolean = false

    @Volatile
    var latestCameraSensorTimestampNs: Long = 0L

    // Chunk R1 Fast-Motion Mask-Offset HUD telemetry (landmark time vs camera frame time)
    @Volatile
    var lastMaskTimeDeltaMs: Float = 0f
        private set

    @Volatile
    var lastMaskSignedDeltaMs: Float = 0f
        private set

    @Volatile
    var lastMaskOffsetPx: Float = 0f
        private set

    @Volatile
    var lastHeadSpeedMps: Float = 0f
        private set

    @Volatile
    var lastHeadAngularSpeedDegPerSec: Float = 0f
        private set

    @Volatile
    var lastIsFastHeadMotion: Boolean = false
        private set

    @Volatile
    var rollingMaskTimeDeltaMs: Float = 0f
        private set

    @Volatile
    var rollingFastMotionMaskDeltaMs: Float = 0f
        private set

    @Volatile
    var rollingFastMotionMaskOffsetPx: Float = 0f
        private set

    @Volatile
    var fastMotionMaskOffsetP95Px: Float = 0f
        private set

    @Volatile
    var maxFastMotionMaskOffsetPx: Float = 0f
        private set

    @Volatile
    var fastMotionSampleCount: Int = 0
        private set

    /**
     * Stationary iris-landmark jitter and stage timing summary for a specific analysis long-edge bucket
     * (`320`, `480`, or `640`) (Flow 4 — Chunk Primitives-1 EC1).
     */
    data class IrisSizeMetrics(
        val analysisSizePx: Int,
        val stationaryJitterStdDevPx: Float,
        val conversionMs: Float,
        val inferenceMs: Float,
        val trackerFps: Float,
        val sampleCount: Int
    )

    // Pre-allocated 3-bucket stationary iris jitter rings (0 -> 320, 1 -> 480, 2 -> 640)
    private val irisBucketCapacity = 60
    private val irisRightX = FloatArray(3 * 60)
    private val irisRightY = FloatArray(3 * 60)
    private val irisLeftX = FloatArray(3 * 60)
    private val irisLeftY = FloatArray(3 * 60)
    private val irisConvMs = FloatArray(3 * 60)
    private val irisInfMs = FloatArray(3 * 60)
    private val irisTrkFps = FloatArray(3 * 60)
    private val irisBucketHead = IntArray(3)
    private val irisBucketSize = IntArray(3)

    @Volatile
    var configuredIrisAnalysisSizePx: Int = 480

    @Volatile
    var cachedIrisJitterLine: String =
        "IrisJitter[480px]: 320:--px • 480:--px • 640:--px"
        private set

    // Latest camera intrinsics & FOV telemetry
    @Volatile
    var lastFovYDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
        private set

    @Volatile
    var lastFovSource: String = ProjectionParams.FOV_SOURCE_FALLBACK
        private set

    @Volatile
    var lastFocalLengthMm: Float = 0f
        private set

    @Volatile
    var lastScalerCropRegion: String = "none"
        private set

    @Volatile
    var lastDepthBits: Int = 24
        private set

    // Latest stage breakdown metrics
    @Volatile
    var lastConversionMs: Float = 0f
        private set

    @Volatile
    var lastUnpackMs: Float = 0f
        private set

    @Volatile
    var lastYuvToRgbMs: Float = 0f
        private set

    @Volatile
    var lastBitmapCopyMs: Float = 0f
        private set

    @Volatile
    var lastRotateScaleMs: Float = 0f
        private set

    @Volatile
    var lastMpImageBuildMs: Float = 0f
        private set

    @Volatile
    var lastSourceWidth: Int = 0
        private set

    @Volatile
    var lastSourceHeight: Int = 0
        private set

    @Volatile
    var lastCropWidth: Int = 0
        private set

    @Volatile
    var lastCropHeight: Int = 0
        private set

    @Volatile
    var lastSourceRotationDeg: Int = 0
        private set

    @Volatile
    var lastPendingWaitMs: Float = 0f
        private set

    @Volatile
    var lastInferenceMs: Float = 0f
        private set

    @Volatile
    var lastQueueMs: Float = 0f
        private set

    @Volatile
    var lastIdleGapPendingMs: Float = 0f
        private set

    @Volatile
    var lastIdleGapNoPendingMs: Float = 0f
        private set

    @Volatile
    var lastPumpDelayMs: Float = 0f
        private set

    @Volatile
    var lastSwapMs: Float = 0f
        private set

    @Volatile
    var lastAgeAtRenderMs: Float = 0f
        private set

    @Volatile
    var lastDrawCpuMs: Float = 0f
        private set

    @Volatile
    var lastPredErrPosMm: Float = 0f
        private set

    @Volatile
    var lastPredErrRotDeg: Float = 0f
        private set

    @Volatile
    var lastPredictionOn: Boolean = true
        private set

    @Volatile
    var lastSampleWasWarmup: Boolean = false
        private set

    // Computed 5s rolling metrics (updated at ~2 Hz or on explicit refresh)
    @Volatile
    var ageP50Ms: Float = 0f
        private set

    @Volatile
    var ageP95Ms: Float = 0f
        private set

    @Volatile
    var cameraFps: Float = 0f
        private set

    @Volatile
    var trackerFps: Float = 0f
        private set

    @Volatile
    var renderFps: Float = 0f
        private set

    @Volatile
    var rollingConversionMs: Float = 0f
        private set

    @Volatile
    var rollingUnpackMs: Float = 0f
        private set

    @Volatile
    var rollingYuvToRgbMs: Float = 0f
        private set

    @Volatile
    var rollingBitmapCopyMs: Float = 0f
        private set

    @Volatile
    var rollingRotateScaleMs: Float = 0f
        private set

    @Volatile
    var rollingMpImageBuildMs: Float = 0f
        private set

    @Volatile
    var rollingPendingWaitMs: Float = 0f
        private set

    @Volatile
    var rollingInferenceMs: Float = 0f
        private set

    @Volatile
    var rollingInferenceBlendOnMs: Float = 0f
        private set

    @Volatile
    var rollingInferenceBlendOffMs: Float = 0f
        private set

    @Volatile
    var rollingIdleGapPendingMs: Float = 0f
        private set

    @Volatile
    var rollingIdleGapNoPendingMs: Float = 0f
        private set

    @Volatile
    var rollingPumpDelayMeanMs: Float = 0f
        private set

    @Volatile
    var rollingPumpDelayMaxMs: Float = 0f
        private set

    @Volatile
    var rollingPredErrPosMm: Float = 0f
        private set

    @Volatile
    var rollingPredErrRotDeg: Float = 0f
        private set

    @Volatile
    var gcCountDelta5s: Int = 0
        private set

    @Volatile
    var cachedHudLine: String =
        "age p50/p95: --/--ms | trk: 0.0fps | cam: 0.0fps | rnd: 0.0fps | wait: 0.0ms | inf: 0.0ms | gc: +0"
        private set

    private var lastHudRefreshNs: Long = 0L

    /**
     * Starts or extends a warm-up window (`5 s` for delegate switch to GPU, `3 s` for filter
     * switch, rebind, or resume). Samples recorded before `warmupUntilNs` are marked `warmup=true`,
     * excluded from HUD rolling averages and `age p50/p95`, and kept in the CSV with `warmup=true`.
     */
    @JvmOverloads
    fun startWarmup(nowNs: Long, durationNs: Long = WARMUP_DEFAULT_DURATION_NS) {
        synchronized(lock) {
            val targetEndNs = nowNs + durationNs.coerceAtLeast(0L)
            if (!isInWarmup(nowNs) || targetEndNs > warmupUntilNs) {
                warmupUntilNs = targetEndNs
            }
        }
    }

    fun isInWarmup(timestampNs: Long): Boolean {
        val until = warmupUntilNs
        return until > 0L && timestampNs < until
    }

    /**
     * Records a governor demote/promote transition reason (`signal name + value`) so it is
     * surfaced on [cachedHudLine] and in the CSV dump.
     */
    fun recordGovernorTransition(reason: String) {
        synchronized(lock) {
            governorReasonLabel = reason
        }
        formatCachedHudLine()
    }

    /**
     * Records derived [ProjectionParams] (`fovYDeg`, `fovSource`, `LENS_FOCAL_LENGTH`, `SCALER_CROP_REGION`).
     */
    fun recordProjectionMetadata(params: ProjectionParams) {
        synchronized(lock) {
            lastFovYDeg = params.fovYDeg
            lastFovSource = params.fovSource
            lastFocalLengthMm = params.focalLengthMm
            lastScalerCropRegion = params.scalerCropDescriptor
        }
    }

    /**
     * Records the active OpenGL depth buffer bit depth (e.g. 24-bit or 16-bit fallback).
     */
    fun recordDepthBits(depthBits: Int) {
        if (depthBits > 0) {
            synchronized(lock) {
                lastDepthBits = depthBits
            }
        }
    }

    /**
     * Records the duration (`swapMs`) of landmarker creation + swap on the dispatch thread.
     */
    fun recordSwapDurationMs(swapMs: Float) {
        if (swapMs.isFinite() && swapMs >= 0f) {
            synchronized(lock) {
                lastSwapMs = swapMs
            }
        }
    }

    /**
     * Records an incoming camera analysis frame arrival timestamp (in `elapsedRealtimeNanos` domain).
     */
    fun recordCameraFrame(timestampNs: Long) {
        synchronized(lock) {
            val tail = (camHead + camSize) % windowCapacity
            if (camSize == windowCapacity) {
                camHead = (camHead + 1) % windowCapacity
            } else {
                camSize++
            }
            camTimestampsNs[tail] = timestampNs
        }
    }

    /**
     * Records a completed perception snapshot and its stage timings.
     * If `doneNs` falls inside the warm-up window, `trkIsWarmup[tail]` is marked `true` so its
     * stage costs are excluded from HUD rolling averages (`rollingConversionMs`, `rollingInferenceMs`, etc.).
     */
    @JvmOverloads
    fun recordTrackerSnapshot(
        snapshot: FaceGeometrySnapshot,
        predErrPosMm: Float = 0f,
        predErrRotDeg: Float = 0f
    ) {
        val doneNs = if (snapshot.inferenceDoneNs > 0L) snapshot.inferenceDoneNs else snapshot.timestampNs
        synchronized(lock) {
            val isWarmup = isInWarmup(doneNs)
            lastSampleWasWarmup = isWarmup
            lastQueueMs = snapshot.queueMs
            lastConversionMs = snapshot.conversionMs
            lastUnpackMs = snapshot.unpackMs
            lastYuvToRgbMs = snapshot.yuvToRgbMs
            lastBitmapCopyMs = snapshot.bitmapCopyMs
            lastRotateScaleMs = snapshot.rotateScaleMs
            lastMpImageBuildMs = snapshot.mpImageBuildMs
            if (snapshot.sourceWidth > 0 && snapshot.sourceHeight > 0) {
                lastSourceWidth = snapshot.sourceWidth
                lastSourceHeight = snapshot.sourceHeight
            }
            if (snapshot.cropWidth > 0 && snapshot.cropHeight > 0) {
                lastCropWidth = snapshot.cropWidth
                lastCropHeight = snapshot.cropHeight
            }
            lastSourceRotationDeg = snapshot.rotationDegrees
            if (snapshot.conversionModeLabel.isNotEmpty()) {
                conversionModeLabel = snapshot.conversionModeLabel
            }
            lastPendingWaitMs = snapshot.pendingWaitMs
            lastInferenceMs = snapshot.inferenceMs
            lastIdleGapPendingMs = snapshot.idleGapPendingMs
            lastIdleGapNoPendingMs = snapshot.idleGapNoPendingMs
            lastPumpDelayMs = snapshot.pumpDelayMs
            if (snapshot.swapMs > 0f && snapshot.swapMs.isFinite()) {
                lastSwapMs = snapshot.swapMs
            }
            if (snapshot.projectionParams !== ProjectionParams.DEFAULT) {
                lastFovYDeg = snapshot.projectionParams.fovYDeg
                lastFovSource = snapshot.projectionParams.fovSource
                lastFocalLengthMm = snapshot.projectionParams.focalLengthMm
                lastScalerCropRegion = snapshot.projectionParams.scalerCropDescriptor
            }
            if (predErrPosMm.isFinite() && predErrPosMm >= 0f) {
                lastPredErrPosMm = predErrPosMm
            }
            if (predErrRotDeg.isFinite() && predErrRotDeg >= 0f) {
                lastPredErrRotDeg = predErrRotDeg
            }

            val tail = (trkHead + trkSize) % windowCapacity
            if (trkSize == windowCapacity) {
                trkHead = (trkHead + 1) % windowCapacity
            } else {
                trkSize++
            }
            trkTimestampsNs[tail] = doneNs
            trkConversionMs[tail] = snapshot.conversionMs
            trkUnpackMs[tail] = snapshot.unpackMs
            trkYuvToRgbMs[tail] = snapshot.yuvToRgbMs
            trkBitmapCopyMs[tail] = snapshot.bitmapCopyMs
            trkRotateScaleMs[tail] = snapshot.rotateScaleMs
            trkMpImageBuildMs[tail] = snapshot.mpImageBuildMs
            trkPendingWaitMs[tail] = snapshot.pendingWaitMs
            trkInferenceMs[tail] = snapshot.inferenceMs
            trkIdleGapPendingMs[tail] = snapshot.idleGapPendingMs
            trkIdleGapNoPendingMs[tail] = snapshot.idleGapNoPendingMs
            trkPumpDelayMs[tail] = snapshot.pumpDelayMs
            trkBlendshapesEnabled[tail] = snapshot.blendshapesEnabled
            trkPredErrPosMm[tail] = lastPredErrPosMm
            trkPredErrRotDeg[tail] = lastPredErrRotDeg
            trkIsWarmup[tail] = isWarmup

            val lm = snapshot.landmarks
            if (!isWarmup && lm != null && lm.size >= 478 * 3 && !lastIsFastHeadMotion) {
                val rawEdge = maxOf(
                    maxOf(snapshot.cropWidth, snapshot.cropHeight),
                    maxOf(snapshot.sourceWidth, snapshot.sourceHeight)
                )
                val snapLongEdge = if (rawEdge > 0) rawEdge else configuredIrisAnalysisSizePx
                recordIrisLandmarkSampleLocked(
                    landmarks = lm,
                    analysisSizePx = snapLongEdge,
                    conversionMs = snapshot.conversionMs,
                    inferenceMs = snapshot.inferenceMs,
                    trackerFps = if (trackerFps > 0f) trackerFps else 30f
                )
            }
        }
    }

    /**
     * Records a prediction-error sample (`posErrMm` in mm, `rotErrDeg` in degrees) evaluated when
     * a new tracker sample arrives.
     */
    fun recordPredictionError(posErrMm: Float, rotErrDeg: Float) {
        synchronized(lock) {
            if (posErrMm.isFinite() && posErrMm >= 0f) {
                lastPredErrPosMm = posErrMm
            }
            if (rotErrDeg.isFinite() && rotErrDeg >= 0f) {
                lastPredErrRotDeg = rotErrDeg
            }
            if (trkSize > 0) {
                val lastIdx = (trkHead + trkSize - 1) % windowCapacity
                trkPredErrPosMm[lastIdx] = lastPredErrPosMm
                trkPredErrRotDeg[lastIdx] = lastPredErrRotDeg
            }
        }
    }

    /**
     * Called from `XelvicGlSurfaceView.onDrawFrame` with zero heap allocations between ~2 Hz HUD updates.
     *
     * Records `ageAtRender` (`renderTimestampNs - faceState.poseTimestampNs`) into the rolling
     * percentile window ONLY when `faceState.lifecycle == FaceLifecycle.State.TRACKING` AND
     * `!isInWarmup(renderTimestampNs)`. Warm-up samples are still written to the CSV ring buffer
     * with `warmup = true`.
     */
    fun recordRenderFrame(
        renderTimestampNs: Long,
        faceState: FaceState,
        drawCpuMs: Float = 0f,
        predictionOn: Boolean = false
    ) {
        var shouldRefreshHud = false
        synchronized(lock) {
            val isWarmup = isInWarmup(renderTimestampNs)
            lastSampleWasWarmup = isWarmup
            lastDrawCpuMs = if (drawCpuMs.isFinite() && drawCpuMs >= 0f) drawCpuMs else 0f
            lastPredictionOn = predictionOn

            if (faceState.projectionParams !== ProjectionParams.DEFAULT) {
                lastFovYDeg = faceState.projectionParams.fovYDeg
                lastFovSource = faceState.projectionParams.fovSource
                lastFocalLengthMm = faceState.projectionParams.focalLengthMm
                lastScalerCropRegion = faceState.projectionParams.scalerCropDescriptor
            }

            // 1. Record render frame timestamp
            val rndTail = (rndHead + rndSize) % windowCapacity
            if (rndSize == windowCapacity) {
                rndHead = (rndHead + 1) % windowCapacity
            } else {
                rndSize++
            }
            rndTimestampsNs[rndTail] = renderTimestampNs

            // 2. Compute ageAtRender while lifecycle == TRACKING; exclude from rolling p50/p95 if warmup=true
            var currentAgeMs = Float.NaN
            if (faceState.lifecycle == FaceLifecycle.State.TRACKING && faceState.poseTimestampNs > 0L) {
                val rawAgeMs = (renderTimestampNs - faceState.poseTimestampNs).toFloat() / 1_000_000f
                if (rawAgeMs.isFinite() && rawAgeMs >= 0f) {
                    currentAgeMs = rawAgeMs
                    lastAgeAtRenderMs = rawAgeMs
                    if (!isWarmup) {
                        val ageTail = (ageHead + ageSize) % windowCapacity
                        if (ageSize == windowCapacity) {
                            ageHead = (ageHead + 1) % windowCapacity
                        } else {
                            ageSize++
                        }
                        ageTimestampsNs[ageTail] = renderTimestampNs
                        ageValuesMs[ageTail] = rawAgeMs
                    }
                }
                recordMaskMotionOffsetLocked(
                    renderTimestampNs = renderTimestampNs,
                    cameraSensorTimestampNs = latestCameraSensorTimestampNs,
                    faceState = faceState,
                    viewportWidth = 1080,
                    viewportHeight = 1920,
                    isWarmup = isWarmup
                )
            } else {
                lastIsFastHeadMotion = false
            }

            if (faceState.conversionMs > 0f || faceState.pendingWaitMs > 0f || faceState.inferenceMs > 0f || faceState.queueMs > 0f) {
                lastConversionMs = faceState.conversionMs
                lastPendingWaitMs = faceState.pendingWaitMs
                lastInferenceMs = faceState.inferenceMs
                lastQueueMs = faceState.queueMs
            }

            if (faceState.predErrPosMm > 0f || faceState.predErrRotDeg > 0f) {
                lastPredErrPosMm = faceState.predErrPosMm
                lastPredErrRotDeg = faceState.predErrRotDeg
                if (trkSize > 0) {
                    val lastTrkIdx = (trkHead + trkSize - 1) % windowCapacity
                    trkPredErrPosMm[lastTrkIdx] = lastPredErrPosMm
                    trkPredErrRotDeg[lastTrkIdx] = lastPredErrRotDeg
                }
            }

            // 3. Check if ~2 Hz HUD & percentile refresh is due
            if (lastHudRefreshNs == 0L || (renderTimestampNs - lastHudRefreshNs) >= hudRefreshIntervalNs) {
                shouldRefreshHud = true
                lastHudRefreshNs = renderTimestampNs
                recomputeMetricsLocked(renderTimestampNs)
            }

            // 4. Append to in-memory CSV ring buffer (zero object allocation; keeps warmup=true rows)
            val csvTail = (csvHead + csvSize) % csvRingCapacity
            if (csvSize == csvRingCapacity) {
                csvHead = (csvHead + 1) % csvRingCapacity
            } else {
                csvSize++
            }
            csvTimestampNs[csvTail] = renderTimestampNs
            csvAgeMs[csvTail] = if (currentAgeMs.isNaN()) -1f else currentAgeMs
            csvConversionMs[csvTail] = lastConversionMs
            csvPendingWaitMs[csvTail] = lastPendingWaitMs
            csvInferenceMs[csvTail] = lastInferenceMs
            csvQueueMs[csvTail] = lastQueueMs
            csvCameraFps[csvTail] = cameraFps
            csvTrackerFps[csvTail] = trackerFps
            csvRenderFps[csvTail] = renderFps
            csvPredictionOn[csvTail] = predictionOn
            csvPredErrPosMm[csvTail] = lastPredErrPosMm
            csvPredErrRotDeg[csvTail] = lastPredErrRotDeg
            csvFovYDeg[csvTail] = lastFovYDeg
            csvFovIsIntrinsics[csvTail] = (lastFovSource == ProjectionParams.FOV_SOURCE_INTRINSICS)
            csvFocalLengthMm[csvTail] = lastFocalLengthMm
            csvScalerCropRegion[csvTail] = lastScalerCropRegion
            csvDepthBits[csvTail] = lastDepthBits
            csvWarmup[csvTail] = isWarmup
            csvGovReason[csvTail] = governorReasonLabel
            csvIdleGapPendingMs[csvTail] = lastIdleGapPendingMs
            csvIdleGapNoPendingMs[csvTail] = lastIdleGapNoPendingMs
            csvPumpDelayMs[csvTail] = lastPumpDelayMs
            csvSwapMs[csvTail] = lastSwapMs
            csvConvMode[csvTail] = conversionModeLabel
            csvUnpackMs[csvTail] = lastUnpackMs
            csvYuvToRgbMs[csvTail] = lastYuvToRgbMs
            csvBitmapCopyMs[csvTail] = lastBitmapCopyMs
            csvRotateScaleMs[csvTail] = lastRotateScaleMs
            csvMpImageBuildMs[csvTail] = lastMpImageBuildMs
            csvSrcWidth[csvTail] = if (lastSourceWidth > 0) lastSourceWidth else faceState.imageWidth
            csvSrcHeight[csvTail] = if (lastSourceHeight > 0) lastSourceHeight else faceState.imageHeight
            csvSrcRotationDeg[csvTail] = lastSourceRotationDeg
        }

        if (shouldRefreshHud) {
            formatCachedHudLine()
        }
    }

    /**
     * Forces an immediate recomputation of the 5-second rolling window metrics and cached HUD line
     * at `nowNs` (useful for unit tests and on-demand inspection).
     */
    fun refreshMetrics(nowNs: Long) {
        synchronized(lock) {
            lastHudRefreshNs = nowNs
            recomputeMetricsLocked(nowNs)
        }
        formatCachedHudLine()
    }

    private fun recomputeMetricsLocked(nowNs: Long) {
        val cutoffNs = nowNs - windowDurationNs

        // Evict expired age samples (> 5s old)
        while (ageSize > 0 && ageTimestampsNs[ageHead] < cutoffNs) {
            ageHead = (ageHead + 1) % windowCapacity
            ageSize--
        }

        // Evict expired camera timestamps (> 5s old)
        while (camSize > 0 && camTimestampsNs[camHead] < cutoffNs) {
            camHead = (camHead + 1) % windowCapacity
            camSize--
        }

        // Evict expired tracker timestamps (> 5s old)
        while (trkSize > 0 && trkTimestampsNs[trkHead] < cutoffNs) {
            trkHead = (trkHead + 1) % windowCapacity
            trkSize--
        }

        // Evict expired render timestamps (> 5s old)
        while (rndSize > 0 && rndTimestampsNs[rndHead] < cutoffNs) {
            rndHead = (rndHead + 1) % windowCapacity
            rndSize--
        }

        // Evict expired mask-offset samples (> 5s old)
        while (maskSize > 0 && maskTimestampsNs[maskHead] < cutoffNs) {
            maskHead = (maskHead + 1) % windowCapacity
            maskSize--
        }

        // Sample ART GC count at ~2 Hz and compute 5s GC-count delta
        val currentGc = gcCountProvider.invoke()
        while (gcSize > 0 && gcTimestampsNs[gcHead] < cutoffNs) {
            gcHead = (gcHead + 1) % GC_WINDOW_CAPACITY
            gcSize--
        }
        val gcTail = (gcHead + gcSize) % GC_WINDOW_CAPACITY
        if (gcSize == GC_WINDOW_CAPACITY) {
            gcHead = (gcHead + 1) % GC_WINDOW_CAPACITY
        } else {
            gcSize++
        }
        gcTimestampsNs[gcTail] = nowNs
        gcCounts[gcTail] = currentGc
        val oldestGc = gcCounts[gcHead]
        gcCountDelta5s = (currentGc - oldestGc).coerceAtLeast(0L).toInt()

        // Compute ageAtRender p50 / p95 using pre-allocated sortScratch (warmup samples already excluded)
        if (ageSize == 0) {
            ageP50Ms = 0f
            ageP95Ms = 0f
        } else {
            for (i in 0 until ageSize) {
                val idx = (ageHead + i) % windowCapacity
                sortScratch[i] = ageValuesMs[idx]
            }
            Arrays.sort(sortScratch, 0, ageSize)
            val p50Idx = ((ageSize - 1) * 0.50f).toInt().coerceIn(0, ageSize - 1)
            val p95Idx = ((ageSize - 1) * 0.95f).toInt().coerceIn(0, ageSize - 1)
            ageP50Ms = sortScratch[p50Idx]
            ageP95Ms = sortScratch[p95Idx]
        }

        // Compute FPS over the active span in the 5s window
        cameraFps = computeRateHzLocked(camTimestampsNs, camHead, camSize, nowNs)
        trackerFps = computeRateHzLocked(trkTimestampsNs, trkHead, trkSize, nowNs)
        renderFps = computeRateHzLocked(rndTimestampsNs, rndHead, rndSize, nowNs)

        // Compute rolling mean conversionMs, per-stage conversion breakdown, pendingWaitMs,
        // inferenceMs (overall + blend ON/OFF), idleGapPendingMs, idleGapNoPendingMs,
        // pumpDelayMs mean/max, and prediction error over non-warmup samples in the 5s window (~2 Hz)
        var nonWarmupCount = 0
        var sumConv = 0f
        var sumUnpack = 0f
        var sumYuv = 0f
        var sumCopy = 0f
        var sumRotScale = 0f
        var sumMpBuild = 0f
        var sumWait = 0f
        var sumInf = 0f
        var sumIdlePend = 0f
        var sumIdleNoPend = 0f
        var sumPumpDelay = 0f
        var maxPumpDelay = 0f
        var blendOnCount = 0
        var sumInfBlendOn = 0f
        var blendOffCount = 0
        var sumInfBlendOff = 0f
        var sumPosErr = 0f
        var sumRotErr = 0f
        for (i in 0 until trkSize) {
            val idx = (trkHead + i) % windowCapacity
            if (!trkIsWarmup[idx]) {
                nonWarmupCount++
                val inf = trkInferenceMs[idx]
                val pump = trkPumpDelayMs[idx]
                sumConv += trkConversionMs[idx]
                sumUnpack += trkUnpackMs[idx]
                sumYuv += trkYuvToRgbMs[idx]
                sumCopy += trkBitmapCopyMs[idx]
                sumRotScale += trkRotateScaleMs[idx]
                sumMpBuild += trkMpImageBuildMs[idx]
                sumWait += trkPendingWaitMs[idx]
                sumInf += inf
                sumIdlePend += trkIdleGapPendingMs[idx]
                sumIdleNoPend += trkIdleGapNoPendingMs[idx]
                sumPumpDelay += pump
                if (pump > maxPumpDelay) {
                    maxPumpDelay = pump
                }
                if (trkBlendshapesEnabled[idx]) {
                    blendOnCount++
                    sumInfBlendOn += inf
                } else {
                    blendOffCount++
                    sumInfBlendOff += inf
                }
                sumPosErr += trkPredErrPosMm[idx]
                sumRotErr += trkPredErrRotDeg[idx]
            }
        }
        if (nonWarmupCount == 0) {
            if (isInWarmup(nowNs)) {
                rollingConversionMs = 0f
                rollingUnpackMs = 0f
                rollingYuvToRgbMs = 0f
                rollingBitmapCopyMs = 0f
                rollingRotateScaleMs = 0f
                rollingMpImageBuildMs = 0f
                rollingPendingWaitMs = 0f
                rollingInferenceMs = 0f
                rollingIdleGapPendingMs = 0f
                rollingIdleGapNoPendingMs = 0f
                rollingPumpDelayMeanMs = 0f
                rollingPumpDelayMaxMs = 0f
                rollingPredErrPosMm = 0f
                rollingPredErrRotDeg = 0f
            } else {
                rollingConversionMs = lastConversionMs
                rollingUnpackMs = lastUnpackMs
                rollingYuvToRgbMs = lastYuvToRgbMs
                rollingBitmapCopyMs = lastBitmapCopyMs
                rollingRotateScaleMs = lastRotateScaleMs
                rollingMpImageBuildMs = lastMpImageBuildMs
                rollingPendingWaitMs = lastPendingWaitMs
                rollingInferenceMs = lastInferenceMs
                rollingIdleGapPendingMs = lastIdleGapPendingMs
                rollingIdleGapNoPendingMs = lastIdleGapNoPendingMs
                rollingPumpDelayMeanMs = lastPumpDelayMs
                rollingPumpDelayMaxMs = lastPumpDelayMs
                rollingPredErrPosMm = lastPredErrPosMm
                rollingPredErrRotDeg = lastPredErrRotDeg
            }
        } else {
            val inv = 1f / nonWarmupCount.toFloat()
            rollingConversionMs = sumConv * inv
            rollingUnpackMs = sumUnpack * inv
            rollingYuvToRgbMs = sumYuv * inv
            rollingBitmapCopyMs = sumCopy * inv
            rollingRotateScaleMs = sumRotScale * inv
            rollingMpImageBuildMs = sumMpBuild * inv
            rollingPendingWaitMs = sumWait * inv
            rollingInferenceMs = sumInf * inv
            rollingIdleGapPendingMs = sumIdlePend * inv
            rollingIdleGapNoPendingMs = sumIdleNoPend * inv
            rollingPumpDelayMeanMs = sumPumpDelay * inv
            rollingPumpDelayMaxMs = maxPumpDelay
            rollingPredErrPosMm = sumPosErr * inv
            rollingPredErrRotDeg = sumRotErr * inv
        }
        if (blendOnCount > 0) {
            rollingInferenceBlendOnMs = sumInfBlendOn / blendOnCount.toFloat()
        }
        if (blendOffCount > 0) {
            rollingInferenceBlendOffMs = sumInfBlendOff / blendOffCount.toFloat()
        }

        // Compute Chunk R1 mask-offset rolling metrics (all tracking samples + fast-head-motion subset)
        if (maskSize == 0) {
            rollingMaskTimeDeltaMs = lastMaskTimeDeltaMs
            rollingFastMotionMaskDeltaMs = 0f
            rollingFastMotionMaskOffsetPx = 0f
            fastMotionMaskOffsetP95Px = 0f
            maxFastMotionMaskOffsetPx = 0f
            fastMotionSampleCount = 0
        } else {
            var sumAllDelta = 0f
            var sumFastDelta = 0f
            var sumFastOffset = 0f
            var maxFastOffset = 0f
            var fastCount = 0
            for (i in 0 until maskSize) {
                val idx = (maskHead + i) % windowCapacity
                val dMs = maskAbsDeltaMs[idx]
                val offPx = maskOffsetPxRing[idx]
                sumAllDelta += dMs
                if (maskIsFastMotion[idx]) {
                    sortScratch[fastCount] = offPx
                    fastCount++
                    sumFastDelta += dMs
                    sumFastOffset += offPx
                    if (offPx > maxFastOffset) {
                        maxFastOffset = offPx
                    }
                }
            }
            rollingMaskTimeDeltaMs = sumAllDelta / maskSize.toFloat()
            fastMotionSampleCount = fastCount
            if (fastCount > 0) {
                Arrays.sort(sortScratch, 0, fastCount)
                val p95Idx = ((fastCount - 1) * 0.95f).toInt().coerceIn(0, fastCount - 1)
                rollingFastMotionMaskDeltaMs = sumFastDelta / fastCount.toFloat()
                rollingFastMotionMaskOffsetPx = sumFastOffset / fastCount.toFloat()
                fastMotionMaskOffsetP95Px = sortScratch[p95Idx]
                maxFastMotionMaskOffsetPx = maxFastOffset
            } else {
                rollingFastMotionMaskDeltaMs = 0f
                rollingFastMotionMaskOffsetPx = 0f
                fastMotionMaskOffsetP95Px = 0f
                maxFastMotionMaskOffsetPx = 0f
            }
        }
    }

    private fun computeRateHzLocked(
        timestampsNs: LongArray,
        head: Int,
        size: Int,
        nowNs: Long
    ): Float {
        if (size <= 1) return 0f
        val firstNs = timestampsNs[head]
        val lastNs = timestampsNs[(head + size - 1) % windowCapacity]
        // Use the span between first and last sample, or active window duration if wider
        val spanNs = maxOf(lastNs - firstNs, nowNs - firstNs).coerceAtMost(windowDurationNs)
        if (spanNs <= 0L) return 0f
        return ((size - 1).toFloat() * 1_000_000_000f) / spanNs.toFloat()
    }

    private fun formatCachedHudLine() {
        val p50 = ageP50Ms
        val p95 = ageP95Ms
        val trk = trackerFps
        val cam = cameraFps
        val rnd = renderFps
        val waitMs = rollingPendingWaitMs
        val inf = rollingInferenceMs
        val gcDelta = gcCountDelta5s
        val status = trackerStatusLabel
        val govReason = governorReasonLabel
        val agePart = if (ageSampleCount() > 0) {
            String.format(Locale.US, "%.1f/%.1fms", p50, p95)
        } else {
            "--/--ms"
        }
        val baseLine = String.format(
            Locale.US,
            "age p50/p95: %s | trk: %.1ffps | cam: %.1ffps | rnd: %.1ffps | wait: %.1fms | inf: %.1fms | gc: +%d",
            agePart,
            trk,
            cam,
            rnd,
            waitMs,
            inf,
            gcDelta
        )
        val withGov = if (govReason.isNotEmpty() && govReason != "none") {
            "$baseLine | gov: $govReason"
        } else {
            baseLine
        }
        cachedHudLine = if (status.isNotEmpty()) "$withGov | $status" else withGov
        cachedIrisJitterLine = formatIrisJitterSummaryLine()
    }

    fun ageSampleCount(): Int = synchronized(lock) { ageSize }

    fun nonWarmupTrackerSampleCount(): Int = synchronized(lock) {
        var count = 0
        for (i in 0 until trkSize) {
            if (!trkIsWarmup[(trkHead + i) % windowCapacity]) count++
        }
        count
    }

    fun csvRowCount(): Int = synchronized(lock) { csvSize }

    /**
     * Generates an on-demand CSV dump of the in-memory ring buffer.
     * Columns (32):
     * `timestamp,ageMs,conversionMs,pendingWaitMs,inferenceMs,queueMs,cameraFps,trackerFps,renderFps,predictionOn,predErrPosMm,predErrRotDeg,fovYDeg,fovSource,focalLengthMm,scalerCropRegion,depthBits,warmup,govReason,idleGapPendingMs,idleGapNoPendingMs,pumpDelayMs,swapMs,convMode,unpackMs,yuvToRgbMs,bitmapCopyMs,rotateScaleMs,mpImageBuildMs,srcWidth,srcHeight,srcRotationDeg`
     */
    fun dumpCsv(): String {
        val sb = StringBuilder()
        dumpCsvTo(sb)
        return sb.toString()
    }

    fun dumpCsvTo(out: Appendable) {
        synchronized(lock) {
            out.append(CSV_HEADER).append('\n')
            for (i in 0 until csvSize) {
                val idx = (csvHead + i) % csvRingCapacity
                val fovSrc = if (csvFovIsIntrinsics[idx]) {
                    ProjectionParams.FOV_SOURCE_INTRINSICS
                } else {
                    ProjectionParams.FOV_SOURCE_FALLBACK
                }
                val cropDesc = csvScalerCropRegion[idx].replace(',', ';')
                val govReason = csvGovReason[idx].replace(',', ';')
                val convModeStr = csvConvMode[idx].replace(',', ';')
                val line = String.format(
                    Locale.US,
                    "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%b,%.3f,%.3f,%.2f,%s,%.2f,%s,%d,%b,%s,%.3f,%.3f,%.3f,%.3f,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d\n",
                    csvTimestampNs[idx],
                    csvAgeMs[idx],
                    csvConversionMs[idx],
                    csvPendingWaitMs[idx],
                    csvInferenceMs[idx],
                    csvQueueMs[idx],
                    csvCameraFps[idx],
                    csvTrackerFps[idx],
                    csvRenderFps[idx],
                    csvPredictionOn[idx],
                    csvPredErrPosMm[idx],
                    csvPredErrRotDeg[idx],
                    csvFovYDeg[idx],
                    fovSrc,
                    csvFocalLengthMm[idx],
                    cropDesc,
                    csvDepthBits[idx],
                    csvWarmup[idx],
                    govReason,
                    csvIdleGapPendingMs[idx],
                    csvIdleGapNoPendingMs[idx],
                    csvPumpDelayMs[idx],
                    csvSwapMs[idx],
                    convModeStr,
                    csvUnpackMs[idx],
                    csvYuvToRgbMs[idx],
                    csvBitmapCopyMs[idx],
                    csvRotateScaleMs[idx],
                    csvMpImageBuildMs[idx],
                    csvSrcWidth[idx],
                    csvSrcHeight[idx],
                    csvSrcRotationDeg[idx]
                )
                out.append(line)
            }
        }
    }

    fun reset(preserveWarmup: Boolean = false, preserveBlendInferenceHistory: Boolean = false) {
        synchronized(lock) {
            ageHead = 0
            ageSize = 0
            camHead = 0
            camSize = 0
            trkHead = 0
            trkSize = 0
            rndHead = 0
            rndSize = 0
            gcHead = 0
            gcSize = 0
            maskHead = 0
            maskSize = 0
            csvHead = 0
            csvSize = 0
            if (!preserveWarmup) {
                warmupUntilNs = 0L
            }
            lastConversionMs = 0f
            lastUnpackMs = 0f
            lastYuvToRgbMs = 0f
            lastBitmapCopyMs = 0f
            lastRotateScaleMs = 0f
            lastMpImageBuildMs = 0f
            lastPendingWaitMs = 0f
            lastInferenceMs = 0f
            lastQueueMs = 0f
            lastIdleGapPendingMs = 0f
            lastIdleGapNoPendingMs = 0f
            lastPumpDelayMs = 0f
            lastAgeAtRenderMs = 0f
            lastDrawCpuMs = 0f
            lastPredErrPosMm = 0f
            lastPredErrRotDeg = 0f
            lastPredictionOn = true
            lastSampleWasWarmup = false
            lastMaskTimeDeltaMs = 0f
            lastMaskSignedDeltaMs = 0f
            lastMaskOffsetPx = 0f
            lastHeadSpeedMps = 0f
            lastHeadAngularSpeedDegPerSec = 0f
            lastIsFastHeadMotion = false
            rollingMaskTimeDeltaMs = 0f
            rollingFastMotionMaskDeltaMs = 0f
            rollingFastMotionMaskOffsetPx = 0f
            fastMotionMaskOffsetP95Px = 0f
            maxFastMotionMaskOffsetPx = 0f
            fastMotionSampleCount = 0
            ageP50Ms = 0f
            ageP95Ms = 0f
            cameraFps = 0f
            trackerFps = 0f
            renderFps = 0f
            rollingConversionMs = 0f
            rollingUnpackMs = 0f
            rollingYuvToRgbMs = 0f
            rollingBitmapCopyMs = 0f
            rollingRotateScaleMs = 0f
            rollingMpImageBuildMs = 0f
            rollingPendingWaitMs = 0f
            rollingInferenceMs = 0f
            if (!preserveBlendInferenceHistory) {
                rollingInferenceBlendOnMs = 0f
                rollingInferenceBlendOffMs = 0f
            }
            rollingIdleGapPendingMs = 0f
            rollingIdleGapNoPendingMs = 0f
            rollingPumpDelayMeanMs = 0f
            rollingPumpDelayMaxMs = 0f
            rollingPredErrPosMm = 0f
            rollingPredErrRotDeg = 0f
            gcCountDelta5s = 0
            for (b in 0 until 3) {
                irisBucketHead[b] = 0
                irisBucketSize[b] = 0
            }
            lastHudRefreshNs = 0L
        }
        formatCachedHudLine()
    }

    /**
     * Cycles [configuredIrisAnalysisSizePx] across `320 -> 480 -> 640 -> 320` and refreshes [cachedIrisJitterLine].
     */
    fun cycleIrisAnalysisSizePx(): Int {
        val next = when (configuredIrisAnalysisSizePx) {
            320 -> 480
            480 -> 640
            else -> 320
        }
        configuredIrisAnalysisSizePx = next
        cachedIrisJitterLine = formatIrisJitterSummaryLine()
        return next
    }

    private fun irisBucketIndexForSize(sizePx: Int): Int = when {
        sizePx <= 380 -> 0 // 320 bucket
        sizePx >= 560 -> 2 // 640 bucket
        else -> 1          // 480 bucket
    }

    private fun irisBucketNominalSize(bucket: Int): Int = when (bucket) {
        0 -> 320
        2 -> 640
        else -> 480
    }

    /**
     * Records a stationary iris-landmark sample (`#468` right iris center and `#473` left iris center)
     * for the corresponding `320 / 480 / 640` analysis long-edge bucket alongside `conversionMs`,
     * `inferenceMs`, and `trackerFps` (Flow 4 — Chunk Primitives-1 EC1).
     */
    @JvmOverloads
    fun recordIrisLandmarkSample(
        landmarks: FloatArray?,
        analysisSizePx: Int,
        conversionMs: Float = this.lastConversionMs,
        inferenceMs: Float = this.lastInferenceMs,
        trackerFps: Float = if (this.trackerFps > 0f) this.trackerFps else 30f,
        isStationary: Boolean = !this.lastIsFastHeadMotion
    ) {
        if (!isStationary || landmarks == null || landmarks.size < 478 * 3) return
        synchronized(lock) {
            recordIrisLandmarkSampleLocked(
                landmarks = landmarks,
                analysisSizePx = analysisSizePx,
                conversionMs = conversionMs,
                inferenceMs = inferenceMs,
                trackerFps = trackerFps
            )
        }
        cachedIrisJitterLine = formatIrisJitterSummaryLine()
    }

    private fun recordIrisLandmarkSampleLocked(
        landmarks: FloatArray,
        analysisSizePx: Int,
        conversionMs: Float,
        inferenceMs: Float,
        trackerFps: Float
    ) {
        val bucket = irisBucketIndexForSize(analysisSizePx)
        val nominalPx = irisBucketNominalSize(bucket).toFloat()
        val rx = landmarks[468 * 3] * nominalPx
        val ry = landmarks[468 * 3 + 1] * nominalPx
        val lx = landmarks[473 * 3] * nominalPx
        val ly = landmarks[473 * 3 + 1] * nominalPx
        if (!rx.isFinite() || !ry.isFinite() || !lx.isFinite() || !ly.isFinite()) return

        val head = irisBucketHead[bucket]
        val size = irisBucketSize[bucket]
        val slot = (head + size) % irisBucketCapacity
        if (size == irisBucketCapacity) {
            irisBucketHead[bucket] = (head + 1) % irisBucketCapacity
        } else {
            irisBucketSize[bucket] = size + 1
        }
        val flatIdx = bucket * irisBucketCapacity + slot
        irisRightX[flatIdx] = rx
        irisRightY[flatIdx] = ry
        irisLeftX[flatIdx] = lx
        irisLeftY[flatIdx] = ly
        irisConvMs[flatIdx] = if (conversionMs.isFinite() && conversionMs >= 0f) conversionMs else 0f
        irisInfMs[flatIdx] = if (inferenceMs.isFinite() && inferenceMs >= 0f) inferenceMs else 0f
        irisTrkFps[flatIdx] = if (trackerFps.isFinite() && trackerFps >= 0f) trackerFps else 0f
    }

    /**
     * Computes stationary iris-landmark jitter (`stationaryJitterStdDevPx`) and mean stage costs
     * (`conversionMs`, `inferenceMs`, `trackerFps`) for the requested `320`, `480`, or `640` bucket.
     */
    fun getIrisSizeMetrics(sizePx: Int): IrisSizeMetrics {
        synchronized(lock) {
            val bucket = irisBucketIndexForSize(sizePx)
            val nominalSize = irisBucketNominalSize(bucket)
            val count = irisBucketSize[bucket]
            if (count <= 0) {
                return IrisSizeMetrics(
                    analysisSizePx = nominalSize,
                    stationaryJitterStdDevPx = 0f,
                    conversionMs = 0f,
                    inferenceMs = 0f,
                    trackerFps = 0f,
                    sampleCount = 0
                )
            }
            val head = irisBucketHead[bucket]
            val baseOffset = bucket * irisBucketCapacity
            var sumRx = 0f
            var sumRy = 0f
            var sumLx = 0f
            var sumLy = 0f
            var sumConv = 0f
            var sumInf = 0f
            var sumFps = 0f
            for (i in 0 until count) {
                val idx = baseOffset + ((head + i) % irisBucketCapacity)
                sumRx += irisRightX[idx]
                sumRy += irisRightY[idx]
                sumLx += irisLeftX[idx]
                sumLy += irisLeftY[idx]
                sumConv += irisConvMs[idx]
                sumInf += irisInfMs[idx]
                sumFps += irisTrkFps[idx]
            }
            val inv = 1f / count.toFloat()
            val meanRx = sumRx * inv
            val meanRy = sumRy * inv
            val meanLx = sumLx * inv
            val meanLy = sumLy * inv

            var varRight = 0f
            var varLeft = 0f
            if (count >= 2) {
                for (i in 0 until count) {
                    val idx = baseOffset + ((head + i) % irisBucketCapacity)
                    val drx = irisRightX[idx] - meanRx
                    val dry = irisRightY[idx] - meanRy
                    val dlx = irisLeftX[idx] - meanLx
                    val dly = irisLeftY[idx] - meanLy
                    varRight += drx * drx + dry * dry
                    varLeft += dlx * dlx + dly * dly
                }
                val denom = (count - 1).toFloat()
                varRight /= denom
                varLeft /= denom
            }
            val stdRight = kotlin.math.sqrt(varRight.coerceAtLeast(0f))
            val stdLeft = kotlin.math.sqrt(varLeft.coerceAtLeast(0f))
            val stdDevPx = 0.5f * (stdRight + stdLeft)

            return IrisSizeMetrics(
                analysisSizePx = nominalSize,
                stationaryJitterStdDevPx = stdDevPx,
                conversionMs = sumConv * inv,
                inferenceMs = sumInf * inv,
                trackerFps = sumFps * inv,
                sampleCount = count
            )
        }
    }

    fun formatIrisJitterSummaryLine(): String {
        val m320 = getIrisSizeMetrics(320)
        val m480 = getIrisSizeMetrics(480)
        val m640 = getIrisSizeMetrics(640)
        fun fmtBucket(m: IrisSizeMetrics): String {
            return if (m.sampleCount >= 2) {
                String.format(
                    Locale.US,
                    "%d:%.2fpx(c%.1f/i%.1f/%.0ffps)",
                    m.analysisSizePx,
                    m.stationaryJitterStdDevPx,
                    m.conversionMs,
                    m.inferenceMs,
                    m.trackerFps
                )
            } else {
                "${m.analysisSizePx}:--px"
            }
        }
        return "IrisJitter[${configuredIrisAnalysisSizePx}px]: ${fmtBucket(m320)} • ${fmtBucket(m480)} • ${fmtBucket(m640)}"
    }

    /**
     * Explicitly records a mask-offset sample comparing `faceState.poseTimestampNs` (landmark time)
     * against `cameraSensorTimestampNs` (camera frame sensor time from `FrameRing`) during head motion.
     */
    @JvmOverloads
    fun recordMaskMotionOffset(
        renderTimestampNs: Long,
        cameraSensorTimestampNs: Long,
        faceState: FaceState,
        viewportWidth: Int = 1080,
        viewportHeight: Int = 1920
    ) {
        synchronized(lock) {
            latestCameraSensorTimestampNs = cameraSensorTimestampNs
            val isWarmup = isInWarmup(renderTimestampNs)
            recordMaskMotionOffsetLocked(
                renderTimestampNs = renderTimestampNs,
                cameraSensorTimestampNs = cameraSensorTimestampNs,
                faceState = faceState,
                viewportWidth = viewportWidth,
                viewportHeight = viewportHeight,
                isWarmup = isWarmup
            )
            recomputeMetricsLocked(renderTimestampNs)
        }
    }

    private fun recordMaskMotionOffsetLocked(
        renderTimestampNs: Long,
        cameraSensorTimestampNs: Long,
        faceState: FaceState,
        viewportWidth: Int,
        viewportHeight: Int,
        isWarmup: Boolean
    ) {
        computeMaskMotionOffset(
            cameraFrameSensorTimestampNs = cameraSensorTimestampNs,
            renderTimestampNs = renderTimestampNs,
            faceState = faceState,
            viewportWidth = viewportWidth,
            viewportHeight = viewportHeight,
            outMetrics = maskMetricScratch
        )
        val absDeltaMs = maskMetricScratch[0]
        val signedDeltaMs = maskMetricScratch[1]
        val offsetPx = maskMetricScratch[2]
        val speedMps = maskMetricScratch[3]
        val angSpeedDegS = maskMetricScratch[4]
        val isFast = speedMps >= FAST_MOTION_LINEAR_SPEED_MPS || angSpeedDegS >= FAST_MOTION_ANGULAR_SPEED_DEG_S

        lastMaskTimeDeltaMs = absDeltaMs
        lastMaskSignedDeltaMs = signedDeltaMs
        lastMaskOffsetPx = offsetPx
        lastHeadSpeedMps = speedMps
        lastHeadAngularSpeedDegPerSec = angSpeedDegS
        lastIsFastHeadMotion = isFast

        if (!isWarmup) {
            val tail = (maskHead + maskSize) % windowCapacity
            if (maskSize == windowCapacity) {
                maskHead = (maskHead + 1) % windowCapacity
            } else {
                maskSize++
            }
            maskTimestampsNs[tail] = renderTimestampNs
            maskAbsDeltaMs[tail] = absDeltaMs
            maskOffsetPxRing[tail] = offsetPx
            maskIsFastMotion[tail] = isFast
        }
    }
}
