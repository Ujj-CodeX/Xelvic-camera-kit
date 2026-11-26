package com.example.engine.governor

import android.content.Context
import android.util.Log
import java.util.Locale

/**
 * PerformanceGovernor (README Section 13 & 21 Step 6):
 *
 * Uses specified telemetry inputs:
 * - Frame-time p50 / p95 (used ONLY when rendering is continuous / vsync-driven)
 * - Tracker rate (Hz)
 * - Result age at render (ms) & tracker stage cost (`conversionMs + inferenceMs`)
 * - Dropped frames
 * - Thermal status
 * - Memory pressure
 *
 * Rules enforced:
 * 1. **No False Visual-Quality or Tracker Demotions**:
 *    - Visual-quality tier changes (Steps 2–4: `Tier A -> B -> C`, texture cap, occluder segments)
 *      are NEVER triggered by `resultAgeMs` or `conversionMs + inferenceMs` alone; they are demoted
 *      ONLY for render `frameTimeP95Ms > 33 ms` or `thermalStatus >= MODERATE` (or memory pressure).
 *    - Tracker-cost throttling (`enableTrackerCostThrottle`, default `false`) is disabled by default
 *      so high `resultAgeMs` or high `conversionMs + inferenceMs` alone never throttles the tracker
 *      below what it can sustain when render and thermal are healthy. When `enableTrackerCostThrottle`
 *      is explicitly enabled, high `resultAgeMs` (`> 120 ms`) with `conversionMs + inferenceMs > 33.3 ms`
 *      may only reduce Step 1 (tracker cadence down to `15 Hz`, analysis size down to `minAnalysisLongEdgePx`),
 *      never Steps 2–5.
 * 2. **Warm-Up Window (5 s after GPU delegate switch, 3 s after filter switch / rebind / resume)**:
 *    - During warm-up, `resultAgeMs` and `conversionMs / inferenceMs` are not fed into governor decisions.
 *    - Single isolated render frames `> 100 ms` (shader-compile hitches) during warm-up are ignored
 *      for `frameWindow` (`frameTimeP95Ms`), without hiding consecutive `> 100 ms` frames or sustained slowness.
 * 3. **Configurable Analysis Size Floor (`minAnalysisLongEdgePx`, default `320`)**:
 *    - `targetAnalysisSizePx` starts at `320` on startup (respecting `minAnalysisLongEdgePx`) and
 *      never drops below `minAnalysisLongEdgePx`.
 * 4. **Preview Quality (Step 5)**:
 *    - `isPreviewQualityReduced` never becomes `true` automatically unless explicitly enabled via
 *      `allowPreviewQualityReduction` or `setDebugPreviewQualityReduced(true)`.
 * 5. **Demote / Promote Reason Telemetry**:
 *    - Every demote or promote records `lastTransitionReason` (`signal name + value`) for the HUD
 *      cached line and CSV dump.
 */
class PerformanceGovernor @JvmOverloads constructor(
    context: Context? = null,
    val isLowEndDevice: Boolean = ThermalMonitor.detectLowEndDevice(context),
    var allowPreviewQualityReduction: Boolean = false,
    val thermalMonitor: ThermalMonitor = ThermalMonitor(context),
    minAnalysisLongEdgePx: Int = ANALYSIS_SIZE_HIGH_PX,
    var enableTrackerCostThrottle: Boolean = false
) : ThermalMonitor.Listener {

    internal val appContext: Context? = context?.applicationContext ?: context

    enum class Tier { A, B, C }

    enum class DegradationStep {
        NONE,
        TRACKER_CADENCE_AND_ANALYSIS_SIZE,
        PACKAGE_TIER_AND_BEHAVIORS,
        LOD_AND_TEXTURE_SIZE,
        OCCLUDER_SEGMENTS,
        PREVIEW_QUALITY
    }

    companion object {
        private const val TAG = "PerformanceGovernor"

        // Normative starting hysteresis values (README Section 13 & Chunk E1)
        const val DWELL_WINDOW_MS = 3000L
        const val DEMOTE_DELAY_MS = 1500L
        const val PROMOTE_HEADROOM_MS = 3000L
        const val MAX_P95_MS = 40.0f
        const val HEADROOM_P95_MS = 34.0f

        // Promotion backoff schedule after repeated frame-p95 demotions: 3s, 6s, 12s, 30s; reset after 60s on one tier
        val PROMOTE_BACKOFF_SCHEDULE_MS = longArrayOf(3000L, 6000L, 12000L, 30000L)
        const val BACKOFF_RESET_STABLE_MS = 60000L

        // Warm-up durations and single-frame shader-compile hitch threshold
        const val WARMUP_DEFAULT_DURATION_MS = 3000L
        const val WARMUP_GPU_DELEGATE_DURATION_MS = 5000L
        const val WARMUP_SINGLE_HITCH_THRESHOLD_MS = 100.0f

        // Tracker cadence bounds (README Section 8 & 13)
        const val MAX_TRACKER_CADENCE_HZ = 30
        const val MID_TRACKER_CADENCE_HZ = 24
        const val LOW_MID_TRACKER_CADENCE_HZ = 20
        const val MIN_TRACKER_CADENCE_HZ = 15

        // Analysis long-edge sizes (README Section 8: start near 320px on the long edge;
        // Chunk Primitives-1 EC1: 480px default / 320, 480, 640 configurable when tracker.iris@1 is active)
        const val ANALYSIS_SIZE_IRIS_MAX_PX = 640
        const val ANALYSIS_SIZE_IRIS_DEFAULT_PX = 480
        const val ANALYSIS_SIZE_HIGH_PX = 320
        const val ANALYSIS_SIZE_MED_PX = 256
        const val ANALYSIS_SIZE_LOW_PX = 224
        val SUPPORTED_IRIS_ANALYSIS_SIZES = intArrayOf(320, 480, 640)

        // Texture max sizes per tier (README Section 5.13)
        const val TEXTURE_MAX_HIGH = 1024
        const val TEXTURE_MAX_MED = 512
        const val TEXTURE_MAX_LOW = 256

        // Result age and tracker cost thresholds (used only when enableTrackerCostThrottle == true)
        const val MAX_ACCEPTABLE_RESULT_AGE_MS = 120.0f
        const val MAX_ACCEPTABLE_TRACKER_COST_MS = 33.3f

        private const val WINDOW_SIZE = 60
    }

    // Configurable timing & p95 thresholds (Chunk E1)
    var dwellWindowMs: Long = DWELL_WINDOW_MS
    var demoteDelayMs: Long = DEMOTE_DELAY_MS
    var promoteHeadroomMs: Long = PROMOTE_HEADROOM_MS
    var maxFrameP95Ms: Float = MAX_P95_MS
    var headroomFrameP95Ms: Float = HEADROOM_P95_MS
    var backoffResetStableMs: Long = BACKOFF_RESET_STABLE_MS

    /**
     * Configurable minimum analysis long-edge resolution floor (default `320`).
     * `targetAnalysisSizePx` never drops below this floor.
     */
    var minAnalysisLongEdgePx: Int = minAnalysisLongEdgePx.coerceIn(ANALYSIS_SIZE_LOW_PX, 640)
        set(value) {
            field = value.coerceIn(ANALYSIS_SIZE_LOW_PX, 640)
            if (targetAnalysisSizePx < field) {
                targetAnalysisSizePx = field
            }
        }

    /**
     * Configurable analysis long-edge resolution (`320`, `480`, or `640`; default `480`) used
     * ONLY while a filter requiring `tracker.iris@1` (`recolor@1` / `eyeColor@1`) is active.
     */
    @Volatile
    var irisAnalysisSizePx: Int = ANALYSIS_SIZE_IRIS_DEFAULT_PX
        private set

    /**
     * True while the currently active filter requires `tracker.iris@1` (`effect.requiresIris == true`).
     * When `false`, `targetAnalysisSizePx` immediately returns to the standard `320` ceiling.
     */
    @Volatile
    var isIrisTrackingRequired: Boolean = false
        private set

    // Current active tier: low-end starts at Tier B (never Tier C); high-end starts at Tier A
    var currentTier: Tier = if (isLowEndDevice) Tier.B else Tier.A
        private set

    // Step 1 controls: Tracker cadence (floor 15 Hz, ceiling 30 Hz) & analysis size (floor minAnalysisLongEdgePx)
    var targetTrackerCadenceHz: Int = MAX_TRACKER_CADENCE_HZ
        private set

    var targetAnalysisSizePx: Int = (if (isLowEndDevice) ANALYSIS_SIZE_LOW_PX else ANALYSIS_SIZE_HIGH_PX)
        .coerceAtLeast(this.minAnalysisLongEdgePx)
        private set

    // Step 3 controls: LOD & texture size cap (Tier B = 512, Tier A = 1024)
    var maxTextureSize: Int = if (isLowEndDevice) TEXTURE_MAX_MED else TEXTURE_MAX_HIGH
        private set

    // Step 4 controls: Occluder ellipsoid segment counts [lat, lon]
    var occluderLatSegments: Int = 16
        private set

    var occluderLonSegments: Int = 24
        private set

    // Step 5 control: Preview quality reduction (ONLY with explicit debug/config approval)
    var isPreviewQualityReduced: Boolean = false
        private set

    // Last executed degradation step for telemetry/testing
    var lastDegradationStep: DegradationStep = DegradationStep.NONE
        private set

    // Reason string of the most recent demote/promote transition (signal name + value)
    @Volatile
    var lastTransitionReason: String = "none"
        private set

    var onTransitionLogged: ((String) -> Unit)? = null

    // Optional package minTier constraint ("A", "B", or "C")
    var packageMinTier: String? = null

    // Measured telemetry inputs
    var frameTimeP50Ms: Float = 16.6f
        private set

    var frameTimeP95Ms: Float = 16.6f
        private set

    // Old System.nanoTime() CPU execution duration around onDrawFrame kept ONLY as a debug stat
    var lastDrawCpuDurationMs: Float = 0.0f
        private set

    @Volatile
    var isContinuousRenderMode: Boolean = true

    var trackerRateHz: Float = 30.0f
        private set

    var resultAgeMs: Float = 0.0f
        private set

    var trackerConversionMs: Float = 0.0f
        private set

    var trackerInferenceMs: Float = 0.0f
        private set

    var droppedFramesCount: Int = 0
        private set

    var thermalStatus: Int = ThermalMonitor.THERMAL_STATUS_NONE
        private set

    var isMemoryPressured: Boolean = false
        private set

    var tierTransitionCount: Int = 0
        private set

    // Warm-up state
    @Volatile
    var warmupUntilMs: Long = 0L
        private set

    @Volatile
    var warmupTriggerReason: String = ""
        private set

    var ignoredWarmupHitchCount: Int = 0
        private set

    private var consecutiveWarmupHitchCount: Int = 0

    // Promotion backoff state after repeated frame-p95 demotions
    var frameP95DemotionCount: Int = 0
        private set

    private var promotedSinceLastFrameP95Demote: Boolean = true

    var tierStableStartTimeMs: Long = 0L
        private set

    @Volatile
    var lastTimeToPromotionMs: Long = -1L
        private set

    val currentPromotionBackoffMs: Long
        get() = when {
            frameP95DemotionCount <= 0 -> promoteHeadroomMs
            frameP95DemotionCount == 1 -> maxOf(promoteHeadroomMs, PROMOTE_BACKOFF_SCHEDULE_MS[0])
            else -> {
                val idx = (frameP95DemotionCount - 1).coerceIn(0, PROMOTE_BACKOFF_SCHEDULE_MS.lastIndex)
                PROMOTE_BACKOFF_SCHEDULE_MS[idx]
            }
        }

    // Pre-allocated ring buffer for zero-allocation frame-time statistics
    private val frameWindow = FloatArray(WINDOW_SIZE)
    private val sortScratch = FloatArray(WINDOW_SIZE)
    private var frameWindowCount = 0
    private var frameWindowIndex = 0

    // Hysteresis and dwell timestamps
    private var lastChangeTimeMs: Long = -DWELL_WINDOW_MS
    private var highLoadStartTimeMs: Long = 0L
    private var headroomStartTimeMs: Long = 0L

    var onMemoryTrimRequested: (() -> Unit)? = null

    init {
        thermalMonitor.addListener(this)
    }

    /**
     * Explicit debug/config setter for Step 5 preview quality reduction.
     */
    fun setDebugPreviewQualityReduced(reduced: Boolean) {
        allowPreviewQualityReduction = reduced
        isPreviewQualityReduced = reduced
        if (reduced) {
            lastDegradationStep = DegradationStep.PREVIEW_QUALITY
            recordTransitionReason("debug:previewQualityReduced=true")
        }
    }

    /**
     * Scopes the higher iris analysis resolution (`irisAnalysisSizePx`, default `480`) strictly
     * to filters requiring `tracker.iris@1` (`recolor@1` / `eyeColor@1`):
     * - When `requiresIris == true`: raises `targetAnalysisSizePx` to `irisAnalysisSizePx` (`480` by default)
     *   for that filter only (even when starting on Tier B on a low-end device, unless severe thermal pressure).
     * - When `requiresIris == false`: immediately restores `targetAnalysisSizePx` to `320`
     *   (or `minAnalysisLongEdgePx` floor) when the iris filter is removed or switched away.
     */
    @JvmOverloads
    fun setFilterIrisRequirement(
        requiresIris: Boolean,
        nowMs: Long = System.currentTimeMillis()
    ) {
        val changed = (isIrisTrackingRequired != requiresIris)
        isIrisTrackingRequired = requiresIris
        if (requiresIris) {
            val desiredPx = maxOf(irisAnalysisSizePx, minAnalysisLongEdgePx)
            if (thermalStatus < ThermalMonitor.THERMAL_STATUS_SEVERE && targetAnalysisSizePx < desiredPx) {
                targetAnalysisSizePx = desiredPx
                if (changed) {
                    recordTransitionReason("iris:on(${targetAnalysisSizePx}px)")
                }
            }
        } else {
            val nonIrisCeilingPx = maxOf(ANALYSIS_SIZE_HIGH_PX, minAnalysisLongEdgePx)
            if (targetAnalysisSizePx > nonIrisCeilingPx) {
                targetAnalysisSizePx = nonIrisCeilingPx
                if (changed) {
                    recordTransitionReason("iris:off(${targetAnalysisSizePx}px)")
                }
            }
        }
        lastTimeToPromotionMs = timeToPromotionMs(nowMs)
    }

    /**
     * Configures the analysis long-edge resolution (`320`, `480`, or `640`) used when an iris filter
     * is active (`isIrisTrackingRequired == true`).
     */
    @JvmOverloads
    fun setConfiguredIrisAnalysisSizePx(
        sizePx: Int,
        nowMs: Long = System.currentTimeMillis()
    ): Int {
        val snapped = when {
            sizePx <= 320 -> 320
            sizePx >= 640 -> 640
            else -> 480
        }
        irisAnalysisSizePx = snapped
        if (isIrisTrackingRequired && thermalStatus < ThermalMonitor.THERMAL_STATUS_SEVERE) {
            targetAnalysisSizePx = maxOf(snapped, minAnalysisLongEdgePx)
            recordTransitionReason("irisSize:${targetAnalysisSizePx}px")
        }
        lastTimeToPromotionMs = timeToPromotionMs(nowMs)
        return irisAnalysisSizePx
    }

    /**
     * Cycles `irisAnalysisSizePx` through `320 -> 480 -> 640 -> 320`.
     */
    @JvmOverloads
    fun cycleConfiguredIrisAnalysisSizePx(nowMs: Long = System.currentTimeMillis()): Int {
        val next = when (irisAnalysisSizePx) {
            320 -> 480
            480 -> 640
            else -> 320
        }
        return setConfiguredIrisAnalysisSizePx(next, nowMs)
    }

    /**
     * Starts or extends a warm-up window (`5,000 ms` for delegate switch to GPU, `3,000 ms` for
     * filter switch, analysis rebind, or resume).
     * During warm-up:
     * - `resultAgeMs` and `conversionMs / inferenceMs` are not fed into governor decisions.
     * - Single isolated render frames `> 100 ms` (shader-compile hitches) are ignored for `frameTimeP95Ms`.
     * - Promotion headroom does not accumulate until warm-up finishes.
     */
    @JvmOverloads
    fun startWarmup(
        nowMs: Long = System.currentTimeMillis(),
        durationMs: Long = WARMUP_DEFAULT_DURATION_MS,
        reason: String = "warmup"
    ) {
        val targetEnd = nowMs + durationMs.coerceAtLeast(0L)
        if (!isInWarmup(nowMs) || targetEnd > warmupUntilMs) {
            warmupUntilMs = targetEnd
        }
        warmupTriggerReason = reason
        consecutiveWarmupHitchCount = 0
        headroomStartTimeMs = 0L
        lastTimeToPromotionMs = timeToPromotionMs(nowMs)
    }

    fun isInWarmup(nowMs: Long): Boolean {
        return warmupUntilMs > 0L && nowMs < warmupUntilMs
    }

    fun isAtMaxQuality(): Boolean {
        val baseHighPx = if (isIrisTrackingRequired) irisAnalysisSizePx else ANALYSIS_SIZE_HIGH_PX
        val highPx = maxOf(baseHighPx, minAnalysisLongEdgePx)
        return !isPreviewQualityReduced &&
            occluderLatSegments >= 16 &&
            occluderLonSegments >= 24 &&
            currentTier == Tier.A &&
            maxTextureSize >= TEXTURE_MAX_HIGH &&
            targetAnalysisSizePx >= highPx &&
            targetTrackerCadenceHz >= MAX_TRACKER_CADENCE_HZ
    }

    /**
     * Returns remaining milliseconds until the next promotion step at [nowMs], or `-1L` when
     * already at maximum quality (`Tier.A`, `30 Hz @ 320 px`, `1024` texture cap, `16x24` segments).
     */
    fun timeToPromotionMs(nowMs: Long): Long {
        if (isAtMaxQuality()) {
            return -1L
        }
        val reqHeadroomMs = currentPromotionBackoffMs
        if (isInWarmup(nowMs)) {
            val warmupRem = (warmupUntilMs - nowMs).coerceAtLeast(0L)
            val dwellRemAtWarmupEnd = (dwellWindowMs - (warmupUntilMs - lastChangeTimeMs)).coerceAtLeast(0L)
            return warmupRem + maxOf(reqHeadroomMs, dwellRemAtWarmupEnd)
        }
        val dwellRem = (dwellWindowMs - (nowMs - lastChangeTimeMs)).coerceAtLeast(0L)
        val headroomRem = if (headroomStartTimeMs > 0L) {
            (reqHeadroomMs - (nowMs - headroomStartTimeMs)).coerceAtLeast(0L)
        } else {
            reqHeadroomMs
        }
        return maxOf(dwellRem, headroomRem)
    }

    fun formatHudPromotionStatus(nowMs: Long): String {
        val remMs = timeToPromotionMs(nowMs)
        return if (remMs < 0L) {
            "promo=max"
        } else {
            String.format(Locale.US, "promo=%.1fs", remMs / 1000.0f)
        }
    }

    fun formatHudGovernorSummary(nowMs: Long, fallbackReason: String = lastTransitionReason): String {
        val reason = if (lastTransitionReason != "none") lastTransitionReason else fallbackReason
        val backoffSec = currentPromotionBackoffMs / 1000L
        val promoStr = formatHudPromotionStatus(nowMs)
        return "Gov: ${targetTrackerCadenceHz}Hz@${targetAnalysisSizePx}px ($reason, lowEnd=$isLowEndDevice, backoff=${backoffSec}s, $promoStr)"
    }

    private fun recordTransitionReason(reason: String) {
        lastTransitionReason = reason
        onTransitionLogged?.invoke(reason)
    }

    /**
     * Resets governor state to initial device-class defaults at `nowMs`.
     */
    fun reset(nowMs: Long = System.currentTimeMillis()) {
        currentTier = if (isLowEndDevice) Tier.B else Tier.A
        targetTrackerCadenceHz = MAX_TRACKER_CADENCE_HZ
        val defaultSizePx = if (isIrisTrackingRequired) {
            irisAnalysisSizePx
        } else if (isLowEndDevice) {
            ANALYSIS_SIZE_LOW_PX
        } else {
            ANALYSIS_SIZE_HIGH_PX
        }
        targetAnalysisSizePx = defaultSizePx.coerceAtLeast(minAnalysisLongEdgePx)
        maxTextureSize = if (isLowEndDevice) TEXTURE_MAX_MED else TEXTURE_MAX_HIGH
        occluderLatSegments = 16
        occluderLonSegments = 24
        isPreviewQualityReduced = false
        lastDegradationStep = DegradationStep.NONE
        lastTransitionReason = "none"
        frameWindowCount = 0
        frameWindowIndex = 0
        droppedFramesCount = 0
        frameTimeP50Ms = 16.6f
        frameTimeP95Ms = 16.6f
        lastDrawCpuDurationMs = 0.0f
        resultAgeMs = 0.0f
        trackerConversionMs = 0.0f
        trackerInferenceMs = 0.0f
        isContinuousRenderMode = true
        lastChangeTimeMs = nowMs
        highLoadStartTimeMs = 0L
        headroomStartTimeMs = 0L
        tierTransitionCount = 0
        warmupUntilMs = 0L
        warmupTriggerReason = ""
        ignoredWarmupHitchCount = 0
        consecutiveWarmupHitchCount = 0
        frameP95DemotionCount = 0
        promotedSinceLastFrameP95Demote = true
        tierStableStartTimeMs = nowMs
        lastTimeToPromotionMs = timeToPromotionMs(nowMs)
    }

    /**
     * Updates tracker telemetry inputs (tracker rate in Hz, result age at render in ms,
     * conversionMs, and inferenceMs). Ignored for age/inference during the warm-up window.
     */
    @JvmOverloads
    fun updateTrackerTelemetry(
        rateHz: Float,
        ageAtRenderMs: Float,
        conversionMs: Float = this.trackerConversionMs,
        inferenceMs: Float = this.trackerInferenceMs,
        nowMs: Long = System.currentTimeMillis()
    ) {
        trackerRateHz = rateHz.coerceAtLeast(0f)
        if (!isInWarmup(nowMs)) {
            resultAgeMs = ageAtRenderMs.coerceAtLeast(0f)
            trackerConversionMs = conversionMs.coerceAtLeast(0f)
            trackerInferenceMs = inferenceMs.coerceAtLeast(0f)
        }
    }

    /**
     * Records a single frame duration and evaluates hysteresis & degradation/promotion rules.
     * Preallocates all buffers to guarantee zero GC allocations in the frame loop.
     *
     * Rules:
     * - `durationMs` is used ONLY when `isContinuousRender == true` (`RENDERMODE_CONTINUOUSLY`).
     *   When `isContinuousRender == false`, `durationMs` is ignored.
     * - During warm-up (`isInWarmup(nowMs)`), `resultAgeAtRenderMs`, `conversionMs`, and `inferenceMs`
     *   are NOT fed into governor decisions, and a single isolated render frame `> 100 ms`
     *   (shader-compile hitch) is ignored for `frameWindow` without hiding sustained slowness.
     */
    @JvmOverloads
    fun recordFrameDuration(
        durationMs: Float,
        nowMs: Long = System.currentTimeMillis(),
        resultAgeAtRenderMs: Float = this.resultAgeMs,
        currentTrackerHz: Float = this.trackerRateHz,
        drawCpuDurationMs: Float = 0.0f,
        isContinuousRender: Boolean = this.isContinuousRenderMode,
        conversionMs: Float = this.trackerConversionMs,
        inferenceMs: Float = this.trackerInferenceMs
    ) {
        this.isContinuousRenderMode = isContinuousRender
        lastDrawCpuDurationMs = if (drawCpuDurationMs.isFinite() && drawCpuDurationMs >= 0f) {
            drawCpuDurationMs
        } else {
            0.0f
        }
        trackerRateHz = currentTrackerHz.coerceAtLeast(0f)

        val inWarmup = isInWarmup(nowMs)
        if (!inWarmup) {
            resultAgeMs = if (resultAgeAtRenderMs.isFinite() && resultAgeAtRenderMs >= 0f) resultAgeAtRenderMs else 0f
            trackerConversionMs = if (conversionMs.isFinite() && conversionMs >= 0f) conversionMs else 0f
            trackerInferenceMs = if (inferenceMs.isFinite() && inferenceMs >= 0f) inferenceMs else 0f
        }

        if (isContinuousRender) {
            val validDuration = if (durationMs.isNaN() || durationMs.isInfinite() || durationMs < 0f) 16.6f else durationMs
            val ignoreSingleWarmupHitch = if (inWarmup && validDuration > WARMUP_SINGLE_HITCH_THRESHOLD_MS) {
                if (consecutiveWarmupHitchCount == 0) {
                    consecutiveWarmupHitchCount = 1
                    ignoredWarmupHitchCount++
                    true
                } else {
                    consecutiveWarmupHitchCount++
                    false
                }
            } else {
                consecutiveWarmupHitchCount = 0
                false
            }

            if (!ignoreSingleWarmupHitch) {
                if (validDuration > maxFrameP95Ms) {
                    droppedFramesCount++
                }

                frameWindow[frameWindowIndex] = validDuration
                frameWindowIndex = (frameWindowIndex + 1) % WINDOW_SIZE
                if (frameWindowCount < WINDOW_SIZE) {
                    frameWindowCount++
                }

                computePercentiles()
            }
        }

        evaluate(nowMs)
    }

    private fun computePercentiles() {
        if (frameWindowCount == 0) {
            frameTimeP50Ms = 16.6f
            frameTimeP95Ms = 16.6f
            return
        }
        System.arraycopy(frameWindow, 0, sortScratch, 0, frameWindowCount)
        java.util.Arrays.sort(sortScratch, 0, frameWindowCount)

        val p50Idx = ((frameWindowCount - 1) * 0.50f).toInt().coerceIn(0, frameWindowCount - 1)
        val p95Idx = ((frameWindowCount - 1) * 0.95f).toInt().coerceIn(0, frameWindowCount - 1)
        frameTimeP50Ms = sortScratch[p50Idx]
        frameTimeP95Ms = sortScratch[p95Idx]
    }

    private fun canDegradeTrackerCost(): Boolean {
        return targetTrackerCadenceHz > MIN_TRACKER_CADENCE_HZ ||
            targetAnalysisSizePx > minAnalysisLongEdgePx
    }

    /**
     * Evaluates governor rules at `nowMs` enforcing the 3-second dwell window,
     * 1.5-second demotion hysteresis (`frameTimeP95Ms > maxFrameP95Ms`), and post-warmup
     * promotion headroom (`frameTimeP95Ms <= headroomFrameP95Ms`) with promotion backoff.
     */
    fun evaluate(nowMs: Long) {
        if (tierStableStartTimeMs == 0L) {
            tierStableStartTimeMs = nowMs
        }
        if (frameP95DemotionCount > 0 && (nowMs - tierStableStartTimeMs) >= backoffResetStableMs) {
            frameP95DemotionCount = 0
            promotedSinceLastFrameP95Demote = true
        }

        val inWarmup = isInWarmup(nowMs)
        if (inWarmup) {
            headroomStartTimeMs = 0L
        }

        // Strict dwell check: NO changes inside the dwell window
        if (nowMs - lastChangeTimeMs < dwellWindowMs) {
            lastTimeToPromotionMs = timeToPromotionMs(nowMs)
            return
        }

        val isFrameP95Overload = frameTimeP95Ms > maxFrameP95Ms
        val renderOrThermalOverloaded = isFrameP95Overload ||
            (thermalStatus >= ThermalMonitor.THERMAL_STATUS_MODERATE) ||
            isMemoryPressured

        val trackerCostMs = trackerConversionMs + trackerInferenceMs
        val trackerCostOverloaded = enableTrackerCostThrottle &&
            !inWarmup &&
            (resultAgeMs > MAX_ACCEPTABLE_RESULT_AGE_MS) &&
            (trackerCostMs > MAX_ACCEPTABLE_TRACKER_COST_MS) &&
            canDegradeTrackerCost()

        if (renderOrThermalOverloaded || trackerCostOverloaded) {
            headroomStartTimeMs = 0L
            if (highLoadStartTimeMs == 0L) {
                highLoadStartTimeMs = nowMs
            } else if (nowMs - highLoadStartTimeMs >= demoteDelayMs) {
                val reason = when {
                    isFrameP95Overload ->
                        String.format(Locale.US, "demote:frameP95=%.1fms", frameTimeP95Ms)
                    thermalStatus >= ThermalMonitor.THERMAL_STATUS_MODERATE ->
                        "demote:thermal=$thermalStatus"
                    isMemoryPressured ->
                        "demote:memPressure=true"
                    else ->
                        String.format(
                            Locale.US,
                            "demote:age=%.1fms(conv+inf=%.1fms)",
                            resultAgeMs,
                            trackerCostMs
                        )
                }
                val prevTier = currentTier
                if (degradeOneStep(nowMs, allowVisualTierDemotion = renderOrThermalOverloaded, reason = reason)) {
                    lastChangeTimeMs = nowMs
                    if (currentTier != prevTier) {
                        tierStableStartTimeMs = nowMs
                    }
                    if (isFrameP95Overload) {
                        if (promotedSinceLastFrameP95Demote || frameP95DemotionCount == 0) {
                            frameP95DemotionCount = (frameP95DemotionCount + 1).coerceAtMost(PROMOTE_BACKOFF_SCHEDULE_MS.size)
                            promotedSinceLastFrameP95Demote = false
                        }
                        tierStableStartTimeMs = nowMs
                    }
                }
                highLoadStartTimeMs = 0L
            }
        } else {
            highLoadStartTimeMs = 0L
            val trackerHeadroomOk = !enableTrackerCostThrottle ||
                inWarmup ||
                (resultAgeMs <= MAX_ACCEPTABLE_RESULT_AGE_MS) ||
                (trackerCostMs <= MAX_ACCEPTABLE_TRACKER_COST_MS)

            val hasHeadroom = !inWarmup &&
                frameTimeP95Ms <= headroomFrameP95Ms &&
                thermalStatus <= ThermalMonitor.THERMAL_STATUS_LIGHT &&
                !isMemoryPressured &&
                trackerHeadroomOk

            if (hasHeadroom && !isAtMaxQuality()) {
                val requiredHeadroomMs = currentPromotionBackoffMs
                if (headroomStartTimeMs == 0L) {
                    headroomStartTimeMs = nowMs
                } else if (nowMs - headroomStartTimeMs >= requiredHeadroomMs) {
                    val prevTier = currentTier
                    val reason = String.format(Locale.US, "promote:headroom(frameP95=%.1fms)", frameTimeP95Ms)
                    if (promoteOneStep(nowMs, reason)) {
                        lastChangeTimeMs = nowMs
                        promotedSinceLastFrameP95Demote = true
                        if (currentTier != prevTier) {
                            tierStableStartTimeMs = nowMs
                        }
                    }
                    headroomStartTimeMs = 0L
                }
            } else {
                headroomStartTimeMs = 0L
            }
        }
        lastTimeToPromotionMs = timeToPromotionMs(nowMs)
    }

    /**
     * Executes the README Section 13 degradation priority order:
     * 1. Reduce tracker cadence (minimum 15 Hz), then analysis size (minimum [minAnalysisLongEdgePx]).
     * 2. Move to a lower package tier and disable its specified behaviors (ONLY when [allowVisualTierDemotion] is true).
     * 3. Reduce LOD and texture size (ONLY when [allowVisualTierDemotion] is true).
     * 4. Reduce occluder segment counts (ONLY when [allowVisualTierDemotion] is true).
     * 5. Reduce preview quality ONLY with explicit product/debug approval ([allowPreviewQualityReduction]).
     */
    private fun degradeOneStep(
        nowMs: Long,
        allowVisualTierDemotion: Boolean,
        reason: String
    ): Boolean {
        // Priority 1a: Reduce tracker cadence down to 15 Hz floor
        if (targetTrackerCadenceHz > MIN_TRACKER_CADENCE_HZ) {
            targetTrackerCadenceHz = when {
                targetTrackerCadenceHz > MID_TRACKER_CADENCE_HZ -> MID_TRACKER_CADENCE_HZ
                targetTrackerCadenceHz > LOW_MID_TRACKER_CADENCE_HZ -> LOW_MID_TRACKER_CADENCE_HZ
                else -> MIN_TRACKER_CADENCE_HZ
            }.coerceAtLeast(MIN_TRACKER_CADENCE_HZ)
            lastDegradationStep = DegradationStep.TRACKER_CADENCE_AND_ANALYSIS_SIZE
            recordTransitionReason(reason)
            Log.w(TAG, "Degraded Step 1a ($reason): tracker cadence -> ${targetTrackerCadenceHz}Hz")
            return true
        }

        // Priority 1b: Reduce analysis frame size down to minAnalysisLongEdgePx floor (default 320)
        val floorPx = minAnalysisLongEdgePx.coerceAtLeast(ANALYSIS_SIZE_LOW_PX)
        if (targetAnalysisSizePx > floorPx) {
            targetAnalysisSizePx = if (targetAnalysisSizePx > ANALYSIS_SIZE_MED_PX && ANALYSIS_SIZE_MED_PX >= floorPx) {
                ANALYSIS_SIZE_MED_PX
            } else {
                floorPx
            }
            lastDegradationStep = DegradationStep.TRACKER_CADENCE_AND_ANALYSIS_SIZE
            recordTransitionReason(reason)
            Log.w(TAG, "Degraded Step 1b ($reason): analysis long edge -> ${targetAnalysisSizePx}px")
            return true
        }

        // Visual quality tiers (Steps 2-5) must NEVER be demoted by tracker age/cost alone!
        if (!allowVisualTierDemotion) {
            return false
        }

        // Priority 2: Move to a lower package tier (A -> B -> C), respecting packageMinTier
        val nextLowerTier = when (currentTier) {
            Tier.A -> Tier.B
            Tier.B -> Tier.C
            Tier.C -> Tier.C
        }
        val clampedLowerTier = TierPolicy.clampToMinTier(nextLowerTier, packageMinTier)
        if (clampedLowerTier != currentTier) {
            currentTier = clampedLowerTier
            tierTransitionCount++
            // Align tier texture cap with the newly entered tier if higher
            val tierDefaultTex = when (currentTier) {
                Tier.A -> TEXTURE_MAX_HIGH
                Tier.B -> TEXTURE_MAX_MED
                Tier.C -> TEXTURE_MAX_LOW
            }
            if (maxTextureSize > tierDefaultTex) {
                maxTextureSize = tierDefaultTex
            }
            lastDegradationStep = DegradationStep.PACKAGE_TIER_AND_BEHAVIORS
            recordTransitionReason(reason)
            Log.w(TAG, "Degraded Step 2 ($reason): package tier -> ${currentTier.name} at $nowMs")
            return true
        }

        // Priority 3: Reduce LOD and texture size (1024 -> 512 -> 256)
        if (maxTextureSize > TEXTURE_MAX_LOW) {
            maxTextureSize = if (maxTextureSize > TEXTURE_MAX_MED) TEXTURE_MAX_MED else TEXTURE_MAX_LOW
            lastDegradationStep = DegradationStep.LOD_AND_TEXTURE_SIZE
            recordTransitionReason(reason)
            Log.w(TAG, "Degraded Step 3 ($reason): maxTextureSize -> $maxTextureSize")
            return true
        }

        // Priority 4: Reduce occluder segment counts (16x24 -> 12x18 -> 8x12)
        if (occluderLatSegments > 8 || occluderLonSegments > 12) {
            if (occluderLatSegments > 12) {
                occluderLatSegments = 12
                occluderLonSegments = 18
            } else {
                occluderLatSegments = 8
                occluderLonSegments = 12
            }
            lastDegradationStep = DegradationStep.OCCLUDER_SEGMENTS
            recordTransitionReason(reason)
            Log.w(TAG, "Degraded Step 4 ($reason): occluder segments -> ${occluderLatSegments}x${occluderLonSegments}")
            return true
        }

        // Priority 5: Reduce preview quality ONLY with explicit debug/config flag
        if (allowPreviewQualityReduction && !isPreviewQualityReduced) {
            isPreviewQualityReduced = true
            lastDegradationStep = DegradationStep.PREVIEW_QUALITY
            recordTransitionReason(reason)
            Log.w(TAG, "Degraded Step 5 ($reason): preview quality reduced (explicit flag enabled)")
            return true
        }

        return false
    }

    /**
     * Promotes quality one step in reverse order after 10s of sustained headroom.
     */
    private fun promoteOneStep(nowMs: Long, reason: String): Boolean {
        // Restore preview quality first if it was reduced
        if (isPreviewQualityReduced) {
            isPreviewQualityReduced = false
            recordTransitionReason(reason)
            Log.i(TAG, "Promoted ($reason): restored full preview quality")
            return true
        }

        // Restore occluder segments
        if (occluderLatSegments < 16 || occluderLonSegments < 24) {
            if (occluderLatSegments < 12) {
                occluderLatSegments = 12
                occluderLonSegments = 18
            } else {
                occluderLatSegments = 16
                occluderLonSegments = 24
            }
            recordTransitionReason(reason)
            Log.i(TAG, "Promoted ($reason): occluder segments -> ${occluderLatSegments}x${occluderLonSegments}")
            return true
        }

        // Promote package tier (C -> B -> A) and corresponding texture size
        if (currentTier != Tier.A) {
            currentTier = when (currentTier) {
                Tier.C -> Tier.B
                Tier.B -> Tier.A
                Tier.A -> Tier.A
            }
            tierTransitionCount++
            maxTextureSize = when (currentTier) {
                Tier.A -> TEXTURE_MAX_HIGH
                Tier.B -> TEXTURE_MAX_MED
                Tier.C -> TEXTURE_MAX_LOW
            }
            recordTransitionReason(reason)
            Log.i(TAG, "Promoted ($reason): package tier -> ${currentTier.name} at $nowMs")
            return true
        }

        // Restore texture size if still below high cap
        if (maxTextureSize < TEXTURE_MAX_HIGH) {
            maxTextureSize = if (maxTextureSize < TEXTURE_MAX_MED) TEXTURE_MAX_MED else TEXTURE_MAX_HIGH
            recordTransitionReason(reason)
            Log.i(TAG, "Promoted ($reason): maxTextureSize -> $maxTextureSize")
            return true
        }

        // Restore analysis size up to ANALYSIS_SIZE_HIGH_PX (320) or irisAnalysisSizePx (480/640) when iris tracking is required
        val baseHighPx = if (isIrisTrackingRequired) irisAnalysisSizePx else ANALYSIS_SIZE_HIGH_PX
        val highPx = maxOf(baseHighPx, minAnalysisLongEdgePx)
        if (targetAnalysisSizePx < highPx) {
            targetAnalysisSizePx = when {
                targetAnalysisSizePx < ANALYSIS_SIZE_MED_PX && ANALYSIS_SIZE_MED_PX < highPx -> ANALYSIS_SIZE_MED_PX
                targetAnalysisSizePx < ANALYSIS_SIZE_HIGH_PX && ANALYSIS_SIZE_HIGH_PX < highPx -> ANALYSIS_SIZE_HIGH_PX
                targetAnalysisSizePx < ANALYSIS_SIZE_IRIS_DEFAULT_PX && ANALYSIS_SIZE_IRIS_DEFAULT_PX < highPx -> ANALYSIS_SIZE_IRIS_DEFAULT_PX
                else -> highPx
            }
            recordTransitionReason(reason)
            Log.i(TAG, "Promoted ($reason): analysis size -> ${targetAnalysisSizePx}px")
            return true
        }

        // Restore tracker cadence up to 30 Hz ceiling
        if (targetTrackerCadenceHz < MAX_TRACKER_CADENCE_HZ) {
            targetTrackerCadenceHz = when {
                targetTrackerCadenceHz < LOW_MID_TRACKER_CADENCE_HZ -> LOW_MID_TRACKER_CADENCE_HZ
                targetTrackerCadenceHz < MID_TRACKER_CADENCE_HZ -> MID_TRACKER_CADENCE_HZ
                else -> MAX_TRACKER_CADENCE_HZ
            }.coerceAtMost(MAX_TRACKER_CADENCE_HZ)
            recordTransitionReason(reason)
            Log.i(TAG, "Promoted ($reason): tracker cadence -> ${targetTrackerCadenceHz}Hz")
            return true
        }

        return false
    }

    override fun onThermalStatusChanged(thermalStatus: Int) {
        this.thermalStatus = thermalStatus
    }

    override fun onMemoryPressure(trimLevel: Int, isLowMemory: Boolean) {
        this.isMemoryPressured = isLowMemory
        if (isLowMemory) {
            onMemoryTrimRequested?.invoke()
        }
    }
}
