package com.example

import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.FrameClock
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.ThermalMonitor
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.scene.AnchorResolver
import com.example.engine.scene.BehaviorRunner
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.PropertySlots
import com.example.engine.scene.SignalBus
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

/**
 * Pure-JVM unit tests for Chunk: Governor False Demotions + Probe Warm-Up:
 * 1. Tier change (visual quality) is NEVER triggered by `resultAge` alone; tracker-cost throttling
 *    is disabled by default (`enableTrackerCostThrottle = false`) so healthy render + thermal with
 *    `conv + inf = 46 ms` and `age = 180 ms` never changes cadence, analysis size, or tier.
 *    When `enableTrackerCostThrottle = true`, high `resultAge` + high `conv + inf` reduces only
 *    Step 1 (tracker cadence / analysis size down to floor), never visual tier.
 * 2. Warm-up durations: `5 s` after delegate switch to GPU, `3 s` after CPU delegate switch,
 *    filter switch, analysis rebind, or resume. During warm-up:
 *    - Age/inference are not fed into governor decisions.
 *    - Single isolated render frames `> 100 ms` (shader-compile hitches) are ignored for `frameP95`
 *      without hiding sustained slowness.
 *    - `LatencyProbe` marks samples `warmup=true`, excludes them from HUD rolling averages and
 *      `age p50/p95`, and retains them in `dumpCsv()` with `warmup=true`.
 * 3. Analysis size floor: default `minAnalysisLongEdgePx = 320` and initial `targetAnalysisSizePx = 320`
 *    on startup (even on low-end device config unless `minAnalysisLongEdgePx` is explicitly lowered).
 * 4. Step 5 (`isPreviewQualityReduced`) never becomes `true` automatically unless explicitly enabled
 *    via debug/config flag (`allowPreviewQualityReduction = true` or `setDebugPreviewQualityReduced(true)`).
 * 5. Continuous render mode gate: frame-interval input is used only when `isContinuousRender == true`.
 * 6. Demote/promote transition reasons (`signal name + value`) are logged on `LatencyProbe.cachedHudLine`
 *    and in `dumpCsv()`.
 */
class GovernorWarmupAndFloorUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    /**
     * Extra Test 5(a) + Requirement 1:
     * - Default `enableTrackerCostThrottle = false`: with healthy render (`frameTimeP95Ms = 16.6 ms`)
     *   and healthy thermal (`THERMAL_STATUS_NONE`), `conv + inf = 46 ms` and `age = 180 ms` must
     *   produce NO change in tracker cadence (`30 Hz`), analysis size (`320 px`), or tier (`Tier.A`).
     * - Even when `enableTrackerCostThrottle = true`, `resultAge` alone with low `conv + inf` (`12 ms`)
     *   produces NO change, and high `age = 180 ms` with `conv + inf = 46 ms` reduces ONLY tracker
     *   cost (cadence down to `15 Hz`, size floored at `320 px`) and NEVER demotes `currentTier`.
     */
    @Test
    fun testHealthyRenderAndThermalHighConvPlusInf46msAndAge180msNeverChangesCadenceSizeOrTier() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        assertFalse(
            "enableTrackerCostThrottle must be disabled by default",
            gov.enableTrackerCostThrottle
        )

        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        // Advance well past 5s dwell window and feed 30 seconds of healthy render (16.6ms) + healthy thermal,
        // but high conv + inf = 46 ms (14 ms conv + 32 ms inf) and high resultAge = 180 ms
        tMs += PerformanceGovernor.DWELL_WINDOW_MS + 200L
        for (i in 0 until 600) {
            tMs += 50L
            gov.recordFrameDuration(
                durationMs = 16.6f,
                nowMs = tMs,
                resultAgeAtRenderMs = 180.0f,
                currentTrackerHz = 21.0f,
                isContinuousRender = true,
                conversionMs = 14.0f,
                inferenceMs = 32.0f // conv + inf = 46.0 ms
            )
        }

        assertEquals(
            "Healthy render + thermal with conv+inf=46ms and age=180ms must NOT reduce tracker cadence by default",
            30,
            gov.targetTrackerCadenceHz
        )
        assertEquals(
            "Healthy render + thermal with conv+inf=46ms and age=180ms must NOT reduce analysis size",
            320,
            gov.targetAnalysisSizePx
        )
        assertEquals(
            "Healthy render + thermal with conv+inf=46ms and age=180ms must NEVER demote visual tier",
            PerformanceGovernor.Tier.A,
            gov.currentTier
        )
        assertEquals("none", gov.lastTransitionReason)

        // Now verify the gated tracker-cost throttle path (enableTrackerCostThrottle = true):
        // 1) High age (180 ms) when conv + inf is fast (12 ms <= 33.3 ms) must NOT throttle even with flag = true
        gov.enableTrackerCostThrottle = true
        for (i in 0 until 200) {
            tMs += 50L
            gov.recordFrameDuration(
                durationMs = 16.6f,
                nowMs = tMs,
                resultAgeAtRenderMs = 180.0f,
                currentTrackerHz = 30.0f,
                isContinuousRender = true,
                conversionMs = 4.0f,
                inferenceMs = 8.0f // conv + inf = 12.0 ms <= 33.3 ms
            )
        }
        assertEquals(
            "High age without high conv+inf must not throttle tracker cadence even when flag is enabled",
            30,
            gov.targetTrackerCadenceHz
        )
        assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)

        // 2) High age (180 ms) caused by high conv + inf (46 ms > 33.3 ms) with flag = true reduces
        // ONLY Step 1 (cadence 30 -> 24 -> 20 -> 15 Hz, size stays at 320 floor) and NEVER demotes Tier A!
        for (cycle in 0 until 10) {
            tMs += PerformanceGovernor.DWELL_WINDOW_MS + 100L
            for (i in 0 until 40) {
                tMs += 50L
                gov.recordFrameDuration(
                    durationMs = 16.6f,
                    nowMs = tMs,
                    resultAgeAtRenderMs = 180.0f,
                    currentTrackerHz = 20.0f,
                    isContinuousRender = true,
                    conversionMs = 14.0f,
                    inferenceMs = 32.0f // conv + inf = 46.0 ms
                )
            }
        }
        assertEquals(15, gov.targetTrackerCadenceHz)
        assertEquals(320, gov.targetAnalysisSizePx)
        assertEquals(
            "Age/tracker-cost must NEVER demote visual tier (Steps 2-5) even after exhausting Step 1",
            PerformanceGovernor.Tier.A,
            gov.currentTier
        )
        assertEquals(16, gov.occluderLatSegments)
        assertEquals(24, gov.occluderLonSegments)
        assertFalse(gov.isPreviewQualityReduced)
        assertTrue(
            "Transition reason must log age and conv+inf",
            gov.lastTransitionReason.startsWith("demote:age=180.0ms(conv+inf=46.0ms)")
        )
    }

    /**
     * Extra Test 5(b) + Requirement 2:
     * - Filter switch starts a 3 s warm-up (`3,000 ms`), GPU delegate switch starts a 5 s warm-up (`5,000 ms`).
     * - A single `150 ms` shader-compile hitch frame inside warm-up is ignored for `frameTimeP95Ms`
     *   and does NOT demote the tier or cadence.
     * - Sustained slow frames inside warm-up (e.g. consecutive `> 100 ms` frames or sustained `45 ms` frames)
     *   are NOT hidden and DO trigger demotion once dwell + hysteresis are satisfied.
     */
    @Test
    fun testFilterSwitchSingle150msHitchInsideWarmupIgnoredWithoutHidingSustainedSlowness() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        // Advance past initial 5s dwell window
        tMs = 16_000L
        for (i in 0 until 30) {
            gov.recordFrameDuration(durationMs = 16.6f, nowMs = tMs + i * 16L)
        }
        tMs = 17_000L

        // Trigger 3s filter-switch warm-up at t = 17_000 ms -> warm-up active until 20_000 ms
        gov.startWarmup(
            nowMs = tMs,
            durationMs = PerformanceGovernor.WARMUP_DEFAULT_DURATION_MS,
            reason = "filterSwitch:neon_glasses"
        )
        assertTrue(gov.isInWarmup(17_500L))
        assertTrue(gov.isInWarmup(19_999L))
        assertFalse(gov.isInWarmup(20_000L))

        // Feed one 150ms shader-compile hitch at t = 17_100 ms followed by normal 16.6ms frames
        gov.recordFrameDuration(
            durationMs = 150.0f,
            nowMs = 17_100L,
            resultAgeAtRenderMs = 220.0f,
            conversionMs = 20.0f,
            inferenceMs = 60.0f
        )
        assertEquals(
            "Single 150ms frame inside warm-up must be ignored as a shader-compile hitch",
            1,
            gov.ignoredWarmupHitchCount
        )
        assertEquals(
            "Warm-up must not feed high resultAge into governor",
            0.0f,
            gov.resultAgeMs,
            0.001f
        )

        for (i in 1..100) {
            gov.recordFrameDuration(
                durationMs = 16.6f,
                nowMs = 17_100L + i * 20L
            )
        }
        assertEquals(
            "frameTimeP95Ms must remain healthy (~16.6ms) after ignoring single 150ms hitch",
            16.6f,
            gov.frameTimeP95Ms,
            0.1f
        )
        assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)
        assertEquals(30, gov.targetTrackerCadenceHz)

        // Verify sustained slowness (e.g. consecutive 120ms frames) inside warm-up is NOT hidden
        val govSustained = PerformanceGovernor(context = null, isLowEndDevice = false)
        govSustained.reset(nowMs = 10_000L)
        val warmupStartMs = 16_000L // past 5s dwell window
        govSustained.startWarmup(
            nowMs = warmupStartMs,
            durationMs = PerformanceGovernor.WARMUP_GPU_DELEGATE_DURATION_MS, // 5s warm-up
            reason = "delegateSwitch:GPU"
        )
        assertEquals(warmupStartMs + 5_000L, govSustained.warmupUntilMs)

        // Feed sustained 120ms frames for 2.0 seconds inside the 5s warm-up window:
        // Only the very first >100ms frame is ignored; subsequent consecutive >100ms frames enter frameWindow
        for (i in 0 until 40) {
            govSustained.recordFrameDuration(
                durationMs = 120.0f,
                nowMs = warmupStartMs + i * 50L
            )
        }
        assertEquals(
            "Only the first isolated >100ms frame may be skipped; consecutive >100ms frames are recorded",
            1,
            govSustained.ignoredWarmupHitchCount
        )
        assertTrue(
            "Sustained slowness inside warm-up must raise frameTimeP95Ms",
            govSustained.frameTimeP95Ms > 100.0f
        )
        assertEquals(
            "Sustained slowness inside warm-up must still demote cadence after 1.5s hysteresis",
            24,
            govSustained.targetTrackerCadenceHz
        )
    }

    /**
     * Requirement 2:
     * Verify `CameraSession` triggers `5 s` warm-up on GPU delegate switch and `3 s` warm-up on
     * CPU delegate switch, filter switch, analysis rebind, and resume, and that `LatencyProbe`
     * excludes `warmup=true` samples from HUD rolling averages and `age p50/p95` while keeping
     * them in `dumpCsv()` with `warmup=true`.
     */
    @Test
    fun testCameraSessionWarmupTriggersAndLatencyProbeWarmupExclusionAndCsvFlag() {
        var fakeClockNs = 10_000_000_000L // 10.0s
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val probe = LatencyProbe(
            windowDurationNs = 5_000_000_000L,
            hudRefreshIntervalNs = 100_000_000L
        )
        val pipeline = SignalPipeline()
        val fakeProvider = object : FaceGeometryProvider {
            override val modelSha256: String = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
            override fun start(config: ProviderConfig) {}
            override fun stop() {}
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {}
            override fun submit(frame: AnalysisFrame) {
                frame.imageProxy?.close()
            }
        }

        val session = CameraSession(
            provider = fakeProvider,
            signalPipeline = pipeline,
            governor = gov,
            latencyProbe = probe
        )

        // 1. Switch to GPU delegate -> 5s warm-up (10.0s .. 15.0s)
        session.setUseGpuDelegate(true)
        assertEquals(15_000L, gov.warmupUntilMs)
        assertEquals("delegateSwitch:GPU", gov.warmupTriggerReason)
        assertEquals(15_000_000_000L, probe.warmupUntilNs)
        assertTrue(probe.isInWarmup(14_999_000_000L))
        assertFalse(probe.isInWarmup(15_000_000_000L))

        // 2. Advance to 20.0s and switch to CPU delegate -> 3s warm-up (20.0s .. 23.0s)
        fakeClockNs = 20_000_000_000L
        session.setUseGpuDelegate(false)
        assertEquals(23_000L, gov.warmupUntilMs)
        assertEquals("delegateSwitch:CPU", gov.warmupTriggerReason)
        assertEquals(23_000_000_000L, probe.warmupUntilNs)

        // 3. Advance to 25.0s and switch filter -> 3s warm-up (25.0s .. 28.0s)
        fakeClockNs = 25_000_000_000L
        val dummyEffect = CompiledEffect(
            packageId = "cat_ears",
            packageName = "Cat Ears",
            schemaVersion = "xfilter/3",
            nodes = emptyList(),
            materials = emptyMap(),
            assets = emptyMap(),
            lighting = null,
            signalBus = SignalBus(0, emptyList(), emptyMap()),
            behaviorRunner = BehaviorRunner(emptyList(), emptyList(), emptyList(), emptyList(), 0),
            anchorResolver = AnchorResolver(emptyMap()),
            propertySlots = PropertySlots(0)
        )
        session.applyEffectRequirements(dummyEffect)
        assertEquals(28_000L, gov.warmupUntilMs)
        assertEquals("filterSwitch:cat_ears", gov.warmupTriggerReason)
        assertEquals(28_000_000_000L, probe.warmupUntilNs)

        // 4. Advance to 30.0s and rebind analysis -> 3s warm-up (30.0s .. 33.0s)
        fakeClockNs = 30_000_000_000L
        session.onAnalysisRebound(formatLabel = "YUV_420_888", resolutionModeLabel = "SMALL")
        assertEquals(33_000L, gov.warmupUntilMs)
        assertEquals(33_000_000_000L, probe.warmupUntilNs)

        // 5. Advance to 35.0s, pause and resume -> 3s warm-up (35.0s .. 38.0s)
        session.onPause()
        fakeClockNs = 35_000_000_000L
        session.onResume(nowElapsedMs = 35_000L)
        assertEquals(38_000L, gov.warmupUntilMs)
        assertEquals("resume", gov.warmupTriggerReason)
        assertEquals(38_000_000_000L, probe.warmupUntilNs)

        // 6. Record 10 warm-up samples during 35.1s .. 36.0s with spiked age (250 ms) and spiked inference (90 ms)
        for (i in 1..10) {
            val sampleNs = 35_000_000_000L + i * 100_000_000L
            fakeClockNs = sampleNs
            val captureNs = sampleNs - 250_000_000L // 250ms age
            val snap = FaceGeometrySnapshot.empty(timestampNs = captureNs).copy(
                inferenceDoneNs = sampleNs,
                conversionMs = 25.0f,
                pendingWaitMs = 15.0f,
                inferenceMs = 90.0f
            )
            probe.recordTrackerSnapshot(snap)
            val fs = FaceState(
                timestampNs = sampleNs,
                poseTimestampNs = captureNs,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true,
                conversionMs = 25.0f,
                pendingWaitMs = 15.0f,
                inferenceMs = 90.0f
            )
            probe.recordRenderFrame(renderTimestampNs = sampleNs, faceState = fs)
        }

        probe.refreshMetrics(36_000_000_000L)
        assertEquals(
            "Warm-up samples must be excluded from ageSampleCount / ageP50 / ageP95",
            0,
            probe.ageSampleCount()
        )
        assertEquals(0.0f, probe.ageP50Ms, 0.001f)
        assertEquals(0.0f, probe.ageP95Ms, 0.001f)
        assertEquals(
            "Warm-up samples must be excluded from rollingInferenceMs",
            0.0f,
            probe.rollingInferenceMs,
            0.001f
        )
        assertEquals(
            "Warm-up samples must be excluded from rollingConversionMs",
            0.0f,
            probe.rollingConversionMs,
            0.001f
        )

        // 7. Record 10 steady-state samples after warm-up expires (38.1s .. 39.0s) with 40 ms age and 12 ms inference
        for (i in 1..10) {
            val sampleNs = 38_000_000_000L + i * 100_000_000L
            fakeClockNs = sampleNs
            val captureNs = sampleNs - 40_000_000L // 40ms age
            val snap = FaceGeometrySnapshot.empty(timestampNs = captureNs).copy(
                inferenceDoneNs = sampleNs,
                conversionMs = 4.0f,
                pendingWaitMs = 1.0f,
                inferenceMs = 12.0f
            )
            probe.recordTrackerSnapshot(snap)
            val fs = FaceState(
                timestampNs = sampleNs,
                poseTimestampNs = captureNs,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true,
                conversionMs = 4.0f,
                pendingWaitMs = 1.0f,
                inferenceMs = 12.0f
            )
            probe.recordRenderFrame(renderTimestampNs = sampleNs, faceState = fs)
        }

        probe.refreshMetrics(39_000_000_000L)
        assertEquals(
            "Only post-warm-up samples must be included in ageSampleCount",
            10,
            probe.ageSampleCount()
        )
        assertEquals(40.0f, probe.ageP50Ms, 0.5f)
        assertEquals(40.0f, probe.ageP95Ms, 0.5f)
        assertEquals(12.0f, probe.rollingInferenceMs, 0.2f)
        assertEquals(4.0f, probe.rollingConversionMs, 0.2f)

        // 8. Verify CSV keeps BOTH warm-up rows (warmup=true) and steady-state rows (warmup=false)
        val csvLines = probe.dumpCsv().trim().lines()
        assertEquals(1 + 20, csvLines.size) // 1 header + 10 warmup + 10 steady-state
        val firstDataCols = csvLines[1].split(",")
        val lastDataCols = csvLines.last().split(",")
        assertEquals("true", firstDataCols[17])  // warmup=true for 35.1s sample
        assertEquals("false", lastDataCols[17])  // warmup=false for 39.0s sample
    }

    /**
     * Extra Test 5(c) + Requirement 3:
     * Default `minAnalysisLongEdgePx = 320` and initial `targetAnalysisSizePx = 320` on startup
     * (even when `isLowEndDevice = true`), and sustained overload never drops `targetAnalysisSizePx`
     * below `minAnalysisLongEdgePx`.
     */
    @Test
    fun testDefaultMinAnalysisLongEdgePx320FloorAndInitialStartupSize() {
        val highEndGov = PerformanceGovernor(context = null, isLowEndDevice = false)
        assertEquals(320, highEndGov.minAnalysisLongEdgePx)
        assertEquals(320, highEndGov.targetAnalysisSizePx)

        val lowEndGov = PerformanceGovernor(context = null, isLowEndDevice = true)
        assertEquals(320, lowEndGov.minAnalysisLongEdgePx)
        assertEquals(
            "Initial targetAnalysisSizePx must respect default 320px floor even on low-end config",
            320,
            lowEndGov.targetAnalysisSizePx
        )

        var tMs = 10_000L
        highEndGov.reset(nowMs = tMs)

        fun triggerOverloadStep() {
            tMs += PerformanceGovernor.DWELL_WINDOW_MS + 100L
            for (i in 0 until 20) {
                highEndGov.recordFrameDuration(durationMs = 45.0f, nowMs = tMs)
            }
            tMs += PerformanceGovernor.DEMOTE_DELAY_MS + 100L
            highEndGov.recordFrameDuration(durationMs = 45.0f, nowMs = tMs)
        }

        // Step 1 reduces cadence 30 -> 24 -> 20 -> 15 Hz
        triggerOverloadStep()
        assertEquals(24, highEndGov.targetTrackerCadenceHz)
        assertEquals(320, highEndGov.targetAnalysisSizePx)

        triggerOverloadStep()
        assertEquals(20, highEndGov.targetTrackerCadenceHz)
        assertEquals(320, highEndGov.targetAnalysisSizePx)

        triggerOverloadStep()
        assertEquals(15, highEndGov.targetTrackerCadenceHz)
        assertEquals(320, highEndGov.targetAnalysisSizePx)

        // Because default minAnalysisLongEdgePx == 320, the next overload step must NOT reduce
        // targetAnalysisSizePx to 256; it must advance directly to Step 2 (Tier A -> Tier B)
        triggerOverloadStep()
        assertEquals(
            "targetAnalysisSizePx must never drop below default 320px floor",
            320,
            highEndGov.targetAnalysisSizePx
        )
        assertEquals(PerformanceGovernor.Tier.B, highEndGov.currentTier)
    }

    /**
     * Requirement 3 (Approved Change 3) & Requirement 4 (Approved Change 4):
     * - `isPreviewQualityReduced` (Step 5) never becomes `true` automatically under sustained overload;
     *   only via explicit debug/config flag (`allowPreviewQualityReduction = true` or `setDebugPreviewQualityReduced(true)`).
     * - Governor frame-interval input is used ONLY when `isContinuousRender == true`; ignored when `false`.
     */
    @Test
    fun testPreviewQualityReducedNeverAutomaticAndContinuousRenderModeGate() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        assertFalse(gov.allowPreviewQualityReduction)
        assertFalse(gov.isPreviewQualityReduced)

        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        // 1. Non-continuous render mode (isContinuousRender = false): 80ms frame intervals must be ignored
        tMs += PerformanceGovernor.DWELL_WINDOW_MS + 200L
        for (i in 0 until 60) {
            tMs += 80L
            gov.recordFrameDuration(
                durationMs = 80.0f,
                nowMs = tMs,
                isContinuousRender = false
            )
        }
        assertEquals(
            "Frame interval must be ignored when isContinuousRender is false",
            30,
            gov.targetTrackerCadenceHz
        )
        assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)

        // 2. Continuous render mode: drive through all degradation steps (cadence -> Tier B -> Tier C -> segments)
        // and verify isPreviewQualityReduced NEVER becomes true automatically
        for (step in 0 until 15) {
            tMs += PerformanceGovernor.DWELL_WINDOW_MS + 100L
            for (i in 0 until 20) {
                gov.recordFrameDuration(durationMs = 50.0f, nowMs = tMs, isContinuousRender = true)
            }
            tMs += PerformanceGovernor.DEMOTE_DELAY_MS + 100L
            gov.recordFrameDuration(durationMs = 50.0f, nowMs = tMs, isContinuousRender = true)
        }
        assertEquals(PerformanceGovernor.Tier.C, gov.currentTier)
        assertEquals(8, gov.occluderLatSegments)
        assertEquals(12, gov.occluderLonSegments)
        assertFalse(
            "isPreviewQualityReduced must NEVER become true automatically",
            gov.isPreviewQualityReduced
        )

        // 3. Explicit debug setter enables Step 5 preview quality reduction
        gov.setDebugPreviewQualityReduced(true)
        assertTrue(gov.isPreviewQualityReduced)
        assertEquals("debug:previewQualityReduced=true", gov.lastTransitionReason)
    }

    /**
     * Requirement 4:
     * Log the reason of every demote/promote (`signal name + value`) on the HUD cached line and in the CSV.
     */
    @Test
    fun testDemoteAndPromoteReasonsLoggedOnHudCachedLineAndCsv() {
        val monitor = ThermalMonitor(context = null)
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false, thermalMonitor = monitor)
        val probe = LatencyProbe()
        gov.onTransitionLogged = { reason ->
            probe.recordGovernorTransition(reason)
        }

        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        // 1. Trigger a frameP95 demotion
        tMs += PerformanceGovernor.DWELL_WINDOW_MS + 100L
        for (i in 0 until 20) {
            gov.recordFrameDuration(durationMs = 44.5f, nowMs = tMs, isContinuousRender = true)
        }
        tMs += PerformanceGovernor.DEMOTE_DELAY_MS + 100L
        gov.recordFrameDuration(durationMs = 44.5f, nowMs = tMs, isContinuousRender = true)

        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals("demote:frameP95=44.5ms", gov.lastTransitionReason)
        assertEquals("demote:frameP95=44.5ms", probe.governorReasonLabel)
        assertTrue(
            "HUD cached line must include governor transition reason",
            probe.cachedHudLine.contains("gov: demote:frameP95=44.5ms")
        )

        // Record a render frame and verify CSV contains the transition reason
        val renderNs = tMs * 1_000_000L
        probe.recordRenderFrame(
            renderTimestampNs = renderNs,
            faceState = FaceState(
                timestampNs = renderNs,
                poseTimestampNs = renderNs - 30_000_000L,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true
            )
        )
        val csvAfterDemote = probe.dumpCsv().trim().lines()
        assertEquals("demote:frameP95=44.5ms", csvAfterDemote.last().split(",")[18])

        // 2. Trigger a promotion after 5s dwell + 10s headroom (flush the 60-sample frameWindow with 14ms frames)
        tMs += PerformanceGovernor.DWELL_WINDOW_MS + 100L
        for (i in 0 until 60) {
            gov.recordFrameDuration(durationMs = 14.0f, nowMs = tMs, isContinuousRender = true)
        }
        tMs += PerformanceGovernor.PROMOTE_HEADROOM_MS + 100L
        gov.recordFrameDuration(durationMs = 14.0f, nowMs = tMs, isContinuousRender = true)

        assertEquals(30, gov.targetTrackerCadenceHz)
        assertEquals("promote:headroom(frameP95=14.0ms)", gov.lastTransitionReason)
        assertTrue(probe.cachedHudLine.contains("gov: promote:headroom(frameP95=14.0ms)"))

        val renderNs2 = tMs * 1_000_000L
        probe.recordRenderFrame(
            renderTimestampNs = renderNs2,
            faceState = FaceState(
                timestampNs = renderNs2,
                poseTimestampNs = renderNs2 - 25_000_000L,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true
            )
        )
        val csvAfterPromote = probe.dumpCsv().trim().lines()
        assertEquals("promote:headroom(frameP95=14.0ms)", csvAfterPromote.last().split(",")[18])
    }
}
