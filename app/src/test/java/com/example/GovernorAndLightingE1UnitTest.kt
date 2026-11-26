package com.example

import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.pkg.Compiler
import com.example.engine.render.MaterialRegistry
import com.example.engine.signal.FaceState
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import kotlin.math.sqrt

/**
 * Pure-JVM unit tests for Chunk E1:
 * 1. Governor startup tier by device class: low-end starts at Tier B (never Tier C), high-end starts at Tier A.
 * 2. Separate configurable HEADROOM_P95_MS (34.0 ms) strictly below MAX_P95_MS (40.0 ms), and p95 values
 *    in the deadband (34.0 ms < p95 <= 40.0 ms) neither promote nor demote.
 * 3. Promotion timing: after warm-up, promotes one step after 3 s of sustained headroom with a 3 s dwell.
 * 4. Promotion backoff after a frame-p95 demotion: waits 3 s, then 6 s, 12 s, 30 s on repeated demotions;
 *    resets the backoff after 60 s on one tier.
 * 5. Borderline load flip-flop test: p95 alternating 41 ms at Tier A and 30 ms at Tier B settles on Tier B
 *    instead of oscillating every few seconds.
 * 6. Filter-switch hitch inside warm-up does not demote, and resultAge alone never demotes.
 * 7. HUD governor line displays transition reason, time-to-promotion, lowEnd=true/false, and current backoff.
 * 8. Multi-light Lambert compilation & shading: real lighting blocks from `sunflower_forehead` and
 *    `periwinkle_forehead` assets (2 lights + ambient), empty-lights case (ambient only), and golden
 *    1-light bit-for-bit equality with the legacy single-light formula.
 */
class GovernorAndLightingE1UnitTest {

    private fun resolveFilterManifestFile(filterId: String): File {
        val candidates = listOf(
            File("app/src/main/assets/filters/$filterId/manifest.json"),
            File("src/main/assets/filters/$filterId/manifest.json")
        )
        return candidates.firstOrNull { it.exists() }
            ?: throw AssertionError("Could not locate manifest.json for $filterId in ${candidates.map { it.absolutePath }}")
    }

    /**
     * Helper to flush the 60-sample frameWindow with a constant frame duration at `nowMs`.
     */
    private fun feedWindowAt(gov: PerformanceGovernor, durationMs: Float, nowMs: Long, count: Int = 60) {
        for (i in 0 until count) {
            gov.recordFrameDuration(
                durationMs = durationMs,
                nowMs = nowMs,
                isContinuousRender = true
            )
        }
    }

    @Test
    fun testStartupTierForLowEndIsTierBAndHighEndIsTierA() {
        val lowEndGov = PerformanceGovernor(context = null, isLowEndDevice = true)
        assertEquals(
            "Low-end device must start at Tier B (never Tier C)",
            PerformanceGovernor.Tier.B,
            lowEndGov.currentTier
        )
        assertNotEquals(PerformanceGovernor.Tier.C, lowEndGov.currentTier)
        assertEquals(PerformanceGovernor.TEXTURE_MAX_MED, lowEndGov.maxTextureSize)
        assertEquals(30, lowEndGov.targetTrackerCadenceHz)
        assertEquals(320, lowEndGov.targetAnalysisSizePx)
        assertEquals(16, lowEndGov.occluderLatSegments)
        assertEquals(24, lowEndGov.occluderLonSegments)

        val highEndGov = PerformanceGovernor(context = null, isLowEndDevice = false)
        assertEquals(
            "High-end device must start at Tier A",
            PerformanceGovernor.Tier.A,
            highEndGov.currentTier
        )
        assertEquals(PerformanceGovernor.TEXTURE_MAX_HIGH, highEndGov.maxTextureSize)
        assertEquals(30, highEndGov.targetTrackerCadenceHz)
        assertEquals(320, highEndGov.targetAnalysisSizePx)
    }

    @Test
    fun testHeadroomP95StrictlyBelowMaxFrameP95AndDeadbandNeitherPromotesNorDemotes() {
        assertEquals(34.0f, PerformanceGovernor.HEADROOM_P95_MS, 0.001f)
        assertEquals(40.0f, PerformanceGovernor.MAX_P95_MS, 0.001f)
        assertTrue(
            "HEADROOM_P95_MS (${PerformanceGovernor.HEADROOM_P95_MS}) must be strictly below MAX_P95_MS (${PerformanceGovernor.MAX_P95_MS})",
            PerformanceGovernor.HEADROOM_P95_MS < PerformanceGovernor.MAX_P95_MS
        )

        val lowEndGov = PerformanceGovernor(context = null, isLowEndDevice = true)
        val highEndGov = PerformanceGovernor(context = null, isLowEndDevice = false)
        assertTrue(lowEndGov.headroomFrameP95Ms < lowEndGov.maxFrameP95Ms)

        var tMs = 10_000L
        lowEndGov.reset(nowMs = tMs)
        highEndGov.reset(nowMs = tMs)

        // Feed p95 values strictly inside the deadband (34.0 ms < p95 <= 40.0 ms) for 40 seconds
        val deadbandSamples = floatArrayOf(34.5f, 36.0f, 37.5f, 39.2f, 40.0f)
        for (step in 0 until 800) {
            tMs += 50L
            val dur = deadbandSamples[step % deadbandSamples.size]
            lowEndGov.recordFrameDuration(durationMs = dur, nowMs = tMs, isContinuousRender = true)
            highEndGov.recordFrameDuration(durationMs = dur, nowMs = tMs, isContinuousRender = true)
        }

        assertEquals(
            "Deadband p95 (34..40 ms) must NOT promote low-end governor from Tier B to Tier A",
            PerformanceGovernor.Tier.B,
            lowEndGov.currentTier
        )
        assertEquals("none", lowEndGov.lastTransitionReason)

        assertEquals(
            "Deadband p95 (34..40 ms) must NOT demote high-end governor from Tier A",
            PerformanceGovernor.Tier.A,
            highEndGov.currentTier
        )
        assertEquals(30, highEndGov.targetTrackerCadenceHz)
        assertEquals("none", highEndGov.lastTransitionReason)
    }

    @Test
    fun testPromotionTimingAfterWarmup3sHeadroomAnd3sDwell() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = true)
        assertEquals(3_000L, PerformanceGovernor.PROMOTE_HEADROOM_MS)
        assertEquals(3_000L, PerformanceGovernor.DWELL_WINDOW_MS)
        assertEquals(3_000L, gov.promoteHeadroomMs)
        assertEquals(3_000L, gov.dwellWindowMs)

        var tMs = 10_000L
        gov.reset(nowMs = tMs)
        assertEquals(PerformanceGovernor.Tier.B, gov.currentTier)

        // Start a 3s warm-up window (10_000 .. 13_000 ms)
        gov.startWarmup(nowMs = tMs, durationMs = 3_000L, reason = "startup")

        // Feed healthy 20ms frames throughout the 3s warm-up (10_000 .. 12_950 ms)
        while (tMs < 12_950L) {
            tMs += 50L
            gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        }
        assertEquals(
            "Must NOT promote while warm-up is still active",
            PerformanceGovernor.Tier.B,
            gov.currentTier
        )

        // First post-warmup frame at t = 13_000 ms starts the 3s headroom timer (and 3s dwell from 10_000 ms is satisfied)
        tMs = 13_000L
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        assertEquals(PerformanceGovernor.Tier.B, gov.currentTier)

        // At t = 15_900 ms (2.9 s of post-warmup headroom): still Tier B
        while (tMs < 15_900L) {
            tMs += 50L
            gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        }
        assertEquals(
            "Must NOT promote before 3s of post-warmup sustained headroom",
            PerformanceGovernor.Tier.B,
            gov.currentTier
        )

        // At t = 16_000 ms (3.0 s of post-warmup sustained headroom): promotes Tier B -> Tier A!
        tMs = 16_000L
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        assertEquals(
            "Must promote from Tier B to Tier A after 3s of post-warmup sustained headroom",
            PerformanceGovernor.Tier.A,
            gov.currentTier
        )
        assertEquals(PerformanceGovernor.TEXTURE_MAX_HIGH, gov.maxTextureSize)
        assertTrue(gov.lastTransitionReason.startsWith("promote:headroom"))
    }

    @Test
    fun testPromotionBackoffSchedule3s6s12s30sAndResetAfter60sOnOneTier() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        fun triggerFrameP95Demotion() {
            tMs += gov.dwellWindowMs + 100L
            feedWindowAt(gov, durationMs = 44.0f, nowMs = tMs)
            tMs += gov.demoteDelayMs + 50L
            gov.recordFrameDuration(durationMs = 44.0f, nowMs = tMs, isContinuousRender = true)
        }

        // 1st frame-p95 demotion (30 Hz -> 24 Hz): backoff must be 3s (3_000 ms)
        triggerFrameP95Demotion()
        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals(1, gov.frameP95DemotionCount)
        assertEquals(3_000L, gov.currentPromotionBackoffMs)

        // Promote back to 30 Hz: waits 3s dwell + 3s headroom
        tMs += gov.dwellWindowMs + 100L
        feedWindowAt(gov, durationMs = 20.0f, nowMs = tMs) // starts headroom timer at tMs
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs + 2_900L, isContinuousRender = true)
        assertEquals("Must not promote at 2.9s when backoff is 3s", 24, gov.targetTrackerCadenceHz)
        tMs += 3_050L
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        assertEquals("Must promote after 3s backoff on 1st demotion", 30, gov.targetTrackerCadenceHz)

        // 2nd frame-p95 demotion: backoff must escalate to 6s (6_000 ms)
        triggerFrameP95Demotion()
        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals(2, gov.frameP95DemotionCount)
        assertEquals(6_000L, gov.currentPromotionBackoffMs)

        tMs += gov.dwellWindowMs + 100L
        feedWindowAt(gov, durationMs = 20.0f, nowMs = tMs) // starts headroom timer at tMs
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs + 5_500L, isContinuousRender = true)
        assertEquals("Must not promote at 5.5s when backoff is 6s", 24, gov.targetTrackerCadenceHz)
        tMs += 6_050L
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        assertEquals("Must promote after 6s backoff on 2nd demotion", 30, gov.targetTrackerCadenceHz)

        // 3rd frame-p95 demotion: backoff must escalate to 12s (12_000 ms)
        triggerFrameP95Demotion()
        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals(3, gov.frameP95DemotionCount)
        assertEquals(12_000L, gov.currentPromotionBackoffMs)

        tMs += gov.dwellWindowMs + 100L
        feedWindowAt(gov, durationMs = 20.0f, nowMs = tMs) // starts headroom timer at tMs
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs + 11_500L, isContinuousRender = true)
        assertEquals("Must not promote at 11.5s when backoff is 12s", 24, gov.targetTrackerCadenceHz)
        tMs += 12_050L
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = tMs, isContinuousRender = true)
        assertEquals("Must promote after 12s backoff on 3rd demotion", 30, gov.targetTrackerCadenceHz)

        // 4th frame-p95 demotion: backoff must escalate to 30s (30_000 ms)
        triggerFrameP95Demotion()
        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals(4, gov.frameP95DemotionCount)
        assertEquals(30_000L, gov.currentPromotionBackoffMs)

        // Verify staying on one tier for 60s (with deadband 36ms frames so it neither promotes nor demotes)
        // resets the backoff back to 3s (3_000 ms)!
        feedWindowAt(gov, durationMs = 36.0f, nowMs = tMs + 1_000L)
        gov.recordFrameDuration(durationMs = 36.0f, nowMs = tMs + 59_000L, isContinuousRender = true)
        assertEquals("Backoff must not reset before 60s on one tier", 30_000L, gov.currentPromotionBackoffMs)

        tMs += 60_100L
        gov.recordFrameDuration(durationMs = 36.0f, nowMs = tMs, isContinuousRender = true)
        assertEquals("Backoff must reset to 0 demotions after 60s on one tier", 0, gov.frameP95DemotionCount)
        assertEquals("Backoff wait must reset to 3s after 60s on one tier", 3_000L, gov.currentPromotionBackoffMs)
    }

    @Test
    fun testBorderlineLoad41msAtTierAAnd30msAtTierBSettlesOnTierBInsteadOfOscillating() {
        // Start a low-end governor at Tier B (or high-end governor); promote to Tier A, then apply
        // borderline load: 41.0 ms whenever at Tier A (> 40 ms threshold) and 30.0 ms whenever at Tier B (<= 34 ms headroom).
        val gov = PerformanceGovernor(context = null, isLowEndDevice = true)
        var tMs = 10_000L
        gov.reset(nowMs = tMs)
        assertEquals(PerformanceGovernor.Tier.B, gov.currentTier)

        var currentLoadedTier: PerformanceGovernor.Tier? = null
        fun stepBorderlineFrame(dtMs: Long) {
            tMs += dtMs
            val tier = gov.currentTier
            val frameMs = if (tier == PerformanceGovernor.Tier.A) 41.0f else 30.0f
            if (tier != currentLoadedTier) {
                currentLoadedTier = tier
                // Fill window so p95 immediately reflects the active tier's steady-state load (41ms at A, 30ms at B)
                feedWindowAt(gov, durationMs = frameMs, nowMs = tMs, count = 60)
            } else {
                gov.recordFrameDuration(durationMs = frameMs, nowMs = tMs, isContinuousRender = true)
            }
        }

        // Simulate 120 seconds (2,400 steps of 50 ms) of borderline load
        var tierBFramesLast45s = 0
        var totalFramesLast45s = 0
        var transitionsInLast30sWindow = 0
        var prevTier = gov.currentTier

        for (step in 1..2400) {
            stepBorderlineFrame(50L)
            if (step > 1500) { // Last 45 seconds of the 120s simulation (after backoff reaches 30s)
                totalFramesLast45s++
                if (gov.currentTier == PerformanceGovernor.Tier.B) {
                    tierBFramesLast45s++
                }
            }
            if (step > 1800) { // Final 30 seconds
                if (gov.currentTier != prevTier) {
                    transitionsInLast30sWindow++
                }
            }
            prevTier = gov.currentTier
        }

        assertEquals(
            "Promotion backoff must reach 30s under repeated borderline Tier A -> Tier B demotions",
            30_000L,
            gov.currentPromotionBackoffMs
        )
        val tierBFraction = tierBFramesLast45s.toFloat() / totalFramesLast45s.toFloat()
        assertTrue(
            "Governor must settle predominantly on Tier B (expected >= 85% of time on Tier B, got ${(tierBFraction * 100f).toInt()}%)",
            tierBFraction >= 0.85f
        )
        assertTrue(
            "Governor must not oscillate every few seconds once settled (transitions in 30s window=$transitionsInLast30sWindow)",
            transitionsInLast30sWindow <= 2
        )
    }

    @Test
    fun testFilterSwitchHitchDoesNotDemoteAndResultAgeAloneNeverDemotes() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        // Advance past initial 3s dwell window
        tMs = 14_000L
        feedWindowAt(gov, durationMs = 16.6f, nowMs = tMs)

        // Trigger filter-switch warm-up (3s) and inject a 165ms shader-compile hitch frame
        gov.startWarmup(
            nowMs = tMs,
            durationMs = PerformanceGovernor.WARMUP_DEFAULT_DURATION_MS,
            reason = "filterSwitch:periwinkle_forehead"
        )
        tMs += 50L
        gov.recordFrameDuration(
            durationMs = 165.0f,
            nowMs = tMs,
            resultAgeAtRenderMs = 260.0f,
            isContinuousRender = true
        )
        assertEquals(
            "Single >100ms frame during filter-switch warm-up must be ignored",
            1,
            gov.ignoredWarmupHitchCount
        )

        // Follow with normal 16.6ms frames for 2s inside warm-up
        for (i in 0 until 40) {
            tMs += 50L
            gov.recordFrameDuration(durationMs = 16.6f, nowMs = tMs, isContinuousRender = true)
        }
        assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)
        assertEquals(30, gov.targetTrackerCadenceHz)
        assertEquals("none", gov.lastTransitionReason)

        // Advance past warm-up and feed high resultAge (250ms) with healthy frameP95 (18ms) for 20s
        tMs = 20_000L
        for (i in 0 until 400) {
            tMs += 50L
            gov.recordFrameDuration(
                durationMs = 18.0f,
                nowMs = tMs,
                resultAgeAtRenderMs = 250.0f,
                conversionMs = 15.0f,
                inferenceMs = 35.0f,
                isContinuousRender = true
            )
        }
        assertEquals(
            "Governor must NEVER demote tier on resultAge alone",
            PerformanceGovernor.Tier.A,
            gov.currentTier
        )
        assertEquals(30, gov.targetTrackerCadenceHz)
    }

    @Test
    fun testHudGovernorLineShowsReasonTimeToPromotionLowEndAndBackoff() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = true)
        val probe = LatencyProbe()
        gov.onTransitionLogged = { reason -> probe.recordGovernorTransition(reason) }

        val t0Ms = 10_000L
        gov.reset(nowMs = t0Ms)

        // At t = 11_000 ms on low-end Tier B: dwell remaining is 2.0s, headroom remaining is 3.0s -> timeToPromotion = 3.0s
        val hudAtStartup = gov.formatHudGovernorSummary(nowMs = 11_000L)
        assertTrue("HUD summary must show lowEnd=true: $hudAtStartup", hudAtStartup.contains("lowEnd=true"))
        assertTrue("HUD summary must show backoff=3s: $hudAtStartup", hudAtStartup.contains("backoff=3s"))
        assertTrue("HUD summary must show promo=3.0s: $hudAtStartup", hudAtStartup.contains("promo=3.0s"))

        // Start accumulating headroom at t = 13_000 ms (past 3s dwell)
        feedWindowAt(gov, durationMs = 20.0f, nowMs = 13_000L)
        // At t = 14_200 ms (1.2s into 3.0s headroom): remaining time to promotion is 1.8s
        gov.recordFrameDuration(durationMs = 20.0f, nowMs = 14_200L, isContinuousRender = true)
        assertEquals(1_800L, gov.timeToPromotionMs(14_200L))

        var fakeClockNs = 14_200_000_000L
        com.example.engine.capture.FrameClock.customElapsedRealtimeSource = { fakeClockNs }
        try {
            val overlay = DebugOverlay(latencyProbe = probe).apply {
                governor = gov
            }
            val fs = FaceState(timestampNs = fakeClockNs)
            overlay.refreshCachedHudStringsIfNeeded(
                faceState = fs,
                fps = 60f,
                tier = gov.currentTier.name,
                fovYCropDeg = 50f,
                fovYViewDeg = 50f,
                force = true
            )
            val hudLine = overlay.cachedGovWarmupLine
            assertTrue("Overlay HUD line must contain lowEnd=true: $hudLine", hudLine.contains("lowEnd=true"))
            assertTrue("Overlay HUD line must contain backoff=3s: $hudLine", hudLine.contains("backoff=3s"))
            assertTrue("Overlay HUD line must contain promo=1.8s: $hudLine", hudLine.contains("promo=1.8s"))

            // Complete promotion to Tier A at t = 16_000 ms -> promo=max
            gov.recordFrameDuration(durationMs = 20.0f, nowMs = 16_000L, isContinuousRender = true)
            assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)
            assertEquals(-1L, gov.timeToPromotionMs(16_000L))

            fakeClockNs = 17_000_000_000L
            overlay.refreshCachedHudStringsIfNeeded(
                faceState = fs.copy(timestampNs = fakeClockNs),
                fps = 60f,
                tier = gov.currentTier.name,
                fovYCropDeg = 50f,
                fovYViewDeg = 50f,
                force = true
            )
            val hudAfterPromote = overlay.cachedGovWarmupLine
            assertTrue(
                "Overlay HUD line must show promote reason: $hudAfterPromote",
                hudAfterPromote.contains("promote:headroom(frameP95=20.0ms)")
            )
            assertTrue(
                "Overlay HUD line must show promo=max when at Tier A max quality: $hudAfterPromote",
                hudAfterPromote.contains("promo=max")
            )
        } finally {
            com.example.engine.capture.FrameClock.customElapsedRealtimeSource = null
        }
    }

    @Test
    fun testRealPackageLightingSunflowerAndPeriwinklePlusEmptyLightsAnd1LightGoldenEquality() {
        // 1. Read the real lighting blocks from sunflower_forehead and periwinkle_forehead manifests
        val sunflowerManifestText = resolveFilterManifestFile("sunflower_forehead").readText()
        val periwinkleManifestText = resolveFilterManifestFile("periwinkle_forehead").readText()

        val sunflowerLighting = Compiler.compileLightingFromJson(sunflowerManifestText)
        val periwinkleLighting = Compiler.compileLightingFromJson(periwinkleManifestText)

        for ((pkgName, compiled) in listOf(
            "sunflower_forehead" to sunflowerLighting,
            "periwinkle_forehead" to periwinkleLighting
        )) {
            assertEquals("$pkgName must compile both directional lights", 2, compiled.lightCount)
            assertArrayEquals(
                "$pkgName ambient",
                floatArrayOf(0.3f, 0.27f, 0.2f),
                compiled.ambient,
                1e-6f
            )
            // Light 0 (key light)
            assertArrayEquals(
                "$pkgName light[0] dirToLight",
                floatArrayOf(-0.35f, 0.65f, 0.68f),
                compiled.lightDirs.sliceArray(0..2),
                1e-6f
            )
            assertArrayEquals(
                "$pkgName light[0] color",
                floatArrayOf(1.0f, 0.91f, 0.72f),
                compiled.lightColors.sliceArray(0..2),
                1e-6f
            )
            assertEquals(0.82f, compiled.lightIntensities[0], 1e-6f)

            // Light 1 (fill light)
            assertArrayEquals(
                "$pkgName light[1] dirToLight",
                floatArrayOf(0.58f, 0.18f, 0.8f),
                compiled.lightDirs.sliceArray(3..5),
                1e-6f
            )
            assertArrayEquals(
                "$pkgName light[1] color",
                floatArrayOf(1.0f, 0.76f, 0.42f),
                compiled.lightColors.sliceArray(3..5),
                1e-6f
            )
            assertEquals(0.28f, compiled.lightIntensities[1], 1e-6f)
        }

        // 2. Empty-lights case (ambient only): "lights": []
        val emptyLightsJson = """
            {
              "lighting": {
                "shading": "Lambert",
                "ambient": [0.3, 0.27, 0.2],
                "lights": []
              }
            }
        """.trimIndent()
        val emptyLighting = Compiler.compileLightingFromJson(emptyLightsJson)
        assertEquals("Empty lights array must compile lightCount = 0", 0, emptyLighting.lightCount)

        // 3. 1-light legacy case (only light[0] of sunflower_forehead)
        val oneLightJson = """
            {
              "lighting": {
                "shading": "Lambert",
                "ambient": [0.3, 0.27, 0.2],
                "lights": [
                  {
                    "dirToLight": [-0.35, 0.65, 0.68],
                    "color": [1.0, 0.91, 0.72],
                    "intensity": 0.82
                  }
                ]
              }
            }
        """.trimIndent()
        val oneLightCompiled = Compiler.compileLightingFromJson(oneLightJson)
        assertEquals(1, oneLightCompiled.lightCount)

        // 4. Golden test grid of surface normals and texture colors:
        //    (a) Verify 0-light equals ambient * tex
        //    (b) Verify 1-light is bit-for-bit (0.0f delta) identical to legacy 1-light formula
        //    (c) Verify 2-light matches package formula C = tex * clamp(ambient + sum_k(intensity_k * color_k * max(dot(N, L_k), 0)), 0, 1)
        val testNormals = arrayOf(
            floatArrayOf(0f, 0f, 1f),
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(-1f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f),
            floatArrayOf(0f, -1f, 0f),
            floatArrayOf(0.6f, 0.2f, 0.77f),
            floatArrayOf(-0.4f, 0.7f, 0.59f),
            floatArrayOf(0f, 0f, -1f)
        )
        val testTextures = arrayOf(
            floatArrayOf(1.0f, 1.0f, 1.0f),
            floatArrayOf(0.95f, 0.78f, 0.12f), // sunflower gold
            floatArrayOf(0.68f, 0.72f, 0.96f)  // periwinkle blue-violet
        )

        for (n in testNormals) {
            for (tex in testTextures) {
                // (a) Empty-lights ambient-only verification
                val ambOnly = MaterialRegistry.evaluateLambertColor(tex, n, emptyLighting)
                assertEquals(tex[0] * 0.3f, ambOnly[0], 1e-6f)
                assertEquals(tex[1] * 0.27f, ambOnly[1], 1e-6f)
                assertEquals(tex[2] * 0.2f, ambOnly[2], 1e-6f)

                // (b) Golden 1-light equality (0.0f delta vs legacy single-light formula)
                val singleNew = MaterialRegistry.evaluateLambertColor(tex, n, oneLightCompiled)
                val singleLegacy = MaterialRegistry.evaluateLegacySingleLightLambertColor(
                    texRgb = tex,
                    normal = n,
                    ambient = oneLightCompiled.ambient,
                    dirToLight = oneLightCompiled.dirToLight,
                    lightColor = oneLightCompiled.lightColor,
                    lightIntensity = oneLightCompiled.lightIntensity
                )
                assertArrayEquals(
                    "1-light output must be bit-for-bit identical to legacy 1-light formula",
                    singleLegacy,
                    singleNew,
                    0.0f
                )

                // (c) Golden 2-light verification against explicit package formula
                val twoLightOut = MaterialRegistry.evaluateLambertColor(tex, n, sunflowerLighting)
                val nLen = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
                val nx = n[0] / nLen
                val ny = n[1] / nLen
                val nz = n[2] / nLen

                val l0Len = sqrt((-0.35f) * (-0.35f) + 0.65f * 0.65f + 0.68f * 0.68f)
                val ndotl0 = maxOf(nx * (-0.35f / l0Len) + ny * (0.65f / l0Len) + nz * (0.68f / l0Len), 0f)

                val l1Len = sqrt(0.58f * 0.58f + 0.18f * 0.18f + 0.8f * 0.8f)
                val ndotl1 = maxOf(nx * (0.58f / l1Len) + ny * (0.18f / l1Len) + nz * (0.8f / l1Len), 0f)

                val expectedR = tex[0] * (0.3f + 0.82f * 1.0f * ndotl0 + 0.28f * 1.0f * ndotl1).coerceIn(0f, 1f)
                val expectedG = tex[1] * (0.27f + 0.82f * 0.91f * ndotl0 + 0.28f * 0.76f * ndotl1).coerceIn(0f, 1f)
                val expectedB = tex[2] * (0.2f + 0.82f * 0.72f * ndotl0 + 0.28f * 0.42f * ndotl1).coerceIn(0f, 1f)

                assertArrayEquals(
                    "2-light Lambert evaluation must match package lighting formula",
                    floatArrayOf(expectedR, expectedG, expectedB),
                    twoLightOut,
                    1e-6f
                )
            }
        }

        // Verify that for normal = (+1, 0, 0) (where light[0] has dot < 0 and light[1] has dot > 0),
        // the 2-light shader produces strictly brighter illumination than the 1-light shader
        val rightNormal = floatArrayOf(1f, 0f, 0f)
        val whiteTex = floatArrayOf(1f, 1f, 1f)
        val oneLightRight = MaterialRegistry.evaluateLambertColor(whiteTex, rightNormal, oneLightCompiled)
        val twoLightRight = MaterialRegistry.evaluateLambertColor(whiteTex, rightNormal, sunflowerLighting)
        assertTrue(
            "Second fill light (+0.58, +0.18, +0.80) must illuminate +X normals where key light (-0.35, +0.65, +0.68) is back-facing",
            twoLightRight[0] > oneLightRight[0] &&
                twoLightRight[1] > oneLightRight[1] &&
                twoLightRight[2] > oneLightRight[2]
        )
    }
}
