package com.example

import android.content.ComponentCallbacks2
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.ThermalMonitor
import com.example.engine.governor.TierPolicy
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.pkg.PackagePreloader
import com.example.engine.render.GlResourceOwner
import com.example.engine.render.TextureCache
import com.example.engine.scene.*
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/**
 * Step 6 Unit Tests — PerformanceGovernor, ThermalMonitor, TierPolicy, PackagePreloader & TextureCache
 * (README Sections 11.4, 13, 14, 15 & 21 Step 6).
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class GovernorPreloadStep6UnitTest {

    @Test
    fun testStartingTierByDeviceClass() {
        val highEndGov = PerformanceGovernor(context = null, isLowEndDevice = false)
        assertEquals("High-end device must start at Tier A", PerformanceGovernor.Tier.A, highEndGov.currentTier)
        assertEquals(30, highEndGov.targetTrackerCadenceHz)
        assertEquals(320, highEndGov.targetAnalysisSizePx)
        assertEquals(1024, highEndGov.maxTextureSize)
        assertEquals(16, highEndGov.occluderLatSegments)
        assertEquals(24, highEndGov.occluderLonSegments)

        val lowEndGov = PerformanceGovernor(context = null, isLowEndDevice = true)
        assertEquals("Low-end device must start at Tier B (never Tier C)", PerformanceGovernor.Tier.B, lowEndGov.currentTier)
        assertEquals(30, lowEndGov.targetTrackerCadenceHz)
        assertEquals("Default minAnalysisLongEdgePx floor is 320 on startup", 320, lowEndGov.targetAnalysisSizePx)
        assertEquals(512, lowEndGov.maxTextureSize)
        assertEquals(16, lowEndGov.occluderLatSegments)
        assertEquals(24, lowEndGov.occluderLonSegments)

        val customLowFloorGov = PerformanceGovernor(
            context = null,
            isLowEndDevice = true,
            minAnalysisLongEdgePx = PerformanceGovernor.ANALYSIS_SIZE_LOW_PX
        )
        assertEquals(224, customLowFloorGov.targetAnalysisSizePx)
    }

    @Test
    fun testGovernorThresholdsHysteresisAndDwellWindow() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        gov.reset(nowMs = 10_000L)

        // 1. Inside 3-second dwell window (10_000 .. 12_999 ms): no changes allowed even under high load (> 40ms)
        for (i in 0 until 30) {
            gov.recordFrameDuration(durationMs = 45.0f, nowMs = 11_000L + i * 50L)
        }
        assertEquals("No changes allowed inside 3s dwell window", 30, gov.targetTrackerCadenceHz)
        assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)

        // 2. Outside 3s dwell window (at 13_500 ms): start high load (> 40ms p95)
        gov.recordFrameDuration(durationMs = 42.0f, nowMs = 13_500L)
        // Before 1.5s (1500ms) demotion hysteresis expires: still at 30 Hz
        gov.recordFrameDuration(durationMs = 42.0f, nowMs = 14_500L)
        assertEquals("Must not degrade before 1.5s sustained high p95", 30, gov.targetTrackerCadenceHz)

        // At 15_050 ms (>= 1500ms of >40ms p95): first degradation step triggers!
        gov.recordFrameDuration(durationMs = 42.0f, nowMs = 15_050L)
        assertEquals("First degradation must reduce tracker cadence from 30Hz to 24Hz", 24, gov.targetTrackerCadenceHz)

        // 3. Anti-flip-flop: Immediately dropping frame time to 12ms inside the new 3s dwell window
        // (15_050 .. 18_049 ms) must NOT promote back
        for (i in 0 until 60) {
            gov.recordFrameDuration(durationMs = 12.0f, nowMs = 15_500L + i * 40L)
        }
        assertEquals("Must not promote inside 3s dwell window", 24, gov.targetTrackerCadenceHz)

        // 4. Outside dwell window (18_500 ms), require 3 seconds (3_000 ms) of sustained headroom (<= 34ms) to promote
        gov.recordFrameDuration(durationMs = 12.0f, nowMs = 18_500L) // starts headroom timer
        gov.recordFrameDuration(durationMs = 12.0f, nowMs = 20_500L) // only 2.0s of headroom -> not yet promoted
        assertEquals(24, gov.targetTrackerCadenceHz)

        gov.recordFrameDuration(durationMs = 12.0f, nowMs = 21_600L) // > 3s of headroom -> promotes back to 30Hz!
        assertEquals("Must promote back to 30Hz after 3s of sustained headroom", 30, gov.targetTrackerCadenceHz)
    }

    @Test
    fun testExactFiveStepDegradationPriorityOrderAnd15HzFloor() {
        val gov = PerformanceGovernor(
            context = null,
            isLowEndDevice = false,
            allowPreviewQualityReduction = false,
            minAnalysisLongEdgePx = PerformanceGovernor.ANALYSIS_SIZE_LOW_PX
        )
        var t = 10_000L
        gov.reset(nowMs = t)

        fun sustainOverloadAndTriggerStep() {
            // Advance past 5-second dwell window
            t += PerformanceGovernor.DWELL_WINDOW_MS + 100L
            for (i in 0 until 20) {
                gov.recordFrameDuration(durationMs = 45.0f, nowMs = t)
            }
            // Advance past 1.5-second demotion hysteresis
            t += PerformanceGovernor.DEMOTE_DELAY_MS + 100L
            gov.recordFrameDuration(durationMs = 45.0f, nowMs = t)
        }

        // Priority 1a: Tracker cadence 30 -> 24 -> 20 -> 15 Hz (never below 15 Hz!)
        sustainOverloadAndTriggerStep()
        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals(PerformanceGovernor.DegradationStep.TRACKER_CADENCE_AND_ANALYSIS_SIZE, gov.lastDegradationStep)

        sustainOverloadAndTriggerStep()
        assertEquals(20, gov.targetTrackerCadenceHz)

        sustainOverloadAndTriggerStep()
        assertEquals(15, gov.targetTrackerCadenceHz)

        // Priority 1b: Analysis size 320 -> 256 -> 224 (cadence stays at 15 Hz floor!)
        sustainOverloadAndTriggerStep()
        assertEquals("Tracker cadence must never drop below 15 Hz", 15, gov.targetTrackerCadenceHz)
        assertEquals(256, gov.targetAnalysisSizePx)

        sustainOverloadAndTriggerStep()
        assertEquals(15, gov.targetTrackerCadenceHz)
        assertEquals(224, gov.targetAnalysisSizePx)

        // Priority 2: Package tier A -> B -> C
        sustainOverloadAndTriggerStep()
        assertEquals(PerformanceGovernor.Tier.B, gov.currentTier)
        assertEquals(PerformanceGovernor.DegradationStep.PACKAGE_TIER_AND_BEHAVIORS, gov.lastDegradationStep)

        sustainOverloadAndTriggerStep()
        assertEquals(PerformanceGovernor.Tier.C, gov.currentTier)
        assertEquals(256, gov.maxTextureSize)

        // Priority 4: Occluder segment counts 16x24 -> 12x18 -> 8x12
        sustainOverloadAndTriggerStep()
        assertEquals(12, gov.occluderLatSegments)
        assertEquals(18, gov.occluderLonSegments)
        assertEquals(PerformanceGovernor.DegradationStep.OCCLUDER_SEGMENTS, gov.lastDegradationStep)

        sustainOverloadAndTriggerStep()
        assertEquals(8, gov.occluderLatSegments)
        assertEquals(12, gov.occluderLonSegments)

        // Priority 5: Without explicit product approval, preview quality must NOT be reduced
        sustainOverloadAndTriggerStep()
        assertFalse("Preview quality must not degrade without explicit product approval", gov.isPreviewQualityReduced)

        // Enable explicit product approval -> Priority 5 reduces preview quality last
        gov.allowPreviewQualityReduction = true
        sustainOverloadAndTriggerStep()
        assertTrue("Preview quality may reduce only last and with explicit product approval", gov.isPreviewQualityReduced)
        assertEquals(PerformanceGovernor.DegradationStep.PREVIEW_QUALITY, gov.lastDegradationStep)
    }

    @Test
    fun testThermalAndMemoryPressureIntegration() {
        val monitor = ThermalMonitor(context = null)
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false, thermalMonitor = monitor)
        var memoryTrimmed = false
        gov.onMemoryTrimRequested = { memoryTrimmed = true }

        var t = 10_000L
        gov.reset(nowMs = t)

        // Fast frame time (14ms), but thermal status becomes MODERATE
        monitor.reportThermalStatus(ThermalMonitor.THERMAL_STATUS_MODERATE)
        assertTrue(monitor.isThermalElevated())
        assertEquals(ThermalMonitor.THERMAL_STATUS_MODERATE, gov.thermalStatus)

        t += PerformanceGovernor.DWELL_WINDOW_MS + 100L
        gov.recordFrameDuration(durationMs = 14.0f, nowMs = t)
        t += PerformanceGovernor.DEMOTE_DELAY_MS + 100L
        gov.recordFrameDuration(durationMs = 14.0f, nowMs = t)

        assertEquals("Elevated thermal status must trigger governor degradation", 24, gov.targetTrackerCadenceHz)

        // Memory pressure trim callback must notify governor and trigger TextureCache trim
        monitor.reportMemoryTrim(ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW, lowMemory = true)
        assertTrue(gov.isMemoryPressured)
        assertTrue("Memory pressure must trigger cache trim callback", memoryTrimmed)
    }

    @Test
    fun testTierPolicyShowHideDisableTextureMaxSizeMinTierAndSingleTier() {
        val n0 = CompiledNode(
            index = 0, id = "lod_a_node", parentIndex = -1, tags = listOf("lod_a"), type = "model",
            anchor = "head_center", basePosition = floatArrayOf(0f, 0f, 0f),
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f), baseScale = floatArrayOf(1f, 1f, 1f),
            rotationFollow = "full", materialId = null, baseVisible = true, baseOpacity = 1f, renderOrder = 0
        )
        val n1 = CompiledNode(
            index = 1, id = "lod_c_node", parentIndex = -1, tags = listOf("lod_c"), type = "model",
            anchor = "head_center", basePosition = floatArrayOf(0f, 0f, 0f),
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f), baseScale = floatArrayOf(1f, 1f, 1f),
            rotationFollow = "full", materialId = null, baseVisible = true, baseOpacity = 1f, renderOrder = 1
        )

        val tierOverrides = mapOf(
            "A" to TierOverride("A", showNodes = setOf(0), hideNodes = setOf(1), disabledBehaviors = emptySet(), textureMaxSize = 1024),
            "B" to TierOverride("B", showNodes = setOf(0), hideNodes = setOf(1), disabledBehaviors = setOf("heavy_spring"), textureMaxSize = 512),
            "C" to TierOverride("C", showNodes = setOf(1), hideNodes = setOf(0), disabledBehaviors = setOf("heavy_spring", "OptionalMap"), textureMaxSize = 256)
        )

        val mapBehavior = BehaviorRunner.CompiledMap(
            id = "OptionalMap", inputSlotIndex = 0, inMin = 0f, inMax = 1f, outMin = 1f, outMax = 2f,
            easing = "linear", clamp = true, smoothingTau = 0.001f, initial = 1f,
            writes = listOf(BehaviorRunner.PropertyWrite(1, BehaviorRunner.PropertyType.SCALE, 0, BehaviorRunner.WriteMode.SET))
        )
        val runner = BehaviorRunner(emptyList(), listOf(mapBehavior), emptyList(), emptyList(), 2)
        val slots = PropertySlots(2)
        val sigBus = SignalBus(2, emptyList(), mapOf("s" to 0)).apply { slotValues[0] = 1f }

        // 1. Test minTier = "B": requesting Tier.C must clamp to Tier.B
        val effectWithMinTier = CompiledEffect(
            packageId = "min_tier_pkg", packageName = "MinTier", schemaVersion = "xfilter/3",
            nodes = listOf(n0, n1), materials = emptyMap(), assets = emptyMap(), lighting = null,
            signalBus = sigBus, behaviorRunner = runner, anchorResolver = AnchorResolver(emptyMap()),
            propertySlots = slots, minTier = "B", tierOverrides = tierOverrides
        )
        val resolvedClamped = TierPolicy.resolveEffectiveTier(effectWithMinTier, PerformanceGovernor.Tier.C, 1024)
        assertEquals("minTier=B must prevent demoting to C", "B", resolvedClamped.tierId)
        assertEquals(512, resolvedClamped.textureMaxSize)

        // 2. Test Tier C behavior disabling and node hiding in BehaviorRunner and SceneEvaluator
        val effectFullTiers = effectWithMinTier.copy(minTier = null, activeTier = "C")
        val tierCConfig = TierPolicy.resolveEffectiveTier(effectFullTiers, PerformanceGovernor.Tier.C, 256)
        assertEquals("C", tierCConfig.tierId)
        assertEquals(256, tierCConfig.textureMaxSize)
        assertTrue(tierCConfig.hideNodes.contains(0))
        assertTrue(tierCConfig.disabledBehaviors.contains("OptionalMap"))

        val faceState = FaceState(isVisible = true, fadeWeight = 1.0f, faceWidth = 0.2f)
        runner.step(
            dt = 0.016f,
            signalBus = sigBus,
            faceState = faceState,
            slots = slots,
            disabledBehaviors = tierCConfig.disabledBehaviors,
            hiddenNodes = tierCConfig.hideNodes
        )

        // OptionalMap is disabled in Tier C -> node 1 scale[0] stays at rest 1.0 instead of 2.0
        assertEquals(1.0f, slots.scale[1 * 3], 0.001f)
        // Node 0 is hidden in Tier C -> visible slot is 0
        assertEquals(0.0f, slots.visible[0], 0.001f)

        val evaluator = SceneEvaluator(maxNodes = 4)
        evaluator.evaluate(effectFullTiers, faceState, slots)
        assertFalse("Node 0 must be hidden in Tier C", evaluator.isNodeVisible[0])
        assertTrue("Node 1 must be visible in Tier C", evaluator.isNodeVisible[1])

        // 3. Single-tier package support
        val singleTierEffect = effectFullTiers.copy(tierOverrides = emptyMap())
        val singleResolved = TierPolicy.resolveEffectiveTier(singleTierEffect, PerformanceGovernor.Tier.B, 512)
        assertEquals("B", singleResolved.tierId)
        assertEquals(512, singleResolved.textureMaxSize)
    }

    @Test
    fun testTextureCacheLruBoundsTextureMaxSizeGlThreadOwnershipAndSafeEviction() {
        val owner = GlResourceOwner()
        val mainGlThreadId = Thread.currentThread().id
        owner.bindGlThread(mainGlThreadId)

        // Create bounded LRU with maxEntries = 3
        val cache = TextureCache(resourceOwner = owner, maxEntries = 3, maxTotalBytes = 10L * 1024L * 1024L)

        // 1. Verify tier-specific textureMaxSize downscaling dimensions
        val (w512, h512) = TextureCache.computeBoundedDimensions(2048, 1024, maxSize = 512)
        assertTrue("Width ($w512) must respect tier maxSize 512", w512 <= 512)
        assertTrue("Height ($h512) must respect tier maxSize 512", h512 <= 512)

        val (w256, h256) = TextureCache.computeBoundedDimensions(2048, 2048, maxSize = 256)
        assertEquals(256, w256)
        assertEquals(256, h256)

        // 2. Stage 4 textures on background worker and spread uploads across frames (1 per frame)
        cache.stageDecodedDimensions("tex_1.png", 1024, 1024, maxSize = 512)
        cache.stageDecodedDimensions("tex_2.png", 1024, 1024, maxSize = 512)
        cache.stageDecodedDimensions("tex_3.png", 1024, 1024, maxSize = 512)
        cache.stageDecodedDimensions("tex_4.png", 1024, 1024, maxSize = 512)
        assertEquals(4, cache.pendingUploadCount())

        // Frame 1: upload 1 texture
        assertEquals(1, cache.stepFrameUploads(maxUploadsPerFrame = 1))
        assertEquals(1, cache.size())
        assertEquals(3, cache.pendingUploadCount())

        // Frame 2 & 3: upload 2 more textures -> cache reaches maxEntries = 3
        assertEquals(1, cache.stepFrameUploads(maxUploadsPerFrame = 1))
        assertEquals(1, cache.stepFrameUploads(maxUploadsPerFrame = 1))
        assertEquals(3, cache.size())
        assertEquals(0, cache.evictionCount)

        // Access tex_1 so tex_2 becomes least-recently-used
        assertNotNull(cache.getEntry("tex_1.png", 512))

        // Frame 4: upload 4th texture -> evicts LRU entry (tex_2) and queues GL deletion!
        assertEquals(1, cache.stepFrameUploads(maxUploadsPerFrame = 1))
        assertEquals("Cache size must stay bounded at maxEntries=3", 3, cache.size())
        assertEquals(1, cache.evictionCount)
        assertNull("LRU entry tex_2 must be evicted", cache.getEntry("tex_2.png", 512))
        assertNotNull("Recently accessed tex_1 must remain in cache", cache.getEntry("tex_1.png", 512))
        assertEquals("Evicted texture must be queued in GlResourceOwner for deferred deletion", 1, owner.getPendingTextureDeletionCount())

        // Drain pending deletions on the GL thread
        val drained = owner.drainPendingDeletions()
        assertEquals(1, drained)
        assertEquals(0, owner.getPendingTextureDeletionCount())

        // 3. Verify GL-thread ownership check rejects uploads/drains from a non-GL thread
        var wrongThreadRejected = false
        val worker = Thread {
            try {
                cache.stepFrameUploads(1)
            } catch (e: IllegalStateException) {
                wrongThreadRejected = true
            }
        }
        worker.start()
        worker.join()
        assertTrue("GL uploads from a non-GL thread must be rejected", wrongThreadRejected)

        // 4. Verify static topology and UV buffer reuse across packages
        val uv1 = cache.sharedCanonicalUvs
        val uv2 = cache.sharedCanonicalUvs
        val tri1 = cache.sharedCanonicalTriangles
        val tri2 = cache.sharedCanonicalTriangles
        assertTrue("Canonical UV buffer must not be empty", uv1.isNotEmpty())
        assertTrue("Canonical triangle index buffer must not be empty", tri1.isNotEmpty())
        assertSame("Static UV buffers must be reused across packages without reallocation", uv1, uv2)
        assertSame("Static topology triangle buffers must be reused across packages without reallocation", tri1, tri2)
    }

    @Test
    fun testPreloadSequencingDistributedUploadsRollbackAndSwitchWithoutRestart() {
        val preloader = PackagePreloader()
        val owner = GlResourceOwner().apply { bindGlThread() }
        val cache = TextureCache(resourceOwner = owner, maxEntries = 16)

        // Resolve bundled reference filter directories
        val firstFilterDir = listOf(
            File("app/src/main/assets/filters/sunflower_forehead"),
            File("src/main/assets/filters/sunflower_forehead")
        ).firstOrNull { it.exists() }
        val secondFilterDir = listOf(
            File("app/src/main/assets/filters/round_glasses"),
            File("src/main/assets/filters/round_glasses")
        ).firstOrNull { it.exists() }

        assertNotNull("Reference filter directory must exist", firstFilterDir)
        assertNotNull("Second reference filter directory must exist", secondFilterDir)

        // 1. Preload first package: Validate -> Compile -> Decode
        val preloadRes1 = preloader.preloadSync("pkg_one", firstFilterDir!!, cache, tierMaxTextureSize = 512)
        assertTrue(preloadRes1 is PackagePreloader.PreloadResult.Success)
        val stages1 = preloader.getCompletedStages("pkg_one")
        assertTrue(stages1.contains(PackagePreloader.Stage.VALIDATED))
        assertTrue(stages1.contains(PackagePreloader.Stage.COMPILED))
        assertTrue(stages1.contains(PackagePreloader.Stage.DECODED))

        // Step GL preload uploads on the GL thread to complete GPU_RESOURCES_BUILT and SHADERS_WARMED
        while (preloader.getPendingGpuUploadCount() > 0 || cache.pendingUploadCount() > 0 || !preloader.shadersWarmed) {
            preloader.stepGlUploads(renderGraph = null, textureCache = cache, maxUploadsPerFrame = 1)
        }
        val finalStages1 = preloader.getCompletedStages("pkg_one")
        assertTrue(finalStages1.contains(PackagePreloader.Stage.GPU_RESOURCES_BUILT))
        assertTrue(finalStages1.contains(PackagePreloader.Stage.SHADERS_WARMED))

        // 2. Verify filter switching never restarts camera or tracker; only tracker output flags change
        val configUpdates = AtomicInteger(0)
        var lastConfig = ProviderConfig()
        val mockProvider = object : FaceGeometryProvider {
            override val modelSha256: String = "mock_sha"
            override fun start(config: ProviderConfig) { lastConfig = config }
            override fun stop() {}
            override fun submit(frame: AnalysisFrame) { frame.imageProxy?.close() }
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {}
            override fun updateConfig(config: ProviderConfig) {
                lastConfig = config
                configUpdates.incrementAndGet()
            }
        }
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val session = CameraSession(mockProvider, SignalPipeline(), gov)
        session.recordCameraBound()
        session.startProvider()
        assertEquals(1, session.cameraStartCount)
        assertEquals(1, session.trackerStartCount)

        val switch1 = preloader.switchFilter("pkg_one", firstFilterDir, currentActive = null, textureCache = cache)
        assertTrue(switch1 is PackagePreloader.SwitchResult.Activated)
        val act1 = switch1 as PackagePreloader.SwitchResult.Activated
        session.applyEffectRequirements(act1.activeEffect)

        val switch2 = preloader.switchFilter("pkg_two", secondFilterDir, currentActive = act1.activeEffect, textureCache = cache)
        assertTrue(switch2 is PackagePreloader.SwitchResult.Activated)
        val act2 = switch2 as PackagePreloader.SwitchResult.Activated
        session.applyEffectRequirements(act2.activeEffect)

        // Camera and tracker start counts must still be 1 (zero restarts!)
        assertEquals("Camera must NOT restart on filter switch", 1, session.cameraStartCount)
        assertEquals("Tracker must NOT restart on filter switch", 1, session.trackerStartCount)
        assertFalse(act2.cameraRestarted)
        assertFalse(act2.trackerRestarted)
        assertTrue("Provider output config must be updated dynamically", configUpdates.get() >= 2)

        // 3. Failure rollback: switching to an invalid/missing package must preserve the previous active filter
        // and expose an explicit failure reason
        val missingDir = File("non_existent_filter_dir_xyz")
        val failedSwitch = preloader.switchFilter("bad_pkg", missingDir, currentActive = act2.activeEffect, textureCache = cache)
        assertTrue("Failed switch must return RolledBack", failedSwitch is PackagePreloader.SwitchResult.RolledBack)
        val rolledBack = failedSwitch as PackagePreloader.SwitchResult.RolledBack
        assertSame("Previous active filter must be preserved on switch failure", act2.activeEffect, rolledBack.preservedEffect)
        assertTrue("Explicit failure reason must be exposed", rolledBack.failureReason.isNotBlank())
    }
}
