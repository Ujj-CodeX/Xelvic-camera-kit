package com.example

import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.example.camera.CameraLifecycleObserver
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.ClockOffsetCalibrator
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.pkg.PackagePreloader
import com.example.engine.render.TransparentOverlayCompositionSurface
import com.example.engine.scene.*
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.lang.reflect.Proxy
import java.util.concurrent.atomic.AtomicInteger

/**
 * Pure-JVM unit tests verifying pause/resume/stop/destroy lifecycle forwarding with a fake
 * [LifecycleOwner]:
 * 1. Short pause (`ON_PAUSE` -> `ON_RESUME`): stops frame processing & Choreographer without
 *    releasing GL resources or restarting the tracker (`trackerStartCount == 1`).
 * 2. Long pause/stop (`ON_PAUSE` -> `ON_STOP` -> `ON_START` -> `ON_RESUME`): releases GL resources
 *    on `ON_STOP`, restores once via the single idempotent entry point (`onSurfaceCreated` or
 *    lazy first draw), and after `>= 45 s` in `ON_STOP` closes the tracker and lazily restarts
 *    it on resume (`trackerStartCount == 2`).
 * 3. `onSurfaceCreated` firing after resume (EGL context lost) & duplicate `onSurfaceCreated`
 *    idempotency (never restores from both `onResume` and `onSurfaceCreated`, never double-creates
 *    or double-frees).
 * 4. Resume resets `FaceLifecycle` to `NO_FACE`, resets behavior/clip/trigger/spring state,
 *    resets `SignalPipeline` filters and velocity (preventing stale velocity extrapolation),
 *    and resets `ClockOffsetCalibrator` + `LatencyProbe` windows.
 * 5. `PackagePreloader` and `TextureCache` suspend/cancel on pause and invalidate/rebuild any
 *    GPU object created before a context loss (`contextGeneration` check).
 * 6. In-flight inference results arriving while paused are dropped, and every `ImageProxy` is
 *    closed exactly once.
 * 7. Strict 5-step dispose order:
 *    `stop_frame_delivery` -> `unbind_camerax` -> `release_gl_on_gl_thread` -> `close_tracker` -> `shutdown_executors`.
 */
class LifecyclePauseResumeUnitTest {

    private class FakeLifecycleOwner : LifecycleOwner {
        val registry = LifecycleRegistry.createUnsafe(this)
        override val lifecycle: Lifecycle
            get() = registry

        fun handleEvent(event: Lifecycle.Event) {
            registry.handleLifecycleEvent(event)
        }
    }

    private class FakeIdleAwareProvider : FaceGeometryProvider {
        override val modelSha256: String = "fake_lifecycle_sha256"
        override var isPaused: Boolean = false
            private set

        var startCount = 0
        var stopCount = 0
        var submittedCount = 0
        var droppedPendingCount = 0
        var lastConfig = ProviderConfig()
        var pendingFrame: AnalysisFrame? = null
        private var listener: ((FaceGeometrySnapshot) -> Unit)? = null

        override fun start(config: ProviderConfig) {
            startCount++
            lastConfig = config
        }

        override fun stop() {
            stopCount++
            isPaused = false
            pendingFrame?.imageProxy?.close()
            pendingFrame = null
        }

        override fun setPaused(paused: Boolean) {
            isPaused = paused
            if (paused) {
                val p = pendingFrame
                if (p != null) {
                    droppedPendingCount++
                    p.imageProxy?.close()
                    pendingFrame = null
                }
            }
        }

        override fun updateConfig(config: ProviderConfig) {
            lastConfig = config
        }

        override fun submit(frame: AnalysisFrame) {
            if (isPaused) {
                frame.imageProxy?.close()
                return
            }
            submittedCount++
            pendingFrame?.imageProxy?.close()
            pendingFrame = frame
        }

        fun consumeAndClosePendingFrame() {
            pendingFrame?.imageProxy?.close()
            pendingFrame = null
        }

        override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {
            this.listener = listener
        }

        fun emitInFlightResult(snapshot: FaceGeometrySnapshot) {
            listener?.invoke(snapshot)
        }
    }

    private fun createCountingImageProxy(
        closeCounter: AtomicInteger,
        sensorTsNs: Long = 100_000_000L
    ): ImageProxy {
        val info = Proxy.newProxyInstance(
            ImageInfo::class.java.classLoader,
            arrayOf(ImageInfo::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getTimestamp" -> sensorTsNs
                "getRotationDegrees" -> 270
                else -> null
            }
        } as ImageInfo

        return Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "close" -> {
                    closeCounter.incrementAndGet()
                    null
                }
                "getWidth" -> 640
                "getHeight" -> 360
                "getFormat" -> 35
                "getImageInfo" -> info
                else -> null
            }
        } as ImageProxy
    }

    private fun buildSyntheticEffectWithBehaviors(): CompiledEffect {
        val node = CompiledNode(
            index = 0,
            id = "prop_node",
            parentIndex = -1,
            tags = emptyList(),
            type = "sprite",
            anchor = "head_center",
            basePosition = floatArrayOf(0f, 0f, 0f),
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
            baseScale = floatArrayOf(1f, 1f, 1f),
            rotationFollow = "full",
            materialId = "mat0",
            baseVisible = true,
            baseOpacity = 1f,
            renderOrder = 0
        )
        val clip = BehaviorRunner.CompiledClip(
            id = "intro_clip",
            targetNodeIndex = 0,
            animationName = "pop",
            loop = false,
            startOn = "faceAcquired",
            speed = 1f,
            holdLast = true,
            isPlaying = false,
            currentTimeSec = 0f,
            durationSec = 1.0f
        )
        val trigger = BehaviorRunner.CompiledTrigger(
            id = "smile_trig",
            inputSlotIndex = 0,
            onEvent = null,
            risingAbove = 0.5f,
            fallingBelow = 0.2f,
            cooldownMs = 100L,
            armedInitially = true,
            onFire = listOf(BehaviorRunner.TriggerAction(action = "playClip", clipId = "intro_clip"))
        )
        val spring = BehaviorRunner.CompiledSpring(
            id = "head_spring",
            targetNodeIndex = 0,
            mode = "rotationInertia",
            stiffness = 120f,
            dampingRatio = 0.6f,
            gain = 1.0f,
            maxAngleDeg = 25f,
            lowPassTau = 0.05f
        )
        val runner = BehaviorRunner(
            clips = listOf(clip),
            maps = emptyList(),
            triggers = listOf(trigger),
            springs = listOf(spring),
            nodeCount = 1
        )
        val bus = SignalBus(
            totalSlots = 1,
            derivedSignals = emptyList(),
            slotIndexMap = mapOf("face.blend.mouthSmileLeft" to 0)
        )
        return CompiledEffect(
            packageId = "lifecycle_test_pkg",
            packageName = "Lifecycle Test",
            schemaVersion = "xfilter/3",
            nodes = listOf(node),
            materials = mapOf("mat0" to CompiledMaterial(id = "mat0", template = "unlit@1", mapAssetId = null, maskAssetId = null)),
            assets = emptyMap(),
            lighting = null,
            signalBus = bus,
            behaviorRunner = runner,
            anchorResolver = AnchorResolver(emptyMap()),
            propertySlots = PropertySlots(1)
        )
    }

    private fun createTrackingSnapshot(timestampNs: Long, tx: Float, ty: Float = 0f, tz: Float = -0.5f): FaceGeometrySnapshot {
        val mat = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            tx, ty, tz, 1f
        )
        return FaceGeometrySnapshot.create(
            timestampNs = timestampNs,
            imageWidth = 640,
            imageHeight = 360,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = FloatArray(478 * 3),
            rawPoseMatrix = mat,
            blendshapes = mapOf("mouthSmileLeft" to 0.8f),
            modelSha256 = "fake_lifecycle_sha256"
        )
    }

    @Test
    fun testShortPauseDoesNotReleaseGlOrRestartTracker() {
        val fakeOwner = FakeLifecycleOwner()
        val provider = FakeIdleAwareProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(null)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, governor, ClockOffsetCalibrator(), probe)
        val surface = TransparentOverlayCompositionSurface()
        val preloader = PackagePreloader()
        val effect = buildSyntheticEffectWithBehaviors()
        var currentClockMs = 1_000L

        surface.activeEffect = effect
        session.startProvider(ProviderConfig(outputPose = true, outputBlendshapes = true))
        session.applyEffectRequirements(effect)
        surface.onSurfaceCreated(1080, 1920)

        val observer = CameraLifecycleObserver.forSessionAndSurface(
            cameraSession = session,
            surface = surface,
            preloader = preloader,
            clockElapsedMs = { currentClockMs }
        )
        fakeOwner.lifecycle.addObserver(observer)

        // Transition ON_CREATE -> ON_START -> ON_RESUME
        fakeOwner.handleEvent(Lifecycle.Event.ON_CREATE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)

        assertFalse(session.isPaused)
        assertFalse(surface.isPaused)
        assertTrue(surface.areGlResourcesAllocated)
        assertEquals(1, surface.glCreateCount)
        assertEquals(0, surface.glReleaseCount)
        assertEquals(1, session.trackerStartCount)

        // Short pause (ON_PAUSE only, e.g., dialog or notification shade)
        currentClockMs = 2_000L
        fakeOwner.handleEvent(Lifecycle.Event.ON_PAUSE)

        assertTrue("CameraSession must be paused on ON_PAUSE", session.isPaused)
        assertTrue("Provider must be IDLE on ON_PAUSE", provider.isPaused)
        assertTrue("Surface must be paused on ON_PAUSE", surface.isPaused)
        assertTrue("Preloader must be paused on ON_PAUSE", preloader.isPaused)
        assertTrue("Short pause must NOT release GL resources", surface.areGlResourcesAllocated)
        assertEquals("Short pause must have 0 GL releases", 0, surface.glReleaseCount)
        assertEquals("Short pause must NOT close tracker", 0, provider.stopCount)

        // Resume from short pause (ON_RESUME)
        currentClockMs = 3_000L
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)

        assertFalse(session.isPaused)
        assertFalse(provider.isPaused)
        assertFalse(surface.isPaused)
        assertFalse(preloader.isPaused)
        assertTrue(surface.areGlResourcesAllocated)
        assertEquals("GL resources must still have been created only once", 1, surface.glCreateCount)
        assertEquals("Tracker must NOT be restarted across short pause", 1, session.trackerStartCount)
    }

    @Test
    fun testLongPauseReleasesGlOnStopAndRestoresOnceAndRestartsTrackerAfter45s() {
        val fakeOwner = FakeLifecycleOwner()
        val provider = FakeIdleAwareProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(null)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, governor, ClockOffsetCalibrator(), probe)
        val surface = TransparentOverlayCompositionSurface()
        var currentClockMs = 10_000L

        session.startProvider(ProviderConfig(outputPose = true, outputBlendshapes = true))
        surface.onSurfaceCreated(1080, 1920)

        val observer = CameraLifecycleObserver.forSessionAndSurface(
            cameraSession = session,
            surface = surface,
            clockElapsedMs = { currentClockMs }
        )
        fakeOwner.lifecycle.addObserver(observer)

        fakeOwner.handleEvent(Lifecycle.Event.ON_CREATE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)
        assertEquals(1, surface.glCreateCount)
        assertEquals(0, surface.glReleaseCount)
        assertEquals(1, session.trackerStartCount)

        // 1. Background stop shorter than 45 s (e.g. 15 s): releases GL on ON_STOP, keeps tracker alive
        currentClockMs = 20_000L
        fakeOwner.handleEvent(Lifecycle.Event.ON_PAUSE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_STOP)

        assertFalse("GL resources must be released on ON_STOP", surface.areGlResourcesAllocated)
        assertEquals(1, surface.glReleaseCount)
        // Redundant onStop / release call must never double-free
        surface.releaseGlResourcesOnGlThread()
        assertEquals("Double-free must be impossible", 1, surface.glReleaseCount)

        // Resume after 15 s (< 45 s): tracker was NOT closed, GL restored once on first draw
        currentClockMs = 35_000L
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)
        assertEquals("Tracker must stay alive for ON_STOP < 45 s", 0, provider.stopCount)
        assertEquals(1, session.trackerStartCount)
        assertFalse("onResume must not eagerly restore GL resources before GL thread draw/surfaceCreated", surface.areGlResourcesAllocated)

        surface.renderFrame(FaceState.EMPTY, dtSec = 0.016f)
        assertTrue("Lazy first draw in renderFrame must restore GL resources once", surface.areGlResourcesAllocated)
        assertEquals(2, surface.glCreateCount)

        // 2. Background stop longer than 45 s (e.g. 50 s): closes provider and lazily restarts on resume
        currentClockMs = 40_000L
        fakeOwner.handleEvent(Lifecycle.Event.ON_PAUSE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_STOP)
        assertEquals(2, surface.glReleaseCount)
        assertFalse(surface.areGlResourcesAllocated)

        // Advance clock by 50 s while stopped (> 45 s threshold)
        currentClockMs = 90_000L
        assertTrue(session.checkLongStopTimeout(currentClockMs))
        assertEquals("Provider must be closed after >= 45 s in ON_STOP", 1, provider.stopCount)
        assertFalse(session.isProviderStarted)
        assertEquals("trackerStartCount is still 1 before resume", 1, session.trackerStartCount)

        // Resume after long stop -> provider is lazily restarted (trackerStartCount becomes 2)
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)
        assertTrue("Provider must be restarted on resume after >45s stop", session.isProviderStarted)
        assertEquals("trackerStartCount must become 2 after >45s ON_STOP", 2, session.trackerStartCount)
    }

    @Test
    fun testDuplicateOnSurfaceCreatedAndOnSurfaceCreatedAfterResumeEglContextLost() {
        val fakeOwner = FakeLifecycleOwner()
        val provider = FakeIdleAwareProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(null)
        val session = CameraSession(provider, pipeline, governor)
        val surface = TransparentOverlayCompositionSurface()

        session.startProvider()
        val observer = CameraLifecycleObserver.forSessionAndSurface(session, surface)
        fakeOwner.lifecycle.addObserver(observer)

        fakeOwner.handleEvent(Lifecycle.Event.ON_CREATE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)

        // 1. Initial onSurfaceCreated creates GL resources once
        surface.onSurfaceCreated(1080, 1920)
        assertEquals(1, surface.glCreateCount)
        assertTrue(surface.areGlResourcesAllocated)

        // 2. Duplicate onSurfaceCreated calls on the same context generation must be skipped
        surface.onSurfaceCreated(1080, 1920)
        surface.onSurfaceCreated(1080, 1920)
        surface.renderFrame(FaceState.EMPTY, dtSec = 0.016f)
        assertEquals("Duplicate onSurfaceCreated and subsequent renderFrame must not re-create GL resources", 1, surface.glCreateCount)

        // 3. Simulate ON_PAUSE -> ON_STOP -> ON_START -> ON_RESUME where EGL context was lost,
        //    so GLSurfaceView fires onSurfaceCreated after resume followed by renderFrame.
        fakeOwner.handleEvent(Lifecycle.Event.ON_PAUSE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_STOP)
        assertFalse(surface.areGlResourcesAllocated)
        assertEquals(1, surface.glReleaseCount)

        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)
        // Must NOT have restored inside onResume
        assertEquals(1, surface.glCreateCount)

        // GLSurfaceView fires onSurfaceCreated on the new EGL context, then onDrawFrame/renderFrame runs
        surface.onSurfaceCreated(1080, 1920)
        assertEquals("onSurfaceCreated after resume must restore GL resources once", 2, surface.glCreateCount)
        assertTrue(surface.areGlResourcesAllocated)

        surface.renderFrame(FaceState.EMPTY, dtSec = 0.016f)
        assertEquals("renderFrame after onSurfaceCreated must not restore a second time", 2, surface.glCreateCount)
    }

    @Test
    fun testResumeResetsLifecycleBehaviorsFiltersVelocityCalibratorAndLatencyProbe() {
        val fakeOwner = FakeLifecycleOwner()
        val provider = FakeIdleAwareProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(null)
        val calibrator = ClockOffsetCalibrator(warmupFrames = 5)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, governor, calibrator, probe)
        val surface = TransparentOverlayCompositionSurface()
        val effect = buildSyntheticEffectWithBehaviors()

        surface.activeEffect = effect
        session.startProvider()
        session.applyEffectRequirements(effect)
        surface.onSurfaceCreated(1080, 1920)

        val observer = CameraLifecycleObserver.forSessionAndSurface(session, surface)
        fakeOwner.lifecycle.addObserver(observer)

        fakeOwner.handleEvent(Lifecycle.Event.ON_CREATE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)

        // 1. Calibrate ClockOffsetCalibrator and populate LatencyProbe
        calibrator.configureSource(ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN)
        for (i in 1..6) {
            val sensorTs = i * 33_000_000L
            val arrivalTs = sensorTs + 50_000_000L
            calibrator.toElapsedRealtimeNs(sensorTs, arrivalTs)
            probe.recordCameraFrame(arrivalTs)
        }
        assertTrue("Calibrator should be calibrated after 5+ warm-up frames", calibrator.isCalibrated)

        // 2. Drive FaceLifecycle to TRACKING with non-zero velocity
        val t0 = 100_000_000L
        val t1 = 250_000_000L // past 100ms easeInMs -> TRACKING
        val t2 = 283_000_000L // 33ms later with +0.05m X motion -> non-zero X velocity
        pipeline.processSnapshot(createTrackingSnapshot(t0, tx = 0.0f), nowNs = t0)
        pipeline.processSnapshot(createTrackingSnapshot(t1, tx = 0.0f), nowNs = t1)
        val trackingState = pipeline.processSnapshot(createTrackingSnapshot(t2, tx = 0.05f), nowNs = t2)
        assertEquals(FaceLifecycle.State.TRACKING, pipeline.lifecycle.currentState)
        val velBeforePause = FloatArray(3)
        pipeline.latencyCompensator.getLinearVelocity(velBeforePause)
        assertTrue("Velocity before pause must be non-zero", kotlin.math.abs(velBeforePause[0]) > 0.1f)

        // 3. Step behaviors on activeEffect so clip is playing, trigger fired (disarmed), and spring displaced
        val stateWithRoll = trackingState.copy(rollDeg = 15f, triggeredEvent = "faceAcquired")
        surface.renderFrame(stateWithRoll, dtSec = 0.02f, renderTimeSec = 0.2f, nowMs = 200L)
        val clip = effect.behaviorRunner.clips.first()
        val trigger = effect.behaviorRunner.triggers.first()
        val spring = effect.behaviorRunner.springs.first()
        assertTrue("Clip should be playing before pause", clip.isPlaying)
        assertTrue("Clip time should have advanced before pause", clip.currentTimeSec > 0f)
        assertFalse("Trigger should have fired and disarmed before pause", trigger.isArmed)
        assertNotEquals("Spring position should be non-zero before pause", 0f, spring.pos)

        // 4. Pause and Resume via lifecycle owner
        fakeOwner.handleEvent(Lifecycle.Event.ON_PAUSE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)

        // 5. Verify all required resets on resume:
        assertEquals("FaceLifecycle must reset to NO_FACE on resume", FaceLifecycle.State.NO_FACE, pipeline.lifecycle.currentState)
        assertEquals(FaceLifecycle.State.NO_FACE, pipeline.getLatestState().lifecycle)

        val velAfterResume = FloatArray(3)
        val angVelAfterResume = FloatArray(3)
        pipeline.latencyCompensator.getLinearVelocity(velAfterResume)
        pipeline.latencyCompensator.getAngularVelocity(angVelAfterResume)
        assertArrayEquals("Linear velocity must be reset to zero on resume", floatArrayOf(0f, 0f, 0f), velAfterResume, 1e-6f)
        assertArrayEquals("Angular velocity must be reset to zero on resume", floatArrayOf(0f, 0f, 0f), angVelAfterResume, 1e-6f)

        val predictedAfterResume = pipeline.consumeLatestPredictedState(renderElapsedNs = t2 + 40_000_000L, predictionEnabled = true)
        assertEquals(FaceLifecycle.State.NO_FACE, predictedAfterResume.lifecycle)
        assertArrayEquals("Prediction must not extrapolate stale velocity after resume", floatArrayOf(0f, 0f, 0f), predictedAfterResume.translation, 1e-6f)

        assertFalse("Clip must be reset to stopped on resume", clip.isPlaying)
        assertEquals("Clip time must be reset to 0 on resume", 0f, clip.currentTimeSec, 1e-6f)
        assertTrue("Trigger must be reset to armedInitially on resume", trigger.isArmed)
        assertEquals("Spring position must be reset to 0 on resume", 0f, spring.pos, 1e-6f)
        assertEquals("Spring velocity must be reset to 0 on resume", 0f, spring.vel, 1e-6f)

        assertFalse("ClockOffsetCalibrator must be reset on resume", calibrator.isCalibrated)
        probe.refreshMetrics(nowNs = t2 + 50_000_000L)
        assertEquals("LatencyProbe cameraFps window must be reset on resume", 0f, probe.cameraFps, 1e-5f)
    }

    @Test
    fun testInFlightResultDroppedWhilePausedAndEveryImageProxyClosedOnce() {
        val provider = FakeIdleAwareProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(null)
        val session = CameraSession(provider, pipeline, governor)
        session.startProvider()

        // 1. Submit frame 1 while active -> accepted and held as pending in FakeIdleAwareProvider
        val closeCount1 = AtomicInteger(0)
        val proxy1 = createCountingImageProxy(closeCount1, sensorTsNs = 100_000_000L)
        assertTrue(session.submitAnalysisFrame(proxy1, timestampNs = 100_000_000L))
        assertEquals("Pending frame not closed until processed or paused", 0, closeCount1.get())

        // 2. Pause session -> pending frame 1 in provider must be immediately closed once and cleared
        session.onPause()
        assertTrue(session.isPaused)
        assertTrue(provider.isPaused)
        assertEquals("Pending frame must be closed exactly once when entering IDLE on pause", 1, closeCount1.get())

        // 3. Submit frame 2 while paused -> rejected and closed once immediately
        val closeCount2 = AtomicInteger(0)
        val proxy2 = createCountingImageProxy(closeCount2, sensorTsNs = 150_000_000L)
        assertFalse(session.submitAnalysisFrame(proxy2, timestampNs = 150_000_000L))
        assertEquals("Frame submitted while paused must be closed exactly once", 1, closeCount2.get())

        // 4. Emit an in-flight inference result while paused -> must be dropped and never update SignalPipeline
        provider.emitInFlightResult(createTrackingSnapshot(timestampNs = 120_000_000L, tx = 0.03f))
        assertEquals("In-flight result arriving while paused must be dropped", 1, session.droppedWhilePausedCount)
        assertEquals("SignalPipeline must remain in NO_FACE when in-flight result is dropped", FaceLifecycle.State.NO_FACE, pipeline.lifecycle.currentState)

        // 5. Resume and submit frame 3 -> accepted and closed once when consumed
        session.onResume()
        val closeCount3 = AtomicInteger(0)
        val proxy3 = createCountingImageProxy(closeCount3, sensorTsNs = 300_000_000L)
        assertTrue(session.submitAnalysisFrame(proxy3, timestampNs = 300_000_000L))
        provider.consumeAndClosePendingFrame()
        assertEquals("Processed frame after resume must be closed exactly once", 1, closeCount3.get())
    }

    @Test
    fun testPreloaderAndTextureCacheSuspendOnPauseAndRebuildAfterContextGenerationChange() {
        val surface = TransparentOverlayCompositionSurface()
        val texCache = surface.renderGraph.textures
        val preloader = PackagePreloader()

        surface.onSurfaceCreated(1080, 1920)
        val gen0 = surface.renderGraph.resourceOwner.contextGeneration

        // Stage and upload a synthetic texture in generation 0
        texCache.stageDecodedDimensions("test_asset.png", srcWidth = 256, srcHeight = 256, maxSize = 1024)
        assertEquals(1, texCache.stepFrameUploads(maxUploadsPerFrame = 1))
        val entryGen0 = texCache.getEntry("test_asset.png", maxSize = 1024)
        assertNotNull(entryGen0)
        assertEquals(gen0, entryGen0!!.contextGeneration)

        // Pause -> staging and frame uploads must be suspended/cancelled
        preloader.onPause(texCache)
        assertTrue(preloader.isPaused)
        assertTrue(texCache.isPaused)
        texCache.stageDecodedDimensions("paused_asset.png", srcWidth = 128, srcHeight = 128, maxSize = 1024)
        assertEquals("Staging while paused must be ignored", 0, texCache.pendingUploadCount())
        assertEquals("stepGlUploads while paused must do 0 work", 0, preloader.stepGlUploads(surface.renderGraph, texCache))

        // Simulate context loss (generation increments) -> GPU object from gen0 must be treated as invalid
        surface.renderGraph.resourceOwner.onContextLost()
        val gen1 = surface.renderGraph.resourceOwner.contextGeneration
        assertTrue(gen1 > gen0)
        assertNull("Texture created before context loss must be treated as invalid", texCache.getEntry("test_asset.png", maxSize = 1024))
        assertEquals(0, texCache.size())

        // Resume and rebuild on demand in generation 1
        preloader.onResume(texCache)
        texCache.stageDecodedDimensions("test_asset.png", srcWidth = 256, srcHeight = 256, maxSize = 1024)
        assertEquals(1, texCache.stepFrameUploads(maxUploadsPerFrame = 1))
        val entryGen1 = texCache.getEntry("test_asset.png", maxSize = 1024)
        assertNotNull(entryGen1)
        assertEquals(gen1, entryGen1!!.contextGeneration)
        assertNotEquals(entryGen0.textureId, entryGen1.textureId)
    }

    @Test
    fun testStrictFiveStepDisposeOrderOnDestroy() {
        val fakeOwner = FakeLifecycleOwner()
        val provider = FakeIdleAwareProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(null)
        val session = CameraSession(provider, pipeline, governor)
        val surface = TransparentOverlayCompositionSurface()
        val preloader = PackagePreloader()
        val detailedLog = ArrayList<String>()

        session.startProvider()
        surface.onSurfaceCreated(1080, 1920)

        val observer = CameraLifecycleObserver.forSessionAndSurface(
            cameraSession = session,
            surface = surface,
            preloader = preloader,
            onUnbindCameraX = { detailedLog.add("camerax_unbound") },
            onShutdownExecutors = { detailedLog.add("executors_shutdown") },
            actionLog = detailedLog
        )
        fakeOwner.lifecycle.addObserver(observer)

        fakeOwner.handleEvent(Lifecycle.Event.ON_CREATE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_START)
        fakeOwner.handleEvent(Lifecycle.Event.ON_RESUME)
        fakeOwner.handleEvent(Lifecycle.Event.ON_PAUSE)
        fakeOwner.handleEvent(Lifecycle.Event.ON_STOP)
        detailedLog.clear()

        // Trigger ON_DESTROY (and verify calling dispose() again in DisposableEffect.onDispose is idempotent)
        fakeOwner.handleEvent(Lifecycle.Event.ON_DESTROY)
        observer.dispose()

        assertEquals(
            listOf(
                CameraLifecycleObserver.DisposeStep.STOP_FRAME_DELIVERY,
                CameraLifecycleObserver.DisposeStep.UNBIND_CAMERAX,
                CameraLifecycleObserver.DisposeStep.RELEASE_GL_ON_GL_THREAD,
                CameraLifecycleObserver.DisposeStep.CLOSE_TRACKER,
                CameraLifecycleObserver.DisposeStep.SHUTDOWN_EXECUTORS
            ),
            observer.executedDisposeSteps
        )

        val expectedOrder = listOf(
            "stop_frame_delivery",
            "unbind_camerax",
            "camerax_unbound",
            "release_gl_on_gl_thread",
            "close_tracker",
            "shutdown_executors",
            "executors_shutdown"
        )
        assertEquals(expectedOrder, detailedLog)
        assertTrue(session.isPaused)
        assertFalse(surface.areGlResourcesAllocated)
        assertEquals(1, provider.stopCount)
        assertTrue(preloader.isShutdown)
    }
}
