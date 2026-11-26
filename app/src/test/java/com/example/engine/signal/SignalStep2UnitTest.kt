package com.example.engine.signal

import com.example.engine.debug.ReplayHarness
import com.example.engine.perception.FaceGeometrySnapshot
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Unit tests for Step 2 — Signal Layer, FacePoseAdapter and FaceState (README Section 9 & 21).
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SignalStep2UnitTest {

    @Test
    fun testPoseMatrixDecompositionAndRecomposition() {
        // Construct 4x4 transform with known translation, 45 degree yaw, and scale
        val yawRad = Math.toRadians(45.0).toFloat()
        val cosY = kotlin.math.cos(yawRad)
        val sinY = kotlin.math.sin(yawRad)

        val originalMatrix = FloatArray(16) { 0f }
        // Scale 1.2, 1.2, 1.2
        originalMatrix[0] = cosY * 1.2f
        originalMatrix[2] = -sinY * 1.2f
        originalMatrix[5] = 1.2f
        originalMatrix[8] = sinY * 1.2f
        originalMatrix[10] = cosY * 1.2f
        // Translation [0.05, -0.02, -0.45]
        originalMatrix[12] = 0.05f
        originalMatrix[13] = -0.02f
        originalMatrix[14] = -0.45f
        originalMatrix[15] = 1.0f

        val decomposed = PoseDecomposer.decompose(originalMatrix)

        // Verify translation
        assertEquals(0.05f, decomposed.translation[0], 0.001f)
        assertEquals(-0.02f, decomposed.translation[1], 0.001f)
        assertEquals(-0.45f, decomposed.translation[2], 0.001f)

        // Verify scale
        assertEquals(1.2f, decomposed.scale[0], 0.001f)
        assertEquals(1.2f, decomposed.scale[1], 0.001f)
        assertEquals(1.2f, decomposed.scale[2], 0.001f)

        // Verify yaw angle ~ 45 degrees
        assertEquals(45f, decomposed.yawDeg, 0.5f)

        // Verify quaternion is normalized
        val q = decomposed.rotation
        val qLen = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        assertEquals(1.0f, qLen, 0.001f)

        // Recompose and verify matrix match
        val recomposed = PoseDecomposer.recompose(decomposed.translation, decomposed.rotation, decomposed.scale)
        for (i in 0 until 16) {
            assertEquals("Matrix mismatch at index $i", originalMatrix[i], recomposed[i], 0.01f)
        }
    }

    @Test
    fun testRejectionOfInvalidPoseValues() {
        assertFalse(PoseDecomposer.isValidPoseMatrix(null))
        assertFalse(PoseDecomposer.isValidPoseMatrix(FloatArray(15)))

        val nanMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        nanMatrix[12] = Float.NaN
        assertFalse(PoseDecomposer.isValidPoseMatrix(nanMatrix))

        val infMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        infMatrix[0] = Float.POSITIVE_INFINITY
        assertFalse(PoseDecomposer.isValidPoseMatrix(infMatrix))

        try {
            PoseDecomposer.decompose(nanMatrix)
            fail("Expected IllegalArgumentException for NaN matrix")
        } catch (e: IllegalArgumentException) {
            // Success
        }
    }

    @Test
    fun testQuaternionSignContinuity() {
        val filter = OneEuroFilterQuaternion()

        // Feed first quaternion q0 = (0, 0, 0, 1)
        val qOut1 = filter.filter(0f, 0f, 0f, 1f, 1_000_000_000L)
        assertEquals(1f, qOut1[3], 0.01f)

        // Feed negated quaternion -q0 = (0, 0, 0, -1) which represents the EXACT same rotation
        // Without sign continuity, this would cause a 180-degree jump artifact.
        val qOut2 = filter.filter(0f, 0f, 0f, -1f, 1_033_333_333L)

        // The filter must enforce sign continuity: output W remains positive
        assertTrue("Quaternion sign continuity violated: w was ${qOut2[3]}", qOut2[3] > 0.9f)
    }

    @Test
    fun testIndependentOneEuroFiltering() {
        val poseFilter = PoseFilter()
        var timestampNs = 1_000_000_000L

        // Feed noisy translation with steady rotation
        val p1 = PoseDecomposer.DecomposedPose(
            translation = floatArrayOf(0f, 0f, -0.5f),
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 0f, pitchDeg = 0f, rollDeg = 0f
        )
        val f1 = poseFilter.filter(p1, timestampNs)

        timestampNs += 33_333_333L
        val p2 = PoseDecomposer.DecomposedPose(
            translation = floatArrayOf(0.01f, 0f, -0.5f), // step in X
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 0f, pitchDeg = 0f, rollDeg = 0f
        )
        val f2 = poseFilter.filter(p2, timestampNs)

        // Filtered X should be smoothed between 0 and 0.01
        assertTrue(f2.translation[0] in 0.0001f..0.0099f)
        // Rotation should remain pure identity
        assertEquals(0f, f2.rotation[0], 0.0001f)
        assertEquals(1f, f2.rotation[3], 0.0001f)
    }

    @Test
    fun testIndependentBlendshapeSmoothing() {
        val pipeline = SignalPipeline()
        val matrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }

        var tNs = 1_000_000_000L
        val s1 = FaceGeometrySnapshot.create(
            timestampNs = tNs, imageWidth = 640, imageHeight = 480, rotationDegrees = 0,
            isFrontCamera = true, isTracking = true, landmarks = FloatArray(478 * 3),
            rawPoseMatrix = matrix,
            blendshapes = mapOf("jawOpen" to 0.0f, "mouthSmileLeft" to 0.8f),
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
            imageToViewTransform = FloatArray(9)
        )
        pipeline.processSnapshot(s1, tNs)

        tNs += 33_333_333L
        val s2 = FaceGeometrySnapshot.create(
            timestampNs = tNs, imageWidth = 640, imageHeight = 480, rotationDegrees = 0,
            isFrontCamera = true, isTracking = true, landmarks = FloatArray(478 * 3),
            rawPoseMatrix = matrix,
            blendshapes = mapOf("jawOpen" to 1.0f, "mouthSmileLeft" to 0.8f), // jawOpen step
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
            imageToViewTransform = FloatArray(9)
        )
        val state2 = pipeline.processSnapshot(s2, tNs)

        val jaw = state2.blendshapes["jawOpen"]!!
        val smile = state2.blendshapes["mouthSmileLeft"]!!

        // jawOpen is smoothed between 0 and 1
        assertTrue("jawOpen was $jaw", jaw in 0.05f..0.95f)
        // mouthSmileLeft remained steady at 0.8f
        assertEquals(0.8f, smile, 0.01f)
    }

    @Test
    fun testBoundedLatencyExtrapolation() {
        val compensator = LatencyCompensator(maxHorizonSec = 0.050f) // 50ms cap

        var tNs = 1_000_000_000L
        compensator.update(floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 0f, 1f), tNs)

        // Velocity = 1 m/s along X
        tNs += 33_333_333L // 33.3ms
        compensator.update(floatArrayOf(0.0333f, 0f, 0f), floatArrayOf(0f, 0f, 0f, 1f), tNs)

        // Predict 16.6ms into future (+16.6ms render time)
        val outT = FloatArray(3)
        val outQ = FloatArray(4)
        compensator.predict(tNs + 16_666_666L, outT, outQ)

        // Expected extrapolation ~ 0.0333 + 1.0 * 0.0166 = ~0.050m
        assertEquals(0.050f, outT[0], 0.005f)

        // Predict 500ms into future (must be capped at maxHorizonSec = 50ms = 0.050s)
        val outTCapped = FloatArray(3)
        compensator.predict(tNs + 500_000_000L, outTCapped, outQ)
        // Max position capped at 0.0333 + 1.0 * 0.050 = ~0.0833m
        assertEquals(0.0833f, outTCapped[0], 0.005f)
    }

    @Test
    fun testFacePoseAdapterCanonicalAxesAndFW() {
        // Test canonical axes
        assertEquals(1f, FacePoseAdapter.AXIS_SUBJECT_RIGHT[0], 0.001f)
        assertEquals(1f, FacePoseAdapter.AXIS_UP[1], 0.001f)
        assertEquals(1f, FacePoseAdapter.AXIS_TOWARD_CAMERA[2], 0.001f)
        assertEquals(-1f, FacePoseAdapter.AXIS_SUBJECT_LEFT[0], 0.001f)

        // Anatomical left vs right
        assertTrue(FacePoseAdapter.isSubjectLeft(-0.05f))
        assertFalse(FacePoseAdapter.isSubjectLeft(0.05f))
        assertTrue(FacePoseAdapter.isSubjectRight(0.05f))
        assertFalse(FacePoseAdapter.isSubjectRight(-0.05f))

        // Pinned cheek landmark IDs
        assertEquals(234, FacePoseAdapter.LANDMARK_CHEEK_RIGHT)
        assertEquals(454, FacePoseAdapter.LANDMARK_CHEEK_LEFT)

        // Test FW computation: FW = ||M · c234 - M · c454||
        val lm = FloatArray(478 * 3) { 0f }
        val i234 = 234 * 3
        val i454 = 454 * 3
        lm[i234] = 0.07f  // subject-right cheek x = 0.07m
        lm[i454] = -0.07f // subject-left cheek x = -0.07m

        val matrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        val fw = FacePoseAdapter.computeFaceWidth(lm, matrix)
        assertEquals(0.14f, fw, 0.001f) // Distance = 0.07 - (-0.07) = 0.14m

        // Test faceLocalToCamera in FW units (positive localX = subject-right -> -X_pose; negative localX = subject-left -> +X_pose)
        val localPos = FacePoseAdapter.faceLocalToCamera(
            localX = 0.5f, localY = 1.0f, localZ = 0.0f,
            matrix = matrix, fw = fw
        )
        assertEquals(-0.5f * 0.14f, localPos[0], 0.001f)
        assertEquals(1.0f * 0.14f, localPos[1], 0.001f)
        assertEquals(0.0f, localPos[2], 0.001f)

        val leftLocalPos = FacePoseAdapter.faceLocalToCamera(
            localX = -0.5f, localY = 1.0f, localZ = 0.0f,
            matrix = matrix, fw = fw
        )
        assertEquals(0.5f * 0.14f, leftLocalPos[0], 0.001f)
    }

    @Test
    fun testFrontCameraSingleStageMirroring() {
        val projFront = FacePoseAdapter.createCameraProjection(
            fovYDeg = 60f, aspectRatio = 16f / 9f, frontFacing = true
        )
        val projRear = FacePoseAdapter.createCameraProjection(
            fovYDeg = 60f, aspectRatio = 16f / 9f, frontFacing = false
        )

        // Front projection must mirror X (negative m[0])
        assertTrue("Front projection must have negative m[0]", projFront[0] < 0f)
        // Rear projection must have positive m[0]
        assertTrue("Rear projection must have positive m[0]", projRear[0] > 0f)
        // Absolute focal values must match
        assertEquals(abs(projRear[0]), abs(projFront[0]), 0.0001f)
    }

    @Test
    fun testFaceLifecycleTransitionsAndReacquisition() {
        val lifecycle = FaceLifecycle(
            LifecycleConfig(holdMs = 100L, fadeMs = 150L, easeInMs = 50L, cancelFade = true)
        )

        var tNs = 1_000_000_000L

        // Initial state
        assertEquals(FaceLifecycle.State.NO_FACE, lifecycle.currentState)

        // 1. Face appears -> ACQUIRING
        val ev1 = lifecycle.update(hasFace = true, nowNs = tNs)
        assertEquals("faceAcquired", ev1)
        assertEquals(FaceLifecycle.State.ACQUIRING, lifecycle.currentState)

        // 2. Advance past easeInMs (50ms) -> TRACKING
        tNs += 60_000_000L
        lifecycle.update(hasFace = true, nowNs = tNs)
        assertEquals(FaceLifecycle.State.TRACKING, lifecycle.currentState)
        assertEquals(1.0f, lifecycle.fadeWeight, 0.001f)

        // 3. Face lost -> HOLD
        tNs += 33_000_000L
        lifecycle.update(hasFace = false, nowNs = tNs)
        assertEquals(FaceLifecycle.State.HOLD, lifecycle.currentState)
        assertEquals(1.0f, lifecycle.fadeWeight, 0.001f)

        // 4. Advance past holdMs (100ms) -> FADING
        tNs += 120_000_000L
        lifecycle.update(hasFace = false, nowNs = tNs)
        assertEquals(FaceLifecycle.State.FADING, lifecycle.currentState)
        assertTrue(lifecycle.fadeWeight in 0.0f..1.0f)

        // 5. Re-acquire during fade! (cancelFade = true)
        lifecycle.update(hasFace = true, nowNs = tNs + 10_000_000L)
        assertEquals(FaceLifecycle.State.TRACKING, lifecycle.currentState)
        assertEquals(1.0f, lifecycle.fadeWeight, 0.001f)

        // 6. Complete fade to NO_FACE with hideAndReset
        lifecycle.update(hasFace = false, nowNs = tNs + 20_000_000L) // back to HOLD
        tNs += 300_000_000L // exceed hold + fade
        val evLost = lifecycle.update(hasFace = false, nowNs = tNs)
        assertEquals("faceLost", evLost)
        assertEquals(FaceLifecycle.State.NO_FACE, lifecycle.currentState)
        assertEquals(0.0f, lifecycle.fadeWeight, 0.001f)
        assertFalse(lifecycle.isVisible)
    }

    @Test
    fun testFaceLifecycleKeepPolicy() {
        val lifecycle = FaceLifecycle(
            LifecycleConfig(holdMs = 50L, fadeMs = 50L, afterFade = "keep")
        )
        var tNs = 1_000_000_000L

        lifecycle.update(hasFace = true, nowNs = tNs)
        tNs += 150_000_000L // exceed hold + fade
        lifecycle.update(hasFace = false, nowNs = tNs)
        tNs += 150_000_000L
        lifecycle.update(hasFace = false, nowNs = tNs)

        assertEquals(FaceLifecycle.State.NO_FACE, lifecycle.currentState)
        assertEquals(0.0f, lifecycle.fadeWeight, 0.001f)
        // In "keep" mode, isVisible remains true after fade
        assertTrue(lifecycle.isVisible)
    }

    @Test
    fun testImmutableFaceStatePublicationAndConcurrency() {
        val pipeline = SignalPipeline()
        val matrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }

        val iterations = 50
        val latch = CountDownLatch(2)
        val readCount = AtomicInteger(0)

        // Thread 1: Writer simulating perception updates
        val writer = Thread {
            for (i in 0 until iterations) {
                val tNs = 1_000_000_000L + (i * 33_333_333L)
                val snapshot = FaceGeometrySnapshot.create(
                    timestampNs = tNs, imageWidth = 640, imageHeight = 480, rotationDegrees = 0,
                    isFrontCamera = true, isTracking = true, landmarks = FloatArray(478 * 3),
                    rawPoseMatrix = matrix, blendshapes = emptyMap(),
                    modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
                    imageToViewTransform = FloatArray(9)
                )
                pipeline.processSnapshot(snapshot, tNs)
                Thread.sleep(2)
            }
            latch.countDown()
        }

        // Thread 2: Reader simulating render thread reading latest state atomically
        val reader = Thread {
            while (latch.count > 1) {
                val state = pipeline.getLatestState()
                assertNotNull(state)
                assertNotNull(state.poseMatrix)
                assertEquals(16, state.poseMatrix.size)
                readCount.incrementAndGet()
                Thread.sleep(1)
            }
            latch.countDown()
        }

        writer.start()
        reader.start()

        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertTrue(readCount.get() > 0)
    }

    @Test
    fun testReplayHarnessVerification() {
        val report = ReplayHarness.runFullVerification()

        // 1. Jitter reduction: filtered must be lower than raw
        assertTrue(
            "Jitter reduction must pass. Reduction was ${report.jitter.jitterReductionPercent}%",
            report.jitter.passed
        )
        assertTrue(report.jitter.filteredStdDevMm < report.jitter.rawStdDevMm)

        // 2. Fast motion tracking without detachment
        assertTrue(
            "Fast motion tracking must pass: lag=${report.fastMotion.maxLagMm}mm, angleErr=${report.fastMotion.maxAngleErrorDeg}deg",
            report.fastMotion.passed
        )

        // 3. Lifecycle state transitions
        assertTrue("Lifecycle transitions must pass", report.lifecycle.passed)
        assertTrue("Full replay report allPassed must be true", report.allPassed)
    }
}
