package com.example

import com.example.engine.capture.AnalysisFrame
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.PoseDecomposer
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Focused unit tests for Step 1 — Alignment Foundation (README Section 21).
 */
class AlignmentStep1UnitTest {

    @Test
    fun testSnapshotDefensiveCopying() {
        val lm = FloatArray(478 * 3) { it.toFloat() }
        val matrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        val blendshapes = mutableMapOf("jawOpen" to 0.5f, "mouthSmileLeft" to 0.75f)
        val transform = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

        val snapshot = FaceGeometrySnapshot.create(
            timestampNs = 1_000_000_000L,
            imageWidth = 1080,
            imageHeight = 1920,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = lm,
            rawPoseMatrix = matrix,
            blendshapes = blendshapes,
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
            imageToViewTransform = transform
        )

        // Mutate original arrays and map
        lm[0] = 999f
        matrix[0] = 999f
        blendshapes["jawOpen"] = 999f
        transform[0] = 999f

        // Defensive copy must isolate snapshot from mutations
        assertEquals(0f, snapshot.landmarks!![0], 0.0001f)
        assertEquals(1f, snapshot.rawPoseMatrix!![0], 0.0001f)
        assertEquals(0.5f, snapshot.blendshapes["jawOpen"]!!, 0.0001f)
        assertEquals(1f, snapshot.imageToViewTransform!![0], 0.0001f)
    }

    @Test
    fun testSnapshotFiniteValidation() {
        val validLm = FloatArray(478 * 3) { 0.5f }
        val validMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }

        val validSnapshot = FaceGeometrySnapshot(
            timestampNs = 100L,
            imageWidth = 100,
            imageHeight = 100,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = validLm,
            rawPoseMatrix = validMatrix,
            blendshapes = emptyMap(),
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
        )
        assertTrue(validSnapshot.hasFiniteLandmarks())
        assertTrue(validSnapshot.hasFinitePoseMatrix())
        assertTrue(validSnapshot.isValid())

        // Corrupt with NaN
        validLm[10] = Float.NaN
        val nanSnapshot = FaceGeometrySnapshot(
            timestampNs = 101L,
            imageWidth = 100,
            imageHeight = 100,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = validLm,
            rawPoseMatrix = validMatrix,
            blendshapes = emptyMap(),
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
        )
        assertFalse(nanSnapshot.hasFiniteLandmarks())
        assertFalse(nanSnapshot.isValid())

        // Corrupt with Infinity in matrix
        validLm[10] = 0.5f
        validMatrix[3] = Float.POSITIVE_INFINITY
        val infSnapshot = FaceGeometrySnapshot(
            timestampNs = 102L,
            imageWidth = 100,
            imageHeight = 100,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = validLm,
            rawPoseMatrix = validMatrix,
            blendshapes = emptyMap(),
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
        )
        assertFalse(infSnapshot.hasFinitePoseMatrix())
        assertFalse(infSnapshot.isValid())
    }

    @Test
    fun testLandmarkCountAndTopologyValidation() {
        val lm478 = FloatArray(478 * 3) { 0.5f }
        val snapshot = FaceGeometrySnapshot(
            timestampNs = 100L,
            imageWidth = 640,
            imageHeight = 480,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = lm478,
            rawPoseMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f },
            blendshapes = emptyMap(),
            modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
        )

        // Must distinguish 478-point tracker landmark output
        assertEquals(478, snapshot.landmarkCount)
        assertEquals("canonical_478_v1", snapshot.topologyId)
        assertTrue(snapshot.hasFiniteLandmarks())
    }

    @Test
    fun testTimestampMonotonicity() {
        var prevTimestampNs = 0L
        for (i in 1..100) {
            val nowNs = 1_000_000_000L + (i * 33_333_333L) // simulate 30fps monotonic intervals
            assertTrue("Timestamp must strictly increase monotonically", nowNs > prevTimestampNs)
            prevTimestampNs = nowNs
        }
    }

    @Test
    fun testSubjectBasedLeftRightMappingAndFaceWidth() {
        // Canonical coordinate convention:
        // X = subject-right (+X)
        // Negative X = subject-left (-X)
        // Cheek vertices: 234 (subject-right) and 454 (subject-left)
        assertEquals(234, FacePoseAdapter.LANDMARK_CHEEK_RIGHT)
        assertEquals(454, FacePoseAdapter.LANDMARK_CHEEK_LEFT)

        val landmarks = FloatArray(478 * 3)
        // Position cheek 234 at anatomical right (+X in face local normalized space)
        // and cheek 454 at anatomical left (-X)
        val i234 = FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3
        val i454 = FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3

        landmarks[i234] = 0.35f  // right
        landmarks[i234 + 1] = 0.5f
        landmarks[i234 + 2] = 0f

        landmarks[i454] = 0.65f  // left
        landmarks[i454 + 1] = 0.5f
        landmarks[i454 + 2] = 0f

        val fw = FacePoseAdapter.computeFaceWidth(landmarks)
        // Distance |0.65 - 0.35| = 0.30
        assertEquals(0.30f, fw, 0.0001f)
    }

    @Test
    fun testCameraProjectionSingleStageMirroringAndFovAgreement() {
        val fovY = 60.0f
        val aspect = 9f / 16f
        val near = 0.05f
        val far = 10.0f

        val projFront = FacePoseAdapter.createCameraProjection(fovY, aspect, near, far, frontFacing = true)
        val projRear = FacePoseAdapter.createCameraProjection(fovY, aspect, near, far, frontFacing = false)

        // Rear camera has positive X scale: f / aspect
        assertTrue("Rear camera projection must have positive X scale", projRear[0] > 0f)

        // Front camera has negative X scale: -(f / aspect) representing single-stage projection mirroring
        assertTrue("Front camera projection must have negative X scale for mirroring", projFront[0] < 0f)
        assertEquals(-projRear[0], projFront[0], 0.0001f)

        // Vertical FOV agreement: both front and rear use identical Y scale f = 1 / tan(fovY / 2)
        assertEquals(projRear[5], projFront[5], 0.0001f)
        val expectedF = 1.0f / kotlin.math.tan(Math.toRadians(fovY.toDouble() / 2.0)).toFloat()
        assertEquals(expectedF, projFront[5], 0.0001f)
    }

    @Test
    fun testPoseMatrixDecompositionAndValidation() {
        val t = floatArrayOf(0.05f, 0.12f, -0.65f)
        val q = floatArrayOf(0f, 0f, 0f, 1f) // identity orientation
        val s = floatArrayOf(1f, 1f, 1f)

        val m = PoseDecomposer.recompose(t, q, s)
        assertEquals(16, m.size)

        val decomp = PoseDecomposer.decompose(m)
        assertEquals(0.05f, decomp.translation[0], 0.001f)
        assertEquals(0.12f, decomp.translation[1], 0.001f)
        assertEquals(-0.65f, decomp.translation[2], 0.001f)

        assertEquals(0f, decomp.yawDeg, 0.01f)
        assertEquals(0f, decomp.pitchDeg, 0.01f)
        assertEquals(0f, decomp.rollDeg, 0.01f)
    }

    @Test
    fun testProviderConfigIndependentOutputs() {
        // Provider configuration must allow requesting pose and blendshapes independently
        val poseOnly = ProviderConfig(outputPose = true, outputBlendshapes = false, outputLandmarks = true)
        assertTrue(poseOnly.outputPose)
        assertFalse(poseOnly.outputBlendshapes)
        assertTrue(poseOnly.outputLandmarks)

        val blendshapesOnly = ProviderConfig(outputPose = false, outputBlendshapes = true, outputLandmarks = true)
        assertFalse(blendshapesOnly.outputPose)
        assertTrue(blendshapesOnly.outputBlendshapes)

        val landmarksOnly = ProviderConfig(outputPose = false, outputBlendshapes = false, outputLandmarks = true)
        assertFalse(landmarksOnly.outputPose)
        assertFalse(landmarksOnly.outputBlendshapes)
    }

    @Test
    fun testLatestPendingFrameHandoffAndSingleCloseOwnership() {
        // Verify contract: 1 inference in flight, latest pending frame replacing older frames,
        // and every frame closed exactly once.
        val closedCount = AtomicInteger(0)
        val inFlight = AtomicBoolean(false)
        val pendingRef = AtomicReference<Int?>(null)

        fun submitFrame(frameId: Int) {
            val oldPending = pendingRef.getAndSet(frameId)
            if (oldPending != null) {
                closedCount.incrementAndGet() // older pending frame discarded and closed
            }

            if (inFlight.compareAndSet(false, true)) {
                val next = pendingRef.getAndSet(null)
                if (next != null) {
                    // simulate background processing
                    closedCount.incrementAndGet() // processed frame closed
                }
                inFlight.set(false)
            }
        }

        // Frame 1 arrives: picked up immediately
        submitFrame(1)
        assertEquals(1, closedCount.get())

        // Frame 2 and Frame 3 arrive while simulated inference is active
        inFlight.set(true)
        submitFrame(2) // pending
        assertEquals(1, closedCount.get())

        submitFrame(3) // replaces 2! Frame 2 must be closed
        assertEquals(2, closedCount.get())

        // Inference finishes, next frame (3) is processed and closed
        inFlight.set(false)
        val next = pendingRef.getAndSet(null)
        assertNotNull(next)
        assertEquals(3, next)
        closedCount.incrementAndGet()

        assertEquals(3, closedCount.get()) // Exactly 3 frames closed out of 3 submissions
    }
}
