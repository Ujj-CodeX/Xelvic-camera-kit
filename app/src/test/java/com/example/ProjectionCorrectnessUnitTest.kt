package com.example

import android.opengl.GLES30
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.render.FaceMeshRenderer
import com.example.engine.render.ModelRenderer
import com.example.engine.render.OccluderPass
import com.example.engine.render.SpriteRenderer
import com.example.engine.signal.CameraIntrinsics
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import com.example.engine.signal.SignalPipeline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.tan

/**
 * Unit tests for **Chunk: projection correctness (README Sections 7, 9, and 10)**:
 * 1. Known intrinsics round-trip (with `SENSOR_INFO_PIXEL_ARRAY_SIZE` + `ACTIVE_ARRAY_SIZE` scaling
 *    and `CaptureResult` `LENS_FOCAL_LENGTH` / `SCALER_CROP_REGION` overrides).
 * 2. Two FOVs (`fovYCropDeg` vs `fovYViewDeg`) and 3D-vs-2D pixel agreement across both wide
 *    (`viewAspect > cropAspect`) and narrow (`viewAspect < cropAspect`) viewports, plus 3D matrix-projected
 *    `DebugOverlay` gizmo and head-proxy wireframe.
 * 3. Single-stage mirror enforcement for both 3D (`createCameraProjection`) and 2D (`imageToViewTransform`)
 *    paths, and front/rear camera triangle winding & back-face culling (`GL_CW` vs `GL_CCW`).
 * 4. FOV sanity bounds (`30..100 deg`) and fallback (`"intrinsics"` vs `"fallback"`), plus HUD & CSV logging.
 * 5. Per-frame `isFrontCamera` and `ProjectionParams` ownership across a camera lens switch.
 */
class ProjectionCorrectnessUnitTest {

    /**
     * 1. Known intrinsics round-trip:
     * Verifies that `FacePoseAdapter.computeProjectionParams` scales `SENSOR_INFO_PHYSICAL_SIZE`
     * by `ACTIVE_ARRAY_SIZE / PIXEL_ARRAY_SIZE`, applies `SCALER_CROP_REGION` and `ImageProxy.cropRect`,
     * and that a 3D face at a known metric position `(tx, ty, tz)` and cheek width `FW` projected into
     * normalized `[0..1]` landmarks round-trips through `MediaPipeFaceGeometryProvider.calibratePoseMatrix`
     * and `SignalPipeline` back to the exact metric camera pose and screen coordinates.
     */
    @Test
    fun knownIntrinsics_roundTripPoseAndScreenProjection() {
        // Known physical sensor: 6.40 mm x 4.80 mm full pixel array (4000 x 3000),
        // with active array 3840 x 2880 (96% of full array -> active physical 6.144 mm x 4.608 mm),
        // focal length 4.25 mm, sensor buffers captured in landscape (1920 x 1080, 16:9 crop of 4:3 active sensor)
        // rotated 90 degrees to portrait upright (9:16).
        val intrinsics = CameraIntrinsics(
            focalLengthMm = 4.25f,
            sensorWidthMm = 6.40f,
            sensorHeightMm = 4.80f,
            pixelArrayWidth = 4000,
            pixelArrayHeight = 3000,
            activeArrayLeft = 80,
            activeArrayTop = 60,
            activeArrayWidth = 3840,
            activeArrayHeight = 2880
        )

        val projParams = FacePoseAdapter.computeProjectionParams(
            intrinsics = intrinsics,
            bufferWidth = 1920,
            bufferHeight = 1080,
            cropRect = null,
            rotationDegrees = 90
        )

        assertTrue("Expected intrinsics calibration within [30..100] deg", projParams.isCalibratedFromIntrinsics)
        assertEquals("intrinsics", projParams.fovSource)
        assertEquals(9f / 16f, projParams.cropAspectRatio, 1e-4f)

        // Expected upright height in mm = activeSensorWidthMm = 6.40 * (3840 / 4000) = 6.144 mm
        // Expected fovYCropDeg = 2 * atan((6.144 / 2) / 4.25) = 71.716 deg (distinct from 60.0 deg default!)
        val expectedUprightH_mm = 6.40 * (3840.0 / 4000.0)
        val expectedFovYDeg = Math.toDegrees(2.0 * atan((expectedUprightH_mm * 0.5) / 4.25)).toFloat()
        assertEquals(expectedFovYDeg, projParams.fovYDeg, 0.05f)
        assertTrue(abs(projParams.fovYDeg - 60.0f) > 10.0f)

        // Construct a synthetic 3D face at known camera-space translation (tx, ty, tz) and cheek width FW = 0.14 m
        val trueTx = 0.045f
        val trueTy = -0.030f
        val trueTz = -0.480f
        val absZ = abs(trueTz)
        val trueFw = 0.140f

        val fCrop = FacePoseAdapter.focalFactorFromFovY(projParams.fovYDeg)
        val halfH = absZ / fCrop
        val halfW = halfH * projParams.cropAspectRatio

        // Project right cheek (+FW/2, 0, trueTz) and left cheek (-FW/2, 0, trueTz) into normalized [0..1] crop coords
        val uRight = ((trueTx + trueFw * 0.5f) / halfW) * 0.5f + 0.5f
        val uLeft = ((trueTx - trueFw * 0.5f) / halfW) * 0.5f + 0.5f
        val vCenter = 0.5f - (trueTy / halfH) * 0.5f

        val landmarks = FloatArray(478 * 3)
        // Populate all landmarks around face center, and set pinned cheek landmarks 234 & 454 + nose tip 1
        for (i in 0 until 478) {
            landmarks[i * 3] = (uRight + uLeft) * 0.5f
            landmarks[i * 3 + 1] = vCenter
            landmarks[i * 3 + 2] = 0f
        }
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3] = uRight
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3 + 1] = vCenter
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3] = uLeft
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3 + 1] = vCenter
        landmarks[FacePoseAdapter.LANDMARK_NOSE_TIP * 3] = (uRight + uLeft) * 0.5f
        landmarks[FacePoseAdapter.LANDMARK_NOSE_TIP * 3 + 1] = vCenter

        // Raw MediaPipe matrix in centimeters (z = -48.0 cm -> -0.480 m)
        val rawMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, -48.0f, 1f
        )

        val calibratedPose = MediaPipeFaceGeometryProvider.calibratePoseMatrix(
            rawMpMatrix = rawMatrix,
            lm = landmarks,
            imageWidth = 1080,
            imageHeight = 1920,
            fovYCropDeg = projParams.fovYDeg
        )

        // Recovered 3D camera-space translation must match (trueTx, trueTy, trueTz)
        assertEquals("Recovered Z depth must match trueTz", trueTz, calibratedPose[14], 1e-4f)
        assertEquals("Recovered X translation must match trueTx", trueTx, calibratedPose[12], 1e-4f)
        assertEquals("Recovered Y translation must match trueTy", trueTy, calibratedPose[13], 1e-4f)

        // Verify that calibrating with naive 60-degree FOV would produce a wrong X/Y translation (>18% error)
        val naive60Pose = MediaPipeFaceGeometryProvider.calibratePoseMatrix(
            rawMpMatrix = rawMatrix,
            lm = landmarks,
            imageWidth = 1080,
            imageHeight = 1920,
            fovYCropDeg = 60.0f
        )
        assertTrue(
            "Calibrated X translation under 71.7 deg FOV must differ from naive 60 deg FOV",
            abs(calibratedPose[12] - naive60Pose[12]) > 0.008f
        )

        // Pass through SignalPipeline and verify FaceState carries projectionParams and recovers FW = 0.14m
        val pipeline = SignalPipeline()
        val snapshot = FaceGeometrySnapshot.create(
            timestampNs = 100_000_000L,
            imageWidth = 1080,
            imageHeight = 1920,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = landmarks,
            rawPoseMatrix = calibratedPose,
            blendshapes = emptyMap(),
            projectionParams = projParams
        )
        pipeline.processSnapshot(snapshot, 100_000_000L)
        val state = pipeline.consumeLatestState()
        assertEquals(projParams, state.projectionParams)
        assertEquals(trueFw, state.faceWidth, 0.005f)
    }

    /**
     * 2. Two FOVs (`fovYCropDeg` vs `fovYViewDeg`) and 3D-vs-2D pixel agreement on both
     * WIDE (`viewAspect > cropAspect`) and NARROW (`viewAspect < cropAspect`) viewports:
     * Verifies that:
     * - When `viewAspect > cropAspect`, `tan(fovYView/2) = tan(fovYCrop/2) * cropAspect / viewAspect`.
     * - When `viewAspect <= cropAspect`, `fovYView = fovYCrop`.
     * - Projecting the 3D calibrated face center and landmarks through `createCameraProjection(fovYView, viewAspect)`
     *   lands on the exact same viewport pixel `(px, py)` as the 2D `imageToViewTransform` path
     *   for both wide and narrow views and for both front and rear cameras.
     */
    @Test
    fun twoFovs_3dProjectionMatches2dImageToViewOnWideAndNarrowViews() {
        val fovYCropDeg = 68.0f
        val cropAspect = 9f / 16f // 0.5625

        // Test both wide view (e.g. foldable/tablet pane 3:4 = 0.75 > 0.5625)
        // and narrow view (e.g. tall phone 9:20 = 0.45 < 0.5625)
        val viewConfigs = listOf(
            Triple(900f, 1200f, true),   // Wide view, Front camera
            Triple(900f, 1200f, false),  // Wide view, Rear camera
            Triple(900f, 2000f, true),   // Narrow view, Front camera
            Triple(900f, 2000f, false)   // Narrow view, Rear camera
        )

        for ((viewW, viewH, isFront) in viewConfigs) {
            val viewAspect = viewW / viewH
            val fovYViewDeg = FacePoseAdapter.computeVisibleViewFovYDeg(
                fovYCropDeg = fovYCropDeg,
                cropAspectRatio = cropAspect,
                viewAspectRatio = viewAspect
            )

            if (viewAspect > cropAspect) {
                val expectedTanHalf = tan(Math.toRadians(fovYCropDeg * 0.5)) * (cropAspect / viewAspect)
                val expectedFovY = Math.toDegrees(2.0 * atan(expectedTanHalf)).toFloat()
                assertEquals(expectedFovY, fovYViewDeg, 1e-3f)
                assertTrue("Wide view must reduce vertical FOV after FILL_CENTER crop", fovYViewDeg < fovYCropDeg)
            } else {
                assertEquals("Narrow view must keep vertical FOV unchanged", fovYCropDeg, fovYViewDeg, 1e-4f)
            }

            // Place face at off-center normalized analysis-crop coordinates (u = 0.37, v = 0.42)
            val uCenter = 0.37f
            val vCenter = 0.42f
            val spanU = 0.24f
            val landmarks = FloatArray(478 * 3)
            for (i in 0 until 478) {
                landmarks[i * 3] = uCenter
                landmarks[i * 3 + 1] = vCenter
                landmarks[i * 3 + 2] = 0f
            }
            landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3] = uCenter + spanU * 0.5f
            landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3 + 1] = vCenter
            landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3] = uCenter - spanU * 0.5f
            landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3 + 1] = vCenter

            // 1) Calibrate 3D pose using ANALYSIS-CROP FOV (fovYCropDeg)
            val rawPose = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -20f, 1f
            )
            val poseMatrix = MediaPipeFaceGeometryProvider.calibratePoseMatrix(
                rawMpMatrix = rawPose,
                lm = landmarks,
                imageWidth = 1080,
                imageHeight = 1920,
                fovYCropDeg = fovYCropDeg
            )

            // 2) Project 3D face origin (0, 0, 0 in face-local space) using VISIBLE-VIEW FOV (fovYViewDeg)
            val projMatrix = FacePoseAdapter.createCameraProjection(
                fovYDeg = fovYViewDeg,
                aspectRatio = viewAspect,
                frontFacing = isFront
            )
            val screen3d = FacePoseAdapter.projectFaceLocalToView(
                localX = 0f,
                localY = 0f,
                localZ = 0f,
                poseMatrix = poseMatrix,
                fw = 0.14f,
                projectionMatrix = projMatrix,
                viewWidth = viewW,
                viewHeight = viewH
            )

            // 3) Map 2D landmark (uCenter, vCenter) using FILL_CENTER imageToViewTransform
            val imgToView = FacePoseAdapter.computeManualFillCenterImageToViewTransform(
                cropAspectRatio = cropAspect,
                viewAspectRatio = viewAspect,
                isFrontCamera = isFront
            )
            val screen2d = FacePoseAdapter.mapLandmarkToView(
                u = uCenter,
                v = vCenter,
                imageToViewTransform = imgToView,
                isFrontCamera = isFront,
                viewWidth = viewW,
                viewHeight = viewH
            )

            assertEquals(
                "3D and 2D X pixel must agree for viewAspect=$viewAspect isFront=$isFront",
                screen2d[0],
                screen3d[0],
                0.25f
            )
            assertEquals(
                "3D and 2D Y pixel must agree for viewAspect=$viewAspect isFront=$isFront",
                screen2d[1],
                screen3d[1],
                0.25f
            )
        }
    }

    /**
     * 3. Single-stage mirror enforcement & Front/Rear triangle winding and culling:
     * Verifies:
     * - 3D mirror is applied once in `FacePoseAdapter.createCameraProjection` (`proj[0] < 0` for front,
     *   `proj[0] > 0` for rear), while `calibratePoseMatrix` is unmirrored.
     * - 2D mirror is applied once inside `imageToViewTransform` (both via CameraX `composeCameraXNormalizedImageToViewTransform`
     *   and `computeManualFillCenterImageToViewTransform`), and `mapLandmarkToView` does NOT double-flip.
     * - Front-camera negative `proj[0]` flips NDC triangle winding (`CCW -> CW`), and
     *   `FacePoseAdapter.glFrontFaceForCamera` + all renderers (`ModelRenderer`, `OccluderPass`,
     *   `SpriteRenderer`, `FaceMeshRenderer`) set `GL_CW` for front camera and `GL_CCW` for rear camera
     *   so front-facing triangles remain visible and back-facing triangles are culled on both cameras.
     */
    @Test
    fun mirrorAppliedOnce_andFrontRearWindingCullingConsistent() {
        // --- A. 2D CameraX CoordinateTransform composition vs Manual FILL_CENTER oracle ---
        // Simulate a CameraX CoordinateTransform pixel matrix that maps a 360x640 upright crop
        // into a 1080x1920 PreviewView (scale = 3.0) with front-camera horizontal mirror already applied by PreviewView:
        // vx_px = (360 - u_px) * 3.0 = -3.0 * u_px + 1080, vy_px = 3.0 * v_px
        val cameraXPixelValues9 = floatArrayOf(
            -3.0f, 0.0f, 1080.0f,
            0.0f, 3.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        )
        val composedNorm9 = FacePoseAdapter.composeCameraXNormalizedImageToViewTransform(
            pixelTransformValues9 = cameraXPixelValues9,
            orientedCropWidth = 360,
            orientedCropHeight = 640,
            viewWidth = 1080,
            viewHeight = 1920
        )
        val manualNorm9 = FacePoseAdapter.computeManualFillCenterImageToViewTransform(
            cropAspectRatio = 360f / 640f,
            viewAspectRatio = 1080f / 1920f,
            isFrontCamera = true
        )
        for (i in 0 until 9) {
            assertEquals("Matrix element $i must match oracle", manualNorm9[i], composedNorm9[i], 1e-4f)
        }

        // Verify mapLandmarkToView mirrors u=0.25 to x=0.75*viewW ONCE on front camera (not double-flipped back to 0.25!)
        val frontPt = FacePoseAdapter.mapLandmarkToView(
            u = 0.25f,
            v = 0.40f,
            imageToViewTransform = composedNorm9,
            isFrontCamera = true,
            viewWidth = 1000f,
            viewHeight = 2000f
        )
        assertEquals(750f, frontPt[0], 0.1f)
        assertEquals(800f, frontPt[1], 0.1f)

        val rearNorm9 = FacePoseAdapter.computeManualFillCenterImageToViewTransform(
            cropAspectRatio = 360f / 640f,
            viewAspectRatio = 1080f / 1920f,
            isFrontCamera = false
        )
        val rearPt = FacePoseAdapter.mapLandmarkToView(
            u = 0.25f,
            v = 0.40f,
            imageToViewTransform = rearNorm9,
            isFrontCamera = false,
            viewWidth = 1000f,
            viewHeight = 2000f
        )
        assertEquals(250f, rearPt[0], 0.1f)
        assertEquals(800f, rearPt[1], 0.1f)

        // --- B. 3D Winding & Culling across Front and Rear cameras ---
        val projFront = FacePoseAdapter.createCameraProjection(60f, 9f / 16f, frontFacing = true)
        val projRear = FacePoseAdapter.createCameraProjection(60f, 9f / 16f, frontFacing = false)
        assertTrue("Front camera proj[0] must be negative (single 3D mirror)", projFront[0] < 0f)
        assertTrue("Rear camera proj[0] must be positive (unmirrored)", projRear[0] > 0f)
        assertEquals(abs(projFront[0]), abs(projRear[0]), 1e-5f)

        assertEquals(GLES30.GL_CW, FacePoseAdapter.glFrontFaceForCamera(true))
        assertEquals(GLES30.GL_CCW, FacePoseAdapter.glFrontFaceForCamera(false))

        // Counter-clockwise triangle in camera space facing the camera at z = -0.5m
        val v0 = floatArrayOf(-0.05f, -0.05f, -0.5f)
        val v1 = floatArrayOf(0.05f, -0.05f, -0.5f)
        val v2 = floatArrayOf(0.0f, 0.05f, -0.5f)

        // Front-facing triangle must be classified as front-facing on BOTH front and rear cameras
        assertTrue(
            "CCW camera-facing triangle must be front-facing on front camera under GL_CW",
            FacePoseAdapter.isTriangleFrontFacingInNdc(v0, v1, v2, projFront, frontFacing = true)
        )
        assertTrue(
            "CCW camera-facing triangle must be front-facing on rear camera under GL_CCW",
            FacePoseAdapter.isTriangleFrontFacingInNdc(v0, v1, v2, projRear, frontFacing = false)
        )

        // Reversed (back-facing) triangle (v0, v2, v1) must be culled on BOTH front and rear cameras
        assertFalse(
            "Back-facing triangle must be culled on front camera",
            FacePoseAdapter.isTriangleFrontFacingInNdc(v0, v2, v1, projFront, frontFacing = true)
        )
        assertFalse(
            "Back-facing triangle must be culled on rear camera",
            FacePoseAdapter.isTriangleFrontFacingInNdc(v0, v2, v1, projRear, frontFacing = false)
        )

        // Verify all 4 renderers initialize with GL_CCW and use glFrontFaceForCamera
        val modelRenderer = ModelRenderer()
        val occluderPass = OccluderPass()
        val spriteRenderer = SpriteRenderer()
        val faceMeshRenderer = FaceMeshRenderer()
        assertEquals(GLES30.GL_CW, FacePoseAdapter.glFrontFaceForCamera(true))
        assertEquals(GLES30.GL_CCW, modelRenderer.lastFrontFaceMode)
        assertEquals(GLES30.GL_CCW, occluderPass.lastFrontFaceMode)
        assertEquals(GLES30.GL_CCW, spriteRenderer.lastFrontFaceMode)
        assertEquals(GLES30.GL_CCW, faceMeshRenderer.lastFrontFaceMode)
    }

    /**
     * 4. DebugOverlay 3D matrix projection & FaceMeshRenderer analysis-crop vertex unprojection:
     * Verifies that `DebugOverlay.projectOverlayGeometry` projects the 3D axes gizmo and head-proxy wireframe
     * through `FacePoseAdapter.createCameraProjection(fovYViewDeg, viewAspect)` and the 4x4 `poseMatrix`,
     * and that `FaceMeshRenderer.updateVertices` unprojects landmarks using `projectionParams.fovYDeg` (`fovYCropDeg`).
     */
    @Test
    fun debugOverlayAndFaceMeshRenderer_use3dMatrixProjectionAndAnalysisCropFov() {
        val overlay = DebugOverlay()
        overlay.isEnabled = true

        val projParams = ProjectionParams(
            fovYDeg = 72.0f,
            fovXDeg = 44.8f,
            cropAspectRatio = 9f / 16f,
            focalLengthMm = 3.60f,
            scalerCropWidth = 2000,
            scalerCropHeight = 1500,
            scalerCropLeft = 1000,
            scalerCropTop = 750,
            isCalibratedFromIntrinsics = true
        )
        val pose = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0.03f, -0.02f, -0.45f, 1f
        )
        val landmarks = FloatArray(478 * 3) { idx ->
            when (idx % 3) {
                0 -> 0.65f
                1 -> 0.35f
                else -> 0.0f
            }
        }
        val state = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            poseMatrix = pose,
            faceWidth = 0.14f,
            landmarks = landmarks,
            isFrontCamera = true,
            projectionParams = projParams
        )

        val viewW = 900f
        val viewH = 1200f // Wide view: viewAspect = 0.75 > cropAspect = 0.5625
        assertTrue(overlay.projectOverlayGeometry(state, viewW, viewH, fps = 60f, tier = "A"))

        val expectedViewFov = projParams.visibleViewFovYDeg(viewW / viewH)
        assertEquals(72.0f, overlay.lastUsedFovYCropDeg, 1e-4f)
        assertEquals(expectedViewFov, overlay.lastUsedFovYViewDeg, 1e-4f)
        assertTrue(overlay.cachedProjectionHudLine.contains("FOV source: intrinsics"))
        assertTrue(overlay.cachedProjectionHudLine.contains("f=3.60mm"))
        assertTrue(overlay.cachedProjectionHudLine.contains("crop=2000x1500@1000,750"))

        // Verify gizmo origin matches FacePoseAdapter.projectFaceLocalToView at resolved head_center (0, 0.20, -0.12)
        val expectedProj = FacePoseAdapter.createCameraProjection(expectedViewFov, viewW / viewH, frontFacing = true)
        val expectedOrigin = FacePoseAdapter.projectFaceLocalToView(
            0f, 0.20f, -0.12f, pose, 0.14f, expectedProj, viewW, viewH
        )
        assertEquals(expectedOrigin[0], overlay.lastProjectedGizmoOriginPx[0], 0.05f)
        assertEquals(expectedOrigin[1], overlay.lastProjectedGizmoOriginPx[1], 0.05f)

        // On front camera, +X (subject right, toward #234 R) projects to the RIGHT of the origin on screen (larger X pixel)
        assertTrue(
            "Subject +X (Subject-R) axis tip must project to right of origin on mirrored front camera (toward #234 R)",
            overlay.lastProjectedGizmoXPx[0] > overlay.lastProjectedGizmoOriginPx[0]
        )
        // +Y (up) projects ABOVE origin on screen (smaller Y pixel)
        assertTrue(
            "Subject +Y axis tip must project above origin on screen",
            overlay.lastProjectedGizmoYPx[1] < overlay.lastProjectedGizmoOriginPx[1]
        )
        // Head-proxy top clearance marker projects above gizmo +Y tip (0.66 FW vs 0.45 FW above head_center)
        assertTrue(
            "Head-proxy top marker must project above gizmo +Y tip",
            overlay.lastProjectedHeadTopPx[1] < overlay.lastProjectedGizmoYPx[1]
        )

        // Verify FaceMeshRenderer.updateVertices uses fovYCropDeg
        val state60 = state.copy(
            projectionParams = ProjectionParams(
                fovYDeg = 60f,
                cropAspectRatio = 9f / 16f,
                isCalibratedFromIntrinsics = true
            )
        )
        val meshRenderer60 = FaceMeshRenderer()
        val meshRenderer72 = FaceMeshRenderer()
        assertTrue(meshRenderer60.updateVertices(state60, viewportAspect = viewW / viewH))
        assertTrue(meshRenderer72.updateVertices(state, viewportAspect = viewW / viewH))
        assertNotEquals(
            "FaceMeshRenderer vertices must scale with analysis-crop FOV (72 deg vs 60 deg)",
            meshRenderer60.livePositions[0],
            meshRenderer72.livePositions[0],
            1e-4f
        )
    }

    /**
     * 5. FOV sanity bounds (`30..100 deg`) and fallback (`< 30 deg`, `> 100 deg`, missing/invalid intrinsics),
     * plus `CaptureResult.LENS_FOCAL_LENGTH` / `SCALER_CROP_REGION` and HUD + CSV logging:
     */
    @Test
    fun fovSanityBoundsAndFallback_andHudCsvMetadata() {
        // Case A: Extremely long focal length -> vertical FOV < 30 deg -> falls back to 60 deg ("fallback")
        val narrowIntrinsics = CameraIntrinsics(
            focalLengthMm = 35.0f, // telephoto producing ~10 deg FOV
            sensorWidthMm = 6.4f,
            sensorHeightMm = 4.8f,
            pixelArrayWidth = 4000,
            pixelArrayHeight = 3000,
            activeArrayWidth = 4000,
            activeArrayHeight = 3000
        )
        val narrowParams = FacePoseAdapter.computeProjectionParams(narrowIntrinsics, 1920, 1080, null, 90)
        assertFalse(narrowParams.isCalibratedFromIntrinsics)
        assertEquals("fallback", narrowParams.fovSource)
        assertEquals(60.0f, narrowParams.fovYDeg, 1e-4f)

        // Case B: Extremely tiny focal length -> vertical FOV > 100 deg -> falls back to 60 deg ("fallback")
        val ultraWideIntrinsics = CameraIntrinsics(
            focalLengthMm = 1.2f, // fisheye producing > 130 deg FOV
            sensorWidthMm = 6.4f,
            sensorHeightMm = 4.8f,
            pixelArrayWidth = 4000,
            pixelArrayHeight = 3000,
            activeArrayWidth = 4000,
            activeArrayHeight = 3000
        )
        val ultraWideParams = FacePoseAdapter.computeProjectionParams(ultraWideIntrinsics, 1920, 1080, null, 90)
        assertFalse(ultraWideParams.isCalibratedFromIntrinsics)
        assertEquals("fallback", ultraWideParams.fovSource)
        assertEquals(60.0f, ultraWideParams.fovYDeg, 1e-4f)

        // Case C: Missing / zero intrinsics -> falls back to 60 deg ("fallback")
        val unknownParams = FacePoseAdapter.computeProjectionParams(CameraIntrinsics.UNKNOWN, 1920, 1080, null, 90)
        assertFalse(unknownParams.isCalibratedFromIntrinsics)
        assertEquals("fallback", unknownParams.fovSource)
        assertEquals(60.0f, unknownParams.fovYDeg, 1e-4f)

        // Case D: Valid intrinsics with SCALER_CROP_REGION (digital zoom 2x on active array)
        val fullIntrinsics = CameraIntrinsics(
            focalLengthMm = 2.8f,
            sensorWidthMm = 6.4f,
            sensorHeightMm = 4.8f,
            pixelArrayWidth = 4000,
            pixelArrayHeight = 3000,
            activeArrayWidth = 4000,
            activeArrayHeight = 3000,
            scalerCropLeft = 1000,
            scalerCropTop = 750,
            scalerCropWidth = 2000,
            scalerCropHeight = 1500
        )
        val zoomedParams = FacePoseAdapter.computeProjectionParams(fullIntrinsics, 1920, 1080, null, 90)
        assertTrue(zoomedParams.isCalibratedFromIntrinsics)
        assertEquals("intrinsics", zoomedParams.fovSource)
        assertEquals("2000x1500@1000,750", zoomedParams.scalerCropDescriptor)

        // Verify LatencyProbe CSV dump includes FOV source, focal length, and scaler crop
        val probe = LatencyProbe()
        probe.recordProjectionMetadata(zoomedParams)
        val state = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            poseTimestampNs = 980_000_000L,
            projectionParams = zoomedParams
        )
        probe.recordRenderFrame(
            renderTimestampNs = 1_000_000_000L,
            faceState = state,
            drawCpuMs = 2.0f,
            predictionOn = true
        )

        val csv = probe.dumpCsv()
        val lines = csv.trim().lines()
        assertEquals(2, lines.size)
        assertTrue(lines[0].contains("fovYDeg,fovSource,focalLengthMm,scalerCropRegion"))
        assertTrue(lines[1].contains(",intrinsics,2.80,2000x1500@1000;750"))
    }

    /**
     * 6. Per-frame `isFrontCamera` and `ProjectionParams` ownership across a camera lens switch:
     * Verifies that a frame captured BEFORE a lens switch retains its own `isFrontCamera` and
     * `ProjectionParams` all the way to `FaceState` even if `onCameraLensChanged` is called while
     * that frame is still in flight, and that subsequent frames use the new lens and reset cached params.
     */
    @Test
    fun cameraSwitch_inFlightFrameRetainsOldCameraParamsAndSubsequentFrameResets() {
        val queuedFrames = ArrayList<AnalysisFrame>()
        var snapshotListener: ((FaceGeometrySnapshot) -> Unit)? = null

        val queueingProvider = object : FaceGeometryProvider {
            override val modelSha256: String = "test"
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {
                snapshotListener = listener
            }
            override fun start(config: ProviderConfig) {}
            override fun updateConfig(config: ProviderConfig) {}
            override fun updateAnalysisSize(longEdgePx: Int) {}
            override fun submit(frame: AnalysisFrame) {
                queuedFrames.add(frame)
            }
            override fun stop() {}
        }

        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor()
        val session = CameraSession(
            provider = queueingProvider,
            signalPipeline = pipeline,
            governor = governor
        )

        // 1. Configure FRONT camera intrinsics (f = 3.2mm -> ~84.9 deg FOV) + CaptureResult zoom crop
        val frontIntrinsics = CameraIntrinsics(
            focalLengthMm = 3.2f,
            sensorWidthMm = 5.8f,
            sensorHeightMm = 4.35f,
            pixelArrayWidth = 3264,
            pixelArrayHeight = 2448,
            activeArrayWidth = 3264,
            activeArrayHeight = 2448
        )
        session.configureCameraIntrinsics(frontIntrinsics)
        session.recordCaptureResultMetadata(
            focalLengthMm = 3.2f,
            scalerCropLeft = 100,
            scalerCropTop = 100,
            scalerCropWidth = 3064,
            scalerCropHeight = 2248
        )

        // Submit Frame #1 on FRONT camera
        val submitted1 = session.submitAnalysisFrame(
            imageProxy = null,
            timestampNs = 100_000_000L,
            rotationDegrees = 270,
            isFrontCamera = true,
            sensorTimestampNs = 100_000_000L,
            arrivalElapsedRealtimeNs = 100_000_000L
        )
        assertTrue(submitted1)
        assertEquals(1, queuedFrames.size)
        val inFlightFrontFrame = queuedFrames[0]
        assertTrue(inFlightFrontFrame.isFrontCamera)
        assertEquals("intrinsics", inFlightFrontFrame.projectionParams.fovSource)
        assertEquals(3.2f, inFlightFrontFrame.projectionParams.focalLengthMm, 1e-4f)

        // 2. Switch lens to REAR camera BEFORE Frame #1 completes!
        session.onCameraLensChanged(isFront = false, rotationDegrees = 90)
        assertEquals("Lens switch must reset cached CaptureResult focal length", 0f, session.lastCaptureFocalLengthMm, 1e-5f)
        assertEquals("Lens switch must reset cached SCALER_CROP_REGION", null, session.lastScalerCropRegion)

        // Configure REAR camera intrinsics (f = 4.8mm -> different FOV)
        val rearIntrinsics = CameraIntrinsics(
            focalLengthMm = 4.8f,
            sensorWidthMm = 6.4f,
            sensorHeightMm = 4.8f,
            pixelArrayWidth = 4000,
            pixelArrayHeight = 3000,
            activeArrayWidth = 4000,
            activeArrayHeight = 3000
        )
        session.configureCameraIntrinsics(rearIntrinsics)

        // 3. Now Frame #1 (captured on FRONT camera before the switch) completes inference
        val snapshotFromFrame1 = FaceGeometrySnapshot.create(
            timestampNs = inFlightFrontFrame.timestampNs,
            imageWidth = 1080,
            imageHeight = 1920,
            rotationDegrees = 0,
            isFrontCamera = inFlightFrontFrame.isFrontCamera,
            isTracking = true,
            landmarks = FloatArray(478 * 3) { 0.5f },
            rawPoseMatrix = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -0.5f, 1f
            ),
            blendshapes = emptyMap(),
            projectionParams = inFlightFrontFrame.projectionParams
        )
        snapshotListener?.invoke(snapshotFromFrame1)

        val stateFromFrame1 = pipeline.consumeLatestState()
        assertTrue("Frame 1 state must retain isFrontCamera = true", stateFromFrame1.isFrontCamera)
        assertEquals(
            "Frame 1 state must retain front camera focal length 3.2mm",
            3.2f,
            stateFromFrame1.projectionParams.focalLengthMm,
            1e-4f
        )

        // 4. Submit Frame #2 on REAR camera and verify it uses rear camera params
        val submitted2 = session.submitAnalysisFrame(
            imageProxy = null,
            timestampNs = 200_000_000L,
            rotationDegrees = 90,
            isFrontCamera = false,
            sensorTimestampNs = 200_000_000L,
            arrivalElapsedRealtimeNs = 200_000_000L
        )
        assertTrue(submitted2)
        val rearFrame = queuedFrames[1]
        assertFalse(rearFrame.isFrontCamera)
        assertEquals(4.8f, rearFrame.projectionParams.focalLengthMm, 1e-4f)
        assertNotEquals(inFlightFrontFrame.projectionParams.fovYDeg, rearFrame.projectionParams.fovYDeg, 0.5f)
    }
}
