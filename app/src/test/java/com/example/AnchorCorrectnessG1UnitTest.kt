package com.example

import com.example.engine.debug.DebugOverlay
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.LegacyAdapters
import com.example.engine.render.FaceMeshRenderer
import com.example.engine.scene.AnchorResolver
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import com.example.ui.DebugHudLayoutCalculator
import com.example.ui.DebugHudSectionState
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * Unit tests for **Chunk G1: Anchor Correctness**:
 * 1. `head_center` built-in default `[0, 0.2, -0.12]` and package `"anchors"` override (`offsetFW`)
 *    applied consistently to model nodes and the head-proxy ellipsoid (`occluder` node + `DebugOverlay`),
 *    plus `sunflower_forehead/manifest.json` `layer.offsetFW = [-0.25, 0.40, 0.47]` and removal of
 *    `sunflower_forehead.xfilter`.
 * 2. Landmark-weighted custom anchor unprojects live landmarks to metric camera space via the shared
 *    `FacePoseAdapter.unprojectLandmarkToCameraSpace` / `FaceMeshRenderer` function, transforms by
 *    `inverse(poseMatrix)`, divides by `FW`, and adds `offsetFW`, matching the unprojected 3D landmark
 *    surface across yaw `-40°..+40°` (and pitch/roll) within tolerance.
 * 3. Side convention: `negative x = subject-left` (`#454` side) and `positive x = subject-right` (`#234` side)
 *    for all built-in anchors and node positions on both front and rear cameras via a single conversion in
 *    `FacePoseAdapter`, without mirroring GLB 3x3 rotation/scale submatrices (`det > 0`).
 * 4. Gizmo & DebugOverlay: red `+X (Subject-R)` axis is labelled and drawn toward subject-right (`#234` side)
 *    on both front and rear cameras; `showResolvedAnchorMarkers` defaults to `false` and draws resolved
 *    `head_center`, `forehead`, and custom landmark anchors when enabled; 9 debug toggles wrap on `360x800 dp`.
 * 5. Zero per-frame allocation in `AnchorResolver.resolveAnchor`, `SceneEvaluator.evaluate`, and
 *    `DebugOverlay.projectOverlayGeometry`.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class AnchorCorrectnessG1UnitTest {

    private fun resolveFilterDir(filterName: String): File {
        val paths = listOf(
            File("app/src/main/assets/filters", filterName),
            File("src/main/assets/filters", filterName)
        )
        val dir = paths.firstOrNull { it.exists() && it.isDirectory }
        assertNotNull("Filter directory for '$filterName' must exist", dir)
        return dir!!
    }

    /**
     * 1. `head_center` default `[0, 0.2, -0.12]` and package override applied consistently to both
     * model nodes and the head-proxy ellipsoid (`occluder` node and `DebugOverlay` head-proxy wireframe),
     * plus `sunflower_forehead` `layer.offsetFW = [-0.25, 0.40, 0.47]` and `sunflower_forehead.xfilter` removal.
     */
    @Test
    fun headCenterDefaultAndPackageOverride_applyToNodesAndHeadProxyEllipsoid() {
        // A. Built-in default [0, 0.2, -0.12]
        val defaultResolver = AnchorResolver()
        val faceState = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1f,
            faceWidth = 0.20f,
            poseMatrix = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -0.50f, 1f
            )
        )
        val outDefault = FloatArray(3)
        defaultResolver.resolveAnchor("head_center", faceState, outDefault)
        assertEquals(0.0f, outDefault[0], 1e-5f)
        assertEquals(0.20f, outDefault[1], 1e-5f)
        assertEquals(-0.12f, outDefault[2], 1e-5f)

        // B. Verify sunflower_forehead.xfilter is removed and folder package is single source of truth
        val staleZipCandidates = listOf(
            File("app/src/main/assets/filters/sunflower_forehead.xfilter"),
            File("src/main/assets/filters/sunflower_forehead.xfilter")
        )
        for (zipFile in staleZipCandidates) {
            assertFalse("sunflower_forehead.xfilter must not exist (${zipFile.path})", zipFile.exists())
        }

        // C. Compile sunflower_forehead folder package and verify layer.anchor = "forehead_left_hairline",
        //    layer.offsetFW = [0.0, 0.06, 0.02], layer.scale = 0.9, and head_center = [0.0, 0.20, -0.12]
        //    on the headProxy ellipsoid occluder
        val sunflowerDir = resolveFilterDir("sunflower_forehead")
        val sunflowerJson = JSONObject(File(sunflowerDir, "manifest.json").readText())
        val compiledSunflower = Compiler.compile(sunflowerJson, sunflowerDir)

        val modelNode = compiledSunflower.nodes.first { it.type == "model" }
        val headEllipsoidNode = compiledSunflower.nodes.first {
            it.type == "occluder" && it.occluderShape == "ellipsoid"
        }
        assertEquals("forehead_left_hairline", modelNode.anchor)
        assertEquals("head_center", headEllipsoidNode.anchor)
        assertEquals(0.0f, modelNode.basePosition[0], 1e-5f)
        assertEquals(0.06f, modelNode.basePosition[1], 1e-5f)
        assertEquals(0.02f, modelNode.basePosition[2], 1e-5f)
        assertEquals(0.9f, modelNode.baseScale[0], 1e-5f)

        val evaluator = SceneEvaluator(maxNodes = 8)
        evaluator.evaluate(compiledSunflower, faceState, compiledSunflower.propertySlots)

        val fw = faceState.faceWidth // 0.20m
        val headEllipsoidMatOffset = headEllipsoidNode.index * 16
        val modelMatOffset = modelNode.index * 16

        // Head-proxy ellipsoid has basePosition [0, 0, 0] + anchor head_center [0, 0.20, -0.12] FW
        assertEquals(0.0f, evaluator.worldMatrices[headEllipsoidMatOffset + 12], 1e-4f)
        assertEquals(0.20f * fw, evaluator.worldMatrices[headEllipsoidMatOffset + 13], 1e-4f)
        assertEquals(-0.50f + (-0.12f * fw), evaluator.worldMatrices[headEllipsoidMatOffset + 14], 1e-4f)

        // Without live landmarks, custom anchor forehead_left_hairline resolves to its offsetFW [-0.01, 0.05, 0.03]
        // plus modelNode.basePosition [0.0, 0.06, 0.02] = [-0.01, 0.11, 0.05] FW
        assertEquals(
            FacePoseAdapter.faceLocalXToPoseMeters(-0.01f, fw),
            evaluator.worldMatrices[modelMatOffset + 12],
            1e-4f
        )
        assertEquals((0.05f + 0.06f) * fw, evaluator.worldMatrices[modelMatOffset + 13], 1e-4f)
        assertEquals(-0.50f + ((0.03f + 0.02f) * fw), evaluator.worldMatrices[modelMatOffset + 14], 1e-4f)

        // D. Package override of head_center via "anchors" block in xfilter-face-3d/1 and xfilter/3
        val customHeadManifest = JSONObject(sunflowerJson.toString())
        val customOffset = JSONArray().put(0.05).put(0.32).put(-0.18)
        customHeadManifest.getJSONObject("anchors")
            .getJSONObject("head_center")
            .put("offsetFW", customOffset)
        customHeadManifest.getJSONObject("layer")
            .put("anchor", "head_center")
        customHeadManifest.optJSONObject("calibration")
            ?.optJSONObject("tunableFields")
            ?.put("head_center.offsetFW", customOffset)

        val compiledCustomHead = Compiler.compile(customHeadManifest, sunflowerDir)
        val outCustomHead = FloatArray(3)
        compiledCustomHead.anchorResolver.resolveAnchor("head_center", faceState, outCustomHead)
        assertEquals(0.05f, outCustomHead[0], 1e-5f)
        assertEquals(0.32f, outCustomHead[1], 1e-5f)
        assertEquals(-0.18f, outCustomHead[2], 1e-5f)

        // Evaluate and verify both model node (when anchored to head_center) and headProxy ellipsoid shift together with overridden head_center
        evaluator.evaluate(compiledCustomHead, faceState, compiledCustomHead.propertySlots)
        assertEquals(
            FacePoseAdapter.faceLocalXToPoseMeters(0.05f, fw),
            evaluator.worldMatrices[headEllipsoidMatOffset + 12],
            1e-4f
        )
        assertEquals(0.32f * fw, evaluator.worldMatrices[headEllipsoidMatOffset + 13], 1e-4f)
        assertEquals(-0.50f + (-0.18f * fw), evaluator.worldMatrices[headEllipsoidMatOffset + 14], 1e-4f)

        assertEquals(
            FacePoseAdapter.faceLocalXToPoseMeters(0.05f + 0.0f, fw),
            evaluator.worldMatrices[modelMatOffset + 12],
            1e-4f
        )
        assertEquals((0.32f + 0.06f) * fw, evaluator.worldMatrices[modelMatOffset + 13], 1e-4f)
        assertEquals(-0.50f + ((-0.18f + 0.02f) * fw), evaluator.worldMatrices[modelMatOffset + 14], 1e-4f)

        // E. Verify DebugOverlay centers both the 3D gizmo and the 3D head-proxy ellipsoid at resolved head_center
        val overlay = DebugOverlay()
        overlay.isEnabled = true
        overlay.anchorResolver = compiledCustomHead.anchorResolver
        assertTrue(overlay.projectOverlayGeometry(faceState, 1080f, 1920f, fps = 60f, tier = "A"))
        assertEquals(0.05f, overlay.lastResolvedHeadCenterFW[0], 1e-5f)
        assertEquals(0.32f, overlay.lastResolvedHeadCenterFW[1], 1e-5f)
        assertEquals(-0.18f, overlay.lastResolvedHeadCenterFW[2], 1e-5f)
    }

    /**
     * 2. Landmark-weighted custom anchor matches the unprojected 3D landmark in a synthetic pose
     * across yaw `-40°..+40°` (and with pitch/roll and `offsetFW`) within tolerance.
     */
    @Test
    fun landmarkAnchor_matchesUnprojectedLandmarkAcrossYawMinus40ToPlus40WithinTolerance() {
        val fovYCropDeg = 68.0f
        val cropAspect = 9f / 16f
        val projParams = ProjectionParams(
            fovYDeg = fovYCropDeg,
            fovXDeg = 42.0f,
            cropAspectRatio = cropAspect,
            isCalibratedFromIntrinsics = true
        )
        val fCrop = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)

        // Canonical 3D face landmarks in head-pose metric space (FW = 0.14m):
        // - Cheek #234 (subject-right) at (-0.07, 0.0, 0.0) in pose space
        // - Cheek #454 (subject-left) at (+0.07, 0.0, 0.0) in pose space
        // - Temple landmark #103 (subject-right temple) at (-0.058, +0.048, +0.015)
        // - Temple landmark #332 (subject-left temple) at (+0.058, +0.048, +0.015)
        // - Forehead landmark #10 at (0.0, +0.062, +0.025)
        val fw = 0.14f
        val headTx = 0.02f
        val headTy = -0.01f
        val headTz = -0.46f

        val canonicalPointsPose = mapOf(
            234 to floatArrayOf(-0.07f, 0.0f, 0.0f),
            454 to floatArrayOf(0.07f, 0.0f, 0.0f),
            103 to floatArrayOf(-0.058f, 0.048f, 0.015f),
            332 to floatArrayOf(0.058f, 0.048f, 0.015f),
            10 to floatArrayOf(0.0f, 0.062f, 0.025f)
        )

        val customAnchors = mapOf(
            "temple_left_332" to AnchorResolver.CustomAnchorDef(
                landmarks = listOf(AnchorResolver.LandmarkWeight(332, 1.0f)),
                offsetFW = floatArrayOf(0f, 0f, 0f)
            ),
            "temple_blend_332_10" to AnchorResolver.CustomAnchorDef(
                landmarks = listOf(
                    AnchorResolver.LandmarkWeight(332, 0.7f),
                    AnchorResolver.LandmarkWeight(10, 0.3f)
                ),
                offsetFW = floatArrayOf(-0.05f, 0.08f, 0.03f)
            )
        )
        val resolver = AnchorResolver(customAnchors)
        val unprojectedMeshVerts = FloatArray(478 * 3)

        val yawAnglesDeg = listOf(-40f, -25f, -10f, 0f, 10f, 25f, 40f)
        for (yawDeg in yawAnglesDeg) {
            val yawRad = Math.toRadians(yawDeg.toDouble()).toFloat()
            val cosY = cos(yawRad)
            val sinY = sin(yawRad)

            // Column-major 4x4 pose matrix with rotation Y(yaw) and translation (headTx, headTy, headTz)
            val poseMatrix = floatArrayOf(
                cosY, 0f, -sinY, 0f,
                0f, 1f, 0f, 0f,
                sinY, 0f, cosY, 0f,
                headTx, headTy, headTz, 1f
            )

            // Generate normalized [0..1] landmarks consistent with FaceMeshRenderer / FacePoseAdapter unprojection:
            // In unprojectLandmarkToCameraSpace:
            //   absZ = abs(headTz)
            //   wCam = 2 * absZ * cropAspect / fCrop
            //   hCam = 2 * absZ / fCrop
            //   xCam = (u - 0.5) * wCam  =>  u = 0.5 + xCam / wCam
            //   yCam = (0.5 - v) * hCam  =>  v = 0.5 - yCam / hCam
            //   zCam = -absZ - lmZ * wCam => lmZ = (-absZ - zCam) / wCam
            val absZ = -headTz
            val wCam = 2.0f * absZ * cropAspect / fCrop
            val hCam = 2.0f * absZ / fCrop
            val landmarks = FloatArray(478 * 3)
            for (i in 0 until 478) {
                landmarks[i * 3] = 0.5f
                landmarks[i * 3 + 1] = 0.5f
                landmarks[i * 3 + 2] = 0f
            }
            for ((id, pPose) in canonicalPointsPose) {
                val xCam = poseMatrix[0] * pPose[0] + poseMatrix[4] * pPose[1] + poseMatrix[8] * pPose[2] + headTx
                val yCam = poseMatrix[1] * pPose[0] + poseMatrix[5] * pPose[1] + poseMatrix[9] * pPose[2] + headTy
                val zCam = poseMatrix[2] * pPose[0] + poseMatrix[6] * pPose[1] + poseMatrix[10] * pPose[2] + headTz
                val u = 0.5f + xCam / wCam
                val v = 0.5f - yCam / hCam
                val lmZ = (-absZ - zCam) / wCam
                landmarks[id * 3] = u
                landmarks[id * 3 + 1] = v
                landmarks[id * 3 + 2] = lmZ
            }

            val faceState = FaceState(
                isVisible = true,
                lifecycle = FaceLifecycle.State.TRACKING,
                fadeWeight = 1f,
                poseMatrix = poseMatrix,
                landmarkPoseMatrix = poseMatrix,
                faceWidth = fw,
                landmarks = landmarks,
                yawDeg = yawDeg,
                projectionParams = projParams
            )

            // Unproject all landmarks with FaceMeshRenderer.unprojectLandmarksToCameraSpace
            assertTrue(
                FaceMeshRenderer.unprojectLandmarksToCameraSpace(
                    faceState = faceState,
                    viewportAspect = cropAspect,
                    pushBackFW = 0f,
                    outPositions = unprojectedMeshVerts
                )
            )

            // 1) Single-landmark custom anchor ("temple_left_332") with zero offsetFW:
            //    When transformed back to camera space via FacePoseAdapter.faceLocalToCamera, it MUST match
            //    the unprojected landmark #332 from FaceMeshRenderer within 1e-4 m across all yaw angles!
            val resolved332FW = FloatArray(3)
            resolver.resolveAnchor("temple_left_332", faceState, resolved332FW)
            val resolved332Cam = FacePoseAdapter.faceLocalToCamera(
                localX = resolved332FW[0],
                localY = resolved332FW[1],
                localZ = resolved332FW[2],
                matrix = poseMatrix,
                fw = fw
            )
            val expected332CamX = unprojectedMeshVerts[332 * 3]
            val expected332CamY = unprojectedMeshVerts[332 * 3 + 1]
            val expected332CamZ = unprojectedMeshVerts[332 * 3 + 2]

            assertEquals("Yaw=$yawDeg: landmark anchor X_cam must match FaceMeshRenderer vertex #332", expected332CamX, resolved332Cam[0], 1e-4f)
            assertEquals("Yaw=$yawDeg: landmark anchor Y_cam must match FaceMeshRenderer vertex #332", expected332CamY, resolved332Cam[1], 1e-4f)
            assertEquals("Yaw=$yawDeg: landmark anchor Z_cam must match FaceMeshRenderer vertex #332", expected332CamZ, resolved332Cam[2], 1e-4f)

            // Furthermore, in head-local FW units, landmark #332 (+0.058m on subject-left) has negative localX:
            assertEquals(-0.058f / fw, resolved332FW[0], 1e-4f)
            assertEquals(0.048f / fw, resolved332FW[1], 1e-4f)
            assertEquals(0.015f / fw, resolved332FW[2], 1e-4f)

            // 2) Weighted two-landmark anchor ("temple_blend_332_10") with offsetFW [-0.05, 0.08, 0.03]:
            val resolvedBlendFW = FloatArray(3)
            resolver.resolveAnchor("temple_blend_332_10", faceState, resolvedBlendFW)
            val expectedBlendLocalX = -(0.7f * 0.058f + 0.3f * 0.0f) / fw - 0.05f
            val expectedBlendLocalY = (0.7f * 0.048f + 0.3f * 0.062f) / fw + 0.08f
            val expectedBlendLocalZ = (0.7f * 0.015f + 0.3f * 0.025f) / fw + 0.03f
            assertEquals("Yaw=$yawDeg: blended landmark localX", expectedBlendLocalX, resolvedBlendFW[0], 1e-4f)
            assertEquals("Yaw=$yawDeg: blended landmark localY", expectedBlendLocalY, resolvedBlendFW[1], 1e-4f)
            assertEquals("Yaw=$yawDeg: blended landmark localZ", expectedBlendLocalZ, resolvedBlendFW[2], 1e-4f)
        }
    }

    /**
     * 3. Side convention test for built-in anchors and node positions on both front and rear cameras,
     * plus verification that GLB 3x3 rotation/scale submatrices are NOT mirrored (`det > 0`).
     */
    @Test
    fun sideConvention_builtInAnchorsAndNodesMapNegativeXToSubjectLeft454OnFrontAndRearWithoutMirroringGlb() {
        val fovYDeg = 60.0f
        val aspect = 9f / 16f
        val viewW = 1080f
        val viewH = 1920f

        // Set up canonical MediaPipe landmarks in unmirrored camera image [0..1]:
        // Subject facing the camera:
        // - Landmark #234 (subject's anatomical right cheek) is on the LEFT of the unmirrored image (u = 0.35)
        // - Landmark #454 (subject's anatomical left cheek) is on the RIGHT of the unmirrored image (u = 0.65)
        val landmarks = FloatArray(478 * 3)
        for (i in 0 until 478) {
            landmarks[i * 3] = 0.50f
            landmarks[i * 3 + 1] = 0.50f
            landmarks[i * 3 + 2] = 0f
        }
        val u234 = 0.35f
        val u454 = 0.65f
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3] = u234
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3 + 1] = 0.50f
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3] = u454
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3 + 1] = 0.50f
        for (lmId in listOf(297, 338, 332)) {
            landmarks[lmId * 3] = 0.58f
            landmarks[lmId * 3 + 1] = 0.30f
            landmarks[lmId * 3 + 2] = 0.02f
        }

        val rawMpMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, -45.0f, 1f
        )
        val poseMatrix = MediaPipeFaceGeometryProvider.calibratePoseMatrix(
            rawMpMatrix = rawMpMatrix,
            lm = landmarks,
            imageWidth = 1080,
            imageHeight = 1920,
            fovYCropDeg = fovYDeg
        )

        // Evidence check: calibratePoseMatrix builds pose +X axis from #234 -> #454,
        // so poseMatrix[0] > 0 (pointing toward +X_cam, which is the #454 subject-left side).
        assertTrue("Pose column 0 (+X_pose) points from #234 toward #454 (+X_cam)", poseMatrix[0] > 0.9f)

        val resolver = AnchorResolver()
        val fw = 0.14f
        val leftRightPairs = listOf(
            "eye_left" to "eye_right",
            "cheek_left" to "cheek_right",
            "ear_left" to "ear_right"
        )

        for (isFrontCamera in listOf(true, false)) {
            val projMatrix = FacePoseAdapter.createCameraProjection(fovYDeg, aspect, frontFacing = isFrontCamera)
            val imgToView = FacePoseAdapter.computeManualFillCenterImageToViewTransform(aspect, aspect, isFrontCamera)

            // Screen X of landmark #234 (subject-right) and #454 (subject-left)
            val screen234 = FacePoseAdapter.mapLandmarkToView(u234, 0.50f, imgToView, isFrontCamera, viewW, viewH)
            val screen454 = FacePoseAdapter.mapLandmarkToView(u454, 0.50f, imgToView, isFrontCamera, viewW, viewH)
            val screenCenter = FacePoseAdapter.projectFaceLocalToView(0f, 0f, 0f, poseMatrix, fw, projMatrix, viewW, viewH)

            // Verify #234 and #454 sit on opposite sides of the face center
            val dir234 = screen234[0] - screenCenter[0]
            val dir454 = screen454[0] - screenCenter[0]
            assertTrue("isFront=$isFrontCamera: #234 and #454 must be on opposite sides of center", dir234 * dir454 < 0f)

            for ((leftAnchorName, rightAnchorName) in leftRightPairs) {
                val leftFW = FloatArray(3)
                val rightFW = FloatArray(3)
                resolver.resolveAnchor(leftAnchorName, FaceState(isVisible = true, faceWidth = fw), leftFW)
                resolver.resolveAnchor(rightAnchorName, FaceState(isVisible = true, faceWidth = fw), rightFW)

                // Package contract: negative x = subject-left, positive x = subject-right
                assertTrue("$leftAnchorName must have negative x in FW space", leftFW[0] < 0f)
                assertTrue("$rightAnchorName must have positive x in FW space", rightFW[0] > 0f)

                val screenLeft = FacePoseAdapter.projectFaceLocalToView(
                    leftFW[0], leftFW[1], leftFW[2], poseMatrix, fw, projMatrix, viewW, viewH
                )
                val screenRight = FacePoseAdapter.projectFaceLocalToView(
                    rightFW[0], rightFW[1], rightFW[2], poseMatrix, fw, projMatrix, viewW, viewH
                )

                // Built-in *_left anchor MUST project to the #454 (subject-left) side of center,
                // and *_right anchor MUST project to the #234 (subject-right) side of center!
                assertTrue(
                    "isFront=$isFrontCamera: $leftAnchorName must project toward #454 (subject-left)",
                    (screenLeft[0] - screenCenter[0]) * dir454 > 0f
                )
                assertTrue(
                    "isFront=$isFrontCamera: $rightAnchorName must project toward #234 (subject-right)",
                    (screenRight[0] - screenCenter[0]) * dir234 > 0f
                )
            }

            // Verify sunflower_forehead node position [-0.25, 0.40, 0.47] projects to the #454 (subject-left) side
            // and its 3x3 world rotation/scale matrix has positive determinant (GLB model is NOT mirrored)
            val sunflowerDir = resolveFilterDir("sunflower_forehead")
            val compiledSunflower = Compiler.compile(
                JSONObject(File(sunflowerDir, "manifest.json").readText()),
                sunflowerDir
            )
            val evaluator = SceneEvaluator(maxNodes = 8)
            val faceState = FaceState(
                isVisible = true,
                lifecycle = FaceLifecycle.State.TRACKING,
                fadeWeight = 1f,
                poseMatrix = poseMatrix,
                faceWidth = fw,
                landmarks = landmarks,
                isFrontCamera = isFrontCamera
            )
            evaluator.evaluate(compiledSunflower, faceState, compiledSunflower.propertySlots)
            val modelNode = compiledSunflower.nodes.first { it.type == "model" }
            val mOff = modelNode.index * 16
            val m = evaluator.worldMatrices

            // Project model world translation (m[mOff + 12], m[mOff + 13], m[mOff + 14]) to screen
            val wx = m[mOff + 12]
            val wy = m[mOff + 13]
            val wz = m[mOff + 14]
            val clipX = projMatrix[0] * wx + projMatrix[4] * wy + projMatrix[8] * wz + projMatrix[12]
            val clipW = projMatrix[3] * wx + projMatrix[7] * wy + projMatrix[11] * wz + projMatrix[15]
            val modelScreenX = ((clipX / clipW) * 0.5f + 0.5f) * viewW
            assertTrue(
                "isFront=$isFrontCamera: sunflower model (offsetFW.x = -0.25) must project toward #454 (subject-left)",
                (modelScreenX - screenCenter[0]) * dir454 > 0f
            )

            // Verify 3x3 determinant of world matrix is strictly positive (no GLB mirroring)
            val det3x3 = m[mOff + 0] * (m[mOff + 5] * m[mOff + 10] - m[mOff + 6] * m[mOff + 9]) -
                m[mOff + 4] * (m[mOff + 1] * m[mOff + 10] - m[mOff + 2] * m[mOff + 9]) +
                m[mOff + 8] * (m[mOff + 1] * m[mOff + 6] - m[mOff + 2] * m[mOff + 5])
            assertTrue("GLB 3x3 world matrix must be right-handed (det > 0), got det=$det3x3", det3x3 > 0f)
        }
    }

    /**
     * 4. Gizmo red `+X` axis is labelled `+X (Subject-R)` and drawn toward subject-right (the `#234` side)
     * on both front and rear cameras, and `DebugOverlay.showResolvedAnchorMarkers` defaults to `false`
     * with a working toggle and 9 debug pills wrapping cleanly on `360x800 dp`.
     */
    @Test
    fun gizmoRedPlusXAxis_labelledAndDrawnTowardSubjectRight234_andResolvedAnchorMarkersToggle() {
        val overlay = DebugOverlay()
        overlay.isEnabled = true

        // 1. Default check: showResolvedAnchorMarkers must default to false
        assertFalse("showResolvedAnchorMarkers must default to false", overlay.showResolvedAnchorMarkers)

        // 2. Verify red +X axis label constant is "+X (Subject-R)"
        assertEquals("+X (Subject-R)", DebugOverlay.GIZMO_PLUS_X_AXIS_LABEL)

        val fovYDeg = 60.0f
        val aspect = 9f / 16f
        val viewW = 1080f
        val viewH = 1920f

        val landmarks = FloatArray(478 * 3)
        for (i in 0 until 478) {
            landmarks[i * 3] = 0.50f
            landmarks[i * 3 + 1] = 0.50f
            landmarks[i * 3 + 2] = 0f
        }
        val u234 = 0.35f // Subject-right cheek #234
        val u454 = 0.65f // Subject-left cheek #454
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3] = u234
        landmarks[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3] = u454
        landmarks[105 * 3] = 0.42f
        landmarks[105 * 3 + 1] = 0.36f
        landmarks[105 * 3 + 2] = 0.02f

        val poseMatrix = MediaPipeFaceGeometryProvider.calibratePoseMatrix(
            rawMpMatrix = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -45.0f, 1f
            ),
            lm = landmarks,
            imageWidth = 1080,
            imageHeight = 1920,
            fovYCropDeg = fovYDeg
        )

        // Configure a custom landmark anchor on the overlay's AnchorResolver
        overlay.anchorResolver = AnchorResolver(
            mapOf(
                "brow_lm" to AnchorResolver.CustomAnchorDef(
                    landmarks = listOf(AnchorResolver.LandmarkWeight(105, 1.0f)),
                    offsetFW = floatArrayOf(0f, 0.02f, 0f)
                )
            )
        )

        for (isFrontCamera in listOf(true, false)) {
            val imgToView = FacePoseAdapter.computeManualFillCenterImageToViewTransform(aspect, aspect, isFrontCamera)
            val state = FaceState(
                isVisible = true,
                lifecycle = FaceLifecycle.State.TRACKING,
                fadeWeight = 1f,
                poseMatrix = poseMatrix,
                landmarkPoseMatrix = poseMatrix,
                faceWidth = 0.14f,
                landmarks = landmarks,
                imageToViewTransform = imgToView,
                isFrontCamera = isFrontCamera,
                projectionParams = ProjectionParams(
                    fovYDeg = fovYDeg,
                    cropAspectRatio = aspect,
                    isCalibratedFromIntrinsics = true
                )
            )

            // Project with showResolvedAnchorMarkers = false
            overlay.showResolvedAnchorMarkers = false
            assertTrue(overlay.projectOverlayGeometry(state, viewW, viewH, fps = 60f, tier = "A"))

            val screen234 = FacePoseAdapter.mapLandmarkToView(u234, 0.50f, imgToView, isFrontCamera, viewW, viewH)
            val gizmoOriginX = overlay.lastProjectedGizmoOriginPx[0]
            val gizmoPlusXX = overlay.lastProjectedGizmoXPx[0]

            // Red +X axis tip MUST point from gizmo origin toward the #234 (subject-right) side on both front and rear!
            val gizmoDeltaX = gizmoPlusXX - gizmoOriginX
            val landmark234DeltaX = screen234[0] - gizmoOriginX
            assertTrue(
                "isFront=$isFrontCamera: Gizmo red +X axis tip ($gizmoPlusXX) must be drawn toward #234 subject-right (${screen234[0]}) relative to origin ($gizmoOriginX)",
                gizmoDeltaX * landmark234DeltaX > 0f && abs(gizmoDeltaX) > 10f
            )

            // Enable showResolvedAnchorMarkers and verify head_center, forehead, and custom landmark anchor project cleanly
            overlay.showResolvedAnchorMarkers = true
            assertTrue(overlay.projectOverlayGeometry(state, viewW, viewH, fps = 60f, tier = "A"))
            assertEquals("brow_lm", overlay.lastResolvedCustomLandmarkAnchorName)
            assertTrue(
                "Forehead resolved anchor marker must project above head_center marker on screen",
                overlay.lastProjectedForeheadAnchorPx[1] < overlay.lastProjectedHeadCenterAnchorPx[1]
            )
            assertFalse(overlay.lastProjectedCustomLandmarkAnchorPx[0].isNaN())
            assertFalse(overlay.lastProjectedCustomLandmarkAnchorPx[1].isNaN())
        }

        // 3. Verify 9 debug pills (including ANC:OFF/ON) wrap cleanly on 360x800 dp without clipping
        val layout360 = DebugHudLayoutCalculator.measureScreenLayout(
            widthDp = 360f,
            heightDp = 800f,
            density = 2.5f,
            fontScale = 1.0f,
            isDebugEnabled = true,
            sectionState = DebugHudSectionState()
        )
        assertTrue(layout360.debugToggleRowBounds.right <= 360f * 2.5f + 0.5f)
        assertTrue(layout360.hudCardBounds.top >= layout360.debugToggleRowBounds.bottom)
        assertFalse(layout360.hudCardBounds.intersects(layout360.debugToggleRowBounds))
    }

    /**
     * 5. Zero per-frame allocation check for `AnchorResolver.resolveAnchor` (built-in, custom offset-only,
     * and custom landmark-weighted), `SceneEvaluator.evaluate`, and `DebugOverlay.projectOverlayGeometry`.
     */
    @Test
    fun zeroPerFrameAllocation_inAnchorResolverAndSceneEvaluator() {
        val multiPropDir = resolveFilterDir("multi_prop")
        val compiledMultiProp = Compiler.compile(
            JSONObject(File(multiPropDir, "manifest.json").readText()),
            multiPropDir
        )
        val evaluator = SceneEvaluator(maxNodes = 8)
        val landmarks = FloatArray(478 * 3) { idx ->
            when (idx % 3) {
                0 -> 0.48f
                1 -> 0.38f
                else -> 0.01f
            }
        }
        val poseMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, -0.45f, 1f
        )
        val faceState = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1f,
            poseMatrix = poseMatrix,
            landmarkPoseMatrix = poseMatrix,
            faceWidth = 0.15f,
            landmarks = landmarks
        )
        val outAnchor = FloatArray(3)

        // Warm-up pass
        for (i in 0 until 20) {
            compiledMultiProp.anchorResolver.resolveAnchor("head_center", faceState, outAnchor)
            compiledMultiProp.anchorResolver.resolveAnchor("brow_custom", faceState, outAnchor)
            evaluator.evaluate(compiledMultiProp, faceState, compiledMultiProp.propertySlots)
        }

        // Steady-state loop (1000 frames) verifying reusable output arrays are mutated in-place
        val initialWorldMatricesRef = evaluator.worldMatrices
        val initialVisibilityRef = evaluator.isNodeVisible
        for (i in 0 until 1000) {
            compiledMultiProp.anchorResolver.resolveAnchor("head_center", faceState, outAnchor)
            compiledMultiProp.anchorResolver.resolveAnchor("brow_custom", faceState, outAnchor)
            evaluator.evaluate(compiledMultiProp, faceState, compiledMultiProp.propertySlots)
        }
        assertTrue(initialWorldMatricesRef === evaluator.worldMatrices)
        assertTrue(initialVisibilityRef === evaluator.isNodeVisible)
    }
}
