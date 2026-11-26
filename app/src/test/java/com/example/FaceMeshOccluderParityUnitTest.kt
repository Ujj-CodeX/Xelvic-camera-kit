package com.example

import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.FaceMeshRenderer
import com.example.engine.render.OccluderPass
import com.example.engine.render.RenderGraph
import com.example.engine.scene.AnchorResolver
import com.example.engine.scene.BehaviorRunner
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.CompiledLighting
import com.example.engine.scene.CompiledMaterial
import com.example.engine.scene.CompiledNode
import com.example.engine.scene.PropertySlots
import com.example.engine.scene.SceneEvaluator
import com.example.engine.scene.SignalBus
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import com.example.engine.signal.SignalPipeline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-JVM unit tests for the FaceMeshRenderer / OccluderPass camera-space unprojection,
 * rigid pose correction (`T = M_pred * inv(M_landmark)`), view-ray `pushBackFW`,
 * pinhole round-trip across front/rear and wide/narrow aspects, and `near = 0.1f` / depth-bits telemetry.
 */
class FaceMeshOccluderParityUnitTest {

    @Test
    fun testRendererAndOccluderClipSpaceParityWithAndWithoutPredictionOffset() {
        val fovYCropDeg = 64.0f
        val cropAspect = 3f / 4f
        val viewAspect = 9f / 16f
        val projParams = ProjectionParams(
            fovYDeg = fovYCropDeg,
            cropAspectRatio = cropAspect,
            near = 0.1f,
            far = 10.0f,
            isCalibratedFromIntrinsics = true
        )

        // Landmark-time unfiltered pose (tz = -0.50m, tx = 0.02m, ty = -0.01m, yaw = 5 deg)
        val landmarkPose = buildRigidPoseMatrix(
            tx = 0.02f,
            ty = -0.01f,
            tz = -0.50f,
            yawDeg = 5.0f,
            pitchDeg = -3.0f,
            rollDeg = 2.0f
        )
        // Filtered + predicted pose at render time (shifted by +15mm X, -8mm Y, -10mm Z, +4 deg yaw)
        val predictedPose = buildRigidPoseMatrix(
            tx = 0.035f,
            ty = -0.018f,
            tz = -0.51f,
            yawDeg = 9.0f,
            pitchDeg = -1.5f,
            rollDeg = 3.5f
        )

        val landmarks = buildSyntheticLandmarksForPose(
            poseMatrix = landmarkPose,
            faceWidthMeters = 0.145f,
            fovYCropDeg = fovYCropDeg,
            cropAspect = cropAspect
        )
        val fw = FacePoseAdapter.computeFaceWidth(
            landmarks = landmarks,
            matrix = landmarkPose,
            imageAspectRatio = cropAspect,
            fovYCropDeg = fovYCropDeg
        )

        for (frontFacing in listOf(true, false)) {
            val state = FaceState(
                timestampNs = 2_050_000_000L,
                poseTimestampNs = 2_000_000_000L,
                lifecycle = FaceLifecycle.State.TRACKING,
                fadeWeight = 1.0f,
                isVisible = true,
                isFrontCamera = frontFacing,
                faceWidth = fw,
                poseMatrix = predictedPose,
                landmarkPoseMatrix = landmarkPose,
                landmarks = landmarks,
                projectionParams = projParams
            )

            val fovYViewDeg = projParams.visibleViewFovYDeg(viewAspect)
            val viewProj = FacePoseAdapter.createCameraProjection(
                fovYDeg = fovYViewDeg,
                aspectRatio = viewAspect,
                near = projParams.near,
                far = projParams.far,
                frontFacing = frontFacing
            )

            val faceMeshRenderer = FaceMeshRenderer()
            val occluderPass = OccluderPass()

            assertTrue(faceMeshRenderer.updateVertices(state, viewportAspect = viewAspect))
            faceMeshRenderer.render(
                viewProjectionMatrix = viewProj,
                textureId = 1,
                maskId = null,
                opacity = 1.0f,
                program = 1,
                frontFacing = frontFacing,
                faceState = state
            )

            occluderPass.renderFaceMesh(
                faceState = state,
                viewProjectionMatrix = viewProj,
                program = 1,
                pushBackFW = 0f,
                viewportAspect = viewAspect
            )

            // 1. Camera-space vertex positions must match identically when pushBackFW == 0f
            for (i in 0 until UvRegistry.VERTEX_COUNT * 3) {
                assertEquals(
                    "Camera-space coordinate [$i] must match between FaceMeshRenderer and OccluderPass (front=$frontFacing)",
                    faceMeshRenderer.livePositions[i],
                    occluderPass.liveFaceMeshPositions[i],
                    1e-6f
                )
            }

            // 2. Effective uMVP (viewProjection * T) must match identically
            for (m in 0 until 16) {
                assertEquals(
                    "Pose-corrected uMVP[$m] must match between FaceMeshRenderer and OccluderPass",
                    faceMeshRenderer.lastEffectiveMvp[m],
                    occluderPass.lastEffectiveFaceMeshMvp[m],
                    1e-6f
                )
            }

            // 3. Clip-space 4D positions [cx, cy, cz, cw] must match for all 468 vertices
            val rendererClip = FaceMeshRenderer.computeClipSpacePositions(
                cameraPositions = faceMeshRenderer.livePositions,
                mvpMatrix = faceMeshRenderer.lastEffectiveMvp
            )
            val occluderClip = FaceMeshRenderer.computeClipSpacePositions(
                cameraPositions = occluderPass.liveFaceMeshPositions,
                mvpMatrix = occluderPass.lastEffectiveFaceMeshMvp
            )
            for (i in rendererClip.indices) {
                assertEquals(
                    "Clip-space coordinate [$i] must be identical between FaceMeshRenderer and OccluderPass (front=$frontFacing)",
                    rendererClip[i],
                    occluderClip[i],
                    1e-5f
                )
            }

            // 4. Verify rigid correction T = M_pred * inv(M_landmark) maps landmark-time origin to predicted origin
            val tMat = faceMeshRenderer.lastRigidCorrectionMatrix
            val lmOriginX = landmarkPose[12]
            val lmOriginY = landmarkPose[13]
            val lmOriginZ = landmarkPose[14]
            val correctedX = tMat[0] * lmOriginX + tMat[4] * lmOriginY + tMat[8] * lmOriginZ + tMat[12]
            val correctedY = tMat[1] * lmOriginX + tMat[5] * lmOriginY + tMat[9] * lmOriginZ + tMat[13]
            val correctedZ = tMat[2] * lmOriginX + tMat[6] * lmOriginY + tMat[10] * lmOriginZ + tMat[14]
            assertEquals("Rigid correction T must map landmark-time X to predicted X", predictedPose[12], correctedX, 1e-5f)
            assertEquals("Rigid correction T must map landmark-time Y to predicted Y", predictedPose[13], correctedY, 1e-5f)
            assertEquals("Rigid correction T must map landmark-time Z to predicted Z", predictedPose[14], correctedZ, 1e-5f)

            // 5. Verify face culling is disabled during face-mesh depth-only occluder draw
            assertFalse(
                "Face culling must be disabled for faceMesh depth-only occluder",
                occluderPass.lastFaceMeshCullFaceEnabled
            )
        }
    }

    @Test
    fun testViewRayPushBackPreservesScreenSpaceAndIncreasesDepth() {
        val fovYCropDeg = 60.0f
        val cropAspect = 3f / 4f
        val viewAspect = 9f / 16f
        val projParams = ProjectionParams(
            fovYDeg = fovYCropDeg,
            cropAspectRatio = cropAspect,
            near = 0.1f,
            far = 10.0f,
            isCalibratedFromIntrinsics = true
        )
        val pose = buildRigidPoseMatrix(tx = 0.04f, ty = -0.03f, tz = -0.48f, yawDeg = 0f, pitchDeg = 0f, rollDeg = 0f)
        val targetFw = 0.150f
        val landmarks = buildSyntheticLandmarksForPose(
            poseMatrix = pose,
            faceWidthMeters = targetFw,
            fovYCropDeg = fovYCropDeg,
            cropAspect = cropAspect
        )

        val state = FaceState(
            timestampNs = 1_000_000_000L,
            poseTimestampNs = 1_000_000_000L,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            isFrontCamera = true,
            faceWidth = targetFw,
            poseMatrix = pose,
            landmarkPoseMatrix = pose,
            landmarks = landmarks,
            projectionParams = projParams
        )

        val viewProj = FacePoseAdapter.createCameraProjection(
            fovYDeg = projParams.visibleViewFovYDeg(viewAspect),
            aspectRatio = viewAspect,
            near = projParams.near,
            far = projParams.far,
            frontFacing = true
        )

        val posUnpushed = FloatArray(UvRegistry.VERTEX_COUNT * 3)
        val posPushed = FloatArray(UvRegistry.VERTEX_COUNT * 3)
        val pushBackFW = 0.08f // 8% of FW = 12mm along each vertex's view ray

        assertTrue(
            FaceMeshRenderer.unprojectLandmarksToCameraSpace(
                faceState = state,
                viewportAspect = viewAspect,
                pushBackFW = 0f,
                outPositions = posUnpushed
            )
        )
        assertTrue(
            FaceMeshRenderer.unprojectLandmarksToCameraSpace(
                faceState = state,
                viewportAspect = viewAspect,
                pushBackFW = pushBackFW,
                outPositions = posPushed
            )
        )

        val clipUnpushed = FaceMeshRenderer.computeClipSpacePositions(posUnpushed, viewProj)
        val clipPushed = FaceMeshRenderer.computeClipSpacePositions(posPushed, viewProj)

        val expectedRayDeltaMeters = pushBackFW * targetFw
        for (v in 0 until UvRegistry.VERTEX_COUNT) {
            val b3 = v * 3
            val ux = posUnpushed[b3]; val uy = posUnpushed[b3 + 1]; val uz = posUnpushed[b3 + 2]
            val px = posPushed[b3]; val py = posPushed[b3 + 1]; val pz = posPushed[b3 + 2]

            val lenU = sqrt(ux * ux + uy * uy + uz * uz)
            val lenP = sqrt(px * px + py * py + pz * pz)
            assertEquals(
                "Vertex $v ray length must increase by pushBackFW * FW",
                lenU + expectedRayDeltaMeters,
                lenP,
                1e-5f
            )

            // Verify screen-space NDC (x_ndc, y_ndc) is unchanged while NDC depth (z_ndc) increases
            val b4 = v * 4
            val ndcXu = clipUnpushed[b4] / clipUnpushed[b4 + 3]
            val ndcYu = clipUnpushed[b4 + 1] / clipUnpushed[b4 + 3]
            val ndcZu = clipUnpushed[b4 + 2] / clipUnpushed[b4 + 3]

            val ndcXp = clipPushed[b4] / clipPushed[b4 + 3]
            val ndcYp = clipPushed[b4 + 1] / clipPushed[b4 + 3]
            val ndcZp = clipPushed[b4 + 2] / clipPushed[b4 + 3]

            assertEquals(
                "View-ray pushback must not shift NDC X for vertex $v",
                ndcXu,
                ndcXp,
                1e-5f
            )
            assertEquals(
                "View-ray pushback must not shift NDC Y for vertex $v",
                ndcYu,
                ndcYp,
                1e-5f
            )
            assertTrue(
                "View-ray pushback must increase NDC depth Z for vertex $v (was $ndcZu, now $ndcZp)",
                ndcZp > ndcZu
            )
        }
    }

    @Test
    fun testIndependentUnprojectionChecksFwCentroidAndPinholeRoundTripAcrossAspectsAndCameras() {
        val cropAspects = listOf(3f / 4f, 9f / 16f, 4f / 3f)
        val viewAspects = listOf(9f / 16f, 3f / 4f, 16f / 9f)
        val cameras = listOf(true, false)

        for (cropAspect in cropAspects) {
            for (viewAspect in viewAspects) {
                for (frontFacing in cameras) {
                    val fovYCropDeg = 62.0f
                    val projParams = ProjectionParams(
                        fovYDeg = fovYCropDeg,
                        cropAspectRatio = cropAspect,
                        near = 0.1f,
                        far = 10.0f,
                        isCalibratedFromIntrinsics = true
                    )

                    val tx = 0.03f
                    val ty = -0.02f
                    val tz = -0.52f
                    val pose = buildRigidPoseMatrix(tx = tx, ty = ty, tz = tz, yawDeg = 0f, pitchDeg = 0f, rollDeg = 0f)
                    val expectedFw = 0.142f
                    val landmarks = buildSyntheticLandmarksForPose(
                        poseMatrix = pose,
                        faceWidthMeters = expectedFw,
                        fovYCropDeg = fovYCropDeg,
                        cropAspect = cropAspect
                    )

                    // 1. Pinhole round-trip check: inject a known 3D point (knownX, knownY, knownZ) at vertex #10 (forehead)
                    val absZ = abs(tz)
                    val f = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)
                    val wCam = 2.0f * absZ * cropAspect / f
                    val hCam = 2.0f * absZ / f
                    val knownX = -0.045f
                    val knownY = 0.065f
                    val knownZ = -0.495f // 25mm closer to camera than -0.52m
                    landmarks[10 * 3] = (knownX / wCam) + 0.5f
                    landmarks[10 * 3 + 1] = -(knownY / hCam) + 0.5f
                    landmarks[10 * 3 + 2] = (-absZ - knownZ) / wCam

                    val measuredFw = FacePoseAdapter.computeFaceWidth(
                        landmarks = landmarks,
                        matrix = pose,
                        imageAspectRatio = cropAspect,
                        fovYCropDeg = fovYCropDeg
                    )
                    assertEquals("computeFaceWidth must match expectedFw", expectedFw, measuredFw, 1e-4f)

                    val state = FaceState(
                        timestampNs = 1_000_000_000L,
                        poseTimestampNs = 1_000_000_000L,
                        lifecycle = FaceLifecycle.State.TRACKING,
                        fadeWeight = 1.0f,
                        isVisible = true,
                        isFrontCamera = frontFacing,
                        faceWidth = measuredFw,
                        poseMatrix = pose,
                        landmarkPoseMatrix = pose,
                        landmarks = landmarks,
                        projectionParams = projParams
                    )

                    val unprojected = FloatArray(UvRegistry.VERTEX_COUNT * 3)
                    assertTrue(
                        FaceMeshRenderer.unprojectLandmarksToCameraSpace(
                            faceState = state,
                            viewportAspect = viewAspect,
                            pushBackFW = 0f,
                            outPositions = unprojected
                        )
                    )

                    // Check A: Distance between unprojected vertices 234 and 454 equals FW within tolerance
                    val i234 = 234 * 3
                    val i454 = 454 * 3
                    val dx = unprojected[i454] - unprojected[i234]
                    val dy = unprojected[i454 + 1] - unprojected[i234 + 1]
                    val dz = unprojected[i454 + 2] - unprojected[i234 + 2]
                    val unprojectedCheekDist = sqrt(dx * dx + dy * dy + dz * dz)
                    assertEquals(
                        "Distance between unprojected #234 and #454 must equal FW (cropAspect=$cropAspect, viewAspect=$viewAspect, front=$frontFacing)",
                        expectedFw,
                        unprojectedCheekDist,
                        1e-4f
                    )

                    // Check B: Mesh centroid and cheek midpoint within tolerance of pose translation (tx, ty, tz)
                    val cheekMidX = 0.5f * (unprojected[i234] + unprojected[i454])
                    val cheekMidY = 0.5f * (unprojected[i234 + 1] + unprojected[i454 + 1])
                    val cheekMidZ = 0.5f * (unprojected[i234 + 2] + unprojected[i454 + 2])
                    assertEquals("Cheek midpoint X must match pose tx", tx, cheekMidX, 1e-4f)
                    assertEquals("Cheek midpoint Y must match pose ty", ty, cheekMidY, 1e-4f)
                    assertEquals("Cheek midpoint Z must match pose tz", tz, cheekMidZ, 1e-4f)

                    var sumX = 0f
                    var sumY = 0f
                    var sumZ = 0f
                    for (v in 0 until UvRegistry.VERTEX_COUNT) {
                        sumX += unprojected[v * 3]
                        sumY += unprojected[v * 3 + 1]
                        sumZ += unprojected[v * 3 + 2]
                    }
                    val centroidX = sumX / UvRegistry.VERTEX_COUNT
                    val centroidY = sumY / UvRegistry.VERTEX_COUNT
                    val centroidZ = sumZ / UvRegistry.VERTEX_COUNT
                    assertEquals("Mesh centroid X within tolerance of pose tx", tx, centroidX, 0.01f)
                    assertEquals("Mesh centroid Y within tolerance of pose ty", ty, centroidY, 0.01f)
                    assertEquals("Mesh centroid Z within tolerance of pose tz", tz, centroidZ, 0.01f)

                    // Check C: Pinhole round-trip at vertex #10 recovers (knownX, knownY, knownZ)
                    assertEquals("Pinhole round-trip X", knownX, unprojected[10 * 3], 1e-5f)
                    assertEquals("Pinhole round-trip Y", knownY, unprojected[10 * 3 + 1], 1e-5f)
                    assertEquals("Pinhole round-trip Z", knownZ, unprojected[10 * 3 + 2], 1e-5f)
                }
            }
        }
    }

    @Test
    fun testNodeOffsetAndScaleIgnoredForFaceMeshAndNearPlaneAndDepthBitsTelemetry() {
        // 1. Verify ProjectionParams.DEFAULT_NEAR is 0.1f
        assertEquals(0.1f, ProjectionParams.DEFAULT_NEAR, 1e-6f)
        assertEquals(0.1f, ProjectionParams.DEFAULT.near, 1e-6f)

        // 2. Verify SignalPipeline carries snapshot.rawPoseMatrix into FaceState.landmarkPoseMatrix
        val pipeline = SignalPipeline()
        val rawPose = buildRigidPoseMatrix(tx = 0.05f, ty = -0.02f, tz = -0.46f, yawDeg = 12f, pitchDeg = -4f, rollDeg = 3f)
        val lm = buildSyntheticLandmarksForPose(
            poseMatrix = rawPose,
            faceWidthMeters = 0.14f,
            fovYCropDeg = 60f,
            cropAspect = 3f / 4f
        )
        val snap = FaceGeometrySnapshot.create(
            timestampNs = 1_000_000_000L,
            imageWidth = 480,
            imageHeight = 640,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = lm,
            rawPoseMatrix = rawPose,
            blendshapes = emptyMap(),
            sensorTimestampNs = 1_000_000_000L
        )
        val state = pipeline.processSnapshot(snap)
        for (i in 0 until 16) {
            assertEquals(
                "FaceState.landmarkPoseMatrix[$i] must preserve tracker-time unfiltered rawPoseMatrix",
                rawPose[i],
                state.landmarkPoseMatrix[i],
                1e-6f
            )
        }

        // 3. Verify RenderGraph ignores node basePosition / baseScale for faceMesh occluder and faceMesh renderer
        val nonIdentityOccluderNode = CompiledNode(
            index = 0,
            id = "face_occ",
            parentIndex = -1,
            tags = emptyList(),
            type = "occluder",
            anchor = "head_center",
            basePosition = floatArrayOf(0.5f, -0.4f, 0.3f), // non-zero offset (must be ignored)
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
            baseScale = floatArrayOf(2.5f, 2.5f, 2.5f),     // non-unit scale (must be ignored)
            rotationFollow = "full",
            materialId = null,
            baseVisible = true,
            baseOpacity = 1f,
            renderOrder = 0,
            occluderShape = "faceMesh",
            pushBackFW = 0f
        )
        val nonIdentityFaceMeshNode = CompiledNode(
            index = 1,
            id = "face_paint",
            parentIndex = -1,
            tags = emptyList(),
            type = "faceMesh",
            anchor = "head_center",
            basePosition = floatArrayOf(-0.3f, 0.2f, -0.1f), // must be ignored
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
            baseScale = floatArrayOf(1.8f, 1.8f, 1.8f),      // must be ignored
            rotationFollow = "full",
            materialId = "mat_face",
            baseVisible = true,
            baseOpacity = 1f,
            renderOrder = 1
        )
        val nodeList = listOf(nonIdentityOccluderNode, nonIdentityFaceMeshNode)
        val effect = CompiledEffect(
            packageId = "test_effect",
            packageName = "Test Effect",
            schemaVersion = "1.0",
            nodes = nodeList,
            materials = mapOf(
                "mat_face" to CompiledMaterial(
                    id = "mat_face",
                    template = "faceTexture@1",
                    mapAssetId = null,
                    maskAssetId = null,
                    blend = "alpha",
                    opacity = 1f
                )
            ),
            assets = emptyMap(),
            lighting = CompiledLighting(),
            signalBus = SignalBus(totalSlots = 0, derivedSignals = emptyList(), slotIndexMap = emptyMap()),
            behaviorRunner = BehaviorRunner(emptyList(), emptyList(), emptyList(), emptyList(), nodeCount = 2),
            anchorResolver = AnchorResolver(),
            propertySlots = PropertySlots(nodeCount = 2),
            tierOverrides = emptyMap()
        )

        val visibleState = state.copy(
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true
        )
        val evaluator = SceneEvaluator(maxNodes = 8)
        evaluator.evaluate(effect, visibleState, effect.propertySlots)

        val renderGraph = RenderGraph()
        renderGraph.render(
            effect = effect,
            faceState = visibleState,
            evaluator = evaluator,
            viewportWidth = 1080,
            viewportHeight = 1920
        )

        val expectedPositions = FloatArray(UvRegistry.VERTEX_COUNT * 3)
        FaceMeshRenderer.unprojectLandmarksToCameraSpace(
            faceState = visibleState,
            viewportAspect = 1080f / 1920f,
            pushBackFW = 0f,
            outPositions = expectedPositions
        )
        for (i in expectedPositions.indices) {
            assertEquals(
                "OccluderPass faceMesh must ignore node offset/scale",
                expectedPositions[i],
                renderGraph.occluder.liveFaceMeshPositions[i],
                1e-6f
            )
            assertEquals(
                "FaceMeshRenderer must ignore node offset/scale",
                expectedPositions[i],
                renderGraph.faceMesh.livePositions[i],
                1e-6f
            )
        }

        // 4. Verify depthBits reporting on LatencyProbe CSV and DebugOverlay HUD
        val probe = LatencyProbe()
        probe.recordDepthBits(24)
        probe.recordRenderFrame(renderTimestampNs = 1_500_000_000L, faceState = visibleState, predictionOn = true)
        val csvLines = probe.dumpCsv().trim().lines()
        val headerCols = csvLines.first().split(",")
        val depthBitsIdx = headerCols.indexOf("depthBits")
        assertTrue("CSV header must include depthBits", depthBitsIdx >= 0)
        assertEquals("24", csvLines.last().split(",")[depthBitsIdx])

        val overlay = DebugOverlay(latencyProbe = probe)
        overlay.projectOverlayGeometry(visibleState, viewWidth = 1080f, viewHeight = 1920f)
        assertTrue(
            "DebugOverlay HUD must report depth=24b (was: ${overlay.cachedProjectionHudLine})",
            overlay.cachedProjectionHudLine.contains("depth=24b")
        )
    }

    private fun buildRigidPoseMatrix(
        tx: Float,
        ty: Float,
        tz: Float,
        yawDeg: Float,
        pitchDeg: Float,
        rollDeg: Float
    ): FloatArray {
        val yaw = Math.toRadians(yawDeg.toDouble()).toFloat()
        val pitch = Math.toRadians(pitchDeg.toDouble()).toFloat()
        val roll = Math.toRadians(rollDeg.toDouble()).toFloat()

        val cy = cos(yaw); val sy = sin(yaw)
        val cp = cos(pitch); val sp = sin(pitch)
        val cr = cos(roll); val sr = sin(roll)

        // Column-major R = Rz(roll) * Rx(pitch) * Ry(yaw)
        val m = FloatArray(16)
        m[0] = cr * cy - sr * sp * sy
        m[1] = sr * cy + cr * sp * sy
        m[2] = -cp * sy
        m[3] = 0f

        m[4] = -sr * cp
        m[5] = cr * cp
        m[6] = sp
        m[7] = 0f

        m[8] = cr * sy + sr * sp * cy
        m[9] = sr * sy - cr * sp * cy
        m[10] = cp * cy
        m[11] = 0f

        m[12] = tx
        m[13] = ty
        m[14] = tz
        m[15] = 1f
        return m
    }

    private fun buildSyntheticLandmarksForPose(
        poseMatrix: FloatArray,
        faceWidthMeters: Float,
        fovYCropDeg: Float,
        cropAspect: Float
    ): FloatArray {
        val tx = poseMatrix[12]
        val ty = poseMatrix[13]
        val tz = poseMatrix[14]
        val absZ = abs(tz).coerceIn(0.25f, 1.20f)
        val f = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)
        val wCam = 2.0f * absZ * cropAspect / f
        val hCam = 2.0f * absZ / f

        val lm = FloatArray(478 * 3)
        val uvs = UvRegistry.CANONICAL_UVS
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            // Distribute vertices symmetrically around (tx, ty, tz) so mesh centroid == (tx, ty, tz)
            val u = uvs[i * 2] - 0.5f
            val v = uvs[i * 2 + 1] - 0.5f
            val vx = tx + u * faceWidthMeters * 0.5f
            val vy = ty - v * faceWidthMeters * 0.5f
            val vz = tz

            lm[i * 3] = (vx / wCam) + 0.5f
            lm[i * 3 + 1] = -(vy / hCam) + 0.5f
            lm[i * 3 + 2] = (-absZ - vz) / wCam
        }

        // Place #234 (subject right cheek) and #454 (subject left cheek) at +/- faceWidthMeters / 2 around (tx, ty, tz)
        val halfFw = faceWidthMeters * 0.5f
        val x234 = tx - halfFw
        val x454 = tx + halfFw
        lm[234 * 3] = (x234 / wCam) + 0.5f
        lm[234 * 3 + 1] = -(ty / hCam) + 0.5f
        lm[234 * 3 + 2] = 0f

        lm[454 * 3] = (x454 / wCam) + 0.5f
        lm[454 * 3 + 1] = -(ty / hCam) + 0.5f
        lm[454 * 3 + 2] = 0f

        return lm
    }
}
