package com.example

import com.example.engine.capture.FrameClock
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.pkg.Compiler as EffectCompiler
import com.example.engine.pkg.FaceRegions
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.GlResourceOwner
import com.example.engine.render.RegionMaskPass
import com.example.engine.render.RenderGraph
import com.example.engine.render.RtPool
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.ControlPointSmoother
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import kotlin.math.sqrt

/**
 * Unit tests for **Flow 3 — Chunk R1 (Generalized)**:
 * 1. Verified `FaceRegions` data table (`NAMED_LOOPS`, `getRegionLoop`) extracted and verified against
 *    pinned `UvRegistry.CANONICAL_UVS` and `UvRegistry.CANONICAL_TRIANGLES` (`FACE_OVAL`, `RIGHT_EYE_CONTOUR`,
 *    `LEFT_EYE_CONTOUR`, `LIPS_OUTER`, `MOUTH_INNER`, `RIGHT_EYEBROW`, `LEFT_EYEBROW`, `RIGHT_IRIS`,
 *    `LEFT_IRIS`, `NOSE_BRIDGE`, `RIGHT_CHEEK`, `LEFT_CHEEK`).
 * 2. Selective 1-Euro control-point smoothing (`ControlPointSmoother`) on `FaceRegions.CONTROL_POINT_INDICES`
 *    with stationary jitter reduction and clean reset on `NO_FACE`.
 * 3. Half-resolution `GL_R8` `RegionMaskPass` FBO allocation (`viewport / 2`), centroid-fan + feathered-skirt
 *    polygon triangulation (`1.0 -> 0.0` soft edge), and deterministic half-res `R8` rasterization.
 * 4. Composable `MaskExpr` producers (`LandmarkLoop`, `IrisCircle`, `LuminanceKey`, `TextureMask`) and
 *    algebra ops (`Union`, `Intersect`, `Subtract`, `Feather`, `Invert`), including built-in canonical
 *    composite masks (`$mask:skin`, `$mask:eyes`, `$mask:mouth`, `$mask:lips`, `$mask:iris`).
 * 5. Manifest `"masks"` compilation, `RtPool` `$mask:<id>` half-resolution `R8` target allocation, and
 *    dynamic `mask:<id>.weight` signal-to-parameter binding via `BehaviorRunner` and `PropertySlots`.
 * 6. Fast-motion mask-offset HUD telemetry (`|landmarkTime - cameraFrameTime|` in `ms` and projected
 *    screen-space shift in `px` during fast head motion) and generic `MASK:OFF/EYES/MOUTH/IRIS/SKIN`
 *    viewer mode cycling in `LatencyProbe` and `DebugOverlay`.
 */
@RunWith(RobolectricTestRunner::class)
class ChunkR1RegionMaskAndTelemetryUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testFaceRegionsVerifiedAgainstPinnedCanonicalTopologyAndNamedLoopsTable() {
        val report = FaceRegions.verifyAgainstPinnedTopology(
            uvs = UvRegistry.CANONICAL_UVS,
            triangles = UvRegistry.CANONICAL_TRIANGLES
        )

        assertTrue(
            "FaceRegions must pass all pinned topology checks against UvRegistry, errorReason=${report.errorReason}",
            report.isValid
        )
        assertNull(report.errorReason)
        assertTrue(
            "FACE_OVAL boundary edge match ratio (${report.boundaryEdgeMatchRatioOval}) must be >= 0.90",
            report.boundaryEdgeMatchRatioOval >= 0.90f
        )
        assertTrue(
            "Mesh edge connectivity ratio (${report.meshEdgeConnectivityRatio}) must be >= 0.95",
            report.meshEdgeConnectivityRatio >= 0.95f
        )

        // Verify exact sizes of region loops and iris sets
        assertEquals(36, FaceRegions.FACE_OVAL.size)
        assertEquals(16, FaceRegions.RIGHT_EYE_CONTOUR.size)
        assertEquals(16, FaceRegions.LEFT_EYE_CONTOUR.size)
        assertEquals(20, FaceRegions.LIPS_OUTER.size)
        assertEquals(20, FaceRegions.MOUTH_INNER.size)
        assertEquals(10, FaceRegions.RIGHT_EYEBROW.size)
        assertEquals(10, FaceRegions.LEFT_EYEBROW.size)
        assertEquals(5, FaceRegions.RIGHT_IRIS.size)
        assertEquals(5, FaceRegions.LEFT_IRIS.size)
        assertEquals(12, FaceRegions.NOSE_BRIDGE.size)
        assertEquals(8, FaceRegions.RIGHT_CHEEK.size)
        assertEquals(8, FaceRegions.LEFT_CHEEK.size)
        assertEquals(468, FaceRegions.RIGHT_IRIS_CENTER)
        assertEquals(473, FaceRegions.LEFT_IRIS_CENTER)

        // Verify NAMED_LOOPS lookup table for all canonical region keys
        val expectedKeys = listOf(
            "face_oval",
            "right_eye",
            "left_eye",
            "lips_outer",
            "lips_inner",
            "mouth_inner",
            "right_eyebrow",
            "left_eyebrow",
            "right_iris",
            "left_iris",
            "nose_bridge",
            "right_cheek",
            "left_cheek"
        )
        for (key in expectedKeys) {
            val loop = FaceRegions.getRegionLoop(key)
            assertNotNull("Expected FaceRegions.getRegionLoop('$key') to return non-null loop", loop)
            assertTrue("Loop '$key' must not be empty", loop!!.isNotEmpty())
        }
        assertSame(FaceRegions.MOUTH_INNER, FaceRegions.getRegionLoop("lips-inner"))

        // Verify CONTROL_POINT_INDICES is strictly deduplicated and sorted
        val controlPoints = FaceRegions.CONTROL_POINT_INDICES
        assertTrue(controlPoints.size in 120..175)
        for (i in 1 until controlPoints.size) {
            assertTrue(
                "CONTROL_POINT_INDICES must be strictly sorted and deduplicated",
                controlPoints[i] > controlPoints[i - 1]
            )
        }
    }

    @Test
    fun testControlPointSmootherReducesStationaryJitterAndLeavesNonControlPointsUntouched() {
        val smoother = ControlPointSmoother(minCutoff = 1.6f, beta = 0.035f, dCutoff = 1.0f)

        // Build base 478-landmark array initialized from canonical UVs
        val baseLandmarks = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            baseLandmarks[i * 3] = UvRegistry.CANONICAL_UVS[i * 2]
            baseLandmarks[i * 3 + 1] = UvRegistry.CANONICAL_UVS[i * 2 + 1]
            baseLandmarks[i * 3 + 2] = 0f
        }
        // Populate iris landmarks (468..477) around canonical eye centers
        for (i in 468..472) {
            baseLandmarks[i * 3] = 0.36f + (i - 468) * 0.005f
            baseLandmarks[i * 3 + 1] = 0.39f
        }
        for (i in 473..477) {
            baseLandmarks[i * 3] = 0.64f + (i - 473) * 0.005f
            baseLandmarks[i * 3 + 1] = 0.39f
        }

        // Pick a control point (e.g. landmark 10 on FACE_OVAL) and a non-control point (e.g. landmark 1, nose tip)
        val controlIdx = 10
        val nonControlIdx = 1
        assertTrue(FaceRegions.CONTROL_POINT_INDICES.contains(controlIdx))
        assertFalse(FaceRegions.CONTROL_POINT_INDICES.contains(nonControlIdx))

        var rawDiffSqSum = 0.0
        var smoothedDiffSqSum = 0.0
        val frameCount = 30
        val baseU = baseLandmarks[controlIdx * 3]

        for (frame in 0 until frameCount) {
            val noisy = baseLandmarks.clone()
            val sign = if (frame % 2 == 0) 1f else -1f
            val jitter = sign * 0.008f
            noisy[controlIdx * 3] = baseU + jitter
            noisy[nonControlIdx * 3] = 0.5f + jitter

            val tsNs = (frame + 1) * 33_333_333L
            val smoothed = smoother.smooth(
                rawLandmarks = noisy,
                timestampNs = tsNs,
                isTracking = true
            )!!
            // Non-control point must pass through unmodified
            assertEquals(noisy[nonControlIdx * 3], smoothed[nonControlIdx * 3], 1e-6f)

            if (frame >= 4) {
                val rawErr = (noisy[controlIdx * 3] - baseU).toDouble()
                val smoothErr = (smoothed[controlIdx * 3] - baseU).toDouble()
                rawDiffSqSum += rawErr * rawErr
                smoothedDiffSqSum += smoothErr * smoothErr
            }
        }

        val rawRms = sqrt(rawDiffSqSum / (frameCount - 4))
        val smoothRms = sqrt(smoothedDiffSqSum / (frameCount - 4))
        assertTrue(
            "Smoothed control-point RMS ($smoothRms) must be significantly lower than raw jitter RMS ($rawRms)",
            smoothRms < rawRms * 0.65
        )

        // Verify clean reset on NO_FACE
        assertNull(smoother.smooth(rawLandmarks = baseLandmarks, timestampNs = 2_000_000_000L, isTracking = false))
    }

    @Test
    fun testRegionMaskPassHalfResFboTriangulationSoftFeatherAndIrisEyelidIntersection() {
        val pass = RegionMaskPass()
        val owner = GlResourceOwner()
        owner.bindGlThread()
        pass.init(owner)
        assertTrue(pass.isInitialized)

        // 1. Half-resolution R8 FBO sizing & resize
        pass.ensureHalfResFbo(viewportWidth = 1080, viewportHeight = 1920, resourceOwner = owner)
        assertEquals(540, pass.fboWidth)
        assertEquals(960, pass.fboHeight)

        // Resize to 720x1280 -> 360x640
        pass.ensureHalfResFbo(viewportWidth = 720, viewportHeight = 1280, resourceOwner = owner)
        assertEquals(360, pass.fboWidth)
        assertEquals(640, pass.fboHeight)

        // 2. Build canonical test landmarks (478 * 3)
        val (faceState, eyeCx, eyeCy) = buildCanonicalFaceStateWithIris()
        val landmarks = faceState.landmarks!!

        // 3. Verify polygon centroid-fan + feathered-skirt triangulation count (27 floats per contour vertex = 3 triangles * 3 verts * 3 floats)
        val writtenFloats = pass.triangulateRegionIntoScratch(
            regionType = RegionMaskPass.RegionType.FACE_OVAL,
            smoothedLandmarks = landmarks,
            faceState = faceState,
            featherNorm = 0.025f
        )
        val triCount = (writtenFloats / 3) / 3
        assertEquals(3 * FaceRegions.FACE_OVAL.size, triCount)

        // 4. Verify soft-edge feathering: interior = 1.0, feather skirt in (0, 1), exterior = 0.0
        val centerWeight = pass.evaluateRegionMaskAtNormalizedViewPt(
            regionType = RegionMaskPass.RegionType.FACE_OVAL,
            normX = 0.50f,
            normY = 0.50f,
            faceState = faceState,
            featherNorm = 0.04f,
            useSmoothedLandmarks = false
        )
        assertEquals(1.0f, centerWeight, 1e-4f)

        val outsideWeight = pass.evaluateRegionMaskAtNormalizedViewPt(
            regionType = RegionMaskPass.RegionType.FACE_OVAL,
            normX = 0.02f,
            normY = 0.02f,
            faceState = faceState,
            featherNorm = 0.04f,
            useSmoothedLandmarks = false
        )
        assertEquals(0.0f, outsideWeight, 1e-4f)

        // Half-res R8 rasterization check
        val r8Buffer = pass.rasterizeRegionsToHalfResR8(
            faceState = faceState,
            viewportWidth = 128,
            viewportHeight = 128,
            regions = listOf(RegionMaskPass.RegionType.FACE_OVAL),
            featherNorm = 0.04f
        )
        assertEquals(64 * 64, r8Buffer.size)
        val midIdx = 32 * 64 + 32
        assertEquals(255, r8Buffer[midIdx].toInt() and 0xFF)
        assertEquals(0, r8Buffer[0].toInt() and 0xFF)
        val featherPixels = r8Buffer.count {
            val v = it.toInt() and 0xFF
            v in 1..254
        }
        assertTrue("Expected soft-edge feather band pixels in (1..254), got $featherPixels", featherPixels > 10)

        // 5. Verify iris disc ∩ eyelid polygon intersection when eyelid closes over upper iris
        val irisCenterWeightOpen = pass.evaluateRegionMaskAtNormalizedViewPt(
            regionType = RegionMaskPass.RegionType.RIGHT_IRIS_CLAMPED,
            normX = eyeCx,
            normY = eyeCy,
            faceState = faceState,
            featherNorm = 0.005f,
            useSmoothedLandmarks = false
        )
        assertTrue("Iris center must be visible when eye contour is open, got $irisCenterWeightOpen", irisCenterWeightOpen > 0.8f)

        // Collapse RIGHT_EYE_CONTOUR vertically around eyeCy + 0.08f (simulating closed eyelid below iris center)
        val closedEyeLandmarks = landmarks.clone()
        for (idx in FaceRegions.RIGHT_EYE_CONTOUR) {
            closedEyeLandmarks[idx * 3 + 1] = eyeCy + 0.08f
        }
        val closedFaceState = faceState.copy(landmarks = closedEyeLandmarks)
        val irisCenterWeightClosed = pass.evaluateRegionMaskAtNormalizedViewPt(
            regionType = RegionMaskPass.RegionType.RIGHT_IRIS_CLAMPED,
            normX = eyeCx,
            normY = eyeCy,
            faceState = closedFaceState,
            featherNorm = 0.005f,
            useSmoothedLandmarks = false
        )
        assertEquals(
            "Iris center must be clipped to 0.0 when outside the eyelid contour polygon",
            0.0f,
            irisCenterWeightClosed,
            1e-4f
        )

        pass.release(owner)
        owner.drainPendingDeletions()
    }

    @Test
    fun testComposableMaskProducersAndAlgebraOps() {
        val pass = RegionMaskPass()
        val (faceState, rightEyeCx, rightEyeCy) = buildCanonicalFaceStateWithIris()
        val landmarks = faceState.landmarks!!

        // Compute mouth inner center (#13 and #14 midpoint) and lower lip tissue point (between #14 inner and #17 outer)
        val mouthInnerCx = (landmarks[13 * 3] + landmarks[14 * 3]) * 0.5f
        val mouthInnerCy = (landmarks[13 * 3 + 1] + landmarks[14 * 3 + 1]) * 0.5f
        val lowerLipTissueX = landmarks[17 * 3]
        val lowerLipTissueY = (landmarks[14 * 3 + 1] + landmarks[17 * 3 + 1]) * 0.5f

        // Compute right cheek center
        var rCheekCx = 0f
        var rCheekCy = 0f
        for (idx in FaceRegions.RIGHT_CHEEK) {
            rCheekCx += landmarks[idx * 3]
            rCheekCy += landmarks[idx * 3 + 1]
        }
        rCheekCx /= FaceRegions.RIGHT_CHEEK.size
        rCheekCy /= FaceRegions.RIGHT_CHEEK.size

        // 1. $mask:skin = face_oval - (left_eye ∪ right_eye) - lips_outer - (left_eyebrow ∪ right_eyebrow)
        val skinAtCheek = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:skin",
            normX = rCheekCx,
            normY = rCheekCy,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals("Skin mask must be 1.0 on cheek skin", 1.0f, skinAtCheek, 1e-4f)

        val skinAtRightEye = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:skin",
            normX = rightEyeCx,
            normY = rightEyeCy,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals("Skin mask must subtract eye opening to 0.0", 0.0f, skinAtRightEye, 1e-4f)

        val skinAtMouth = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:skin",
            normX = mouthInnerCx,
            normY = mouthInnerCy,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals("Skin mask must subtract lips_outer to 0.0", 0.0f, skinAtMouth, 1e-4f)

        // 2. $mask:lips = lips_outer - mouth_inner
        val lipsOnLowerLip = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:lips",
            normX = lowerLipTissueX,
            normY = lowerLipTissueY,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertTrue("Lips mask must be > 0.8 on outer lip tissue, got $lipsOnLowerLip", lipsOnLowerLip > 0.8f)

        val lipsInsideOpenMouth = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:lips",
            normX = mouthInnerCx,
            normY = mouthInnerCy,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals("Lips mask must subtract inner mouth cavity to 0.0", 0.0f, lipsInsideOpenMouth, 1e-4f)

        // 3. $mask:iris = (right_iris_circle ∩ right_eye) ∪ (left_iris_circle ∩ left_eye)
        val irisOpenVal = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:iris",
            normX = rightEyeCx,
            normY = rightEyeCy,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals("Composable iris mask must be 1.0 at open iris center", 1.0f, irisOpenVal, 1e-4f)

        // Close right eye -> composable Intersect(IrisCircle, LandmarkLoop("right_eye")) clips iris to 0.0
        val closedEyeLm = landmarks.clone()
        for (idx in FaceRegions.RIGHT_EYE_CONTOUR) {
            closedEyeLm[idx * 3 + 1] = rightEyeCy + 0.08f
        }
        val closedState = faceState.copy(landmarks = closedEyeLm)
        val irisClosedVal = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:iris",
            normX = rightEyeCx,
            normY = rightEyeCy,
            faceState = closedState,
            useSmoothedLandmarks = false
        )
        assertEquals("Composable iris mask must be 0.0 when eyelid closes over iris", 0.0f, irisClosedVal, 1e-4f)

        // 4. LuminanceKey + Invert + Feather algebra ops
        val glareProtectedIrisExpr = RegionMaskPass.MaskExpr.Intersect(
            RegionMaskPass.BUILTIN_RIGHT_IRIS_MASK,
            RegionMaskPass.MaskExpr.LuminanceKey(
                sourceTarget = "\$camera",
                minLuma = 0.08f,
                maxLuma = 0.82f,
                feather = 0.05f
            )
        )
        val midtoneIrisWeight = pass.evaluateMaskExprAtNormalizedViewPt(
            expr = glareProtectedIrisExpr,
            normX = rightEyeCx,
            normY = rightEyeCy,
            faceState = faceState,
            sampleLuma = 0.45f,
            useSmoothedLandmarks = false
        )
        assertEquals("Midtone iris pixel must pass luminance key with weight 1.0", 1.0f, midtoneIrisWeight, 1e-4f)

        val glareIrisWeight = pass.evaluateMaskExprAtNormalizedViewPt(
            expr = glareProtectedIrisExpr,
            normX = rightEyeCx,
            normY = rightEyeCy,
            faceState = faceState,
            sampleLuma = 0.96f,
            useSmoothedLandmarks = false
        )
        assertEquals("Specular glare pixel (luma=0.96) must be suppressed to 0.0", 0.0f, glareIrisWeight, 1e-4f)

        val invertedSkinExpr = RegionMaskPass.MaskExpr.Invert(RegionMaskPass.BUILTIN_SKIN_MASK)
        val invertedAtCheek = pass.evaluateMaskExprAtNormalizedViewPt(
            expr = invertedSkinExpr,
            normX = rCheekCx,
            normY = rCheekCy,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals(0.0f, invertedAtCheek, 1e-4f)
    }

    @Test
    fun testManifestMasksCompilationRtPoolAllocationAndDynamicMaskWeightBinding() {
        val manifestJson = JSONObject(
            """
            {
              "schemaVersion": "1.12.0",
              "id": "r1_composable_masks_test",
              "name": "Composable Masks Test",
              "masks": {
                "glare_free_iris": {
                  "op": "intersect",
                  "a": "${'$'}mask:iris",
                  "b": {
                    "op": "luminanceKey",
                    "source": "${'$'}camera",
                    "minLuma": 0.05,
                    "maxLuma": 0.85,
                    "feather": 0.06
                  }
                },
                "blush_cheeks": {
                  "op": "feather",
                  "input": {
                    "op": "union",
                    "a": "right_cheek",
                    "b": "left_cheek"
                  },
                  "radius": 0.025
                }
              },
              "passes": [
                { "id": "cam_in", "stage": "CAMERA_FRAME", "writes": "${'$'}camera" },
                { "id": "skin_mask_pass", "template": "regionMask@1", "writes": "${'$'}mask:skin", "scale": "HALF" },
                { "id": "iris_mask_pass", "template": "regionMask@1", "writes": "${'$'}mask:glare_free_iris", "scale": "HALF" },
                { "id": "recolor_eyes", "template": "recolor@1", "reads": ["${'$'}camera", "${'$'}mask:glare_free_iris"], "writes": "${'$'}output" }
              ],
              "nodes": [
                { "id": "root", "type": "group" }
              ],
              "behaviors": [
                {
                  "id": "blink_fades_iris_mask",
                  "type": "map",
                  "input": "face.blend.eyeBlinkRight",
                  "inputRange": [0.0, 1.0],
                  "outputRange": [1.0, 0.0],
                  "easing": "linear",
                  "smoothingTau": 0.0,
                  "writes": [
                    { "path": "mask:glare_free_iris.weight", "mode": "set" }
                  ]
                }
              ]
            }
            """.trimIndent()
        )

        val compiled = EffectCompiler.compile(manifestJson, File("."))
        assertTrue(compiled.maskDefs.containsKey("skin"))
        assertTrue(compiled.maskDefs.containsKey("iris"))
        assertTrue(compiled.maskDefs.containsKey("glare_free_iris"))
        assertTrue(compiled.maskDefs.containsKey("blush_cheeks"))

        val (faceStateOpen, rightEyeCx, rightEyeCy) = buildCanonicalFaceStateWithIris(
            blendshapes = mapOf("eyeBlinkRight" to 0.0f)
        )
        val evaluator = SceneEvaluator(maxNodes = 16)
        val renderGraph = RenderGraph()
        renderGraph.init()

        // Step signals & behaviors with eye open (eyeBlinkRight = 0.0 -> mask:glare_free_iris.weight = 1.0)
        compiled.signalBus.evaluate(faceStateOpen, 1.0f)
        compiled.behaviorRunner.step(
            0.016f,
            compiled.signalBus,
            faceStateOpen,
            compiled.propertySlots,
            1000L
        )
        assertEquals(1.0f, compiled.propertySlots.getDynamicParam("mask:glare_free_iris.weight", 1.0f), 1e-3f)

        evaluator.evaluate(compiled, faceStateOpen, compiled.propertySlots)
        renderGraph.render(
            effect = compiled,
            faceState = faceStateOpen,
            evaluator = evaluator,
            viewportWidth = 1080,
            viewportHeight = 1920
        )

        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin"))
        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:glare_free_iris"))

        val skinRt = renderGraph.rtPool.getNamedTarget("\$mask:skin")
        val irisRt = renderGraph.rtPool.getNamedTarget("\$mask:glare_free_iris")
        assertNotNull(skinRt)
        assertNotNull(irisRt)
        assertEquals(540, skinRt!!.width)
        assertEquals(960, skinRt.height)
        assertEquals(RtPool.RtFormat.R8, skinRt.format)
        assertEquals(540, irisRt!!.width)
        assertEquals(960, irisRt.height)
        assertEquals(RtPool.RtFormat.R8, irisRt.format)

        // Step signals & behaviors with eye 75% closed (eyeBlinkRight = 0.75 -> mask:glare_free_iris.weight = 0.25)
        val faceStateBlinking = faceStateOpen.copy(blendshapes = mapOf("eyeBlinkRight" to 0.75f))
        compiled.signalBus.evaluate(faceStateBlinking, 1.033f)
        compiled.behaviorRunner.step(
            0.016f,
            compiled.signalBus,
            faceStateBlinking,
            compiled.propertySlots,
            1033L
        )
        val dynamicWeight = compiled.propertySlots.getDynamicParam("mask:glare_free_iris.weight", 1.0f)
        assertEquals(0.25f, dynamicWeight, 1e-2f)

        val weightedMaskVal = renderGraph.regionMask.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "glare_free_iris",
            normX = rightEyeCx,
            normY = rightEyeCy,
            faceState = faceStateBlinking,
            customMasks = compiled.maskDefs,
            maskWeight = dynamicWeight,
            sampleLuma = 0.45f,
            useSmoothedLandmarks = false
        )
        assertEquals(0.25f, weightedMaskVal, 1e-2f)

        renderGraph.release()
    }

    @Test
    fun testFastMotionMaskOffsetTelemetryAndDebugHudFormatting() {
        var fakeNowNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeNowNs }

        val probe = LatencyProbe()
        val overlay = DebugOverlay(latencyProbe = probe)

        // 1. Stationary head pose with 25ms landmark-to-camera delta -> motion: STILL, 0.0px shift
        val stillState = FaceState(
            lifecycle = FaceLifecycle.State.TRACKING,
            isVisible = true,
            fadeWeight = 1.0f,
            faceWidth = 0.14f,
            translation = floatArrayOf(0f, 0f, -0.50f),
            poseTimestampNs = 975_000_000L,
            velocity = floatArrayOf(0.01f, 0f, 0f),
            angularVelocity = floatArrayOf(0f, 0f, 0f),
            projectionParams = ProjectionParams.DEFAULT
        )
        probe.recordMaskMotionOffset(
            renderTimestampNs = fakeNowNs,
            cameraSensorTimestampNs = 1_000_000_000L,
            faceState = stillState,
            viewportWidth = 1080,
            viewportHeight = 1920
        )

        assertEquals(25.0f, probe.lastMaskTimeDeltaMs, 0.1f)
        assertFalse("Stationary head must not trigger fast head motion flag", probe.lastIsFastHeadMotion)
        assertTrue("Stationary head offset px should be small (< 3px)", probe.lastMaskOffsetPx < 3.0f)

        overlay.refreshCachedHudStringsIfNeeded(
            faceState = stillState,
            fps = 60f,
            tier = "A",
            force = true
        )
        assertTrue(
            "HUD line must report STILL motion when head is stationary: ${overlay.cachedMaskOffsetLine}",
            overlay.cachedMaskOffsetLine.contains("motion: STILL")
        )
        assertTrue(
            "MaskView HUD line must default to MASK:OFF: ${overlay.cachedMaskViewerLine}",
            overlay.cachedMaskViewerLine.contains("MASK:OFF")
        )

        // Cycle DebugMaskMode: OFF -> EYES -> MOUTH -> IRIS -> SKIN -> OFF
        var mode = RegionMaskPass.DebugMaskMode.OFF
        mode = mode.next()
        assertEquals(RegionMaskPass.DebugMaskMode.EYES, mode)
        mode = mode.next()
        assertEquals(RegionMaskPass.DebugMaskMode.MOUTH, mode)
        mode = mode.next()
        assertEquals(RegionMaskPass.DebugMaskMode.IRIS, mode)
        mode = mode.next()
        assertEquals(RegionMaskPass.DebugMaskMode.SKIN, mode)
        overlay.debugMaskMode = mode
        overlay.refreshCachedHudStringsIfNeeded(
            faceState = stillState,
            fps = 60f,
            tier = "A",
            force = true
        )
        assertTrue(overlay.cachedMaskViewerLine.contains("MASK:SKIN"))
        assertTrue(overlay.cachedMaskViewerLine.contains("\$mask:skin"))
        assertEquals(RegionMaskPass.DebugMaskMode.OFF, mode.next())

        // 2. Fast head turn (0.45 m/s linear + 95 deg/s yaw with 33.3ms landmark lag behind camera frame)
        fakeNowNs = 1_600_000_000L
        val fastYawRadPerSec = Math.toRadians(95.0).toFloat()
        val fastState = FaceState(
            lifecycle = FaceLifecycle.State.TRACKING,
            isVisible = true,
            fadeWeight = 1.0f,
            faceWidth = 0.14f,
            translation = floatArrayOf(0f, 0f, -0.50f),
            poseTimestampNs = 1_566_700_000L,
            velocity = floatArrayOf(0.45f, 0.05f, 0f),
            angularVelocity = floatArrayOf(0f, fastYawRadPerSec, 0f),
            projectionParams = ProjectionParams.DEFAULT
        )
        probe.recordMaskMotionOffset(
            renderTimestampNs = fakeNowNs,
            cameraSensorTimestampNs = 1_600_000_000L,
            faceState = fastState,
            viewportWidth = 1080,
            viewportHeight = 1920
        )

        assertTrue("Fast head motion must be detected", probe.lastIsFastHeadMotion)
        assertEquals(33.3f, probe.lastMaskTimeDeltaMs, 0.2f)
        assertEquals(33.3f, probe.rollingFastMotionMaskDeltaMs, 0.2f)
        assertTrue(
            "Projected screen-space mask offset during fast head motion must be > 15px, got ${probe.lastMaskOffsetPx}",
            probe.lastMaskOffsetPx > 15.0f
        )
        assertEquals(1, probe.fastMotionSampleCount)
        assertTrue(probe.fastMotionMaskOffsetP95Px > 15.0f)

        overlay.refreshCachedHudStringsIfNeeded(
            faceState = fastState,
            fps = 60f,
            tier = "A",
            force = true
        )
        val fastLine = overlay.cachedMaskOffsetLine
        assertTrue("Expected MaskOffset prefix in '$fastLine'", fastLine.startsWith("MaskOffset: dt="))
        assertTrue("Expected fast delta in '$fastLine'", fastLine.contains("(fast: 33.3ms)"))
        assertTrue("Expected FAST motion label in '$fastLine'", fastLine.contains("motion: FAST("))
        assertTrue(
            "Expected non-zero fast p95/max px in '$fastLine'",
            !fastLine.contains("fast p95/max: 0.0/0.0px")
        )
    }

    private fun buildCanonicalFaceStateWithIris(
        blendshapes: Map<String, Float> = emptyMap()
    ): Triple<FaceState, Float, Float> {
        val landmarks = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            landmarks[i * 3] = UvRegistry.CANONICAL_UVS[i * 2]
            landmarks[i * 3 + 1] = UvRegistry.CANONICAL_UVS[i * 2 + 1]
            landmarks[i * 3 + 2] = 0f
        }
        // Open inner mouth slightly (#13 upper inner lip, #14 lower inner lip) so mouth_inner has non-zero area
        landmarks[13 * 3 + 1] -= 0.015f
        landmarks[14 * 3 + 1] += 0.015f

        var rEyeCx = 0f
        var rEyeCy = 0f
        for (idx in FaceRegions.RIGHT_EYE_CONTOUR) {
            rEyeCx += landmarks[idx * 3]
            rEyeCy += landmarks[idx * 3 + 1]
        }
        rEyeCx /= FaceRegions.RIGHT_EYE_CONTOUR.size
        rEyeCy /= FaceRegions.RIGHT_EYE_CONTOUR.size

        var lEyeCx = 0f
        var lEyeCy = 0f
        for (idx in FaceRegions.LEFT_EYE_CONTOUR) {
            lEyeCx += landmarks[idx * 3]
            lEyeCy += landmarks[idx * 3 + 1]
        }
        lEyeCx /= FaceRegions.LEFT_EYE_CONTOUR.size
        lEyeCy /= FaceRegions.LEFT_EYE_CONTOUR.size

        val irisRadius = 0.022f
        landmarks[468 * 3] = rEyeCx; landmarks[468 * 3 + 1] = rEyeCy
        landmarks[469 * 3] = rEyeCx + irisRadius; landmarks[469 * 3 + 1] = rEyeCy
        landmarks[470 * 3] = rEyeCx; landmarks[470 * 3 + 1] = rEyeCy - irisRadius
        landmarks[471 * 3] = rEyeCx - irisRadius; landmarks[471 * 3 + 1] = rEyeCy
        landmarks[472 * 3] = rEyeCx; landmarks[472 * 3 + 1] = rEyeCy + irisRadius

        landmarks[473 * 3] = lEyeCx; landmarks[473 * 3 + 1] = lEyeCy
        landmarks[474 * 3] = lEyeCx + irisRadius; landmarks[474 * 3 + 1] = lEyeCy
        landmarks[475 * 3] = lEyeCx; landmarks[475 * 3 + 1] = lEyeCy - irisRadius
        landmarks[476 * 3] = lEyeCx - irisRadius; landmarks[476 * 3 + 1] = lEyeCy
        landmarks[477 * 3] = lEyeCx; landmarks[477 * 3 + 1] = lEyeCy + irisRadius

        val identityTransform = floatArrayOf(
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        )
        val faceState = FaceState(
            lifecycle = FaceLifecycle.State.TRACKING,
            isVisible = true,
            fadeWeight = 1.0f,
            landmarks = landmarks,
            blendshapes = blendshapes,
            isFrontCamera = false,
            imageToViewTransform = identityTransform,
            poseTimestampNs = 100_000_000L
        )
        return Triple(faceState, rEyeCx, rEyeCy)
    }
}
