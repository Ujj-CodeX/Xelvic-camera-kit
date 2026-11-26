package com.example

import android.graphics.Bitmap
import android.graphics.Color
import com.example.engine.capture.CameraSession
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.pkg.CapabilityRegistry
import com.example.engine.pkg.Compiler
import com.example.engine.render.CompositionMode
import com.example.engine.render.FaceMeshRenderer
import com.example.engine.render.PassStageKind
import com.example.engine.render.RecolorPass
import com.example.engine.render.RenderGraph
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import kotlin.math.abs

/**
 * Flow 4 — Chunk Primitives-1 (`FM1 + EC1`) Unit & Golden Conformance Suite:
 * 1. Generalized `faceTexture@1` & Blend Modes (`alpha`, `multiply`, `softLight`, `screen`),
 *    dynamic `mouth_inner` subtraction on open mouth, eyelid blink attenuation, and `cull = false`
 *    default for +-40 deg yaw.
 * 2. Generalized `recolor@1` (`RecolorPass`) with default `strength <= 0.70` cap, fibrous luminance
 *    detail preservation, specular corneal/eyeglasses glare protection (`Y in [0.78, 0.92]`),
 *    blink attenuation, and off-center gaze containment (`irisCircle intersect eyeContour`).
 * 3. `PassGraph` / `RenderGraph` execution order: `faceTexture@1` and `recolor@1` execute in
 *    `CAMERA_PIXEL_PASSES` strictly before `CLEAR_DEPTH`, `OCCLUDERS`, and 3D props.
 * 4. `320 / 480 / 640` stationary iris-landmark jitter (`stationaryJitterStdDevPx`) and
 *    `conv / inf / trk` telemetry in `LatencyProbe` and `DebugOverlay`.
 * 5. Scoped `PerformanceGovernor` iris resolution (`480` default while `tracker.iris@1` active,
 *    even on Tier B low-end start; immediately restored to `320` when removed; no visual tier
 *    demotion from high `resultAgeMs` alone).
 */
@RunWith(RobolectricTestRunner::class)
class ChunkPrimitives1Fm1Ec1UnitTest {

    private fun buildSyntheticLandmarks478(
        rightIrisShiftX: Float = 0f,
        rightIrisShiftY: Float = 0f,
        mouthOpenDeltaV: Float = 0f
    ): FloatArray {
        val lm = FloatArray(478 * 3)
        // Default neutral positions around center (0.5, 0.5)
        for (i in 0 until 478) {
            lm[i * 3] = 0.50f
            lm[i * 3 + 1] = 0.50f
            lm[i * 3 + 2] = 0f
        }

        // Right eye contour around (0.36, 0.42), half-width ~0.045, half-height ~0.022
        val rightEyeIndices = intArrayOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
        for ((k, idx) in rightEyeIndices.withIndex()) {
            val angle = (2.0 * Math.PI * k) / rightEyeIndices.size
            lm[idx * 3] = 0.36f + 0.045f * kotlin.math.cos(angle).toFloat()
            lm[idx * 3 + 1] = 0.42f + 0.022f * kotlin.math.sin(angle).toFloat()
        }

        // Left eye contour around (0.64, 0.42), half-width ~0.045, half-height ~0.022
        val leftEyeIndices = intArrayOf(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
        for ((k, idx) in leftEyeIndices.withIndex()) {
            val angle = (2.0 * Math.PI * k) / leftEyeIndices.size
            lm[idx * 3] = 0.64f + 0.045f * kotlin.math.cos(angle).toFloat()
            lm[idx * 3 + 1] = 0.42f + 0.022f * kotlin.math.sin(angle).toFloat()
        }

        // Right iris: center #468 + ring #469..472 (radius ~0.018)
        val rCx = 0.36f + rightIrisShiftX
        val rCy = 0.42f + rightIrisShiftY
        val irisR = 0.018f
        lm[468 * 3] = rCx
        lm[468 * 3 + 1] = rCy
        lm[469 * 3] = rCx + irisR; lm[469 * 3 + 1] = rCy
        lm[470 * 3] = rCx; lm[470 * 3 + 1] = rCy - irisR
        lm[471 * 3] = rCx - irisR; lm[471 * 3 + 1] = rCy
        lm[472 * 3] = rCx; lm[472 * 3 + 1] = rCy + irisR

        // Left iris: center #473 + ring #474..477 (radius ~0.018)
        val lCx = 0.64f
        val lCy = 0.42f
        lm[473 * 3] = lCx
        lm[473 * 3 + 1] = lCy
        lm[474 * 3] = lCx + irisR; lm[474 * 3 + 1] = lCy
        lm[475 * 3] = lCx; lm[475 * 3 + 1] = lCy - irisR
        lm[476 * 3] = lCx - irisR; lm[476 * 3 + 1] = lCy
        lm[477 * 3] = lCx; lm[477 * 3 + 1] = lCy + irisR

        // Cheek width landmarks #234 and #454 (distance = 0.28)
        lm[234 * 3] = 0.36f; lm[234 * 3 + 1] = 0.55f
        lm[454 * 3] = 0.64f; lm[454 * 3 + 1] = 0.55f

        // Inner mouth upper #13 and lower #14
        lm[13 * 3] = 0.50f; lm[13 * 3 + 1] = 0.70f
        lm[14 * 3] = 0.50f; lm[14 * 3 + 1] = 0.70f + mouthOpenDeltaV

        return lm
    }

    @Test
    fun testFaceTextureBlendModesAndDynamicMouthBlinkAndDoubleSidedDefault() {
        // 1. Verify BlendMode channel math (alpha, multiply, softLight, screen)
        val base = floatArrayOf(0.60f, 0.45f, 0.35f)
        val tint = floatArrayOf(0.80f, 0.30f, 0.50f)

        val multOut = FaceMeshRenderer.blendRgbWithBase(base, tint, 1.0f, FaceMeshRenderer.BlendMode.MULTIPLY)
        assertEquals(0.60f * 0.80f, multOut[0], 1e-4f)
        assertEquals(0.45f * 0.30f, multOut[1], 1e-4f)
        assertEquals(0.35f * 0.50f, multOut[2], 1e-4f)

        val screenOut = FaceMeshRenderer.blendRgbWithBase(base, tint, 1.0f, FaceMeshRenderer.BlendMode.SCREEN)
        assertEquals(1f - (1f - 0.60f) * (1f - 0.80f), screenOut[0], 1e-4f)
        assertEquals(1f - (1f - 0.45f) * (1f - 0.30f), screenOut[1], 1e-4f)

        val softOut = FaceMeshRenderer.blendRgbWithBase(base, tint, 1.0f, FaceMeshRenderer.BlendMode.SOFT_LIGHT)
        val expectedSoftR = (1f - 2f * 0.80f) * (0.60f * 0.60f) + 2f * 0.80f * 0.60f
        assertEquals(expectedSoftR, softOut[0], 1e-4f)

        // 2. Verify dynamic mouth_inner subtraction when mouth opens & eyelid blink attenuation
        val faceMeshRenderer = FaceMeshRenderer()
        val texRgba = floatArrayOf(0.90f, 0.25f, 0.45f, 1.0f)
        val colorTint = floatArrayOf(1f, 1f, 1f, 1f)

        // Inner mouth UV (0.50, 0.72): when mouthOpenWeight = 0.95, alpha is dynamically subtracted
        val mouthClosedFrag = faceMeshRenderer.evaluateEffectiveFragment(
            u = 0.50f,
            v = 0.73f,
            texRgba = texRgba,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 0.85f,
            tintColor = colorTint,
            protectEyes = true,
            protectLips = false,
            protectMouthInterior = true,
            mouthOpenWeight = 0.0f,
            eyeBlinkWeight = 0f
        )
        val mouthOpenFrag = faceMeshRenderer.evaluateEffectiveFragment(
            u = 0.50f,
            v = 0.73f,
            texRgba = texRgba,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 0.85f,
            tintColor = colorTint,
            protectEyes = true,
            protectLips = false,
            protectMouthInterior = true,
            mouthOpenWeight = 0.95f,
            eyeBlinkWeight = 0f
        )
        assertTrue(
            "Open mouth interior must dynamically subtract faceTexture@1 alpha (open=${mouthOpenFrag[3]} < closed=${mouthClosedFrag[3]})",
            mouthOpenFrag[3] < mouthClosedFrag[3] && mouthOpenFrag[3] < 0.08f
        )

        // Eyelid edge UV (0.3496, 0.410): when eyeBlinkWeight = 0.95, alpha is dynamically attenuated
        val eyeOpenFrag = faceMeshRenderer.evaluateEffectiveFragment(
            u = 0.3496f,
            v = 0.410f,
            texRgba = texRgba,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 0.85f,
            tintColor = colorTint,
            protectEyes = true,
            protectLips = true,
            protectMouthInterior = true,
            mouthOpenWeight = 0f,
            eyeBlinkWeight = 0f
        )
        val eyeBlinkFrag = faceMeshRenderer.evaluateEffectiveFragment(
            u = 0.3496f,
            v = 0.410f,
            texRgba = texRgba,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 0.85f,
            tintColor = colorTint,
            protectEyes = true,
            protectLips = true,
            protectMouthInterior = true,
            mouthOpenWeight = 0f,
            eyeBlinkWeight = 0.95f
        )
        assertTrue(
            "Blinking eyelid region must dynamically attenuate faceTexture@1 alpha (blink=${eyeBlinkFrag[3]} < open=${eyeOpenFrag[3]})",
            eyeBlinkFrag[3] < eyeOpenFrag[3] && eyeBlinkFrag[3] < 0.08f
        )

        // Cheek UV (0.22, 0.58): remains at full expected opacity even while blinking/mouth open
        val cheekFrag = faceMeshRenderer.evaluateEffectiveFragment(
            u = 0.22f,
            v = 0.58f,
            texRgba = texRgba,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 0.85f,
            tintColor = colorTint,
            protectEyes = true,
            protectLips = true,
            protectMouthInterior = true,
            mouthOpenWeight = 0.95f,
            eyeBlinkWeight = 0.95f
        )
        assertEquals(0.85f, cheekFrag[3], 1e-4f)

        // 3. Verify Compiler defaults faceTexture@1 to cull = false (doubleSided = true) for +-40 deg yaw
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_fm1_pkg",
              "name": "Test FM1",
              "version": "1.0.0",
              "assets": [
                { "id": "tex_blush", "type": "image", "path": "blush.png" }
              ],
              "materials": {
                "mat_blush": {
                  "template": "faceTexture@1",
                  "blend": "softLight",
                  "params": {
                    "map": "tex_blush",
                    "opacity": 0.65
                  }
                }
              },
              "nodes": [
                {
                  "id": "node_blush",
                  "type": "faceMesh",
                  "material": "mat_blush"
                }
              ]
            }
            """.trimIndent()
        )
        val compiled = Compiler.compile(manifestJson, File("."))
        val mat = compiled.materials["mat_blush"]
        assertNotNull(mat)
        assertFalse("faceTexture@1 must default to cull = false so +-40 deg yaw never culls far cheek", mat!!.cull)
        assertTrue("faceTexture@1 must default to doubleSided = true", mat.doubleSided)
        assertEquals("softLight", mat.blend)
        assertTrue(compiled.requestedBlendshapes.contains("jawOpen"))
        assertTrue(compiled.requestedBlendshapes.contains("eyeBlinkLeft"))
    }

    @Test
    fun testRecolorPassStrengthCapLuminanceDetailGlareBlinkAndGazeContainment() {
        val recolorPass = RecolorPass()
        val landmarks = buildSyntheticLandmarks478()

        // 1. Default strength cap <= 0.70 when unspecified or above 0.70
        val manifestWithHighStrength = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_ec1_pkg",
              "name": "Test EC1",
              "version": "1.0.0",
              "requires": {
                "features": ["tracker.iris@1", "pixel.recolor@1"]
              },
              "materials": {
                "mat_iris": {
                  "template": "recolor@1",
                  "params": {
                    "color": [0.18, 0.72, 0.58, 1.0],
                    "opacity": 0.95
                  }
                }
              },
              "nodes": [
                {
                  "id": "iris_node",
                  "type": "eyeColor",
                  "material": "mat_iris",
                  "params": {
                    "strength": 0.95
                  }
                }
              ]
            }
            """.trimIndent()
        )
        val compiled = Compiler.compile(manifestWithHighStrength, File("."))
        assertTrue("Package with eyeColor/recolor must set requiresIris = true", compiled.requiresIris)
        val irisNode = compiled.nodes.first { it.id == "iris_node" }
        val resolvedStrength = irisNode.postParams["strength"] ?: 1f
        assertTrue(
            "Default recolor strength must be capped at 0.70 (got $resolvedStrength)",
            resolvedStrength <= RecolorPass.MAX_DEFAULT_STRENGTH + 1e-5f
        )

        val targetR = irisNode.postParams["colorR"] ?: 0.18f
        val targetG = irisNode.postParams["colorG"] ?: 0.72f
        val targetB = irisNode.postParams["colorB"] ?: 0.58f
        val preserveLuma = irisNode.postParams["preserveLuminance"] ?: RecolorPass.DEFAULT_PRESERVE_LUMINANCE
        val glareLow = irisNode.postParams["glareLowLuma"] ?: RecolorPass.DEFAULT_GLARE_LOW_LUMA
        val glareHigh = irisNode.postParams["glareHighLuma"] ?: RecolorPass.DEFAULT_GLARE_HIGH_LUMA

        // 2. Fibrous luminance detail preservation: darker iris fiber vs lighter iris fiber keep contrast
        val recoloredDark = FloatArray(3)
        val recoloredLight = FloatArray(3)
        RecolorPass.recolorPixelRgb(
            baseR = 0.24f, baseG = 0.20f, baseB = 0.16f,
            targetR = targetR, targetG = targetG, targetB = targetB,
            maskWeight = 0.90f,
            strength = resolvedStrength,
            preserveLuminance = preserveLuma,
            glareLowLuma = glareLow,
            glareHighLuma = glareHigh,
            outRgb = recoloredDark
        )
        RecolorPass.recolorPixelRgb(
            baseR = 0.52f, baseG = 0.45f, baseB = 0.38f,
            targetR = targetR, targetG = targetG, targetB = targetB,
            maskWeight = 0.90f,
            strength = resolvedStrength,
            preserveLuminance = preserveLuma,
            glareLowLuma = glareLow,
            glareHighLuma = glareHigh,
            outRgb = recoloredLight
        )
        val lumaAfterDark = RecolorPass.luminance(recoloredDark[0], recoloredDark[1], recoloredDark[2])
        val lumaAfterLight = RecolorPass.luminance(recoloredLight[0], recoloredLight[1], recoloredLight[2])
        assertTrue(
            "Fibrous luminance detail must be preserved (light=$lumaAfterLight > dark=$lumaAfterDark)",
            lumaAfterLight > lumaAfterDark + 0.10f
        )

        // 3. Glasses-on / strong corneal reflection protection (Y >= 0.92 stays uncolored)
        val cornealGlareRgb = floatArrayOf(0.95f, 0.96f, 0.98f)
        val glareLuma = RecolorPass.luminance(cornealGlareRgb[0], cornealGlareRgb[1], cornealGlareRgb[2])
        val glareFactor = RecolorPass.computeGlareProtectionFactor(
            luma = glareLuma,
            glareLowLuma = glareLow,
            glareHighLuma = glareHigh
        )
        assertEquals(0.0f, glareFactor, 1e-5f)
        val recoloredGlare = FloatArray(3)
        val appliedGlareW = RecolorPass.recolorPixelRgb(
            baseR = cornealGlareRgb[0], baseG = cornealGlareRgb[1], baseB = cornealGlareRgb[2],
            targetR = targetR, targetG = targetG, targetB = targetB,
            maskWeight = 1.0f,
            strength = resolvedStrength,
            preserveLuminance = preserveLuma,
            glareLowLuma = glareLow,
            glareHighLuma = glareHigh,
            outRgb = recoloredGlare
        )
        assertEquals(0.0f, appliedGlareW, 1e-5f)
        assertEquals(cornealGlareRgb[0], recoloredGlare[0], 1e-4f)
        assertEquals(cornealGlareRgb[1], recoloredGlare[1], 1e-4f)
        assertEquals(cornealGlareRgb[2], recoloredGlare[2], 1e-4f)

        // 4. Blink fade: open eye has positive iris mask weight; closed eye (blink = 1.0) fades to 0
        val openState = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            landmarks = landmarks,
            blendshapes = mapOf("eyeBlinkLeft" to 0.0f, "eyeBlinkRight" to 0.0f),
            isFrontCamera = false
        )
        val closedState = openState.copy(
            blendshapes = mapOf("eyeBlinkLeft" to 1.0f, "eyeBlinkRight" to 1.0f)
        )
        // Sample inside right iris ring (x = 0.36 + 0.010 = 0.37, y = 0.42)
        val openEval = RecolorPass.evaluateIrisMaskWeightAtPixel(
            xPx = 0.37f * 480f,
            yPx = 0.42f * 480f,
            faceState = openState,
            viewportWidth = 480,
            viewportHeight = 480,
            featherPx = 1.5f
        )
        val closedEval = RecolorPass.evaluateIrisMaskWeightAtPixel(
            xPx = 0.37f * 480f,
            yPx = 0.42f * 480f,
            faceState = closedState,
            viewportWidth = 480,
            viewportHeight = 480,
            featherPx = 1.5f
        )
        assertTrue("Open eye iris ring pixel must have positive mask weight (got $openEval)", openEval > 0.35f)
        assertEquals("Closed eye (blink=1.0) must fade iris mask weight to 0.0", 0.0f, closedEval, 1e-4f)

        // 5. Off-center gaze containment (irisCircle intersect eyeContour):
        // Shift right iris upward so top of iris circle extends above upper eyelid (y = 0.385 < upper lid 0.398)
        val upwardGazeLandmarks = buildSyntheticLandmarks478(rightIrisShiftY = -0.016f)
        val upwardGazeState = openState.copy(landmarks = upwardGazeLandmarks)
        // Sample at (0.36, 0.388): inside the shifted iris circle, but OUTSIDE eye contour!
        val outsideEyelidEval = RecolorPass.evaluateIrisMaskWeightAtPixel(
            xPx = 0.36f * 480f,
            yPx = 0.388f * 480f,
            faceState = upwardGazeState,
            viewportWidth = 480,
            viewportHeight = 480,
            featherPx = 1.0f
        )
        assertEquals(
            "Iris pixel outside eyelid contour during off-center gaze must be clipped to 0.0",
            0.0f,
            outsideEyelidEval,
            1e-4f
        )

        // 6. CPU Bitmap execution parity (applyNodeToBitmap)
        val bmp = Bitmap.createBitmap(120, 120, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.rgb(95, 75, 60))
        val modifiedPixels = recolorPass.applyNodeToBitmap(
            bitmap = bmp,
            node = irisNode,
            effect = compiled,
            faceState = openState,
            opacity = 1.0f
        )
        assertTrue("RecolorPass.applyNodeToBitmap must modify iris pixels on visible face", modifiedPixels > 0)
    }

    @Test
    fun testCameraPixelPassesOrderBeforeOccludersAnd3DProps() {
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_order_pkg",
              "name": "Test Order",
              "version": "1.0.0",
              "assets": [
                { "id": "tex_makeup", "type": "image", "path": "makeup.png" }
              ],
              "materials": {
                "mat_makeup": { "template": "faceTexture@1", "blend": "multiply", "params": { "map": "tex_makeup" } },
                "mat_iris": { "template": "recolor@1", "params": { "color": [0.2, 0.6, 0.8, 1.0], "strength": 0.65 } },
                "mat_occ": { "template": "occluder@1" },
                "mat_glasses": { "template": "unlit@1", "blend": "alpha" }
              },
              "nodes": [
                { "id": "glasses_3d", "type": "sprite", "material": "mat_glasses", "renderOrder": 10 },
                { "id": "head_occ", "type": "occluder", "material": "mat_occ", "renderOrder": 0 },
                { "id": "iris_recolor", "type": "eyeColor", "material": "mat_iris", "renderOrder": 2 },
                { "id": "cheek_makeup", "type": "faceMesh", "material": "mat_makeup", "renderOrder": 1 }
              ]
            }
            """.trimIndent()
        )
        val compiled = Compiler.compile(manifestJson, File("."))
        val stageKinds = compiled.passGraph.passes.map { it.stageKind }
        val camPixelIdx = stageKinds.indexOf(PassStageKind.CAMERA_PIXEL_PASSES)
        val clearDepthIdx = stageKinds.indexOf(PassStageKind.CLEAR_DEPTH)
        val occluderIdx = stageKinds.indexOf(PassStageKind.OCCLUDERS)
        val transparentIdx = stageKinds.indexOf(PassStageKind.TRANSPARENT_NODES)

        assertTrue("PassGraph must include CAMERA_PIXEL_PASSES", camPixelIdx >= 0)
        assertTrue("CAMERA_PIXEL_PASSES must execute before CLEAR_DEPTH", camPixelIdx < clearDepthIdx)
        assertTrue("CLEAR_DEPTH must execute before OCCLUDERS", clearDepthIdx < occluderIdx)
        assertTrue("OCCLUDERS must execute before TRANSPARENT_NODES (3D props)", occluderIdx < transparentIdx)
    }

    @Test
    fun testStationaryIrisJitterTelemetryAt320And480And640AndScopedGovernorIrisSize() {
        val probe = LatencyProbe()
        val baseLandmarks = buildSyntheticLandmarks478()

        // Feed stationary samples into 320, 480, and 640 buckets with decreasing normalized jitter
        val offsets320 = floatArrayOf(-0.0020f, 0.0020f, -0.0018f, 0.0018f, -0.0022f, 0.0022f)
        val offsets480 = floatArrayOf(-0.0009f, 0.0009f, -0.0008f, 0.0008f, -0.0010f, 0.0010f)
        val offsets640 = floatArrayOf(-0.0005f, 0.0005f, -0.0004f, 0.0004f, -0.0005f, 0.0005f)

        for (i in offsets320.indices) {
            val lm320 = baseLandmarks.clone().also {
                it[468 * 3] += offsets320[i]
                it[473 * 3 + 1] -= offsets320[i]
            }
            probe.recordIrisLandmarkSample(lm320, analysisSizePx = 320, conversionMs = 1.8f, inferenceMs = 7.5f, trackerFps = 30f, isStationary = true)

            val lm480 = baseLandmarks.clone().also {
                it[468 * 3] += offsets480[i]
                it[473 * 3 + 1] -= offsets480[i]
            }
            probe.recordIrisLandmarkSample(lm480, analysisSizePx = 480, conversionMs = 2.4f, inferenceMs = 8.9f, trackerFps = 30f, isStationary = true)

            val lm640 = baseLandmarks.clone().also {
                it[468 * 3] += offsets640[i]
                it[473 * 3 + 1] -= offsets640[i]
            }
            probe.recordIrisLandmarkSample(lm640, analysisSizePx = 640, conversionMs = 3.5f, inferenceMs = 11.2f, trackerFps = 28f, isStationary = true)
        }

        val m320 = probe.getIrisSizeMetrics(320)
        val m480 = probe.getIrisSizeMetrics(480)
        val m640 = probe.getIrisSizeMetrics(640)

        assertEquals(6, m320.sampleCount)
        assertEquals(6, m480.sampleCount)
        assertEquals(6, m640.sampleCount)
        assertTrue("320 bucket jitter must be > 0", m320.stationaryJitterStdDevPx > 0.1f)
        assertTrue("480 bucket jitter must be lower than 320 bucket jitter", m480.stationaryJitterStdDevPx < m320.stationaryJitterStdDevPx)
        assertTrue("640 bucket jitter must be lower than 480 bucket jitter", m640.stationaryJitterStdDevPx < m480.stationaryJitterStdDevPx)

        val overlay = DebugOverlay(latencyProbe = probe)
        val state = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            landmarks = baseLandmarks
        )
        overlay.refreshCachedHudStringsIfNeeded(state, fps = 30f, tier = "A", force = true)
        assertTrue(overlay.cachedIrisJitterLine.contains("IrisJitter[480px]"))
        assertTrue(overlay.cachedIrisJitterLine.contains("320:"))
        assertTrue(overlay.cachedIrisJitterLine.contains("480:"))
        assertTrue(overlay.cachedIrisJitterLine.contains("640:"))

        // Verify PerformanceGovernor scopes 480 analysis size to tracker.iris@1 filters only (even on low-end Tier B start)
        val lowEndGov = PerformanceGovernor(isLowEndDevice = true)
        assertEquals(PerformanceGovernor.Tier.B, lowEndGov.currentTier)
        assertEquals(320, lowEndGov.targetAnalysisSizePx)

        lowEndGov.setFilterIrisRequirement(requiresIris = true, nowMs = 1000L)
        assertEquals(
            "Activating a tracker.iris@1 filter must raise targetAnalysisSizePx to 480 even on Tier B low-end start",
            480,
            lowEndGov.targetAnalysisSizePx
        )

        // Cycling iris analysis size 480 -> 640 -> 320 -> 480
        assertEquals(640, lowEndGov.cycleConfiguredIrisAnalysisSizePx(nowMs = 1100L))
        assertEquals(640, lowEndGov.targetAnalysisSizePx)
        assertEquals(320, lowEndGov.cycleConfiguredIrisAnalysisSizePx(nowMs = 1200L))
        assertEquals(480, lowEndGov.cycleConfiguredIrisAnalysisSizePx(nowMs = 1300L))

        // Removing the iris filter immediately restores targetAnalysisSizePx to 320
        lowEndGov.setFilterIrisRequirement(requiresIris = false, nowMs = 1400L)
        assertEquals(
            "Removing tracker.iris@1 filter must immediately restore targetAnalysisSizePx to 320",
            320,
            lowEndGov.targetAnalysisSizePx
        )

        // Verify high resultAgeMs alone never demotes visual tier
        val highEndGov = PerformanceGovernor(isLowEndDevice = false)
        highEndGov.setFilterIrisRequirement(requiresIris = true, nowMs = 1000L)
        assertEquals(PerformanceGovernor.Tier.A, highEndGov.currentTier)
        assertEquals(480, highEndGov.targetAnalysisSizePx)
        for (step in 1..20) {
            highEndGov.recordFrameDuration(
                durationMs = 16.5f,
                nowMs = 1000L + step * 500L,
                resultAgeAtRenderMs = 185.0f,
                currentTrackerHz = 24.0f,
                conversionMs = 18.0f,
                inferenceMs = 22.0f
            )
        }
        assertEquals(
            "High resultAgeMs alone must NEVER demote visual tier (Tier A)",
            PerformanceGovernor.Tier.A,
            highEndGov.currentTier
        )
    }
}
