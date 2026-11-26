package com.example

import android.graphics.Bitmap
import android.graphics.Color
import com.example.engine.photo.OffscreenRenderer
import com.example.engine.pkg.CapabilityRegistry
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.PackageValidator
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.PostRenderer
import com.example.engine.render.RegionMaskPass
import com.example.engine.render.RenderGraph
import com.example.engine.render.SpriteRenderer
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.ControlPointSmoother
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
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
import kotlin.math.hypot

/**
 * Flow 5 — Chunk Primitives-2 (`CF1 + Snap-Level Beauty, Soft Focus & Subtle Face Morph`)
 * Verification Suite:
 *
 * 1. Capability registration & manifest compilation for `post.skinSoften@1`, `post.bilateral@1`,
 *    `post.softFocus@1`, `post.warp@1`, `post.colorMatrix@1`, `post.vignette@1`, and `node.overlay@1`.
 * 2. Half-res (`0.5x`) `skinSoften` / `bilateral` with `$mask:skin` region protection and high-frequency
 *    pore `detailPreserve` re-injection (eyes, lips, and background remain 100% sharp with `delta == 0`).
 * 3. Quarter-res (`0.25x`) `softFocus` Kawase highlight halation with dark-edge contrast preservation.
 * 4. 1-Euro smoothed landmark-driven `warp` (`jawSlim`, `eyeEnlarge`, `noseSlim`, `chinVLine`) with
 *    zero stationary jitter and strictly `0.0` UV displacement outside `face_oval`.
 * 5. `colorMatrix` (brightness, contrast, saturation, temperature, tint, sepia, B&W) and `vignette` radial falloff.
 * 6. Screen-space `node.overlay@1` (`cover` / `contain`, `safeArea`, and unmirrored front-camera orientation
 *    executing in Stage 8 `OVERLAY_NODES` after `POST_NODES`).
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ChunkPrimitives2Cf1BeautyUnitTest {

    private fun buildSyntheticFaceState(
        imageWidth: Int = 128,
        imageHeight: Int = 128,
        isFrontCamera: Boolean = true,
        timestampNs: Long = 100_000_000L
    ): FaceState {
        val uvs = UvRegistry.CANONICAL_UVS
        val landmarks = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        // Map canonical face UVs into centered normalized camera space [0.25..0.75, 0.20..0.80]
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            val u = uvs[i * 2]
            val v = uvs[i * 2 + 1]
            // Note: on front camera, FacePoseAdapter.mapLandmarkToView mirrors X (1 - u) when identity transform is used
            val rawU = 0.25f + u * 0.50f
            val rawV = 0.20f + v * 0.60f
            landmarks[i * 3] = if (isFrontCamera) (1.0f - rawU) else rawU
            landmarks[i * 3 + 1] = rawV
            landmarks[i * 3 + 2] = -0.02f
        }
        // Populate 468..477 iris landmarks around left (#33/#133) and right (#263/#362) eye centers
        val rEyeU = (landmarks[33 * 3] + landmarks[133 * 3]) * 0.5f
        val rEyeV = (landmarks[159 * 3 + 1] + landmarks[145 * 3 + 1]) * 0.5f
        landmarks[468 * 3] = rEyeU
        landmarks[468 * 3 + 1] = rEyeV
        for (k in 0 until 4) {
            val ang = k * Math.PI * 0.5
            landmarks[(469 + k) * 3] = rEyeU + (Math.cos(ang) * 0.015).toFloat()
            landmarks[(469 + k) * 3 + 1] = rEyeV + (Math.sin(ang) * 0.015).toFloat()
        }

        val lEyeU = (landmarks[263 * 3] + landmarks[362 * 3]) * 0.5f
        val lEyeV = (landmarks[386 * 3 + 1] + landmarks[374 * 3 + 1]) * 0.5f
        landmarks[473 * 3] = lEyeU
        landmarks[473 * 3 + 1] = lEyeV
        for (k in 0 until 4) {
            val ang = k * Math.PI * 0.5
            landmarks[(474 + k) * 3] = lEyeU + (Math.cos(ang) * 0.015).toFloat()
            landmarks[(474 + k) * 3 + 1] = lEyeV + (Math.sin(ang) * 0.015).toFloat()
        }

        val pose = FloatArray(16).apply {
            this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
            this[14] = -0.45f
        }
        return FaceState.create(
            timestampNs = timestampNs,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = pose,
            translation = floatArrayOf(0f, 0f, -0.45f),
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 0f,
            pitchDeg = 0f,
            rollDeg = 0f,
            faceWidth = 0.14f,
            landmarks = landmarks,
            blendshapes = emptyMap(),
            acquisitionTimeNs = timestampNs,
            imageToViewTransform = null,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            isFrontCamera = isFrontCamera
        )
    }

    @Test
    fun testFlow5CapabilitiesAndPostEffectContracts() {
        val expectedCapabilities = listOf(
            "post.skinSoften@1",
            "post.bilateral@1",
            "post.softFocus@1",
            "post.warp@1",
            "post.colorMatrix@1",
            "post.vignette@1",
            "node.overlay@1",
            "pass.skinSoften@1",
            "pass.bilateral@1",
            "pass.softFocus@1",
            "pass.warp@1",
            "pass.colorMatrix@1",
            "pass.vignette@1"
        )
        for (cap in expectedCapabilities) {
            assertTrue("CapabilityRegistry must support $cap", CapabilityRegistry.isFeatureSupported(cap))
        }

        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_snap_soft_focus_beauty",
              "name": "Snap Soft Focus Beauty Test",
              "requires": {
                "runtime": 3,
                "features": [
                  "node.post@1",
                  "post.bilateral@1",
                  "post.softFocus@1",
                  "post.warp@1",
                  "post.colorMatrix@1",
                  "post.vignette@1",
                  "node.overlay@1"
                ]
              },
              "params": {
                "smoothKnob": { "type": "float", "default": 0.55, "min": 0.0, "max": 1.0 },
                "glowKnob": { "type": "float", "default": 0.40, "min": 0.0, "max": 1.0 },
                "jawKnob": { "type": "float", "default": 0.35, "min": 0.0, "max": 1.0 }
              },
              "nodes": [
                {
                  "id": "subtle_morph",
                  "type": "post",
                  "effect": "warp",
                  "params": {
                    "jawSlim": "${'$'}jawKnob",
                    "eyeEnlarge": 0.25,
                    "noseSlim": 0.20,
                    "chinVLine": 0.15,
                    "radiusFW": 0.40
                  },
                  "renderOrder": 90
                },
                {
                  "id": "skin_bilateral",
                  "type": "post",
                  "effect": "bilateral",
                  "mask": "${'$'}mask:skin",
                  "params": {
                    "amount": "${'$'}smoothKnob",
                    "radius": 3.0,
                    "edgePreserve": 0.75,
                    "detailPreserve": 0.35
                  },
                  "renderOrder": 100
                },
                {
                  "id": "lens_soft_focus",
                  "type": "post",
                  "effect": "softFocus",
                  "params": {
                    "amount": "${'$'}glowKnob",
                    "radius": 4.0,
                    "threshold": 0.55,
                    "warmth": 0.12
                  },
                  "renderOrder": 110
                },
                {
                  "id": "tone_matrix",
                  "type": "post",
                  "effect": "colorMatrix",
                  "params": {
                    "brightness": 0.03,
                    "contrast": 1.06,
                    "saturation": 1.04,
                    "temperature": 0.08
                  },
                  "renderOrder": 120
                },
                {
                  "id": "edge_vignette",
                  "type": "post",
                  "effect": "vignette",
                  "params": {
                    "radius": 0.80,
                    "softness": 0.45,
                    "amount": 0.25
                  },
                  "renderOrder": 130
                },
                {
                  "id": "polaroid_frame",
                  "type": "overlay",
                  "fit": "contain",
                  "safeArea": true,
                  "renderOrder": 200
                }
              ]
            }
            """.trimIndent()
        )

        val validation = PackageValidator.validateManifest(manifestJson)
        assertTrue("Flow 5 beauty + softFocus + warp + overlay manifest must validate: ${validation.errorReason}", validation.isValid)

        val compiled = Compiler.compile(manifestJson, File("."))
        assertEquals(6, compiled.nodes.size)
        val overlayNode = compiled.nodes.first { it.id == "polaroid_frame" }
        assertEquals("overlay", overlayNode.type)
        assertEquals("contain", overlayNode.overlayFit)
        assertTrue(overlayNode.overlaySafeArea)
    }

    @Test
    fun testSkinSoftenAndBilateralWithSkinMaskPreservesEyesLipsBackgroundAndPoreDetail() {
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_masked_skin_soften",
              "name": "Masked Skin Soften",
              "nodes": [
                {
                  "id": "skin_pass",
                  "type": "post",
                  "effect": "bilateral",
                  "mask": "${'$'}mask:skin",
                  "params": {
                    "amount": 0.85,
                    "radius": 3.0,
                    "edgePreserve": 0.80,
                    "detailPreserve": 0.0
                  },
                  "renderOrder": 100
                }
              ]
            }
            """.trimIndent()
        )
        val effectNoDetail = Compiler.compile(manifestJson, File("."))
        val faceState = buildSyntheticFaceState(128, 128)
        val evaluator = SceneEvaluator(maxNodes = 16)
        effectNoDetail.signalBus.evaluate(faceState, 0.1f)
        evaluator.evaluate(effectNoDetail, faceState, effectNoDetail.propertySlots)

        // Verify RenderGraph leases HALF (0.5x) RtPool target and evaluates $mask:skin
        val renderGraph = RenderGraph()
        renderGraph.render(effectNoDetail, faceState, evaluator, 256, 256)
        assertTrue("bilateral/skinSoften must lease a HALF-res RtPool target", renderGraph.post.lastUsedHalfResTarget)
        assertTrue(
            "bilateral with \$mask:skin must evaluate \$mask:skin into RtPool",
            renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin")
        )

        // Construct synthetic 96x96 test image with:
        // - Cheek skin region (mask == 1.0): mid-frequency blotches (3x3 blocks of ±18) + fine pore texture
        // - Protected eye region & background (mask == 0.0): high-contrast razor-sharp pattern
        val w = 96
        val h = 96
        val explicitSkinMask = FloatArray(w * h)
        val bmpNoDetail = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val inSkinBox = (x in 28..68 && y in 28..68)
                val inProtectedEyeOrLip = (x in 36..44 && y in 36..44) || (x in 44..52 && y in 56..62)
                val isSkin = inSkinBox && !inProtectedEyeOrLip
                explicitSkinMask[y * w + x] = if (isSkin) 1.0f else 0.0f

                val blotch = if (((x / 2) + (y / 2)) % 2 == 0) 18 else -18
                val pore = if ((x + y) % 2 == 0) 6 else -6
                val r = (182 + blotch + pore).coerceIn(0, 255)
                val g = (142 + blotch + pore).coerceIn(0, 255)
                val b = (124 + blotch + pore).coerceIn(0, 255)
                bmpNoDetail.setPixel(x, y, Color.argb(255, r, g, b))
            }
        }
        val bmpOriginal = bmpNoDetail.copy(Bitmap.Config.ARGB_8888, false)
        val bmpWithDetail = bmpNoDetail.copy(Bitmap.Config.ARGB_8888, true)

        val post = PostRenderer()
        post.applyPostNodesToBitmap(
            composedBitmap = bmpNoDetail,
            effect = effectNoDetail,
            isNodeVisible = BooleanArray(1) { true },
            faceState = faceState,
            skinMaskWeights = explicitSkinMask
        )

        // 1. Protected eye pixel (40, 40), lip pixel (48, 58), and background pixel (8, 8) must have ZERO modification!
        assertEquals(
            "Protected eye pixel (mask=0) must remain 100% sharp with zero color change",
            bmpOriginal.getPixel(40, 40),
            bmpNoDetail.getPixel(40, 40)
        )
        assertEquals(
            "Protected lip pixel (mask=0) must remain 100% sharp with zero color change",
            bmpOriginal.getPixel(48, 58),
            bmpNoDetail.getPixel(48, 58)
        )
        assertEquals(
            "Background pixel outside face_oval (mask=0) must remain 100% unchanged",
            bmpOriginal.getPixel(8, 8),
            bmpNoDetail.getPixel(8, 8)
        )

        // 2. Interior cheek skin pixel (52, 46) must have reduced blotch variance
        val origSkinDiff = abs(Color.red(bmpOriginal.getPixel(52, 46)) - Color.red(bmpOriginal.getPixel(54, 46)))
        val smoothSkinDiff = abs(Color.red(bmpNoDetail.getPixel(52, 46)) - Color.red(bmpNoDetail.getPixel(54, 46)))
        assertTrue(
            "Skin bilateral smoothing must reduce blotch variance inside \$mask:skin (was $origSkinDiff, now $smoothSkinDiff)",
            smoothSkinDiff < origSkinDiff
        )

        // 3. High-frequency pore detailPreserve = 0.90 retains more fine micro-contrast than detailPreserve = 0.0
        val manifestDetailJson = JSONObject(manifestJson.toString()).apply {
            getJSONArray("nodes").getJSONObject(0).getJSONObject("params").put("detailPreserve", 0.90)
        }
        val effectWithDetail = Compiler.compile(manifestDetailJson, File("."))
        post.applyPostNodesToBitmap(
            composedBitmap = bmpWithDetail,
            effect = effectWithDetail,
            isNodeVisible = BooleanArray(1) { true },
            faceState = faceState,
            skinMaskWeights = explicitSkinMask
        )
        val noDetailMicroDiff = abs(Color.red(bmpNoDetail.getPixel(52, 46)) - Color.red(bmpNoDetail.getPixel(53, 46)))
        val withDetailMicroDiff = abs(Color.red(bmpWithDetail.getPixel(52, 46)) - Color.red(bmpWithDetail.getPixel(53, 46)))
        assertTrue(
            "detailPreserve > 0 must re-inject high-frequency pore micro-texture ($withDetailMicroDiff vs $noDetailMicroDiff)",
            withDetailMicroDiff >= noDetailMicroDiff
        )
    }

    @Test
    fun testSoftFocusQuarterResKawaseHighlightHalationPreservesDarkContrast() {
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_soft_focus_lens",
              "name": "Soft Focus Lens",
              "nodes": [
                {
                  "id": "soft_focus_pass",
                  "type": "post",
                  "effect": "softFocus",
                  "params": {
                    "amount": 0.75,
                    "radius": 4.0,
                    "threshold": 0.50,
                    "warmth": 0.20
                  },
                  "renderOrder": 100
                }
              ]
            }
            """.trimIndent()
        )
        val effect = Compiler.compile(manifestJson, File("."))
        val faceState = buildSyntheticFaceState()
        val evaluator = SceneEvaluator(maxNodes = 16)
        effect.signalBus.evaluate(faceState, 0.1f)
        evaluator.evaluate(effect, faceState, effect.propertySlots)

        val renderGraph = RenderGraph()
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertTrue(
            "softFocus must lease a QUARTER-res (0.25x) RtPool target for Kawase highlight bloom",
            renderGraph.post.lastUsedQuarterResTarget
        )

        // Create a 64x64 bitmap with a bright highlight region (center 28..36) next to a midtone region
        // and a dark eyelash region (10..16)
        val bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        for (y in 0 until 64) {
            for (x in 0 until 64) {
                val c = when {
                    x in 28..36 && y in 28..36 -> Color.rgb(250, 245, 240) // Bright specular highlight
                    x in 8..16 && y in 8..16 -> Color.rgb(18, 18, 20)      // Dark eyelash/pupil
                    else -> Color.rgb(120, 110, 105)                       // Midtone skin
                }
                bmp.setPixel(x, y, c)
            }
        }
        val origMidtoneAdjacent = bmp.getPixel(26, 32)
        val origDarkLash = bmp.getPixel(12, 12)

        val post = PostRenderer()
        post.applyPostNodesToBitmap(bmp, effect, BooleanArray(1) { true }, faceState)

        val afterMidtoneAdjacent = bmp.getPixel(26, 32)
        val afterDarkLash = bmp.getPixel(12, 12)

        assertTrue(
            "softFocus Kawase halation must bloom bright highlights into adjacent midtones",
            Color.red(afterMidtoneAdjacent) > Color.red(origMidtoneAdjacent)
        )
        assertTrue(
            "softFocus dark contrast guard must keep dark eyelash pixels dark (< 30 RGB)",
            Color.red(afterDarkLash) < 30 && abs(Color.red(afterDarkLash) - Color.red(origDarkLash)) <= 8
        )
    }

    @Test
    fun testSubtleFaceWarpLandmarkDisplacementZeroStationaryJitterAndZeroBackgroundDistortion() {
        val faceState = buildSyntheticFaceState(128, 128)
        val warpParams = mapOf(
            "jawSlim" to 0.65f,
            "eyeEnlarge" to 0.60f,
            "noseSlim" to 0.55f,
            "chinVLine" to 0.50f,
            "radiusFW" to 0.45f,
            "intensity" to 1.0f
        )

        val noseBridge = PostRenderer.getLandmarkNormalizedView(faceState, 6)!!
        val leftJaw = PostRenderer.getLandmarkNormalizedView(faceState, 172)!!
        val rightJaw = PostRenderer.getLandmarkNormalizedView(faceState, 397)!!
        val leftAla = PostRenderer.getLandmarkNormalizedView(faceState, 129)!!
        val chinTip = PostRenderer.getLandmarkNormalizedView(faceState, 152)!!

        // 1. Jaw slimming pulls left & right jaw points INWARD toward the face midline (noseBridge[0])
        val deltaLeftJaw = FloatArray(2)
        val deltaRightJaw = FloatArray(2)
        PostRenderer.computeWarpForwardDisplacementUv(leftJaw[0], leftJaw[1], faceState, warpParams, 1f, deltaLeftJaw)
        PostRenderer.computeWarpForwardDisplacementUv(rightJaw[0], rightJaw[1], faceState, warpParams, 1f, deltaRightJaw)

        val distLeftBefore = abs(leftJaw[0] - noseBridge[0])
        val distLeftAfter = abs((leftJaw[0] + deltaLeftJaw[0]) - noseBridge[0])
        val distRightBefore = abs(rightJaw[0] - noseBridge[0])
        val distRightAfter = abs((rightJaw[0] + deltaRightJaw[0]) - noseBridge[0])

        assertTrue("jawSlim must move left jaw inward toward midline", distLeftAfter < distLeftBefore)
        assertTrue("jawSlim must move right jaw inward toward midline", distRightAfter < distRightBefore)

        // 2. Nose slimming pulls nasal alar wing (#129) inward toward midline
        val deltaLeftAla = FloatArray(2)
        PostRenderer.computeWarpForwardDisplacementUv(leftAla[0], leftAla[1], faceState, warpParams, 1f, deltaLeftAla)
        val alaDistBefore = abs(leftAla[0] - noseBridge[0])
        val alaDistAfter = abs((leftAla[0] + deltaLeftAla[0]) - noseBridge[0])
        assertTrue("noseSlim must move nasal alar wing inward toward midline", alaDistAfter < alaDistBefore)

        // 3. Chin V-Line tapers chin (#152) upward (negative dV) when chinVLine > 0
        val deltaChin = FloatArray(2)
        PostRenderer.computeWarpForwardDisplacementUv(chinTip[0], chinTip[1], faceState, warpParams, 1f, deltaChin)
        assertTrue("Positive chinVLine must lift chin upward (dV < 0)", deltaChin[1] < 0f)

        // 4. Eye enlargement expands points near the eye contour radially outward from the eye centroid
        val lOuter = PostRenderer.getLandmarkNormalizedView(faceState, 33)!!
        val lInner = PostRenderer.getLandmarkNormalizedView(faceState, 133)!!
        val lTop = PostRenderer.getLandmarkNormalizedView(faceState, 159)!!
        val lBot = PostRenderer.getLandmarkNormalizedView(faceState, 145)!!
        val eyeCenterU = (lOuter[0] + lInner[0] + lTop[0] + lBot[0]) * 0.25f
        val eyeCenterV = (lOuter[1] + lInner[1] + lTop[1] + lBot[1]) * 0.25f
        val sampleNearEyeU = eyeCenterU + (lOuter[0] - eyeCenterU) * 0.55f
        val sampleNearEyeV = eyeCenterV + (lOuter[1] - eyeCenterV) * 0.55f
        val deltaEye = FloatArray(2)
        PostRenderer.computeWarpForwardDisplacementUv(
            sampleNearEyeU,
            sampleNearEyeV,
            faceState,
            mapOf("eyeEnlarge" to 0.8f, "intensity" to 1.0f),
            1f,
            deltaEye
        )
        val eyeRadiusBefore = hypot(sampleNearEyeU - eyeCenterU, sampleNearEyeV - eyeCenterV)
        val eyeRadiusAfter = hypot(
            sampleNearEyeU + deltaEye[0] - eyeCenterU,
            sampleNearEyeV + deltaEye[1] - eyeCenterV
        )
        assertTrue("eyeEnlarge must expand points near the eye contour outward from eye center", eyeRadiusAfter > eyeRadiusBefore)

        // 5. Strict 0.0f background distortion outside face_oval
        val bgPoints = arrayOf(
            0.02f to 0.02f,
            0.98f to 0.02f,
            0.02f to 0.98f,
            0.98f to 0.98f,
            0.05f to 0.50f,
            0.95f to 0.50f
        )
        val deltaBg = FloatArray(2)
        for ((bu, bv) in bgPoints) {
            PostRenderer.computeWarpForwardDisplacementUv(bu, bv, faceState, warpParams, 1f, deltaBg)
            assertEquals("Background point ($bu, $bv) outside face_oval must have 0.0 dU", 0.0f, deltaBg[0], 1e-7f)
            assertEquals("Background point ($bu, $bv) outside face_oval must have 0.0 dV", 0.0f, deltaBg[1], 1e-7f)
        }

        // 6. Zero stationary jitter across 30 consecutive stationary frames with 1-Euro ControlPointSmoother
        val smoother = ControlPointSmoother()
        val refDelta = FloatArray(2)
        for (frame in 0 until 30) {
            val tsNs = 100_000_000L + frame * 16_666_667L
            val smoothedLm = smoother.smooth(faceState.landmarks, tsNs, true)!!
            val fState = faceState.copy(landmarks = smoothedLm.clone(), timestampNs = tsNs, poseTimestampNs = tsNs)
            val curDelta = FloatArray(2)
            PostRenderer.computeWarpForwardDisplacementUv(leftJaw[0], leftJaw[1], fState, warpParams, 1f, curDelta)
            if (frame == 0) {
                refDelta[0] = curDelta[0]
                refDelta[1] = curDelta[1]
            } else {
                assertEquals("Stationary 1-Euro smoothed landmarks must produce 0.0 warp jitter in dU", refDelta[0], curDelta[0], 1e-6f)
                assertEquals("Stationary 1-Euro smoothed landmarks must produce 0.0 warp jitter in dV", refDelta[1], curDelta[1], 1e-6f)
            }
        }
    }

    @Test
    fun testColorMatrixAndVignetteGradingMath() {
        // 1. Test Black & White colorMatrix (bw = 1.0)
        val bwManifest = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_bw_vignette",
              "name": "B&W + Vignette",
              "nodes": [
                {
                  "id": "bw_pass",
                  "type": "post",
                  "effect": "colorMatrix",
                  "params": {
                    "bw": 1.0,
                    "contrast": 1.15
                  },
                  "renderOrder": 100
                },
                {
                  "id": "vig_pass",
                  "type": "post",
                  "effect": "vignette",
                  "params": {
                    "radius": 0.40,
                    "softness": 0.50,
                    "amount": 0.65
                  },
                  "renderOrder": 110
                }
              ]
            }
            """.trimIndent()
        )
        val effect = Compiler.compile(bwManifest, File("."))
        val bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.rgb(210, 120, 70)) // Warm orange-terracotta input

        val post = PostRenderer()
        val applied = post.applyPostNodesToBitmap(bmp, effect, BooleanArray(2) { true })
        assertEquals(2, applied)

        val centerPx = bmp.getPixel(32, 32)
        val cornerPx = bmp.getPixel(1, 1)

        // B&W mode must produce neutral greyscale (R == G == B)
        assertEquals(Color.red(centerPx), Color.green(centerPx))
        assertEquals(Color.green(centerPx), Color.blue(centerPx))

        // Vignette must darken the corner relative to the center
        assertTrue(
            "Vignette must darken outer corners relative to center (center=${Color.red(centerPx)}, corner=${Color.red(cornerPx)})",
            Color.red(cornerPx) < Color.red(centerPx)
        )
    }

    @Test
    fun testOverlayNodeCoverContainSafeAreaAndUnmirroredFrontCameraOrientation() {
        // 1. Verify "contain" + safeArea insets placement inside viewport and preserves aspect ratio
        val containPlacement = SpriteRenderer.computeOverlayPlacement(
            viewportWidth = 1000,
            viewportHeight = 2000,
            assetWidth = 1000,
            assetHeight = 1000,
            fit = "contain",
            safeArea = true
        )
        assertFalse("Screen-space overlay must NEVER be mirrored horizontally", containPlacement.isMirroredHorizontally)
        assertTrue("Safe-area must inset left edge", containPlacement.dstLeftPx >= 30f - 1e-3f)
        assertTrue("Safe-area must inset top edge", containPlacement.dstTopPx >= 100f - 1e-3f)
        val drawnW = containPlacement.dstRightPx - containPlacement.dstLeftPx
        val drawnH = containPlacement.dstBottomPx - containPlacement.dstTopPx
        assertEquals("Contain fit of 1:1 asset must remain square", drawnW, drawnH, 1.0f)

        // 2. Verify "cover" fills viewport and crops excess UVs symmetrically without mirroring
        val coverPlacement = SpriteRenderer.computeOverlayPlacement(
            viewportWidth = 1000,
            viewportHeight = 2000,
            assetWidth = 2000,
            assetHeight = 2000,
            fit = "cover",
            safeArea = false
        )
        assertEquals(0f, coverPlacement.dstLeftPx, 1e-3f)
        assertEquals(1000f, coverPlacement.dstRightPx, 1e-3f)
        assertEquals(0f, coverPlacement.dstTopPx, 1e-3f)
        assertEquals(2000f, coverPlacement.dstBottomPx, 1e-3f)
        assertTrue("Cover horizontal UV window must stay unmirrored (u0 < u1)", coverPlacement.u0 < coverPlacement.u1)
        assertEquals(0.25f, coverPlacement.u0, 1e-3f)
        assertEquals(0.75f, coverPlacement.u1, 1e-3f)

        // 3. Verify asymmetric left-side badge on overlayBitmap remains on the LEFT side of targetBitmap
        // even when isFrontCamera = true
        val targetBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        targetBmp.eraseColor(Color.BLACK)
        val overlayBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        overlayBmp.eraseColor(Color.TRANSPARENT)
        for (y in 20..44) {
            for (x in 4..14) {
                overlayBmp.setPixel(x, y, Color.argb(255, 255, 40, 40)) // Red badge on LEFT side
            }
        }

        SpriteRenderer.applyOverlayToBitmap(
            targetBitmap = targetBmp,
            overlayBitmap = overlayBmp,
            fit = "cover",
            safeArea = false,
            opacity = 1.0f,
            blendMode = "alpha",
            isFrontCamera = true
        )

        assertEquals("Left-side overlay badge must stay on the left (unmirrored)", 255, Color.red(targetBmp.getPixel(8, 32)))
        assertEquals("Right side of target bitmap must remain untouched black", 0, Color.red(targetBmp.getPixel(56, 32)))
    }

    @Test
    fun testNoMakeupBeautyPackageValidationCompilationTiersAndExecution() {
        val pkgDir = listOf(
            File("app/src/main/assets/filters/no_makeup_beauty"),
            File("src/main/assets/filters/no_makeup_beauty")
        ).firstOrNull { it.exists() && it.isDirectory }
        assertNotNull("no_makeup_beauty package directory must exist in assets/filters", pkgDir)

        val manifestJson = JSONObject(File(pkgDir!!, "manifest.json").readText())
        val manifestVal = PackageValidator.validateManifest(manifestJson)
        assertTrue(
            "no_makeup_beauty manifest must pass strict PackageValidator: ${manifestVal.errorReason}",
            manifestVal.isValid
        )

        val assetsArr = manifestJson.getJSONArray("assets")
        for (i in 0 until assetsArr.length()) {
            val a = assetsArr.getJSONObject(i)
            val assetRelPath = a.optString("path", a.optString("uri", ""))
            val assetFile = File(pkgDir, assetRelPath)
            val assetVal = PackageValidator.validateAssetFile(
                assetFile,
                a.optString("sha256", null),
                a.optLong("bytes", 0L)
            )
            assertTrue("Asset '${assetFile.name}' must pass SHA-256 and byte validation: ${assetVal.errorReason}", assetVal.isValid)
        }

        val nodesArr = manifestJson.getJSONArray("nodes")
        val paramsJson = manifestJson.optJSONObject("params")
        for (i in 0 until nodesArr.length()) {
            val nodeVal = PostRenderer.validatePostNode(nodesArr.getJSONObject(i), paramsJson)
            assertTrue("Post node $i must pass PostRenderer.validatePostNode: ${nodeVal.errorReason}", nodeVal.isValid)
        }

        val effect = Compiler.compile(manifestJson, pkgDir)
        assertEquals("no_makeup_beauty", effect.packageId)
        assertEquals("No Makeup Beauty", effect.packageName)
        assertEquals("xfilter/3", effect.schemaVersion)
        assertEquals(5, effect.nodes.size)

        val faceState = buildSyntheticFaceState(256, 256)
        val evaluator = SceneEvaluator(maxNodes = 16)
        effect.signalBus.evaluate(faceState, 0.1f)
        evaluator.evaluate(effect, faceState, effect.propertySlots)

        val renderGraph = RenderGraph()

        // 1. Tier A: All 5 post passes execute; half-res bilateral + $mask:skin and quarter-res softFocus
        effect.activeTier = "A"
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertEquals(5, renderGraph.lastRenderedPostCount)
        assertEquals(
            listOf(
                "face_subtle_morph",
                "skin_bilateral_smooth",
                "dreamy_soft_focus",
                "natural_warm_grade",
                "portrait_vignette"
            ),
            renderGraph.post.lastExecutedPasses.map { it.nodeId }
        )
        assertTrue(renderGraph.post.lastUsedHalfResTarget)
        assertTrue(renderGraph.post.lastUsedQuarterResTarget)
        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin"))

        // 2. Tier B: All 5 post passes execute with Tier B paramOverrides (radius=2.0, softFocus amount=0.14)
        effect.activeTier = "B"
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertEquals(5, renderGraph.lastRenderedPostCount)
        val tierBSkinParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "skin_bilateral_smooth" }.params
        val tierBSoftParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "dreamy_soft_focus" }.params
        assertEquals(2.0f, tierBSkinParams["radius"] ?: 0f, 1e-4f)
        assertEquals(0.14f, tierBSoftParams["amount"] ?: 0f, 1e-4f)

        // 3. Tier C: hideNodes hides face_subtle_morph and skin_bilateral_smooth -> 3 passes execute
        effect.activeTier = "C"
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertEquals(3, renderGraph.lastRenderedPostCount)
        assertEquals(
            listOf("dreamy_soft_focus", "natural_warm_grade", "portrait_vignette"),
            renderGraph.post.lastExecutedPasses.map { it.nodeId }
        )
        val tierCSoftParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "dreamy_soft_focus" }.params
        val tierCVigParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "portrait_vignette" }.params
        assertEquals(0.10f, tierCSoftParams["amount"] ?: 0f, 1e-4f)
        assertEquals(0.05f, tierCVigParams["amount"] ?: 0f, 1e-4f)

        // 4. OffscreenRenderer photo capture with Tier A active
        effect.activeTier = "A"
        val cameraBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        cameraBmp.eraseColor(Color.rgb(210, 165, 145))
        val photoResult = OffscreenRenderer().renderToPhoto(
            baseBitmap = cameraBmp,
            effect = effect,
            faceState = faceState,
            evaluator = evaluator,
            forceCpuFallback = true
        )
        assertEquals(64, photoResult.bitmap.width)
        assertEquals(64, photoResult.bitmap.height)
        assertTrue("All 5 beauty post nodes must be applied during photo capture", photoResult.renderedNodeCount >= 5)
    }

    @Test
    fun testLightLensPackageValidationCompilationAlignmentAndExecution() {
        val pkgDir = listOf(
            File("app/src/main/assets/filters/light_lens"),
            File("src/main/assets/filters/light_lens")
        ).firstOrNull { it.exists() && it.isDirectory }
        assertNotNull("light_lens package directory must exist in assets/filters", pkgDir)

        val manifestJson = JSONObject(File(pkgDir!!, "manifest.json").readText())
        val manifestVal = PackageValidator.validateManifest(manifestJson)
        assertTrue(
            "light_lens manifest must pass strict PackageValidator: ${manifestVal.errorReason}",
            manifestVal.isValid
        )

        val assetsArr = manifestJson.getJSONArray("assets")
        for (i in 0 until assetsArr.length()) {
            val a = assetsArr.getJSONObject(i)
            val assetRelPath = a.optString("path", a.optString("uri", ""))
            val assetFile = File(pkgDir, assetRelPath)
            val assetVal = PackageValidator.validateAssetFile(
                assetFile,
                a.optString("sha256").takeIf { it.isNotEmpty() },
                a.optLong("bytes", 0L)
            )
            assertTrue("Asset '${assetFile.name}' must pass SHA-256 and byte validation: ${assetVal.errorReason}", assetVal.isValid)
        }

        val effect = Compiler.compile(manifestJson, pkgDir)
        assertEquals("light_lens", effect.packageId)
        assertEquals("Light Lens", effect.packageName)
        assertEquals("xfilter/3", effect.schemaVersion)
        assertEquals(7, effect.nodes.size)
        assertFalse("light_lens uses \$mask:lips and \$mask:cheeks (not iris), so requiresIris must be false", effect.requiresIris)

        // Verify hex color parsing & luminancePreserve on natural_lip_tint (#B96F7B) and soft_cheek_blush (#C98288)
        val lipNode = effect.nodes.first { it.id == "natural_lip_tint" }
        val cheekNode = effect.nodes.first { it.id == "soft_cheek_blush" }
        assertEquals(0xB9 / 255f, lipNode.postParams["colorR"] ?: 0f, 1e-3f)
        assertEquals(0x6F / 255f, lipNode.postParams["colorG"] ?: 0f, 1e-3f)
        assertEquals(0x7B / 255f, lipNode.postParams["colorB"] ?: 0f, 1e-3f)
        assertEquals(0.96f, lipNode.postParams["preserveLuminance"] ?: 0f, 1e-4f)
        assertEquals(0.12f, lipNode.postParams["strength"] ?: 0f, 1e-4f)

        assertEquals(0xC9 / 255f, cheekNode.postParams["colorR"] ?: 0f, 1e-3f)
        assertEquals(0x82 / 255f, cheekNode.postParams["colorG"] ?: 0f, 1e-3f)
        assertEquals(0x88 / 255f, cheekNode.postParams["colorB"] ?: 0f, 1e-3f)
        assertEquals(0.94f, cheekNode.postParams["preserveLuminance"] ?: 0f, 1e-4f)
        assertEquals(0.075f, cheekNode.postParams["strength"] ?: 0f, 1e-4f)

        val faceState = buildSyntheticFaceState(256, 256)
        val lm = faceState.landmarks!!
        // Open inner mouth (#13 upper inner lip) and outer lower lip (#17) so mouth_inner cavity and lower lip tissue are well-separated
        lm[13 * 3 + 1] -= 0.015f
        lm[17 * 3 + 1] += 0.015f
        val regionMask = RegionMaskPass()

        // Anatomical alignment check for $mask:lips, $mask:cheeks, and $mask:skin in normalized view space
        val pt14 = PostRenderer.getLandmarkNormalizedView(faceState, 14)!!
        val pt17 = PostRenderer.getLandmarkNormalizedView(faceState, 17)!!
        val pt13 = PostRenderer.getLandmarkNormalizedView(faceState, 13)!!
        val lowerLipX = pt17[0]
        val lowerLipY = (pt14[1] + pt17[1]) * 0.5f
        val mouthInnerX = (pt13[0] + pt14[0]) * 0.5f
        val mouthInnerY = (pt13[1] + pt14[1]) * 0.5f
        var rightCheekX = 0f
        var rightCheekY = 0f
        for (idx in com.example.engine.pkg.FaceRegions.RIGHT_CHEEK) {
            val pt = PostRenderer.getLandmarkNormalizedView(faceState, idx)!!
            rightCheekX += pt[0]
            rightCheekY += pt[1]
        }
        rightCheekX /= com.example.engine.pkg.FaceRegions.RIGHT_CHEEK.size
        rightCheekY /= com.example.engine.pkg.FaceRegions.RIGHT_CHEEK.size

        assertTrue(
            "Lips mask must align with lower lip tissue",
            regionMask.evaluateNamedMaskAtNormalizedViewPt("\$mask:lips", lowerLipX, lowerLipY, faceState, useSmoothedLandmarks = false) > 0.8f
        )
        assertEquals(
            "Lips mask must be 0.0 inside mouth inner cavity",
            0.0f,
            regionMask.evaluateNamedMaskAtNormalizedViewPt("\$mask:lips", mouthInnerX, mouthInnerY, faceState, useSmoothedLandmarks = false),
            1e-4f
        )
        assertEquals(
            "Cheeks mask must be 1.0 at right cheek center",
            1.0f,
            regionMask.evaluateNamedMaskAtNormalizedViewPt("\$mask:cheeks", rightCheekX, rightCheekY, faceState, useSmoothedLandmarks = false),
            1e-4f
        )
        assertEquals(
            "Cheeks mask must be 0.0 on lips",
            0.0f,
            regionMask.evaluateNamedMaskAtNormalizedViewPt("\$mask:cheeks", lowerLipX, lowerLipY, faceState, useSmoothedLandmarks = false),
            1e-4f
        )

        val evaluator = SceneEvaluator(maxNodes = 16)
        effect.signalBus.evaluate(faceState, 0.1f)
        effect.behaviorRunner.step(0.016f, effect.signalBus, faceState, effect.propertySlots, 100L)
        evaluator.evaluate(effect, faceState, effect.propertySlots)

        val renderGraph = RenderGraph()

        // 1. Tier A: 2 recolor passes ($mask:lips, $mask:cheeks) + 5 post passes execute
        effect.activeTier = "A"
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertEquals(2, renderGraph.lastRenderedCameraPixelCount)
        assertEquals(5, renderGraph.lastRenderedPostCount)
        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:lips"))
        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:cheeks"))
        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin"))

        // 2. Tier B: paramOverrides apply (post:skin_bilateral_smooth.radius=1.9, post:light_soft_focus.amount=0.12)
        effect.activeTier = "B"
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertEquals(2, renderGraph.lastRenderedCameraPixelCount)
        assertEquals(5, renderGraph.lastRenderedPostCount)
        val tierBSkinParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "skin_bilateral_smooth" }.params
        val tierBSoftParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "light_soft_focus" }.params
        assertEquals(1.9f, tierBSkinParams["radius"] ?: 0f, 1e-4f)
        assertEquals(0.12f, tierBSoftParams["amount"] ?: 0f, 1e-4f)

        // 3. Tier C: hideNodes hides face_subtle_morph and skin_bilateral_smooth -> 2 recolor + 3 post passes
        effect.activeTier = "C"
        renderGraph.render(effect, faceState, evaluator, 256, 256)
        assertEquals(2, renderGraph.lastRenderedCameraPixelCount)
        assertEquals(listOf("natural_lip_tint", "soft_cheek_blush"), renderGraph.recolor.lastExecutedRecords.map { it.id })
        assertEquals(3, renderGraph.lastRenderedPostCount)
        assertEquals(
            listOf("light_soft_focus", "light_tone_grade", "portrait_vignette"),
            renderGraph.post.lastExecutedPasses.map { it.nodeId }
        )
        val tierCSoftParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "light_soft_focus" }.params
        val tierCVigParams = renderGraph.post.lastExecutedPasses.first { it.nodeId == "portrait_vignette" }.params
        assertEquals(0.08f, tierCSoftParams["amount"] ?: 0f, 1e-4f)
        assertEquals(0.035f, tierCVigParams["amount"] ?: 0f, 1e-4f)

        // 4. OffscreenRenderer photo capture with Tier A active (verifies lip & cheek recolor + 5 post passes on photo)
        effect.activeTier = "A"
        val cameraBmp = Bitmap.createBitmap(96, 96, Bitmap.Config.ARGB_8888)
        cameraBmp.eraseColor(Color.rgb(210, 175, 160))
        val photoResult = OffscreenRenderer().renderToPhoto(
            baseBitmap = cameraBmp,
            effect = effect,
            faceState = faceState,
            evaluator = evaluator,
            forceCpuFallback = true
        )
        assertEquals(96, photoResult.bitmap.width)
        assertEquals(96, photoResult.bitmap.height)
        assertTrue(
            "All 7 Light Lens nodes (2 recolor + 5 post) must execute during photo capture, got ${photoResult.renderedNodeCount}",
            photoResult.renderedNodeCount >= 7
        )
    }

    @Test
    fun testContinuousSwitchingBetween3dArVrAndBeautyFiltersSeamlessFlowAndZeroLandmarkerReinit() {
        fun loadFilterDir(id: String): File {
            return listOf(
                File("app/src/main/assets/filters/$id"),
                File("src/main/assets/filters/$id")
            ).first { it.exists() && it.isDirectory }
        }

        val sunflowerDir = loadFilterDir("sunflower_forehead")
        val noMakeupDir = loadFilterDir("no_makeup_beauty")
        val lightLensDir = loadFilterDir("light_lens")
        val refBeautyDir = loadFilterDir("ref_beauty")
        val roundGlassesDir = loadFilterDir("round_glasses")

        val sunflowerEffect = Compiler.compile(JSONObject(File(sunflowerDir, "manifest.json").readText()), sunflowerDir)
        val noMakeupEffect = Compiler.compile(JSONObject(File(noMakeupDir, "manifest.json").readText()), noMakeupDir)
        val lightLensEffect = Compiler.compile(JSONObject(File(lightLensDir, "manifest.json").readText()), lightLensDir)
        val refBeautyEffect = Compiler.compile(JSONObject(File(refBeautyDir, "manifest.json").readText()), refBeautyDir)
        val roundGlassesEffect = Compiler.compile(JSONObject(File(roundGlassesDir, "manifest.json").readText()), roundGlassesDir)

        // 1. All 3 beauty filters must compile with requiresPose = true
        assertTrue("no_makeup_beauty must require pose for SignalPipeline tracking", noMakeupEffect.requiresPose)
        assertTrue("light_lens must require pose for SignalPipeline tracking", lightLensEffect.requiresPose)
        assertTrue("light_lens with recolor nodes must require blendshapes for blink attenuation", lightLensEffect.requiresBlendshapes)
        assertTrue(lightLensEffect.requestedBlendshapes.containsAll(listOf("eyeBlinkLeft", "eyeBlinkRight")))
        assertTrue("ref_beauty must require pose for SignalPipeline tracking", refBeautyEffect.requiresPose)

        // 2. Verify continuous switching 3D AR/VR <-> Beauty via CameraSession.applyEffectRequirements
        // never tears down/recreates the MediaPipe Landmarker, always updates activeEntry.config,
        // and continuously delivers valid pose + landmarks through MediaPipeFaceGeometryProvider.handleResult
        // -> SignalPipeline.processSnapshot across many frames (> 350ms hold+fade window per filter).
        val syntheticFaceTemplate = buildSyntheticFaceState(256, 256)
        val templateLm = syntheticFaceTemplate.landmarks!!
        val landmarks478 = List(478) { idx ->
            com.google.mediapipe.tasks.components.containers.NormalizedLandmark.create(
                templateLm[idx * 3],
                templateLm[idx * 3 + 1],
                templateLm[idx * 3 + 2]
            )
        }
        val mpPoseMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, -0.45f, 1f
        )
        val mpBlendList = java.util.Optional.of(
            listOf(
                listOf(
                    com.google.mediapipe.tasks.components.containers.Category.create(0.05f, 0, "eyeBlinkLeft", "eyeBlinkLeft"),
                    com.google.mediapipe.tasks.components.containers.Category.create(0.05f, 1, "eyeBlinkRight", "eyeBlinkRight"),
                    com.google.mediapipe.tasks.components.containers.Category.create(0.20f, 2, "jawOpen", "jawOpen")
                )
            )
        )
        val autoValueClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.AutoValue_FaceLandmarkerResult")
        val resultCtor = autoValueClass.declaredConstructors.first().apply { isAccessible = true }
        val mockMpResult = resultCtor.newInstance(
            1000L,
            listOf(landmarks478),
            mpBlendList,
            java.util.Optional.of(listOf(mpPoseMatrix))
        ) as com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

        var fakeClockNs = 100_000_000L
        com.example.engine.capture.FrameClock.customElapsedRealtimeSource = { fakeClockNs }
        try {
            val createdHandles = mutableListOf<com.example.engine.perception.MediaPipeFaceGeometryProvider.LandmarkerHandle>()
            val mockCtx = object : android.content.ContextWrapper(null) {
                override fun getAssets(): android.content.res.AssetManager? = null
            }
            val provider = com.example.engine.perception.MediaPipeFaceGeometryProvider(
                context = mockCtx,
                dispatchExecutorOverride = java.util.concurrent.Executor { r -> r.run() },
                reaperExecutorOverride = java.util.concurrent.Executor { r -> r.run() }
            )
            provider.enforceCadenceLimit = false
            provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
                val handle = object : com.example.engine.perception.MediaPipeFaceGeometryProvider.LandmarkerHandle {
                    override val delegate = delegate
                    override var isClosed: Boolean = false
                    override fun detectAsync(mpImage: com.google.mediapipe.framework.image.MPImage, timestampMs: Long) {
                        resultListener(mockMpResult, mpImage)
                    }
                    override fun close() {
                        isClosed = true
                    }
                }
                createdHandles.add(handle)
                handle
            }

            val signalPipeline = com.example.engine.signal.SignalPipeline(
                lifecycle = FaceLifecycle(easeInMs = 0L)
            )
            val governor = com.example.engine.governor.PerformanceGovernor(context = null, isLowEndDevice = false)
            val cameraSession = com.example.engine.capture.CameraSession(
                provider = provider,
                signalPipeline = signalPipeline,
                governor = governor
            )
            cameraSession.recordCameraBound()
            cameraSession.startProvider(com.example.engine.perception.ProviderConfig(analysisLongEdgePx = 320))
            assertEquals("Initial startProvider() must create exactly 1 landmarker handle", 1, createdHandles.size)
            assertEquals(1, cameraSession.trackerStartCount)

            val renderGraph = RenderGraph()
            val evaluator = SceneEvaluator(maxNodes = 32)
            val switchSequence = listOf(
                sunflowerEffect,     // 3D AR/VR with blendshapes
                noMakeupEffect,      // Beauty (post warp + bilateral $mask:skin + softFocus)
                lightLensEffect,     // Beauty (recolor $mask:lips/$mask:cheeks + post)
                roundGlassesEffect,  // 3D AR/VR without blendshapes
                noMakeupEffect,      // Back to Beauty
                refBeautyEffect,     // Reference Beauty
                sunflowerEffect,     // Back to 3D AR/VR
                lightLensEffect      // Back to Beauty
            )

            val dummyMpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(
                Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
            ).build()
            val activeInFlightField = provider.javaClass.getDeclaredField("activeInFlightDispatchId").apply { isAccessible = true }
            val inFlightFlagField = provider.javaClass.getDeclaredField("isInferenceInFlight").apply { isAccessible = true }
            val currentDispatchPauseGenField = provider.javaClass.getDeclaredField("currentDispatchPauseGeneration").apply { isAccessible = true }
            val pauseGenField = provider.javaClass.getDeclaredField("pauseGeneration").apply { isAccessible = true }

            // Process 1 initial acquisition frame so FaceLifecycle transitions NO_FACE -> ACQUIRING -> TRACKING
            activeInFlightField.setLong(provider, 1L)
            currentDispatchPauseGenField.setInt(provider, pauseGenField.getInt(provider))
            (inFlightFlagField.get(provider) as java.util.concurrent.atomic.AtomicBoolean).set(true)
            provider.handleResult(
                callbackDispatchId = 1L,
                sourceEntry = provider.activeEntry,
                result = mockMpResult,
                mpImage = dummyMpImage
            )

            for ((stepIdx, effect) in switchSequence.withIndex()) {
                // Execute real CameraSession.applyEffectRequirements(effect)
                cameraSession.applyEffectRequirements(effect)

                // Assert Step B & Step D fix: zero landmarker teardown/recreation and activeEntry.config.outputPose == true
                assertEquals(
                    "Filter switch #$stepIdx (${effect.packageId}) must NOT recreate FaceLandmarker",
                    1,
                    createdHandles.size
                )
                assertEquals(1, cameraSession.trackerStartCount)
                assertFalse(
                    "Filter switch #$stepIdx (${effect.packageId}) must NOT close active FaceLandmarker",
                    createdHandles[0].isClosed
                )
                assertTrue(
                    "Filter switch #$stepIdx (${effect.packageId}) must keep activeEntry.config.outputPose = true",
                    provider.activeEntry!!.config.outputPose
                )

                // Simulate 15 consecutive frames (~500ms per filter switch, well beyond 200ms HOLD + 250ms FADING window)
                // directly through MediaPipeFaceGeometryProvider.handleResult -> CameraSession listener -> SignalPipeline.processSnapshot
                for (frame in 0 until 15) {
                    fakeClockNs += 33_333_333L
                    val dispatchId = (stepIdx * 100 + frame + 1).toLong()
                    activeInFlightField.setLong(provider, dispatchId)
                    currentDispatchPauseGenField.setInt(provider, pauseGenField.getInt(provider))
                    (inFlightFlagField.get(provider) as java.util.concurrent.atomic.AtomicBoolean).set(true)

                    provider.handleResult(
                        callbackDispatchId = dispatchId,
                        sourceEntry = provider.activeEntry,
                        result = mockMpResult,
                        mpImage = dummyMpImage
                    )

                    val faceState = signalPipeline.getLatestState()

                    assertEquals(
                        "Step #$stepIdx (${effect.packageId}) frame $frame must stay in TRACKING state (even after >450ms)",
                        FaceLifecycle.State.TRACKING,
                        faceState.lifecycle
                    )
                    assertTrue(
                        "Step #$stepIdx (${effect.packageId}) frame $frame must remain visible (fadeWeight=${faceState.fadeWeight})",
                        faceState.isVisible && faceState.fadeWeight > 0.9f
                    )
                    assertNotNull(
                        "Step #$stepIdx (${effect.packageId}) frame $frame must preserve non-null landmarks",
                        faceState.landmarks
                    )

                    effect.signalBus.evaluate(faceState, fakeClockNs / 1_000_000_000f)
                    effect.behaviorRunner.step(0.033f, effect.signalBus, faceState, effect.propertySlots, fakeClockNs / 1_000_000L)
                    evaluator.evaluate(effect, faceState, effect.propertySlots)
                    renderGraph.render(effect, faceState, evaluator, 256, 256)

                    if (effect.packageId == "no_makeup_beauty") {
                        assertEquals(5, renderGraph.lastRenderedPostCount)
                        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin"))
                    } else if (effect.packageId == "light_lens") {
                        assertEquals(2, renderGraph.lastRenderedCameraPixelCount)
                        assertEquals(5, renderGraph.lastRenderedPostCount)
                        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:lips"))
                        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:cheeks"))
                        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin"))
                    } else if (effect.packageId == "ref_beauty") {
                        assertEquals(2, renderGraph.lastRenderedPostCount)
                        assertTrue(renderGraph.regionMask.lastEvaluatedMaskIds.contains("\$mask:skin"))
                    }
                }
            }

            cameraSession.stop()
        } finally {
            com.example.engine.capture.FrameClock.customElapsedRealtimeSource = null
        }
    }

    @Test
    fun testChunk2BeautySustainedExecutionZeroRedundantSmoothingContextRecreationAndRetiringLandmarkerSafety() {
        fun loadFilterDir(id: String): File {
            return listOf(
                File("app/src/main/assets/filters/$id"),
                File("src/main/assets/filters/$id")
            ).first { it.exists() && it.isDirectory }
        }
        val lightLensDir = loadFilterDir("light_lens")
        val noMakeupDir = loadFilterDir("no_makeup_beauty")
        val lightLensEffect = Compiler.compile(JSONObject(File(lightLensDir, "manifest.json").readText()), lightLensDir)
        val noMakeupEffect = Compiler.compile(JSONObject(File(noMakeupDir, "manifest.json").readText()), noMakeupDir)

        // 1. Root Cause 2D Verification:
        // Across 120 sustained frames of a multi-mask beauty filter (light_lens evaluates
        // renderRegionsToHalfResFbo + $mask:lips + $mask:cheeks + $mask:skin + warp in a single frame),
        // ControlPointSmoother must step its 118*(x,y) 1-Euro filters ONLY ONCE per unique frame timestamp.
        val renderGraph = RenderGraph()
        val singleSceneSurface = com.example.engine.render.SingleSceneCompositionSurface(renderGraph)
        singleSceneSurface.onSurfaceCreated(256, 256)
        singleSceneSurface.activeEffect = lightLensEffect

        val smoother = renderGraph.regionMask.smoother
        smoother.reset()
        assertEquals(0, smoother.smoothedFrameCount)

        for (frameIdx in 1..120) {
            val tsNs = frameIdx * 33_333_333L
            val faceState = buildSyntheticFaceState(imageWidth = 256, imageHeight = 256, timestampNs = tsNs)

            // Simulate all 5 per-frame beauty landmark smoothing requests at the same timestampNs:
            // (1) CAMERA_PIXEL_PASSES renderRegionsToHalfResFbo
            // (2) $mask:lips
            // (3) $mask:cheeks
            // (4) $mask:skin
            // (5) PostRenderer warp with a custom outLandmarks buffer
            val customOut = FloatArray(smoother.smoothedLandmarks.size)
            val s1 = smoother.smooth(faceState.landmarks, tsNs, true)
            val s2 = smoother.smooth(faceState.landmarks, tsNs, true)
            val s3 = smoother.smooth(faceState.landmarks, tsNs, true)
            val s4 = smoother.smooth(faceState.landmarks, tsNs, true)
            val s5 = smoother.smooth(faceState.landmarks, tsNs, true, customOut)

            assertNotNull(s1)
            assertTrue("Same-frame smooth() calls must return cached smoothedLandmarks in O(1)", s1 === s2)
            assertTrue(s1 === s3)
            assertTrue(s1 === s4)
            org.junit.Assert.assertArrayEquals(
                "Custom outLandmarks buffer on same frame must receive identical cached smoothed landmarks",
                s1!!,
                s5!!,
                1e-6f
            )
            assertEquals(
                "Frame $frameIdx: ControlPointSmoother must increment smoothedFrameCount exactly once per unique timestampNs",
                frameIdx,
                smoother.smoothedFrameCount
            )

            singleSceneSurface.renderFrame(
                faceState = faceState,
                dtSec = 0.033f,
                renderTimeSec = frameIdx * 0.033f,
                nowMs = frameIdx * 33L
            )
            assertEquals(
                "RenderGraph execution on frame $frameIdx must not re-step ControlPointSmoother for the same timestampNs",
                frameIdx,
                smoother.smoothedFrameCount
            )
        }

        // 2. Root Cause 2B & 2C Verification:
        // Context loss & recreation on SingleSceneCompositionSurface / RenderGraph must reset all stale
        // FBO/texture/program/VAO/VBO handles in SingleSceneCompositionSurface, RegionMaskPass, RecolorPass,
        // and PostRenderer, and clear uploadedMaskCache.
        val uploadedMaskCacheField = renderGraph.regionMask.javaClass.getDeclaredField("uploadedMaskCache").apply {
            isAccessible = true
        }
        @Suppress("UNCHECKED_CAST")
        val uploadedMaskCache = uploadedMaskCacheField.get(renderGraph.regionMask) as MutableMap<String, Any>

        // Populate uploadedMaskCache with an entry and verify expr keying + context recreation clear
        val cacheEntryClass = renderGraph.regionMask.javaClass.declaredClasses.first {
            it.simpleName == "UploadedMaskCacheEntry"
        }
        val ctor = cacheEntryClass.getDeclaredConstructor(
            Long::class.javaPrimitiveType,
            Float::class.javaPrimitiveType,
            Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType,
            RegionMaskPass.MaskExpr::class.java
        ).apply { isAccessible = true }

        val entryA = ctor.newInstance(1000L, 1.0f, 7, 128, 128, lightLensEffect.maskDefs["skin"]!!)
        val entryB = ctor.newInstance(1000L, 1.0f, 7, 128, 128, RegionMaskPass.MaskExpr.LandmarkLoop("lips_outer"))
        assertFalse(
            "UploadedMaskCacheEntry with different MaskExpr for the same mask name must not compare equal",
            entryA == entryB
        )
        uploadedMaskCache["\$mask:skin"] = entryA
        assertTrue(uploadedMaskCache.isNotEmpty())

        // Simulate dirty non-zero GL handles before context loss across all render passes
        val postPingPongFboField = renderGraph.post.javaClass.getDeclaredField("pingPongFbo").apply { isAccessible = true }
        val recolorSceneCopyField = renderGraph.recolor.javaClass.getDeclaredField("sceneCopyTexId").apply { isAccessible = true }
        val recolorCapturedFlagField = renderGraph.recolor.javaClass.getDeclaredField("sceneCopyCapturedThisFrame").apply { isAccessible = true }
        val surfaceSceneFboField = singleSceneSurface.javaClass.getDeclaredField("sceneFbo").apply { isAccessible = true }
        val spriteVaoField = renderGraph.sprites.javaClass.getDeclaredField("vao").apply { isAccessible = true }
        val faceMeshVaoField = renderGraph.faceMesh.javaClass.getDeclaredField("vao").apply { isAccessible = true }
        val matUnlitField = renderGraph.materials.javaClass.getDeclaredField("programUnlit").apply { isAccessible = true }

        postPingPongFboField.setInt(renderGraph.post, 42)
        recolorSceneCopyField.setInt(renderGraph.recolor, 43)
        recolorCapturedFlagField.setBoolean(renderGraph.recolor, true)
        surfaceSceneFboField.setInt(singleSceneSurface, 44)
        spriteVaoField.setInt(renderGraph.sprites, 45)
        faceMeshVaoField.setInt(renderGraph.faceMesh, 46)
        matUnlitField.setInt(renderGraph.materials, 47)

        // Trigger context loss and recreation
        singleSceneSurface.activeEffect = noMakeupEffect
        singleSceneSurface.onContextLostAndRecreated()

        assertEquals("PostRenderer.pingPongFbo must be reset to 0 on context recreation", 0, postPingPongFboField.getInt(renderGraph.post))
        assertEquals("RecolorPass.sceneCopyTexId must be reset to 0 on context recreation", 0, recolorSceneCopyField.getInt(renderGraph.recolor))
        assertFalse("RecolorPass.sceneCopyCapturedThisFrame must be reset on context recreation", recolorCapturedFlagField.getBoolean(renderGraph.recolor))
        assertEquals("SingleSceneCompositionSurface.sceneFbo must be reset to 0 on context recreation", 0, surfaceSceneFboField.getInt(singleSceneSurface))
        assertEquals("SpriteRenderer.vao must be reset to 0 on context recreation", 0, spriteVaoField.getInt(renderGraph.sprites))
        assertEquals("FaceMeshRenderer.vao must be reset to 0 on context recreation", 0, faceMeshVaoField.getInt(renderGraph.faceMesh))
        assertTrue("MaterialRegistry.programUnlit stale handle (47) must be replaced on context recreation", matUnlitField.getInt(renderGraph.materials) != 47)
        assertTrue("RegionMaskPass.uploadedMaskCache must be cleared on context recreation", uploadedMaskCache.isEmpty())

        // Verify per-frame RecolorPass.sceneCopyCapturedThisFrame reset on every renderFrame
        recolorCapturedFlagField.setBoolean(renderGraph.recolor, true)
        val resumeState = buildSyntheticFaceState(256, 256).copy(
            timestampNs = 2_000_000_000L,
            poseTimestampNs = 2_000_000_000L
        )
        singleSceneSurface.renderFrame(
            faceState = resumeState,
            dtSec = 0.016f,
            renderTimeSec = 2.0f,
            nowMs = 2000L
        )
        assertFalse(
            "RenderGraph.render must reset RecolorPass.sceneCopyCapturedThisFrame at the start of every frame so camera copy never freezes",
            recolorCapturedFlagField.getBoolean(renderGraph.recolor)
        )

        // 3. Root Cause 2A Verification:
        // If a stale callback completes after a watchdog reset has already started a newer dispatchId (dispatch #2),
        // releaseInFlightSlotAndRetiringLandmarkers(expectedDispatchId = 1L) must NOT clobber dispatch #2 or close
        // dispatch #2's retiring entries prematurely.
        val mockCtx = object : android.content.ContextWrapper(null) {
            override fun getAssets(): android.content.res.AssetManager? = null
        }
        val provider = com.example.engine.perception.MediaPipeFaceGeometryProvider(context = mockCtx)
        val activeInFlightField = provider.javaClass.getDeclaredField("activeInFlightDispatchId").apply { isAccessible = true }
        val inFlightFlagField = provider.javaClass.getDeclaredField("isInferenceInFlight").apply { isAccessible = true }
        val releaseMethod = provider.javaClass.getDeclaredMethod(
            "releaseInFlightSlotAndRetiringLandmarkers",
            Long::class.javaPrimitiveType
        ).apply { isAccessible = true }

        activeInFlightField.setLong(provider, 2L)
        val inFlightAtomic = inFlightFlagField.get(provider) as java.util.concurrent.atomic.AtomicBoolean
        inFlightAtomic.set(true)

        // Late callback from old dispatchId = 1L calls releaseInFlightSlotAndRetiringLandmarkers(1L)
        releaseMethod.invoke(provider, 1L)
        assertEquals("Newer activeInFlightDispatchId (2L) must remain intact when stale dispatchId (1L) finishes", 2L, activeInFlightField.getLong(provider))
        assertTrue("isInferenceInFlight must remain true for dispatchId 2L", inFlightAtomic.get())

        // Matching callback for dispatchId = 2L releases cleanly
        releaseMethod.invoke(provider, 2L)
        assertEquals(0L, activeInFlightField.getLong(provider))
        assertFalse(inFlightAtomic.get())

        // Verify stop() atomically clears activeInFlightDispatchId and isInferenceInFlight under landmarkerLock
        activeInFlightField.setLong(provider, 3L)
        inFlightAtomic.set(true)
        provider.stop()
        assertEquals("stop() must clear activeInFlightDispatchId under landmarkerLock", 0L, activeInFlightField.getLong(provider))
        assertFalse("stop() must clear isInferenceInFlight under landmarkerLock", inFlightAtomic.get())
    }

    @Test
    fun testChunk3AndChunk4BeautyFilterPackageAlignmentAndGpuMaskPathAudit() {
        fun loadFilterDir(id: String): File {
            return listOf(
                File("app/src/main/assets/filters/$id"),
                File("src/main/assets/filters/$id")
            ).first { it.exists() && it.isDirectory }
        }

        val noMakeupDir = loadFilterDir("no_makeup_beauty")
        val lightLensDir = loadFilterDir("light_lens")
        val refBeautyDir = loadFilterDir("ref_beauty")

        val noMakeupEffect = Compiler.compile(JSONObject(File(noMakeupDir, "manifest.json").readText()), noMakeupDir)
        val lightLensEffect = Compiler.compile(JSONObject(File(lightLensDir, "manifest.json").readText()), lightLensDir)
        val refBeautyEffect = Compiler.compile(JSONObject(File(refBeautyDir, "manifest.json").readText()), refBeautyDir)

        // 1. Audit no_makeup_beauty (5 post nodes: warp, bilateral $mask:skin, softFocus with face.fade, colorMatrix, vignette)
        assertEquals(5, noMakeupEffect.nodes.count { it.type == "post" })
        assertTrue(noMakeupEffect.requiresPose)
        assertEquals(478, noMakeupEffect.requiredLandmarkCount)
        assertTrue(noMakeupEffect.maskDefs.containsKey("skin"))
        assertEquals(setOf("A", "B", "C"), noMakeupEffect.tierOverrides.keys)

        // 2. Audit light_lens (2 recolor nodes: $mask:lips, $mask:cheeks + 5 post nodes)
        val lightLensRecolorNodes = lightLensEffect.nodes.filter { it.type == "recolor" }
        assertEquals(2, lightLensRecolorNodes.size)
        assertEquals(5, lightLensEffect.nodes.count { it.type == "post" })
        assertTrue(lightLensEffect.requiresPose)
        assertTrue(lightLensEffect.requiresBlendshapes)
        assertTrue(lightLensEffect.requestedBlendshapes.containsAll(listOf("eyeBlinkLeft", "eyeBlinkRight")))
        val lipRecolor = lightLensRecolorNodes.first { it.id == "natural_lip_tint" }
        val cheekRecolor = lightLensRecolorNodes.first { it.id == "soft_cheek_blush" }
        assertEquals("\$mask:lips", lipRecolor.faceMeshRegion)
        assertEquals("\$mask:cheeks", cheekRecolor.faceMeshRegion)
        assertEquals(0.96f, lipRecolor.postParams["preserveLuminance"] ?: 0f, 1e-4f)
        assertEquals(0.94f, cheekRecolor.postParams["preserveLuminance"] ?: 0f, 1e-4f)
        assertEquals(setOf("A", "B", "C"), lightLensEffect.tierOverrides.keys)

        // 3. Audit ref_beauty (2 post nodes: skinSoften $mask:skin, colorGrade)
        assertEquals(2, refBeautyEffect.nodes.count { it.type == "post" })
        assertTrue(refBeautyEffect.requiresPose)
        assertEquals(478, refBeautyEffect.requiredLandmarkCount)
        assertTrue(refBeautyEffect.maskDefs.containsKey("skin"))

        // 4. Verify collectGpuMaskTriangles handles all mask expressions (including standalone/feathered LuminanceKey)
        // on the hardware GPU path without falling back to CPU evaluation.
        val regionMaskPass = RegionMaskPass()
        val faceState = buildSyntheticFaceState(256, 256)
        val smoothed = regionMaskPass.smoother.smooth(faceState.landmarks, faceState.timestampNs, true)!!
        val collectMethod = regionMaskPass.javaClass.getDeclaredMethod(
            "collectGpuMaskTriangles",
            RegionMaskPass.MaskExpr::class.java,
            FloatArray::class.java,
            FaceState::class.java,
            Boolean::class.javaPrimitiveType,
            Float::class.javaPrimitiveType,
            IntArray::class.java
        ).apply { isAccessible = true }

        for ((maskName, maskExpr) in lightLensEffect.maskDefs) {
            val counts = IntArray(2)
            val ok = collectMethod.invoke(regionMaskPass, maskExpr, smoothed, faceState, false, 0f, counts) as Boolean
            assertTrue("Mask '$maskName' must stay on hardware GPU triangle collection path", ok)
        }
        val featheredLumaExpr = RegionMaskPass.MaskExpr.Intersect(
            RegionMaskPass.MaskExpr.LandmarkLoop("lips_outer"),
            RegionMaskPass.MaskExpr.Feather(RegionMaskPass.MaskExpr.LuminanceKey("\$camera", 0.0f, 0.85f, 0.10f), 0.01f)
        )
        val lumaCounts = IntArray(2)
        val featheredLumaOk = collectMethod.invoke(regionMaskPass, featheredLumaExpr, smoothed, faceState, false, 0f, lumaCounts) as Boolean
        assertTrue("Feathered LuminanceKey expression must stay on hardware GPU path", featheredLumaOk)
        assertTrue("Additive triangle count must be positive", lumaCounts[0] > 0)
    }
}


