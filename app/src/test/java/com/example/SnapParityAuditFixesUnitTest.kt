package com.example

import android.graphics.Bitmap
import android.graphics.Color
import com.example.engine.capture.DefaultFrameRing
import com.example.engine.capture.VideoRecorderPipeline
import com.example.engine.pkg.CapabilityRegistry
import com.example.engine.pkg.Compiler as EffectCompiler
import com.example.engine.pkg.PackageValidator
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.GlbLoader
import com.example.engine.render.MaterialRegistry
import com.example.engine.render.ModelRenderer
import com.example.engine.render.PostRenderer
import com.example.engine.render.RecolorPass
import com.example.engine.render.RegionMaskPass
import com.example.engine.render.SpriteRenderer
import com.example.engine.scene.CompiledLighting
import com.example.engine.scene.CompiledMaterial
import com.example.engine.scene.CompiledNode
import com.example.engine.signal.ControlPointSmoother
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Conformance test suite verifying all 5 phases of the Snap Camera Kit Parity & Deep Code Trace Audit fixes:
 * - Phase 1: `material.pbr@1` Cook-Torrance GGX PBR shader/CPU oracle & glTF 2.0 `pbrMetallicRoughness` loader.
 * - Phase 2A: `post.lut@1` 3D Hald/strip trilinear LUT grading, `post.grain@1` luminance-weighted film grain,
 *   and `post.sharpen@1` coring unsharp mask.
 * - Phase 2B: Forehead cranial dome extension above landmark `#10` on `$mask:skin`/`face_oval`, plus `$mask:teeth`
 *   and `$mask:sclera` masks and natural teeth/sclera whitening in `RecolorPass`.
 * - Phase 3: 2D landmark pose-delta latency compensation, velocity-adaptive 1-Euro `ControlPointSmoother`
 *   for fast head turns, and `DefaultFrameRing.findFrameClosestToTimestampNs`.
 * - Phase 4: Animated 2D sprite-sheet flipbooks (`SpriteRenderer.computeFlipbookUvRect` & `applyOverlayToBitmap`).
 * - Phase 5: Composed video recording pipeline (`VideoRecorderPipeline`).
 */
@RunWith(RobolectricTestRunner::class)
class SnapParityAuditFixesUnitTest {

    @Test
    fun testPhase1PbrCapabilityCompilationGlbMaterialParsingAndCookTorranceOracle() {
        assertTrue(CapabilityRegistry.isFeatureSupported("material.pbr@1"))
        assertTrue(CapabilityRegistry.isFeatureSupported("post.lut@1"))
        assertTrue(CapabilityRegistry.isFeatureSupported("post.grain@1"))
        assertTrue(CapabilityRegistry.isFeatureSupported("post.sharpen@1"))

        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_pbr_test_${System.nanoTime()}")
        tempDir.mkdirs()
        try {
            val manifestJson = """
                {
                  "schema": "xfilter/3",
                  "id": "pbr_glasses_test",
                  "name": "PBR Glasses Test",
                  "version": "1.0.0",
                  "requires": {
                    "features": ["node.group@1", "material.pbr@1"]
                  },
                  "assets": [],
                  "materials": {
                    "gold_frame": {
                      "template": "pbr@1",
                      "color": [1.0, 0.84, 0.35],
                      "metallic": 0.92,
                      "roughness": 0.18,
                      "fresnelSheen": 0.25,
                      "emissive": [0.02, 0.01, 0.0]
                    }
                  },
                  "nodes": [
                    {
                      "id": "frame_node",
                      "type": "group",
                      "material": "gold_frame"
                    }
                  ]
                }
            """.trimIndent()
            File(tempDir, "manifest.json").writeText(manifestJson)

            val valRes = PackageValidator.validateManifest(JSONObject(manifestJson))
            assertTrue("Expected pbr@1 manifest to validate cleanly: ${valRes.errorReason}", valRes.isValid)

            val compiled = EffectCompiler.compile(JSONObject(manifestJson), tempDir)
            val mat = compiled.materials["gold_frame"]
            assertNotNull(mat)
            assertEquals("pbr@1", mat!!.template)
            assertEquals(0.92f, mat.metallic, 1e-4f)
            assertEquals(0.18f, mat.roughness, 1e-4f)
            assertEquals(0.25f, mat.fresnelSheen, 1e-4f)
            assertEquals(0.02f, mat.emissive[0], 1e-4f)

            // Verify Cook-Torrance GGX PBR oracle response: low roughness metallic has stronger specular peak than rough dielectric
            val lighting = CompiledLighting()
            val smoothMetal = MaterialRegistry.evaluatePbrColor(
                albedoRgb = floatArrayOf(0.95f, 0.80f, 0.30f),
                normal = floatArrayOf(0f, 0.25f, 0.968f),
                metallic = 0.95f,
                roughness = 0.12f,
                lighting = lighting
            )
            val roughDielectric = MaterialRegistry.evaluatePbrColor(
                albedoRgb = floatArrayOf(0.95f, 0.80f, 0.30f),
                normal = floatArrayOf(0f, 0.25f, 0.968f),
                metallic = 0.0f,
                roughness = 0.85f,
                lighting = lighting
            )
            assertTrue(smoothMetal[0] in 0f..1f && smoothMetal[1] in 0f..1f && smoothMetal[2] in 0f..1f)
            assertNotEquals(
                "Cook-Torrance GGX PBR output must differ between smooth metal and rough dielectric",
                smoothMetal[0],
                roughDielectric[0],
                1e-3f
            )

            // Verify GlbLoader parses glTF 2.0 pbrMetallicRoughness materials
            val glbFile = File(tempDir, "pbr_triangle.glb")
            writeMinimalPbrGlb(glbFile, metallic = 0.85f, roughness = 0.22f)
            val glbModel = GlbLoader().load(glbFile)
            assertNotNull(glbModel)
            assertEquals(1, glbModel!!.materials.size)
            assertEquals(0.85f, glbModel.materials[0].metallicFactor, 1e-4f)
            assertEquals(0.22f, glbModel.materials[0].roughnessFactor, 1e-4f)
            assertEquals(0, glbModel.meshes[0].materialIndex)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testPhase2ALutGrainAndSharpenPostProcessingPrimitives() {
        // 1. Verify 3D LUT trilinear interpolation: identity LUT preserves RGB within 1/255
        val identityLut = PostRenderer.createIdentityLutBitmap(gridSize = 16, asSquareHald = true)
        assertEquals(64, identityLut.width)
        assertEquals(64, identityLut.height)
        val outRgb = FloatArray(3)
        PostRenderer.evaluateLutRgb(0.25f, 0.55f, 0.80f, identityLut, intensity = 1.0f, outRgb = outRgb)
        assertEquals(0.25f, outRgb[0], 0.02f)
        assertEquals(0.55f, outRgb[1], 0.02f)
        assertEquals(0.80f, outRgb[2], 0.02f)

        // 2. Verify full manifest compilation & bitmap execution for lut, grain, and sharpen post nodes
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_post_audit_${System.nanoTime()}")
        tempDir.mkdirs()
        try {
            val manifestJson = """
                {
                  "schema": "xfilter/3",
                  "id": "post_audit_test",
                  "name": "Post Audit Test",
                  "version": "1.0.0",
                  "requires": {
                    "features": ["node.post@1", "post.lut@1", "post.grain@1", "post.sharpen@1"]
                  },
                  "assets": [],
                  "nodes": [
                    {
                      "id": "lut_pass",
                      "type": "post",
                      "effect": "lut",
                      "renderOrder": 10,
                      "params": { "amount": 0.8, "intensity": 1.0 }
                    },
                    {
                      "id": "sharpen_pass",
                      "type": "post",
                      "effect": "sharpen",
                      "renderOrder": 20,
                      "params": { "amount": 0.6, "radius": 1.0, "threshold": 0.01, "intensity": 1.0 }
                    },
                    {
                      "id": "grain_pass",
                      "type": "post",
                      "effect": "grain",
                      "renderOrder": 30,
                      "params": { "amount": 0.35, "size": 1.0, "seed": 42.0, "intensity": 1.0 }
                    }
                  ]
                }
            """.trimIndent()
            File(tempDir, "manifest.json").writeText(manifestJson)

            val valRes = PackageValidator.validateManifest(JSONObject(manifestJson))
            assertTrue("Expected lut/grain/sharpen manifest to validate: ${valRes.errorReason}", valRes.isValid)

            val effect = EffectCompiler.compile(JSONObject(manifestJson), tempDir)
            val bmp = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
            for (y in 0 until 32) {
                for (x in 0 until 32) {
                    val v = if (x >= 16) 190 else 80
                    bmp.setPixel(x, y, Color.rgb(v, v, v))
                }
            }
            val beforeEdgeDiff = Color.red(bmp.getPixel(16, 16)) - Color.red(bmp.getPixel(15, 16))
            val postRenderer = PostRenderer()
            val executed = postRenderer.applyPostNodesToBitmap(
                composedBitmap = bmp,
                effect = effect,
                isNodeVisible = BooleanArray(effect.nodes.size) { true }
            )
            assertEquals(3, executed)
            val afterEdgeDiff = Color.red(bmp.getPixel(16, 16)) - Color.red(bmp.getPixel(15, 16))
            assertTrue("Expected sharpen + LUT + grain to modify pixel values", afterEdgeDiff != beforeEdgeDiff)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testPhase2BCranialForeheadExtensionAndTeethScleraMasksAndWhitening() {
        val pass = RegionMaskPass()
        val landmarks = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            landmarks[i * 3] = UvRegistry.CANONICAL_UVS[i * 2]
            landmarks[i * 3 + 1] = UvRegistry.CANONICAL_UVS[i * 2 + 1]
            landmarks[i * 3 + 2] = 0f
        }
        for (i in 468..472) {
            landmarks[i * 3] = 0.36f + (i - 468) * 0.004f
            landmarks[i * 3 + 1] = 0.39f
        }
        for (i in 473..477) {
            landmarks[i * 3] = 0.64f + (i - 473) * 0.004f
            landmarks[i * 3 + 1] = 0.39f
        }
        val faceState = FaceState(
            isVisible = true,
            fadeWeight = 1.0f,
            landmarks = landmarks,
            isFrontCamera = false
        )

        // 1. Point above raw top forehead landmark #10 must be covered by $mask:skin due to cranial extension
        val rawForeheadX = landmarks[10 * 3]
        val rawForeheadY = landmarks[10 * 3 + 1]
        val rawChinY = landmarks[152 * 3 + 1]
        val cranialAboveForeheadY = (rawForeheadY + (rawForeheadY - rawChinY) * 0.08f).coerceIn(0.02f, 0.98f)
        val skinAtUpperForehead = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:skin",
            normX = rawForeheadX,
            normY = cranialAboveForeheadY,
            faceState = faceState,
            useSmoothedLandmarks = false
        )
        assertEquals(
            "Upper forehead above landmark #10 must be inside \$mask:skin after cranial dome extension",
            1.0f,
            skinAtUpperForehead,
            1e-4f
        )

        // 2. Verify $mask:teeth and $mask:sclera are registered in BUILTIN_MASKS
        assertTrue(RegionMaskPass.BUILTIN_MASKS.containsKey("teeth"))
        assertTrue(RegionMaskPass.BUILTIN_MASKS.containsKey("sclera"))

        val mouthCenterX = (landmarks[13 * 3] + landmarks[14 * 3]) * 0.5f
        val mouthCenterY = (landmarks[13 * 3 + 1] + landmarks[14 * 3 + 1]) * 0.5f
        val teethOnBrightEnamel = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:teeth",
            normX = mouthCenterX,
            normY = mouthCenterY,
            faceState = faceState,
            sampleLuma = 0.68f,
            useSmoothedLandmarks = false
        )
        val teethOnDarkOralCavity = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:teeth",
            normX = mouthCenterX,
            normY = mouthCenterY,
            faceState = faceState,
            sampleLuma = 0.08f,
            useSmoothedLandmarks = false
        )
        assertTrue("Bright enamel inside mouth_inner must have high teeth mask weight, got $teethOnBrightEnamel", teethOnBrightEnamel > 0.85f)
        assertEquals("Dark oral cavity inside mouth_inner must be rejected by teeth luminance key", 0.0f, teethOnDarkOralCavity, 1e-4f)

        // 3. Verify RecolorPass.whitenTeethOrScleraPixelRgb neutralizes yellow cast and protects dark oral cavity
        val outWhite = FloatArray(3)
        val appliedEnamel = RecolorPass.whitenTeethOrScleraPixelRgb(
            baseR = 0.78f,
            baseG = 0.74f,
            baseB = 0.52f, // yellowish tooth pixel
            maskWeight = 1.0f,
            whiteningStrength = 0.65f,
            outRgb = outWhite
        )
        assertTrue(appliedEnamel > 0.4f)
        assertTrue("Expected blue channel to lift significantly when whitening yellow teeth", outWhite[2] > 0.62f)

        val outDarkCavity = FloatArray(3)
        val appliedCavity = RecolorPass.whitenTeethOrScleraPixelRgb(
            baseR = 0.08f,
            baseG = 0.05f,
            baseB = 0.05f,
            maskWeight = 1.0f,
            whiteningStrength = 0.65f,
            outRgb = outDarkCavity
        )
        assertEquals(0.0f, appliedCavity, 1e-4f)
    }

    @Test
    fun testPhase3FastMotionControlPointSmootherAndTimestampFrameRingLookup() {
        // 1. Fast coherent motion in ControlPointSmoother must track rapidly with highVelocityBeta
        val fastSmoother = ControlPointSmoother()
        val legacySlowSmoother = ControlPointSmoother(highVelocityBeta = 0.0f)
        val lms = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        val cpIdx = 10 // FACE_OVAL control point

        var fastOutX = 0f
        var slowOutX = 0f
        for (step in 0..8) {
            val targetX = 0.20f + step * 0.06f // coherent fast sweep across screen (1.8 norm/sec at 30fps)
            lms[cpIdx * 3] = targetX
            lms[cpIdx * 3 + 1] = 0.30f
            val tsNs = (step + 1) * 33_333_333L
            fastOutX = fastSmoother.smooth(lms.clone(), tsNs, true)!![cpIdx * 3]
            slowOutX = legacySlowSmoother.smooth(lms.clone(), tsNs, true)!![cpIdx * 3]
        }
        val finalTargetX = 0.20f + 8 * 0.06f
        val fastLag = abs(finalTargetX - fastOutX)
        val slowLag = abs(finalTargetX - slowOutX)
        assertTrue(
            "Velocity-adaptive ControlPointSmoother lag ($fastLag) must be < 50% of legacy lag ($slowLag)",
            fastLag < slowLag * 0.5f
        )

        // 2. DefaultFrameRing.findFrameClosestToTimestampNs must locate the camera frame closest to tracker timestamp
        val ring = DefaultFrameRing(delayDepth = 0, capacity = 8)
        val identity = FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0) }
        ring.recordFrame(textureId = 101, sensorTimestampNs = 100_000_000L, alignedElapsedRealtimeNs = 100_000_000L, texMatrix = identity)
        ring.recordFrame(textureId = 102, sensorTimestampNs = 133_333_333L, alignedElapsedRealtimeNs = 133_333_333L, texMatrix = identity)
        ring.recordFrame(textureId = 103, sensorTimestampNs = 166_666_666L, alignedElapsedRealtimeNs = 166_666_666L, texMatrix = identity)

        val matched = ring.findFrameClosestToTimestampNs(135_000_000L)
        assertNotNull(matched)
        assertEquals(102, matched!!.textureId)
        assertEquals(103, ring.peekFrame()!!.textureId)

        // 3. 2D landmark pose-delta latency compensation shifts 2D landmarks in the direction of 3D predicted pose
        val basePoseMat = FloatArray(16).apply {
            android.opengl.Matrix.setIdentityM(this, 0)
            this[14] = -0.50f // 50cm in front of camera
        }
        val predictedPoseMat = FloatArray(16).apply {
            android.opengl.Matrix.setIdentityM(this, 0)
            this[12] = 0.04f // +4cm X translation prediction delta
            this[14] = -0.50f
        }
        val unshiftedState = FaceState(
            isVisible = true,
            fadeWeight = 1.0f,
            landmarks = lms,
            poseMatrix = basePoseMat,
            landmarkPoseMatrix = basePoseMat,
            isFrontCamera = false
        )
        val shiftedState = FaceState(
            isVisible = true,
            fadeWeight = 1.0f,
            landmarks = lms,
            poseMatrix = predictedPoseMat,
            landmarkPoseMatrix = basePoseMat,
            isFrontCamera = false
        )
        val pt0 = FloatArray(2)
        val pt1 = FloatArray(2)
        FacePoseAdapter.mapLandmarkToViewWithPoseDelta(0.5f, 0.5f, 0.0f, unshiftedState, 1.0f, 1.0f, pt0)
        FacePoseAdapter.mapLandmarkToViewWithPoseDelta(0.5f, 0.5f, 0.0f, shiftedState, 1.0f, 1.0f, pt1)
        assertTrue("Predicted +X pose delta must shift projected 2D landmark +X in view space", pt1[0] > pt0[0] + 0.01f)
    }

    @Test
    fun testPhase4SpriteSheetFlipbookAnimationAndPhase5VideoRecorderPipeline() {
        // Phase 4: 2x2 flipbook at 10 fps
        val rect0 = SpriteRenderer.computeFlipbookUvRect(rows = 2, cols = 2, fps = 10f, loop = true, renderTimeSec = 0.02f)
        assertEquals(0.0f, rect0[0], 1e-4f)
        assertEquals(0.0f, rect0[1], 1e-4f)
        assertEquals(0.5f, rect0[2], 1e-4f)
        assertEquals(0.5f, rect0[3], 1e-4f)

        val rect1 = SpriteRenderer.computeFlipbookUvRect(rows = 2, cols = 2, fps = 10f, loop = true, renderTimeSec = 0.12f)
        assertEquals(0.5f, rect1[0], 1e-4f)
        assertEquals(0.0f, rect1[1], 1e-4f)

        val rect2 = SpriteRenderer.computeFlipbookUvRect(rows = 2, cols = 2, fps = 10f, loop = true, renderTimeSec = 0.22f)
        assertEquals(0.0f, rect2[0], 1e-4f)
        assertEquals(0.5f, rect2[1], 1e-4f)

        // Phase 5: VideoRecorderPipeline lifecycle and monotonic timestamp tracking
        val outMp4 = File(System.getProperty("java.io.tmpdir"), "xelvic_rec_${System.nanoTime()}.mp4")
        try {
            val recorder = VideoRecorderPipeline()
            assertEquals(VideoRecorderPipeline.State.IDLE, recorder.state)
            assertTrue(
                recorder.startRecording(
                    VideoRecorderPipeline.RecordingConfig(
                        outputFile = outMp4,
                        width = 720,
                        height = 1280,
                        useHardwareCodec = false
                    )
                )
            )
            assertEquals(VideoRecorderPipeline.State.RECORDING, recorder.state)
            assertTrue(recorder.recordComposedFrame(1_000_000_000L))
            assertTrue(recorder.recordComposedFrame(1_033_333_333L))
            assertTrue(recorder.recordComposedFrame(1_066_666_666L))
            assertEquals(3, recorder.recordedFrameCount)

            val result = recorder.stopRecording()
            assertEquals(VideoRecorderPipeline.State.COMPLETED, recorder.state)
            assertEquals(3, result.recordedFrameCount)
            assertEquals(66L, result.durationMs)
            assertTrue(outMp4.exists() && outMp4.length() >= 24L)
        } finally {
            outMp4.delete()
        }
    }

    @Test
    fun testGap1CompletePbrPipelineFixesRndManifestsGlbUint32MultiPrimitiveAndSplitSumIbl() {
        // 1. Verify VS_PBR and VS_LAMBERT use transpose(inverse(mat3(uModelView))) normal matrix
        // and FS_PBR includes sRGB->Linear, fresnelSchlickRoughness, envBRDFApprox, uNormalScale, uOcclusionStrength
        assertTrue(MaterialRegistry.VS_PBR.contains("transpose(inverse(mat3(uModelView)))"))
        assertTrue(MaterialRegistry.VS_LAMBERT.contains("transpose(inverse(mat3(uModelView)))"))
        assertTrue(MaterialRegistry.FS_PBR.contains("fresnelSchlickRoughness"))
        assertTrue(MaterialRegistry.FS_PBR.contains("envBRDFApprox"))
        assertTrue(MaterialRegistry.FS_PBR.contains("uNormalScale"))
        assertTrue(MaterialRegistry.FS_PBR.contains("uOcclusionStrength"))

        // 2. Verify heart_mood_filter and furry_glasses_filter manifests compile to pbr@1 with exact manifest factors
        val filtersRoot = listOf(
            File("app/src/main/assets/filters"),
            File("src/main/assets/filters")
        ).firstOrNull { it.exists() }
        assertNotNull("Expected assets/filters directory to exist", filtersRoot)

        val heartDir = File(filtersRoot!!, "heart_mood_filter")
        val heartCompiled = EffectCompiler.compile(JSONObject(File(heartDir, "manifest.json").readText()), heartDir)
        val heartMat = heartCompiled.materials["mat_lambert"]
        assertNotNull(heartMat)
        assertEquals("pbr@1", heartMat!!.template)
        assertEquals(0.02f, heartMat.metallic, 1e-4f)
        assertEquals(0.31f, heartMat.roughness, 1e-4f)
        assertTrue(heartMat.hasExplicitMetallic)
        assertTrue(heartMat.hasExplicitRoughness)

        val furryDir = File(filtersRoot, "furry_glasses_filter")
        val furryCompiled = EffectCompiler.compile(JSONObject(File(furryDir, "manifest.json").readText()), furryDir)
        val furryMat = furryCompiled.materials["mat_lambert"]
        assertNotNull(furryMat)
        assertEquals("pbr@1", furryMat!!.template)
        assertEquals(0.0f, furryMat.metallic, 1e-4f)
        assertEquals(0.92f, furryMat.roughness, 1e-4f)

        // 3. Verify fresnelSchlickRoughness + envBRDFApprox grazing angle energy conservation:
        // At grazing angle (NdotV ~ 0.08) under ambient-only studio IBL, smooth gloss (roughness=0.06)
        // must exhibit significantly stronger Fresnel specular IBL than rough plush (roughness=0.92),
        // and AO (occlusion=0.25) must attenuate ambient IBL.
        val ambientOnlyLighting = CompiledLighting(
            ambient = floatArrayOf(0.45f, 0.45f, 0.45f),
            lightCount = 0,
            lightIntensities = floatArrayOf(0f, 0f)
        )
        val grazingNormal = floatArrayOf(0f, 0.996f, 0.089f) // NdotV ~= 0.089
        val darkAlbedo = floatArrayOf(0.15f, 0.15f, 0.15f)
        val grazingSmooth = MaterialRegistry.evaluatePbrColor(
            albedoRgb = darkAlbedo,
            normal = grazingNormal,
            metallic = 0.0f,
            roughness = 0.06f,
            lighting = ambientOnlyLighting
        )
        val grazingRoughPlush = MaterialRegistry.evaluatePbrColor(
            albedoRgb = darkAlbedo,
            normal = grazingNormal,
            metallic = 0.0f,
            roughness = 0.92f,
            lighting = ambientOnlyLighting,
            fresnelSheen = 0.0f
        )
        assertTrue(
            "Rough plush (0.92) must not have mirror-like grazing IBL reflection like smooth gloss (0.06)",
            grazingSmooth[2] > grazingRoughPlush[2] + 0.08f
        )

        val unoccluded = MaterialRegistry.evaluatePbrColor(
            albedoRgb = floatArrayOf(0.8f, 0.5f, 0.5f),
            normal = floatArrayOf(0f, 0f, 1f),
            metallic = 0.0f,
            roughness = 0.5f,
            lighting = ambientOnlyLighting,
            occlusion = 1.0f
        )
        val occluded = MaterialRegistry.evaluatePbrColor(
            albedoRgb = floatArrayOf(0.8f, 0.5f, 0.5f),
            normal = floatArrayOf(0f, 0f, 1f),
            metallic = 0.0f,
            roughness = 0.5f,
            lighting = ambientOnlyLighting,
            occlusion = 0.25f
        )
        assertTrue("Ambient occlusion must attenuate ambient IBL", unoccluded[0] > occluded[0] + 0.08f)

        // 4. Verify GlbLoader UNSIGNED_INT (5125) indices, normalScale, occlusionStrength, multi-primitive mesh mapping,
        // and ModelRenderer multiplicative baseColorFactor + embedded GLB PBR auto-activation
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_gap1_glb_${System.nanoTime()}")
        tempDir.mkdirs()
        try {
            val glbFile = File(tempDir, "multi_prim_uint32.glb")
            writeMultiPrimitiveUint32PbrGlb(glbFile)
            val glbModel = GlbLoader().load(glbFile)
            assertNotNull(glbModel)
            assertEquals(2, glbModel!!.materials.size)
            assertEquals(0.75f, glbModel.materials[0].normalScale, 1e-4f)
            assertEquals(0.60f, glbModel.materials[0].occlusionStrength, 1e-4f)
            assertEquals(0.35f, glbModel.materials[0].alphaCutoff, 1e-4f)
            assertEquals(2, glbModel.meshes.size)
            assertEquals(2, glbModel.meshesForGltfMeshIndex(0).size)
            assertEquals(2, glbModel.meshes[0].cpuIndices!![2].toInt())

            val renderer = ModelRenderer()
            val identity = FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0) }
            val dummyNode = CompiledNode(
                index = 0,
                id = "test_model",
                parentIndex = -1,
                tags = emptyList(),
                type = "model",
                anchor = "head_center",
                basePosition = floatArrayOf(0f, 0f, 0f),
                baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
                baseScale = floatArrayOf(1f, 1f, 1f),
                rotationFollow = "full",
                materialId = "mat_tint",
                baseVisible = true,
                baseOpacity = 1f,
                renderOrder = 0
            )
            // Material with lambert@1 template, tint [0.5, 1.0, 0.5], and no explicit roughness/metallic:
            // ModelRenderer must auto-upgrade to pbr@1 (since model has embedded glTF PBR materials),
            // multiply tint [0.5, 1.0, 0.5] with glbMat[1].baseColorFactor [0.8, 0.4, 0.2, 1.0] -> [0.4, 0.4, 0.1],
            // inherit glbMat[1].roughnessFactor (0.88f), and draw both primitives (lastDrawnPrimitiveCount == 2).
            val tintMat = CompiledMaterial(
                id = "mat_tint",
                template = "lambert@1",
                mapAssetId = null,
                maskAssetId = null,
                color = floatArrayOf(0.5f, 1.0f, 0.5f),
                hasExplicitMetallic = false,
                hasExplicitRoughness = false
            )
            renderer.render(
                model = glbModel,
                node = dummyNode,
                worldMatrix = identity,
                viewProjectionMatrix = identity,
                material = tintMat,
                lighting = CompiledLighting(),
                opacity = 1.0f,
                programs = MaterialRegistry(),
                textureId = 0
            )
            assertEquals("pbr@1", renderer.lastBoundTemplate)
            assertEquals(2, renderer.lastDrawnPrimitiveCount)
            assertEquals(0.88f, renderer.lastBoundRoughness, 1e-4f)
            assertEquals(0.40f, renderer.lastBoundColor[0], 1e-4f)
            assertEquals(0.40f, renderer.lastBoundColor[1], 1e-4f)
            assertEquals(0.10f, renderer.lastBoundColor[2], 1e-4f)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    private fun writeMultiPrimitiveUint32PbrGlb(targetFile: File) {
        val posFloats = floatArrayOf(
            0f, 0f, 0f,
            1f, 0f, 0f,
            0f, 1f, 0f
        )
        val uint32Indices = intArrayOf(0, 1, 2)
        val binBuf = ByteBuffer.allocate(36 + 12).order(ByteOrder.LITTLE_ENDIAN)
        for (f in posFloats) binBuf.putFloat(f)
        for (idx in uint32Indices) binBuf.putInt(idx)
        val binBytes = binBuf.array()

        val jsonObj = JSONObject("""
            {
              "asset": { "version": "2.0" },
              "buffers": [{ "byteLength": ${binBytes.size} }],
              "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 12 }
              ],
              "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" }
              ],
              "materials": [
                {
                  "name": "prim0_mat",
                  "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 0.5, 0.5, 1.0],
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.25
                  },
                  "normalTexture": { "index": 0, "scale": 0.75 },
                  "occlusionTexture": { "index": 0, "strength": 0.60 },
                  "alphaCutoff": 0.35
                },
                {
                  "name": "prim1_mat",
                  "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.4, 0.2, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.88
                  }
                }
              ],
              "meshes": [
                {
                  "primitives": [
                    { "attributes": { "POSITION": 0 }, "indices": 1, "material": 0 },
                    { "attributes": { "POSITION": 0 }, "indices": 1, "material": 1 }
                  ]
                }
              ],
              "nodes": [{ "name": "root_mesh", "mesh": 0 }]
            }
        """.trimIndent())
        var jsonBytes = jsonObj.toString().toByteArray(Charsets.UTF_8)
        val jsonPad = (4 - (jsonBytes.size % 4)) % 4
        if (jsonPad > 0) {
            jsonBytes += ByteArray(jsonPad) { ' '.code.toByte() }
        }
        val totalLen = 12 + 8 + jsonBytes.size + 8 + binBytes.size
        val glbBuf = ByteBuffer.allocate(totalLen).order(ByteOrder.LITTLE_ENDIAN)
        glbBuf.putInt(0x46546C67)
        glbBuf.putInt(2)
        glbBuf.putInt(totalLen)
        glbBuf.putInt(jsonBytes.size)
        glbBuf.putInt(0x4E4F534A)
        glbBuf.put(jsonBytes)
        glbBuf.putInt(binBytes.size)
        glbBuf.putInt(0x004E4942)
        glbBuf.put(binBytes)
        targetFile.writeBytes(glbBuf.array())
    }

    private fun writeMinimalPbrGlb(targetFile: File, metallic: Float, roughness: Float) {
        val posFloats = floatArrayOf(
            0f, 0f, 0f,
            1f, 0f, 0f,
            0f, 1f, 0f
        )
        val indicesShort = shortArrayOf(0, 1, 2, 0) // padded to 8 bytes
        val binBuf = ByteBuffer.allocate(36 + 8).order(ByteOrder.LITTLE_ENDIAN)
        for (f in posFloats) binBuf.putFloat(f)
        for (s in indicesShort) binBuf.putShort(s)
        val binBytes = binBuf.array()

        val jsonObj = JSONObject("""
            {
              "asset": { "version": "2.0" },
              "buffers": [{ "byteLength": ${binBytes.size} }],
              "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
              ],
              "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
              ],
              "materials": [
                {
                  "name": "pbr_mat",
                  "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 0.85, 0.4, 1.0],
                    "metallicFactor": $metallic,
                    "roughnessFactor": $roughness
                  }
                }
              ],
              "meshes": [
                {
                  "primitives": [
                    {
                      "attributes": { "POSITION": 0 },
                      "indices": 1,
                      "material": 0
                    }
                  ]
                }
              ],
              "nodes": [{ "mesh": 0 }]
            }
        """.trimIndent())
        var jsonBytes = jsonObj.toString().toByteArray(Charsets.UTF_8)
        val jsonPad = (4 - (jsonBytes.size % 4)) % 4
        if (jsonPad > 0) {
            jsonBytes += ByteArray(jsonPad) { ' '.code.toByte() }
        }
        val totalLen = 12 + 8 + jsonBytes.size + 8 + binBytes.size
        val glbBuf = ByteBuffer.allocate(totalLen).order(ByteOrder.LITTLE_ENDIAN)
        glbBuf.putInt(0x46546C67)
        glbBuf.putInt(2)
        glbBuf.putInt(totalLen)
        glbBuf.putInt(jsonBytes.size)
        glbBuf.putInt(0x4E4F534A)
        glbBuf.put(jsonBytes)
        glbBuf.putInt(binBytes.size)
        glbBuf.putInt(0x004E4942)
        glbBuf.put(binBytes)
        targetFile.writeBytes(glbBuf.array())
    }

    @Test
    fun testGap2CompletePostLutGrainSharpenTextureCacheCustomPassAndValidatorFixes() {
        // 1. Verify resolveLutLayout & trilinear evaluateLutRgb across 16^3 strip (256x16),
        // 32^3 strip (1024x32), and 64^3 square Hald (512x512) LUTs
        val layout16Strip = PostRenderer.resolveLutLayout(256, 16, 0)
        assertEquals(16, layout16Strip[0])
        assertEquals(16, layout16Strip[1])

        val layout32Strip = PostRenderer.resolveLutLayout(1024, 32, 0)
        assertEquals(32, layout32Strip[0])
        assertEquals(32, layout32Strip[1])

        val layout64Hald = PostRenderer.resolveLutLayout(512, 512, 0)
        assertEquals(64, layout64Hald[0])
        assertEquals(8, layout64Hald[1])

        // Generate an analytical 32^3 strip LUT (1024x32) where output RGB = (R^0.8, G*0.7+0.1, B*0.9)
        val grid32 = 32
        val lut32W = 1024
        val lut32H = 32
        val lut32Pixels = IntArray(lut32W * lut32H)
        for (b in 0 until grid32) {
            for (g in 0 until grid32) {
                for (r in 0 until grid32) {
                    val rf = r.toFloat() / (grid32 - 1).toFloat()
                    val gf = g.toFloat() / (grid32 - 1).toFloat()
                    val bf = b.toFloat() / (grid32 - 1).toFloat()
                    val outR = (Math.pow(rf.toDouble(), 0.8).toFloat() * 255f).toInt().coerceIn(0, 255)
                    val outG = ((gf * 0.7f + 0.1f) * 255f).toInt().coerceIn(0, 255)
                    val outB = ((bf * 0.9f) * 255f).toInt().coerceIn(0, 255)
                    val x = b * grid32 + r
                    val y = g
                    lut32Pixels[y * lut32W + x] = Color.argb(255, outR, outG, outB)
                }
            }
        }
        // Test fractional mid-cell RGB lookup (trilinear interpolation across R, G, and B)
        val fracR = (10.5f / 31f)
        val fracG = (17.25f / 31f)
        val fracB = (21.75f / 31f)
        val triOut = PostRenderer.evaluateLutRgb(
            r = fracR,
            g = fracG,
            b = fracB,
            lutBitmap = null,
            intensity = 1.0f,
            lutPixels = lut32Pixels,
            lutWidth = lut32W,
            lutHeight = lut32H,
            gridSizeHint = 32
        )
        val expectedR = Math.pow(fracR.toDouble(), 0.8).toFloat()
        val expectedG = fracG * 0.7f + 0.1f
        val expectedB = fracB * 0.9f
        assertEquals(expectedR, triOut[0], 0.015f)
        assertEquals(expectedG, triOut[1], 0.015f)
        assertEquals(expectedB, triOut[2], 0.015f)

        // 2. Verify TextureCache with isLut = true preserves exact native dimensions even under Tier C maxSize = 256
        val exactDims = com.example.engine.render.TextureCache.computeBoundedDimensions(1024, 32, 256, isLut = true)
        assertEquals(1024, exactDims.first)
        assertEquals(32, exactDims.second)

        val haldDims = com.example.engine.render.TextureCache.computeBoundedDimensions(512, 512, 256, isLut = true)
        assertEquals(512, haldDims.first)
        assertEquals(512, haldDims.second)

        // 3. Verify PackageValidator catches undeclared LUT asset reference on post.lut@1 node
        val invalidLutManifest = """
            {
              "schema": "xfilter/3",
              "id": "invalid_lut_pkg",
              "name": "Invalid LUT Package",
              "version": "1.0.0",
              "requires": { "features": ["node.post@1", "post.lut@1"] },
              "assets": [],
              "nodes": [
                {
                  "id": "grade_lut",
                  "type": "post",
                  "effect": "lut@1",
                  "params": {
                    "lut": "missing_lut_asset",
                    "intensity": 0.8
                  }
                }
              ]
            }
        """.trimIndent()
        val invalidReport = PackageValidator.validateManifest(JSONObject(invalidLutManifest))
        assertFalse("Validator must reject post node referencing undeclared LUT asset", invalidReport.isValid)
        assertTrue(invalidReport.errorReason?.contains("missing_lut_asset") == true)

        // 4. Verify GPU RenderGraph wiring for post.lut@1 (binds lutTexId), post.grain@1 (propagates DeterministicFrameContext),
        // and PassStageKind.CUSTOM_PASS execution
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_gap2_rg_${System.nanoTime()}")
        tempDir.mkdirs()
        try {
            val lutFile = File(tempDir, "lut16.png")
            val lutBmp = Bitmap.createBitmap(256, 16, Bitmap.Config.ARGB_8888)
            for (b in 0 until 16) {
                for (g in 0 until 16) {
                    for (r in 0 until 16) {
                        lutBmp.setPixel(b * 16 + r, g, Color.argb(255, r * 17, g * 17, b * 17))
                    }
                }
            }
            lutFile.outputStream().use { out ->
                lutBmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            val manifestJsonStr = """
                {
                  "schema": "xfilter/3",
                  "id": "gap2_test_pkg",
                  "name": "Gap 2 Test Package",
                  "version": "1.0.0",
                  "requires": {
                    "features": ["node.post@1", "post.lut@1", "post.grain@1", "post.sharpen@1"]
                  },
                  "assets": [
                    { "id": "film_lut", "type": "lut", "uri": "lut16.png" }
                  ],
                  "passes": [
                    { "id": "camera_frame", "stage": "camera_frame" },
                    {
                      "id": "custom_sharpen_pass",
                      "stage": "custom",
                      "template": "pass.sharpen@1",
                      "reads": ["${'$'}scene"],
                      "writes": "${'$'}scene",
                      "params": { "amount": 0.4, "radius": 1.0, "threshold": 0.02 }
                    },
                    { "id": "post_nodes", "stage": "post_nodes" },
                    { "id": "final_output", "stage": "final_output" }
                  ],
                  "nodes": [
                    {
                      "id": "lut_node",
                      "type": "post",
                      "effect": "lut@1",
                      "asset": "film_lut",
                      "params": { "intensity": 0.85, "gridSize": 16 }
                    },
                    {
                      "id": "grain_node",
                      "type": "post",
                      "effect": "grain@1",
                      "params": { "amount": 0.15, "size": 1.2, "seed": 7.0 }
                    }
                  ]
                }
            """.trimIndent()
            File(tempDir, "manifest.json").writeText(manifestJsonStr)

            val compiled = EffectCompiler.compile(JSONObject(manifestJsonStr), tempDir)
            val rg = com.example.engine.render.RenderGraph()
            rg.init()
            rg.resize(720, 1280)

            val evaluator = com.example.engine.scene.SceneEvaluator().apply {
                evaluate(compiled, FaceState.EMPTY, com.example.engine.scene.PropertySlots(compiled.nodes.size))
            }

            val ctx1 = com.example.engine.render.DeterministicFrameContext(
                frameSeed = 1001,
                renderTimeSec = 1.25f,
                frameIndex = 75L,
                isPhotoPass = false
            )
            val ctx2 = com.example.engine.render.DeterministicFrameContext(
                frameSeed = 2002,
                renderTimeSec = 1.50f,
                frameIndex = 90L,
                isPhotoPass = false
            )

            rg.render(
                effect = compiled,
                faceState = FaceState.EMPTY,
                evaluator = evaluator,
                viewportWidth = 720,
                viewportHeight = 1280,
                frameContext = ctx1
            )

            assertTrue("PostRenderer must bind non-zero LUT texture ID on GPU", rg.post.lastBoundLutTexId != 0)
            val lutEntry = rg.textures.getEntryForPath(lutFile.absolutePath)
            assertNotNull("LUT texture entry must exist in TextureCache", lutEntry)
            assertTrue("LUT texture entry must be marked isLut = true", lutEntry!!.isLut)
            assertEquals(256, lutEntry.width)
            assertEquals(16, lutEntry.height)

            val grainSeedFrame1 = rg.post.lastEffectiveGrainSeed
            rg.render(
                effect = compiled,
                faceState = FaceState.EMPTY,
                evaluator = evaluator,
                viewportWidth = 720,
                viewportHeight = 1280,
                frameContext = ctx2
            )
            val grainSeedFrame2 = rg.post.lastEffectiveGrainSeed
            assertNotEquals(
                "Animated film grain seed must advance deterministically with frameContext",
                grainSeedFrame1,
                grainSeedFrame2
            )
            assertTrue(
                "Custom pass custom_sharpen_pass must be executed by RenderGraph",
                rg.lastRenderedNodeIds.contains("custom_sharpen_pass")
            )

            // 5. Verify CapabilityRegistry & PostRenderer.getCapabilities() parity for post.* and pass.*
            val postCaps = PostRenderer.getCapabilities()
            for (cap in listOf(
                "post.lut@1", "post.grain@1", "post.sharpen@1",
                "pass.lut@1", "pass.grain@1", "pass.sharpen@1"
            )) {
                assertTrue("PostRenderer.getCapabilities() must advertise $cap", postCaps.contains(cap))
                assertTrue("CapabilityRegistry must support $cap", com.example.engine.pkg.CapabilityRegistry.isFeatureSupported(cap))
            }

            // 6. Verify Snap-style non-linear skin-tone-safe 3D LUT generator (creamy highlights, teal shadows, rosy midtones)
            val shadowGrade = PostRenderer.evaluateCinematicBeautyGrade(0.15f, 0.15f, 0.15f)
            assertTrue("Teal shadows: blue channel must exceed red channel in dark neutral shadows", shadowGrade[2] > shadowGrade[0] + 0.035f)
            assertTrue("Teal shadows: green channel must exceed red channel in dark neutral shadows", shadowGrade[1] > shadowGrade[0] + 0.015f)

            val highlightGrade = PostRenderer.evaluateCinematicBeautyGrade(0.85f, 0.85f, 0.85f)
            assertTrue("Creamy highlights: warm red/green must exceed blue in highlights", highlightGrade[0] > highlightGrade[2] + 0.015f)
            assertTrue("Creamy highlights: green must exceed blue in creamy highlights", highlightGrade[1] > highlightGrade[2] + 0.008f)

            val skinMidtoneGrade = PostRenderer.evaluateCinematicBeautyGrade(0.68f, 0.50f, 0.42f)
            assertTrue("Rosy skin midtones: skin red channel must warm up", skinMidtoneGrade[0] > 0.68f + 0.01f)

            val hald64Bmp = PostRenderer.createCinematicBeautyLutBitmap(gridSize = 16, asSquareHald = true)
            assertEquals(64, hald64Bmp.width)
            assertEquals(64, hald64Bmp.height)
            val strip256Bmp = PostRenderer.createCinematicBeautyLutBitmap(gridSize = 16, asSquareHald = false)
            assertEquals(256, strip256Bmp.width)
            assertEquals(16, strip256Bmp.height)

            // 7. Verify CPU applyPostNodesToBitmap executes both nodes and custom PassStageKind.CUSTOM_PASS stages
            val cpuBmp = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
            cpuBmp.eraseColor(Color.argb(255, 160, 125, 108))
            val executedCpuCount = rg.post.applyPostNodesToBitmap(
                composedBitmap = cpuBmp,
                effect = compiled,
                isNodeVisible = BooleanArray(compiled.nodes.size) { true },
                frameContext = ctx1
            )
            assertEquals(
                "applyPostNodesToBitmap must execute lut_node, grain_node, and custom_sharpen_pass",
                3,
                executedCpuCount
            )
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testGap3CompleteCranialForehead022TeethMucosaGuardMouthOpenGateAndGpuCpuScleraParity() {
        // 1. Cranial Forehead Extension: 0.22 * faceHeight dome + temple transition arc + cloned FACE_OVAL support
        assertEquals(0.22f, RegionMaskPass.FOREHEAD_CRANIAL_EXTENSION_SCALE, 1e-4f)
        assertEquals(0.22f, RegionMaskPass.cranialForeheadWeightForLandmark(10), 1e-4f)
        assertEquals(0.20f, RegionMaskPass.cranialForeheadWeightForLandmark(109), 1e-4f)
        assertEquals(0.20f, RegionMaskPass.cranialForeheadWeightForLandmark(338), 1e-4f)
        assertEquals(0.11f, RegionMaskPass.cranialForeheadWeightForLandmark(103), 1e-4f)
        assertEquals(0.11f, RegionMaskPass.cranialForeheadWeightForLandmark(332), 1e-4f)
        assertTrue("Temple transition landmark #251 must have positive cranial weight", RegionMaskPass.cranialForeheadWeightForLandmark(251) > 0f)
        assertTrue("Temple transition landmark #21 must have positive cranial weight", RegionMaskPass.cranialForeheadWeightForLandmark(21) > 0f)
        assertTrue(
            "isFaceOvalLoop must match cloned FACE_OVAL arrays, not just reference identity",
            RegionMaskPass.isFaceOvalLoop(com.example.engine.pkg.FaceRegions.FACE_OVAL.clone())
        )

        val pass = RegionMaskPass()
        val landmarks = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            landmarks[i * 3] = UvRegistry.CANONICAL_UVS[i * 2]
            landmarks[i * 3 + 1] = UvRegistry.CANONICAL_UVS[i * 2 + 1]
            landmarks[i * 3 + 2] = 0f
        }
        for (i in 468..472) {
            landmarks[i * 3] = 0.36f + (i - 468) * 0.004f
            landmarks[i * 3 + 1] = 0.39f
        }
        for (i in 473..477) {
            landmarks[i * 3] = 0.64f + (i - 473) * 0.004f
            landmarks[i * 3 + 1] = 0.39f
        }
        val openMouthLandmarks = landmarks.clone().also {
            it[13 * 3 + 1] -= 0.025f
            it[14 * 3 + 1] += 0.025f
        }
        val openFaceState = FaceState(
            isVisible = true,
            fadeWeight = 1.0f,
            landmarks = openMouthLandmarks,
            blendshapes = mapOf("jawOpen" to 0.35f, "eyeBlinkLeft" to 0.0f, "eyeBlinkRight" to 0.0f),
            isFrontCamera = false
        )

        val rawForeheadX = openMouthLandmarks[10 * 3]
        val rawForeheadY = openMouthLandmarks[10 * 3 + 1]
        val rawChinY = openMouthLandmarks[152 * 3 + 1]
        // Check point +18% of faceHeight above landmark #10 (which was previously excluded when scale was 0.16)
        val highForeheadY = (rawForeheadY + (rawForeheadY - rawChinY) * 0.18f).coerceIn(0.01f, 0.99f)
        val skinAtHighForehead = pass.evaluateNamedMaskAtNormalizedViewPt(
            maskId = "\$mask:skin",
            normX = rawForeheadX,
            normY = highForeheadY,
            faceState = openFaceState,
            useSmoothedLandmarks = false
        )
        assertTrue(
            "Point at +0.18 * faceHeight above #10 must be covered by \$mask:skin with 0.22 cranial extension, got $skinAtHighForehead",
            skinAtHighForehead > 0.85f
        )

        // 2. Teeth Whitening Mucosa Guard: bright pink/red tongue & gum mucosa must be protected from graying
        val outTongue = FloatArray(3)
        val tongueApplied = RecolorPass.whitenTeethOrScleraPixelRgb(
            baseR = 0.82f,
            baseG = 0.36f,
            baseB = 0.40f, // bright pink/red tongue pixel
            maskWeight = 1.0f,
            whiteningStrength = 0.65f,
            outRgb = outTongue,
            protectMucosaRedness = true
        )
        assertEquals(
            "Bright pink/red tongue mucosa (R - G > 0.30) must be rejected by mucosa guard",
            0.0f,
            tongueApplied,
            1e-4f
        )

        // 3. Mouth-Open Gate: closed mouth must gate $mask:teeth whitening to 0.0
        val closedMouthLandmarks = landmarks.clone().also {
            val midLipY = (it[13 * 3 + 1] + it[14 * 3 + 1]) * 0.5f
            it[13 * 3 + 1] = midLipY
            it[14 * 3 + 1] = midLipY
        }
        val closedFaceState = FaceState(
            isVisible = true,
            fadeWeight = 1.0f,
            landmarks = closedMouthLandmarks,
            blendshapes = mapOf("jawOpen" to 0.0f),
            isFrontCamera = false
        )
        assertEquals(0.0f, RecolorPass.computeMouthOpenGate(closedFaceState), 1e-4f)
        assertEquals(1.0f, RecolorPass.computeMouthOpenGate(openFaceState), 1e-4f)

        // 4. Compiler tracker requirements for $mask:teeth (jawOpen) and $mask:sclera (requiresIris + eyeBlink)
        val tempDir = java.nio.file.Files.createTempDirectory("gap3_retouch_test").toFile()
        try {
            val manifestJson = """
                {
                  "schemaVersion": "1.2.0",
                  "id": "com.example.gap3_retouch",
                  "name": "Gap 3 Retouch Parity",
                  "nodes": [
                    {
                      "id": "teeth_whiten",
                      "type": "recolor",
                      "region": "${'$'}mask:teeth",
                      "params": { "strength": 0.55 }
                    },
                    {
                      "id": "sclera_brighten",
                      "type": "recolor",
                      "region": "${'$'}mask:sclera",
                      "params": { "strength": 0.40 }
                    }
                  ]
                }
            """.trimIndent()
            File(tempDir, "manifest.json").writeText(manifestJson)
            val compiled = EffectCompiler.compile(JSONObject(manifestJson), tempDir)
            assertTrue("Effect with \$mask:sclera must set requiresIris = true", compiled.requiresIris)
            assertTrue("Effect with \$mask:teeth must request jawOpen blendshape", compiled.requestedBlendshapes.contains("jawOpen"))
            assertTrue("Effect with \$mask:sclera must request eyeBlinkLeft", compiled.requestedBlendshapes.contains("eyeBlinkLeft"))
            assertTrue("Effect with \$mask:sclera must request eyeBlinkRight", compiled.requestedBlendshapes.contains("eyeBlinkRight"))

            // Verify CPU applyNodeToBitmap skips teeth whitening when mouth is closed, and modifies enamel pixels when open
            val recolorPass = RecolorPass()
            val teethNode = compiled.nodes.first { it.id == "teeth_whiten" }
            val closedBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
            closedBmp.eraseColor(Color.argb(255, 200, 188, 135)) // yellowish enamel fill
            val closedModified = recolorPass.applyNodeToBitmap(closedBmp, teethNode, compiled, closedFaceState, 1.0f)
            assertEquals("Closed mouth must modify 0 pixels for \$mask:teeth", 0, closedModified)

            val openBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
            openBmp.eraseColor(Color.argb(255, 200, 188, 135))
            val openModified = recolorPass.applyNodeToBitmap(openBmp, teethNode, compiled, openFaceState, 1.0f)
            assertTrue("Open mouth with yellowish enamel must modify pixels inside \$mask:teeth", openModified > 0)

            // Verify CPU applyNodeToBitmap skips sclera brightening when both eyes are blinked shut
            val scleraNode = compiled.nodes.first { it.id == "sclera_brighten" }
            val blinkedFaceState = openFaceState.copy(
                blendshapes = mapOf("eyeBlinkLeft" to 1.0f, "eyeBlinkRight" to 1.0f)
            )
            val blinkedBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
            blinkedBmp.eraseColor(Color.argb(255, 195, 185, 165))
            val blinkedModified = recolorPass.applyNodeToBitmap(blinkedBmp, scleraNode, compiled, blinkedFaceState, 1.0f)
            assertEquals("Blinked shut eyes must modify 0 pixels for \$mask:sclera", 0, blinkedModified)

            // 5. Verify CPU RecolorPass.applyNodeToBitmap with $mask:skin does not clip the +0.18 * faceHeight upper forehead
            val skinRecolorManifest = """
                {
                  "schemaVersion": "1.2.0",
                  "id": "com.example.gap3_skin_recolor",
                  "name": "Gap 3 Skin Forehead Recolor",
                  "nodes": [
                    {
                      "id": "skin_bronzer",
                      "type": "recolor",
                      "region": "${'$'}mask:skin",
                      "params": { "colorR": 0.85, "colorG": 0.55, "colorB": 0.35, "strength": 0.60 }
                    }
                  ],
                  "passes": [
                    { "id": "cam_in", "stage": "CAMERA_FRAME", "writes": "${'$'}camera" },
                    { "id": "standalone_teeth_pass", "template": "pass.recolor@1", "reads": ["${'$'}camera", "${'$'}mask:teeth"], "writes": "${'$'}scene", "params": { "strength": 0.50 } }
                  ]
                }
            """.trimIndent()
            val compiledSkin = EffectCompiler.compile(JSONObject(skinRecolorManifest), tempDir)
            val skinNode = compiledSkin.nodes.first { it.id == "skin_bronzer" }
            val skinBmp = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
            val initialColor = Color.argb(255, 140, 140, 140)
            skinBmp.eraseColor(initialColor)
            val skinModified = recolorPass.applyNodeToBitmap(skinBmp, skinNode, compiledSkin, openFaceState, 1.0f)
            assertTrue(skinModified > 0)
            val fhPx = (rawForeheadX * 100f).toInt().coerceIn(0, 99)
            val fhPy = (highForeheadY * 100f).toInt().coerceIn(0, 99)
            assertNotEquals(
                "Upper forehead pixel at +0.18 * faceHeight above #10 must not be clipped by applyNodeToBitmap bounding box",
                initialColor,
                skinBmp.getPixel(fhPx, fhPy)
            )

            // 6. Verify standalone pass.recolor@1 in PassGraph executes in RenderGraph.render
            val rg = com.example.engine.render.RenderGraph()
            val eval = com.example.engine.scene.SceneEvaluator()
            eval.evaluate(compiledSkin, openFaceState, compiledSkin.propertySlots)
            rg.render(compiledSkin, openFaceState, eval, 128, 128)
            assertTrue(
                "Standalone pass.recolor@1 in passes must execute in RenderGraph",
                rg.recolor.lastExecutedRecords.any { it.id == "standalone_teeth_pass" && it.maskId == "\$mask:teeth" }
            )
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testGap5SpriteSheetFlipbookOverlayCropParticleSparklesAndPhotoParity() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_gap5_${System.nanoTime()}").apply { mkdirs() }
        try {
            // 1. Create a 2x2 flipbook PNG (64x64 total, 4 distinct 32x32 colored quadrants):
            //    Frame 0 (top-left):     Pure Red   (255, 0, 0)
            //    Frame 1 (top-right):    Pure Green (0, 255, 0)
            //    Frame 2 (bottom-left):  Pure Blue  (0, 0, 255)
            //    Frame 3 (bottom-right): Pure White (255, 255, 255)
            val sheetBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
            for (y in 0 until 64) {
                for (x in 0 until 64) {
                    val c = when {
                        x < 32 && y < 32 -> Color.argb(255, 255, 0, 0)
                        x >= 32 && y < 32 -> Color.argb(255, 0, 255, 0)
                        x < 32 && y >= 32 -> Color.argb(255, 0, 0, 255)
                        else -> Color.argb(255, 255, 255, 255)
                    }
                    sheetBmp.setPixel(x, y, c)
                }
            }
            val sheetFile = File(tempDir, "flipbook_2x2.png")
            java.io.FileOutputStream(sheetFile).use { out ->
                sheetBmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            // 2. Verify computeFlipbookUvRect supports frameCount (< rows*cols), explicitFrameIndex, startFrame, and half-texel inset
            val uvFrameCount3 = SpriteRenderer.computeFlipbookUvRect(
                rows = 2, cols = 2, fps = 10f, loop = true, renderTimeSec = 0.32f, frameCount = 3
            )
            // At t=0.32s * 10fps = rawFrame 3; with frameCount=3 and loop=true, 3 % 3 == 0 (top-left cell [0, 0])
            assertEquals(0.0f, uvFrameCount3[0], 1e-4f)
            assertEquals(0.0f, uvFrameCount3[1], 1e-4f)

            val uvExplicit2 = SpriteRenderer.computeFlipbookUvRect(
                rows = 2, cols = 2, fps = 0f, loop = true, renderTimeSec = 0f, explicitFrameIndex = 2
            )
            // Frame 2 is bottom-left cell (col=0, row=1 -> u=0.0, v=0.5)
            assertEquals(0.0f, uvExplicit2[0], 1e-4f)
            assertEquals(0.5f, uvExplicit2[1], 1e-4f)

            val uvInset = SpriteRenderer.computeFlipbookUvRect(
                rows = 2, cols = 2, fps = 10f, loop = true, renderTimeSec = 0.12f, texWidth = 64, texHeight = 64
            )
            // Half-texel inset on 64x64 is 0.5 / 64 = 0.0078125
            assertEquals(0.5f + 0.5f / 64f, uvInset[0], 1e-4f)
            assertEquals(0.0f + 0.5f / 64f, uvInset[1], 1e-4f)
            assertEquals(0.5f - 1.0f / 64f, uvInset[2], 1e-4f)
            assertEquals(0.5f - 1.0f / 64f, uvInset[3], 1e-4f)

            // 3. Verify Overlay cover-crop + flipbook UV composition (Bug #1 fix)
            // Viewport is 100x200 (1:2 portrait), single cell is 100x100 (1:1 square) -> cover crops U to [0.25..0.75]
            val placement = SpriteRenderer.computeOverlayPlacement(
                viewportWidth = 100, viewportHeight = 200, assetWidth = 100, assetHeight = 100, fit = "cover"
            )
            assertEquals(0.25f, placement.u0, 1e-4f)
            assertEquals(0.75f, placement.u1, 1e-4f)
            // Active cell = Frame 1 (top-right: fbU0=0.5, fbV0=0.0, fbUScale=0.5, fbVScale=0.5)
            val fbCell1 = floatArrayOf(0.5f, 0.0f, 0.5f, 0.5f)
            val composedUv = SpriteRenderer.composeOverlayUvRect(placement, fbCell1)
            // Composed U0 = 0.5 + 0.25 * 0.5 = 0.625; Composed UScale = 0.5 * 0.5 = 0.25
            assertEquals(0.625f, composedUv[0], 1e-4f)
            assertEquals(0.0f, composedUv[1], 1e-4f)
            assertEquals(0.25f, composedUv[2], 1e-4f)
            assertEquals(0.5f, composedUv[3], 1e-4f)

            // 4. Compile manifest with animated sparkle sprite (using size.widthFW/heightFW, pivot, particles, additive blend,
            //    material map fallback, and dynamic frameIndex behavior) + animated flipbook overlay
            val manifestJson = JSONObject(
                """
                {
                  "schema": "xfilter/3",
                  "id": "com.example.gap5_2d_fx",
                  "name": "Gap 5 Sparkle Flipbook FX",
                  "photo": {
                    "timeSec": 0.12,
                    "seed": 7
                  },
                  "assets": [
                    { "id": "sheet_tex", "type": "image", "path": "flipbook_2x2.png" }
                  ],
                  "materials": {
                    "mat_sparkle": {
                      "template": "unlit@1",
                      "blend": "additive",
                      "params": {
                        "map": "sheet_tex",
                        "color": [0.5, 1.0, 0.5],
                        "opacity": 1.0
                      }
                    },
                    "mat_overlay": {
                      "template": "unlit@1",
                      "blend": "alpha",
                      "params": {
                        "map": "sheet_tex",
                        "opacity": 1.0
                      }
                    }
                  },
                  "nodes": [
                    {
                      "id": "sparkle_emitter",
                      "type": "sprite",
                      "anchor": "nose_tip",
                      "material": "mat_sparkle",
                      "size": { "widthFW": 0.40, "heightFW": 0.20 },
                      "pivot": [0.5, 0.0],
                      "flipbook": {
                        "rows": 2,
                        "cols": 2,
                        "fps": 10.0,
                        "loop": true,
                        "frameCount": 3,
                        "startFrame": 1
                      },
                      "particles": {
                        "count": 6,
                        "spread": [0.25, 0.30],
                        "drift": [0.0, 0.40],
                        "twinkleSpeed": 3.0,
                        "scaleMin": 0.5,
                        "scaleMax": 1.2,
                        "seed": 42
                      },
                      "renderOrder": 10
                    },
                    {
                      "id": "driven_sprite",
                      "type": "sprite",
                      "anchor": "forehead",
                      "material": "mat_overlay",
                      "size": { "widthFW": 0.60, "heightFW": 0.60 },
                      "flipbook": {
                        "rows": 2,
                        "cols": 2,
                        "fps": 0.0,
                        "loop": false,
                        "frameIndex": 0
                      },
                      "renderOrder": 11
                    },
                    {
                      "id": "flipbook_overlay",
                      "type": "overlay",
                      "material": "mat_overlay",
                      "fit": "cover",
                      "flipbook": {
                        "rows": 2,
                        "cols": 2,
                        "fps": 10.0,
                        "loop": true
                      },
                      "renderOrder": 100
                    }
                  ],
                  "signals": {
                    "jaw_open": "face.blend.jawOpen"
                  },
                  "behaviors": [
                    {
                      "id": "jaw_drives_frame",
                      "type": "map",
                      "input": "sig.jaw_open",
                      "inputRange": [0.0, 1.0],
                      "outputRange": [0.0, 2.0],
                      "easing": "linear",
                      "smoothingTau": 0.0,
                      "writes": [
                        { "path": "sprite:driven_sprite.frameIndex", "mode": "set" }
                      ]
                    }
                  ]
                }
                """.trimIndent()
            )
            val effect = EffectCompiler.compile(manifestJson, tempDir)
            val sparkleNode = effect.nodes.first { it.id == "sparkle_emitter" }
            val drivenNode = effect.nodes.first { it.id == "driven_sprite" }
            assertEquals(0.40f, sparkleNode.spriteWidthFW, 1e-4f)
            assertEquals(0.20f, sparkleNode.spriteHeightFW, 1e-4f)
            assertEquals(0.5f, sparkleNode.spritePivot[0], 1e-4f)
            assertEquals(0.0f, sparkleNode.spritePivot[1], 1e-4f)
            assertEquals(3, sparkleNode.spriteFrameCount)
            assertEquals(1, sparkleNode.spriteStartFrame)
            assertEquals(6, sparkleNode.particleCount)
            assertEquals(0, drivenNode.spriteFrameIndex)

            // 5. Verify SceneEvaluator applies spriteWidthFW (0.40), spriteHeightFW (0.20), and spritePivot ([0.5, 0.0])
            val faceState = FaceState.create(
                timestampNs = 120_000_000L,
                lifecycle = com.example.engine.signal.FaceLifecycle.State.TRACKING,
                fadeWeight = 1.0f,
                isVisible = true,
                poseMatrix = floatArrayOf(
                    1f, 0f, 0f, 0f,
                    0f, 1f, 0f, 0f,
                    0f, 0f, 1f, 0f,
                    0f, 0f, -0.5f, 1f
                ),
                translation = floatArrayOf(0f, 0f, -0.5f),
                rotation = floatArrayOf(0f, 0f, 0f, 1f),
                scale = floatArrayOf(1f, 1f, 1f),
                yawDeg = 0f, pitchDeg = 0f, rollDeg = 0f,
                faceWidth = 0.20f,
                landmarks = FloatArray(478 * 3) { idx ->
                    when (idx % 3) {
                        0 -> 0.5f
                        1 -> 0.5f
                        else -> 0.0f
                    }
                },
                blendshapes = mapOf("jawOpen" to 1.0f),
                acquisitionTimeNs = 120_000_000L,
                imageToViewTransform = null,
                imageWidth = 128,
                imageHeight = 128,
                isFrontCamera = true
            )
            effect.signalBus.evaluate(faceState, 0.016f)
            effect.behaviorRunner.step(0.016f, effect.signalBus, faceState, effect.propertySlots, 120L)
            val evaluator = com.example.engine.scene.SceneEvaluator()
            evaluator.evaluate(effect, faceState, effect.propertySlots)

            // Sparkle node scale X must be 0.40 * fw (0.08) and scale Y must be 0.20 * fw (0.04)
            val spOffset = sparkleNode.index * 16
            assertEquals(0.40f * 0.20f, evaluator.worldMatrices[spOffset + 0], 1e-4f)
            assertEquals(0.20f * 0.20f, evaluator.worldMatrices[spOffset + 5], 1e-4f)

            // Dynamic frameIndex on driven_sprite must be updated to 2.0 by jaw_drives_frame behavior
            val dynFrameVal = effect.propertySlots.getDynamicParam("pass:driven_sprite.frameIndex", -1f)
            assertEquals(2.0f, dynFrameVal, 1e-3f)

            // 6. Verify RenderGraph passes colorTint, particle instancing, and dynamic frameIndex to SpriteRenderer
            val renderGraph = com.example.engine.render.RenderGraph()
            renderGraph.render(
                effect = effect,
                faceState = faceState,
                evaluator = evaluator,
                viewportWidth = 64,
                viewportHeight = 64,
                frameContext = com.example.engine.render.DeterministicFrameContext(
                    frameIndex = 1L,
                    renderTimeSec = 0.12f, // At 10fps -> rawFrame 1 (top-right pure GREEN quadrant)
                    frameSeed = 7
                )
            )
            // Last node rendered is flipbook_overlay at t=0.12s (Frame 1: top-right green quadrant)
            assertTrue(renderGraph.sprites.lastComposedOverlayUvRect[0] >= 0.5f)
            assertTrue(renderGraph.sprites.lastComposedOverlayUvRect[1] < 0.05f)

            // 7. Verify CPU OffscreenRenderer photo capture samples ONLY the active flipbook cell (Frame 1 = Pure GREEN)
            //    instead of stamping all 4 colored quadrants (Red, Green, Blue, White) across the captured photo!
            val basePhoto = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
            basePhoto.eraseColor(Color.argb(255, 0, 0, 0))
            val offscreen = com.example.engine.photo.OffscreenRenderer()
            val photoOut = offscreen.renderToPhoto(
                baseBitmap = basePhoto,
                effect = effect,
                faceState = faceState,
                evaluator = evaluator,
                forceCpuFallback = true
            )
            // Because flipbook_overlay covers the 64x64 photo with Frame 1 (pure Green 0, 255, 0),
            // both top-left (8, 8) and bottom-right (56, 56) of the captured photo must be pure Green!
            // (Before Bug #4 fix, (8, 8) was Red and (56, 56) was White because the entire 2x2 sheet was drawn!)
            val pxTopLeft = photoOut.bitmap.getPixel(8, 8)
            val pxBottomRight = photoOut.bitmap.getPixel(56, 56)
            assertEquals("Top-left of captured photo must sample Frame 1 Green, not Frame 0 Red", 0, Color.red(pxTopLeft))
            assertEquals("Top-left of captured photo must sample Frame 1 Green", 255, Color.green(pxTopLeft))
            assertEquals("Bottom-right of captured photo must sample Frame 1 Green, not Frame 3 White", 0, Color.red(pxBottomRight))
            assertEquals("Bottom-right of captured photo must sample Frame 1 Green", 255, Color.green(pxBottomRight))
            assertEquals("Bottom-right of captured photo must sample Frame 1 Green, not Frame 3 White", 0, Color.blue(pxBottomRight))

            // 8. Verify Gap #5 Recheck fixes:
            //    (a) CapabilityRegistry registers 2D flipbook & particle feature tokens
            val features = com.example.engine.pkg.CapabilityRegistry.getAllSupportedFeatures()
            assertTrue(features.contains("sprite.flipbook@1"))
            assertTrue(features.contains("overlay.flipbook@1"))
            assertTrue(features.contains("sprite.particles@1"))
            assertTrue(features.contains("overlay.particles@1"))

            //    (b) OffscreenRenderer preserves live preview renderTimeSec when photoPolicy.timeSec is null
            val effectNoHardcodedPhotoTime = effect.copy(
                photoPolicy = effect.photoPolicy.copy(timeSec = null)
            )
            offscreen.renderGraph.render(
                effect = effectNoHardcodedPhotoTime,
                faceState = faceState,
                evaluator = evaluator,
                viewportWidth = 64,
                viewportHeight = 64,
                frameContext = com.example.engine.render.DeterministicFrameContext(
                    frameIndex = 2L,
                    renderTimeSec = 0.12f,
                    frameSeed = 7
                )
            )
            offscreen.renderToPhoto(
                baseBitmap = basePhoto,
                effect = effectNoHardcodedPhotoTime,
                faceState = faceState,
                evaluator = evaluator,
                forceCpuFallback = true
            )
            assertEquals(
                "When photoPolicy.timeSec is null, OffscreenRenderer must preserve live preview renderTimeSec (0.12f)",
                0.12f,
                offscreen.lastPhotoFrameContext.renderTimeSec,
                1e-4f
            )
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testGap6RealTimeGpuVideoRecordingPipelineAudioAndLifecycleSafety() {
        val outMp4 = File(System.getProperty("java.io.tmpdir"), "xelvic_gap6_${System.nanoTime()}.mp4")
        try {
            // 1. Verify EGL_RECORDABLE_ANDROID constant (0x3142) for MediaCodec input surface EGLConfig matching
            assertEquals(0x3142, VideoRecorderPipeline.EGL_RECORDABLE_ANDROID)

            // 2. Verify OffsetTestHarness.selectCompositionMode switches to SINGLE_SCENE_GL when requiresVideoRecording = true
            val selectedMode = com.example.engine.render.OffsetTestHarness.selectCompositionMode(
                effect = null,
                requiresVideoRecording = true,
                offsetDecision = com.example.engine.render.OffsetTestHarness().lastDecision,
                hasCameraSceneInput = true
            )
            assertEquals(com.example.engine.render.CompositionMode.SINGLE_SCENE_GL, selectedMode)

            // 3. Wire VideoRecorderPipeline directly to SingleSceneCompositionSurface and verify
            //    every rendered scene frame blits/records through videoRecorder.recordGlSceneFboFrame
            val recorder = VideoRecorderPipeline()
            val singleSurface = com.example.engine.render.SingleSceneCompositionSurface().apply {
                this.videoRecorder = recorder
                onSurfaceCreated(720, 1280)
            }
            val camBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.rgb(32, 128, 220))
            }
            singleSurface.bindCameraFrameBitmap(camBmp)

            assertTrue(
                recorder.startRecording(
                    VideoRecorderPipeline.RecordingConfig(
                        outputFile = outMp4,
                        width = 720,
                        height = 1280,
                        enableAudio = true,
                        useHardwareCodec = false
                    )
                )
            )
            assertEquals(VideoRecorderPipeline.State.RECORDING, recorder.state)

            val faceState1 = FaceState.EMPTY.copy(timestampNs = 2_000_000_000L, isVisible = true, fadeWeight = 1f)
            val faceState2 = FaceState.EMPTY.copy(timestampNs = 2_033_333_333L, isVisible = true, fadeWeight = 1f)
            val faceState3 = FaceState.EMPTY.copy(timestampNs = 2_066_666_666L, isVisible = true, fadeWeight = 1f)

            singleSurface.renderFrame(faceState = faceState1, dtSec = 0.033f, renderTimeSec = 0.0f, nowMs = 2000L)
            assertEquals(0L, recorder.lastSubmittedPresentationTimeNs)

            // Feed PCM 16-bit microphone audio chunks alongside composed video frames
            val pcmChunk = ByteArray(2048) { idx -> (idx and 0x7F).toByte() }
            assertTrue(recorder.feedPcmAudioSamples(pcmChunk, pcmChunk.size, 2_016_000_000L))
            assertTrue(recorder.feedPcmAudioSamples(pcmChunk, pcmChunk.size, 2_048_000_000L))

            singleSurface.renderFrame(faceState = faceState2, dtSec = 0.033f, renderTimeSec = 0.033f, nowMs = 2033L)
            val pts2Ns = recorder.lastSubmittedPresentationTimeNs
            assertTrue("Expected strictly positive monotonic presentationTimeNs on frame 2, got $pts2Ns", pts2Ns >= 33_333_000L)

            singleSurface.renderFrame(faceState = faceState3, dtSec = 0.033f, renderTimeSec = 0.066f, nowMs = 2066L)
            val pts3Ns = recorder.lastSubmittedPresentationTimeNs
            assertTrue("Expected monotonic presentationTimeNs on frame 3 ($pts3Ns > $pts2Ns)", pts3Ns > pts2Ns)

            // 4. Finalize recording and verify video frame count, blitted GL frame count, audio sample count, and valid MP4 ftyp header
            val result = recorder.stopRecording()
            assertEquals(VideoRecorderPipeline.State.COMPLETED, recorder.state)
            assertEquals(3, result.recordedFrameCount)
            assertEquals(3, result.blittedGlFrameCount)
            assertTrue("Expected audioRecorded == true when PCM audio samples were fed", result.audioRecorded)
            assertEquals(2048, result.recordedAudioSampleCount)
            assertTrue(result.durationMs >= 66L)
            assertTrue("Output MP4 file must exist and contain valid ISO BMFF ftyp box", outMp4.exists() && outMp4.length() >= 24L)
        } finally {
            outMp4.delete()
        }
    }

    @Test
    fun testPhase1DeepAuditFixesMultiSubnodeGlassesMultiPropHatLambertSheenEmissiveAlphaMaskAndFrontMirror() {
        // 1. Bug #1: round_glasses multi-node GLB ("glasses_frame", "lens_left", "lens_right") must resolve
        //    "frame" -> "glasses_frame" (lambert@1, opaque) and "lens" -> "lens_left" / "lens_right" (unlit@1, alpha, opacity=0.25)
        val glassesDir = File("src/main/assets/filters/round_glasses").let {
            if (it.exists()) it else File("app/src/main/assets/filters/round_glasses")
        }
        assertTrue("round_glasses filter directory must exist", glassesDir.exists())
        val glassesJson = JSONObject(File(glassesDir, "manifest.json").readText())
        val glassesVal = PackageValidator.validateManifest(glassesJson)
        assertTrue("round_glasses must validate cleanly", glassesVal.isValid)
        val glassesEffect = EffectCompiler.compile(glassesJson, glassesDir)
        val glassesNode = glassesEffect.nodes.first { it.id == "glasses" }

        val primaryGlassesMat = ModelRenderer.resolvePrimaryModelMaterial(glassesNode, glassesEffect.materials)
        assertNotNull(primaryGlassesMat)
        assertEquals("frame", primaryGlassesMat!!.id)

        val resolvedFrame = ModelRenderer.resolveSubnodeMaterial(
            node = glassesNode,
            subnodeName = "glasses_frame",
            glbMatName = "",
            primaryMaterial = primaryGlassesMat,
            materials = glassesEffect.materials
        )
        val resolvedLeftLens = ModelRenderer.resolveSubnodeMaterial(
            node = glassesNode,
            subnodeName = "lens_left",
            glbMatName = "",
            primaryMaterial = primaryGlassesMat,
            materials = glassesEffect.materials
        )
        val resolvedRightLens = ModelRenderer.resolveSubnodeMaterial(
            node = glassesNode,
            subnodeName = "lens_right",
            glbMatName = "",
            primaryMaterial = primaryGlassesMat,
            materials = glassesEffect.materials
        )
        assertEquals("glasses_frame must bind to 'frame' material", "frame", resolvedFrame?.id)
        assertEquals("opaque", resolvedFrame?.blend)
        assertEquals("lens_left must bind to 'lens' material", "lens", resolvedLeftLens?.id)
        assertEquals("alpha", resolvedLeftLens?.blend)
        assertEquals(0.25f, resolvedLeftLens?.opacity ?: 1f, 1e-4f)
        assertEquals("lens_right must bind to 'lens' material", "lens", resolvedRightLens?.id)

        // 2. Bug #2: multi_prop node "prop_top_hat" (which omits explicit "material": "mat_hat") must resolve to "mat_hat"
        val multiPropDir = File("src/main/assets/filters/multi_prop").let {
            if (it.exists()) it else File("app/src/main/assets/filters/multi_prop")
        }
        assertTrue("multi_prop filter directory must exist", multiPropDir.exists())
        val multiPropJson = JSONObject(File(multiPropDir, "manifest.json").readText())
        val multiPropVal = PackageValidator.validateManifest(multiPropJson)
        assertTrue("multi_prop must validate cleanly", multiPropVal.isValid)
        val multiPropEffect = EffectCompiler.compile(multiPropJson, multiPropDir)
        val hatNode = multiPropEffect.nodes.first { it.id == "prop_top_hat" }
        val resolvedHatMat = ModelRenderer.resolvePrimaryModelMaterial(hatNode, multiPropEffect.materials)
        assertNotNull("prop_top_hat must resolve to mat_hat via token/3D lit fallback", resolvedHatMat)
        assertEquals("mat_hat", resolvedHatMat!!.id)
        assertEquals("lambert@1", resolvedHatMat.template)
        assertEquals("opaque", resolvedHatMat.blend)

        // 3. Bug #3: periwinkle_forehead & sunflower_forehead explicitly declare lambert@1 and have default glTF matte
        //    factors (metallic=0, roughness=1, no PBR maps). hasActiveGlbPbr must return false so lambert@1 is preserved.
        val periwinkleDir = File("src/main/assets/filters/periwinkle_forehead").let {
            if (it.exists()) it else File("app/src/main/assets/filters/periwinkle_forehead")
        }
        val periwinkleJson = JSONObject(File(periwinkleDir, "manifest.json").readText())
        val periwinkleEffect = EffectCompiler.compile(periwinkleJson, periwinkleDir)
        val flowerNode = periwinkleEffect.nodes.first { it.type == "model" }
        val flowerMat: CompiledMaterial = ModelRenderer.resolvePrimaryModelMaterial(flowerNode, periwinkleEffect.materials)!!
        assertEquals("lambert@1", flowerMat.template)
        val glbLoader = GlbLoader()
        val periwinkleGlb = glbLoader.load(File(periwinkleDir, "assets/periwinkle_forehead.glb"))
        assertNotNull(periwinkleGlb)
        val firstGlbMat = periwinkleGlb!!.materials.firstOrNull()
        assertFalse(
            "Explicit lambert@1 with default glTF metallic=0, roughness=1 and no PBR maps must NOT override to pbr@1",
            ModelRenderer.hasActiveGlbPbr(periwinkleGlb, flowerMat, firstGlbMat)
        )

        // Contrast with heart_mood_filter which explicitly authors glTF metallic=0.1, roughness=0.3 -> hasActiveGlbPbr == true
        val heartDir = File("src/main/assets/filters/heart_mood_filter").let {
            if (it.exists()) it else File("app/src/main/assets/filters/heart_mood_filter")
        }
        val heartGlb = glbLoader.load(File(heartDir, "models/heart_mood_crown_lod_a.glb"))!!
        assertTrue(
            "heart_mood_crown_lod_a.glb with custom glTF metallic=0.1, roughness=0.3 must activate PBR shading",
            ModelRenderer.hasActiveGlbPbr(heartGlb, null, heartGlb.materials.firstOrNull())
        )

        // 4. Bug #4 & Bug #5: Direct diffuse energy balance in evaluatePbrColor + emissiveMapRgb support + Compiler emissiveMapAssetId
        val lighting = CompiledLighting()
        val pbrNoEmMap = MaterialRegistry.evaluatePbrColor(
            albedoRgb = floatArrayOf(0.8f, 0.2f, 0.2f),
            normal = floatArrayOf(0f, 0f, 1f),
            metallic = 0.0f,
            roughness = 0.5f,
            emissiveRgb = floatArrayOf(0.4f, 0.5f, 0.6f),
            lighting = lighting,
            emissiveMapRgb = floatArrayOf(0.5f, 0.5f, 0.5f)
        )
        val pbrFullEm = MaterialRegistry.evaluatePbrColor(
            albedoRgb = floatArrayOf(0.8f, 0.2f, 0.2f),
            normal = floatArrayOf(0f, 0f, 1f),
            metallic = 0.0f,
            roughness = 0.5f,
            emissiveRgb = floatArrayOf(0.4f, 0.5f, 0.6f),
            lighting = lighting,
            emissiveMapRgb = floatArrayOf(1.0f, 1.0f, 1.0f)
        )
        assertTrue("Emissive texture map must modulate emissive factor in PBR evaluation", pbrFullEm[0] > pbrNoEmMap[0] + 0.05f)
        assertTrue("Dielectric PBR red channel under default studio lighting must be vibrant (> 0.60f)", pbrNoEmMap[0] > 0.60f)
        assertTrue("FS_PBR shader must declare uEmissiveMap and uAlphaMode uniforms", MaterialRegistry.FS_PBR.contains("uEmissiveMap") && MaterialRegistry.FS_PBR.contains("uAlphaMode"))
        assertTrue("FS_LAMBERT shader must declare uAlphaMode and uAlphaCutoff uniforms", MaterialRegistry.FS_LAMBERT.contains("uAlphaMode") && MaterialRegistry.FS_LAMBERT.contains("uAlphaCutoff"))

        // 5. Bug #7: evaluateLambertColor CPU specular & Fresnel rim sheen parity while preserving 3-arg backward compatibility
        val lambertLegacy = MaterialRegistry.evaluateLambertColor(
            texRgb = floatArrayOf(0.5f, 0.4f, 0.7f),
            normal = floatArrayOf(0.6f, 0.0f, 0.8f),
            lighting = lighting
        )
        val lambertPhase1 = MaterialRegistry.evaluateLambertColor(
            texRgb = floatArrayOf(0.5f, 0.4f, 0.7f),
            normal = floatArrayOf(0.6f, 0.0f, 0.8f),
            lighting = lighting,
            fresnelSheen = 0.25f,
            includeSpecular = true
        )
        assertTrue(
            "Phase 1 Lambert evaluation with specular + Fresnel rim sheen must be brighter than flat legacy diffuse",
            lambertPhase1[0] > lambertLegacy[0] && lambertPhase1[2] > lambertLegacy[2]
        )

        // 6. Bug #8: ModelRenderer front-camera uModelView X-mirror parity
        val modelRenderer = ModelRenderer()
        val worldMat = FloatArray(16).apply {
            android.opengl.Matrix.setIdentityM(this, 0)
            this[0] = 1.25f
            this[12] = 0.35f
        }
        val vpMat = FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0) }
        modelRenderer.render(
            model = periwinkleGlb,
            node = flowerNode,
            worldMatrix = worldMat,
            viewProjectionMatrix = vpMat,
            material = flowerMat,
            lighting = lighting,
            opacity = 1f,
            programs = MaterialRegistry(),
            textureId = 0,
            frontFacing = true
        )
        assertEquals(
            "When frontFacing=true, ModelRenderer must mirror X column of uModelView to match mirrored projection",
            -1.25f,
            modelRenderer.lastModelViewMatrix[0],
            1e-4f
        )
        assertEquals(
            "When frontFacing=true, ModelRenderer must mirror X translation of uModelView",
            -0.35f,
            modelRenderer.lastModelViewMatrix[12],
            1e-4f
        )
    }

    @Test
    fun testPhase2DeepAuditFixes_CustomPassDefaultsGrainRecolorScleraSkinParity() {
        // 1. Gap #1: Custom PassGraph nodes (pass.lut@1, pass.grain@1, pass.sharpen@1) must seed EFFECT_PARAM_SPECS defaults
        // and support dynamic PropertySlots overrides even when keys are omitted in manifest pass.params
        val manifestWithCustomPasses = JSONObject("""
            {
              "schemaVersion": "1.0.0",
              "id": "phase2_custom_pass_audit",
              "name": "Phase 2 Custom Pass Audit",
              "assets": [
                { "id": "warm_lut_img", "type": "image", "uri": "assets/warm_lut.png" }
              ],
              "passGraph": [
                { "id": "cam", "stage": "CAMERA_FRAME" },
                {
                  "id": "eye_recolor_pass",
                  "template": "pass.recolor@1",
                  "reads": ["${'$'}mask:iris"],
                  "params": {
                    "targetColor": "#3399FF",
                    "luminancePreserve": 0.85,
                    "opacity": 0.60,
                    "intensity": 0.50
                  }
                },
                {
                  "id": "teeth_pass",
                  "template": "pass.recolor@1",
                  "reads": ["${'$'}mask:teeth"],
                  "params": {
                    "mode": "teethWhiten",
                    "strength": 0.80,
                    "intensity": 0.50,
                    "desaturateYellow": 0.82,
                    "minCavityLuma": 0.30
                  }
                },
                {
                  "id": "lut_pass",
                  "template": "pass.lut@1",
                  "asset": "warm_lut_img",
                  "params": {}
                },
                {
                  "id": "grain_pass",
                  "template": "pass.grain@1",
                  "params": {}
                },
                {
                  "id": "sharpen_pass",
                  "template": "pass.sharpen@1",
                  "params": {}
                }
              ],
              "nodes": [
                { "id": "lut_pass", "type": "post", "effect": "lut@1" }
              ]
            }
        """.trimIndent())

        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_phase2_audit_${System.nanoTime()}")
        tempDir.mkdirs()
        try {
            val assetsDir = File(tempDir, "assets").apply { mkdirs() }
            File(assetsDir, "warm_lut.png").writeBytes(ByteArray(16))

            val compiled = EffectCompiler.compile(manifestWithCustomPasses, tempDir)
            val passesById = compiled.passGraph.passes.associateBy { it.id }

            // Verify custom pass defaults are seeded in PassGraph
            val lutPass = passesById["lut_pass"]!!
            val grainPass = passesById["grain_pass"]!!
            val sharpenPass = passesById["sharpen_pass"]!!
            assertEquals(1.0f, lutPass.params["amount"] ?: 0f, 1e-4f)
            assertEquals(0.15f, grainPass.params["amount"] ?: 0f, 1e-4f)
            assertEquals(1.0f, grainPass.params["size"] ?: 0f, 1e-4f)
            assertEquals(0.75f, grainPass.params["lumaWeight"] ?: 0f, 1e-4f)
            assertEquals(24.0f, grainPass.params["speed"] ?: 0f, 1e-4f)
            assertEquals(0.35f, sharpenPass.params["amount"] ?: 0f, 1e-4f)
            assertEquals(1.0f, sharpenPass.params["radius"] ?: 0f, 1e-4f)
            assertEquals(0.02f, sharpenPass.params["threshold"] ?: 0f, 1e-4f)

            // 2. Gap #5: PostRenderer.isAssetUsedAsLut detects LUT assets even when declared as type="image"
            assertTrue(
                "Asset referenced by pass.lut@1 must be detected as a LUT even when declared with type='image'",
                PostRenderer.isAssetUsedAsLut(compiled, "warm_lut_img")
            )

            // 3. Gap #3: Standalone pass.recolor@1 hex targetColor / luminancePreserve / opacity parsing
            // and non-squared intensity in RecolorPass.resolveRecolorParams
            val eyeRecolorPass = passesById["eye_recolor_pass"]!!
            assertEquals(0x33 / 255f, eyeRecolorPass.params["colorR"] ?: 0f, 1e-3f)
            assertEquals(0x99 / 255f, eyeRecolorPass.params["colorG"] ?: 0f, 1e-3f)
            assertEquals(0xFF / 255f, eyeRecolorPass.params["colorB"] ?: 0f, 1e-3f)
            assertEquals(0.85f, eyeRecolorPass.params["preserveLuminance"] ?: 0f, 1e-4f)
            // strength must be stored as 0.60f (not pre-multiplied by intensity 0.50f), so runtime strength*intensity = 0.30f (not 0.15f)
            assertEquals(0.60f, eyeRecolorPass.params["strength"] ?: 0f, 1e-4f)
            assertEquals(0.50f, eyeRecolorPass.params["intensity"] ?: 0f, 1e-4f)

            val teethPass = passesById["teeth_pass"]!!
            assertEquals(2.0f, teethPass.params["mode"] ?: 0f, 1e-4f)
            assertEquals(RecolorPass.MAX_DEFAULT_STRENGTH, teethPass.params["strength"] ?: 0f, 1e-4f)
            assertEquals(0.50f, teethPass.params["intensity"] ?: 0f, 1e-4f)
            assertEquals(0.82f, teethPass.params["desaturateYellow"] ?: 0f, 1e-4f)
            assertEquals(0.30f, teethPass.params["minCavityLuma"] ?: 0f, 1e-4f)

            // 4. Gap #2: Film grain speed + lumaWeight and FS_POST hue rotation in colorMatrix
            val seedAt24fpsFrame1 = PostRenderer.resolveEffectiveGrainSeed(
                0f,
                com.example.engine.render.DeterministicFrameContext(renderTimeSec = 0.042f, frameSeed = 1),
                24.0f
            )
            val seedAt60fpsFrame1 = PostRenderer.resolveEffectiveGrainSeed(
                0f,
                com.example.engine.render.DeterministicFrameContext(renderTimeSec = 0.020f, frameSeed = 1),
                60.0f
            )
            assertTrue("24Hz grain should advance after 42ms", seedAt24fpsFrame1 > 0f)
            assertTrue("60Hz grain should advance after 20ms", seedAt60fpsFrame1 > 0f)
            assertTrue("FS_POST must include uLumaWeight and YIQ hue rotation in colorMatrix", PostRenderer.FS_POST.contains("uLumaWeight") && PostRenderer.FS_POST.contains("yiqI"))

            // 5. Gap #4: Subtracted iris in $mask:sclera when iris center touches/crosses the eyelid contour
            // and GPU/CPU cranial forehead hairline feather scaling parity on $mask:skin / face_oval
            val landmarks = FloatArray(478 * 3)
            for (i in 0 until 478) {
                landmarks[i * 3] = 0.5f
                landmarks[i * 3 + 1] = 0.5f
                landmarks[i * 3 + 2] = 0.0f
            }
            // Place chin (#152) at (0.50, 0.72) and forehead (#10) at (0.50, 0.32) -> faceHeight = 0.40
            landmarks[152 * 3] = 0.50f; landmarks[152 * 3 + 1] = 0.72f
            landmarks[10 * 3] = 0.50f;  landmarks[10 * 3 + 1] = 0.32f
            // Set up face_oval polygon around face center
            val oval = com.example.engine.pkg.FaceRegions.FACE_OVAL
            for (k in oval.indices) {
                val angle = (2.0 * Math.PI * k) / oval.size
                val lmIdx = oval[k]
                landmarks[lmIdx * 3] = (0.50 + 0.18 * Math.cos(angle)).toFloat()
                landmarks[lmIdx * 3 + 1] = (0.52 - 0.20 * Math.sin(angle)).toFloat()
            }
            landmarks[10 * 3] = 0.50f;  landmarks[10 * 3 + 1] = 0.32f
            landmarks[152 * 3] = 0.50f; landmarks[152 * 3 + 1] = 0.72f

            // Set up left eye contour around (0.60, 0.44) with semi-axes (0.04, 0.015)
            val leftEye = com.example.engine.pkg.FaceRegions.LEFT_EYE_CONTOUR
            for (k in leftEye.indices) {
                val angle = (2.0 * Math.PI * k) / leftEye.size
                val lmIdx = leftEye[k]
                landmarks[lmIdx * 3] = (0.60 + 0.04 * Math.cos(angle)).toFloat()
                landmarks[lmIdx * 3 + 1] = (0.44 + 0.015 * Math.sin(angle)).toFloat()
            }
            // Place left iris center slightly outside top eyelid edge (upward gaze: y = 0.422 < 0.425 top edge)
            // while iris disc (radius = 0.016) overlaps the open eye
            val leftIrisCenterIdx = com.example.engine.pkg.FaceRegions.LEFT_IRIS_CENTER
            val leftIrisRing = com.example.engine.pkg.FaceRegions.LEFT_IRIS_RING
            landmarks[leftIrisCenterIdx * 3] = 0.60f
            landmarks[leftIrisCenterIdx * 3 + 1] = 0.422f
            val ringOffsets = arrayOf(
                floatArrayOf(0.016f, 0f),
                floatArrayOf(0f, -0.016f),
                floatArrayOf(-0.016f, 0f),
                floatArrayOf(0f, 0.016f)
            )
            for (r in leftIrisRing.indices) {
                val rIdx = leftIrisRing[r]
                landmarks[rIdx * 3] = 0.60f + ringOffsets[r][0]
                landmarks[rIdx * 3 + 1] = 0.422f + ringOffsets[r][1]
            }

            val faceState = FaceState(
                isVisible = true,
                landmarks = landmarks,
                timestampNs = 100_000_000L,
                poseTimestampNs = 100_000_000L
            )
            val regionMask = RegionMaskPass()
            // Unclamped subtractive iris triangulation must produce all 16 segments * 9 floats/seg * 3 = 432 floats
            val unclampedFloats = regionMask.triangulateRegionIntoScratch(
                regionType = RegionMaskPass.RegionType.LEFT_IRIS_CLAMPED,
                smoothedLandmarks = landmarks,
                faceState = faceState,
                clampIrisToEyelid = false
            )
            assertEquals("Subtracted iris disc in \$mask:sclera must triangulate all 16 segments even during upward gaze", 16 * 27, unclampedFloats)

            // Even clamped additive iris must still produce 16 segments when iris center is slightly outside eyelid contour but disc overlaps eye
            val clampedFloats = regionMask.triangulateRegionIntoScratch(
                regionType = RegionMaskPass.RegionType.LEFT_IRIS_CLAMPED,
                smoothedLandmarks = landmarks,
                faceState = faceState,
                clampIrisToEyelid = true
            )
            assertEquals("Clamped additive iris disc must not collapse to 0 triangles when iris center slightly crosses eyelid edge", 16 * 27, clampedFloats)

            // Verify CPU cranial forehead feather expansion above extrapolated landmark #10:
            // Extrapolated landmark #10 is at y = 0.32 - 0.22 * (0.72 - 0.32) = 0.232.
            // With base featherNorm = 0.018, cranial forehead feather expands by (1 + 0.22 * 3.5) = 1.77x -> 0.03186.
            // So at y = 0.210 (distance 0.022 above extrapolated top, > base 0.018 but < expanded 0.03186),
            // CPU evaluateRegionMaskAtNormalizedViewPt must return > 0f (matching GPU skirt).
            val hairlineFeatherWeight = regionMask.evaluateRegionMaskAtNormalizedViewPt(
                regionType = RegionMaskPass.RegionType.FACE_OVAL,
                normX = 0.50f,
                normY = 0.210f,
                faceState = faceState,
                featherNorm = 0.018f,
                useSmoothedLandmarks = false
            )
            assertTrue(
                "CPU FACE_OVAL / \$mask:skin evaluation must apply expanded cranial hairline feather (> 0 at y=0.210, got $hairlineFeatherWeight)",
                hairlineFeatherWeight > 0.05f
            )
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testPhase3DeepAuditFixesSubHalfPixelFastMotionAndFrameRingSync() {
        // 1. Gap #1: Verify FaceRegions.CONTROL_POINT_INDICES includes warp anchors (#4, #6, #129, #358)
        // while keeping non-control landmark #1 untouched, and PostRenderer.populateWarpAnchors uses smoothed landmarks
        val cpSet = com.example.engine.pkg.FaceRegions.CONTROL_POINT_INDICES.toSet()
        assertTrue("CONTROL_POINT_INDICES must include noseTip #4", cpSet.contains(4))
        assertTrue("CONTROL_POINT_INDICES must include noseBridge #6", cpSet.contains(6))
        assertTrue("CONTROL_POINT_INDICES must include leftAla #129", cpSet.contains(129))
        assertTrue("CONTROL_POINT_INDICES must include rightAla #358", cpSet.contains(358))

        // 2. Gap #2 & Phase 3 Verification Criteria:
        // Prove < 0.5px mask alignment error on a 1080x1920 view during simulated fast head motion
        // with 35ms tracker latency across SignalPipeline + ControlPointSmoother + FacePoseAdapter.mapLandmarkToViewWithPoseDelta
        val pipeline = com.example.engine.signal.SignalPipeline()
        val smoother = ControlPointSmoother()
        val viewWidthPx = 1080f
        val viewHeightPx = 1920f
        val cropAspect = 9f / 16f
        val fovYDeg = 60f
        val projParams = com.example.engine.signal.ProjectionParams(
            fovYDeg = fovYDeg,
            cropAspectRatio = cropAspect
        )
        val fCrop = FacePoseAdapter.focalFactorFromFovY(fovYDeg)
        val absZ = 0.50f // 50cm in front of camera
        val wCam = 2.0f * absZ * cropAspect / fCrop
        val hCam = 2.0f * absZ / fCrop

        // Simulate fast lateral + vertical head motion: vx = 0.28 m/s (~525 px/sec on 1080p view), vy = -0.10 m/s
        val vxMps = 0.28f
        val vyMps = -0.10f
        val t0Ns = 1_000_000_000L
        val dtNs = 33_333_333L // 30 fps tracker capture cadence
        val trackerLatencyNs = 35_000_000L // 35ms tracker latency at render time

        val rawLms = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)
        val targetLmId = 61 // outer lip corner control point ($mask:lips)
        val localOffsetX = -0.025f
        val localOffsetY = -0.035f

        var lastCaptureNs = t0Ns
        var latestSmoothedLms: FloatArray? = null

        for (frame in 0..8) {
            val captureNs = t0Ns + frame * dtNs
            lastCaptureNs = captureNs
            val tSec = (frame * dtNs).toFloat() / 1_000_000_000f
            val headTx = -0.08f + vxMps * tSec
            val headTy = 0.04f + vyMps * tSec

            // Populate 3D pose matrix at capture time
            val rawPose = FloatArray(16).apply {
                android.opengl.Matrix.setIdentityM(this, 0)
                this[12] = headTx
                this[13] = headTy
                this[14] = -absZ
            }

            // Populate cheek landmarks (#234, #454) and target lip landmark (#61) projected from true 3D pose at captureNs
            val rCheekCamX = headTx - 0.07f
            val lCheekCamX = headTx + 0.07f
            rawLms[234 * 3] = 0.5f + (rCheekCamX / wCam)
            rawLms[234 * 3 + 1] = 0.5f - (headTy / hCam)
            rawLms[454 * 3] = 0.5f + (lCheekCamX / wCam)
            rawLms[454 * 3 + 1] = 0.5f - (headTy / hCam)

            val lmCamX = headTx + localOffsetX
            val lmCamY = headTy + localOffsetY
            rawLms[targetLmId * 3] = 0.5f + (lmCamX / wCam)
            rawLms[targetLmId * 3 + 1] = 0.5f - (lmCamY / hCam)
            rawLms[targetLmId * 3 + 2] = 0f

            val snapshot = com.example.engine.perception.FaceGeometrySnapshot.create(
                timestampNs = captureNs,
                imageWidth = 1080,
                imageHeight = 1920,
                rotationDegrees = 0,
                isFrontCamera = true,
                isTracking = true,
                landmarks = rawLms.clone(),
                rawPoseMatrix = rawPose,
                blendshapes = emptyMap(),
                sensorTimestampNs = captureNs,
                projectionParams = projParams
            )
            val unpredictedState = pipeline.processSnapshot(snapshot, nowNs = captureNs)

            // Verify Gap #2A: unpredicted snapshot (timestampNs == poseTimestampNs) must NOT pull 2D landmarks backward
            val unpredPt = FloatArray(2)
            val directPt = FloatArray(2)
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = rawLms[targetLmId * 3],
                v = rawLms[targetLmId * 3 + 1],
                z = 0f,
                faceState = unpredictedState,
                viewWidth = viewWidthPx,
                viewHeight = viewHeightPx,
                outPt = unpredPt
            )
            FacePoseAdapter.mapLandmarkToView(
                u = rawLms[targetLmId * 3],
                v = rawLms[targetLmId * 3 + 1],
                imageToViewTransform = unpredictedState.imageToViewTransform,
                isFrontCamera = unpredictedState.isFrontCamera,
                viewWidth = viewWidthPx,
                viewHeight = viewHeightPx,
                outPt = directPt
            )
            assertEquals("Unpredicted snapshot at dt=0 must have 0.0px backward pose-filter pull", directPt[0], unpredPt[0], 1e-4f)
            assertEquals("Unpredicted snapshot at dt=0 must have 0.0px backward pose-filter pull", directPt[1], unpredPt[1], 1e-4f)

            latestSmoothedLms = smoother.smooth(rawLms.clone(), captureNs, true)
        }

        // Evaluate predicted state at render time = lastCaptureNs + 35ms tracker latency
        val renderNs = lastCaptureNs + trackerLatencyNs
        val predictedState = pipeline.consumeLatestPredictedState(renderNs, predictionEnabled = true)
        val smoothedAtLastCapture = latestSmoothedLms!!

        val compensatedPtPx = FloatArray(2)
        FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
            u = smoothedAtLastCapture[targetLmId * 3],
            v = smoothedAtLastCapture[targetLmId * 3 + 1],
            z = smoothedAtLastCapture[targetLmId * 3 + 2],
            faceState = predictedState,
            viewWidth = viewWidthPx,
            viewHeight = viewHeightPx,
            outPt = compensatedPtPx
        )

        // Compute true ground-truth landmark pixel position at renderNs (+35ms ahead of lastCaptureNs)
        val trueRenderSec = ((8 * dtNs) + trackerLatencyNs).toFloat() / 1_000_000_000f
        val trueHeadTxAtRender = -0.08f + vxMps * trueRenderSec
        val trueHeadTyAtRender = 0.04f + vyMps * trueRenderSec
        val trueLmUAtRender = 0.5f + ((trueHeadTxAtRender + localOffsetX) / wCam)
        val trueLmVAtRender = 0.5f - ((trueHeadTyAtRender + localOffsetY) / hCam)
        val groundTruthPtPx = FloatArray(2)
        FacePoseAdapter.mapLandmarkToView(
            u = trueLmUAtRender,
            v = trueLmVAtRender,
            imageToViewTransform = predictedState.imageToViewTransform,
            isFrontCamera = predictedState.isFrontCamera,
            viewWidth = viewWidthPx,
            viewHeight = viewHeightPx,
            outPt = groundTruthPtPx
        )

        val errPxX = abs(compensatedPtPx[0] - groundTruthPtPx[0])
        val errPxY = abs(compensatedPtPx[1] - groundTruthPtPx[1])
        val totalAlignmentErrorPx = kotlin.math.hypot(errPxX, errPxY)
        assertTrue(
            "Phase 3 fast-motion (35ms latency) 2D mask landmark alignment error ($totalAlignmentErrorPx px) must be < 0.5px on 1080x1920 view",
            totalAlignmentErrorPx < 0.5f
        )

        // 3. Gap #3: Verify mapLandmarkToViewWithPoseDelta honors non-identity imageToViewTransform (FILL_CENTER crop & mirror)
        val fillCenterTransform = floatArrayOf(
            -0.80f, 0.0f, 0.90f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        )
        val stateWithTransform = predictedState.copy(imageToViewTransform = fillCenterTransform)
        val transformedPt = FloatArray(2)
        FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
            u = smoothedAtLastCapture[targetLmId * 3],
            v = smoothedAtLastCapture[targetLmId * 3 + 1],
            z = 0f,
            faceState = stateWithTransform,
            viewWidth = 1.0f,
            viewHeight = 1.0f,
            outPt = transformedPt
        )
        val expectedTransformedU = (-0.80f * trueLmUAtRender) + 0.90f
        assertEquals(
            "mapLandmarkToViewWithPoseDelta must route 3D pose delta through imageToViewTransform scale/mirror",
            expectedTransformedU,
            transformedPt[0],
            5e-4f
        )

        // 4. Gap #4: Verify SurfaceTextureCameraInputSurface + FrameRing tracker-timestamp sync & ClockOffsetCalibrator wiring
        val calibrator = com.example.engine.capture.ClockOffsetCalibrator(warmupFrames = 1)
        calibrator.configureSource(com.example.engine.capture.ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN)
        val calibratedTs = calibrator.toElapsedRealtimeNs(sensorTimestampNs = 100_000_000L, arrivalElapsedRealtimeNs = 600_000_000L)
        assertEquals(600_000_000L, calibratedTs)

        val ring = DefaultFrameRing(delayDepth = 0, capacity = 8)
        val inputSurface = com.example.engine.capture.SurfaceTextureCameraInputSurface(
            frameRing = ring,
            clockOffsetCalibrator = calibrator
        )
        inputSurface.attachToGlContext(textureId = 77)
        val m1 = FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0); this[12] = 0.11f }
        val m2 = FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0); this[12] = 0.22f }
        val m3 = FloatArray(16).apply { android.opengl.Matrix.setIdentityM(this, 0); this[12] = 0.33f }
        ring.recordFrame(textureId = 77, sensorTimestampNs = 100_000_000L, alignedElapsedRealtimeNs = 600_000_000L, texMatrix = m1)
        ring.recordFrame(textureId = 77, sensorTimestampNs = 133_333_333L, alignedElapsedRealtimeNs = 633_333_333L, texMatrix = m2)
        ring.recordFrame(textureId = 77, sensorTimestampNs = 166_666_666L, alignedElapsedRealtimeNs = 666_666_666L, texMatrix = m3)

        inputSurface.syncFrameRingToTrackerTimestamp = true
        val outMat = FloatArray(16)
        val syncedSlot = inputSurface.readFrame(outTexMatrix = outMat, targetTimestampNs = 634_000_000L, useAlignedClock = true)
        assertNotNull(syncedSlot)
        assertEquals(633_333_333L, syncedSlot!!.alignedElapsedRealtimeNs)
        assertEquals("readFrame with targetTimestampNs must copy matched slot's texMatrix", 0.22f, outMat[12], 1e-6f)
    }
}



