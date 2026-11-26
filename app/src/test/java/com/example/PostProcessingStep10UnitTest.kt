package com.example

import android.graphics.Bitmap
import android.graphics.Color
import android.opengl.Matrix
import com.example.engine.capture.AnalysisFrame
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.photo.StillPipeline
import com.example.engine.pkg.CapabilityRegistry
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.PackageValidator
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.CompositionMode
import com.example.engine.render.NodeRendererRegistry
import com.example.engine.render.OffsetTestHarness
import com.example.engine.render.PostRenderer
import com.example.engine.render.RenderGraph
import com.example.engine.render.SingleSceneCompositionSurface
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs

/**
 * Step 10 Verification Suite:
 * Xelvic Post-Processing and Beauty Engine (README Sections 5.9, 7, 11.1, 11.2, 11.5, 19, 21 Step 10).
 *
 * Covers:
 * 1. Post-node schema and typed parameter validation (`colorGrade`, `blur`, `skinSoften`, `$param` resolution)
 * 2. Allow-listed effect registration and rejection of unknown effects, custom shaders/scripts, or invalid params
 * 3. RenderGraph ordering with post nodes after transparent nodes (exact 7-pass order from Section 11.1)
 * 4. PostRenderer execution on composed camera + virtual effect output
 * 5. Single-scene camera-texture composition when post effects are active (`SINGLE_SCENE_GL`)
 * 6. `ref_beauty` package validation, compilation, RenderGraph execution, tier degradation, and still-photo parity
 * 7. Existing filter and renderer regression across all 6 reference packages
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class PostProcessingStep10UnitTest {

    private fun resolveFilterDir(packageId: String): File {
        val candidates = listOf(
            File("app/src/main/assets/filters/$packageId"),
            File("src/main/assets/filters/$packageId"),
            File("../app/src/main/assets/filters/$packageId")
        )
        return candidates.firstOrNull { it.exists() && File(it, "manifest.json").exists() }
            ?: error("Could not locate filter package directory for $packageId")
    }

    private fun compileReferenceFilter(packageId: String): CompiledEffect {
        val dir = resolveFilterDir(packageId)
        val manifestJson = JSONObject(File(dir, "manifest.json").readText())
        val validation = PackageValidator.validateManifest(manifestJson)
        assertTrue("Package $packageId must pass validation: ${validation.errorReason}", validation.isValid)
        return Compiler.compile(manifestJson, dir)
    }

    private fun buildCanonicalSnapshot(
        tx: Float = 0.0f,
        ty: Float = 0.0f,
        tz: Float = -0.45f,
        imageWidth: Int = 576,
        imageHeight: Int = 1024,
        isFrontCamera: Boolean = true,
        blendshapes: Map<String, Float> = mapOf("mouthOpen" to 0.25f, "smile" to 0.5f)
    ): FaceGeometrySnapshot {
        val pose = FloatArray(16)
        Matrix.setIdentityM(pose, 0)
        pose[12] = tx
        pose[13] = ty
        pose[14] = tz

        val aspect = imageWidth.toFloat() / imageHeight.toFloat().coerceAtLeast(1f)
        val halfH = abs(tz) * 0.57735026f
        val halfW = halfH * aspect
        val fw = 0.14f

        val uvs = UvRegistry.CANONICAL_UVS
        val landmarks = FloatArray(UvRegistry.VERTEX_COUNT * 3)
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            val u = uvs[i * 2]
            val v = uvs[i * 2 + 1]
            val localX = (u - 0.5f) * fw
            val localY = (0.5f - v) * (fw * 1.25f)
            val camX = tx + localX
            val camY = ty + localY
            val normX = (0.5f + (camX / (2f * halfW))).coerceIn(0.05f, 0.95f)
            val normY = (0.5f - (camY / (2f * halfH))).coerceIn(0.05f, 0.95f)
            landmarks[i * 3] = normX
            landmarks[i * 3 + 1] = normY
            landmarks[i * 3 + 2] = 0f
        }

        val idTransform = FloatArray(16)
        Matrix.setIdentityM(idTransform, 0)

        return FaceGeometrySnapshot.create(
            timestampNs = 16_666_666L,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = 0,
            isFrontCamera = isFrontCamera,
            isTracking = true,
            landmarks = landmarks,
            rawPoseMatrix = pose,
            blendshapes = blendshapes,
            imageToViewTransform = idTransform
        )
    }

    private fun buildSyntheticFaceState(
        tx: Float = 0.0f,
        ty: Float = 0.0f,
        tz: Float = -0.45f,
        imageWidth: Int = 576,
        imageHeight: Int = 1024,
        isFrontCamera: Boolean = true
    ): FaceState {
        val snapshot = buildCanonicalSnapshot(tx, ty, tz, imageWidth, imageHeight, isFrontCamera)
        val pose = FloatArray(16)
        Matrix.setIdentityM(pose, 0)
        pose[12] = tx
        pose[13] = ty
        pose[14] = tz
        return FaceState.create(
            timestampNs = snapshot.timestampNs,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = pose,
            translation = floatArrayOf(tx, ty, tz),
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 0f,
            pitchDeg = 0f,
            rollDeg = 0f,
            faceWidth = 0.14f,
            landmarks = snapshot.landmarks,
            blendshapes = snapshot.blendshapes,
            acquisitionTimeNs = snapshot.timestampNs,
            imageToViewTransform = snapshot.imageToViewTransform,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            isFrontCamera = isFrontCamera
        )
    }

    // -------------------------------------------------------------------------
    // 1. Post-node schema & typed parameter validation + 2. Allow-list & Rejection
    // -------------------------------------------------------------------------

    @Test
    fun testAllowListedEffectsAndCapabilitiesRegistered() {
        assertEquals(
            setOf("colorGrade", "blur", "skinSoften"),
            PostRenderer.ALLOW_LISTED_EFFECTS
        )
        assertTrue(CapabilityRegistry.isFeatureSupported("node.post@1"))
        assertTrue(CapabilityRegistry.isFeatureSupported("post.colorGrade@1"))
        assertTrue(CapabilityRegistry.isFeatureSupported("post.blur@1"))
        assertTrue(CapabilityRegistry.isFeatureSupported("post.skinSoften@1"))
        assertFalse(CapabilityRegistry.isFeatureSupported("post.customShader@1"))

        assertTrue(NodeRendererRegistry.isNodeTypeSupported("post"))
        assertEquals("node.post@1", NodeRendererRegistry.getCapabilityForNodeType("post"))
    }

    @Test
    fun testPostNodeSchemaAndTypedParameterValidation() {
        val validManifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_valid_post",
              "name": "Valid Post Filter",
              "requires": {
                "runtime": 3,
                "features": ["node.post@1", "post.skinSoften@1", "post.colorGrade@1", "post.blur@1"]
              },
              "params": {
                "soften": { "type": "float", "default": 0.4, "min": 0.0, "max": 1.0 },
                "warm": { "type": "float", "default": 0.15, "min": -1.0, "max": 1.0 }
              },
              "nodes": [
                {
                  "id": "soften_pass",
                  "type": "post",
                  "effect": "skinSoften",
                  "params": {
                    "amount": "${'$'}soften",
                    "radius": 2.5,
                    "detailPreserve": 0.6
                  },
                  "renderOrder": 100
                },
                {
                  "id": "blur_pass",
                  "type": "post",
                  "effect": "blur",
                  "params": {
                    "radius": 2.0,
                    "intensity": 0.5
                  },
                  "renderOrder": 101
                },
                {
                  "id": "grade_pass",
                  "type": "post",
                  "effect": "colorGrade",
                  "params": {
                    "warmth": "${'$'}warm",
                    "contrast": 1.1,
                    "saturation": 1.05,
                    "brightness": 0.02,
                    "intensity": 1.0
                  },
                  "renderOrder": 102
                }
              ]
            }
            """.trimIndent()
        )

        val validation = PackageValidator.validateManifest(validManifestJson)
        assertTrue("Valid post manifest must pass validation: ${validation.errorReason}", validation.isValid)

        val compiled = Compiler.compile(validManifestJson, File("."))
        assertEquals(3, compiled.nodes.size)

        val softenNode = compiled.nodes.first { it.id == "soften_pass" }
        assertEquals("skinSoften", softenNode.postEffect)
        assertEquals(0.4f, softenNode.postParams["amount"] ?: 0f, 1e-4f)
        assertEquals(2.5f, softenNode.postParams["radius"] ?: 0f, 1e-4f)

        val gradeNode = compiled.nodes.first { it.id == "grade_pass" }
        assertEquals("colorGrade", gradeNode.postEffect)
        assertEquals(0.15f, gradeNode.postParams["warmth"] ?: 0f, 1e-4f)
        assertEquals(1.1f, gradeNode.postParams["contrast"] ?: 0f, 1e-4f)
    }

    @Test
    fun testRejectUnknownEffectsCustomShadersAndInvalidParams() {
        // 1. Unknown effect identifier
        val unknownEffectJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "bad_effect",
              "name": "Bad Effect",
              "nodes": [
                {
                  "id": "post_1",
                  "type": "post",
                  "effect": "vignetteKaleidoscope",
                  "params": { "amount": 0.5 }
                }
              ]
            }
            """.trimIndent()
        )
        val res1 = PackageValidator.validateManifest(unknownEffectJson)
        assertFalse(res1.isValid)
        assertTrue(res1.errorReason!!.contains("unknown or unsupported"))

        // 2. Missing effect identifier
        val missingEffectJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "missing_effect",
              "name": "Missing Effect",
              "nodes": [
                {
                  "id": "post_1",
                  "type": "post",
                  "params": { "amount": 0.5 }
                }
              ]
            }
            """.trimIndent()
        )
        val res2 = PackageValidator.validateManifest(missingEffectJson)
        assertFalse(res2.isValid)
        assertTrue(res2.errorReason!!.contains("missing required 'effect'"))

        // 3. Forbidden custom shader / executable code keys in params
        val customShaderJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "custom_shader_attempt",
              "name": "Custom Shader Attempt",
              "nodes": [
                {
                  "id": "post_1",
                  "type": "post",
                  "effect": "colorGrade",
                  "params": {
                    "warmth": 0.2,
                    "fragmentShader": "void main() { gl_FragColor = vec4(1.0); }"
                  }
                }
              ]
            }
            """.trimIndent()
        )
        val res3 = PackageValidator.validateManifest(customShaderJson)
        assertFalse(res3.isValid)
        assertTrue(res3.errorReason!!.contains("forbidden shader/script"))

        // 4. Out-of-range parameter value (skinSoften amount > 1.0)
        val outOfRangeJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "out_of_range_post",
              "name": "Out Of Range Post",
              "nodes": [
                {
                  "id": "post_1",
                  "type": "post",
                  "effect": "skinSoften",
                  "params": {
                    "amount": 2.5
                  }
                }
              ]
            }
            """.trimIndent()
        )
        val res4 = PackageValidator.validateManifest(outOfRangeJson)
        assertFalse(res4.isValid)
        assertTrue(res4.errorReason!!.contains("out of supported range"))

        // 5. Unsupported parameter name for effect
        val unknownParamJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "unknown_param_post",
              "name": "Unknown Param Post",
              "nodes": [
                {
                  "id": "post_1",
                  "type": "post",
                  "effect": "blur",
                  "params": {
                    "radius": 0.4,
                    "nonExistentKnob": 0.5
                  }
                }
              ]
            }
            """.trimIndent()
        )
        val res5 = PackageValidator.validateManifest(unknownParamJson)
        assertFalse(res5.isValid)
        assertTrue(res5.errorReason!!.contains("does not support parameter 'nonExistentKnob'"))

        // 6. Undeclared $param reference
        val undeclaredRefJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "undeclared_ref_post",
              "name": "Undeclared Ref Post",
              "nodes": [
                {
                  "id": "post_1",
                  "type": "post",
                  "effect": "skinSoften",
                  "params": {
                    "amount": "${'$'}missingKnob"
                  }
                }
              ]
            }
            """.trimIndent()
        )
        val res6 = PackageValidator.validateManifest(undeclaredRefJson)
        assertFalse(res6.isValid)
        assertTrue(res6.errorReason!!.contains("undeclared manifest param"))
    }

    // -------------------------------------------------------------------------
    // 3. RenderGraph ordering with post nodes after transparent nodes
    // -------------------------------------------------------------------------

    @Test
    fun testRenderGraphExactSevenPassOrderAndPostRenderOrderSorting() {
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_full_pass_order",
              "name": "Full Pass Order Test",
              "anchors": {
                "nose": { "type": "landmark", "index": 1 }
              },
              "materials": {
                "mat_opaque": { "template": "lambert@1", "blend": "opaque", "depthTest": true, "depthWrite": true },
                "mat_trans": { "template": "unlit@1", "blend": "alpha", "depthTest": false, "depthWrite": false }
              },
              "nodes": [
                {
                  "id": "post_late",
                  "type": "post",
                  "effect": "colorGrade",
                  "params": { "warmth": 0.25 },
                  "renderOrder": 200
                },
                {
                  "id": "trans_sprite",
                  "type": "sprite",
                  "anchor": "nose",
                  "material": "mat_trans",
                  "renderOrder": 10
                },
                {
                  "id": "face_occ",
                  "type": "occluder",
                  "shape": "ellipsoid",
                  "renderOrder": 0
                },
                {
                  "id": "post_early",
                  "type": "post",
                  "effect": "skinSoften",
                  "params": { "amount": 0.4, "radius": 2.0 },
                  "renderOrder": 100
                }
              ]
            }
            """.trimIndent()
        )

        val validation = PackageValidator.validateManifest(manifestJson)
        assertTrue("Manifest must be valid: ${validation.errorReason}", validation.isValid)
        val compiled = Compiler.compile(manifestJson, File("."))

        val faceState = buildSyntheticFaceState()
        val evaluator = SceneEvaluator(maxNodes = 64)
        compiled.signalBus.evaluate(faceState, 0.5f)
        compiled.behaviorRunner.step(0.016f, compiled.signalBus, faceState, compiled.propertySlots, 500L)
        evaluator.evaluate(compiled, faceState, compiled.propertySlots)

        val renderGraph = RenderGraph()
        renderGraph.render(
            effect = compiled,
            faceState = faceState,
            evaluator = evaluator,
            viewportWidth = 1080,
            viewportHeight = 1920
        )

        // Verify exact 9-pass order (Chunk B0):
        // 1. Camera frame -> 2. Camera-pixel passes -> 3. Clear depth -> 4. Occluders -> 5. Opaque nodes -> 6. Transparent nodes -> 7. Post-processing nodes -> 8. Overlay nodes -> 9. Final output
        assertEquals(
            listOf(
                RenderGraph.RenderPass.CAMERA_FRAME,
                RenderGraph.RenderPass.CAMERA_PIXEL_PASSES,
                RenderGraph.RenderPass.CLEAR_DEPTH,
                RenderGraph.RenderPass.OCCLUDERS,
                RenderGraph.RenderPass.OPAQUE_NODES,
                RenderGraph.RenderPass.TRANSPARENT_NODES,
                RenderGraph.RenderPass.POST_NODES,
                RenderGraph.RenderPass.OVERLAY_NODES,
                RenderGraph.RenderPass.FINAL_OUTPUT
            ),
            renderGraph.lastPassOrder
        )

        // Verify post nodes executed in ascending renderOrder (post_early=100 before post_late=200)
        assertEquals(2, renderGraph.lastRenderedPostCount)
        assertEquals(2, renderGraph.post.lastExecutedPasses.size)
        assertEquals("post_early", renderGraph.post.lastExecutedPasses[0].nodeId)
        assertEquals("skinSoften", renderGraph.post.lastExecutedPasses[0].effect)
        assertEquals("post_late", renderGraph.post.lastExecutedPasses[1].nodeId)
        assertEquals("colorGrade", renderGraph.post.lastExecutedPasses[1].effect)
    }

    // -------------------------------------------------------------------------
    // 4. PostRenderer execution on composed camera + virtual effect output
    // 5. Single-scene camera-texture composition when post effects are active
    // -------------------------------------------------------------------------

    @Test
    fun testSingleSceneCompositionAndPostExecutionOnComposedScene() {
        val refBeautyEffect = compileReferenceFilter("ref_beauty")

        // Section 11.5: Presence of a post node must automatically select SINGLE_SCENE_GL
        val mode = OffsetTestHarness.selectCompositionMode(
            effect = refBeautyEffect,
            requiresVideoRecording = false,
            offsetDecision = null
        )
        assertEquals(CompositionMode.SINGLE_SCENE_GL, mode)

        // Create a test camera frame with distinct skin-region detail and a virtual prop pixel
        val cameraFrame = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        for (y in 0 until 128) {
            for (x in 0 until 128) {
                // High-frequency skin micro-texture (delta = ±10, below edge-preservation threshold)
                val delta = if ((x + y) % 2 == 0) 10 else -10
                val r = (185 + delta).coerceIn(0, 255)
                val g = (145 + delta).coerceIn(0, 255)
                val b = (125 + delta).coerceIn(0, 255)
                cameraFrame.setPixel(x, y, Color.argb(255, r, g, b))
            }
        }
        // Virtual overlay patch in upper-left background to prove colorGrade applies to composed scene
        for (oy in 8..12) {
            for (ox in 8..12) {
                cameraFrame.setPixel(ox, oy, Color.argb(255, 128, 128, 128))
            }
        }
        val originalOverlayPixel = cameraFrame.getPixel(10, 10)
        val originalCenterDiff = abs(
            Color.red(cameraFrame.getPixel(64, 64)) - Color.red(cameraFrame.getPixel(65, 64))
        )

        val renderGraph = RenderGraph()
        val evaluator = SceneEvaluator(maxNodes = 64)
        val faceState = buildSyntheticFaceState(imageWidth = 128, imageHeight = 128)

        val singleSurface = SingleSceneCompositionSurface(renderGraph, evaluator)
        singleSurface.activeEffect = refBeautyEffect
        singleSurface.onSurfaceCreated(128, 128)
        singleSurface.bindCameraTexture(textureId = 7, isExternalOes = false)
        singleSurface.bindCameraFrameBitmap(cameraFrame, alreadyMirroredForFrontCamera = true)

        singleSurface.renderFrame(
            faceState = faceState,
            dtSec = 0.016f,
            renderTimeSec = 0.25f,
            nowMs = 250L
        )

        val composed = singleSurface.lastComposedSceneBitmap
        assertNotNull("SingleSceneCompositionSurface must produce composed scene output", composed)
        assertEquals(2, renderGraph.lastRenderedPostCount)

        // Verify background/virtual overlay pixel was warm-graded (red increased, blue decreased)
        val gradedOverlayPixel = composed!!.getPixel(10, 10)
        assertNotEquals(originalOverlayPixel, gradedOverlayPixel)
        assertTrue(
            "Warm colorGrade must increase red channel on composed scene",
            Color.red(gradedOverlayPixel) > Color.red(originalOverlayPixel)
        )
        assertTrue(
            "Warm colorGrade must decrease blue channel on composed scene",
            Color.blue(gradedOverlayPixel) < Color.blue(originalOverlayPixel)
        )

        // Verify face skin center checkerboard high-frequency contrast was softened by skinSoften + colorGrade
        val softenedDiff = abs(
            Color.red(composed.getPixel(64, 64)) - Color.red(composed.getPixel(65, 64))
        )
        assertTrue(
            "skinSoften must reduce high-frequency micro-contrast inside the skin region (was $originalCenterDiff, now $softenedDiff)",
            softenedDiff < originalCenterDiff
        )
        singleSurface.release()
    }

    // -------------------------------------------------------------------------
    // 6. ref_beauty package validation, compilation, RenderGraph execution & StillPipeline parity
    // -------------------------------------------------------------------------

    @Test
    fun testRefBeautyPackageValidationCompilationTierDegradationAndPhotoParity() {
        val dir = resolveFilterDir("ref_beauty")
        val manifestJson = JSONObject(File(dir, "manifest.json").readText())

        // 1. Validate schema, features, and asset integrity
        val validation = PackageValidator.validateManifest(manifestJson)
        assertTrue("ref_beauty must pass PackageValidator: ${validation.errorReason}", validation.isValid)

        // 2. Compile package without any engine special cases
        val compiled = Compiler.compile(manifestJson, dir)
        assertEquals("ref_beauty", compiled.packageId)
        assertEquals(2, compiled.nodes.size)
        assertTrue(compiled.nodes.all { it.type == "post" })

        val skinNode = compiled.nodes.first { it.id == "beauty_skin_soften" }
        val gradeNode = compiled.nodes.first { it.id == "beauty_warm_grade" }
        assertEquals("skinSoften", skinNode.postEffect)
        assertEquals(0.45f, skinNode.postParams["amount"] ?: 0f, 1e-4f)
        assertEquals(2.5f, skinNode.postParams["radius"] ?: 0f, 1e-4f)

        assertEquals("colorGrade", gradeNode.postEffect)
        assertEquals(0.18f, gradeNode.postParams["warmth"] ?: 0f, 1e-4f)

        // 3. Verify Tier C disables expensive skinSoften pass while keeping colorGrade active
        val faceState = buildSyntheticFaceState()
        val evaluator = SceneEvaluator(maxNodes = 64)
        val renderGraph = RenderGraph()

        compiled.activeTier = "C"
        val tierCEffective = TierPolicy.resolveEffectiveTier(compiled, PerformanceGovernor.Tier.C, 256)
        compiled.signalBus.evaluate(faceState, 0.5f)
        compiled.behaviorRunner.step(
            dt = 0.016f,
            signalBus = compiled.signalBus,
            faceState = faceState,
            slots = compiled.propertySlots,
            nowMs = 500L,
            disabledBehaviors = tierCEffective.disabledBehaviors,
            hiddenNodes = tierCEffective.hideNodes
        )
        evaluator.evaluate(compiled, faceState, compiled.propertySlots)
        renderGraph.render(compiled, faceState, evaluator, 1080, 1920)
        assertEquals(1, renderGraph.lastRenderedPostCount)
        assertEquals("beauty_warm_grade", renderGraph.post.lastExecutedPasses.single().nodeId)

        // 4. Verify Tier A executes both skinSoften and colorGrade in order
        compiled.activeTier = "A"
        val tierAEffective = TierPolicy.resolveEffectiveTier(compiled, PerformanceGovernor.Tier.A, 1024)
        compiled.signalBus.evaluate(faceState, 0.5f)
        compiled.behaviorRunner.step(
            dt = 0.016f,
            signalBus = compiled.signalBus,
            faceState = faceState,
            slots = compiled.propertySlots,
            nowMs = 500L,
            disabledBehaviors = tierAEffective.disabledBehaviors,
            hiddenNodes = tierAEffective.hideNodes
        )
        evaluator.evaluate(compiled, faceState, compiled.propertySlots)
        renderGraph.render(compiled, faceState, evaluator, 1080, 1920)
        assertEquals(2, renderGraph.lastRenderedPostCount)
        assertEquals(
            listOf("beauty_skin_soften", "beauty_warm_grade"),
            renderGraph.post.lastExecutedPasses.map { it.nodeId }
        )

        // 5. Verify StillPipeline photo parity for ref_beauty
        val fakeProvider = object : FaceGeometryProvider {
            override val modelSha256: String = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
            override fun start(config: ProviderConfig) {}
            override fun updateConfig(config: ProviderConfig) {}
            override fun submit(frame: AnalysisFrame) {}
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {}
            override fun detectStillImage(
                bitmap: Bitmap,
                rotationDegrees: Int,
                isFrontCamera: Boolean,
                timestampNs: Long
            ): FaceGeometrySnapshot {
                return buildCanonicalSnapshot(
                    tx = 0.0f,
                    ty = 0.0f,
                    tz = -0.45f,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    isFrontCamera = isFrontCamera
                )
            }
            override fun stop() {}
        }

        val stillPipeline = StillPipeline(faceProvider = fakeProvider)
        val rawPhoto = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        rawPhoto.eraseColor(Color.rgb(175, 138, 122))

        val photoResult1 = stillPipeline.processStillPhoto(
            rawCaptureBitmap = rawPhoto,
            effect = compiled,
            isFrontCamera = true,
            rotationDegrees = 0,
            targetAspectRatio = 1f
        )
        val photoResult2 = stillPipeline.processStillPhoto(
            rawCaptureBitmap = rawPhoto,
            effect = compiled,
            isFrontCamera = true,
            rotationDegrees = 0,
            targetAspectRatio = 1f
        )

        assertTrue(photoResult1.faceDetected)
        assertTrue(photoResult1.effectApplied)
        assertTrue(photoResult1.renderedNodeCount >= 2)
        // Verify deterministic output across repeated captures
        assertEquals(
            photoResult1.bitmap.getPixel(128, 128),
            photoResult2.bitmap.getPixel(128, 128)
        )
        assertNotEquals(
            "StillPipeline must apply ref_beauty post effects to captured photo",
            Color.rgb(175, 138, 122),
            photoResult1.bitmap.getPixel(128, 128)
        )
        stillPipeline.release()
    }

    // -------------------------------------------------------------------------
    // 7. Existing filter and renderer regression across all 6 reference packages
    // -------------------------------------------------------------------------

    @Test
    fun testAllSixReferencePackagesCompileAndExecuteWithoutRegression() {
        val packageIds = listOf(
            "ref_sticker_2d",
            "round_glasses",
            "multi_prop",
            "ref_face_paint",
            "sunflower_forehead",
            "ref_beauty"
        )

        val renderGraph = RenderGraph()
        val evaluator = SceneEvaluator(maxNodes = 64)
        val faceState = buildSyntheticFaceState()

        for (pkgId in packageIds) {
            val effect = compileReferenceFilter(pkgId)
            effect.signalBus.evaluate(faceState, 0.25f)
            effect.behaviorRunner.step(0.016f, effect.signalBus, faceState, effect.propertySlots, 250L)
            evaluator.evaluate(effect, faceState, effect.propertySlots)
            renderGraph.render(effect, faceState, evaluator, 1080, 1920)

            assertEquals(
                "All 9 render passes must execute in canonical order for $pkgId",
                9,
                renderGraph.lastPassOrder.size
            )
            if (pkgId == "ref_beauty") {
                assertEquals(2, renderGraph.lastRenderedPostCount)
            } else {
                assertEquals(0, renderGraph.lastRenderedPostCount)
            }
        }
    }
}
