package com.example

import com.example.engine.pkg.Compiler as EffectCompiler
import com.example.engine.render.DeterministicFrameContext
import com.example.engine.render.PassGraph
import com.example.engine.render.RenderGraph
import com.example.engine.render.RenderTargetRef
import com.example.engine.render.RtPool
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

/**
 * ChunkG0PassGraphRtPoolAndBindingUnitTest (Flow 2 — Chunk G0):
 *
 * Verifies the filter-agnostic engine foundations introduced in Chunk G0:
 * 1. Zero-regression gate on the canonical 9-stage PassGraph (`CAMERA_FRAME -> CAMERA_PIXEL_PASSES ->
 *    CLEAR_DEPTH -> OCCLUDERS -> OPAQUE_NODES -> TRANSPARENT_NODES -> POST_NODES -> OVERLAY_NODES -> FINAL_OUTPUT`).
 * 2. Custom data-driven PassGraph compilation & execution with named render targets
 *    (`$camera`, `$scene`, `$mask:<id>`, `$rt:<name>`, `$output`).
 * 3. `RtPool` handle reuse across frames, `FULL`/`HALF`/`QUARTER` scale sizing (`RGBA8` & `R8`),
 *    and `GlResourceOwner` tracking.
 * 4. Data-driven TierPolicy (`tiers[tier].disablePasses`, `passScaleOverrides`, `paramOverrides`).
 * 5. Universal SignalBus -> BehaviorRunner -> PropertySlots parameter binding for `PASS_UNIFORM`,
 *    `MATERIAL_PARAM`, and `MASK_WEIGHT` (live & photo paths).
 * 6. `DeterministicFrameContext` seed & noise parity between Live Preview and Still Photo.
 */
@RunWith(RobolectricTestRunner::class)
class ChunkG0PassGraphRtPoolAndBindingUnitTest {

    private fun createTrackingFaceState(
        smile: Float = 0.0f,
        mouthOpen: Float = 0.0f
    ): FaceState {
        val landmarks = FloatArray(478 * 3)
        for (i in 0 until 478) {
            landmarks[i * 3] = 0.5f + (i % 11 - 5) * 0.01f
            landmarks[i * 3 + 1] = 0.5f + (i % 7 - 3) * 0.01f
            landmarks[i * 3 + 2] = -0.02f
        }
        val pose = FloatArray(16)
        android.opengl.Matrix.setIdentityM(pose, 0)
        pose[14] = -0.55f
        return FaceState.EMPTY.copy(
            timestampNs = 100_000_000L,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = pose,
            faceWidth = 0.16f,
            landmarks = landmarks,
            blendshapes = mapOf(
                "mouthSmileLeft" to smile,
                "mouthSmileRight" to smile,
                "jawOpen" to mouthOpen
            ),
            imageWidth = 1080,
            imageHeight = 1920,
            isFrontCamera = true
        )
    }

    @Test
    fun canonical9StagePassGraph_preservesZeroRegressionPassOrder() {
        val manifest = JSONObject(
            """
            {
              "schemaVersion": "1.12.0",
              "id": "com.xelvic.test.g0.canonical",
              "name": "G0 Canonical 9-Stage Test",
              "assets": [
                { "id": "paint_tex", "type": "image", "path": "assets/paint.png" }
              ],
              "materials": {
                "paint_mat": { "template": "faceTexture@1", "blend": "alpha", "params": { "map": "paint_tex", "opacity": 0.9 } },
                "prop_mat": { "template": "unlit@1", "blend": "opaque", "params": { "opacity": 1.0 } },
                "lens_mat": { "template": "unlit@1", "blend": "alpha", "params": { "opacity": 0.6 } }
              },
              "nodes": [
                { "id": "face_paint", "type": "faceMesh", "material": "paint_mat", "renderOrder": 0 },
                { "id": "head_occ", "type": "occluder", "shape": "faceMesh", "pushBackFW": 0.015 },
                { "id": "frame_prop", "type": "sprite", "anchor": "eye_center", "material": "prop_mat", "renderOrder": 1 },
                { "id": "alpha_lens", "type": "sprite", "anchor": "left_eye", "material": "lens_mat", "renderOrder": 2 },
                { "id": "warm_grade", "type": "post", "effect": "colorGrade", "params": { "warmth": 0.2, "intensity": 0.8 }, "renderOrder": 10 },
                { "id": "frame_overlay", "type": "overlay", "renderOrder": 20 }
              ]
            }
            """.trimIndent()
        )

        val effect = EffectCompiler.compile(manifest, File("."))
        assertFalse(effect.passGraph.isCustomManifestGraph)
        assertEquals(9, effect.passGraph.passes.size)

        val evaluator = SceneEvaluator()
        val renderGraph = RenderGraph()
        val faceState = createTrackingFaceState()

        effect.signalBus.evaluate(faceState, 0.5f)
        effect.behaviorRunner.step(0.016f, effect.signalBus, faceState, effect.propertySlots, 500L)
        evaluator.evaluate(effect, faceState, effect.propertySlots)
        renderGraph.render(effect, faceState, evaluator, 1080, 1920)

        val expectedLegacyOrder = listOf(
            RenderGraph.RenderPass.CAMERA_FRAME,
            RenderGraph.RenderPass.CAMERA_PIXEL_PASSES,
            RenderGraph.RenderPass.CLEAR_DEPTH,
            RenderGraph.RenderPass.OCCLUDERS,
            RenderGraph.RenderPass.OPAQUE_NODES,
            RenderGraph.RenderPass.TRANSPARENT_NODES,
            RenderGraph.RenderPass.POST_NODES,
            RenderGraph.RenderPass.OVERLAY_NODES,
            RenderGraph.RenderPass.FINAL_OUTPUT
        )
        assertEquals(expectedLegacyOrder, renderGraph.lastPassOrder)

        val expectedPassIds = listOf(
            PassGraph.PASS_ID_CAMERA_FRAME,
            PassGraph.PASS_ID_CAMERA_PIXEL,
            PassGraph.PASS_ID_CLEAR_DEPTH,
            PassGraph.PASS_ID_OCCLUDERS,
            PassGraph.PASS_ID_OPAQUE,
            PassGraph.PASS_ID_TRANSPARENT,
            PassGraph.PASS_ID_POST,
            PassGraph.PASS_ID_OVERLAY,
            PassGraph.PASS_ID_FINAL_OUTPUT
        )
        assertEquals(expectedPassIds, renderGraph.lastExecutedPassNodeIds)
        assertEquals(9, renderGraph.lastExecutedPassRecords.size)
    }

    @Test
    fun rtPool_reusesTemporaryHandlesAcrossFrames_andTracksScalesAndFormats() {
        val renderGraph = RenderGraph()
        val pool = renderGraph.rtPool
        renderGraph.resourceOwner.bindGlThread()

        // 1. Named half-res R8 mask target ($mask:skin -> 540x960, 1 byte/px)
        val maskRt = pool.acquireNamed(
            name = "\$mask:skin",
            baseWidth = 1080,
            baseHeight = 1920,
            format = RtPool.RtFormat.R8,
            scaleFactor = 0.5f,
            withDepth = false,
            resourceOwner = renderGraph.resourceOwner
        )
        assertEquals(540, maskRt.width)
        assertEquals(960, maskRt.height)
        assertEquals(540L * 960L * 1L, maskRt.byteSize)

        // 2. Named quarter-res RGBA8 target ($rt:bloomQuarter -> 270x480, 4 bytes/px)
        val quarterRt = pool.acquireNamed(
            name = "\$rt:bloomQuarter",
            baseWidth = 1080,
            baseHeight = 1920,
            format = RtPool.RtFormat.RGBA8,
            scaleFactor = 0.25f,
            withDepth = false,
            resourceOwner = renderGraph.resourceOwner
        )
        assertEquals(270, quarterRt.width)
        assertEquals(480, quarterRt.height)
        assertEquals(270L * 480L * 4L, quarterRt.byteSize)

        // 3. Temporary half-res RGBA8 target leased in Frame 1, released via endFrame(), and re-leased in Frame 2
        val tempFrame1 = pool.acquireTemp(
            baseWidth = 1080,
            baseHeight = 1920,
            format = RtPool.RtFormat.RGBA8,
            scaleFactor = 0.5f,
            debugLabel = "pingPongA",
            resourceOwner = renderGraph.resourceOwner
        )
        val allocCountAfterFrame1 = pool.allocationCount
        assertEquals(3, allocCountAfterFrame1)

        pool.endFrame()

        // Frame 2: acquire both named and temp targets again -> zero new allocations!
        val maskRtFrame2 = pool.acquireNamed(
            name = "\$mask:skin",
            baseWidth = 1080,
            baseHeight = 1920,
            format = RtPool.RtFormat.R8,
            scaleFactor = 0.5f,
            withDepth = false,
            resourceOwner = renderGraph.resourceOwner
        )
        val tempFrame2 = pool.acquireTemp(
            baseWidth = 1080,
            baseHeight = 1920,
            format = RtPool.RtFormat.RGBA8,
            scaleFactor = 0.5f,
            debugLabel = "pingPongA",
            resourceOwner = renderGraph.resourceOwner
        )

        assertEquals(allocCountAfterFrame1, pool.allocationCount)
        assertEquals(maskRt.fboId, maskRtFrame2.fboId)
        assertEquals(maskRt.texId, maskRtFrame2.texId)
        assertEquals(tempFrame1.fboId, tempFrame2.fboId)
        assertEquals(tempFrame1.texId, tempFrame2.texId)
        assertTrue(pool.cacheHitCount >= 2)

        // Clean release removes tracked handles from GlResourceOwner
        pool.releaseAll(renderGraph.resourceOwner)
        assertEquals(0, pool.totalAllocatedTargets)
    }

    @Test
    fun customDataDrivenPassGraph_andTierDisablePassesAndScaleOverrides_workDeclaratively() {
        val manifest = JSONObject(
            """
            {
              "schemaVersion": "1.12.0",
              "id": "com.xelvic.test.g0.custom_graph",
              "name": "G0 Custom PassGraph & Tier Policy Test",
              "passes": [
                { "id": "camera_in", "stage": "camera_frame", "reads": ["${'$'}camera"], "writes": "${'$'}scene" },
                { "id": "skin_mask_gen", "stage": "custom", "template": "mask.skin@1", "reads": ["${'$'}camera"], "writes": "${'$'}mask:skin", "scale": 0.5 },
                { "id": "bilateral_beauty", "stage": "custom", "template": "beauty.bilateral@1", "reads": ["${'$'}scene", "${'$'}mask:skin"], "writes": "${'$'}rt:softSkin", "scale": 0.5, "params": { "radius": 4.0, "sigmaColor": 0.15 } },
                { "id": "film_grain", "stage": "custom", "template": "post.grain@1", "reads": ["${'$'}rt:softSkin"], "writes": "${'$'}scene", "scale": 1.0, "params": { "strength": 0.25 } },
                { "id": "present_out", "stage": "final_output", "reads": ["${'$'}scene"], "writes": "${'$'}output" }
              ],
              "tiers": [
                {
                  "id": "A",
                  "textureMaxSize": 1024
                },
                {
                  "id": "C",
                  "textureMaxSize": 512,
                  "disablePasses": ["film_grain"],
                  "passScaleOverrides": {
                    "bilateral_beauty": 0.25
                  },
                  "paramOverrides": {
                    "pass:bilateral_beauty.radius": 2.0
                  }
                }
              ]
            }
            """.trimIndent()
        )

        val effect = EffectCompiler.compile(manifest, File("."))
        assertTrue(effect.passGraph.isCustomManifestGraph)
        assertEquals(5, effect.passGraph.passes.size)

        // Verify RenderTargetRef parsing on bilateral_beauty pass
        val bilateralPass = effect.passGraph.passes[2]
        assertEquals(RenderTargetRef.TargetKind.SCENE, bilateralPass.parsedReads[0].kind)
        assertEquals(RenderTargetRef.TargetKind.MASK, bilateralPass.parsedReads[1].kind)
        assertEquals("skin", bilateralPass.parsedReads[1].identifier)
        assertEquals(RenderTargetRef.TargetKind.NAMED_RT, bilateralPass.parsedWrites.kind)
        assertEquals("softSkin", bilateralPass.parsedWrites.identifier)

        val evaluator = SceneEvaluator()
        val renderGraph = RenderGraph()
        val faceState = createTrackingFaceState()

        // Render in Tier A: all 5 passes execute; bilateral_beauty runs at scale 0.5 (540x960) with radius 4.0
        effect.activeTier = "A"
        evaluator.evaluate(effect, faceState, effect.propertySlots)
        renderGraph.render(effect, faceState, evaluator, 1080, 1920)

        assertEquals(
            listOf("camera_in", "skin_mask_gen", "bilateral_beauty", "film_grain", "present_out"),
            renderGraph.lastExecutedPassNodeIds
        )
        assertTrue(renderGraph.lastSkippedPassNodeIds.isEmpty())
        val tierABilateralRecord = renderGraph.lastExecutedPassRecords.first { it.passId == "bilateral_beauty" }
        assertEquals(0.5f, tierABilateralRecord.effectiveScaleFactor, 1e-4f)
        assertEquals(4.0f, tierABilateralRecord.resolvedParams["radius"] ?: 0f, 1e-4f)

        val softSkinRtTierA = renderGraph.rtPool.getNamedTarget("\$rt:softSkin")
        assertNotNull(softSkinRtTierA)
        assertEquals(540, softSkinRtTierA!!.width)
        assertEquals(960, softSkinRtTierA.height)

        // Render in Tier C: film_grain is disabled via data, bilateral_beauty drops to 0.25 scale (270x480) and radius 2.0
        effect.activeTier = "C"
        evaluator.evaluate(effect, faceState, effect.propertySlots)
        renderGraph.render(effect, faceState, evaluator, 1080, 1920)

        assertEquals(
            listOf("camera_in", "skin_mask_gen", "bilateral_beauty", "present_out"),
            renderGraph.lastExecutedPassNodeIds
        )
        assertEquals(listOf("film_grain"), renderGraph.lastSkippedPassNodeIds)
        val tierCBilateralRecord = renderGraph.lastExecutedPassRecords.first { it.passId == "bilateral_beauty" }
        assertEquals(0.25f, tierCBilateralRecord.effectiveScaleFactor, 1e-4f)
        assertEquals(2.0f, tierCBilateralRecord.resolvedParams["radius"] ?: 0f, 1e-4f)

        val softSkinRtTierC = renderGraph.rtPool.getNamedTarget("\$rt:softSkin")
        assertNotNull(softSkinRtTierC)
        assertEquals(270, softSkinRtTierC!!.width)
        assertEquals(480, softSkinRtTierC.height)
    }

    @Test
    fun universalSignalToParameterBinding_updatesPassUniformsMaterialParamsAndMaskWeights() {
        val manifest = JSONObject(
            """
            {
              "schemaVersion": "1.12.0",
              "id": "com.xelvic.test.g0.param_binding",
              "name": "G0 Universal Parameter Binding Test",
              "assets": [
                { "id": "blush_tex", "type": "image", "path": "assets/blush.png" }
              ],
              "materials": {
                "blush_mat": { "template": "faceTexture@1", "blend": "alpha", "params": { "map": "blush_tex", "opacity": 0.4 } }
              },
              "nodes": [
                { "id": "blush_mesh", "type": "faceMesh", "material": "blush_mat" },
                { "id": "warm_post", "type": "post", "effect": "colorGrade", "params": { "warmth": 0.1, "intensity": 0.9 } }
              ],
              "passes": [
                { "id": "camera_frame", "stage": "camera_frame" },
                { "id": "camera_pixel_passes", "stage": "camera_pixel_passes" },
                { "id": "post_nodes", "stage": "post_nodes", "params": { "glareStrength": 0.2 } },
                { "id": "final_output", "stage": "final_output" }
              ],
              "behaviors": [
                {
                  "id": "smile_to_warmth_and_blush",
                  "type": "map",
                  "input": "face.blend.mouthSmileLeft",
                  "inputRange": [0.0, 1.0],
                  "outputRange": [0.2, 0.85],
                  "easing": "linear",
                  "smoothingTau": 0.001,
                  "initial": 0.2,
                  "writes": [
                    { "path": "pass:warm_post.warmth", "mode": "set" },
                    { "path": "pass:post_nodes.glareStrength", "mode": "set" },
                    { "path": "mat:blush_mat.opacity", "mode": "set" },
                    { "path": "mask:cheeks.weight", "mode": "set" }
                  ]
                }
              ]
            }
            """.trimIndent()
        )

        val effect = EffectCompiler.compile(manifest, File("."))
        val evaluator = SceneEvaluator()
        val renderGraph = RenderGraph()

        // Evaluate with smile = 1.0 (settled via photo evaluation and live step)
        val smilingState = createTrackingFaceState(smile = 1.0f)
        effect.signalBus.evaluate(smilingState, 1.0f)
        effect.behaviorRunner.evaluateForPhoto(
            photoPolicy = effect.photoPolicy,
            signalBus = effect.signalBus,
            faceState = smilingState,
            slots = effect.propertySlots
        )
        evaluator.evaluate(effect, smilingState, effect.propertySlots)
        renderGraph.render(effect, smilingState, evaluator, 1080, 1920)

        assertEquals(0.85f, effect.propertySlots.getDynamicParam("pass:warm_post.warmth"), 1e-3f)
        assertEquals(0.85f, effect.propertySlots.getDynamicParam("pass:post_nodes.glareStrength"), 1e-3f)
        assertEquals(0.85f, effect.propertySlots.getDynamicParam("mat:blush_mat.opacity"), 1e-3f)
        assertEquals(0.85f, effect.propertySlots.getDynamicParam("mask:cheeks.weight"), 1e-3f)

        // Verify PostRenderer received the dynamically bound warmth parameter (0.85 instead of static 0.1)
        assertEquals(1, renderGraph.post.lastExecutedPasses.size)
        val executedPost = renderGraph.post.lastExecutedPasses[0]
        assertEquals(0.85f, executedPost.params["warmth"] ?: 0f, 1e-3f)

        // Verify PassGraph execution record received the dynamically bound glareStrength (0.85)
        val postStageRecord = renderGraph.lastExecutedPassRecords.first { it.passId == "post_nodes" }
        assertEquals(0.85f, postStageRecord.resolvedParams["glareStrength"] ?: 0f, 1e-3f)
    }

    @Test
    fun deterministicFrameContext_matchesSeedAndNoiseBetweenPreviewAndPhoto() {
        val pkgId = "com.xelvic.filter.vintage_grain"
        val timeSec = 1.25f

        val previewCtx = DeterministicFrameContext.forPreview(
            packageId = pkgId,
            renderTimeSec = timeSec,
            frameIndex = 42L,
            explicitSeed = null
        )
        val photoCtx = DeterministicFrameContext.forPhoto(
            packageId = pkgId,
            photoTimeSec = timeSec,
            explicitSeed = null
        )

        assertEquals(previewCtx.frameSeed, photoCtx.frameSeed)
        for (y in 0 until 8) {
            for (x in 0 until 8) {
                assertEquals(
                    previewCtx.seededNoise01(x, y, salt = 7),
                    photoCtx.seededNoise01(x, y, salt = 7),
                    1e-6f
                )
            }
        }
    }
}
