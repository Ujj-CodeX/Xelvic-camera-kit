package com.example

import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.PackageValidator
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.FaceMeshRenderer
import com.example.engine.render.GlResourceOwner
import com.example.engine.render.MaterialRegistry
import com.example.engine.render.RenderGraph
import com.example.engine.render.TextureCache
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs

/**
 * Step 8 Unit & Conformance Tests — Xelvic Face Mesh Texture Pipeline (README Section 5.7, 5.8, 11.4, 15 & 21 Step 8).
 *
 * Covers:
 * 1. Exact task-model SHA-256 verification against `app/src/main/assets/face_landmarker.task`
 * 2. UV registry identity, 468-vertex / 898-triangle topology compatibility, and task bundle parity
 * 3. Landmark-to-UV correspondence (midline, vertical/horizontal anatomical ordering, bilateral symmetry)
 * 4. Incorrect and legacy canonical OBJ UV layout rejection (MediaPipe issue #5760)
 * 5. FaceMeshRenderer live vertex updates across frontal, smile, blink, and head turn poses
 * 6. Static topology and UV buffer reuse across multi-frame rendering
 * 7. Mask application and protected facial regions (eyes, lips, mouth interior)
 * 8. Premultiplied alpha blending and material opacity scaling
 * 9. TextureCache bounds, tier size limits, off-thread decode staging, and GlResourceOwner disposal
 * 10. Invalid package rejection & end-to-end `ref_face_paint` conformance verification
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class FaceMeshTextureStep8UnitTest {

    private fun resolveTaskModelFile(): File {
        val candidates = listOf(
            File("app/src/main/assets/face_landmarker.task"),
            File("src/main/assets/face_landmarker.task"),
            File("../app/src/main/assets/face_landmarker.task")
        )
        return candidates.firstOrNull { it.exists() } ?: candidates.first()
    }

    private fun resolveRefFacePaintDir(): File {
        val candidates = listOf(
            File("app/src/main/assets/filters/ref_face_paint"),
            File("src/main/assets/filters/ref_face_paint"),
            File("../app/src/main/assets/filters/ref_face_paint")
        )
        return candidates.firstOrNull { it.exists() } ?: candidates.first()
    }

    /**
     * Synthesizes a 478-landmark tracked face derived from the pinned `CANONICAL_UVS`
     * with configurable expression deformations (smile, blink, jawOpen) and head yaw/pitch.
     */
    private fun createTrackedFaceState(
        yawDeg: Float = 0f,
        pitchDeg: Float = 0f,
        smile: Float = 0f,
        blink: Float = 0f,
        jawOpen: Float = 0f,
        depthMeters: Float = -0.50f
    ): FaceState {
        val uvs = UvRegistry.CANONICAL_UVS
        val landmarks = FloatArray(478 * 3)
        val yawRad = Math.toRadians(yawDeg.toDouble()).toFloat()
        val pitchRad = Math.toRadians(pitchDeg.toDouble()).toFloat()
        val sinYaw = kotlin.math.sin(yawRad)
        val cosYaw = kotlin.math.cos(yawRad)

        for (i in 0 until 468) {
            var u = uvs[i * 2]
            var v = uvs[i * 2 + 1]

            // Apply realistic expression deformations to specific landmark groups
            when (i) {
                61 -> { // Right mouth corner moves right (smaller u) and up (smaller v) on smile
                    u -= 0.035f * smile
                    v -= 0.025f * smile
                }
                291 -> { // Left mouth corner moves left (larger u) and up (smaller v) on smile
                    u += 0.035f * smile
                    v -= 0.025f * smile
                }
                13 -> { // Upper inner lip
                    v -= 0.010f * jawOpen
                }
                14, 17, 152 -> { // Lower inner lip, lower outer lip, chin move down on jawOpen
                    v += 0.040f * jawOpen
                }
                159, 386 -> { // Upper eyelids move down toward lower eyelids on blink
                    v += 0.018f * blink
                }
            }

            // Center around 0.5 and apply 3D head yaw & pitch rotation
            val relX = (u - 0.5f) * 0.45f
            val relY = (v - 0.5f) * 0.55f
            val baseZ = -0.04f * kotlin.math.cos((u - 0.5f) * 3.14f)

            val rotX = relX * cosYaw - baseZ * sinYaw
            val rotZ = relX * sinYaw + baseZ * cosYaw
            val rotY = relY + pitchRad * 0.12f

            landmarks[i * 3] = 0.5f + rotX
            landmarks[i * 3 + 1] = 0.5f + rotY
            landmarks[i * 3 + 2] = rotZ
        }
        // Populate iris refinement landmarks 468..477
        for (i in 468 until 478) {
            landmarks[i * 3] = 0.5f
            landmarks[i * 3 + 1] = 0.4f
            landmarks[i * 3 + 2] = -0.02f
        }

        val poseMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        poseMatrix[0] = cosYaw
        poseMatrix[2] = sinYaw
        poseMatrix[8] = -sinYaw
        poseMatrix[10] = cosYaw
        poseMatrix[12] = 0f
        poseMatrix[13] = 0f
        poseMatrix[14] = depthMeters

        return FaceState(
            isVisible = true,
            fadeWeight = 1.0f,
            faceWidth = 0.14f,
            poseMatrix = poseMatrix,
            landmarks = landmarks,
            isFrontCamera = true
        )
    }

    @Test
    fun test1_exactTaskModelSha256AndBundleVerification() {
        val taskFile = resolveTaskModelFile()
        assertTrue("Pinned face_landmarker.task must exist at ${taskFile.path}", taskFile.exists())

        val bundleVerify = UvRegistry.verifyAgainstTaskBundle(taskFile)
        assertTrue(
            "Task bundle verification failed: ${bundleVerify.errorReason}",
            bundleVerify.isValid
        )

        // Verify exact model SHA-256 acceptance and rejection
        assertTrue(UvRegistry.verifyModelSha256(UvRegistry.PINNED_MODEL_SHA256).isValid)
        assertTrue(UvRegistry.verifyModelSha256(UvRegistry.PINNED_MODEL_SHA256.uppercase()).isValid)

        val wrongHashResult = UvRegistry.verifyModelSha256(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        assertFalse("Wrong model SHA-256 must be rejected", wrongHashResult.isValid)
        assertTrue(wrongHashResult.errorReason!!.contains("mismatch", ignoreCase = true))
    }

    @Test
    fun test2_uvRegistryIdentityAndTopologyCompatibility() {
        assertEquals("mediapipe_canonical_468_v1", UvRegistry.UV_LAYOUT_ID)
        assertEquals(468, UvRegistry.VERTEX_COUNT)
        assertEquals(478, UvRegistry.TRACKER_LANDMARK_COUNT)
        assertEquals(898, UvRegistry.TRIANGLE_COUNT)
        assertEquals(2694, UvRegistry.INDEX_COUNT)
        assertEquals(936, UvRegistry.CANONICAL_UVS.size)
        assertEquals(2694, UvRegistry.CANONICAL_TRIANGLES.size)

        assertEquals(UvRegistry.EXPECTED_UV_LAYOUT_SHA256, UvRegistry.computeUvLayoutSha256())
        assertEquals(UvRegistry.EXPECTED_TRIANGLES_SHA256, UvRegistry.computeTrianglesSha256())

        val layoutResult = UvRegistry.verifyUvLayout()
        assertTrue("Pinned UV layout verification failed: ${layoutResult.errorReason}", layoutResult.isValid)

        // Verify FaceGeometrySnapshot compatibility check
        val validSnapshot = FaceGeometrySnapshot.create(
            timestampNs = 1_000_000L,
            imageWidth = 480,
            imageHeight = 640,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = FloatArray(478 * 3),
            rawPoseMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f },
            blendshapes = emptyMap(),
            modelSha256 = UvRegistry.PINNED_MODEL_SHA256
        )
        assertTrue(UvRegistry.verifySnapshotCompatibility(validSnapshot).isValid)

        // Reject snapshot with mismatched model hash
        val badHashSnapshot = validSnapshot.copy(modelSha256 = "bad_sha256_hash")
        assertFalse(UvRegistry.verifySnapshotCompatibility(badHashSnapshot).isValid)

        // Reject snapshot with insufficient landmark count (< 468)
        val shortLandmarkSnapshot = validSnapshot.copy(landmarks = FloatArray(400 * 3))
        assertFalse(UvRegistry.verifySnapshotCompatibility(shortLandmarkSnapshot).isValid)
    }

    @Test
    fun test3_landmarkToUvCorrespondenceAndSymmetry() {
        val corrResult = UvRegistry.verifyLandmarkUvCorrespondence()
        assertTrue("Landmark-to-UV correspondence failed: ${corrResult.errorReason}", corrResult.isValid)

        // Verify midline landmarks sit on u == 0.5000
        val midlineLandmarks = intArrayOf(10, 168, 6, 1, 2, 0, 13, 14, 17, 152)
        for (lm in midlineLandmarks) {
            val uv = UvRegistry.getVertexUv(lm)
            assertEquals("Landmark #$lm must lie on midline u=0.5", 0.5f, uv[0], 0.008f)
        }

        // Verify anatomical vertical progression (v increases from forehead to chin)
        val vForehead = UvRegistry.getVertexUv(10)[1]
        val vNoseBridge = UvRegistry.getVertexUv(168)[1]
        val vNoseTip = UvRegistry.getVertexUv(1)[1]
        val vUpperLip = UvRegistry.getVertexUv(0)[1]
        val vLowerLip = UvRegistry.getVertexUv(17)[1]
        val vChin = UvRegistry.getVertexUv(152)[1]

        assertTrue(vForehead < vNoseBridge)
        assertTrue(vNoseBridge < vNoseTip)
        assertTrue(vNoseTip < vUpperLip)
        assertTrue(vUpperLip < vLowerLip)
        assertTrue(vLowerLip < vChin)

        // Verify anatomical horizontal progression (u increases from subject-right #234 to subject-left #454)
        val uRightCheek = UvRegistry.getVertexUv(234)[0]
        val uRightEye = UvRegistry.getVertexUv(33)[0]
        val uNoseTip = UvRegistry.getVertexUv(1)[0]
        val uLeftEye = UvRegistry.getVertexUv(263)[0]
        val uLeftCheek = UvRegistry.getVertexUv(454)[0]

        assertTrue(uRightCheek < uRightEye)
        assertTrue(uRightEye < uNoseTip)
        assertTrue(uNoseTip < uLeftEye)
        assertTrue(uLeftEye < uLeftCheek)
    }

    @Test
    fun test4_rejectLegacyCanonicalObjAndIncompatibleUvLayouts() {
        // 1. Reject legacy canonical OBJ UV layout IDs (MediaPipe issue #5760)
        for (legacyId in UvRegistry.REJECTED_LEGACY_UV_LAYOUT_IDS) {
            assertFalse("Legacy UV ID '$legacyId' must be rejected", UvRegistry.isSupportedUvLayoutId(legacyId))
            val res = UvRegistry.verifyUvLayout(uvLayoutId = legacyId)
            assertFalse("verifyUvLayout must reject legacy ID '$legacyId'", res.isValid)
        }

        // 2. Reject V-flipped OBJ UV array even if it has 468 vertices (936 floats)
        val flippedObjUvs = UvRegistry.CANONICAL_UVS.clone()
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            flippedObjUvs[i * 2 + 1] = 1.0f - flippedObjUvs[i * 2 + 1]
        }
        val flippedRes = UvRegistry.verifyUvLayout(candidateUvs = flippedObjUvs)
        assertFalse("V-flipped OBJ UV coordinates must be rejected even with 468 vertices", flippedRes.isValid)

        // 3. Reject synthetic grid UV layout even if it has 468 vertices
        val syntheticGridUvs = FloatArray(UvRegistry.VERTEX_COUNT * 2) { idx ->
            val vert = idx / 2
            if (idx % 2 == 0) (vert % 24) / 24f else (vert / 24) / 20f
        }
        val gridRes = UvRegistry.verifyUvLayout(candidateUvs = syntheticGridUvs)
        assertFalse("Synthetic grid UV layout must be rejected", gridRes.isValid)

        // 4. Reject scrambled vertex ordering (simulating MediaPipe #5760 vertex permutation)
        val scrambledUvs = UvRegistry.CANONICAL_UVS.clone()
        // Swap nose tip (#1) and chin (#152) UVs
        val u1 = scrambledUvs[2]; val v1 = scrambledUvs[3]
        scrambledUvs[2] = scrambledUvs[152 * 2]
        scrambledUvs[3] = scrambledUvs[152 * 2 + 1]
        scrambledUvs[152 * 2] = u1
        scrambledUvs[152 * 2 + 1] = v1
        val scrambledRes = UvRegistry.verifyUvLayout(candidateUvs = scrambledUvs)
        assertFalse("Permuted vertex UV ordering must be rejected", scrambledRes.isValid)
    }

    @Test
    fun test5_faceMeshRendererLiveVertexUpdatesAcrossExpressionsAndHeadTurns() {
        val renderer = FaceMeshRenderer()
        renderer.init()

        // Frame 1: Frontal neutral pose
        val neutralState = createTrackedFaceState(yawDeg = 0f, smile = 0f, blink = 0f, jawOpen = 0f)
        assertTrue(renderer.updateVertices(neutralState, viewportAspect = 9f / 16f))

        val neutralRightMouth = renderer.getLiveVertexPosition(61)
        val neutralLeftMouth = renderer.getLiveVertexPosition(291)
        val neutralChin = renderer.getLiveVertexPosition(152)
        val neutralUpperLid = renderer.getLiveVertexPosition(159)
        val neutralNose = renderer.getLiveVertexPosition(1)

        // Frame 2: Smile + Jaw Open expression
        val smileState = createTrackedFaceState(yawDeg = 0f, smile = 1.0f, blink = 0f, jawOpen = 0.8f)
        assertTrue(renderer.updateVertices(smileState, viewportAspect = 9f / 16f))

        val smileRightMouth = renderer.getLiveVertexPosition(61)
        val smileLeftMouth = renderer.getLiveVertexPosition(291)
        val openChin = renderer.getLiveVertexPosition(152)

        // Mouth corners widen and lift in camera space (+Y is up)
        val neutralMouthWidth = smileLeftMouth[0] - smileRightMouth[0]
        val baseMouthWidth = neutralLeftMouth[0] - neutralRightMouth[0]
        assertTrue("Mouth width must increase during smile", neutralMouthWidth > baseMouthWidth)
        assertTrue("Right mouth corner must lift (+Y) during smile", smileRightMouth[1] > neutralRightMouth[1])
        assertTrue("Left mouth corner must lift (+Y) during smile", smileLeftMouth[1] > neutralLeftMouth[1])
        assertTrue("Chin must drop (-Y) when jaw opens", openChin[1] < neutralChin[1])

        // Frame 3: Blink expression
        val blinkState = createTrackedFaceState(yawDeg = 0f, smile = 0f, blink = 1.0f, jawOpen = 0f)
        assertTrue(renderer.updateVertices(blinkState, viewportAspect = 9f / 16f))
        val blinkUpperLid = renderer.getLiveVertexPosition(159)
        assertTrue("Upper eyelid (#159) must move downward (-Y) during blink", blinkUpperLid[1] < neutralUpperLid[1])

        // Frame 4: Head yaw turn (+25 deg)
        val yawState = createTrackedFaceState(yawDeg = 25f, smile = 0f, blink = 0f, jawOpen = 0f)
        assertTrue(renderer.updateVertices(yawState, viewportAspect = 9f / 16f))
        val yawRightCheek = renderer.getLiveVertexPosition(234)
        val yawLeftCheek = renderer.getLiveVertexPosition(454)
        val yawNose = renderer.getLiveVertexPosition(1)

        assertNotEquals("Nose X must shift under head yaw rotation", neutralNose[0], yawNose[0], 0.001f)
        assertNotEquals("Right and left cheeks must have asymmetric depth Z under head yaw", yawRightCheek[2], yawLeftCheek[2], 0.005f)
    }

    @Test
    fun test6_staticTopologyAndUvReuseWithoutFrameAllocations() {
        val resourceOwner = GlResourceOwner()
        resourceOwner.bindGlThread()
        val renderer = FaceMeshRenderer()
        renderer.init(resourceOwner)

        assertEquals("Static triangle topology must be uploaded exactly once at init", 1, renderer.topologyUploadCount)
        assertEquals("Static UV layout must be uploaded exactly once at init", 1, renderer.uvUploadCount)
        assertEquals(2694, renderer.indexCount)

        val state = createTrackedFaceState()
        val viewProj = FloatArray(16) { if (it % 5 == 0) 1f else 0f }

        // Simulate 120 live frames
        for (frame in 0 until 120) {
            assertTrue(renderer.updateVertices(state, viewportAspect = 9f / 16f))
            renderer.render(
                viewProjectionMatrix = viewProj,
                textureId = 1,
                maskId = 2,
                opacity = 0.9f,
                program = 1
            )
        }

        assertEquals("Static topology must NOT be re-uploaded during frame loop", 1, renderer.topologyUploadCount)
        assertEquals("Static UVs must NOT be re-uploaded during frame loop", 1, renderer.uvUploadCount)
        assertEquals("Live vertices must update every frame", 120, renderer.vertexUpdateCount)
        assertEquals("Render call count must match 120 frames", 120, renderer.renderCallCount)

        renderer.release(resourceOwner)
        assertFalse(renderer.isInitialized)
    }

    @Test
    fun test7_maskApplicationAndProtectedFacialRegions() {
        val renderer = FaceMeshRenderer()
        val paintSample = floatArrayOf(0.8f, 0.2f, 0.4f, 0.9f) // premultiplied RGBA

        // 1. Protected eye landmarks (#33 right eye corner, #159 right upper lid, #263 left eye corner, #386 left upper lid)
        for (eyeLm in intArrayOf(33, 159, 145, 263, 386, 374)) {
            assertEquals("Vertex protection weight for eye landmark #$eyeLm must be 0", 0f, UvRegistry.VERTEX_PROTECTION_WEIGHTS[eyeLm], 0.0001f)
            val uv = UvRegistry.getVertexUv(eyeLm)
            val frag = renderer.evaluateEffectiveFragment(
                u = uv[0],
                v = uv[1],
                texRgba = paintSample,
                isTexPremultiplied = true,
                maskValue = 1.0f,
                materialOpacity = 1.0f,
                protectEyes = true,
                protectLips = true,
                protectMouthInterior = true
            )
            assertEquals("Eye landmark #$eyeLm must be protected (0 alpha)", 0f, frag[3], 0.05f)
        }

        // 2. Protected mouth interior landmarks (#13 upper inner lip, #14 lower inner lip, #78, #308)
        for (mouthLm in intArrayOf(13, 14, 78, 308)) {
            assertEquals("Vertex protection weight for mouth interior #$mouthLm must be 0", 0f, UvRegistry.VERTEX_PROTECTION_WEIGHTS[mouthLm], 0.0001f)
            val uv = UvRegistry.getVertexUv(mouthLm)
            val frag = renderer.evaluateEffectiveFragment(
                u = uv[0],
                v = uv[1],
                texRgba = paintSample,
                isTexPremultiplied = true,
                maskValue = 1.0f,
                materialOpacity = 1.0f,
                protectEyes = true,
                protectLips = true,
                protectMouthInterior = true
            )
            assertEquals("Mouth interior landmark #$mouthLm must be protected (0 alpha)", 0f, frag[3], 0.001f)
        }

        // 3. Protected outer lip landmarks (#0 upper lip top, #17 lower lip bottom)
        for (lipLm in intArrayOf(0, 17, 61, 291)) {
            assertEquals("Vertex protection weight for lip landmark #$lipLm must be 0", 0f, UvRegistry.VERTEX_PROTECTION_WEIGHTS[lipLm], 0.0001f)
        }
        val uvUpperLip = UvRegistry.getVertexUv(0)
        val fragLip = renderer.evaluateEffectiveFragment(
            u = uvUpperLip[0],
            v = uvUpperLip[1],
            texRgba = paintSample,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 1.0f,
            protectEyes = true,
            protectLips = true,
            protectMouthInterior = true
        )
        assertEquals("Outer lip center (#0) must be protected (0 alpha)", 0f, fragLip[3], 0.05f)

        // 4. Unprotected paint regions: Forehead (#10), Right Cheek (#234), Left Cheek (#454), Nose Bridge (#168)
        for (paintLm in intArrayOf(10, 234, 454, 168)) {
            assertEquals("Vertex protection weight for paint landmark #$paintLm must be 1", 1f, UvRegistry.VERTEX_PROTECTION_WEIGHTS[paintLm], 0.0001f)
            val uv = UvRegistry.getVertexUv(paintLm)
            val frag = renderer.evaluateEffectiveFragment(
                u = uv[0],
                v = uv[1],
                texRgba = paintSample,
                isTexPremultiplied = true,
                maskValue = 1.0f,
                materialOpacity = 0.9f,
                protectEyes = true,
                protectLips = true,
                protectMouthInterior = true
            )
            assertTrue("Paint landmark #$paintLm must receive positive paint alpha (got ${frag[3]})", frag[3] > 0.75f)
        }
    }

    @Test
    fun test8_premultipliedAlphaBlendingAndMaterialOpacity() {
        val renderer = FaceMeshRenderer()
        // Forehead UV (#10) has regionMask = 1.0
        val uv = UvRegistry.getVertexUv(10)

        // Case A: Premultiplied texture input (r=0.4, g=0.2, b=0.1, a=0.5) with opacity=0.8, mask=0.5 -> effectiveMask = 0.4
        val premultTex = floatArrayOf(0.4f, 0.2f, 0.1f, 0.5f)
        val outPremult = renderer.evaluateEffectiveFragment(
            u = uv[0],
            v = uv[1],
            texRgba = premultTex,
            isTexPremultiplied = true,
            maskValue = 0.5f,
            materialOpacity = 0.8f
        )
        // Premultiplied RGB must NOT be multiplied by tex.a a second time!
        assertEquals(0.4f * 0.4f, outPremult[0], 0.0001f)
        assertEquals(0.2f * 0.4f, outPremult[1], 0.0001f)
        assertEquals(0.1f * 0.4f, outPremult[2], 0.0001f)
        assertEquals(0.5f * 0.4f, outPremult[3], 0.0001f)

        // Case B: Straight (unpremultiplied) texture input (r=0.8, g=0.4, b=0.2, a=0.5) -> baseRgb = rgb * a = (0.4, 0.2, 0.1)
        val straightTex = floatArrayOf(0.8f, 0.4f, 0.2f, 0.5f)
        val outStraight = renderer.evaluateEffectiveFragment(
            u = uv[0],
            v = uv[1],
            texRgba = straightTex,
            isTexPremultiplied = false,
            maskValue = 0.5f,
            materialOpacity = 0.8f
        )
        assertArrayEquals(outPremult, outStraight, 0.0001f)

        // Case C: Zero opacity produces transparent zeroed premultiplied output
        val outZero = renderer.evaluateEffectiveFragment(
            u = uv[0],
            v = uv[1],
            texRgba = premultTex,
            isTexPremultiplied = true,
            maskValue = 1.0f,
            materialOpacity = 0.0f
        )
        assertArrayEquals(floatArrayOf(0f, 0f, 0f, 0f), outZero, 0.0001f)
    }

    @Test
    fun test9_textureCacheBoundsAndResourceOwnership() {
        val owner = GlResourceOwner()
        owner.bindGlThread()
        val cache = TextureCache(resourceOwner = owner, maxEntries = 8, maxTotalBytes = 256 * 1024L)

        val pkgDir = resolveRefFacePaintDir()
        val paintFile = File(pkgDir, "assets/paint.png")
        val maskFile = File(pkgDir, "assets/paint_mask.png")
        assertTrue(paintFile.exists())
        assertTrue(maskFile.exists())

        // Off-thread decode staging into bounded cache
        assertTrue(cache.preloadDecodeToStaging(paintFile.absolutePath, maxSize = 512))
        assertTrue(cache.preloadDecodeToStaging(maskFile.absolutePath, maxSize = 512))
        assertEquals(2, cache.pendingUploadCount())

        // Step distributed upload on GL thread (1 per frame)
        val uploaded1 = cache.stepFrameUploads(maxUploadsPerFrame = 1)
        assertEquals(1, uploaded1)
        assertEquals(1, cache.pendingUploadCount())

        val uploaded2 = cache.stepFrameUploads(maxUploadsPerFrame = 1)
        assertEquals(1, uploaded2)
        assertEquals(0, cache.pendingUploadCount())

        assertTrue(cache.totalBytes <= cache.maxTotalBytes)
        assertEquals(2, cache.size())

        cache.releaseAll()
        owner.drainPendingDeletions()
        assertEquals(0L, cache.totalBytes)
        assertEquals(0, cache.size())
    }

    @Test
    fun test10_invalidPackageRejectionAndRefFacePaintConformance() {
        // 1. Reject faceTexture@1 material missing params.map
        val missingMapJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "bad_no_map",
              "name": "Bad No Map",
              "materials": {
                "mat_paint": { "template": "faceTexture@1", "params": {} }
              },
              "nodes": [
                { "id": "mesh", "type": "faceMesh", "material": "mat_paint" }
              ]
            }
            """.trimIndent()
        )
        val resNoMap = PackageValidator.validateManifest(missingMapJson)
        assertFalse(resNoMap.isValid)
        assertTrue(resNoMap.errorReason!!.contains("map"))

        // 2. Reject faceTexture@1 material with legacy canonical OBJ UV layout
        val legacyUvJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "bad_legacy_uv",
              "name": "Bad Legacy UV",
              "materials": {
                "mat_paint": {
                  "template": "faceTexture@1",
                  "params": { "map": "tex", "uvLayout": "canonical_face_model_obj" }
                }
              },
              "nodes": [
                { "id": "mesh", "type": "faceMesh", "material": "mat_paint" }
              ]
            }
            """.trimIndent()
        )
        val resLegacyUv = PackageValidator.validateManifest(legacyUvJson)
        assertFalse(resLegacyUv.isValid)
        assertTrue(resLegacyUv.errorReason!!.contains("UV layout"))

        // 3. Reject faceMesh node bound to non-faceTexture@1 material
        val wrongMatJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "bad_wrong_mat",
              "name": "Bad Wrong Mat",
              "materials": {
                "mat_unlit": { "template": "unlit@1", "params": {} }
              },
              "nodes": [
                { "id": "mesh", "type": "faceMesh", "material": "mat_unlit" }
              ]
            }
            """.trimIndent()
        )
        val resWrongMat = PackageValidator.validateManifest(wrongMatJson)
        assertFalse(resWrongMat.isValid)
        assertTrue(resWrongMat.errorReason!!.contains("faceTexture@1"))

        // 4. Validate and compile the reference `ref_face_paint` conformance package
        val refDir = resolveRefFacePaintDir()
        assertTrue("ref_face_paint directory must exist", refDir.exists())
        val manifestText = File(refDir, "manifest.json").readText()
        val manifestJson = JSONObject(manifestText)
        val validRes = PackageValidator.validateManifest(manifestJson)
        assertTrue("ref_face_paint manifest validation failed: ${validRes.errorReason}", validRes.isValid)

        val compiled = Compiler.compile(manifestJson, refDir)
        assertEquals("ref_face_paint", compiled.packageId)
        assertTrue(compiled.requiredLandmarkCount >= 468)
        assertTrue(compiled.requiresPose)

        val mat = compiled.materials["mat_paint"]
        assertNotNull(mat)
        assertEquals("faceTexture@1", mat!!.template)
        assertEquals("paint_texture", mat.mapAssetId)
        assertEquals("paint_mask", mat.maskAssetId)
        assertEquals(UvRegistry.UV_LAYOUT_ID, mat.uvLayout)
        assertEquals(0.9f, mat.opacity, 0.001f)
        assertTrue(mat.protectEyes)
        assertTrue(mat.protectLips)
        assertTrue(mat.protectMouthInterior)

        // Verify tier texture size overrides (A=1024, B=512, C=256)
        assertEquals(1024, TierPolicy.resolveEffectiveTier(compiled, PerformanceGovernor.Tier.A, 1024).textureMaxSize)
        assertEquals(512, TierPolicy.resolveEffectiveTier(compiled, PerformanceGovernor.Tier.B, 1024).textureMaxSize)
        assertEquals(256, TierPolicy.resolveEffectiveTier(compiled, PerformanceGovernor.Tier.C, 1024).textureMaxSize)

        // 5. Execute RenderGraph + SceneEvaluator across frontal, smile, blink, and head turn poses
        val renderGraph = RenderGraph()
        renderGraph.init()
        val evaluator = SceneEvaluator()

        val testPoses = listOf(
            createTrackedFaceState(yawDeg = 0f, smile = 0f, blink = 0f),
            createTrackedFaceState(yawDeg = 0f, smile = 1f, blink = 0f, jawOpen = 0.6f),
            createTrackedFaceState(yawDeg = 0f, smile = 0f, blink = 1f),
            createTrackedFaceState(yawDeg = -22f, pitchDeg = 10f, smile = 0.5f, blink = 0f)
        )

        for (pose in testPoses) {
            evaluator.evaluate(compiled, pose, compiled.propertySlots)
            renderGraph.render(compiled, pose, evaluator, viewportWidth = 1080, viewportHeight = 1920)
        }

        assertEquals(4, renderGraph.faceMesh.vertexUpdateCount)
        assertEquals(4, renderGraph.faceMesh.renderCallCount)
        assertEquals(1, renderGraph.faceMesh.topologyUploadCount)
        assertEquals(1, renderGraph.faceMesh.uvUploadCount)
        assertEquals(0.9f, renderGraph.faceMesh.lastRenderedOpacity, 0.001f)

        renderGraph.release()
    }
}
