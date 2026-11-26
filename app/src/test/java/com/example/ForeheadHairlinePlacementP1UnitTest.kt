package com.example

import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.pkg.UvRegistry
import com.example.engine.scene.AnchorResolver
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import com.squareup.moshi.Moshi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Unit tests for **Chunk P1: Forehead-Left Hairline Placement (Package-Only)**:
 *
 * Verifies for both `sunflower_forehead` and `periwinkle_forehead`:
 * 1. Custom landmark anchor `"forehead_left_hairline"` is defined in `manifest.json` `"anchors"`
 *    with upper-left forehead hairline landmarks (`#297`, `#338`, `#332`), weights summing to `1.0`,
 *    and `offsetFW` synced with `calibration.tunableFields`.
 * 2. All custom anchor landmark IDs (`#297`, `#338`, `#332`) are verified from the repo's pinned
 *    468-point canonical topology (`face_landmarker.task` via `UvRegistry.extractTaskBundleGeometry`)
 *    to lie on the subject-left (`#454`) side (`canon.X > 0`, `UV.u > 0.5`, `localX < 0`), opposite
 *    `#234` (subject-right).
 * 3. `layer.anchor` is `"forehead_left_hairline"`, `layer.scale` is `0.9` for `sunflower_forehead`
 *    and `1.0` for `periwinkle_forehead`, `layer.offsetFW` is in sync with `calibration.tunableFields`,
 *    and no `.xfilter` duplicate exists.
 * 4. In a synthetic frontal pose with synthetic landmarks generated from the repo's canonical 468-point
 *    landmark coordinates:
 *    - The resolved flower center is above the subject-left eyebrow reference (`#334` upper eyebrow arch
 *      on the `#333` side, and `#282` lower eyebrow arch) by at least `flowerRadius + 0.10 FW`
 *      (`0.19 + 0.10 = 0.29 FW` for sunflower; `0.11 + 0.10 = 0.21 FW` for periwinkle), ensuring the
 *      bottom edge stays at least `0.10 FW` above the eyebrow.
 *    - The resolved flower center is inside the forehead band (below the hairline reference `#10` level
 *      plus the allowed bloom overlap, with top edge touching/slightly overlapping the hairline and
 *      bottom edge strictly below `#10`).
 *    - The resolved flower center is on the subject-left side (`X_cam > 0` in unmirrored camera space on
 *      the `#454` side, and `screenX < screenCenterX` on screen-left in the mirrored front preview).
 */
class ForeheadHairlinePlacementP1UnitTest {

    private val moshi = Moshi.Builder().build()

    @Suppress("UNCHECKED_CAST")
    private fun readManifestMap(filterName: String): Pair<File, Map<String, Any?>> {
        val candidates = listOf(
            File("app/src/main/assets/filters", filterName),
            File("src/main/assets/filters", filterName)
        )
        val dir = candidates.firstOrNull { it.exists() && it.isDirectory }
        assertNotNull("Filter directory for '$filterName' must exist", dir)
        val manifestFile = File(dir!!, "manifest.json")
        assertTrue("manifest.json for '$filterName' must exist", manifestFile.exists())
        val adapter = moshi.adapter(Map::class.java)
        val map = adapter.fromJson(manifestFile.readText()) as? Map<String, Any?>
        assertNotNull("Parsed manifest map for '$filterName' must not be null", map)
        return dir to map!!
    }

    private fun resolveTaskFile(): File {
        val candidates = listOf(
            File("app/src/main/assets/face_landmarker.task"),
            File("src/main/assets/face_landmarker.task")
        )
        val file = candidates.firstOrNull { it.exists() && it.isFile }
        assertNotNull("face_landmarker.task must exist in assets", file)
        return file!!
    }

    private fun toFloatList(value: Any?): List<Float> {
        val list = value as? List<*> ?: error("Expected list of numbers, got: $value")
        return list.map { (it as Number).toFloat() }
    }

    @Suppress("UNCHECKED_CAST")
    private fun buildAnchorResolverFromManifest(manifest: Map<String, Any?>): AnchorResolver {
        val anchorsMap = manifest["anchors"] as? Map<String, Any?> ?: emptyMap()
        val customAnchors = mutableMapOf<String, AnchorResolver.CustomAnchorDef>()
        for ((name, rawDef) in anchorsMap) {
            val defMap = rawDef as? Map<String, Any?> ?: continue
            val offList = (defMap["offsetFW"] as? List<*>)?.map { (it as Number).toFloat() } ?: listOf(0f, 0f, 0f)
            val offsetFW = floatArrayOf(
                offList.getOrElse(0) { 0f },
                offList.getOrElse(1) { 0f },
                offList.getOrElse(2) { 0f }
            )
            val lmListRaw = defMap["landmarks"] as? List<*>
            val landmarks = lmListRaw?.mapNotNull { item ->
                val lmMap = item as? Map<String, Any?> ?: return@mapNotNull null
                val id = (lmMap["id"] as? Number)?.toInt() ?: return@mapNotNull null
                val weight = (lmMap["weight"] as? Number)?.toFloat() ?: return@mapNotNull null
                AnchorResolver.LandmarkWeight(id = id, weight = weight)
            } ?: emptyList()
            customAnchors[name] = AnchorResolver.CustomAnchorDef(
                landmarks = landmarks,
                offsetFW = offsetFW
            )
        }
        return AnchorResolver(customAnchors)
    }

    /**
     * Verifies manifest structure, single-source-of-truth (no `.xfilter`), landmark IDs `#297, #338, #332`,
     * weights summing to `1.0`, scale (`0.9` for sunflower, `1.0` for periwinkle), and `calibration.tunableFields`
     * synchronization for both packages.
     */
    @Test
    @Suppress("UNCHECKED_CAST")
    fun manifestsDefineForeheadLeftHairlineAnchorAndKeepTunableFieldsInSync() {
        val specs = listOf(
            Triple("sunflower_forehead", 0.9f, listOf(0.0f, 0.06f, 0.02f)),
            Triple("periwinkle_forehead", 1.0f, listOf(0.0f, 0.02f, 0.02f))
        )
        val expectedAnchorOffsets = mapOf(
            "sunflower_forehead" to listOf(-0.01f, 0.05f, 0.03f),
            "periwinkle_forehead" to listOf(-0.01f, 0.02f, 0.02f)
        )

        for ((pkgName, expectedScale, expectedLayerOffset) in specs) {
            // 1. Single source of truth: no .xfilter file
            val zipCandidates = listOf(
                File("app/src/main/assets/filters/$pkgName.xfilter"),
                File("src/main/assets/filters/$pkgName.xfilter")
            )
            for (zip in zipCandidates) {
                assertFalse("$pkgName.xfilter must not exist (${zip.path})", zip.exists())
            }

            val (_, manifest) = readManifestMap(pkgName)
            val anchors = manifest["anchors"] as? Map<String, Any?>
            assertNotNull("$pkgName must contain 'anchors'", anchors)

            val hairlineAnchor = anchors!!["forehead_left_hairline"] as? Map<String, Any?>
            assertNotNull("$pkgName must define 'anchors.forehead_left_hairline'", hairlineAnchor)

            // 2. Verify landmarks [297 (0.5), 338 (0.25), 332 (0.25)] and weights summing to 1.0
            val landmarks = hairlineAnchor!!["landmarks"] as? List<Map<String, Any?>>
            assertNotNull("$pkgName forehead_left_hairline must define 'landmarks'", landmarks)
            assertEquals(3, landmarks!!.size)

            val idToWeight = landmarks.associate {
                ((it["id"] as Number).toInt()) to ((it["weight"] as Number).toFloat())
            }
            assertEquals(setOf(297, 338, 332), idToWeight.keys)
            assertEquals(0.50f, idToWeight[297]!!, 1e-6f)
            assertEquals(0.25f, idToWeight[338]!!, 1e-6f)
            assertEquals(0.25f, idToWeight[332]!!, 1e-6f)
            val weightSum = idToWeight.values.sum()
            assertEquals("$pkgName landmark weights must sum to 1.0", 1.0f, weightSum, 1e-6f)

            val anchorOffsetFW = toFloatList(hairlineAnchor["offsetFW"])
            val expectedAnchorOff = expectedAnchorOffsets.getValue(pkgName)
            assertEquals(3, anchorOffsetFW.size)
            for (i in 0..2) {
                assertEquals(expectedAnchorOff[i], anchorOffsetFW[i], 1e-5f)
            }

            // 3. Verify layer.anchor, layer.scale, and layer.offsetFW
            val layer = manifest["layer"] as? Map<String, Any?>
            assertNotNull("$pkgName must contain 'layer'", layer)
            assertEquals("forehead_left_hairline", layer!!["anchor"])
            assertEquals(expectedScale, (layer["scale"] as Number).toFloat(), 1e-5f)

            val layerOffsetFW = toFloatList(layer["offsetFW"])
            assertEquals(3, layerOffsetFW.size)
            for (i in 0..2) {
                assertEquals(expectedLayerOffset[i], layerOffsetFW[i], 1e-5f)
            }

            // 4. Verify calibration.tunableFields is in sync with layer and forehead_left_hairline
            val calibration = manifest["calibration"] as? Map<String, Any?>
            val tunableFields = calibration?.get("tunableFields") as? Map<String, Any?>
            assertNotNull("$pkgName must contain 'calibration.tunableFields'", tunableFields)

            assertEquals(expectedScale, (tunableFields!!["layer.scale"] as Number).toFloat(), 1e-5f)
            val tunableLayerOffset = toFloatList(tunableFields["layer.offsetFW"])
            val tunableAnchorOffset = toFloatList(tunableFields["forehead_left_hairline.offsetFW"])
            for (i in 0..2) {
                assertEquals(
                    "$pkgName calibration.tunableFields['layer.offsetFW'][$i] must match layer.offsetFW[$i]",
                    layerOffsetFW[i],
                    tunableLayerOffset[i],
                    1e-5f
                )
                assertEquals(
                    "$pkgName calibration.tunableFields['forehead_left_hairline.offsetFW'][$i] must match anchors.forehead_left_hairline.offsetFW[$i]",
                    anchorOffsetFW[i],
                    tunableAnchorOffset[i],
                    1e-5f
                )
            }
        }
    }

    /**
     * Synthesizes a frontal pose and synthetic landmarks from the repo's canonical 468-point topology
     * (`face_landmarker.task`) and verifies for both `sunflower_forehead` and `periwinkle_forehead`:
     * 1. Landmark verification: `#297`, `#338`, `#332`, `#333`, `#334`, `#282` lie on the `#454`
     *    (subject-left) side (`X_canon > 0`, `UV.u > 0.5`, `localX < 0`), opposite `#234` (subject-right).
     * 2. Eyebrow clearance: resolved flower center Y is above the subject-left eyebrow reference (`#334`
     *    upper eyebrow arch on the `#333` side, as well as `#282` lower eyebrow arch) by at least
     *    `flowerRadius + 0.10 FW` (`0.19 + 0.10 = 0.29 FW` unscaled / `0.174 + 0.10 = 0.274 FW` scaled
     *    for sunflower; `0.11 + 0.10 = 0.21 FW` for periwinkle), and the flower's bottom edge is also
     *    above mid-forehead `#333`.
     * 3. Forehead band & hairline overlap: resolved flower center Y is inside the forehead band (below
     *    the hairline reference `#10` level `Y_10` plus the allowed bloom overlap `<= flowerRadius`),
     *    with the top edge (`Y_center + scaledRadius`) overlapping `#10` and the bottom edge
     *    (`Y_center - scaledRadius`) strictly below `#10`.
     * 4. Subject-left placement: `localX < 0` in face-local FW space, `X_cam > 0` in unmirrored camera
     *    space (same side as `#454`), and `screenX < screenCenterX` (screen-left) in the mirrored
     *    front-camera preview.
     */
    @Test
    @Suppress("UNCHECKED_CAST")
    fun syntheticFrontalPoseAndCanonicalLandmarks_placesBothFlowersAtForeheadLeftHairlineWithEyebrowClearance() {
        val taskGeometry = UvRegistry.extractTaskBundleGeometry(resolveTaskFile())
        assertNotNull("Canonical geometry from face_landmarker.task must be extracted", taskGeometry)
        val canonPos = taskGeometry!!.canonicalPositions
        val canonUvs = UvRegistry.CANONICAL_UVS
        assertEquals(468 * 3, canonPos.size)

        // Helper to read canonical 3D vertex (cm)
        fun canonVertex(id: Int): FloatArray = floatArrayOf(
            canonPos[id * 3],
            canonPos[id * 3 + 1],
            canonPos[id * 3 + 2]
        )

        val c234 = canonVertex(FacePoseAdapter.LANDMARK_CHEEK_RIGHT) // #234 subject-right cheek
        val c454 = canonVertex(FacePoseAdapter.LANDMARK_CHEEK_LEFT)  // #454 subject-left cheek

        // Verify #454 is on +X_canon / u > 0.5 (subject-left) and #234 is on -X_canon / u < 0.5 (subject-right)
        assertTrue("#234 must have negative canonical X (subject-right)", c234[0] < -7.0f)
        assertTrue("#454 must have positive canonical X (subject-left)", c454[0] > 7.0f)
        assertTrue("#234 must have canonical UV.u < 0.5", canonUvs[234 * 2] < 0.1f)
        assertTrue("#454 must have canonical UV.u > 0.5", canonUvs[454 * 2] > 0.9f)

        // Verify all 3 hairline anchor landmarks (#297, #338, #332) and eyebrow/forehead reference landmarks
        // (#333, #334, #282) are on the same (#454 subject-left) side as #454
        for (id in listOf(297, 338, 332, 333, 334, 282)) {
            val v = canonVertex(id)
            val u = canonUvs[id * 2]
            assertTrue("Landmark #$id must be on #454 subject-left side (canon.X > 0), got ${v[0]}", v[0] > 0f)
            assertTrue("Landmark #$id must have canonical UV.u > 0.5 (subject-left), got $u", u > 0.5f)
        }

        // Compute canonical cheek midpoint and canonical FW (in meters)
        val midX = 0.5f * (c234[0] + c454[0])
        val midY = 0.5f * (c234[1] + c454[1])
        val midZ = 0.5f * (c234[2] + c454[2])
        val dx = c454[0] - c234[0]
        val dy = c454[1] - c234[1]
        val dz = c454[2] - c234[2]
        val fwCm = sqrt(dx * dx + dy * dy + dz * dz)
        val fwMeters = fwCm * 0.01f // ~0.153284 m

        // Synthesize a frontal pose at Z = -0.48 m in camera space and project all 468 canonical vertices
        // into normalized [0..1] MediaPipe landmarks using the exact camera frustum math of FacePoseAdapter
        val fovYCropDeg = 60.0f
        val cropAspect = 9f / 16f
        val fCrop = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)
        val headTz = -0.48f
        val absZ = -headTz
        val wCam = 2.0f * absZ * cropAspect / fCrop
        val hCam = 2.0f * absZ / fCrop

        val frontalPoseMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, headTz, 1f
        )

        val landmarks = FloatArray(478 * 3)
        for (id in 0 until 468) {
            // Metric offset from cheek_mid in meters
            val xPoseM = (canonPos[id * 3] - midX) * 0.01f
            val yPoseM = (canonPos[id * 3 + 1] - midY) * 0.01f
            val zPoseM = (canonPos[id * 3 + 2] - midZ) * 0.01f

            val xCam = xPoseM
            val yCam = yPoseM
            val zCam = headTz + zPoseM

            landmarks[id * 3] = 0.5f + xCam / wCam
            landmarks[id * 3 + 1] = 0.5f - yCam / hCam
            landmarks[id * 3 + 2] = (-absZ - zCam) / wCam
        }

        val projParams = ProjectionParams(
            fovYDeg = fovYCropDeg,
            fovXDeg = 36.0f,
            cropAspectRatio = cropAspect,
            isCalibratedFromIntrinsics = true
        )

        val faceState = FaceState(
            isVisible = true,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1f,
            poseMatrix = frontalPoseMatrix,
            landmarkPoseMatrix = frontalPoseMatrix,
            faceWidth = fwMeters,
            landmarks = landmarks,
            isFrontCamera = true,
            projectionParams = projParams
        )

        // Resolve canonical reference landmarks in face-local FW units via the same unproject + inverse(pose) path
        fun resolveSingleLandmarkFW(id: Int): FloatArray {
            val singleResolver = AnchorResolver(
                mapOf(
                    "lm_$id" to AnchorResolver.CustomAnchorDef(
                        landmarks = listOf(AnchorResolver.LandmarkWeight(id, 1.0f)),
                        offsetFW = floatArrayOf(0f, 0f, 0f)
                    )
                )
            )
            val out = FloatArray(3)
            singleResolver.resolveAnchor("lm_$id", faceState, out)
            return out
        }

        val lm10FW = resolveSingleLandmarkFW(10)   // Hairline center reference (#10): Y ~ +0.495 FW
        val lm333FW = resolveSingleLandmarkFW(333) // Mid-forehead left reference (#333): Y ~ +0.368 FW
        val lm334FW = resolveSingleLandmarkFW(334) // Upper eyebrow arch peak on #333 side (#334): Y ~ +0.289 FW
        val lm282FW = resolveSingleLandmarkFW(282) // Lower eyebrow arch on #333 side (#282): Y ~ +0.250 FW

        assertEquals(0.0f, lm10FW[0], 1e-4f)
        assertEquals(0.495089f, lm10FW[1], 1e-3f)
        assertEquals(0.368393f, lm333FW[1], 1e-3f)
        assertEquals(0.289391f, lm334FW[1], 1e-3f)
        assertEquals(0.250189f, lm282FW[1], 1e-3f)

        // Package test specifications:
        // - sunflower_forehead: unscaled radius = 0.1936 FW (~0.19 FW), scale = 0.9 -> scaledRadius = 0.1742 FW
        // - periwinkle_forehead: unscaled radius = 0.11 FW, scale = 1.0 -> scaledRadius = 0.11 FW
        data class FlowerPackageExpectation(
            val pkgName: String,
            val unscaledRadiusFW: Float,
            val scaledRadiusFW: Float,
            val expectedCenterFW: FloatArray
        )

        val packageExpectations = listOf(
            FlowerPackageExpectation(
                pkgName = "sunflower_forehead",
                unscaledRadiusFW = 0.19f,
                scaledRadiusFW = 0.1936f * 0.9f, // 0.17424 FW
                expectedCenterFW = floatArrayOf(-0.239526f, 0.583658f, 0.443658f)
            ),
            FlowerPackageExpectation(
                pkgName = "periwinkle_forehead",
                unscaledRadiusFW = 0.11f,
                scaledRadiusFW = 0.11f * 1.0f, // 0.11 FW
                expectedCenterFW = floatArrayOf(-0.239526f, 0.513658f, 0.433658f)
            )
        )

        val viewW = 1080f
        val viewH = 1920f
        val frontProjMatrix = FacePoseAdapter.createCameraProjection(fovYCropDeg, cropAspect, frontFacing = true)
        val screenCenter = FacePoseAdapter.projectFaceLocalToView(
            0f, 0f, 0f, frontalPoseMatrix, fwMeters, frontProjMatrix, viewW, viewH
        )

        for (spec in packageExpectations) {
            val (_, manifest) = readManifestMap(spec.pkgName)
            val resolver = buildAnchorResolverFromManifest(manifest)
            val layer = manifest["layer"] as Map<String, Any?>
            val layerAnchorName = layer["anchor"] as String
            val layerOffsetFW = toFloatList(layer["offsetFW"])

            val anchorResolvedFW = FloatArray(3)
            resolver.resolveAnchor(layerAnchorName, faceState, anchorResolvedFW)

            // Final resolved flower center in face-local FW space = resolved anchor + layer.offsetFW
            val centerFW = floatArrayOf(
                anchorResolvedFW[0] + layerOffsetFW[0],
                anchorResolvedFW[1] + layerOffsetFW[1],
                anchorResolvedFW[2] + layerOffsetFW[2]
            )

            assertEquals("${spec.pkgName} center X (FW)", spec.expectedCenterFW[0], centerFW[0], 1e-3f)
            assertEquals("${spec.pkgName} center Y (FW)", spec.expectedCenterFW[1], centerFW[1], 1e-3f)
            assertEquals("${spec.pkgName} center Z (FW)", spec.expectedCenterFW[2], centerFW[2], 1e-3f)

            // 1. Above the eyebrow reference (#334 upper eyebrow arch on #333 side, and #282) by at least
            //    flower radius + 0.10 FW (tested against BOTH unscaled radius and scaled radius):
            val minCenterAboveEyebrowUnscaled = lm334FW[1] + spec.unscaledRadiusFW + 0.10f
            val minCenterAboveEyebrowScaled = lm334FW[1] + spec.scaledRadiusFW + 0.10f
            assertTrue(
                "${spec.pkgName}: flower center Y (${centerFW[1]}) must be >= eyebrow #334 (${lm334FW[1]}) + unscaledRadius (${spec.unscaledRadiusFW}) + 0.10 = $minCenterAboveEyebrowUnscaled",
                centerFW[1] >= minCenterAboveEyebrowUnscaled - 1e-4f
            )
            assertTrue(
                "${spec.pkgName}: flower center Y (${centerFW[1]}) must be >= eyebrow #334 (${lm334FW[1]}) + scaledRadius (${spec.scaledRadiusFW}) + 0.10 = $minCenterAboveEyebrowScaled",
                centerFW[1] >= minCenterAboveEyebrowScaled - 1e-4f
            )

            val bottomEdgeFW = centerFW[1] - spec.scaledRadiusFW
            val topEdgeFW = centerFW[1] + spec.scaledRadiusFW
            val clearanceAboveUpperBrow = bottomEdgeFW - lm334FW[1]
            val clearanceAboveLowerBrow = bottomEdgeFW - lm282FW[1]
            assertTrue(
                "${spec.pkgName}: bottom edge ($bottomEdgeFW) must stay at least 0.10 FW above upper eyebrow #334 (${lm334FW[1]}), got $clearanceAboveUpperBrow",
                clearanceAboveUpperBrow >= 0.10f
            )
            assertTrue(
                "${spec.pkgName}: bottom edge ($bottomEdgeFW) must stay at least 0.10 FW above lower eyebrow #282 (${lm282FW[1]}), got $clearanceAboveLowerBrow",
                clearanceAboveLowerBrow >= 0.10f
            )
            assertTrue(
                "${spec.pkgName}: bottom edge ($bottomEdgeFW) must be above mid-forehead #333 (${lm333FW[1]})",
                bottomEdgeFW >= lm333FW[1]
            )

            // 2. Inside the forehead band (below hairline reference #10 level plus allowed bloom overlap <= radius,
            //    with bottom edge strictly inside the forehead below #10 and top edge overlapping #10):
            val maxAllowedCenterY = lm10FW[1] + spec.scaledRadiusFW
            assertTrue(
                "${spec.pkgName}: center Y (${centerFW[1]}) must be below hairline #10 (${lm10FW[1]}) + allowed overlap (${spec.scaledRadiusFW}) = $maxAllowedCenterY",
                centerFW[1] <= maxAllowedCenterY
            )
            assertTrue(
                "${spec.pkgName}: bottom edge ($bottomEdgeFW) must be inside the forehead below hairline #10 (${lm10FW[1]})",
                bottomEdgeFW < lm10FW[1]
            )
            assertTrue(
                "${spec.pkgName}: top edge ($topEdgeFW) must touch/overlap hairline #10 (${lm10FW[1]})",
                topEdgeFW > lm10FW[1]
            )

            // 3. Subject-left placement:
            //    - localX < 0 in face-local FW space
            //    - X_cam > 0 in unmirrored camera space (same side as #454)
            //    - screenX < screenCenterX (screen-left) in mirrored front-camera preview
            assertTrue("${spec.pkgName}: localX (${centerFW[0]}) must be negative (subject-left)", centerFW[0] < -0.15f)

            val camPos = FacePoseAdapter.faceLocalToCamera(
                localX = centerFW[0],
                localY = centerFW[1],
                localZ = centerFW[2],
                matrix = frontalPoseMatrix,
                fw = fwMeters
            )
            assertTrue(
                "${spec.pkgName}: X_cam (${camPos[0]}) must be > 0 in unmirrored camera space (#454 subject-left side)",
                camPos[0] > 0.02f
            )

            val screenPos = FacePoseAdapter.projectFaceLocalToView(
                centerFW[0],
                centerFW[1],
                centerFW[2],
                frontalPoseMatrix,
                fwMeters,
                frontProjMatrix,
                viewW,
                viewH
            )
            assertTrue(
                "${spec.pkgName}: screenX (${screenPos[0]}) must be on screen-left (< ${screenCenter[0]}) in mirrored front preview",
                screenPos[0] < screenCenter[0]
            )
        }
    }

    /**
     * Regression test verifying that when a face moves to or past the right/left screen edge
     * (`x234 <= 0.01f`, `x234 < 0f`, `x454 >= 0.99f`, `x454 > 1f`), `FacePoseAdapter.computeFaceWidth`
     * and `SignalPipeline` keep `faceWidth` consistent (~0.14m..0.16m) instead of dropping the
     * camera-frustum `wCam`/`hCam` conversion and blowing up ~4x to ~0.50m.
     */
    @Test
    fun movingFaceToScreenEdgeOrProfile_keepsFaceWidthAndFlowerScaleConsistentWithoutBlowUp() {
        val fovYCropDeg = 60.0f
        val cropAspect = 9f / 16f
        val fCrop = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)
        val absZ = 0.42f
        val wCam = 2.0f * absZ * cropAspect / fCrop
        val hCam = 2.0f * absZ / fCrop

        val trueFw = 0.14f
        val normSpanX = trueFw / wCam // ~0.513 in normalized crop width

        val poseMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, -absZ, 1f
        )

        // Test horizontal face center shifts from center (0.50) all the way to and past both borders
        // including where x234 < 0.01 / < 0.0 and where x454 > 0.99 / > 1.0
        val centerOffsetsU = listOf(0.50f, 0.35f, 0.26f, 0.25f, 0.18f, 0.10f, 0.65f, 0.74f, 0.75f, 0.82f, 0.90f)
        for (centerU in centerOffsetsU) {
            val lm = FloatArray(478 * 3) { idx -> if (idx % 3 == 2) 0f else 0.50f }
            val u234 = centerU - normSpanX * 0.5f
            val u454 = centerU + normSpanX * 0.5f
            lm[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3] = u234
            lm[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3 + 1] = 0.52f
            lm[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3] = u454
            lm[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3 + 1] = 0.52f
            lm[FacePoseAdapter.LANDMARK_FOREHEAD * 3] = centerU
            lm[FacePoseAdapter.LANDMARK_FOREHEAD * 3 + 1] = 0.52f - (0.16f / hCam) * 0.5f
            lm[FacePoseAdapter.LANDMARK_CHIN * 3] = centerU
            lm[FacePoseAdapter.LANDMARK_CHIN * 3 + 1] = 0.52f + (0.16f / hCam) * 0.5f
            lm[33 * 3 + 1] = 0.44f

            val measuredFw = FacePoseAdapter.computeFaceWidth(
                landmarks = lm,
                matrix = poseMatrix,
                imageAspectRatio = cropAspect,
                fovYCropDeg = fovYCropDeg
            )
            assertEquals(
                "At centerU=$centerU (u234=$u234, u454=$u454), computeFaceWidth must stay ~0.14m without 4x edge jump",
                trueFw,
                measuredFw,
                0.015f
            )
        }
    }
}
