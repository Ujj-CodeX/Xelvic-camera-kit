package com.example.engine.pkg

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * FaceRegions (Flow 2 — Chunk R1):
 *
 * Single authoritative registry of facial region landmark loops and control points extracted and
 * verified directly against the repository's pinned MediaPipe 468-vertex topology
 * ([UvRegistry.CANONICAL_UVS] and [UvRegistry.CANONICAL_TRIANGLES]) plus the 10 refined iris
 * landmarks (`468..477` in [UvRegistry.SNAPSHOT_TOPOLOGY_478_ID]).
 *
 * Regions provided:
 * - [FACE_OVAL]: 36-vertex closed outer boundary loop of the 468-vertex canonical mesh.
 * - [RIGHT_EYE_CONTOUR]: 16-vertex subject-right eye opening loop (`#33` outer -> lower lid -> `#133` inner -> upper lid).
 * - [LEFT_EYE_CONTOUR]: 16-vertex subject-left eye opening loop (`#362` inner -> lower lid -> `#263` outer -> upper lid).
 * - [LIPS_OUTER]: 20-vertex outer lip contour loop (`#61` right corner -> lower -> `#291` left corner -> upper).
 * - [MOUTH_INNER]: 20-vertex inner mouth aperture loop (`#78` right inner corner -> lower -> `#308` left inner corner -> upper).
 * - [RIGHT_EYEBROW]: 10-vertex subject-right eyebrow closed loop.
 * - [LEFT_EYEBROW]: 10-vertex subject-left eyebrow closed loop.
 * - [RIGHT_IRIS]: 5-landmark right iris (`#468` center + `#469..#472` ring).
 * - [LEFT_IRIS]: 5-landmark left iris (`#473` center + `#474..#477` ring).
 */
object FaceRegions {

    /**
     * Closed 36-vertex outer perimeter loop of the 468-vertex canonical face mesh,
     * starting at top midline forehead (`#10`), traversing subject-left cheek (`#454`),
     * bottom chin (`#152`), and subject-right cheek (`#234`).
     * Every consecutive edge in this loop is a boundary edge of [UvRegistry.CANONICAL_TRIANGLES].
     */
    val FACE_OVAL: IntArray = intArrayOf(
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    )

    /**
     * Subject-right eye opening contour (16 vertices):
     * `#33` (outer corner) -> lower eyelid (`7, 163, 144, 145, 153, 154, 155`) ->
     * `#133` (inner corner) -> upper eyelid (`173, 157, 158, 159, 160, 161, 246`).
     */
    val RIGHT_EYE_CONTOUR: IntArray = UvRegistry.RIGHT_EYE_LANDMARKS.clone()

    /**
     * Subject-left eye opening contour (16 vertices):
     * `#362` (inner corner) -> lower eyelid (`382, 381, 380, 374, 373, 390, 249`) ->
     * `#263` (outer corner) -> upper eyelid (`466, 388, 387, 386, 385, 384, 398`).
     */
    val LEFT_EYE_CONTOUR: IntArray = UvRegistry.LEFT_EYE_LANDMARKS.clone()

    /**
     * Outer lip contour (20 vertices):
     * `#61` (subject-right corner) -> lower outer lip -> `#291` (subject-left corner) -> upper outer lip.
     */
    val LIPS_OUTER: IntArray = UvRegistry.LIPS_OUTER_LANDMARKS.clone()

    /**
     * Inner mouth aperture contour (20 vertices):
     * `#78` (subject-right inner corner) -> lower inner lip (`#14` bottom center) ->
     * `#308` (subject-left inner corner) -> upper inner lip (`#13` top center).
     */
    val MOUTH_INNER: IntArray = UvRegistry.MOUTH_INTERIOR_LANDMARKS.clone()

    /**
     * Subject-right eyebrow closed contour loop (10 vertices: upper arch -> inner tip -> lower arch).
     */
    val RIGHT_EYEBROW: IntArray = intArrayOf(
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    )

    /**
     * Subject-left eyebrow closed contour loop (10 vertices: upper arch -> inner tip -> lower arch).
     */
    val LEFT_EYEBROW: IntArray = intArrayOf(
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276
    )

    /**
     * Subject-right iris landmarks (MediaPipe 478 topology):
     * `#468` is the iris center; `#469, #470, #471, #472` are the 4 cardinal perimeter points.
     */
    const val RIGHT_IRIS_CENTER: Int = 468
    val RIGHT_IRIS_RING: IntArray = intArrayOf(469, 470, 471, 472)
    val RIGHT_IRIS: IntArray = intArrayOf(468, 469, 470, 471, 472)

    /**
     * Subject-left iris landmarks (MediaPipe 478 topology):
     * `#473` is the iris center; `#474, #475, #476, #477` are the 4 cardinal perimeter points.
     */
    const val LEFT_IRIS_CENTER: Int = 473
    val LEFT_IRIS_RING: IntArray = intArrayOf(474, 475, 476, 477)
    val LEFT_IRIS: IntArray = intArrayOf(473, 474, 475, 476, 477)

    /**
     * Closed 12-vertex nasal bridge contour loop (`#168` glabella -> right bridge flank -> `#4` supratip ->
     * `#1` nose tip -> left bridge flank -> `#168`), verified against [UvRegistry.CANONICAL_TRIANGLES].
     */
    val NOSE_BRIDGE: IntArray = intArrayOf(
        168, 193, 122, 196, 3, 51, 4, 281, 248, 419, 351, 417
    )

    /**
     * Closed 8-vertex subject-right cheek contour loop (`#116, #117, #118, #101, #36, #205, #187, #123`).
     */
    val RIGHT_CHEEK: IntArray = intArrayOf(
        116, 117, 118, 101, 36, 205, 187, 123
    )

    /**
     * Closed 8-vertex subject-left cheek contour loop (`#345, #346, #347, #330, #266, #425, #411, #352`),
     * bilateral mirror of [RIGHT_CHEEK].
     */
    val LEFT_CHEEK: IntArray = intArrayOf(
        345, 346, 347, 330, 266, 425, 411, 352
    )

    /**
     * Data-driven table of named landmark loops (Flow 3 — Chunk R1 Generalized).
     * Allows manifests and composable [com.example.engine.render.RegionMaskPass.MaskExpr] definitions
     * to reference any facial region loop by canonical name without code changes.
     */
    val NAMED_LOOPS: Map<String, IntArray> = linkedMapOf(
        "face_oval" to FACE_OVAL,
        "right_eye" to RIGHT_EYE_CONTOUR,
        "left_eye" to LEFT_EYE_CONTOUR,
        "lips_outer" to LIPS_OUTER,
        "lips_inner" to MOUTH_INNER,
        "mouth_inner" to MOUTH_INNER,
        "right_eyebrow" to RIGHT_EYEBROW,
        "left_eyebrow" to LEFT_EYEBROW,
        "right_iris" to RIGHT_IRIS,
        "left_iris" to LEFT_IRIS,
        "nose_bridge" to NOSE_BRIDGE,
        "right_cheek" to RIGHT_CHEEK,
        "left_cheek" to LEFT_CHEEK
    )

    /**
     * Resolves a named landmark loop (case-insensitive, normalizing `-` to `_`) from [NAMED_LOOPS].
     */
    fun getRegionLoop(name: String): IntArray? {
        val key = name.trim().lowercase().replace('-', '_')
        return NAMED_LOOPS[key]
    }

    /**
     * Key eyelid vertical opening landmark pairs for rapid aperture ratio checks:
     * - Right eye: upper lid `#159`, lower lid `#145`, corners `#33` & `#133`
     * - Left eye: upper lid `#386`, lower lid `#374`, corners `#362` & `#263`
     * - Inner mouth: upper inner lip `#13`, lower inner lip `#14`, corners `#78` & `#308`
     */
    const val RIGHT_EYE_UPPER_LID = 159
    const val RIGHT_EYE_LOWER_LID = 145
    const val RIGHT_EYE_OUTER_CORNER = 33
    const val RIGHT_EYE_INNER_CORNER = 133

    const val LEFT_EYE_UPPER_LID = 386
    const val LEFT_EYE_LOWER_LID = 374
    const val LEFT_EYE_INNER_CORNER = 362
    const val LEFT_EYE_OUTER_CORNER = 263

    const val MOUTH_INNER_UPPER = 13
    const val MOUTH_INNER_LOWER = 14
    const val MOUTH_INNER_RIGHT_CORNER = 78
    const val MOUTH_INNER_LEFT_CORNER = 308

    /**
     * Additional nasal bridge (`#6`) and nasal alar wing (`#129`, `#358`) anchor landmarks used by
     * `PostRenderer` (`post.warp@1`) so all warp control points are 1-Euro smoothed in lockstep
     * with cheek, jaw, eye, and chin contour landmarks.
     */
    val WARP_ANCHOR_LANDMARKS: IntArray = intArrayOf(6, 129, 358)

    /**
     * Sorted, deduplicated list of all landmark indices used across all [FaceRegions] contours,
     * iris circles, and [WARP_ANCHOR_LANDMARKS]. Used by `ControlPointSmoother` so 1-Euro smoothing
     * filters only the active region/warp control points rather than all 478 landmarks.
     */
    val CONTROL_POINT_INDICES: IntArray by lazy {
        val set = java.util.TreeSet<Int>()
        for (id in FACE_OVAL) set.add(id)
        for (id in RIGHT_EYE_CONTOUR) set.add(id)
        for (id in LEFT_EYE_CONTOUR) set.add(id)
        for (id in LIPS_OUTER) set.add(id)
        for (id in MOUTH_INNER) set.add(id)
        for (id in RIGHT_EYEBROW) set.add(id)
        for (id in LEFT_EYEBROW) set.add(id)
        for (id in RIGHT_IRIS) set.add(id)
        for (id in LEFT_IRIS) set.add(id)
        for (id in NOSE_BRIDGE) set.add(id)
        for (id in RIGHT_CHEEK) set.add(id)
        for (id in LEFT_CHEEK) set.add(id)
        for (id in WARP_ANCHOR_LANDMARKS) set.add(id)
        set.toIntArray()
    }

    data class RegionVerificationResult(
        val isValid: Boolean,
        val errorReason: String? = null,
        val boundaryEdgeMatchRatioOval: Float = 0f,
        val meshEdgeConnectivityRatio: Float = 0f
    ) {
        companion object {
            fun success(ovalBoundaryRatio: Float, connectivityRatio: Float) =
                RegionVerificationResult(true, null, ovalBoundaryRatio, connectivityRatio)

            fun fail(reason: String) =
                RegionVerificationResult(false, reason, 0f, 0f)
        }
    }

    /**
     * Extracts all undirected mesh edges `(min(u, v), max(u, v))` and their triangle incidence counts
     * from [triangles] (`2694` indices = `898` triangles).
     * Edges with incidence count == 1 are topological boundary edges of the 468-vertex mesh.
     */
    @JvmOverloads
    fun extractMeshEdgeIncidence(
        triangles: ShortArray = UvRegistry.CANONICAL_TRIANGLES
    ): Map<Long, Int> {
        val edgeCounts = HashMap<Long, Int>(triangles.size)
        val triCount = triangles.size / 3
        for (t in 0 until triCount) {
            val i0 = triangles[t * 3].toInt() and 0xFFFF
            val i1 = triangles[t * 3 + 1].toInt() and 0xFFFF
            val i2 = triangles[t * 3 + 2].toInt() and 0xFFFF
            incrementEdge(edgeCounts, i0, i1)
            incrementEdge(edgeCounts, i1, i2)
            incrementEdge(edgeCounts, i2, i0)
        }
        return edgeCounts
    }

    private fun incrementEdge(map: HashMap<Long, Int>, a: Int, b: Int) {
        val key = edgeKey(a, b)
        map[key] = (map[key] ?: 0) + 1
    }

    fun edgeKey(a: Int, b: Int): Long {
        val lo = min(a, b).toLong()
        val hi = max(a, b).toLong()
        return (lo shl 32) or (hi and 0xFFFFFFFFL)
    }

    /**
     * Verifies all [FaceRegions] landmark lists against the repository's pinned topology
     * ([UvRegistry.CANONICAL_UVS] and [UvRegistry.CANONICAL_TRIANGLES]):
     * 1. Range & uniqueness of indices in each loop.
     * 2. Topological connectivity of consecutive loop vertices in [UvRegistry.CANONICAL_TRIANGLES]
     *    (1-hop or 2-hop mesh edge adjacency) and outer boundary membership of [FACE_OVAL].
     * 3. Bilateral UV symmetry between right and left contours (`RIGHT_EYE_CONTOUR` vs `LEFT_EYE_CONTOUR`,
     *    `RIGHT_EYEBROW` vs `LEFT_EYEBROW`, and left/right sides of `FACE_OVAL`, `LIPS_OUTER`, `MOUTH_INNER`).
     * 4. Anatomical UV hierarchy (`v(BROW) < v(EYE) < v(LIPS)` and `MOUTH_INNER` strictly nested inside `LIPS_OUTER`).
     * 5. Iris landmark indices (`468..477`) validity for 478-landmark snapshots.
     */
    @JvmOverloads
    fun verifyAgainstPinnedTopology(
        uvs: FloatArray = UvRegistry.CANONICAL_UVS,
        triangles: ShortArray = UvRegistry.CANONICAL_TRIANGLES
    ): RegionVerificationResult {
        if (uvs.size != UvRegistry.VERTEX_COUNT * 2) {
            return RegionVerificationResult.fail("Invalid UV array size: ${uvs.size}")
        }
        if (triangles.size != UvRegistry.INDEX_COUNT) {
            return RegionVerificationResult.fail("Invalid triangle index count: ${triangles.size}")
        }

        val meshLoops = mapOf(
            "FACE_OVAL" to FACE_OVAL,
            "RIGHT_EYE_CONTOUR" to RIGHT_EYE_CONTOUR,
            "LEFT_EYE_CONTOUR" to LEFT_EYE_CONTOUR,
            "LIPS_OUTER" to LIPS_OUTER,
            "MOUTH_INNER" to MOUTH_INNER,
            "RIGHT_EYEBROW" to RIGHT_EYEBROW,
            "LEFT_EYEBROW" to LEFT_EYEBROW,
            "NOSE_BRIDGE" to NOSE_BRIDGE,
            "RIGHT_CHEEK" to RIGHT_CHEEK,
            "LEFT_CHEEK" to LEFT_CHEEK
        )

        // 1. Validate index bounds and uniqueness within each contour loop
        for ((name, loop) in meshLoops) {
            if (loop.size < 6) {
                return RegionVerificationResult.fail("Region $name has too few vertices (${loop.size})")
            }
            val seen = HashSet<Int>(loop.size)
            for (idx in loop) {
                if (idx !in 0 until UvRegistry.VERTEX_COUNT) {
                    return RegionVerificationResult.fail("Region $name contains out-of-range vertex index $idx")
                }
                if (!seen.add(idx)) {
                    return RegionVerificationResult.fail("Region $name contains duplicate vertex index $idx")
                }
            }
        }

        // 2. Build mesh edge incidence and adjacency sets from CANONICAL_TRIANGLES
        val edgeIncidence = extractMeshEdgeIncidence(triangles)
        val neighbors = Array(UvRegistry.VERTEX_COUNT) { HashSet<Int>() }
        for (key in edgeIncidence.keys) {
            val u = (key ushr 32).toInt()
            val v = (key and 0xFFFFFFFFL).toInt()
            if (u in 0 until UvRegistry.VERTEX_COUNT && v in 0 until UvRegistry.VERTEX_COUNT) {
                neighbors[u].add(v)
                neighbors[v].add(u)
            }
        }

        // Check FACE_OVAL boundary edge incidence in CANONICAL_TRIANGLES
        var ovalBoundaryEdges = 0
        for (i in FACE_OVAL.indices) {
            val a = FACE_OVAL[i]
            val b = FACE_OVAL[(i + 1) % FACE_OVAL.size]
            val count = edgeIncidence[edgeKey(a, b)] ?: 0
            if (count == 1) {
                ovalBoundaryEdges++
            }
        }
        val ovalBoundaryRatio = ovalBoundaryEdges.toFloat() / FACE_OVAL.size.toFloat()
        if (ovalBoundaryRatio < 0.90f) {
            return RegionVerificationResult.fail(
                "FACE_OVAL boundary edge match ratio too low ($ovalBoundaryRatio); contour does not match canonical mesh perimeter"
            )
        }

        // Check topological connectivity (1-hop or 2-hop in CANONICAL_TRIANGLES) across all mesh loops
        var totalLoopEdges = 0
        var connectedLoopEdges = 0
        for ((name, loop) in meshLoops) {
            for (i in loop.indices) {
                val a = loop[i]
                val b = loop[(i + 1) % loop.size]
                totalLoopEdges++
                val isOneHop = neighbors[a].contains(b)
                val isTwoHop = isOneHop || neighbors[a].any { mid -> neighbors[mid].contains(b) }
                if (isTwoHop) {
                    connectedLoopEdges++
                } else {
                    return RegionVerificationResult.fail(
                        "Region $name edge ($a -> $b) is disconnected in UvRegistry.CANONICAL_TRIANGLES"
                    )
                }
            }
        }
        val connectivityRatio = connectedLoopEdges.toFloat() / totalLoopEdges.coerceAtLeast(1).toFloat()

        // 3. Bilateral symmetry in UvRegistry.CANONICAL_UVS
        // 3a. Right eye vs Left eye (RIGHT_EYE_CONTOUR starts at outer corner #33;
        // LEFT_EYE_CONTOUR starts at inner corner #362 with outer corner #263 at index 8,
        // so bilateral mirror of right[i] is left[(8 - i + 16) % 16])
        if (RIGHT_EYE_CONTOUR.size != LEFT_EYE_CONTOUR.size) {
            return RegionVerificationResult.fail("Eye contour size mismatch")
        }
        val eyeLen = RIGHT_EYE_CONTOUR.size
        for (i in RIGHT_EYE_CONTOUR.indices) {
            val rIdx = RIGHT_EYE_CONTOUR[i]
            val lIdx = LEFT_EYE_CONTOUR[(8 - i + eyeLen) % eyeLen]
            val ru = uvs[rIdx * 2]; val rv = uvs[rIdx * 2 + 1]
            val lu = uvs[lIdx * 2]; val lv = uvs[lIdx * 2 + 1]
            if (ru >= 0.5f || lu <= 0.5f) {
                return RegionVerificationResult.fail(
                    "Eye contour side violation at index $i: right #$rIdx u=$ru, left #$lIdx u=$lu"
                )
            }
            if (abs((ru + lu) - 1.0f) > 0.015f || abs(rv - lv) > 0.015f) {
                return RegionVerificationResult.fail(
                    "Eye contour bilateral UV symmetry violation for pair (#$rIdx, #$lIdx): uSum=${ru + lu}, vDiff=${abs(rv - lv)}"
                )
            }
        }

        // 3b. Right eyebrow vs Left eyebrow
        if (RIGHT_EYEBROW.size != LEFT_EYEBROW.size) {
            return RegionVerificationResult.fail("Eyebrow contour size mismatch")
        }
        for (i in RIGHT_EYEBROW.indices) {
            val rIdx = RIGHT_EYEBROW[i]
            val lIdx = LEFT_EYEBROW[i]
            val ru = uvs[rIdx * 2]; val rv = uvs[rIdx * 2 + 1]
            val lu = uvs[lIdx * 2]; val lv = uvs[lIdx * 2 + 1]
            if (ru >= 0.5f || lu <= 0.5f) {
                return RegionVerificationResult.fail(
                    "Eyebrow contour side violation at index $i: right #$rIdx u=$ru, left #$lIdx u=$lu"
                )
            }
            if (abs((ru + lu) - 1.0f) > 0.015f || abs(rv - lv) > 0.015f) {
                return RegionVerificationResult.fail(
                    "Eyebrow contour bilateral UV symmetry violation for pair (#$rIdx, #$lIdx): uSum=${ru + lu}, vDiff=${abs(rv - lv)}"
                )
            }
        }

        // 3c. Right cheek vs Left cheek
        if (RIGHT_CHEEK.size != LEFT_CHEEK.size) {
            return RegionVerificationResult.fail("Cheek contour size mismatch")
        }
        for (i in RIGHT_CHEEK.indices) {
            val rIdx = RIGHT_CHEEK[i]
            val lIdx = LEFT_CHEEK[i]
            val ru = uvs[rIdx * 2]; val rv = uvs[rIdx * 2 + 1]
            val lu = uvs[lIdx * 2]; val lv = uvs[lIdx * 2 + 1]
            if (ru >= 0.5f || lu <= 0.5f) {
                return RegionVerificationResult.fail(
                    "Cheek contour side violation at index $i: right #$rIdx u=$ru, left #$lIdx u=$lu"
                )
            }
            if (abs((ru + lu) - 1.0f) > 0.02f || abs(rv - lv) > 0.02f) {
                return RegionVerificationResult.fail(
                    "Cheek contour bilateral UV symmetry violation for pair (#$rIdx, #$lIdx): uSum=${ru + lu}, vDiff=${abs(rv - lv)}"
                )
            }
        }

        // 3d. Nose bridge horizontal centering around u = 0.5
        var noseMeanU = 0f
        for (idx in NOSE_BRIDGE) {
            noseMeanU += uvs[idx * 2]
        }
        noseMeanU /= NOSE_BRIDGE.size.toFloat()
        if (abs(noseMeanU - 0.5f) > 0.02f) {
            return RegionVerificationResult.fail("NOSE_BRIDGE horizontal mean U ($noseMeanU) not centered at 0.5")
        }

        // 4. Anatomical hierarchy & nesting in UV space
        val rBrowMeanV = meanV(RIGHT_EYEBROW, uvs)
        val lBrowMeanV = meanV(LEFT_EYEBROW, uvs)
        val rEyeMeanV = meanV(RIGHT_EYE_CONTOUR, uvs)
        val lEyeMeanV = meanV(LEFT_EYE_CONTOUR, uvs)
        val lipsOuterMeanV = meanV(LIPS_OUTER, uvs)
        val mouthInnerMeanV = meanV(MOUTH_INNER, uvs)

        if (!(rBrowMeanV < rEyeMeanV && lBrowMeanV < lEyeMeanV && rEyeMeanV < lipsOuterMeanV)) {
            return RegionVerificationResult.fail(
                "Anatomical vertical ordering violation: browV=$rBrowMeanV, eyeV=$rEyeMeanV, lipsV=$lipsOuterMeanV"
            )
        }
        if (abs(lipsOuterMeanV - mouthInnerMeanV) > 0.02f) {
            return RegionVerificationResult.fail(
                "MOUTH_INNER vertical center ($mouthInnerMeanV) misaligned with LIPS_OUTER ($lipsOuterMeanV)"
            )
        }

        // Verify MOUTH_INNER bounding box is strictly inside LIPS_OUTER bounding box
        val outerBounds = uvBounds(LIPS_OUTER, uvs)
        val innerBounds = uvBounds(MOUTH_INNER, uvs)
        if (!(innerBounds[0] > outerBounds[0] &&
              innerBounds[1] > outerBounds[1] &&
              innerBounds[2] < outerBounds[2] &&
              innerBounds[3] < outerBounds[3])
        ) {
            return RegionVerificationResult.fail(
                "MOUTH_INNER UV bounds ${innerBounds.contentToString()} not strictly nested inside LIPS_OUTER ${outerBounds.contentToString()}"
            )
        }

        // 5. Verify iris landmark indices (468..477)
        for (id in RIGHT_IRIS) {
            if (id !in UvRegistry.VERTEX_COUNT until UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS) {
                return RegionVerificationResult.fail("Invalid RIGHT_IRIS landmark index $id")
            }
        }
        for (id in LEFT_IRIS) {
            if (id !in UvRegistry.VERTEX_COUNT until UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS) {
                return RegionVerificationResult.fail("Invalid LEFT_IRIS landmark index $id")
            }
        }

        return RegionVerificationResult.success(
            ovalBoundaryRatio = ovalBoundaryRatio,
            connectivityRatio = connectivityRatio
        )
    }

    /**
     * Computes the normalized inner-mouth opening ratio (`verticalInnerOpening / horizontalInnerWidth`)
     * from 468/478 landmarks (`#13` upper inner lip, `#14` lower inner lip, `#78` right corner, `#308` left corner).
     * Zero heap allocations.
     */
    fun computeMouthOpeningRatio(landmarks: FloatArray?): Float {
        if (landmarks == null || landmarks.size < UvRegistry.VERTEX_COUNT * 3) return 0f
        val x13 = landmarks[MOUTH_INNER_UPPER * 3]
        val y13 = landmarks[MOUTH_INNER_UPPER * 3 + 1]
        val x14 = landmarks[MOUTH_INNER_LOWER * 3]
        val y14 = landmarks[MOUTH_INNER_LOWER * 3 + 1]
        val x78 = landmarks[MOUTH_INNER_RIGHT_CORNER * 3]
        val y78 = landmarks[MOUTH_INNER_RIGHT_CORNER * 3 + 1]
        val x308 = landmarks[MOUTH_INNER_LEFT_CORNER * 3]
        val y308 = landmarks[MOUTH_INNER_LEFT_CORNER * 3 + 1]

        val vert = hypot(x14 - x13, y14 - y13)
        val horiz = hypot(x308 - x78, y308 - y78).coerceAtLeast(1e-4f)
        return (vert / horiz).coerceIn(0f, 2f)
    }

    private fun meanV(indices: IntArray, uvs: FloatArray): Float {
        var sum = 0f
        for (idx in indices) {
            sum += uvs[idx * 2 + 1]
        }
        return sum / indices.size.coerceAtLeast(1).toFloat()
    }

    private fun uvBounds(indices: IntArray, uvs: FloatArray): FloatArray {
        var minU = Float.POSITIVE_INFINITY
        var minV = Float.POSITIVE_INFINITY
        var maxU = Float.NEGATIVE_INFINITY
        var maxV = Float.NEGATIVE_INFINITY
        for (idx in indices) {
            val u = uvs[idx * 2]
            val v = uvs[idx * 2 + 1]
            if (u < minU) minU = u
            if (v < minV) minV = v
            if (u > maxU) maxU = u
            if (v > maxV) maxV = v
        }
        return floatArrayOf(minU, minV, maxU, maxV)
    }
}
