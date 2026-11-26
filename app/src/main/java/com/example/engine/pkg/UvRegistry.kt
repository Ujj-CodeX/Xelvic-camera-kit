package com.example.engine.pkg

import com.example.engine.perception.FaceGeometrySnapshot
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Base64
import java.util.zip.ZipInputStream
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * UvRegistry: Single authoritative registry for the pinned MediaPipe 468-point face-mesh UV layout,
 * tessellation topology, anatomical region masks, and task-bundle SHA-256 verification
 * (README Section 6, 11.2, 19.2 & 21 Step 8).
 *
 * Enforces:
 * - Pinned task bundle SHA-256 (`PINNED_MODEL_SHA256`) matching `app/src/main/assets/face_landmarker.task`.
 * - Pinned 468-vertex UV layout (`CANONICAL_UVS`) and 898-triangle tessellation (`CANONICAL_TRIANGLES`)
 *   extracted directly from `geometry_pipeline_metadata_landmarks.binarypb` inside the pinned task bundle.
 * - Independent landmark-to-UV correspondence verification (midline symmetry, vertical/horizontal anatomical
 *   ordering, bilateral symmetry).
 * - Strict rejection of legacy unverified canonical OBJ UV layouts (MediaPipe issue #5760 ordering incompatibility),
 *   mismatched model SHA-256 hashes, mismatched topologies, or UV arrays that only match vertex count.
 */
object UvRegistry {

    const val TOPOLOGY_ID = "mediapipe_canonical_468_v1"
    const val UV_LAYOUT_ID = TOPOLOGY_ID
    const val SNAPSHOT_TOPOLOGY_478_ID = "canonical_478_v1"
    const val VERTEX_COUNT = 468
    const val TRACKER_LANDMARK_COUNT_WITH_IRIS = 478
    const val TRACKER_LANDMARK_COUNT = TRACKER_LANDMARK_COUNT_WITH_IRIS
    const val TRIANGLE_COUNT = 898
    const val INDEX_COUNT = TRIANGLE_COUNT * 3 // 2694 uint16 indices

    /**
     * Exact SHA-256 of the pinned `app/src/main/assets/face_landmarker.task` bundle.
     */
    const val PINNED_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

    /**
     * Exact SHA-256 of the 468-vertex canonical UV float buffer (936 big-endian IEEE-754 floats)
     * extracted from `geometry_pipeline_metadata_landmarks.binarypb` in `face_landmarker.task`.
     */
    const val EXPECTED_UV_LAYOUT_SHA256 = "f94390be44716e3c4f06383feb8152028883c2fe6c59ebe96b3b55851fab401f"

    /**
     * Exact SHA-256 of the 898-triangle canonical index buffer (2694 big-endian uint16 shorts)
     * extracted from `geometry_pipeline_metadata_landmarks.binarypb` in `face_landmarker.task`.
     */
    const val EXPECTED_TRIANGLES_SHA256 = "111a06ca10dc394df8d1ce0c35e89cadf4cee61f5268a7d74b27de576ffa1ebf"

    private val SUPPORTED_UV_LAYOUT_IDS = setOf(
        TOPOLOGY_ID,
        "canonical_468_v1",
        "mediapipe_468_v1",
        "default"
    )

    private val SUPPORTED_RUNTIME_TOPOLOGY_IDS = setOf(
        TOPOLOGY_ID,
        SNAPSHOT_TOPOLOGY_478_ID,
        "canonical_468_v1"
    )

    // Known legacy OBJ UV identifiers rejected per MediaPipe issue #5760
    val REJECTED_LEGACY_OBJ_IDS = setOf(
        "legacy_canonical_obj",
        "canonical_face_model.obj",
        "canonical_face_model_obj",
        "mediapipe_legacy_obj",
        "obj_uv_v1",
        "unverified_obj"
    )
    val REJECTED_LEGACY_UV_LAYOUT_IDS = REJECTED_LEGACY_OBJ_IDS

    // Anatomical landmark indices for protected regions (eyes, lips, mouth interior)
    val RIGHT_EYE_LANDMARKS = intArrayOf(
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    )
    val LEFT_EYE_LANDMARKS = intArrayOf(
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    )
    val LIPS_OUTER_LANDMARKS = intArrayOf(
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    )
    val MOUTH_INTERIOR_LANDMARKS = intArrayOf(
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
    )

    data class UvVerificationResult(
        val isValid: Boolean,
        val errorReason: String? = null
    ) {
        companion object {
            fun success() = UvVerificationResult(true, null)
            fun fail(reason: String) = UvVerificationResult(false, reason)
        }
    }

    data class TaskBundleGeometry(
        val modelSha256: String,
        val vertexCount: Int,
        val triangleCount: Int,
        val canonicalPositions: FloatArray, // 468 * 3
        val canonicalUvs: FloatArray,       // 468 * 2
        val canonicalTriangles: ShortArray  // 2694
    )

    // Base64-encoded 936 IEEE-754 big-endian floats from geometry_pipeline_metadata_landmarks.binarypb
    private const val CANONICAL_UVS_B64 =
        "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"

    // Base64-encoded 2694 big-endian uint16 triangle indices from geometry_pipeline_metadata_landmarks.binarypb
    private const val CANONICAL_TRIANGLES_B64 =
        "AK0AmwCFAPYAIQAHAX4BjgFqAQcB0gD5ATQBnwFEAE4AXwC/AWQBhQEIAH8AIgCiAXABCAGFAIsAogAiAQsAAAEuACUASAAAAAsB" +
        "LgAAAAsAAABIAV0BwwFeAHgAeQDnAcQBXgHDAOgA5wB5AQsBLgENACUAJwBIAS8BDQEuAEkASAAnAWUBVwFeAIAAeQByARUBXgFX" +
        "AC8AcgB5AV4BxAFlAHkAgADoAcUBZQHEAOkA6ACAASsBTQEpAEUAQwBoAUwBKQFNAGcAaABDAK8AmAGMAK8AqwCYAXkBjACYAJQA" +
        "mACrAX0BgAF+AJoAmwCdAY4BfgGAAK0AnQCbARgBWwFKADIAZQB2AVwBSgFbAHcAdgBlAQ0BLwEOACcAKABJATABDgEvAEoASQAo" +
        "AAkBUACXAAkAlwBrAVEAlwFQAGwAawCXAVgBFgFoAHMAgwAwARcBaAEWADEAMACDAQYBrwGiACAAwgDTAagBogGvAMwA0wDCATAB" +
        "mAEOAEoAKAC4AZkBDgGYALkAuAAoARABNgGXACoAtwBQAZ8BlwE2AL8AUAC3AUIBDgGaAFwAugAoAZkBmgEOALkAKAC6AVsBwQFc" +
        "AHYAdwDlAcIBXAHBAOYA5QB3AbIBsAGuANYA0gDUAaYBrgGwAMoA1ADSATkBOgASAFMAEgBUABEAEgE6ABEAVAASATMBdwEyAE0A" +
        "TACSASMBMgF3AD0AkgBMAQMBgwEEAB0AHgCgAYQBBAGDAKEAoAAeAR4BngGAADgAnQC+AY4BgAGeAK0AvgCdAaIBqAGWAMIAtgDM" +
        "AU8BlgGoAGoAzAC2AW8BoAFsAIoAhwDAAbIBbAGgANYAwACHAYcBpwFHAKUAYgDLAWYBRwGnAIEAywBiASoBLQEcAEQANgBHAPsB" +
        "HAEtABUARwA2AAQBEwAFAAQABQAtARkABQETADMALQAFAP4BdQD9ABgAFwCQAXYA/QF1AJEAkAAXAUABQQEzAFoATQBbAXcBMwFB" +
        "AJIAWwBNARgBqQGbADIAuwDNAasBmwGpAM8AzQC7AaUBOQDIAMkAyABTABIAyAE5ABIAUwDIAU8BQQGWAGoAtgBbAZUBlgFBALUA" +
        "WwC2AZUBQQGUALUAtABbAUABlAFBAFoAWwC0ABEBOgAQABEAEABUATsAEAE6AFUAVAAQAakBCgGqAM0AzgAkAacBqgEKAMsAJADO" +
        "AXEBjAGQAIwAsACrAXkBkAGMAJQAqwCwAYcBDQFCAKUAXAAnAQ4BQgENACgAJwBcAaEB0QGdAMEAvQD1AdABnQHRAPQA9QC9AQEB" +
        "AgGCABsAnwAcAYEBggECAJ4AHACfAQQBhAHTAB4A9wChAdIB0wGEAPYAoQD3APgByAGjAAMAxADsAY8BowHIAK4A7ADEAU0BKgFM" +
        "AGgAZwBEARwBTAEqADYARABnAR0ACAGhADcAwQAIAKgBoQAIAKgACADBAVQBBQFaAG8AdQAfAcABWgEFAOQAHwB1AR0BoQG5ADcA" +
        "3QDBAZ0BuQGhAL0AwQDdAUcBzAFGAGIAYQDwAUgBRgHMAGMA8ABhARUBYwFJAC8AZAB+AXMBSQFjAI4AfgBkATUBiAG2AE8A2gCm" +
        "AbcBtgGIANsApgDaAX0BfgEAAJoAGgCbAVUBAAF+AHAAmwAaAWgBFwGkAIMAxgAxAa0BpAEXANEAMQDGAW0BbAF7AIgAlgCHAYoB" +
        "ewFsAKkAhwCWAWMBFQG1AH4A2QAvAVcBtQEVAHIALwDZAbsBvAEaAN8ANADgARsBGgG8ADUA4AA0ARkBEwFrADMAhgAtAbgBawET" +
        "ANwALQCGAa8BBgGLANMAqgAgAXEBiwEGAIwAIACqAVEBKwFSAGwAbQBFASkBUgErAEMARQBtAU8BEQFBAGoAWwArAXcBQQERAJIA" +
        "KwBbAVwBwgFdAHcAeADmAcMBXQHCAOcA5gB4AdMBZwFWAPcAcQCCAb4BVgFnAOIAggBxARoBGwFOADQAaQA1ASUBTgEbAD8ANQBp" +
        "APoBygHOABQA8gDuAc0BzgHKAPEA7gDyARQBYQEsAC4ARgB8AX8BLAFhAJwAfABGAUUBJAFEAGAAXwA+ATQBRAEkAE4APgBfARsB" +
        "FAElADUAPwAuASwBJQEUAEYALgA/Ab8BCAFZAOMAdAAiAXQBWQEIAI8AIgB0AWABWQFaAHsAdQB0AVQBWgFZAG8AdAB1AAEAEwES" +
        "AAEALAATAWIBEgATAH0AEwAsAPgBGQHIAAMA7AAzAWsByAEZAIYAMwDsAakBqgGrAM0AzwDOAbQBqwGqANgAzgDPAXwBfQD8AJkA" +
        "FgCaAQAA/AF9ABoAmgAWAYcBiQENAKUAJwCnAQsBDQGJACUApwAnAMcBrADIAMcAyADQAaUAyAGsAMkA0ADIAUoBSQEKAGUAJABk" +
        "AXMBCgFJAI4AZAAkAaYBsAERAMoAKwDUAR8BEQGwADkA1AArASIA+gFIADwAYwAUAc4BSAD6APIAFABjAQIBHgGBABwAngA4AYAB" +
        "gQEeAJ0AOACeAVYBvgFhAHEAfADiAQkBYQG+ACMA4gB8AQEBggEDABsAHQCfAYMBAwGCAKAAnwAdAa4BpgGvANIA0wDKAagBrwGm" +
        "AMwAygDTAb0BVgEUAOEALgBxAWEBFAFWAHwAcQAuAagBpgFPAMwAagDKAREBTwGmACsAygBqATIBJAEzAEwATQA+AUUBMwEkAGAA" +
        "PgBNAW4BvwFgAIkAewDjAVkBYAG/AHQA4wB7AS4BDAEvAEgASQAmAQ8BLwEMACkAJgBJAXMBZgEKAI4AJACBAacBCgFmAMsAgQAk" +
        "AUcBJgHMAGIA8ABAAccBzAEmAOsAQADwASYBSwEWAEAAMABmARcBFgFLADEAZgAwAS8BDwEwAEkASgApARABMAEPACoAKQBKAasB" +
        "tAGyAM8A1gDYAbABsgG0ANQA2ADWATABEAGYAEoAuAAqAZcBmAEQALcAKgC4AYoBrgGLAKkAqgDSAa8BiwGuANMA0gCqAYsBcQF6" +
        "AKoAlQCMAZABegFxALAAjACVASgBTgErAEIARQBpAU0BKwFOAGgAaQBFAaEAqAFfAMEAegCoAAYBXwCoAAYAqAB6ARgBmwFgADIA" +
        "ewC7AXgBYAGbAJMAuwB7AT8BQAFFAFkAYABaATMBRQFAAE0AWgBgAR0BJwFQADcAawBBASgBUAEnAEIAQQBrAZQBQAGTALQAswBa" +
        "AT8BkwFAAFkAWgCzAUoBXAFJAGUAZAB3AV0BSQFcAHgAdwBkAU4BJQFNAGkAaAA/ASoBTQElAEQAPwBoAUMBxgFuAF0AiQDqAb8B" +
        "bgHGAOMA6gCJABABOwAPABAADwBVATwADwE7AFYAVQAPAa0BFwFmANEAgQAxAUsBZgEXAGYAMQCBAA8BPAAOAA8ADgBWAT0ADgE8" +
        "AFcAVgAOAAgBHQAJAAgACQA3AVAACQEdAGsANwAJAUkBXQEVAGQALwB4AV4BFQFdAHkAeAAvAPwA/QF8ABYAmQAXAXYBfAD9AJEA" +
        "FwCZAZIBkwE+ALIAWACzAT8BPgGTAFkAswBYAV8ABgGjAHoAxAAGAMUBowAGAMUABgDEAUQBPgFFAF8AYABYAT8BRQE+AFkAWABg" +
        "AY0BbwFtAKwAiACKAWwBbQFvAIcAigCIASABswGNADoArADXAW8BjQGzAIoA1wCsAbYBtwFYANoAcwDbARYBWAG3ADAA2wBzAQ8B" +
        "NwEQACkAKgBRATYBEAE3AFAAUQAqAAUBGQDDAAUAwwAzAPgAwwEZAAMAMwDDAREBHwF3ACsAkgA5ASMBdwEfAD0AOQCSAYwBrACv" +
        "AKsArwDQAMcArwGsAMcA0ACvAQwBOAEPACYAKQBSATcBDwE4AFEAUgApAbwBvQEbAOAANQDhARQBGwG9AC4A4QA1AP4BUwF1ABgA" +
        "kABuAYYBdQFTAKMAbgCQAScBGgEoAEEAQgA0AU4BKAEaAGkANABCAVoBwAFbAHUAdgDkAcEBWwHAAOUA5AB2AcYBZAG/AOoA4wB/" +
        "AQgBvwFkACIAfwDjAVABKAFRAGsAbABCASsBUQEoAEUAQgBsAJcBUQAKAJcACgBsAVIACgFRAG0AbAAKARYBtwEmADAAQADbAccB" +
        "JgG3AOsA2wBAAZcBnwEkALcAPgC/ATQBJAGfAE4AvwA+AWYBcwGtAIEA0QCOAWMBrQFzAH4AjgDRAVkBdAFUAHQAbwCPAQkBVAF0" +
        "ACMAjwBvAYQBhgHSAKEA9gCjAPkB0gGGAAcAowD2AWABWgEYAHsAMgB1AVsBGAFaAHYAdQAyAScBugEaAEEANADeAbsBGgG6AN8A" +
        "3gA0ABMAXgFiABMAfQBeAXIBYgBeAI0AXgB9AScBHQG6AEEA3gA3AbkBugEdAN0ANwDeAaMAxQD4AMQAAwDFAMMA+ADFAMMAxQAD" +
        "AWcBBwD/AIIAGQAhAPkA/wEHAAcAIQAZARMBEgG4AC0A3AAsAckBuAESAO0ALADcASwBfwEtAEYARwCcAXABLQF/AIsAnABHAaEB" +
        "XwHRAMEA9QB6AZwB0QFfALwAegD1AdIBBwHTAPYA9wAhAWcB0wEHAIIAIQD3AYUA+wFwAKIAiwAVAS0BcAD7AEcAFQCLAXYBggF8" +
        "AJEAmQCfAYEBfAGCAJ4AnwCZAXsBigF6AJYAlQCpAYsBegGKAKoAqQCVAV8BowGcAHoAvADEAY8BnAGjAK4AxAC8AaoBQgG0AM4A" +
        "2ABcAZoBtAFCALoAXADYAYMBdQGEAKAAoQCQAYYBhAF1AKMAkAChAYkBRgCkAKcApABhAAIApAFGAAIAYQCkAWIBcgHNAH0A8QCN" +
        "Ac4BzQFyAPIAjQDxAAABCwCkAAAApAAlAYkApAELAKcAJQCkAAsADAEuAAsASAAMAQwBLgAMACYADABIAYIBdgGDAJ8AoACRAXUB" +
        "gwF2AJAAkQCgAAwADQEMAAwAJgANATgBDAANAFIADQAmASUBLAEqAD8ARABGAS0BKgEsAEcARgBEAVQBCQEFAG8AHwAjAb4BBQEJ" +
        "AOIAIwAfAXwBgQF9AJkAmgCeAYABfQGBAJ0AngCaARgBSgGpADIAzQBlAQoBqQFKACQAZQDNAacBhwGqAMsAzgClAUIBqgGHAFwA" +
        "pQDOAa0BYwGkANEAxgB+AbUBpAFjANkAfgDGAYcBRwGJAKUApwBiAUYBiQFHAGEAYgCnAckBtgG4AO0A3ADaAVgBuAG2AHMA2gDc" +
        "AX4BagFVAJsAcACFAc8BVQFqAPMAhQBwAckBzQHLAO0A7wDxAcoBywHNAO4A8QDvAbIBrgFsANYAhwDSAYoBbAGuAKkA0gCHAZ4B" +
        "zwGOAL4ArQDzAWoBjgHPAIUA8wCtAQYBrAFxACAAjADQAYwBcQGsAKsA0ACMAckBEgHNAO0A8QAsAWIBzQESAH0ALADxATwBkwE9" +
        "AFYAVwCzAZIBPQGTALIAswBXATsBlAE8AFUAVgC0AZMBPAGUALMAtABWAToBlQE7AFQAVQC1AZQBOwGVALQAtQBVATkBlgE6AFMA" +
        "VAC2AZUBOgGWALUAtgBUAaIBlgGlAMIAyQC2ATkBpQGWAFMAtgDJAW4BkQFDAIkAXQCxAWkBQwGRAIQAsQBdAZgBlwEyALgATAC3" +
        "ASQBMgGXAD4AtwBMAZgBMgGZALgAuQBMASMBmQEyAD0ATAC5AZoBmQEfALoAOQC5ASMBHwGZAD0AuQA5AbQBmgGwANgA1AC6AR8B" +
        "sAGaADkAugDUAbIBoAGrANYAzwDAAZsBqwGgALsAwADPAQgBcAF0ACIAjwCLAX8BdAFwAJwAiwCPAckBywG2AO0A2gDvATUBtgHL" +
        "AE8A7wDaAWABeAFuAHsAiQCTAZEBbgF4ALEAkwCJAAQAAQETAAQALQABARIBEwABACwAAQAtAawBBgGlANAAyQAgAaIBpQEGAMIA" +
        "IADJAUcBZgEmAGIAQACBAUsBJgFmAGYAgQBAAW8BswGgAIoAwADXAbEBoAGzANUA1wDAAccBtwEhAOsAOwDbAYgBIQG3AKYA2wA7" +
        "AUgBzgFGAGMAYQDyAXIBRgHOAI0A8gBhAUYBcgACAGEAAgCNAF4AAgFyAF4AjQACAcwBxwExAPAASwDrASEBMQHHADsA6wBLAcAB" +
        "UwHBAOQA5QBuAP4BwQFTABgAbgDlAQUBvgD/AB8AGQDiAWcA/wG+AIIA4gAZAcEA/gHCAOUA5gAYAP0BwgD+ABcAGADmAcIA/QHD" +
        "AOYA5wAXAPwBwwD9ABYAFwDnAcMA/AHEAOcA6AAWAQABxAD8ABoAFgDoAQABVQHEABoA6ABwAcUBxAFVAOkAcADoAZ0B0AGeAL0A" +
        "vgD0Ac8BngHQAPMA9AC+AbkBnQEeAN0AOAC9AZ4BHgGdAL4AvQA4AbkBHgG6AN0A3gA4AQIBugEeABwAOADeAboBAgG7AN4A3wAc" +
        "AQEBuwECABsAHADfAbwBuwEDAOAAHQDfAQEBAwG7ABsA3wAdAQMBBAG8AB0A4AAeAb0BvAEEAOEAHgDgAQQB0wG9AB4A4QD3AVYB" +
        "vQHTAHEA9wDhAPoBNQHKABQA7gBPAcsBygE1AO8ATwDuASIBMQGIADwApgBLASEBiAExADsASwCmAcwBMQFIAPAAYwBLASIBSAEx" +
        "ADwASwBjAXgBsQGRAJMAsQDVAbMBkQGxANcA1QCxAPoBIgE1ABQATwA8AYgBNQEiAKYAPABPAZsBoAF4ALsAkwDAAbEBeAGgANUA" +
        "wACTAVUBzwHFAHAA6QDzAdABxQHPAPQA8wDpAcUB0AFlAOkAgAD0AdEBZQHQAPUA9ACAAZwBVwHRALwA9QByAWUB0QFXAIAAcgD1" +
        "AbUBVwGPANkArgByAZwBjwFXALwAcgCuAWsBuAFoAIYAgwDcAVgBaAG4AHMA3ACDAcgBpAGPAOwArgDGAbUBjwGkANkAxgCuAcgB" +
        "awGkAOwAxgCGAWgBpAFrAIMAhgDGAWkBkQEgAIQAOgCxAbMBIAGRANcAsQA6AWEBCQF/AHwAnAAjAXQBfwEJAI8AIwCcAP8A+QFT" +
        "ABkAbgAHAYYBUwD5AKMABwBuAQUA/wHAAB8A5AAZAVMBwAD/AG4AGQDkAA4BPQANAA4ADQBXATgADQE9AFIAVwANAT0BkgE4AFcA" +
        "UgCyATcBOAGSAFEAsgBSAZIBPgE3ALIAUQBYATYBNwE+AFAAWABRAT4BRAE2AFgAUABfAZ8BNgFEAL8AXwBQ"

    /**
     * Pinned verified canonical UV coordinates for 468 vertices [u0, v0, u1, v1, ...].
     */
    val CANONICAL_UVS: FloatArray by lazy {
        val bytes = Base64.getDecoder().decode(CANONICAL_UVS_B64)
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).asFloatBuffer()
        val arr = FloatArray(VERTEX_COUNT * 2)
        fb.get(arr)
        arr
    }

    /**
     * Pinned verified face-mesh triangle indices (2694 uint16 shorts = 898 triangles).
     */
    val CANONICAL_TRIANGLES: ShortArray by lazy {
        val bytes = Base64.getDecoder().decode(CANONICAL_TRIANGLES_B64)
        val sb = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).asShortBuffer()
        val arr = ShortArray(INDEX_COUNT)
        sb.get(arr)
        arr
    }

    /**
     * Static per-vertex anatomical protection weights [0..1] for the 468 vertices.
     * Protected regions (right eye, left eye, mouth interior, and inner lip opening) have weight 0.0f.
     */
    val VERTEX_PROTECTION_WEIGHTS: FloatArray by lazy {
        val weights = FloatArray(VERTEX_COUNT) { 1.0f }
        for (idx in RIGHT_EYE_LANDMARKS) {
            if (idx in 0 until VERTEX_COUNT) weights[idx] = 0.0f
        }
        for (idx in LEFT_EYE_LANDMARKS) {
            if (idx in 0 until VERTEX_COUNT) weights[idx] = 0.0f
        }
        for (idx in MOUTH_INTERIOR_LANDMARKS) {
            if (idx in 0 until VERTEX_COUNT) weights[idx] = 0.0f
        }
        for (idx in LIPS_OUTER_LANDMARKS) {
            if (idx in 0 until VERTEX_COUNT) weights[idx] = 0.0f
        }
        weights
    }

    val UV_LAYOUT_SHA256: String by lazy {
        computeUvSha256(CANONICAL_UVS)
    }

    val TOPOLOGY_TRIANGLES_SHA256: String by lazy {
        computeTrianglesSha256(CANONICAL_TRIANGLES)
    }

    @JvmOverloads
    fun computeUvSha256(uvs: FloatArray = CANONICAL_UVS): String {
        val md = MessageDigest.getInstance("SHA-256")
        val byteBuf = ByteBuffer.allocate(uvs.size * 4).order(ByteOrder.BIG_ENDIAN)
        for (f in uvs) byteBuf.putFloat(f)
        md.update(byteBuf.array())
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun computeUvLayoutSha256(uvs: FloatArray = CANONICAL_UVS): String = computeUvSha256(uvs)

    @JvmOverloads
    fun computeTrianglesSha256(triangles: ShortArray = CANONICAL_TRIANGLES): String {
        val md = MessageDigest.getInstance("SHA-256")
        val byteBuf = ByteBuffer.allocate(triangles.size * 2).order(ByteOrder.BIG_ENDIAN)
        for (s in triangles) byteBuf.putShort(s)
        md.update(byteBuf.array())
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun computeBytesSha256(bytes: ByteArray): String {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(bytes)
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun getVertexUv(vertexIndex: Int, out: FloatArray = FloatArray(2)): FloatArray {
        require(vertexIndex in 0 until VERTEX_COUNT) {
            "Vertex index out of range: $vertexIndex (expected 0 until $VERTEX_COUNT)"
        }
        out[0] = CANONICAL_UVS[vertexIndex * 2]
        out[1] = CANONICAL_UVS[vertexIndex * 2 + 1]
        return out
    }

    fun isSupportedUvLayoutId(uvLayoutId: String?): Boolean {
        if (uvLayoutId.isNullOrBlank()) return true
        if (uvLayoutId in REJECTED_LEGACY_OBJ_IDS) return false
        return uvLayoutId in SUPPORTED_UV_LAYOUT_IDS
    }

    fun isProtectedVertex(vertexIndex: Int): Boolean {
        if (vertexIndex !in 0 until VERTEX_COUNT) return false
        return VERTEX_PROTECTION_WEIGHTS[vertexIndex] < 0.5f
    }

    /**
     * Evaluates the continuous UV-space anatomical region protection mask in [0.0, 1.0].
     * Mirrors the exact elliptical smoothstep protection math used in `FS_FACE_TEXTURE` so both
     * shader rendering and deterministic CPU/JVM conformance tests agree.
     *
     * Returns 0.0 inside protected eye/lip/mouth-interior cores and 1.0 on paintable skin regions.
     */
    fun evaluateRegionProtectionMask(
        u: Float,
        v: Float,
        protectEyes: Boolean = true,
        protectLips: Boolean = true,
        protectMouthInterior: Boolean = true
    ): Float {
        if (!u.isFinite() || !v.isFinite() || u < 0f || u > 1f || v < 0f || v > 1f) return 0f

        var weight = 1.0f

        if (protectEyes) {
            // Right eye center (0.3496, 0.3780), radii (0.076, 0.032) covering #33..#133 and #159..#145
            val rx = (u - 0.3496f) / 0.076f
            val ry = (v - 0.3780f) / 0.032f
            val rDist = sqrt(rx * rx + ry * ry)
            val rEyeMask = smoothstep(0.88f, 1.15f, rDist)
            weight = minOf(weight, rEyeMask)

            // Left eye center (0.6504, 0.3780), radii (0.076, 0.032) covering #362..#263 and #386..#374
            val lx = (u - 0.6504f) / 0.076f
            val ly = (v - 0.3780f) / 0.032f
            val lDist = sqrt(lx * lx + ly * ly)
            val lEyeMask = smoothstep(0.88f, 1.15f, lDist)
            weight = minOf(weight, lEyeMask)
        }

        if (protectMouthInterior) {
            // Inner mouth aperture center (0.5000, 0.6939), radii (0.098, 0.025)
            val mx = (u - 0.5000f) / 0.098f
            val my = (v - 0.6939f) / 0.025f
            val mDist = sqrt(mx * mx + my * my)
            val mouthMask = smoothstep(0.85f, 1.15f, mDist)
            weight = minOf(weight, mouthMask)
        }

        if (protectLips) {
            // Outer lips contour center (0.5000, 0.6937), radii (0.130, 0.048) covering #61..#291 and #0..#17
            val lx = (u - 0.5000f) / 0.130f
            val ly = (v - 0.6937f) / 0.048f
            val lDist = sqrt(lx * lx + ly * ly)
            val lipMask = smoothstep(0.88f, 1.15f, lDist)
            weight = minOf(weight, lipMask)
        }

        return weight.coerceIn(0f, 1f)
    }

    private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
        val t = ((x - edge0) / (edge1 - edge0).coerceAtLeast(1e-5f)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    /**
     * Verifies that the provided task model SHA-256 matches the pinned MediaPipe `face_landmarker.task` bundle.
     */
    fun verifyModelSha256(modelSha256: String?): UvVerificationResult {
        if (modelSha256.isNullOrBlank()) {
            return UvVerificationResult.fail("Missing task model SHA-256")
        }
        if (!modelSha256.equals(PINNED_MODEL_SHA256, ignoreCase = true)) {
            return UvVerificationResult.fail(
                "Task model SHA-256 mismatch: expected $PINNED_MODEL_SHA256, got $modelSha256"
            )
        }
        return UvVerificationResult.success()
    }

    /**
     * Verifies landmark/vertex ordering and anatomical UV correspondence across the 468 vertices.
     * Never assumes UV compatibility based solely on vertex count (468).
     * Explicitly detects and rejects legacy canonical OBJ UV ordering (MediaPipe issue #5760),
     * V-flipped OBJ UVs, synthetic grid UVs, and permuted vertex orderings.
     */
    @JvmOverloads
    fun verifyLandmarkUvCorrespondence(uvs: FloatArray = CANONICAL_UVS): UvVerificationResult {
        if (uvs.size != VERTEX_COUNT * 2) {
            return UvVerificationResult.fail(
                "UV array length mismatch: expected ${VERTEX_COUNT * 2} floats for $VERTEX_COUNT vertices, got ${uvs.size}"
            )
        }

        for (i in uvs.indices) {
            val v = uvs[i]
            if (!v.isFinite() || v < 0.0f || v > 1.0f) {
                return UvVerificationResult.fail(
                    "Invalid UV coordinate at float index $i (vertex ${i / 2}): $v (must be finite in [0.0, 1.0])"
                )
            }
        }

        fun uOf(lm: Int): Float = uvs[lm * 2]
        fun vOf(lm: Int): Float = uvs[lm * 2 + 1]

        // 1. Midline anatomical landmarks must lie along u = 0.5000
        val midlineLandmarks = intArrayOf(1, 10, 13, 14, 152)
        for (lm in midlineLandmarks) {
            val u = uOf(lm)
            if (abs(u - 0.5f) > 0.008f) {
                return UvVerificationResult.fail(
                    "Anatomical midline landmark #$lm has incompatible U=$u (expected ~0.5000; possible legacy OBJ UV ordering #5760)"
                )
            }
        }

        // 2. Vertical anatomical ordering (V increases from forehead to chin in MediaPipe canonical UV layout)
        // Forehead (#10: ~0.1071) < Eyes (#33/#263: ~0.3756) < Nose tip (#1: ~0.5475) < Lips (#13/#14: ~0.6923) < Chin (#152: ~0.9545)
        val vForehead = vOf(10)
        val vRightEye = vOf(33)
        val vLeftEye = vOf(263)
        val vNoseTip = vOf(1)
        val vUpperLip = vOf(13)
        val vLowerLip = vOf(14)
        val vChin = vOf(152)

        if (!(vForehead < vRightEye && vRightEye < vNoseTip && vNoseTip < vUpperLip && vUpperLip <= vLowerLip && vLowerLip < vChin)) {
            return UvVerificationResult.fail(
                "Incompatible vertical landmark-to-UV ordering (vForehead=$vForehead, vEye=$vRightEye, vNose=$vNoseTip, vLip=$vUpperLip, vChin=$vChin); legacy OBJ V-flip or ordering mismatch (#5760)"
            )
        }

        // 3. Horizontal anatomical ordering (U increases from subject-right cheek #234 to subject-left cheek #454)
        val uRightCheek = uOf(234)
        val uRightEyeOuter = uOf(33)
        val uRightEyeInner = uOf(133)
        val uLeftEyeInner = uOf(362)
        val uLeftEyeOuter = uOf(263)
        val uLeftCheek = uOf(454)

        if (!(uRightCheek < uRightEyeOuter &&
              uRightEyeOuter < uRightEyeInner &&
              uRightEyeInner < 0.5f &&
              0.5f < uLeftEyeInner &&
              uLeftEyeInner < uLeftEyeOuter &&
              uLeftEyeOuter < uLeftCheek)
        ) {
            return UvVerificationResult.fail(
                "Incompatible horizontal landmark-to-UV ordering (u234=$uRightCheek, u33=$uRightEyeOuter, u133=$uRightEyeInner, u362=$uLeftEyeInner, u263=$uLeftEyeOuter, u454=$uLeftCheek)"
            )
        }

        // 4. Bilateral anatomical symmetry across right/left landmark pairs
        val symmetricPairs = arrayOf(
            234 to 454, // cheeks
            33 to 263,  // outer eyes
            133 to 362, // inner eyes
            61 to 291,  // mouth corners
            50 to 280   // mid cheeks
        )
        for ((rightIdx, leftIdx) in symmetricPairs) {
            val uSum = uOf(rightIdx) + uOf(leftIdx)
            val vDiff = abs(vOf(rightIdx) - vOf(leftIdx))
            if (abs(uSum - 1.0f) > 0.01f || vDiff > 0.01f) {
                return UvVerificationResult.fail(
                    "Bilateral UV symmetry violation for pair (#$rightIdx, #$leftIdx): uSum=$uSum, vDiff=$vDiff (incompatible UV vertex ordering #5760)"
                )
            }
        }

        return UvVerificationResult.success()
    }

    /**
     * Verifies a UV layout ID, UV coordinate buffer, triangle index buffer, and model SHA-256
     * against the pinned `UvRegistry` specifications.
     */
    @JvmOverloads
    fun verifyUvLayout(
        uvLayoutId: String? = TOPOLOGY_ID,
        candidateUvs: FloatArray = CANONICAL_UVS,
        candidateTriangles: ShortArray = CANONICAL_TRIANGLES,
        modelSha256: String = PINNED_MODEL_SHA256
    ): UvVerificationResult {
        if (!uvLayoutId.isNullOrBlank() && uvLayoutId in REJECTED_LEGACY_OBJ_IDS) {
            return UvVerificationResult.fail(
                "Rejected legacy canonical OBJ UV layout '$uvLayoutId' due to MediaPipe issue #5760 vertex ordering incompatibility"
            )
        }
        if (!isSupportedUvLayoutId(uvLayoutId)) {
            return UvVerificationResult.fail(
                "Unsupported or unverified UV layout ID '$uvLayoutId' (expected '$TOPOLOGY_ID')"
            )
        }

        val shaRes = verifyModelSha256(modelSha256)
        if (!shaRes.isValid) return shaRes

        if (candidateTriangles.size != INDEX_COUNT) {
            return UvVerificationResult.fail(
                "Triangle index count mismatch: expected $INDEX_COUNT indices ($TRIANGLE_COUNT triangles), got ${candidateTriangles.size}"
            )
        }

        for (i in 0 until TRIANGLE_COUNT) {
            val i0 = candidateTriangles[i * 3].toInt()
            val i1 = candidateTriangles[i * 3 + 1].toInt()
            val i2 = candidateTriangles[i * 3 + 2].toInt()
            if (i0 !in 0 until VERTEX_COUNT || i1 !in 0 until VERTEX_COUNT || i2 !in 0 until VERTEX_COUNT) {
                return UvVerificationResult.fail(
                    "Triangle $i has out-of-range vertex index ($i0, $i1, $i2) for $VERTEX_COUNT-vertex topology"
                )
            }
            if (i0 == i1 || i1 == i2 || i0 == i2) {
                return UvVerificationResult.fail(
                    "Triangle $i is degenerate ($i0, $i1, $i2)"
                )
            }
        }

        val corrRes = verifyLandmarkUvCorrespondence(candidateUvs)
        if (!corrRes.isValid) return corrRes

        val actualUvSha = computeUvSha256(candidateUvs)
        if (!actualUvSha.equals(EXPECTED_UV_LAYOUT_SHA256, ignoreCase = true)) {
            return UvVerificationResult.fail(
                "UV layout SHA-256 mismatch: expected $EXPECTED_UV_LAYOUT_SHA256, got $actualUvSha"
            )
        }

        val actualTriSha = computeTrianglesSha256(candidateTriangles)
        if (!actualTriSha.equals(EXPECTED_TRIANGLES_SHA256, ignoreCase = true)) {
            return UvVerificationResult.fail(
                "Topology triangle SHA-256 mismatch: expected $EXPECTED_TRIANGLES_SHA256, got $actualTriSha"
            )
        }

        return UvVerificationResult.success()
    }

    /**
     * Verifies runtime `FaceGeometrySnapshot` compatibility with the pinned 468-point UV layout and model hash.
     */
    fun verifySnapshotCompatibility(snapshot: FaceGeometrySnapshot): UvVerificationResult {
        val shaRes = verifyModelSha256(snapshot.modelSha256)
        if (!shaRes.isValid) return shaRes

        if (snapshot.topologyId !in SUPPORTED_RUNTIME_TOPOLOGY_IDS) {
            return UvVerificationResult.fail(
                "Incompatible snapshot topologyId '${snapshot.topologyId}' (expected one of $SUPPORTED_RUNTIME_TOPOLOGY_IDS)"
            )
        }

        if (snapshot.isTracking) {
            val lmCount = snapshot.landmarkCount
            if (lmCount < VERTEX_COUNT) {
                return UvVerificationResult.fail(
                    "Insufficient tracker landmarks for 468-point face mesh: got $lmCount, need at least $VERTEX_COUNT"
                )
            }
            if (!snapshot.hasFiniteLandmarks()) {
                return UvVerificationResult.fail("Snapshot contains non-finite landmark coordinates")
            }
        }

        return UvVerificationResult.success()
    }

    /**
     * Extracts the canonical 468-vertex mesh (3D positions, UV coordinates, and 898 triangle indices)
     * directly from `geometry_pipeline_metadata_landmarks.binarypb` inside the MediaPipe `.task` bundle
     * and verifies it against the pinned `UvRegistry` hashes and landmark-to-UV correspondence rules.
     */
    fun verifyAgainstTaskBundle(taskFile: File): UvVerificationResult {
        if (!taskFile.exists() || !taskFile.isFile) {
            return UvVerificationResult.fail("Task bundle file does not exist: ${taskFile.path}")
        }
        val bytes = taskFile.readBytes()
        return verifyAgainstTaskBundleBytes(bytes)
    }

    fun verifyAgainstTaskBundleBytes(taskBytes: ByteArray): UvVerificationResult {
        val actualModelSha = computeBytesSha256(taskBytes)
        val modelShaRes = verifyModelSha256(actualModelSha)
        if (!modelShaRes.isValid) return modelShaRes

        val extracted = try {
            extractTaskBundleGeometry(taskBytes)
        } catch (e: Exception) {
            return UvVerificationResult.fail("Failed to extract geometry metadata from task bundle: ${e.message}")
        }

        if (extracted.vertexCount != VERTEX_COUNT) {
            return UvVerificationResult.fail(
                "Task bundle canonical mesh vertex count mismatch: expected $VERTEX_COUNT, got ${extracted.vertexCount}"
            )
        }
        if (extracted.triangleCount != TRIANGLE_COUNT) {
            return UvVerificationResult.fail(
                "Task bundle canonical mesh triangle count mismatch: expected $TRIANGLE_COUNT, got ${extracted.triangleCount}"
            )
        }

        return verifyUvLayout(
            uvLayoutId = TOPOLOGY_ID,
            candidateUvs = extracted.canonicalUvs,
            candidateTriangles = extracted.canonicalTriangles,
            modelSha256 = extracted.modelSha256
        )
    }

    fun extractTaskBundleGeometry(taskFile: File): TaskBundleGeometry {
        return extractTaskBundleGeometry(taskFile.readBytes())
    }

    fun extractTaskBundleGeometry(taskBytes: ByteArray): TaskBundleGeometry {
        val modelSha = computeBytesSha256(taskBytes)
        val pbBytes = findBinaryPbInArchiveBytes(taskBytes, "geometry_pipeline_metadata_landmarks.binarypb")
            ?: throw IllegalArgumentException("Task bundle missing geometry_pipeline_metadata_landmarks.binarypb")

        // Field 1 of EnvironmentMetadata is canonical_mesh (Mesh3d message, wire_type = 2)
        val meshBytes = extractFirstLengthDelimitedField(pbBytes, targetFieldNumber = 1)
            ?: throw IllegalArgumentException("geometry_pipeline_metadata_landmarks.binarypb missing canonical_mesh (field 1)")

        // In Mesh3d: field 3 is repeated float vertex_buffer (X, Y, Z, U, V), field 4 is repeated uint32 index_buffer
        val vertexFloats = ArrayList<Float>(VERTEX_COUNT * 5)
        val indexInts = ArrayList<Int>(INDEX_COUNT)
        parseMesh3dProto(meshBytes, vertexFloats, indexInts)

        if (vertexFloats.size != VERTEX_COUNT * 5) {
            throw IllegalArgumentException(
                "Unexpected Mesh3d vertex_buffer float count: ${vertexFloats.size} (expected ${VERTEX_COUNT * 5})"
            )
        }
        if (indexInts.size != INDEX_COUNT) {
            throw IllegalArgumentException(
                "Unexpected Mesh3d index_buffer count: ${indexInts.size} (expected $INDEX_COUNT)"
            )
        }

        val positions = FloatArray(VERTEX_COUNT * 3)
        val uvs = FloatArray(VERTEX_COUNT * 2)
        for (i in 0 until VERTEX_COUNT) {
            positions[i * 3] = vertexFloats[i * 5]
            positions[i * 3 + 1] = vertexFloats[i * 5 + 1]
            positions[i * 3 + 2] = vertexFloats[i * 5 + 2]
            uvs[i * 2] = vertexFloats[i * 5 + 3]
            uvs[i * 2 + 1] = vertexFloats[i * 5 + 4]
        }

        val triangles = ShortArray(INDEX_COUNT) { idx ->
            val v = indexInts[idx]
            if (v !in 0 until VERTEX_COUNT) {
                throw IllegalArgumentException("Out of range Mesh3d index $v at position $idx")
            }
            v.toShort()
        }

        return TaskBundleGeometry(
            modelSha256 = modelSha,
            vertexCount = VERTEX_COUNT,
            triangleCount = TRIANGLE_COUNT,
            canonicalPositions = positions,
            canonicalUvs = uvs,
            canonicalTriangles = triangles
        )
    }

    /**
     * Searches an uncompressed/compressed ZIP or MediaPipe `.task` / `.tflite` bundle for `targetEntryName`,
     * supporting both top-level entries and nested `.tflite` model archives.
     */
    private fun findBinaryPbInArchiveBytes(archiveBytes: ByteArray, targetEntryName: String): ByteArray? {
        // 1. Try standard ZipInputStream (including nested .tflite archives)
        try {
            ZipInputStream(ByteArrayInputStream(archiveBytes)).use { zis ->
                while (true) {
                    val entry = zis.nextEntry ?: break
                    val entryBytes = zis.readBytes()
                    if (entry.name.endsWith(targetEntryName)) {
                        return entryBytes
                    }
                    if (entry.name.endsWith(".tflite") || entry.name.endsWith(".task") || entry.name.endsWith(".zip")) {
                        val nested = findBinaryPbInArchiveBytes(entryBytes, targetEntryName)
                        if (nested != null) return nested
                    }
                }
            }
        } catch (_: Exception) {
            // Fall through to raw ZIP local-header scan
        }

        // 2. Scan raw ZIP Local File Headers (PK\x03\x04) in case a TFLite flatbuffer precedes the appended ZIP archive
        val nameBytes = targetEntryName.toByteArray(Charsets.UTF_8)
        var i = 0
        while (i + 30 <= archiveBytes.size) {
            if (archiveBytes[i] == 0x50.toByte() &&
                archiveBytes[i + 1] == 0x4B.toByte() &&
                archiveBytes[i + 2] == 0x03.toByte() &&
                archiveBytes[i + 3] == 0x04.toByte()
            ) {
                val bb = ByteBuffer.wrap(archiveBytes, i, 30).order(ByteOrder.LITTLE_ENDIAN)
                val compression = bb.getShort(i + 8).toInt() and 0xFFFF
                val compSize = bb.getInt(i + 18)
                val nameLen = bb.getShort(i + 26).toInt() and 0xFFFF
                val extraLen = bb.getShort(i + 28).toInt() and 0xFFFF
                val nameStart = i + 30
                val dataStart = nameStart + nameLen + extraLen
                if (nameStart + nameLen <= archiveBytes.size && compSize > 0 && dataStart + compSize <= archiveBytes.size) {
                    val entryName = String(archiveBytes, nameStart, nameLen, Charsets.UTF_8)
                    val payload = archiveBytes.copyOfRange(dataStart, dataStart + compSize)
                    val decoded = if (compression == 0) {
                        payload
                    } else {
                        try {
                            java.util.zip.InflaterInputStream(
                                ByteArrayInputStream(payload),
                                java.util.zip.Inflater(true)
                            ).readBytes()
                        } catch (_: Exception) {
                            null
                        }
                    }
                    if (decoded != null) {
                        if (entryName.endsWith(targetEntryName)) {
                            return decoded
                        }
                        if (entryName.endsWith(".tflite") || entryName.endsWith(".task")) {
                            val nested = findBinaryPbInArchiveBytes(decoded, targetEntryName)
                            if (nested != null) return nested
                        }
                    }
                    i = dataStart + compSize
                    continue
                }
            }
            i++
        }
        return null
    }

    private fun extractFirstLengthDelimitedField(buf: ByteArray, targetFieldNumber: Int): ByteArray? {
        var pos = 0
        while (pos < buf.size) {
            val tag = readVarint(buf, pos)
            pos = tag.nextPos
            val fieldNum = (tag.value ushr 3).toInt()
            val wireType = (tag.value and 0x07L).toInt()
            when (wireType) {
                0 -> {
                    val v = readVarint(buf, pos)
                    pos = v.nextPos
                }
                2 -> {
                    val lenVar = readVarint(buf, pos)
                    pos = lenVar.nextPos
                    val length = lenVar.value.toInt()
                    if (fieldNum == targetFieldNumber) {
                        return buf.copyOfRange(pos, pos + length)
                    }
                    pos += length
                }
                5 -> pos += 4
                1 -> pos += 8
                else -> throw IllegalArgumentException("Unsupported protobuf wire type $wireType")
            }
        }
        return null
    }

    private fun parseMesh3dProto(
        buf: ByteArray,
        outVertices: MutableList<Float>,
        outIndices: MutableList<Int>
    ) {
        var pos = 0
        while (pos < buf.size) {
            val tag = readVarint(buf, pos)
            pos = tag.nextPos
            val fieldNum = (tag.value ushr 3).toInt()
            val wireType = (tag.value and 0x07L).toInt()
            when (wireType) {
                0 -> {
                    val v = readVarint(buf, pos)
                    pos = v.nextPos
                    if (fieldNum == 4) {
                        outIndices.add(v.value.toInt())
                    }
                }
                5 -> {
                    val bits = (buf[pos].toInt() and 0xFF) or
                            ((buf[pos + 1].toInt() and 0xFF) shl 8) or
                            ((buf[pos + 2].toInt() and 0xFF) shl 16) or
                            ((buf[pos + 3].toInt() and 0xFF) shl 24)
                    pos += 4
                    if (fieldNum == 3) {
                        outVertices.add(Float.fromBits(bits))
                    }
                }
                2 -> {
                    val lenVar = readVarint(buf, pos)
                    pos = lenVar.nextPos
                    val length = lenVar.value.toInt()
                    val end = pos + length
                    if (fieldNum == 3) {
                        while (pos + 4 <= end) {
                            val bits = (buf[pos].toInt() and 0xFF) or
                                    ((buf[pos + 1].toInt() and 0xFF) shl 8) or
                                    ((buf[pos + 2].toInt() and 0xFF) shl 16) or
                                    ((buf[pos + 3].toInt() and 0xFF) shl 24)
                            pos += 4
                            outVertices.add(Float.fromBits(bits))
                        }
                    } else if (fieldNum == 4) {
                        while (pos < end) {
                            val v = readVarint(buf, pos)
                            pos = v.nextPos
                            outIndices.add(v.value.toInt())
                        }
                    } else {
                        pos = end
                    }
                }
                1 -> pos += 8
                else -> throw IllegalArgumentException("Unsupported Mesh3d wire type $wireType")
            }
        }
    }

    private data class VarintResult(val value: Long, val nextPos: Int)

    private fun readVarint(buf: ByteArray, startPos: Int): VarintResult {
        var pos = startPos
        var result = 0L
        var shift = 0
        while (pos < buf.size) {
            val b = buf[pos++].toInt()
            result = result or ((b.toLong() and 0x7FL) shl shift)
            if ((b and 0x80) == 0) {
                return VarintResult(result, pos)
            }
            shift += 7
        }
        throw IllegalArgumentException("Truncated varint in protobuf stream")
    }
}
