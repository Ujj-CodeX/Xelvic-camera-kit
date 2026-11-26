package com.example.engine.render

import android.opengl.EGL14
import android.opengl.GLES30
import com.example.engine.pkg.FaceRegions
import com.example.engine.pkg.UvRegistry
import com.example.engine.scene.PropertySlots
import com.example.engine.signal.ControlPointSmoother
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * RegionMaskPass (Flow 3 — Chunk R1 Generalized):
 *
 * Composable mask producer and mask algebra engine operating on 1-Euro smoothed control-point
 * landmarks ([ControlPointSmoother]) and named landmark loops ([FaceRegions.NAMED_LOOPS]).
 *
 * Supports:
 * - **Mask Producers**:
 *   - [MaskExpr.LandmarkLoop]: Any named contour loop from [FaceRegions.NAMED_LOOPS] (`"face_oval"`,
 *     `"right_eye"`, `"left_eye"`, `"lips_outer"`, `"mouth_inner"`, `"right_eyebrow"`, `"left_eyebrow"`,
 *     `"nose_bridge"`, `"right_cheek"`, `"left_cheek"`, etc.) with soft radial feathering.
 *   - [MaskExpr.IrisCircle]: Parametric iris disc (`left` or `right`, `radiusScale`, `featherNorm`)
 *     derived from 478 iris landmarks (`#468..#472` / `#473..#477`) or eye contour fallback.
 *   - [MaskExpr.TextureMask]: Static or package-supplied mask asset reference.
 *   - [MaskExpr.LuminanceKey]: Smoothstep luminance key (`minLuma..maxLuma`, `feather`) for
 *     specular highlight / glasses glare suppression or shadow protection.
 * - **Mask Algebra Ops**:
 *   - [MaskExpr.Union]: $\max(A, B)$
 *   - [MaskExpr.Intersect]: $A \times B$
 *   - [MaskExpr.Subtract]: $\max(0, A - B)$
 *   - [MaskExpr.Feather]: Softens edges by `radiusNorm`
 *   - [MaskExpr.Invert]: $1 - A$
 * - **Built-in Canonical Composite Masks** ([BUILTIN_MASKS]):
 *   - `"skin"` (`$mask:skin`): `face_oval - (left_eye ∪ right_eye) - lips_outer - (left_eyebrow ∪ right_eyebrow)`
 *   - `"eyes"` (`$mask:eyes`): `left_eye ∪ right_eye`
 *   - `"mouth"` (`$mask:mouth`): `mouth_inner`
 *   - `"lips"` (`$mask:lips`): `lips_outer - mouth_inner`
 *   - `"iris"` (`$mask:iris`): `(right_iris_circle ∩ right_eye) ∪ (left_iris_circle ∩ left_eye)`
 *   - `"right_iris"` (`$mask:right_iris`), `"left_iris"` (`$mask:left_iris`)
 *   - `"cheeks"` (`$mask:cheeks`): `right_cheek ∪ left_cheek`
 */
class RegionMaskPass {

    enum class RegionType {
        FACE_OVAL,
        RIGHT_EYE,
        LEFT_EYE,
        LIPS_OUTER,
        MOUTH_INNER,
        RIGHT_EYEBROW,
        LEFT_EYEBROW,
        RIGHT_IRIS_CLAMPED,
        LEFT_IRIS_CLAMPED
    }

    /**
     * Generic Debug HUD & Canvas mask viewer modes (`MASK:OFF/EYES/MOUTH/IRIS/SKIN`).
     */
    enum class DebugMaskMode(val maskId: String?) {
        OFF(null),
        EYES("eyes"),
        MOUTH("mouth"),
        IRIS("iris"),
        SKIN("skin");

        fun next(): DebugMaskMode {
            val vals = values()
            return vals[(ordinal + 1) % vals.size]
        }
    }

    /**
     * Composable Mask Expression AST (Flow 3 — Chunk R1 Generalized).
     */
    sealed class MaskExpr {
        data class LandmarkLoop(
            val regionName: String,
            val featherNorm: Float = DEFAULT_FEATHER_NORM
        ) : MaskExpr()

        data class IrisCircle(
            val isLeft: Boolean,
            val radiusScale: Float = 1.0f,
            val featherNorm: Float = 0.006f
        ) : MaskExpr()

        data class TextureMask(
            val assetId: String
        ) : MaskExpr()

        data class LuminanceKey(
            val sourceTarget: String = "\$camera",
            val minLuma: Float = 0.0f,
            val maxLuma: Float = 0.85f,
            val feather: Float = 0.10f
        ) : MaskExpr()

        data class Union(val a: MaskExpr, val b: MaskExpr) : MaskExpr()
        data class Intersect(val a: MaskExpr, val b: MaskExpr) : MaskExpr()
        data class Subtract(val a: MaskExpr, val b: MaskExpr) : MaskExpr()
        data class Feather(val input: MaskExpr, val radiusNorm: Float) : MaskExpr()
        data class Invert(val input: MaskExpr) : MaskExpr()
    }

    companion object {
        const val DEFAULT_FEATHER_NORM = 0.012f
        const val IRIS_CIRCLE_SEGMENTS = 16
        private const val MAX_CONTOUR_POINTS = 64
        // Each contour point produces 1 interior triangle (3 verts) + 2 skirt triangles (6 verts) = 9 verts * 3 floats
        private const val MAX_FLOATS_PER_REGION = MAX_CONTOUR_POINTS * 9 * 3
        private const val MAX_TOTAL_VERTEX_FLOATS = MAX_FLOATS_PER_REGION * 9

        val BUILTIN_EYES_MASK: MaskExpr = MaskExpr.Union(
            MaskExpr.LandmarkLoop("left_eye", 0.008f),
            MaskExpr.LandmarkLoop("right_eye", 0.008f)
        )

        val BUILTIN_MOUTH_MASK: MaskExpr = MaskExpr.LandmarkLoop("mouth_inner", 0.008f)

        val BUILTIN_LIPS_MASK: MaskExpr = MaskExpr.Subtract(
            MaskExpr.LandmarkLoop("lips_outer", 0.008f),
            MaskExpr.LandmarkLoop("mouth_inner", 0.005f)
        )

        val BUILTIN_RIGHT_IRIS_MASK: MaskExpr = MaskExpr.Intersect(
            MaskExpr.IrisCircle(isLeft = false, radiusScale = 1.0f, featherNorm = 0.005f),
            MaskExpr.LandmarkLoop("right_eye", 0.004f)
        )

        val BUILTIN_LEFT_IRIS_MASK: MaskExpr = MaskExpr.Intersect(
            MaskExpr.IrisCircle(isLeft = true, radiusScale = 1.0f, featherNorm = 0.005f),
            MaskExpr.LandmarkLoop("left_eye", 0.004f)
        )

        val BUILTIN_IRIS_MASK: MaskExpr = MaskExpr.Union(
            BUILTIN_RIGHT_IRIS_MASK,
            BUILTIN_LEFT_IRIS_MASK
        )

        val BUILTIN_SKIN_MASK: MaskExpr = MaskExpr.Subtract(
            MaskExpr.Subtract(
                MaskExpr.Subtract(
                    MaskExpr.LandmarkLoop("face_oval", DEFAULT_FEATHER_NORM),
                    BUILTIN_EYES_MASK
                ),
                MaskExpr.LandmarkLoop("lips_outer", 0.008f)
            ),
            MaskExpr.Union(
                MaskExpr.LandmarkLoop("left_eyebrow", 0.008f),
                MaskExpr.LandmarkLoop("right_eyebrow", 0.008f)
            )
        )

        const val FOREHEAD_CRANIAL_EXTENSION_SCALE = 0.22f

        /**
         * Returns the anatomical cranial dome extension weight `[0.0 .. 0.22]` along the
         * chin-to-forehead vector (`#152 -> #10`) for upper forehead and temple transition
         * landmarks on `FACE_OVAL`, extending `$mask:skin` and `face_oval` above MediaPipe's
         * top landmark `#10` by `0.22 * faceHeight` to cover the full upper forehead up to the hairline.
         */
        fun cranialForeheadWeightForLandmark(lmIdx: Int): Float {
            return when (lmIdx) {
                10 -> 0.22f
                338, 109 -> 0.20f
                297, 67 -> 0.16f
                332, 103 -> 0.11f
                284, 54 -> 0.06f
                251, 21 -> 0.025f
                389, 162 -> 0.008f
                else -> 0f
            }
        }

        /**
         * Checks whether a landmark loop index array represents `FACE_OVAL`, matching both
         * reference identity and cloned/deserialized `FACE_OVAL` arrays.
         */
        fun isFaceOvalLoop(indices: IntArray): Boolean {
            return indices === FaceRegions.FACE_OVAL ||
                (indices.size == FaceRegions.FACE_OVAL.size &&
                    indices[0] == 10 &&
                    indices[18] == 152)
        }

        val BUILTIN_CHEEKS_MASK: MaskExpr = MaskExpr.Union(
            MaskExpr.LandmarkLoop("right_cheek", 0.025f),
            MaskExpr.LandmarkLoop("left_cheek", 0.025f)
        )

        val BUILTIN_SCLERA_MASK: MaskExpr = MaskExpr.Subtract(
            MaskExpr.Union(
                MaskExpr.LandmarkLoop("left_eye", 0.005f),
                MaskExpr.LandmarkLoop("right_eye", 0.005f)
            ),
            MaskExpr.Union(
                MaskExpr.IrisCircle(isLeft = true, radiusScale = 1.05f, featherNorm = 0.006f),
                MaskExpr.IrisCircle(isLeft = false, radiusScale = 1.05f, featherNorm = 0.006f)
            )
        )

        val BUILTIN_TEETH_MASK: MaskExpr = MaskExpr.Intersect(
            MaskExpr.LandmarkLoop("mouth_inner", 0.008f),
            MaskExpr.LuminanceKey("\$camera", minLuma = 0.28f, maxLuma = 0.98f, feather = 0.08f)
        )

        val BUILTIN_MASKS: Map<String, MaskExpr> = linkedMapOf(
            "skin" to BUILTIN_SKIN_MASK,
            "eyes" to BUILTIN_EYES_MASK,
            "mouth" to BUILTIN_MOUTH_MASK,
            "lips" to BUILTIN_LIPS_MASK,
            "iris" to BUILTIN_IRIS_MASK,
            "right_iris" to BUILTIN_RIGHT_IRIS_MASK,
            "left_iris" to BUILTIN_LEFT_IRIS_MASK,
            "cheeks" to BUILTIN_CHEEKS_MASK,
            "sclera" to BUILTIN_SCLERA_MASK,
            "teeth" to BUILTIN_TEETH_MASK,
            "nose_bridge" to MaskExpr.LandmarkLoop("nose_bridge", DEFAULT_FEATHER_NORM),
            "face_oval" to MaskExpr.LandmarkLoop("face_oval", DEFAULT_FEATHER_NORM)
        )

        /**
         * Compiles an optional `"masks"` JSON block from an `xfilter/3` manifest into a map of
         * named [MaskExpr] definitions merged on top of [BUILTIN_MASKS].
         */
        fun compileMasksFromJson(masksObj: JSONObject?): Map<String, MaskExpr> {
            val result = LinkedHashMap<String, MaskExpr>(BUILTIN_MASKS)
            if (masksObj == null) return result
            for (maskKey in masksObj.keys()) {
                val cleanKey = maskKey.removePrefix("\$mask:").trim()
                val item = masksObj.opt(maskKey)
                val expr = parseMaskNode(item, result)
                if (expr != null && cleanKey.isNotEmpty()) {
                    result[cleanKey] = expr
                }
            }
            return result
        }

        fun parseMaskExprFromJson(
            json: JSONObject,
            knownMasks: Map<String, MaskExpr> = BUILTIN_MASKS
        ): MaskExpr {
            return parseMaskNode(json, knownMasks)
                ?: throw IllegalArgumentException("Invalid mask expression JSON: $json")
        }

        private fun parseMaskNode(
            raw: Any?,
            knownMasks: Map<String, MaskExpr>
        ): MaskExpr? {
            if (raw == null) return null
            if (raw is String) {
                val key = raw.trim().removePrefix("\$mask:").lowercase().replace('-', '_')
                knownMasks[key]?.let { return it }
                if (FaceRegions.getRegionLoop(key) != null) {
                    return MaskExpr.LandmarkLoop(key, DEFAULT_FEATHER_NORM)
                }
                throw IllegalArgumentException("Unknown mask or region reference '$raw'")
            }
            if (raw is JSONObject) {
                val op = raw.optString("op", raw.optString("type", "landmarkLoop")).trim()
                return when (op.lowercase()) {
                    "landmarkloop", "region", "loop" -> {
                        val region = raw.optString("region", raw.optString("loop", "face_oval"))
                            .trim().lowercase().replace('-', '_')
                        if (FaceRegions.getRegionLoop(region) == null) {
                            throw IllegalArgumentException("Mask landmarkLoop references unknown region '$region'")
                        }
                        val feather = raw.optDouble("feather", DEFAULT_FEATHER_NORM.toDouble()).toFloat()
                        MaskExpr.LandmarkLoop(region, feather)
                    }
                    "iriscircle", "iris" -> {
                        val side = raw.optString("side", "right").trim().lowercase()
                        val isLeft = (side == "left" || side == "l")
                        val scale = raw.optDouble("radiusScale", raw.optDouble("scale", 1.0)).toFloat()
                        val feather = raw.optDouble("feather", 0.006).toFloat()
                        MaskExpr.IrisCircle(isLeft = isLeft, radiusScale = scale, featherNorm = feather)
                    }
                    "texturemask", "texture" -> {
                        val asset = raw.optString("asset", raw.optString("map", "")).trim()
                        if (asset.isEmpty()) {
                            throw IllegalArgumentException("Mask textureMask requires non-empty 'asset'")
                        }
                        MaskExpr.TextureMask(asset)
                    }
                    "luminancekey", "lumakey" -> {
                        val src = raw.optString("source", "\$camera").trim()
                        val minL = raw.optDouble("minLuma", 0.0).toFloat().coerceIn(0f, 1f)
                        val maxL = raw.optDouble("maxLuma", 0.85).toFloat().coerceIn(minL, 1f)
                        val feather = raw.optDouble("feather", 0.10).toFloat().coerceIn(0.001f, 0.5f)
                        MaskExpr.LuminanceKey(src, minL, maxL, feather)
                    }
                    "union", "max", "or" -> {
                        val a = parseMaskNode(raw.opt("a"), knownMasks)
                            ?: throw IllegalArgumentException("Mask union missing operand 'a'")
                        val b = parseMaskNode(raw.opt("b"), knownMasks)
                            ?: throw IllegalArgumentException("Mask union missing operand 'b'")
                        MaskExpr.Union(a, b)
                    }
                    "intersect", "min", "and", "multiply" -> {
                        val a = parseMaskNode(raw.opt("a"), knownMasks)
                            ?: throw IllegalArgumentException("Mask intersect missing operand 'a'")
                        val b = parseMaskNode(raw.opt("b"), knownMasks)
                            ?: throw IllegalArgumentException("Mask intersect missing operand 'b'")
                        MaskExpr.Intersect(a, b)
                    }
                    "subtract", "sub", "diff" -> {
                        val a = parseMaskNode(raw.opt("a"), knownMasks)
                            ?: throw IllegalArgumentException("Mask subtract missing operand 'a'")
                        val b = parseMaskNode(raw.opt("b"), knownMasks)
                            ?: throw IllegalArgumentException("Mask subtract missing operand 'b'")
                        MaskExpr.Subtract(a, b)
                    }
                    "feather", "blur", "smooth" -> {
                        val input = parseMaskNode(raw.opt("input") ?: raw.opt("a"), knownMasks)
                            ?: throw IllegalArgumentException("Mask feather missing 'input'")
                        val radius = raw.optDouble("radius", raw.optDouble("feather", DEFAULT_FEATHER_NORM.toDouble())).toFloat()
                        MaskExpr.Feather(input, radius)
                    }
                    "invert", "not" -> {
                        val input = parseMaskNode(raw.opt("input") ?: raw.opt("a"), knownMasks)
                            ?: throw IllegalArgumentException("Mask invert missing 'input'")
                        MaskExpr.Invert(input)
                    }
                    else -> throw IllegalArgumentException("Unsupported mask operator '$op'")
                }
            }
            return null
        }

        private const val VS_REGION_MASK = """#version 300 es
            layout(location = 0) in vec2 aPosNdc;
            layout(location = 1) in float aWeight;
            uniform float uFlipY;
            out float vWeight;
            void main() {
                vWeight = aWeight;
                float fy = (abs(uFlipY) > 0.01) ? uFlipY : 1.0;
                gl_Position = vec4(aPosNdc.x, aPosNdc.y * fy, 0.0, 1.0);
            }
        """

        private const val FS_REGION_MASK = """#version 300 es
            precision mediump float;
            in float vWeight;
            uniform float uRegionSign;
            out vec4 fragColor;
            void main() {
                float w = clamp(vWeight, 0.0, 1.0) * uRegionSign;
                fragColor = vec4(w, w, w, w);
            }
        """

        private val IRIS_COS = FloatArray(IRIS_CIRCLE_SEGMENTS) { i ->
            cos(2.0 * PI * i / IRIS_CIRCLE_SEGMENTS).toFloat()
        }
        private val IRIS_SIN = FloatArray(IRIS_CIRCLE_SEGMENTS) { i ->
            sin(2.0 * PI * i / IRIS_CIRCLE_SEGMENTS).toFloat()
        }
    }

    val smoother = ControlPointSmoother()

    var fboId: Int = 0
        private set
    var maskTextureId: Int = 0
        private set
    var fboWidth: Int = 0
        private set
    var fboHeight: Int = 0
        private set

    private var program: Int = 0
    private var vao: Int = 0
    private var vbo: Int = 0

    var isInitialized: Boolean = false
        private set

    var lastTriangulatedVertexCount: Int = 0
        private set
    var lastTriangulatedTriangleCount: Int = 0
        private set
    var lastRenderedRegionCount: Int = 0
        private set

    // Pre-allocated direct buffer for zero-allocation dynamic polygon VBO uploads
    private val vertexBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(MAX_TOTAL_VERTEX_FLOATS * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val vertexScratch = FloatArray(MAX_TOTAL_VERTEX_FLOATS)

    // Pre-allocated scratch arrays for 2D normalized view coordinates [0..1, 0..1]
    private val contourNormX = FloatArray(MAX_CONTOUR_POINTS)
    private val contourNormY = FloatArray(MAX_CONTOUR_POINTS)
    private val outerNormX = FloatArray(MAX_CONTOUR_POINTS)
    private val outerNormY = FloatArray(MAX_CONTOUR_POINTS)
    private val eyeClipNormX = FloatArray(MAX_CONTOUR_POINTS)
    private val eyeClipNormY = FloatArray(MAX_CONTOUR_POINTS)
    private val ptScratch = FloatArray(2)

    @JvmOverloads
    fun init(resourceOwner: GlResourceOwner? = null) {
        fboId = 0
        maskTextureId = 0
        fboWidth = 0
        fboHeight = 0
        program = 0
        vao = 0
        vbo = 0
        uploadedMaskCache.clear()
        try {
            if (hasActiveGlContext()) {
                program = compileProgram(VS_REGION_MASK, FS_REGION_MASK)
                if (program != 0) {
                    resourceOwner?.trackProgram(program)
                }
                val vaos = IntArray(1)
                val vbos = IntArray(1)
                GLES30.glGenVertexArrays(1, vaos, 0)
                GLES30.glGenBuffers(1, vbos, 0)
                vao = vaos[0]
                vbo = vbos[0]
                resourceOwner?.trackVao(vao)
                resourceOwner?.trackBuffer(vbo)

                if (vao != 0 && vbo != 0) {
                    GLES30.glBindVertexArray(vao)
                    GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
                    GLES30.glBufferData(
                        GLES30.GL_ARRAY_BUFFER,
                        MAX_TOTAL_VERTEX_FLOATS * 4,
                        null,
                        GLES30.GL_DYNAMIC_DRAW
                    )
                    val stride = 3 * 4
                    GLES30.glEnableVertexAttribArray(0)
                    GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, stride, 0)
                    GLES30.glEnableVertexAttribArray(1)
                    GLES30.glVertexAttribPointer(1, 1, GLES30.GL_FLOAT, false, stride, 2 * 4)
                    GLES30.glBindVertexArray(0)
                    GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
                }
            }
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
        isInitialized = true
    }

    /**
     * Ensures the half-resolution (`max(1, viewportWidth / 2)` × `max(1, viewportHeight / 2)`)
     * `GL_R8` FBO and texture are allocated for the given full-resolution viewport.
     */
    fun ensureHalfResFbo(
        viewportWidth: Int,
        viewportHeight: Int,
        resourceOwner: GlResourceOwner? = null
    ) {
        val targetW = max(1, viewportWidth / 2)
        val targetH = max(1, viewportHeight / 2)
        if (fboWidth == targetW && fboHeight == targetH && (fboId != 0 || !hasActiveGlContext())) {
            return
        }
        fboWidth = targetW
        fboHeight = targetH

        if (!hasActiveGlContext()) return
        try {
            if (fboId == 0) {
                val fbos = IntArray(1)
                GLES30.glGenFramebuffers(1, fbos, 0)
                fboId = fbos[0]
                if (fboId != 0) resourceOwner?.trackFramebuffer(fboId)
            }
            if (maskTextureId == 0) {
                val texs = IntArray(1)
                GLES30.glGenTextures(1, texs, 0)
                maskTextureId = texs[0]
                if (maskTextureId != 0) resourceOwner?.trackTexture(maskTextureId)
            }
            if (fboId != 0 && maskTextureId != 0) {
                val prevDrawFbo = IntArray(1)
                val prevReadFbo = IntArray(1)
                val prevTex2D = IntArray(1)
                GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevDrawFbo, 0)
                GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, prevReadFbo, 0)
                GLES30.glGetIntegerv(GLES30.GL_TEXTURE_BINDING_2D, prevTex2D, 0)

                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTextureId)
                GLES30.glTexImage2D(
                    GLES30.GL_TEXTURE_2D,
                    0,
                    GLES30.GL_R8,
                    fboWidth,
                    fboHeight,
                    0,
                    GLES30.GL_RED,
                    GLES30.GL_UNSIGNED_BYTE,
                    null
                )
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
                GLES30.glFramebufferTexture2D(
                    GLES30.GL_FRAMEBUFFER,
                    GLES30.GL_COLOR_ATTACHMENT0,
                    GLES30.GL_TEXTURE_2D,
                    maskTextureId,
                    0
                )
                GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, prevDrawFbo[0])
                GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevReadFbo[0])
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, prevTex2D[0])
            }
        } catch (_: Throwable) {
        }
    }

    /**
     * Smooths control-point landmarks in [faceState], triangulates the requested [regions] with
     * feathered soft edges (`[xNdc, yNdc, weight]`), and renders them into the half-res `GL_R8` FBO.
     *
     * Returns the number of triangles generated across all [regions].
     */
    @JvmOverloads
    fun renderRegionsToHalfResFbo(
        faceState: FaceState,
        viewportWidth: Int,
        viewportHeight: Int,
        regions: List<RegionType> = listOf(
            RegionType.FACE_OVAL,
            RegionType.RIGHT_EYE,
            RegionType.LEFT_EYE,
            RegionType.MOUTH_INNER
        ),
        featherNorm: Float = DEFAULT_FEATHER_NORM,
        resourceOwner: GlResourceOwner? = null
    ): Int {
        ensureHalfResFbo(viewportWidth, viewportHeight, resourceOwner)
        val isTracking = faceState.isVisible && faceState.lifecycle != FaceLifecycle.State.NO_FACE
        val tsNs = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
        val smoothed = smoother.smooth(faceState.landmarks, tsNs, isTracking)
        if (smoothed == null) {
            lastTriangulatedVertexCount = 0
            lastTriangulatedTriangleCount = 0
            lastRenderedRegionCount = 0
            return 0
        }

        var floatOffset = 0
        var regionCount = 0
        for (i in regions.indices) {
            val rType = regions[i]
            val addedFloats = triangulateRegionIntoScratch(
                regionType = rType,
                smoothedLandmarks = smoothed,
                faceState = faceState,
                featherNorm = featherNorm,
                outFloats = vertexScratch,
                startFloatIdx = floatOffset
            )
            if (addedFloats > 0) {
                floatOffset += addedFloats
                regionCount++
            }
        }

        val totalVerts = floatOffset / 3
        lastTriangulatedVertexCount = totalVerts
        lastTriangulatedTriangleCount = totalVerts / 3
        lastRenderedRegionCount = regionCount

        if (totalVerts == 0 || !hasActiveGlContext() || fboId == 0 || program == 0 || vao == 0 || vbo == 0) {
            return lastTriangulatedTriangleCount
        }

        try {
            val prevDrawFbo = IntArray(1)
            val prevReadFbo = IntArray(1)
            val prevViewport = IntArray(4)
            GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevDrawFbo, 0)
            GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, prevReadFbo, 0)
            GLES30.glGetIntegerv(GLES30.GL_VIEWPORT, prevViewport, 0)

            vertexBuffer.clear()
            vertexBuffer.put(vertexScratch, 0, floatOffset)
            (vertexBuffer as java.nio.Buffer).position(0)

            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            GLES30.glViewport(0, 0, fboWidth, fboHeight)
            GLES30.glColorMask(true, false, false, false)
            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            GLES30.glDepthMask(false)
            GLES30.glClearColor(0f, 0f, 0f, 0f)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

            GLES30.glEnable(GLES30.GL_BLEND)
            GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE)
            GLES30.glBlendEquation(GLES30.GL_MAX)

            GLES30.glUseProgram(program)
            val uSignLoc = GLES30.glGetUniformLocation(program, "uRegionSign")
            if (uSignLoc >= 0) GLES30.glUniform1f(uSignLoc, 1.0f)
            val uFlipYLoc = GLES30.glGetUniformLocation(program, "uFlipY")
            if (uFlipYLoc >= 0) GLES30.glUniform1f(uFlipYLoc, 1.0f)

            GLES30.glBindVertexArray(vao)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
            GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, floatOffset * 4, vertexBuffer)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, totalVerts)
            GLES30.glBindVertexArray(0)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)

            // Restore blend equation and framebuffer state
            GLES30.glBlendEquation(GLES30.GL_FUNC_ADD)
            GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)
            GLES30.glDisable(GLES30.GL_BLEND)
            GLES30.glColorMask(true, true, true, true)
            GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, prevDrawFbo[0])
            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevReadFbo[0])
            GLES30.glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3])
        } catch (_: Throwable) {
        }

        return lastTriangulatedTriangleCount
    }

    /**
     * Triangulates a single [regionType] contour from [smoothedLandmarks] into [outFloats] starting
     * at [startFloatIdx], formatted as `[xNdc, yNdc, weight]` per vertex (3 vertices per triangle).
     *
     * Returns the number of floats written. Zero heap allocations.
     */
    fun triangulateRegionIntoScratch(
        regionType: RegionType,
        smoothedLandmarks: FloatArray,
        faceState: FaceState,
        featherNorm: Float = DEFAULT_FEATHER_NORM,
        outFloats: FloatArray = vertexScratch,
        startFloatIdx: Int = 0,
        irisRadiusScale: Float = 1.0f,
        clampIrisToEyelid: Boolean = true
    ): Int {
        val count = extractRegionContourNormalizedViewPts(
            regionType = regionType,
            landmarks = smoothedLandmarks,
            faceState = faceState,
            outNormX = contourNormX,
            outNormY = contourNormY,
            irisRadiusScale = irisRadiusScale,
            clampIrisToEyelid = clampIrisToEyelid
        )
        val loopIndices = if (regionType == RegionType.FACE_OVAL) FaceRegions.FACE_OVAL else null
        return triangulateContourPointsIntoScratch(
            count = count,
            featherNorm = featherNorm,
            outFloats = outFloats,
            startFloatIdx = startFloatIdx,
            loopIndices = loopIndices
        )
    }

    private fun triangulateLoopIndicesIntoScratch(
        loopIndices: IntArray,
        smoothedLandmarks: FloatArray,
        faceState: FaceState,
        featherNorm: Float = DEFAULT_FEATHER_NORM,
        outFloats: FloatArray = vertexScratch,
        startFloatIdx: Int = 0
    ): Int {
        if (smoothedLandmarks.size < UvRegistry.VERTEX_COUNT * 3) return 0
        val count = mapLoopToNormalizedView(
            indices = loopIndices,
            landmarks = smoothedLandmarks,
            faceState = faceState,
            outNormX = contourNormX,
            outNormY = contourNormY
        )
        return triangulateContourPointsIntoScratch(
            count = count,
            featherNorm = featherNorm,
            outFloats = outFloats,
            startFloatIdx = startFloatIdx,
            loopIndices = loopIndices
        )
    }

    private fun triangulateContourPointsIntoScratch(
        count: Int,
        featherNorm: Float,
        outFloats: FloatArray,
        startFloatIdx: Int,
        loopIndices: IntArray? = null
    ): Int {
        if (count < 3) return 0

        // Compute polygon centroid in normalized view space [0..1, 0..1]
        var cx = 0f
        var cy = 0f
        for (i in 0 until count) {
            cx += contourNormX[i]
            cy += contourNormY[i]
        }
        cx /= count.toFloat()
        cy /= count.toFloat()

        val safeFeather = featherNorm.coerceIn(0.0005f, 0.15f)
        val isFaceOval = loopIndices != null && isFaceOvalLoop(loopIndices)
        for (i in 0 until count) {
            val dx = contourNormX[i] - cx
            val dy = contourNormY[i] - cy
            val len = sqrt(dx * dx + dy * dy).coerceAtLeast(1e-5f)
            // Expand hairline feather smoothly along the extrapolated upper forehead arc
            val cranialW = if (isFaceOval && i < loopIndices!!.size) {
                cranialForeheadWeightForLandmark(loopIndices[i])
            } else {
                0f
            }
            val vertexFeather = (safeFeather * (1f + cranialW * 3.5f)).coerceIn(0.0005f, 0.18f)
            outerNormX[i] = contourNormX[i] + (dx / len) * vertexFeather
            outerNormY[i] = contourNormY[i] + (dy / len) * vertexFeather
        }

        val cNdcX = cx * 2f - 1f
        val cNdcY = 1f - cy * 2f
        var ptr = startFloatIdx

        for (i in 0 until count) {
            if (ptr + 27 > outFloats.size) break
            val next = (i + 1) % count

            val x0 = contourNormX[i] * 2f - 1f
            val y0 = 1f - contourNormY[i] * 2f
            val x1 = contourNormX[next] * 2f - 1f
            val y1 = 1f - contourNormY[next] * 2f

            val ox0 = outerNormX[i] * 2f - 1f
            val oy0 = 1f - outerNormY[i] * 2f
            val ox1 = outerNormX[next] * 2f - 1f
            val oy1 = 1f - outerNormY[next] * 2f

            // 1. Interior fan triangle: (centroid, v_i, v_next) all with weight = 1.0f
            outFloats[ptr++] = cNdcX; outFloats[ptr++] = cNdcY; outFloats[ptr++] = 1.0f
            outFloats[ptr++] = x0;    outFloats[ptr++] = y0;    outFloats[ptr++] = 1.0f
            outFloats[ptr++] = x1;    outFloats[ptr++] = y1;    outFloats[ptr++] = 1.0f

            // 2. Feathered skirt triangle A: (v_i [1.0], outer_i [0.0], outer_next [0.0])
            outFloats[ptr++] = x0;    outFloats[ptr++] = y0;    outFloats[ptr++] = 1.0f
            outFloats[ptr++] = ox0;   outFloats[ptr++] = oy0;   outFloats[ptr++] = 0.0f
            outFloats[ptr++] = ox1;   outFloats[ptr++] = oy1;   outFloats[ptr++] = 0.0f

            // 3. Feathered skirt triangle B: (v_i [1.0], outer_next [0.0], v_next [1.0])
            outFloats[ptr++] = x0;    outFloats[ptr++] = y0;    outFloats[ptr++] = 1.0f
            outFloats[ptr++] = ox1;   outFloats[ptr++] = oy1;   outFloats[ptr++] = 0.0f
            outFloats[ptr++] = x1;    outFloats[ptr++] = y1;    outFloats[ptr++] = 1.0f
        }

        return ptr - startFloatIdx
    }

    /**
     * Extracts the 2D normalized view coordinates `[0..1, 0..1]` for [regionType] into
     * `[outNormX, outNormY]` and returns the vertex count.
     *
     * For [RegionType.RIGHT_IRIS_CLAMPED] and [RegionType.LEFT_IRIS_CLAMPED], constructs a
     * 16-segment iris disc from the iris center & perimeter landmarks (`#468..#472` / `#473..#477`)
     * and intersects each spoke with the eyelid opening polygon (`RIGHT_EYE_CONTOUR` / `LEFT_EYE_CONTOUR`)
     * so the eyelid clips the iris during blinks or squints.
     */
    fun extractRegionContourNormalizedViewPts(
        regionType: RegionType,
        landmarks: FloatArray,
        faceState: FaceState,
        outNormX: FloatArray = contourNormX,
        outNormY: FloatArray = contourNormY,
        irisRadiusScale: Float = 1.0f,
        clampIrisToEyelid: Boolean = true
    ): Int {
        if (landmarks.size < UvRegistry.VERTEX_COUNT * 3) return 0

        return when (regionType) {
            RegionType.FACE_OVAL -> mapLoopToNormalizedView(FaceRegions.FACE_OVAL, landmarks, faceState, outNormX, outNormY)
            RegionType.RIGHT_EYE -> mapLoopToNormalizedView(FaceRegions.RIGHT_EYE_CONTOUR, landmarks, faceState, outNormX, outNormY)
            RegionType.LEFT_EYE -> mapLoopToNormalizedView(FaceRegions.LEFT_EYE_CONTOUR, landmarks, faceState, outNormX, outNormY)
            RegionType.LIPS_OUTER -> mapLoopToNormalizedView(FaceRegions.LIPS_OUTER, landmarks, faceState, outNormX, outNormY)
            RegionType.MOUTH_INNER -> mapLoopToNormalizedView(FaceRegions.MOUTH_INNER, landmarks, faceState, outNormX, outNormY)
            RegionType.RIGHT_EYEBROW -> mapLoopToNormalizedView(FaceRegions.RIGHT_EYEBROW, landmarks, faceState, outNormX, outNormY)
            RegionType.LEFT_EYEBROW -> mapLoopToNormalizedView(FaceRegions.LEFT_EYEBROW, landmarks, faceState, outNormX, outNormY)
            RegionType.RIGHT_IRIS_CLAMPED -> buildIrisClampedToEyelid(
                irisCenterIdx = FaceRegions.RIGHT_IRIS_CENTER,
                irisRingIndices = FaceRegions.RIGHT_IRIS_RING,
                eyeContourIndices = FaceRegions.RIGHT_EYE_CONTOUR,
                landmarks = landmarks,
                faceState = faceState,
                outNormX = outNormX,
                outNormY = outNormY,
                radiusScale = irisRadiusScale,
                clampToEyelid = clampIrisToEyelid
            )
            RegionType.LEFT_IRIS_CLAMPED -> buildIrisClampedToEyelid(
                irisCenterIdx = FaceRegions.LEFT_IRIS_CENTER,
                irisRingIndices = FaceRegions.LEFT_IRIS_RING,
                eyeContourIndices = FaceRegions.LEFT_EYE_CONTOUR,
                landmarks = landmarks,
                faceState = faceState,
                outNormX = outNormX,
                outNormY = outNormY,
                radiusScale = irisRadiusScale,
                clampToEyelid = clampIrisToEyelid
            )
        }
    }

    private fun mapLoopToNormalizedView(
        indices: IntArray,
        landmarks: FloatArray,
        faceState: FaceState,
        outNormX: FloatArray,
        outNormY: FloatArray
    ): Int {
        val count = min(indices.size, min(outNormX.size, outNormY.size))
        val isFaceOval = isFaceOvalLoop(indices)
        var cranialDx = 0f
        var cranialDy = 0f
        if (isFaceOval && landmarks.size >= 153 * 3) {
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = landmarks[10 * 3],
                v = landmarks[10 * 3 + 1],
                z = landmarks[10 * 3 + 2],
                faceState = faceState,
                viewWidth = 1.0f,
                viewHeight = 1.0f,
                outPt = ptScratch
            )
            val fX = ptScratch[0]
            val fY = ptScratch[1]
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = landmarks[152 * 3],
                v = landmarks[152 * 3 + 1],
                z = landmarks[152 * 3 + 2],
                faceState = faceState,
                viewWidth = 1.0f,
                viewHeight = 1.0f,
                outPt = ptScratch
            )
            val cX = ptScratch[0]
            val cY = ptScratch[1]
            cranialDx = fX - cX
            cranialDy = fY - cY
        }
        for (i in 0 until count) {
            val lmIdx = indices[i]
            val baseIdx = lmIdx * 3
            if (baseIdx + 2 >= landmarks.size) return 0
            val u = landmarks[baseIdx]
            val v = landmarks[baseIdx + 1]
            val z = landmarks[baseIdx + 2]
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = u,
                v = v,
                z = z,
                faceState = faceState,
                viewWidth = 1.0f,
                viewHeight = 1.0f,
                outPt = ptScratch
            )
            val cranialW = if (isFaceOval) cranialForeheadWeightForLandmark(lmIdx) else 0f
            outNormX[i] = (ptScratch[0] + cranialDx * cranialW).coerceIn(0.01f, 0.99f)
            outNormY[i] = (ptScratch[1] + cranialDy * cranialW).coerceIn(0.01f, 0.99f)
        }
        return count
    }

    private fun buildIrisClampedToEyelid(
        irisCenterIdx: Int,
        irisRingIndices: IntArray,
        eyeContourIndices: IntArray,
        landmarks: FloatArray,
        faceState: FaceState,
        outNormX: FloatArray,
        outNormY: FloatArray,
        radiusScale: Float = 1.0f,
        clampToEyelid: Boolean = true
    ): Int {
        val eyeCount = mapLoopToNormalizedView(
            indices = eyeContourIndices,
            landmarks = landmarks,
            faceState = faceState,
            outNormX = eyeClipNormX,
            outNormY = eyeClipNormY
        )
        if (eyeCount < 3) return 0

        // Determine iris center & radius in normalized view space (from 478 landmarks if present, else eye contour center)
        val cx: Float
        val cy: Float
        val baseRadius: Float
        var eyeSumX = 0f
        var eyeSumY = 0f
        var eyeMinX = Float.POSITIVE_INFINITY
        var eyeMaxX = Float.NEGATIVE_INFINITY
        var eyeMinY = Float.POSITIVE_INFINITY
        var eyeMaxY = Float.NEGATIVE_INFINITY
        for (i in 0 until eyeCount) {
            val ex = eyeClipNormX[i]
            val ey = eyeClipNormY[i]
            eyeSumX += ex
            eyeSumY += ey
            if (ex < eyeMinX) eyeMinX = ex
            if (ex > eyeMaxX) eyeMaxX = ex
            if (ey < eyeMinY) eyeMinY = ey
            if (ey > eyeMaxY) eyeMaxY = ey
        }
        val eyeCentroidX = eyeSumX / eyeCount.toFloat()
        val eyeCentroidY = eyeSumY / eyeCount.toFloat()

        if (landmarks.size >= UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3) {
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = landmarks[irisCenterIdx * 3],
                v = landmarks[irisCenterIdx * 3 + 1],
                z = landmarks[irisCenterIdx * 3 + 2],
                faceState = faceState,
                viewWidth = 1.0f,
                viewHeight = 1.0f,
                outPt = ptScratch
            )
            cx = ptScratch[0]
            cy = ptScratch[1]

            var rSum = 0f
            for (rIdx in irisRingIndices) {
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = landmarks[rIdx * 3],
                    v = landmarks[rIdx * 3 + 1],
                    z = landmarks[rIdx * 3 + 2],
                    faceState = faceState,
                    viewWidth = 1.0f,
                    viewHeight = 1.0f,
                    outPt = ptScratch
                )
                rSum += hypot(ptScratch[0] - cx, ptScratch[1] - cy)
            }
            baseRadius = (rSum / irisRingIndices.size.toFloat()).coerceIn(0.002f, 0.15f)
        } else {
            cx = eyeCentroidX
            cy = eyeCentroidY
            baseRadius = ((eyeMaxY - eyeMinY) * 0.55f).coerceIn(0.002f, 0.10f)
        }
        val radius = (baseRadius * radiusScale.coerceIn(0.2f, 3.0f)).coerceIn(0.002f, 0.25f)

        // When used as a subtractive cutout (e.g. $mask:sclera = (left_eye U right_eye) - iris),
        // emit the unclamped 16-segment iris disc so GL_FUNC_REVERSE_SUBTRACT clips at per-pixel
        // FBO accuracy even when the iris center touches or crosses the eyelid contour during gaze.
        if (!clampToEyelid) {
            for (i in 0 until IRIS_CIRCLE_SEGMENTS) {
                outNormX[i] = cx + IRIS_COS[i] * radius
                outNormY[i] = cy + IRIS_SIN[i] * radius
            }
            return IRIS_CIRCLE_SEGMENTS
        }

        // If eyelid opening has collapsed or does not overlap the iris disc, return 0 vertices (fully clipped)
        val overlapLeft = max(cx - radius, eyeMinX)
        val overlapRight = min(cx + radius, eyeMaxX)
        val overlapTop = max(cy - radius, eyeMinY)
        val overlapBottom = min(cy + radius, eyeMaxY)
        if ((overlapRight - overlapLeft) < 1e-4f || (overlapBottom - overlapTop) < 1e-4f) {
            return 0
        }

        // If the iris center lands on or slightly outside the 16-point eyelid contour during extreme
        // upward/downward gaze while the disc still overlaps the open eye, pull the ray origin slightly
        // toward the eye contour centroid so spokes into the open eye remain clipped smoothly rather than dropping to 0.
        var rayOx = cx
        var rayOy = cy
        if (!isPointInPolygon(rayOx, rayOy, eyeClipNormX, eyeClipNormY, eyeCount)) {
            val distToEyeBoundary = distanceToPolygonBoundary(cx, cy, eyeClipNormX, eyeClipNormY, eyeCount)
            if (distToEyeBoundary >= radius) {
                return 0
            }
            for (step in 1..8) {
                val alpha = step / 8.0f
                val candX = cx + (eyeCentroidX - cx) * alpha
                val candY = cy + (eyeCentroidY - cy) * alpha
                if (isPointInPolygon(candX, candY, eyeClipNormX, eyeClipNormY, eyeCount)) {
                    rayOx = candX
                    rayOy = candY
                    break
                }
            }
        }

        // For each of the 16 iris disc directions, clamp the ray distance against the eye contour polygon
        var positiveSpokes = 0
        for (i in 0 until IRIS_CIRCLE_SEGMENTS) {
            val dirX = IRIS_COS[i]
            val dirY = IRIS_SIN[i]
            val maxPolyDist = rayPolygonBoundaryDistance(
                ox = rayOx,
                oy = rayOy,
                dirX = dirX,
                dirY = dirY,
                polyX = eyeClipNormX,
                polyY = eyeClipNormY,
                polyCount = eyeCount
            )
            // Measure radial limit relative to the ray origin and clamp against the iris disc perimeter around (cx, cy)
            val dxOrigin = rayOx - cx
            val dyOrigin = rayOy - cy
            val bProj = dxOrigin * dirX + dyOrigin * dirY
            val cQuad = dxOrigin * dxOrigin + dyOrigin * dyOrigin - radius * radius
            val discr = bProj * bProj - cQuad
            val maxCircleDist = if (discr >= 0f) (-bProj + sqrt(discr)).coerceAtLeast(0f) else 0f
            val clampedR = if (maxPolyDist.isFinite() && maxPolyDist >= 0f) {
                min(maxCircleDist, maxPolyDist)
            } else {
                0f
            }
            if (clampedR > 1e-4f) {
                positiveSpokes++
            }
            outNormX[i] = rayOx + dirX * clampedR
            outNormY[i] = rayOy + dirY * clampedR
        }
        return if (positiveSpokes >= 3) IRIS_CIRCLE_SEGMENTS else 0
    }

    /**
     * Evaluates the continuous feathered mask weight `[0.0 .. 1.0]` of [regionType] at normalized
     * view coordinate `(normX, normY)` in `[0..1, 0..1]`.
     *
     * - Returns `1.0f` strictly inside the region contour polygon.
     * - Smoothly falls off `1.0f -> 0.0f` across `[0 .. featherNorm]` outside the polygon boundary.
     * - For [RegionType.RIGHT_IRIS_CLAMPED] / [RegionType.LEFT_IRIS_CLAMPED], returns `0.0f` outside
     *   the eyelid contour when the eyelid clips the iris.
     */
    fun evaluateRegionMaskAtNormalizedViewPt(
        regionType: RegionType,
        normX: Float,
        normY: Float,
        faceState: FaceState,
        featherNorm: Float = DEFAULT_FEATHER_NORM,
        useSmoothedLandmarks: Boolean = true
    ): Float {
        val rawLm = faceState.landmarks ?: return 0f
        val lm = if (useSmoothedLandmarks) {
            val ts = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
            smoother.smooth(rawLm, ts, faceState.isVisible) ?: rawLm
        } else {
            rawLm
        }
        val count = extractRegionContourNormalizedViewPts(
            regionType = regionType,
            landmarks = lm,
            faceState = faceState,
            outNormX = contourNormX,
            outNormY = contourNormY
        )
        if (count < 3) return 0f

        val inside = isPointInPolygon(normX, normY, contourNormX, contourNormY, count)
        if (inside) return 1.0f

        val loopIndices = if (regionType == RegionType.FACE_OVAL) FaceRegions.FACE_OVAL else null
        return evaluateFeatheredFalloffOutsidePolygon(
            px = normX,
            py = normY,
            polyX = contourNormX,
            polyY = contourNormY,
            count = count,
            baseFeatherNorm = featherNorm,
            loopIndices = loopIndices
        )
    }

    /**
     * Rasterizes the union (maximum) of [regions] into a half-resolution (`max(1, viewportWidth / 2)` ×
     * `max(1, viewportHeight / 2)`) single-channel `R8` byte buffer (`0..255` unsigned).
     */
    @JvmOverloads
    fun rasterizeRegionsToHalfResR8(
        faceState: FaceState,
        viewportWidth: Int,
        viewportHeight: Int,
        regions: List<RegionType>,
        featherNorm: Float = DEFAULT_FEATHER_NORM,
        outBuffer: ByteArray? = null
    ): ByteArray {
        val w = max(1, viewportWidth / 2)
        val h = max(1, viewportHeight / 2)
        fboWidth = w
        fboHeight = h
        val buf = if (outBuffer != null && outBuffer.size >= w * h) outBuffer else ByteArray(w * h)
        java.util.Arrays.fill(buf, 0, w * h, 0.toByte())

        val rawLm = faceState.landmarks ?: return buf
        val ts = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
        val lm = smoother.smooth(rawLm, ts, faceState.isVisible) ?: return buf

        for (rType in regions) {
            val count = extractRegionContourNormalizedViewPts(
                regionType = rType,
                landmarks = lm,
                faceState = faceState,
                outNormX = contourNormX,
                outNormY = contourNormY
            )
            if (count < 3) continue

            val safeFeather = featherNorm.coerceAtLeast(1e-5f)
            val loopIndices = if (rType == RegionType.FACE_OVAL) FaceRegions.FACE_OVAL else null
            val maxBoundFeather = if (loopIndices != null) (safeFeather * 4.5f).coerceAtMost(0.18f) else safeFeather
            var minX = 1f; var minY = 1f; var maxX = 0f; var maxY = 0f
            for (i in 0 until count) {
                val px = contourNormX[i]
                val py = contourNormY[i]
                if (px < minX) minX = px
                if (py < minY) minY = py
                if (px > maxX) maxX = px
                if (py > maxY) maxY = py
            }
            val xStart = ((minX - maxBoundFeather) * w).toInt().coerceIn(0, w - 1)
            val xEnd = ((maxX + maxBoundFeather) * w).toInt().coerceIn(0, w - 1)
            val yStart = ((minY - maxBoundFeather) * h).toInt().coerceIn(0, h - 1)
            val yEnd = ((maxY + maxBoundFeather) * h).toInt().coerceIn(0, h - 1)

            for (py in yStart..yEnd) {
                val ny = (py + 0.5f) / h.toFloat()
                val rowBase = py * w
                for (px in xStart..xEnd) {
                    val nx = (px + 0.5f) / w.toFloat()
                    val weight = if (isPointInPolygon(nx, ny, contourNormX, contourNormY, count)) {
                        1.0f
                    } else {
                        evaluateFeatheredFalloffOutsidePolygon(
                            px = nx,
                            py = ny,
                            polyX = contourNormX,
                            polyY = contourNormY,
                            count = count,
                            baseFeatherNorm = safeFeather,
                            loopIndices = loopIndices
                        )
                    }
                    if (weight > 0f) {
                        val newByte = (weight * 255f + 0.5f).toInt().coerceIn(0, 255)
                        val prevByte = buf[rowBase + px].toInt() and 0xFF
                        if (newByte > prevByte) {
                            buf[rowBase + px] = newByte.toByte()
                        }
                    }
                }
            }
        }
        return buf
    }

    private val evaluatedMaskIdsScratch = ArrayList<String>(8)
    val lastEvaluatedMaskIds: List<String>
        get() = evaluatedMaskIdsScratch

    fun clearEvaluatedMasks() {
        evaluatedMaskIdsScratch.clear()
    }

    /**
     * Evaluates any composable [MaskExpr] tree at normalized view coordinate `(normX, normY)` in `[0..1, 0..1]`
     * with zero heap allocations.
     */
    @JvmOverloads
    fun evaluateMaskExprAtNormalizedViewPt(
        expr: MaskExpr,
        normX: Float,
        normY: Float,
        faceState: FaceState,
        sampleLuma: Float = 0.5f,
        useSmoothedLandmarks: Boolean = true,
        extraFeatherNorm: Float = 0f
    ): Float {
        val rawLm = faceState.landmarks ?: return 0f
        val lm = if (useSmoothedLandmarks) {
            val ts = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
            smoother.smooth(rawLm, ts, faceState.isVisible) ?: rawLm
        } else {
            rawLm
        }
        return evaluateMaskExprWithLandmarks(
            expr = expr,
            normX = normX,
            normY = normY,
            landmarks = lm,
            faceState = faceState,
            sampleLuma = sampleLuma,
            extraFeatherNorm = extraFeatherNorm
        ).coerceIn(0f, 1f)
    }

    /**
     * Resolves a named mask (`"skin"`, `"eyes"`, `"mouth"`, `"lips"`, `"iris"`, or a manifest-defined
     * mask in [customMasks]) and evaluates it at `(normX, normY)` scaled by [maskWeight].
     */
    @JvmOverloads
    fun evaluateNamedMaskAtNormalizedViewPt(
        maskId: String,
        normX: Float,
        normY: Float,
        faceState: FaceState,
        customMasks: Map<String, MaskExpr> = emptyMap(),
        maskWeight: Float = 1.0f,
        sampleLuma: Float = 0.5f,
        useSmoothedLandmarks: Boolean = true
    ): Float {
        val cleanKey = maskId.removePrefix("\$mask:").trim().lowercase().replace('-', '_')
        val expr = customMasks[cleanKey]
            ?: BUILTIN_MASKS[cleanKey]
            ?: FaceRegions.getRegionLoop(cleanKey)?.let { MaskExpr.LandmarkLoop(cleanKey, DEFAULT_FEATHER_NORM) }
            ?: return 0f
        val rawVal = evaluateMaskExprAtNormalizedViewPt(
            expr = expr,
            normX = normX,
            normY = normY,
            faceState = faceState,
            sampleLuma = sampleLuma,
            useSmoothedLandmarks = useSmoothedLandmarks
        )
        return (rawVal * maskWeight.coerceIn(0f, 1f)).coerceIn(0f, 1f)
    }

    private class CachedLoopData(maxPoints: Int) {
        val normX = FloatArray(maxPoints)
        val normY = FloatArray(maxPoints)
        var count: Int = 0
        var minX: Float = 1f
        var minY: Float = 1f
        var maxX: Float = 0f
        var maxY: Float = 0f
    }

    private val preparedLoopCache = HashMap<String, CachedLoopData>(16)
    private var isLoopCacheActive: Boolean = false
    private var r8ScratchBuffer: ByteArray = ByteArray(0)
    private var r8UploadDirectBuffer: ByteBuffer? = null

    private fun prepareNamedLoopCache(landmarks: FloatArray, faceState: FaceState) {
        if (landmarks.size < UvRegistry.VERTEX_COUNT * 3) return
        for ((name, indices) in FaceRegions.NAMED_LOOPS) {
            val cached = preparedLoopCache.getOrPut(name) { CachedLoopData(MAX_CONTOUR_POINTS) }
            val cnt = mapLoopToNormalizedView(indices, landmarks, faceState, cached.normX, cached.normY)
            cached.count = cnt
            var mnX = 1f; var mnY = 1f; var mxX = 0f; var mxY = 0f
            for (i in 0 until cnt) {
                val x = cached.normX[i]
                val y = cached.normY[i]
                if (x < mnX) mnX = x
                if (y < mnY) mnY = y
                if (x > mxX) mxX = x
                if (y > mxY) mxY = y
            }
            cached.minX = mnX
            cached.minY = mnY
            cached.maxX = mxX
            cached.maxY = mxY
        }
    }

    private fun containsInvertOp(expr: MaskExpr): Boolean {
        return when (expr) {
            is MaskExpr.Invert -> true
            is MaskExpr.Union -> containsInvertOp(expr.a) || containsInvertOp(expr.b)
            is MaskExpr.Intersect -> containsInvertOp(expr.a) || containsInvertOp(expr.b)
            is MaskExpr.Subtract -> containsInvertOp(expr.a) || containsInvertOp(expr.b)
            is MaskExpr.Feather -> containsInvertOp(expr.input)
            else -> false
        }
    }

    private data class UploadedMaskCacheEntry(
        val timestampNs: Long,
        val weight: Float,
        val texId: Int,
        val width: Int,
        val height: Int,
        val expr: MaskExpr
    )

    private val uploadedMaskCache = HashMap<String, UploadedMaskCacheEntry>(8)
    private var glRasterScratchBuffer: ByteArray = ByteArray(0)
    private var glUpscaleScratchBuffer: ByteArray = ByteArray(0)

    /**
     * Computes tight normalized `[minX, minY, maxX, maxY]` bounds for [expr] using [preparedLoopCache].
     */
    private fun computeExprBounds(expr: MaskExpr, outBounds: FloatArray): Boolean {
        return when (expr) {
            is MaskExpr.LandmarkLoop -> {
                val clean = expr.regionName.trim().lowercase().replace('-', '_')
                val loop = preparedLoopCache[clean] ?: preparedLoopCache["face_oval"]
                if (loop != null && loop.count >= 3) {
                    val baseF = max(expr.featherNorm, 0.015f)
                    val f = if (clean == "face_oval") (baseF * 4.5f).coerceAtMost(0.18f) else baseF
                    outBounds[0] = (loop.minX - f).coerceIn(0f, 1f)
                    outBounds[1] = (loop.minY - f).coerceIn(0f, 1f)
                    outBounds[2] = (loop.maxX + f).coerceIn(0f, 1f)
                    outBounds[3] = (loop.maxY + f).coerceIn(0f, 1f)
                    true
                } else false
            }
            is MaskExpr.IrisCircle -> {
                val loopName = if (expr.isLeft) "left_eye" else "right_eye"
                val loop = preparedLoopCache[loopName] ?: preparedLoopCache["face_oval"]
                if (loop != null && loop.count >= 3) {
                    val f = max(expr.featherNorm, 0.02f)
                    outBounds[0] = (loop.minX - f).coerceIn(0f, 1f)
                    outBounds[1] = (loop.minY - f).coerceIn(0f, 1f)
                    outBounds[2] = (loop.maxX + f).coerceIn(0f, 1f)
                    outBounds[3] = (loop.maxY + f).coerceIn(0f, 1f)
                    true
                } else false
            }
            is MaskExpr.Subtract -> computeExprBounds(expr.a, outBounds)
            is MaskExpr.Intersect -> {
                val bA = FloatArray(4)
                val bB = FloatArray(4)
                val okA = computeExprBounds(expr.a, bA)
                val okB = computeExprBounds(expr.b, bB)
                if (okA && okB) {
                    outBounds[0] = max(bA[0], bB[0])
                    outBounds[1] = max(bA[1], bB[1])
                    outBounds[2] = min(bA[2], bB[2])
                    outBounds[3] = min(bA[3], bB[3])
                    outBounds[2] >= outBounds[0] && outBounds[3] >= outBounds[1]
                } else if (okA) {
                    System.arraycopy(bA, 0, outBounds, 0, 4)
                    true
                } else if (okB) {
                    System.arraycopy(bB, 0, outBounds, 0, 4)
                    true
                } else false
            }
            is MaskExpr.Union -> {
                val bA = FloatArray(4)
                val bB = FloatArray(4)
                val okA = computeExprBounds(expr.a, bA)
                val okB = computeExprBounds(expr.b, bB)
                if (okA && okB) {
                    outBounds[0] = min(bA[0], bB[0])
                    outBounds[1] = min(bA[1], bB[1])
                    outBounds[2] = max(bA[2], bB[2])
                    outBounds[3] = max(bA[3], bB[3])
                    true
                } else false
            }
            is MaskExpr.Feather -> {
                if (computeExprBounds(expr.input, outBounds)) {
                    val f = max(expr.radiusNorm, 0f)
                    outBounds[0] = (outBounds[0] - f).coerceIn(0f, 1f)
                    outBounds[1] = (outBounds[1] - f).coerceIn(0f, 1f)
                    outBounds[2] = (outBounds[2] + f).coerceIn(0f, 1f)
                    outBounds[3] = (outBounds[3] + f).coerceIn(0f, 1f)
                    true
                } else false
            }
            else -> false
        }
    }

    /**
     * Evaluates a composable [MaskExpr] tree across a half-resolution (`max(1, viewportWidth / 2)` ×
     * `max(1, viewportHeight / 2)`) single-channel `R8` byte buffer (`0..255` unsigned), scaled by [maskWeight].
     */
    @JvmOverloads
    fun evaluateMaskExprToHalfResR8(
        expr: MaskExpr,
        faceState: FaceState,
        viewportWidth: Int,
        viewportHeight: Int,
        maskWeight: Float = 1.0f,
        lumaBuffer: FloatArray? = null,
        outBuffer: ByteArray? = null,
        useSmoothedLandmarks: Boolean = true
    ): ByteArray {
        val w = max(1, viewportWidth / 2)
        val h = max(1, viewportHeight / 2)
        fboWidth = w
        fboHeight = h
        val total = w * h
        val buf = when {
            outBuffer != null && outBuffer.size >= total -> outBuffer
            r8ScratchBuffer.size == total -> r8ScratchBuffer
            else -> ByteArray(total).also { r8ScratchBuffer = it }
        }
        java.util.Arrays.fill(buf, 0, total, 0.toByte())

        val clampedWeight = maskWeight.coerceIn(0f, 1f)
        if (!faceState.isVisible || clampedWeight <= 0f) return buf
        val rawLm = faceState.landmarks ?: return buf
        val lm = if (useSmoothedLandmarks) {
            val ts = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
            smoother.smooth(rawLm, ts, faceState.isVisible) ?: return buf
        } else {
            rawLm
        }

        prepareNamedLoopCache(lm, faceState)
        isLoopCacheActive = true
        try {
            val hasInvert = containsInvertOp(expr)
            val exprBounds = FloatArray(4)
            val hasTightBounds = !hasInvert && computeExprBounds(expr, exprBounds)
            val faceOvalBounds = preparedLoopCache["face_oval"]
            val padNorm = 0.08f
            val xStart: Int
            val xEnd: Int
            val yStart: Int
            val yEnd: Int
            if (hasTightBounds) {
                xStart = (exprBounds[0] * w).toInt().coerceIn(0, w - 1)
                xEnd = (exprBounds[2] * w).toInt().coerceIn(0, w - 1)
                yStart = (exprBounds[1] * h).toInt().coerceIn(0, h - 1)
                yEnd = (exprBounds[3] * h).toInt().coerceIn(0, h - 1)
            } else if (!hasInvert && faceOvalBounds != null && faceOvalBounds.count >= 3) {
                xStart = ((faceOvalBounds.minX - padNorm) * w).toInt().coerceIn(0, w - 1)
                xEnd = ((faceOvalBounds.maxX + padNorm) * w).toInt().coerceIn(0, w - 1)
                yStart = ((faceOvalBounds.minY - padNorm) * h).toInt().coerceIn(0, h - 1)
                yEnd = ((faceOvalBounds.maxY + padNorm) * h).toInt().coerceIn(0, h - 1)
            } else {
                xStart = 0
                xEnd = w - 1
                yStart = 0
                yEnd = h - 1
            }

            if (xStart <= xEnd && yStart <= yEnd) {
                for (py in yStart..yEnd) {
                    val ny = (py + 0.5f) / h.toFloat()
                    val rowBase = py * w
                    for (px in xStart..xEnd) {
                        val nx = (px + 0.5f) / w.toFloat()
                        val idx = rowBase + px
                        val sampleLuma = if (lumaBuffer != null && idx < lumaBuffer.size) lumaBuffer[idx] else 0.5f
                        val v = evaluateMaskExprWithLandmarks(
                            expr = expr,
                            normX = nx,
                            normY = ny,
                            landmarks = lm,
                            faceState = faceState,
                            sampleLuma = sampleLuma,
                            extraFeatherNorm = 0f
                        ) * clampedWeight
                        if (v > 0f) {
                            buf[idx] = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt().coerceIn(0, 255).toByte()
                        }
                    }
                }
            }
        } finally {
            isLoopCacheActive = false
        }
        return buf
    }

    /**
     * Acquires or updates the named `$mask:<maskId>` half-resolution `GL_R8` render target in [rtPool]
     * using the dynamic weight from [propertySlots] (`mask:<maskId>.weight`).
     */
    @JvmOverloads
    fun evaluateMaskToRtPool(
        maskId: String,
        expr: MaskExpr,
        faceState: FaceState,
        viewportWidth: Int,
        viewportHeight: Int,
        rtPool: RtPool,
        propertySlots: PropertySlots? = null,
        scaleFactor: Float = RtPool.RtScale.HALF.factor,
        resourceOwner: GlResourceOwner? = null
    ): RtPool.PooledRenderTarget {
        val cleanId = maskId.removePrefix("\$mask:").trim()
        val rtName = "\$mask:$cleanId"
        val weight = propertySlots?.getDynamicParam("mask:$cleanId.weight", 1.0f) ?: 1.0f
        val target = rtPool.acquireNamed(
            name = rtName,
            baseWidth = viewportWidth,
            baseHeight = viewportHeight,
            format = RtPool.RtFormat.R8,
            scaleFactor = scaleFactor,
            withDepth = false,
            resourceOwner = resourceOwner
        )
        val alreadyEvaluatedThisFrame = evaluatedMaskIdsScratch.contains(rtName)
        if (!alreadyEvaluatedThisFrame) {
            evaluatedMaskIdsScratch.add(rtName)
        }
        if (hasActiveGlContext() && target.texId != 0 && weight > 0f && faceState.isVisible) {
            val faceTs = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
            val prevEntry = uploadedMaskCache[rtName]
            if (prevEntry != null &&
                prevEntry.expr == expr &&
                (alreadyEvaluatedThisFrame || (prevEntry.timestampNs == faceTs && faceTs > 0L)) &&
                kotlin.math.abs(prevEntry.weight - weight) < 1e-4f &&
                prevEntry.texId == target.texId &&
                prevEntry.width == target.width &&
                prevEntry.height == target.height
            ) {
                return target
            }

            // Fast path: Hardware GPU polygon rasterization directly into target.fboId (zero CPU pixel loop, zero texture upload)
            if (target.fboId != 0 && program != 0 && vao != 0 && vbo != 0) {
                val renderedOnGpu = tryRenderMaskExprToGpuFbo(
                    expr = expr,
                    faceState = faceState,
                    target = target,
                    maskWeight = weight
                )
                if (renderedOnGpu) {
                    uploadedMaskCache[rtName] = UploadedMaskCacheEntry(
                        timestampNs = faceTs,
                        weight = weight,
                        texId = target.texId,
                        width = target.width,
                        height = target.height,
                        expr = expr
                    )
                    return target
                }
            }

            // Fallback path for complex/custom masks (e.g. LuminanceKey, Intersect, Invert): bounded grid <= 144
            val maxSide = max(target.width, target.height).coerceAtLeast(1)
            val rasterScale = if (maxSide > 144) 144.0f / maxSide.toFloat() else 1.0f
            val rw = max(1, (target.width * rasterScale).toInt())
            val rh = max(1, (target.height * rasterScale).toInt())
            val rTotal = rw * rh
            if (glRasterScratchBuffer.size < rTotal) {
                glRasterScratchBuffer = ByteArray(rTotal)
            }
            val smallBytes = evaluateMaskExprToHalfResR8(
                expr = expr,
                faceState = faceState,
                viewportWidth = rw * 2,
                viewportHeight = rh * 2,
                maskWeight = weight,
                outBuffer = glRasterScratchBuffer
            )

            try {
                val byteCount = target.width * target.height
                val uploadSource: ByteArray = if (rw == target.width && rh == target.height) {
                    smallBytes
                } else {
                    if (glUpscaleScratchBuffer.size < byteCount) {
                        glUpscaleScratchBuffer = ByteArray(byteCount)
                    }
                    val dst = glUpscaleScratchBuffer
                    val tw = target.width
                    val th = target.height
                    for (y in 0 until th) {
                        val sy = (y * rh) / th
                        val srcRow = sy * rw
                        val dstRow = y * tw
                        for (x in 0 until tw) {
                            val sx = (x * rw) / tw
                            dst[dstRow + x] = smallBytes[srcRow + sx]
                        }
                    }
                    dst
                }

                var byteBuf = r8UploadDirectBuffer
                if (byteBuf == null || byteBuf.capacity() < byteCount) {
                    byteBuf = ByteBuffer.allocateDirect(byteCount).order(ByteOrder.nativeOrder())
                    r8UploadDirectBuffer = byteBuf
                }
                byteBuf.clear()
                byteBuf.put(uploadSource, 0, byteCount)
                byteBuf.position(0)
                val prevTex2D = IntArray(1)
                val prevUnpackAlign = IntArray(1)
                GLES30.glGetIntegerv(GLES30.GL_TEXTURE_BINDING_2D, prevTex2D, 0)
                GLES30.glGetIntegerv(GLES30.GL_UNPACK_ALIGNMENT, prevUnpackAlign, 0)
                GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, target.texId)
                GLES30.glTexSubImage2D(
                    GLES30.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    target.width,
                    target.height,
                    GLES30.GL_RED,
                    GLES30.GL_UNSIGNED_BYTE,
                    byteBuf
                )
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, prevTex2D[0])
                GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, if (prevUnpackAlign[0] > 0) prevUnpackAlign[0] else 4)
                uploadedMaskCache[rtName] = UploadedMaskCacheEntry(
                    timestampNs = faceTs,
                    weight = weight,
                    texId = target.texId,
                    width = target.width,
                    height = target.height,
                    expr = expr
                )
            } catch (_: Throwable) {
            }
        }
        return target
    }

    private val subtractiveVertexScratch = FloatArray(MAX_TOTAL_VERTEX_FLOATS)

    private fun tryRenderMaskExprToGpuFbo(
        expr: MaskExpr,
        faceState: FaceState,
        target: RtPool.PooledRenderTarget,
        maskWeight: Float
    ): Boolean {
        val rawLm = faceState.landmarks ?: return false
        val tsNs = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
        val smoothed = smoother.smooth(rawLm, tsNs, faceState.isVisible) ?: return false

        val counts = IntArray(2) // [0] = additiveFloats, [1] = subtractiveFloats
        if (!collectGpuMaskTriangles(expr, smoothed, faceState, isSubtracted = false, extraFeather = 0f, counts = counts)) {
            return false
        }
        val addFloats = counts[0]
        val subFloats = counts[1]
        if (addFloats <= 0) return false

        return try {
            val prevDrawFbo = IntArray(1)
            val prevReadFbo = IntArray(1)
            val prevViewport = IntArray(4)
            GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevDrawFbo, 0)
            GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, prevReadFbo, 0)
            GLES30.glGetIntegerv(GLES30.GL_VIEWPORT, prevViewport, 0)

            // Prevent texture-framebuffer feedback loop if target.texId was left bound on GL_TEXTURE0 or GL_TEXTURE1
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, target.fboId)
            GLES30.glViewport(0, 0, target.width, target.height)
            GLES30.glColorMask(true, false, false, false)
            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            GLES30.glDepthMask(false)
            GLES30.glClearColor(0f, 0f, 0f, 0f)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

            GLES30.glEnable(GLES30.GL_BLEND)
            GLES30.glUseProgram(program)
            val uSignLoc = GLES30.glGetUniformLocation(program, "uRegionSign")
            if (uSignLoc >= 0) GLES30.glUniform1f(uSignLoc, maskWeight.coerceIn(0f, 1f))
            // Flip Y to match top-down viewUv convention in PostRenderer and RecolorPass
            val uFlipYLoc = GLES30.glGetUniformLocation(program, "uFlipY")
            if (uFlipYLoc >= 0) GLES30.glUniform1f(uFlipYLoc, -1.0f)

            GLES30.glBindVertexArray(vao)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)

            // 1. Draw positive mask regions with GL_MAX
            GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE)
            GLES30.glBlendEquation(GLES30.GL_MAX)
            vertexBuffer.clear()
            vertexBuffer.put(vertexScratch, 0, addFloats)
            (vertexBuffer as java.nio.Buffer).position(0)
            GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, addFloats * 4, vertexBuffer)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, addFloats / 3)

            // 2. Draw subtracted cutout regions (e.g. eyes/lips/eyebrows for skin, mouth_inner for lips)
            if (subFloats > 0) {
                GLES30.glBlendEquation(GLES30.GL_FUNC_REVERSE_SUBTRACT)
                vertexBuffer.clear()
                vertexBuffer.put(subtractiveVertexScratch, 0, subFloats)
                (vertexBuffer as java.nio.Buffer).position(0)
                GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, subFloats * 4, vertexBuffer)
                GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, subFloats / 3)
            }

            GLES30.glBindVertexArray(0)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
            if (uFlipYLoc >= 0) GLES30.glUniform1f(uFlipYLoc, 1.0f)
            GLES30.glBlendEquation(GLES30.GL_FUNC_ADD)
            GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)
            GLES30.glDisable(GLES30.GL_BLEND)
            GLES30.glColorMask(true, true, true, true)
            GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, prevDrawFbo[0])
            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevReadFbo[0])
            GLES30.glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3])
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun collectGpuMaskTriangles(
        expr: MaskExpr,
        smoothedLandmarks: FloatArray,
        faceState: FaceState,
        isSubtracted: Boolean,
        extraFeather: Float,
        counts: IntArray
    ): Boolean {
        return when (expr) {
            is MaskExpr.LandmarkLoop -> {
                val loop = FaceRegions.getRegionLoop(expr.regionName) ?: return false
                val targetArr = if (isSubtracted) subtractiveVertexScratch else vertexScratch
                val startIdx = if (isSubtracted) counts[1] else counts[0]
                val added = triangulateLoopIndicesIntoScratch(
                    loopIndices = loop,
                    smoothedLandmarks = smoothedLandmarks,
                    faceState = faceState,
                    featherNorm = (expr.featherNorm + extraFeather).coerceAtLeast(0.001f),
                    outFloats = targetArr,
                    startFloatIdx = startIdx
                )
                if (isSubtracted) counts[1] += added else counts[0] += added
                true
            }
            is MaskExpr.IrisCircle -> {
                val rType = if (expr.isLeft) RegionType.LEFT_IRIS_CLAMPED else RegionType.RIGHT_IRIS_CLAMPED
                val targetArr = if (isSubtracted) subtractiveVertexScratch else vertexScratch
                val startIdx = if (isSubtracted) counts[1] else counts[0]
                val added = triangulateRegionIntoScratch(
                    regionType = rType,
                    smoothedLandmarks = smoothedLandmarks,
                    faceState = faceState,
                    featherNorm = (expr.featherNorm + extraFeather).coerceAtLeast(0.001f),
                    outFloats = targetArr,
                    startFloatIdx = startIdx,
                    irisRadiusScale = expr.radiusScale,
                    clampIrisToEyelid = !isSubtracted
                )
                if (isSubtracted) counts[1] += added else counts[0] += added
                true
            }
            is MaskExpr.Union -> {
                collectGpuMaskTriangles(expr.a, smoothedLandmarks, faceState, isSubtracted, extraFeather, counts) &&
                    collectGpuMaskTriangles(expr.b, smoothedLandmarks, faceState, isSubtracted, extraFeather, counts)
            }
            is MaskExpr.Intersect -> {
                // 1. IrisCircle intersected with eye loop: RIGHT_IRIS_CLAMPED / LEFT_IRIS_CLAMPED already clamps to eye loop
                if (expr.a is MaskExpr.IrisCircle && expr.b is MaskExpr.LandmarkLoop) {
                    collectGpuMaskTriangles(expr.a, smoothedLandmarks, faceState, isSubtracted, extraFeather, counts)
                } else if (expr.b is MaskExpr.IrisCircle && expr.a is MaskExpr.LandmarkLoop) {
                    collectGpuMaskTriangles(expr.b, smoothedLandmarks, faceState, isSubtracted, extraFeather, counts)
                } else if (isGpuEvaluatedLuminanceKey(expr.b)) {
                    // Luminance glare protection is evaluated per-pixel in RecolorPass FS_RECOLOR shader
                    collectGpuMaskTriangles(expr.a, smoothedLandmarks, faceState, isSubtracted, extraFeather, counts)
                } else if (isGpuEvaluatedLuminanceKey(expr.a)) {
                    collectGpuMaskTriangles(expr.b, smoothedLandmarks, faceState, isSubtracted, extraFeather, counts)
                } else if (!isSubtracted && expr.b is MaskExpr.Subtract) {
                    // 2. Sub-region (e.g. left_cheek U right_cheek) intersected with $mask:skin (face_oval - cutouts):
                    // additive sub-region with the same subtractive cutouts
                    collectGpuMaskTriangles(expr.a, smoothedLandmarks, faceState, false, extraFeather, counts) &&
                        collectGpuCutoutsOnly(expr.b, smoothedLandmarks, faceState, extraFeather, counts)
                } else if (!isSubtracted && expr.a is MaskExpr.Subtract) {
                    collectGpuMaskTriangles(expr.b, smoothedLandmarks, faceState, false, extraFeather, counts) &&
                        collectGpuCutoutsOnly(expr.a, smoothedLandmarks, faceState, extraFeather, counts)
                } else {
                    false
                }
            }
            is MaskExpr.Subtract -> {
                if (isSubtracted) return false
                collectGpuMaskTriangles(expr.a, smoothedLandmarks, faceState, false, extraFeather, counts) &&
                    collectGpuMaskTriangles(expr.b, smoothedLandmarks, faceState, true, extraFeather, counts)
            }
            is MaskExpr.Feather -> {
                collectGpuMaskTriangles(
                    expr = expr.input,
                    smoothedLandmarks = smoothedLandmarks,
                    faceState = faceState,
                    isSubtracted = isSubtracted,
                    extraFeather = extraFeather + expr.radiusNorm.coerceAtLeast(0f),
                    counts = counts
                )
            }
            is MaskExpr.LuminanceKey -> {
                // Per-pixel luminance glare protection is evaluated in RecolorPass FS_RECOLOR shader
                true
            }
            else -> false
        }
    }

    private fun isGpuEvaluatedLuminanceKey(expr: MaskExpr): Boolean {
        return when (expr) {
            is MaskExpr.LuminanceKey -> true
            is MaskExpr.Feather -> isGpuEvaluatedLuminanceKey(expr.input)
            else -> false
        }
    }

    private fun collectGpuCutoutsOnly(
        expr: MaskExpr,
        smoothedLandmarks: FloatArray,
        faceState: FaceState,
        extraFeather: Float,
        counts: IntArray
    ): Boolean {
        return when (expr) {
            is MaskExpr.Subtract -> {
                val leftOk = if (expr.a is MaskExpr.Subtract) {
                    collectGpuCutoutsOnly(expr.a, smoothedLandmarks, faceState, extraFeather, counts)
                } else {
                    true
                }
                leftOk && collectGpuMaskTriangles(expr.b, smoothedLandmarks, faceState, true, extraFeather, counts)
            }
            else -> true
        }
    }

    private fun evaluateMaskExprWithLandmarks(
        expr: MaskExpr,
        normX: Float,
        normY: Float,
        landmarks: FloatArray,
        faceState: FaceState,
        sampleLuma: Float,
        extraFeatherNorm: Float
    ): Float {
        return when (expr) {
            is MaskExpr.LandmarkLoop -> {
                val fNorm = (expr.featherNorm + extraFeatherNorm).coerceAtLeast(1e-5f)
                val loop = FaceRegions.getRegionLoop(expr.regionName)
                if (isLoopCacheActive) {
                    val cached = preparedLoopCache[expr.regionName]
                    if (cached != null) {
                        if (cached.count < 3) return 0f
                        val maxBoundFeather = if (loop != null && isFaceOvalLoop(loop)) {
                            (fNorm * 4.5f).coerceAtMost(0.18f)
                        } else {
                            fNorm
                        }
                        if (normX < cached.minX - maxBoundFeather || normX > cached.maxX + maxBoundFeather ||
                            normY < cached.minY - maxBoundFeather || normY > cached.maxY + maxBoundFeather
                        ) {
                            return 0f
                        }
                        if (isPointInPolygon(normX, normY, cached.normX, cached.normY, cached.count)) return 1.0f
                        return evaluateFeatheredFalloffOutsidePolygon(
                            px = normX,
                            py = normY,
                            polyX = cached.normX,
                            polyY = cached.normY,
                            count = cached.count,
                            baseFeatherNorm = fNorm,
                            loopIndices = loop
                        )
                    }
                }
                if (loop == null) return 0f
                evaluateLoopMaskAtPt(
                    loopIndices = loop,
                    normX = normX,
                    normY = normY,
                    landmarks = landmarks,
                    faceState = faceState,
                    featherNorm = fNorm
                )
            }
            is MaskExpr.IrisCircle -> {
                evaluateIrisCircleAtPt(
                    isLeft = expr.isLeft,
                    radiusScale = expr.radiusScale,
                    normX = normX,
                    normY = normY,
                    landmarks = landmarks,
                    faceState = faceState,
                    featherNorm = (expr.featherNorm + extraFeatherNorm).coerceAtLeast(1e-5f)
                )
            }
            is MaskExpr.TextureMask -> {
                // Static texture mask defaults to full weight inside face_oval when sampled analytically
                evaluateLoopMaskAtPt(
                    loopIndices = FaceRegions.FACE_OVAL,
                    normX = normX,
                    normY = normY,
                    landmarks = landmarks,
                    faceState = faceState,
                    featherNorm = (DEFAULT_FEATHER_NORM + extraFeatherNorm).coerceAtLeast(1e-5f)
                )
            }
            is MaskExpr.LuminanceKey -> {
                val luma = sampleLuma.coerceIn(0f, 1f)
                val f = (expr.feather + extraFeatherNorm).coerceAtLeast(1e-4f)
                val lowWeight = if (expr.minLuma <= 0f) {
                    1.0f
                } else {
                    smoothstep(expr.minLuma - f, expr.minLuma + f, luma)
                }
                val highWeight = if (expr.maxLuma >= 1f) {
                    1.0f
                } else {
                    1.0f - smoothstep(expr.maxLuma - f, expr.maxLuma + f, luma)
                }
                (lowWeight * highWeight).coerceIn(0f, 1f)
            }
            is MaskExpr.Union -> {
                val va = evaluateMaskExprWithLandmarks(expr.a, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                val vb = evaluateMaskExprWithLandmarks(expr.b, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                max(va, vb)
            }
            is MaskExpr.Intersect -> {
                val va = evaluateMaskExprWithLandmarks(expr.a, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                if (va <= 0f) 0f else {
                    val vb = evaluateMaskExprWithLandmarks(expr.b, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                    (va * vb).coerceIn(0f, 1f)
                }
            }
            is MaskExpr.Subtract -> {
                val va = evaluateMaskExprWithLandmarks(expr.a, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                if (va <= 0f) 0f else {
                    val vb = evaluateMaskExprWithLandmarks(expr.b, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                    max(0f, va - vb)
                }
            }
            is MaskExpr.Feather -> {
                evaluateMaskExprWithLandmarks(
                    expr = expr.input,
                    normX = normX,
                    normY = normY,
                    landmarks = landmarks,
                    faceState = faceState,
                    sampleLuma = sampleLuma,
                    extraFeatherNorm = extraFeatherNorm + expr.radiusNorm.coerceAtLeast(0f)
                )
            }
            is MaskExpr.Invert -> {
                val v = evaluateMaskExprWithLandmarks(expr.input, normX, normY, landmarks, faceState, sampleLuma, extraFeatherNorm)
                (1f - v).coerceIn(0f, 1f)
            }
        }
    }

    private fun evaluateLoopMaskAtPt(
        loopIndices: IntArray,
        normX: Float,
        normY: Float,
        landmarks: FloatArray,
        faceState: FaceState,
        featherNorm: Float
    ): Float {
        if (landmarks.size < UvRegistry.VERTEX_COUNT * 3) return 0f
        val count = mapLoopToNormalizedView(loopIndices, landmarks, faceState, contourNormX, contourNormY)
        if (count < 3) return 0f
        if (isPointInPolygon(normX, normY, contourNormX, contourNormY, count)) return 1.0f
        return evaluateFeatheredFalloffOutsidePolygon(
            px = normX,
            py = normY,
            polyX = contourNormX,
            polyY = contourNormY,
            count = count,
            baseFeatherNorm = featherNorm,
            loopIndices = loopIndices
        )
    }

    private fun evaluateFeatheredFalloffOutsidePolygon(
        px: Float,
        py: Float,
        polyX: FloatArray,
        polyY: FloatArray,
        count: Int,
        baseFeatherNorm: Float,
        loopIndices: IntArray? = null
    ): Float {
        val safeFeather = baseFeatherNorm.coerceIn(1e-5f, 0.15f)
        val isFaceOval = loopIndices != null && isFaceOvalLoop(loopIndices)
        var minDistSq = Float.POSITIVE_INFINITY
        var closestEdgeFeather = safeFeather
        for (i in 0 until count) {
            val j = (i + 1) % count
            val ax = polyX[i]; val ay = polyY[i]
            val bx = polyX[j]; val by = polyY[j]
            val abx = bx - ax
            val aby = by - ay
            val lenSq = abx * abx + aby * aby
            val t = if (lenSq > 1e-8f) {
                (((px - ax) * abx + (py - ay) * aby) / lenSq).coerceIn(0f, 1f)
            } else {
                0f
            }
            val cx = ax + t * abx
            val cy = ay + t * aby
            val dx = px - cx
            val dy = py - cy
            val dSq = dx * dx + dy * dy
            if (dSq < minDistSq) {
                minDistSq = dSq
                if (isFaceOval && i < loopIndices!!.size && j < loopIndices.size) {
                    val wA = cranialForeheadWeightForLandmark(loopIndices[i])
                    val wB = cranialForeheadWeightForLandmark(loopIndices[j])
                    val cranialW = wA + (wB - wA) * t
                    closestEdgeFeather = (safeFeather * (1f + cranialW * 3.5f)).coerceIn(1e-5f, 0.18f)
                }
            }
        }
        val dist = sqrt(minDistSq)
        if (dist >= closestEdgeFeather) return 0.0f
        val normT = (1.0f - (dist / closestEdgeFeather)).coerceIn(0f, 1f)
        return normT * normT * (3f - 2f * normT)
    }

    private fun evaluateIrisCircleAtPt(
        isLeft: Boolean,
        radiusScale: Float,
        normX: Float,
        normY: Float,
        landmarks: FloatArray,
        faceState: FaceState,
        featherNorm: Float
    ): Float {
        if (landmarks.size < UvRegistry.VERTEX_COUNT * 3) return 0f
        val centerIdx = if (isLeft) FaceRegions.LEFT_IRIS_CENTER else FaceRegions.RIGHT_IRIS_CENTER
        val ringIndices = if (isLeft) FaceRegions.LEFT_IRIS_RING else FaceRegions.RIGHT_IRIS_RING
        val eyeIndices = if (isLeft) FaceRegions.LEFT_EYE_CONTOUR else FaceRegions.RIGHT_EYE_CONTOUR

        val cx: Float
        val cy: Float
        val baseRadius: Float
        if (landmarks.size >= UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3) {
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = landmarks[centerIdx * 3],
                v = landmarks[centerIdx * 3 + 1],
                z = landmarks[centerIdx * 3 + 2],
                faceState = faceState,
                viewWidth = 1.0f,
                viewHeight = 1.0f,
                outPt = ptScratch
            )
            cx = ptScratch[0]
            cy = ptScratch[1]
            var rSum = 0f
            for (rIdx in ringIndices) {
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = landmarks[rIdx * 3],
                    v = landmarks[rIdx * 3 + 1],
                    z = landmarks[rIdx * 3 + 2],
                    faceState = faceState,
                    viewWidth = 1.0f,
                    viewHeight = 1.0f,
                    outPt = ptScratch
                )
                rSum += hypot(ptScratch[0] - cx, ptScratch[1] - cy)
            }
            baseRadius = (rSum / ringIndices.size.toFloat()).coerceIn(0.002f, 0.15f)
        } else {
            val eyeCount = mapLoopToNormalizedView(eyeIndices, landmarks, faceState, eyeClipNormX, eyeClipNormY)
            if (eyeCount < 3) return 0f
            var sx = 0f; var sy = 0f
            var minY = Float.POSITIVE_INFINITY; var maxY = Float.NEGATIVE_INFINITY
            for (i in 0 until eyeCount) {
                val ex = eyeClipNormX[i]; val ey = eyeClipNormY[i]
                sx += ex; sy += ey
                if (ey < minY) minY = ey
                if (ey > maxY) maxY = ey
            }
            cx = sx / eyeCount.toFloat()
            cy = sy / eyeCount.toFloat()
            baseRadius = ((maxY - minY) * 0.55f).coerceIn(0.002f, 0.10f)
        }

        val radius = (baseRadius * radiusScale.coerceIn(0.2f, 3.0f)).coerceAtLeast(1e-4f)
        val dist = hypot(normX - cx, normY - cy)
        if (dist <= radius) return 1.0f
        val excess = dist - radius
        if (excess >= featherNorm) return 0.0f
        val t = (1.0f - (excess / featherNorm)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
        val span = (edge1 - edge0).let { if (kotlin.math.abs(it) < 1e-6f) 1e-6f else it }
        val t = ((x - edge0) / span).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun isPointInPolygon(
        px: Float,
        py: Float,
        polyX: FloatArray,
        polyY: FloatArray,
        count: Int
    ): Boolean {
        var inside = false
        var j = count - 1
        for (i in 0 until count) {
            val xi = polyX[i]; val yi = polyY[i]
            val xj = polyX[j]; val yj = polyY[j]
            val intersect = ((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / ((yj - yi).let { if (kotlin.math.abs(it) < 1e-7f) 1e-7f else it }) + xi)
            if (intersect) inside = !inside
            j = i
        }
        return inside
    }

    private fun distanceToPolygonBoundary(
        px: Float,
        py: Float,
        polyX: FloatArray,
        polyY: FloatArray,
        count: Int
    ): Float {
        var minDistSq = Float.POSITIVE_INFINITY
        for (i in 0 until count) {
            val j = (i + 1) % count
            val ax = polyX[i]; val ay = polyY[i]
            val bx = polyX[j]; val by = polyY[j]
            val abx = bx - ax
            val aby = by - ay
            val lenSq = abx * abx + aby * aby
            val t = if (lenSq > 1e-8f) {
                (((px - ax) * abx + (py - ay) * aby) / lenSq).coerceIn(0f, 1f)
            } else {
                0f
            }
            val cx = ax + t * abx
            val cy = ay + t * aby
            val dx = px - cx
            val dy = py - cy
            val dSq = dx * dx + dy * dy
            if (dSq < minDistSq) minDistSq = dSq
        }
        return sqrt(minDistSq)
    }

    /**
     * Computes the distance along ray `(ox, oy) + t * (dirX, dirY)` (`t >= 0`) to the boundary of
     * polygon `[polyX, polyY]`. If `(ox, oy)` is outside the polygon, returns `0f`.
     */
    private fun rayPolygonBoundaryDistance(
        ox: Float,
        oy: Float,
        dirX: Float,
        dirY: Float,
        polyX: FloatArray,
        polyY: FloatArray,
        polyCount: Int
    ): Float {
        if (!isPointInPolygon(ox, oy, polyX, polyY, polyCount)) return 0f
        var minPositiveT = Float.POSITIVE_INFINITY
        for (i in 0 until polyCount) {
            val j = (i + 1) % polyCount
            val ax = polyX[i]; val ay = polyY[i]
            val bx = polyX[j]; val by = polyY[j]
            val sx = bx - ax
            val sy = by - ay
            // Solve: O + t * D = A + u * S
            val denom = dirX * sy - dirY * sx
            if (kotlin.math.abs(denom) > 1e-7f) {
                val aox = ax - ox
                val aoy = ay - oy
                val t = (aox * sy - aoy * sx) / denom
                val u = (aox * dirY - aoy * dirX) / denom
                if (t >= 0f && u in -1e-4f..1.0001f) {
                    if (t < minPositiveT) minPositiveT = t
                }
            }
        }
        return minPositiveT
    }

    @JvmOverloads
    fun release(resourceOwner: GlResourceOwner? = null) {
        if (resourceOwner != null) {
            if (fboId != 0) resourceOwner.queueDeleteFramebuffer(fboId)
            if (maskTextureId != 0) resourceOwner.queueDeleteTexture(maskTextureId)
            if (vao != 0) resourceOwner.queueDeleteVao(vao)
            if (vbo != 0) resourceOwner.queueDeleteBuffer(vbo)
            if (program != 0) resourceOwner.queueDeleteProgram(program)
        } else {
            try {
                if (fboId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
                if (maskTextureId != 0) GLES30.glDeleteTextures(1, intArrayOf(maskTextureId), 0)
                if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
                if (vbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
                if (program != 0) GLES30.glDeleteProgram(program)
            } catch (_: Throwable) {
            }
        }
        fboId = 0
        maskTextureId = 0
        vao = 0
        vbo = 0
        program = 0
        fboWidth = 0
        fboHeight = 0
        isInitialized = false
        smoother.reset()
    }

    private fun compileProgram(vsSource: String, fsSource: String): Int {
        val vs = compileShader(GLES30.GL_VERTEX_SHADER, vsSource)
        val fs = compileShader(GLES30.GL_FRAGMENT_SHADER, fsSource)
        if (vs == 0 || fs == 0) return 0
        val prog = GLES30.glCreateProgram()
        GLES30.glAttachShader(prog, vs)
        GLES30.glAttachShader(prog, fs)
        GLES30.glLinkProgram(prog)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, linked, 0)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        if (linked[0] == 0) {
            GLES30.glDeleteProgram(prog)
            return 0
        }
        return prog
    }

    private fun compileShader(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src)
        GLES30.glCompileShader(s)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            GLES30.glDeleteShader(s)
            return 0
        }
        return s
    }

    private fun hasActiveGlContext(): Boolean {
        return try {
            val ctx = EGL14.eglGetCurrentContext()
            ctx != null && ctx != EGL14.EGL_NO_CONTEXT
        } catch (_: Throwable) {
            false
        }
    }
}
