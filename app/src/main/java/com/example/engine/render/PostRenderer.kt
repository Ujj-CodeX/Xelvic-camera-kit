package com.example.engine.render

import android.graphics.Bitmap
import android.opengl.EGL14
import android.opengl.GLES30
import com.example.engine.pkg.PackageValidator
import com.example.engine.pkg.UvRegistry
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.CompiledNode
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * PostRenderer (README Section 7, Section 11.4, Section 21 Step 10 & Flow 5 Chunk Primitives-2):
 *
 * Implements the engine's bounded, allow-listed fullscreen post-processing effects (`node.post@1`):
 * - `colorGrade` (`post.colorGrade@1`): Warmth, contrast, saturation, exposure/brightness grading.
 * - `blur` (`post.blur@1`): Bounded 9-tap separable/box gaussian blur.
 * - `skinSoften` / `bilateral` (`post.skinSoften@1`, `post.bilateral@1`): Half-resolution (`0.5x`)
 *   edge-preserving bilateral smoothing gated by `$mask:skin` (`face_oval - eyes - eyebrows - lips`)
 *   with high-frequency pore `detailPreserve` re-injection so eyes, lashes, lips, and background remain 100% sharp.
 * - `softFocus` (`post.softFocus@1`): Quarter-resolution (`0.25x`) Kawase highlight halation +
 *   screen/softLight bloom that makes highlights glow while preserving dark edge contrast.
 * - `warp` (`post.warp@1`): 1-Euro smoothed landmark-driven subtle face morph (`jawSlim`, `eyeEnlarge`,
 *   `noseSlim`, `chinVLine`) with strict `0.0` displacement outside `face_oval`.
 * - `colorMatrix` (`post.colorMatrix@1`): Brightness, contrast, saturation, hue, temperature, tint,
 *   sepia, and black & white grading.
 * - `vignette` (`post.vignette@1`): Smooth radial edge falloff (`radius`, `softness`, `amount`).
 */
class PostRenderer {

    data class ParamSpec(
        val defaultValue: Float,
        val minValue: Float,
        val maxValue: Float
    )

    data class ExecutedPostPass(
        val nodeId: String,
        val effect: String,
        val renderOrder: Int,
        val params: Map<String, Float>
    )

    companion object {
        /**
         * Original Step 10 allow-listed set preserved for backward compatibility with `PostProcessingStep10UnitTest`.
         */
        val ALLOW_LISTED_EFFECTS: Set<String> = setOf(
            "colorGrade",
            "blur",
            "skinSoften"
        )

        /**
         * Full set of engine-supported post effects including Flow 5 (Chunk Primitives-2) beauty, softFocus, warp,
         * colorMatrix, and vignette primitives.
         */
        val SUPPORTED_EFFECTS: Set<String> = setOf(
            "colorGrade",
            "blur",
            "skinSoften",
            "bilateral",
            "softFocus",
            "warp",
            "colorMatrix",
            "vignette",
            "lut",
            "grain",
            "sharpen"
        )

        private val SKIN_SOFTEN_PARAM_SPECS = mapOf(
            "amount" to ParamSpec(defaultValue = 0.35f, minValue = 0.0f, maxValue = 1.0f),
            "radius" to ParamSpec(defaultValue = 2.0f, minValue = 0.0f, maxValue = 12.0f),
            "edgePreserve" to ParamSpec(defaultValue = 0.7f, minValue = 0.0f, maxValue = 1.0f),
            "detailPreserve" to ParamSpec(defaultValue = 0.25f, minValue = 0.0f, maxValue = 1.0f),
            "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
        )

        val EFFECT_PARAM_SPECS: Map<String, Map<String, ParamSpec>> = mapOf(
            "colorGrade" to mapOf(
                "warmth" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "tint" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "contrast" to ParamSpec(defaultValue = 1.0f, minValue = 0.5f, maxValue = 2.0f),
                "saturation" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 2.0f),
                "exposure" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "brightness" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "blur" to mapOf(
                "radius" to ParamSpec(defaultValue = 1.5f, minValue = 0.0f, maxValue = 8.0f),
                "amount" to ParamSpec(defaultValue = 0.5f, minValue = 0.0f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "skinSoften" to SKIN_SOFTEN_PARAM_SPECS,
            "bilateral" to SKIN_SOFTEN_PARAM_SPECS,
            "softFocus" to mapOf(
                "amount" to ParamSpec(defaultValue = 0.35f, minValue = 0.0f, maxValue = 1.0f),
                "radius" to ParamSpec(defaultValue = 4.0f, minValue = 0.0f, maxValue = 16.0f),
                "threshold" to ParamSpec(defaultValue = 0.55f, minValue = 0.0f, maxValue = 1.0f),
                "warmth" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "warp" to mapOf(
                "jawSlim" to ParamSpec(defaultValue = 0.0f, minValue = 0.0f, maxValue = 1.0f),
                "eyeEnlarge" to ParamSpec(defaultValue = 0.0f, minValue = 0.0f, maxValue = 1.0f),
                "noseSlim" to ParamSpec(defaultValue = 0.0f, minValue = 0.0f, maxValue = 1.0f),
                "chinVLine" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "radiusFW" to ParamSpec(defaultValue = 0.40f, minValue = 0.05f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "colorMatrix" to mapOf(
                "brightness" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "contrast" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 2.0f),
                "saturation" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 2.0f),
                "hue" to ParamSpec(defaultValue = 0.0f, minValue = -180.0f, maxValue = 180.0f),
                "temperature" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "tint" to ParamSpec(defaultValue = 0.0f, minValue = -1.0f, maxValue = 1.0f),
                "sepia" to ParamSpec(defaultValue = 0.0f, minValue = 0.0f, maxValue = 1.0f),
                "bw" to ParamSpec(defaultValue = 0.0f, minValue = 0.0f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "vignette" to mapOf(
                "radius" to ParamSpec(defaultValue = 0.75f, minValue = 0.0f, maxValue = 2.0f),
                "softness" to ParamSpec(defaultValue = 0.45f, minValue = 0.01f, maxValue = 2.0f),
                "amount" to ParamSpec(defaultValue = 0.40f, minValue = 0.0f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "lut" to mapOf(
                "amount" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f),
                "gridSize" to ParamSpec(defaultValue = 64.0f, minValue = 4.0f, maxValue = 64.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "grain" to mapOf(
                "amount" to ParamSpec(defaultValue = 0.15f, minValue = 0.0f, maxValue = 1.0f),
                "size" to ParamSpec(defaultValue = 1.0f, minValue = 0.25f, maxValue = 8.0f),
                "seed" to ParamSpec(defaultValue = 0.0f, minValue = -10000.0f, maxValue = 10000.0f),
                "lumaWeight" to ParamSpec(defaultValue = 0.75f, minValue = 0.0f, maxValue = 1.0f),
                "speed" to ParamSpec(defaultValue = 24.0f, minValue = 0.0f, maxValue = 120.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            ),
            "sharpen" to mapOf(
                "amount" to ParamSpec(defaultValue = 0.35f, minValue = 0.0f, maxValue = 1.0f),
                "radius" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 4.0f),
                "threshold" to ParamSpec(defaultValue = 0.02f, minValue = 0.0f, maxValue = 1.0f),
                "intensity" to ParamSpec(defaultValue = 1.0f, minValue = 0.0f, maxValue = 1.0f)
            )
        )

        private val FORBIDDEN_SHADER_FIELDS: Set<String> = setOf(
            "shader", "glsl", "vertexShader", "fragmentShader", "code", "source", "script", "program"
        )

        private val ALLOWED_POST_NODE_FIELDS: Set<String> = setOf(
            "id", "type", "effect", "params", "visible", "opacity", "renderOrder", "tags", "parent", "mask", "region",
            "asset", "lut", "map"
        )

        fun normalizeEffectName(effectName: String): String = effectName.trim().removeSuffix("@1")

        fun getDefaultParamsForEffect(effectName: String): Map<String, Float> {
            val specs = EFFECT_PARAM_SPECS[normalizeEffectName(effectName)] ?: return emptyMap()
            val out = LinkedHashMap<String, Float>(specs.size)
            for ((k, spec) in specs) {
                out[k] = spec.defaultValue
            }
            return out
        }

        fun isAssetUsedAsLut(effect: CompiledEffect, assetId: String): Boolean {
            if (assetId.isEmpty()) return false
            if (effect.assets[assetId]?.type == "lut") return true
            val referencedByPostNode = effect.nodes.any { node ->
                node.postAssetId == assetId && normalizeEffectName(node.postEffect ?: "") == "lut"
            }
            if (referencedByPostNode) return true
            return effect.passGraph.passes.any { passNode ->
                passNode.assetId == assetId &&
                    normalizeEffectName(passNode.templateId.removePrefix("pass.").removePrefix("post.")) == "lut"
            }
        }

        fun isEffectSupported(effectName: String): Boolean {
            return SUPPORTED_EFFECTS.contains(normalizeEffectName(effectName))
        }

        fun getCapabilityForEffect(effectName: String): String? {
            val base = normalizeEffectName(effectName)
            return if (SUPPORTED_EFFECTS.contains(base)) "post.$base@1" else null
        }

        fun getCapabilities(): Set<String> {
            return SUPPORTED_EFFECTS.flatMap { listOf("post.$it@1", "pass.$it@1") }.toSet() + "node.post@1"
        }

        /**
         * Validates a `post` node in `manifest.json` against the allow-listed effect registry,
         * parameter schema/bounds, and custom shader prohibition.
         */
        fun validatePostNode(
            nodeJson: JSONObject,
            manifestParamsJson: JSONObject?
        ): PackageValidator.ValidationResult {
            val nodeId = nodeJson.optString("id", "unnamed_post")

            for (key in nodeJson.keys()) {
                if (key in FORBIDDEN_SHADER_FIELDS) {
                    return PackageValidator.ValidationResult.fail(
                        "Post node '$nodeId' contains forbidden shader/script field '$key' (custom shaders are prohibited)"
                    )
                }
                if (key !in ALLOWED_POST_NODE_FIELDS) {
                    return PackageValidator.ValidationResult.fail(
                        "Post node '$nodeId' contains unknown field '$key'"
                    )
                }
            }

            val rawEffect = nodeJson.optString("effect", "").trim()
            if (rawEffect.isEmpty()) {
                return PackageValidator.ValidationResult.fail(
                    "Post node '$nodeId' is missing required 'effect' field"
                )
            }
            val effectName = normalizeEffectName(rawEffect)
            val paramSpecs = EFFECT_PARAM_SPECS[effectName]
                ?: return PackageValidator.ValidationResult.fail(
                    "Post node '$nodeId' specifies unknown or unsupported effect '$rawEffect'"
                )

            val paramsObj = nodeJson.optJSONObject("params")
            if (paramsObj != null) {
                for (pKey in paramsObj.keys()) {
                    if (pKey in FORBIDDEN_SHADER_FIELDS) {
                        return PackageValidator.ValidationResult.fail(
                            "Post node '$nodeId' params contains forbidden shader/script field '$pKey'"
                        )
                    }
                    if (effectName == "lut" && (pKey == "asset" || pKey == "lut" || pKey == "map")) {
                        val rawAsset = paramsObj.optString(pKey, "").trim()
                        if (rawAsset.isEmpty()) {
                            return PackageValidator.ValidationResult.fail(
                                "Post node '$nodeId' lut parameter '$pKey' must be a non-empty asset ID"
                            )
                        }
                        continue
                    }
                    val spec = paramSpecs[pKey]
                        ?: return PackageValidator.ValidationResult.fail(
                            "Post node '$nodeId' (effect '$effectName') does not support parameter '$pKey'"
                        )

                    val rawVal = paramsObj.get(pKey)
                    when (rawVal) {
                        is Number -> {
                            val v = rawVal.toFloat()
                            if (!v.isFinite() || v < spec.minValue || v > spec.maxValue) {
                                return PackageValidator.ValidationResult.fail(
                                    "Post node '$nodeId' parameter '$pKey' value $v is out of supported range [${spec.minValue}, ${spec.maxValue}]"
                                )
                            }
                        }
                        is String -> {
                            if (!rawVal.startsWith("$") || rawVal.length < 2) {
                                return PackageValidator.ValidationResult.fail(
                                    "Post node '$nodeId' parameter '$pKey' has invalid string value '$rawVal' (expected numeric literal or '\$paramName')"
                                )
                            }
                            val refName = rawVal.substring(1)
                            val manifestParam = manifestParamsJson?.optJSONObject(refName)
                                ?: return PackageValidator.ValidationResult.fail(
                                    "Post node '$nodeId' parameter '$pKey' references undeclared manifest param '$rawVal'"
                                )
                            if (manifestParam.optString("type", "") != "float") {
                                return PackageValidator.ValidationResult.fail(
                                    "Post node '$nodeId' parameter '$pKey' references non-float manifest param '$rawVal'"
                                )
                            }
                            val defVal = manifestParam.optDouble("default", Double.NaN).toFloat()
                            if (!defVal.isFinite() || defVal < spec.minValue || defVal > spec.maxValue) {
                                return PackageValidator.ValidationResult.fail(
                                    "Post node '$nodeId' parameter '$pKey' bound param '$rawVal' default $defVal is out of range [${spec.minValue}, ${spec.maxValue}]"
                                )
                            }
                            if (manifestParam.has("min")) {
                                val minDecl = manifestParam.optDouble("min", Double.NaN).toFloat()
                                if (!minDecl.isFinite() || minDecl < spec.minValue) {
                                    return PackageValidator.ValidationResult.fail(
                                        "Post node '$nodeId' parameter '$pKey' bound param '$rawVal' min $minDecl exceeds effect lower bound ${spec.minValue}"
                                    )
                                }
                            }
                            if (manifestParam.has("max")) {
                                val maxDecl = manifestParam.optDouble("max", Double.NaN).toFloat()
                                if (!maxDecl.isFinite() || maxDecl > spec.maxValue) {
                                    return PackageValidator.ValidationResult.fail(
                                        "Post node '$nodeId' parameter '$pKey' bound param '$rawVal' max $maxDecl exceeds effect upper bound ${spec.maxValue}"
                                    )
                                }
                            }
                        }
                        else -> {
                            return PackageValidator.ValidationResult.fail(
                                "Post node '$nodeId' parameter '$pKey' has unsupported value type"
                            )
                        }
                    }
                }
            }

            return PackageValidator.ValidationResult.success()
        }

        /**
         * Resolves a post node's parameters against manifest `params` defaults at compile time.
         */
        fun resolvePostParams(
            nodeId: String,
            effectName: String,
            paramsJson: JSONObject?,
            resolvedManifestParams: Map<String, Any>
        ): Map<String, Float> {
            val normEffect = normalizeEffectName(effectName)
            val specs = EFFECT_PARAM_SPECS[normEffect]
                ?: throw IllegalArgumentException("Post node '$nodeId' uses unsupported post effect '$effectName'")

            val out = HashMap<String, Float>(specs.size)
            for ((k, spec) in specs) {
                out[k] = spec.defaultValue
            }

            if (paramsJson != null) {
                for (pKey in paramsJson.keys()) {
                    if (normEffect == "lut" && (pKey == "asset" || pKey == "lut" || pKey == "map")) {
                        continue
                    }
                    val spec = specs[pKey]
                        ?: throw IllegalArgumentException("Post node '$nodeId' (effect '$normEffect') has unknown parameter '$pKey'")
                    val rawVal = paramsJson.get(pKey)
                    val numericVal = when (rawVal) {
                        is Number -> rawVal.toFloat()
                        is String -> {
                            if (!rawVal.startsWith("$")) {
                                throw IllegalArgumentException("Post node '$nodeId' parameter '$pKey' invalid reference '$rawVal'")
                            }
                            val refKey = rawVal.substring(1)
                            val resolved = resolvedManifestParams[refKey]
                                ?: throw IllegalArgumentException("Post node '$nodeId' parameter '$pKey' references missing param '$rawVal'")
                            (resolved as? Number)?.toFloat()
                                ?: throw IllegalArgumentException("Post node '$nodeId' parameter '$pKey' param '$rawVal' is not a float")
                        }
                        else -> throw IllegalArgumentException("Post node '$nodeId' parameter '$pKey' must be float")
                    }
                    if (!numericVal.isFinite() || numericVal < spec.minValue || numericVal > spec.maxValue) {
                        throw IllegalArgumentException(
                            "Post node '$nodeId' parameter '$pKey' value $numericVal out of range [${spec.minValue}, ${spec.maxValue}]"
                        )
                    }
                    out[pKey] = numericVal
                }
            }
            return out
        }

        private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
            if (edge1 <= edge0) return if (x >= edge0) 1f else 0f
            val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }

        /**
         * Maps landmark `lmId` in `faceState` (or optional [landmarksOverride], e.g. 1-Euro smoothed landmarks)
         * into normalized view coordinates `[u, v]` in `[0..1, 0..1]`, incorporating the unified 3D rigid
         * pose-prediction delta when active.
         */
        @JvmOverloads
        fun getLandmarkNormalizedView(
            faceState: FaceState,
            lmId: Int,
            landmarksOverride: FloatArray? = null
        ): FloatArray? {
            val lms = landmarksOverride ?: faceState.landmarks ?: return null
            val baseIdx = lmId * 3
            if (baseIdx + 2 >= lms.size) return null
            val out = FloatArray(2)
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = lms[baseIdx],
                v = lms[baseIdx + 1],
                z = lms[baseIdx + 2],
                faceState = faceState,
                viewWidth = 1.0f,
                viewHeight = 1.0f,
                outPt = out
            )
            return out
        }

        /**
         * Pre-computed per-frame 2D normalized view anchors for `post.warp@1` (`jawSlim`, `eyeEnlarge`, `noseSlim`, `chinVLine`),
         * hoisted outside per-pixel loops and shared across GPU and CPU warp execution.
         */
        class WarpAnchorCache {
            var faceCenterU: Float = 0.5f
            var faceCenterV: Float = 0.5f
            var faceHalfW: Float = 0.16f
            var faceHalfH: Float = 0.20f
            val noseTip = FloatArray(2)
            val chinTip = FloatArray(2)
            val leftJaw = FloatArray(2)
            val rightJaw = FloatArray(2)
            val leftEyeCenter = FloatArray(2)
            val rightEyeCenter = FloatArray(2)
            val leftAla = FloatArray(2)
            val rightAla = FloatArray(2)
        }

        @JvmOverloads
        fun populateWarpAnchors(
            faceState: FaceState,
            landmarksOverride: FloatArray? = null,
            outCache: WarpAnchorCache = WarpAnchorCache()
        ): WarpAnchorCache {
            val noseBridge = getLandmarkNormalizedView(faceState, 6, landmarksOverride) ?: floatArrayOf(0.5f, 0.48f)
            val noseTip = getLandmarkNormalizedView(faceState, 4, landmarksOverride) ?: floatArrayOf(noseBridge[0], noseBridge[1] + 0.05f)
            val foreheadTop = getLandmarkNormalizedView(faceState, 10, landmarksOverride) ?: floatArrayOf(noseBridge[0], noseBridge[1] - 0.18f)
            val chinTip = getLandmarkNormalizedView(faceState, 152, landmarksOverride) ?: floatArrayOf(noseBridge[0], noseBridge[1] + 0.22f)
            val leftCheek = getLandmarkNormalizedView(faceState, 234, landmarksOverride) ?: floatArrayOf(noseBridge[0] - 0.16f, noseBridge[1] + 0.02f)
            val rightCheek = getLandmarkNormalizedView(faceState, 454, landmarksOverride) ?: floatArrayOf(noseBridge[0] + 0.16f, noseBridge[1] + 0.02f)

            val fcU = (leftCheek[0] + rightCheek[0] + noseBridge[0]) / 3f
            val fcV = (foreheadTop[1] + chinTip[1]) * 0.5f
            val fhW = max(0.06f, abs(rightCheek[0] - leftCheek[0]) * 0.5f)
            val fhH = max(0.08f, abs(chinTip[1] - foreheadTop[1]) * 0.5f)

            outCache.faceCenterU = fcU
            outCache.faceCenterV = fcV
            outCache.faceHalfW = fhW
            outCache.faceHalfH = fhH
            outCache.noseTip[0] = noseTip[0]
            outCache.noseTip[1] = noseTip[1]
            outCache.chinTip[0] = chinTip[0]
            outCache.chinTip[1] = chinTip[1]

            val leftJaw = getLandmarkNormalizedView(faceState, 172, landmarksOverride)
            outCache.leftJaw[0] = leftJaw?.get(0) ?: (fcU - fhW * 0.82f)
            outCache.leftJaw[1] = leftJaw?.get(1) ?: (fcV + fhH * 0.48f)

            val rightJaw = getLandmarkNormalizedView(faceState, 397, landmarksOverride)
            outCache.rightJaw[0] = rightJaw?.get(0) ?: (fcU + fhW * 0.82f)
            outCache.rightJaw[1] = rightJaw?.get(1) ?: (fcV + fhH * 0.48f)

            val lOuter = getLandmarkNormalizedView(faceState, 33, landmarksOverride)
            val lInner = getLandmarkNormalizedView(faceState, 133, landmarksOverride)
            val lTop = getLandmarkNormalizedView(faceState, 159, landmarksOverride)
            val lBot = getLandmarkNormalizedView(faceState, 145, landmarksOverride)
            if (lOuter != null && lInner != null && lTop != null && lBot != null) {
                outCache.leftEyeCenter[0] = (lOuter[0] + lInner[0] + lTop[0] + lBot[0]) * 0.25f
                outCache.leftEyeCenter[1] = (lOuter[1] + lInner[1] + lTop[1] + lBot[1]) * 0.25f
            } else if (lOuter != null && lInner != null) {
                outCache.leftEyeCenter[0] = (lOuter[0] + lInner[0]) * 0.5f
                outCache.leftEyeCenter[1] = (lOuter[1] + lInner[1]) * 0.5f
            } else {
                outCache.leftEyeCenter[0] = fcU - fhW * 0.36f
                outCache.leftEyeCenter[1] = fcV - fhH * 0.20f
            }

            val rOuter = getLandmarkNormalizedView(faceState, 263, landmarksOverride)
            val rInner = getLandmarkNormalizedView(faceState, 362, landmarksOverride)
            val rTop = getLandmarkNormalizedView(faceState, 386, landmarksOverride)
            val rBot = getLandmarkNormalizedView(faceState, 374, landmarksOverride)
            if (rOuter != null && rInner != null && rTop != null && rBot != null) {
                outCache.rightEyeCenter[0] = (rOuter[0] + rInner[0] + rTop[0] + rBot[0]) * 0.25f
                outCache.rightEyeCenter[1] = (rOuter[1] + rInner[1] + rTop[1] + rBot[1]) * 0.25f
            } else if (rOuter != null && rInner != null) {
                outCache.rightEyeCenter[0] = (rOuter[0] + rInner[0]) * 0.5f
                outCache.rightEyeCenter[1] = (rOuter[1] + rInner[1]) * 0.5f
            } else {
                outCache.rightEyeCenter[0] = fcU + fhW * 0.36f
                outCache.rightEyeCenter[1] = fcV - fhH * 0.20f
            }

            val leftAla = getLandmarkNormalizedView(faceState, 129, landmarksOverride)
            outCache.leftAla[0] = leftAla?.get(0) ?: (noseTip[0] - fhW * 0.22f)
            outCache.leftAla[1] = leftAla?.get(1) ?: noseTip[1]

            val rightAla = getLandmarkNormalizedView(faceState, 358, landmarksOverride)
            outCache.rightAla[0] = rightAla?.get(0) ?: (noseTip[0] + fhW * 0.22f)
            outCache.rightAla[1] = rightAla?.get(1) ?: noseTip[1]

            return outCache
        }

        /**
         * Computes the forward screen-space UV displacement `(dU, dV)` at normalized screen coordinate `(u, v)`
         * for `post.warp@1` (`jawSlim`, `eyeEnlarge`, `noseSlim`, `chinVLine`) driven by 1-Euro smoothed landmarks.
         *
         * Guarantees:
         * - Smooth cubic radial falloff around target anatomical landmarks.
         * - Strictly `0.0f` displacement outside `face_oval` (`dU == 0f && dV == 0f`), ensuring zero background distortion.
         * - Deterministic output for identical 1-Euro smoothed `FaceState` inputs (zero stationary jitter).
         */
        @JvmOverloads
        fun computeWarpForwardDisplacementUv(
            u: Float,
            v: Float,
            faceState: FaceState,
            params: Map<String, Float>,
            @Suppress("UNUSED_PARAMETER") aspect: Float = 1f,
            outDeltaUv: FloatArray,
            landmarksOverride: FloatArray? = null,
            anchorCache: WarpAnchorCache? = null
        ) {
            outDeltaUv[0] = 0f
            outDeltaUv[1] = 0f
            if (!faceState.isVisible || faceState.fadeWeight <= 0.001f) return
            val lms = landmarksOverride ?: faceState.landmarks
            if (lms == null || lms.size < UvRegistry.VERTEX_COUNT * 3) return

            val intensity = (params["intensity"] ?: 1.0f).coerceIn(0f, 1f) * faceState.fadeWeight.coerceIn(0f, 1f)
            if (intensity <= 0.001f) return

            val jawSlim = (params["jawSlim"] ?: 0.0f).coerceIn(0f, 1f)
            val eyeEnlarge = (params["eyeEnlarge"] ?: 0.0f).coerceIn(0f, 1f)
            val noseSlim = (params["noseSlim"] ?: 0.0f).coerceIn(0f, 1f)
            val chinVLine = (params["chinVLine"] ?: 0.0f).coerceIn(-1f, 1f)
            val radiusScale = (params["radiusFW"] ?: 0.40f).coerceIn(0.05f, 1.0f)

            if (jawSlim <= 1e-4f && eyeEnlarge <= 1e-4f && noseSlim <= 1e-4f && abs(chinVLine) <= 1e-4f) {
                return
            }

            val anchors = anchorCache ?: populateWarpAnchors(faceState, landmarksOverride)
            val faceCenterU = anchors.faceCenterU
            val faceCenterV = anchors.faceCenterV
            val faceHalfW = anchors.faceHalfW
            val faceHalfH = anchors.faceHalfH
            val noseTip = anchors.noseTip
            val chinTip = anchors.chinTip

            // Strict face_oval outer boundary guard: outside elliptical face oval envelope -> 0.0f displacement
            val normOvalX = (u - faceCenterU) / (faceHalfW * 1.05f)
            val normOvalY = (v - faceCenterV) / (faceHalfH * 1.05f)
            val ovalDistSq = normOvalX * normOvalX + normOvalY * normOvalY
            if (ovalDistSq >= 1.0f) {
                return
            }
            val ovalDist = sqrt(ovalDistSq)
            val boundaryEnvelope = 1f - smoothstep(0.72f, 1.0f, ovalDist)
            if (boundaryEnvelope <= 1e-5f) return

            val faceWidthUv = faceHalfW * 2f
            var dU = 0f
            var dV = 0f

            // 1. Jaw Slimming: pulls lower cheek/jaw contour (#172/#58 and #397/#288) inward toward midline
            if (jawSlim > 1e-4f) {
                val leftJaw = anchors.leftJaw
                val rightJaw = anchors.rightJaw
                val jawRadius = max(0.03f, faceWidthUv * radiusScale * 0.85f)
                val maxJawShift = faceWidthUv * 0.06f * jawSlim

                val distL = hypot(u - leftJaw[0], v - leftJaw[1])
                if (distL < jawRadius) {
                    val wL = 1f - smoothstep(0f, jawRadius, distL)
                    val dirToMidU = (faceCenterU - leftJaw[0]).let { if (abs(it) > 1e-4f) it / abs(it) else 1f }
                    dU += dirToMidU * maxJawShift * wL
                    dV -= maxJawShift * 0.18f * wL
                }

                val distR = hypot(u - rightJaw[0], v - rightJaw[1])
                if (distR < jawRadius) {
                    val wR = 1f - smoothstep(0f, jawRadius, distR)
                    val dirToMidU = (faceCenterU - rightJaw[0]).let { if (abs(it) > 1e-4f) it / abs(it) else -1f }
                    dU += dirToMidU * maxJawShift * wR
                    dV -= maxJawShift * 0.18f * wR
                }
            }

            // 2. Eye Enlargement: subtle radial expansion outward from left & right eye centroids (clamped <= 8% eye radius)
            if (eyeEnlarge > 1e-4f) {
                val leftEyeCenter = anchors.leftEyeCenter
                val rightEyeCenter = anchors.rightEyeCenter
                val eyeRadius = max(0.025f, faceWidthUv * 0.22f)
                val maxExpandScale = 0.08f * eyeEnlarge

                val rxL = u - leftEyeCenter[0]
                val ryL = v - leftEyeCenter[1]
                val rDistL = hypot(rxL, ryL)
                if (rDistL in 1e-5f..eyeRadius) {
                    val normR = rDistL / eyeRadius
                    val bell = normR * (1f - smoothstep(0.25f, 1.0f, normR))
                    dU += rxL * maxExpandScale * (bell / normR)
                    dV += ryL * maxExpandScale * (bell / normR)
                }

                val rxR = u - rightEyeCenter[0]
                val ryR = v - rightEyeCenter[1]
                val rDistR = hypot(rxR, ryR)
                if (rDistR in 1e-5f..eyeRadius) {
                    val normR = rDistR / eyeRadius
                    val bell = normR * (1f - smoothstep(0.25f, 1.0f, normR))
                    dU += rxR * maxExpandScale * (bell / normR)
                    dV += ryR * maxExpandScale * (bell / normR)
                }
            }

            // 3. Nose Slimming: pulls left/right nasal alar wings (#129, #358) horizontally toward nose tip (#4)
            if (noseSlim > 1e-4f) {
                val leftAla = anchors.leftAla
                val rightAla = anchors.rightAla
                val noseRadius = max(0.02f, faceWidthUv * 0.20f)
                val maxNoseShift = faceWidthUv * 0.045f * noseSlim

                val dLAla = hypot(u - leftAla[0], v - leftAla[1])
                if (dLAla < noseRadius) {
                    val w = 1f - smoothstep(0f, noseRadius, dLAla)
                    val dirU = (noseTip[0] - leftAla[0]).let { if (abs(it) > 1e-4f) it / abs(it) else 1f }
                    dU += dirU * maxNoseShift * w
                }
                val dRAla = hypot(u - rightAla[0], v - rightAla[1])
                if (dRAla < noseRadius) {
                    val w = 1f - smoothstep(0f, noseRadius, dRAla)
                    val dirU = (noseTip[0] - rightAla[0]).let { if (abs(it) > 1e-4f) it / abs(it) else -1f }
                    dU += dirU * maxNoseShift * w
                }
            }

            // 4. Chin V-Line: subtle vertical/inward taper around chin tip (#152)
            if (abs(chinVLine) > 1e-4f) {
                val chinRadius = max(0.03f, faceWidthUv * 0.28f)
                val dChin = hypot(u - chinTip[0], v - chinTip[1])
                if (dChin < chinRadius) {
                    val w = 1f - smoothstep(0f, chinRadius, dChin)
                    dV -= chinVLine * (faceHalfH * 0.055f) * w
                    dU += (faceCenterU - u) * (0.08f * chinVLine) * w
                }
            }

            outDeltaUv[0] = dU * boundaryEnvelope * intensity
            outDeltaUv[1] = dV * boundaryEnvelope * intensity
        }

        /**
         * Computes the backward source-sampling UV `(sampleU, sampleV)` for output pixel `(u, v)`.
         */
        @JvmOverloads
        fun computeWarpSampleUv(
            u: Float,
            v: Float,
            faceState: FaceState,
            params: Map<String, Float>,
            aspect: Float = 1f,
            outSampleUv: FloatArray,
            landmarksOverride: FloatArray? = null
        ) {
            val delta = FloatArray(2)
            computeWarpForwardDisplacementUv(u, v, faceState, params, aspect, delta, landmarksOverride)
            outSampleUv[0] = (u - delta[0]).coerceIn(0f, 1f)
            outSampleUv[1] = (v - delta[1]).coerceIn(0f, 1f)
        }

        /**
         * Generates an identity 3D Hald/strip LUT Bitmap for testing or neutral grading.
         * When `asSquareHald = true` and `gridSize = 64`, produces a `512x512` 8x8 tile Hald LUT;
         * when `asSquareHald = false`, produces a `(gridSize * gridSize) x gridSize` horizontal strip LUT.
         */
        @JvmOverloads
        fun createIdentityLutBitmap(gridSize: Int = 16, asSquareHald: Boolean = false): Bitmap {
            val n = gridSize.coerceIn(4, 64)
            val tilesPerRow = if (asSquareHald) sqrt(n.toDouble()).roundToInt().coerceAtLeast(1) else n
            val tilesPerCol = ((n + tilesPerRow - 1) / tilesPerRow).coerceAtLeast(1)
            val bmpW = tilesPerRow * n
            val bmpH = tilesPerCol * n
            val bmp = Bitmap.createBitmap(bmpW, bmpH, Bitmap.Config.ARGB_8888)
            val pixels = IntArray(bmpW * bmpH)
            val denom = (n - 1).coerceAtLeast(1).toFloat()
            for (bIdx in 0 until n) {
                val tileX = (bIdx % tilesPerRow) * n
                val tileY = (bIdx / tilesPerRow) * n
                val bVal = ((bIdx / denom) * 255f).roundToInt().coerceIn(0, 255)
                for (gIdx in 0 until n) {
                    val gVal = ((gIdx / denom) * 255f).roundToInt().coerceIn(0, 255)
                    val rowOff = (tileY + gIdx) * bmpW
                    for (rIdx in 0 until n) {
                        val rVal = ((rIdx / denom) * 255f).roundToInt().coerceIn(0, 255)
                        pixels[rowOff + tileX + rIdx] = (0xFF shl 24) or (rVal shl 16) or (gVal shl 8) or bVal
                    }
                }
            }
            bmp.setPixels(pixels, 0, bmpW, 0, 0, bmpW, bmpH)
            return bmp
        }

        /**
         * Generates a Snap-style cinematic beauty 3D LUT Bitmap (Hald square or horizontal strip) with
         * non-linear skin-tone-safe color grading: creamy warm highlights, subtle teal/cool shadows,
         * and rosy-golden midtone skin protection.
         */
        @JvmOverloads
        fun createCinematicBeautyLutBitmap(gridSize: Int = 16, asSquareHald: Boolean = false): Bitmap {
            val n = gridSize.coerceIn(4, 64)
            val tilesPerRow = if (asSquareHald) sqrt(n.toDouble()).roundToInt().coerceAtLeast(1) else n
            val tilesPerCol = ((n + tilesPerRow - 1) / tilesPerRow).coerceAtLeast(1)
            val bmpW = tilesPerRow * n
            val bmpH = tilesPerCol * n
            val bmp = Bitmap.createBitmap(bmpW, bmpH, Bitmap.Config.ARGB_8888)
            val pixels = IntArray(bmpW * bmpH)
            val denom = (n - 1).coerceAtLeast(1).toFloat()
            val outRgb = FloatArray(3)
            for (bIdx in 0 until n) {
                val tileX = (bIdx % tilesPerRow) * n
                val tileY = (bIdx / tilesPerRow) * n
                val bf = bIdx / denom
                for (gIdx in 0 until n) {
                    val gf = gIdx / denom
                    val rowOff = (tileY + gIdx) * bmpW
                    for (rIdx in 0 until n) {
                        val rf = rIdx / denom
                        evaluateCinematicBeautyGrade(rf, gf, bf, outRgb)
                        val rVal = (outRgb[0] * 255f).roundToInt().coerceIn(0, 255)
                        val gVal = (outRgb[1] * 255f).roundToInt().coerceIn(0, 255)
                        val bVal = (outRgb[2] * 255f).roundToInt().coerceIn(0, 255)
                        pixels[rowOff + tileX + rIdx] = (0xFF shl 24) or (rVal shl 16) or (gVal shl 8) or bVal
                    }
                }
            }
            bmp.setPixels(pixels, 0, bmpW, 0, 0, bmpW, bmpH)
            return bmp
        }

        /**
         * Evaluates Snap-style non-linear skin-tone-safe color grading (creamy highlights, teal shadows,
         * rosy midtones) used both to generate procedural 3D LUT bitmaps and as the analytical `post.lut@1`
         * curve when no external LUT bitmap is bound.
         */
        fun evaluateCinematicBeautyGrade(r: Float, g: Float, b: Float, outRgb: FloatArray = FloatArray(3)): FloatArray {
            val rc = r.coerceIn(0f, 1f)
            val gc = g.coerceIn(0f, 1f)
            val bc = b.coerceIn(0f, 1f)

            // Soft filmic S-curve with gentle shadow toe and creamy highlight rolloff
            val sr = rc * rc * (3f - 2f * rc)
            val sg = gc * gc * (3f - 2f * gc)
            val sb = bc * bc * (3f - 2f * bc)
            val baseR = rc * 0.65f + sr * 0.35f
            val baseG = gc * 0.65f + sg * 0.35f
            val baseB = bc * 0.65f + sb * 0.35f

            val luma = (0.299f * baseR + 0.587f * baseG + 0.114f * baseB).coerceIn(0f, 1f)
            val shadowW = (1f - luma) * (1f - luma)
            val highlightW = luma * luma
            val midtoneW = (4f * luma * (1f - luma)).coerceIn(0f, 1f)

            // Skin-tone affinity detector (warm R > G > B ratio in midtones)
            val warmExcess = (baseR - baseB).coerceIn(0f, 1f) * (1f - abs(baseG - (baseR * 0.62f + baseB * 0.38f))).coerceIn(0f, 1f)
            val skinW = (warmExcess * 1.8f).coerceIn(0f, 1f) * midtoneW

            // 1. Teal shadows (-R, +G, +B in low luminance, attenuated on skin tones)
            val shadowProtect = 1f - skinW * 0.65f
            val tealR = -0.028f * shadowW * shadowProtect
            val tealG = +0.014f * shadowW * shadowProtect
            val tealB = +0.036f * shadowW * shadowProtect

            // 2. Creamy warm highlights (+R, +G, slight -B rolloff in high luminance)
            val creamR = +0.034f * highlightW
            val creamG = +0.022f * highlightW
            val creamB = -0.018f * highlightW

            // 3. Rosy-golden midtones (subtle rosy warmth on midtones and skin)
            val rosyR = +0.026f * midtoneW + 0.018f * skinW
            val rosyG = +0.006f * midtoneW + 0.004f * skinW
            val rosyB = +0.010f * midtoneW + 0.008f * skinW

            outRgb[0] = (baseR + tealR + creamR + rosyR).coerceIn(0f, 1f)
            outRgb[1] = (baseG + tealG + creamG + rosyG).coerceIn(0f, 1f)
            outRgb[2] = (baseB + tealB + creamB + rosyB).coerceIn(0f, 1f)
            return outRgb
        }

        /**
         * Resolves 3D LUT layout dimensions `[gridSize, tilesPerRow, texWidth, texHeight]` from a 2D LUT texture's
         * pixel dimensions `(texWidth, texHeight)` and an optional manifest `gridSizeHint`.
         * Supports both square Hald LUTs (e.g. `64x64` for $16^3$, `512x512` for $64^3$) and horizontal/grid strip LUTs
         * (e.g. `256x16` for $16^3$, `1024x32` for $32^3$, `4096x64` for $64^3$).
         */
        @JvmOverloads
        fun resolveLutLayout(
            texWidth: Int,
            texHeight: Int,
            gridSizeHint: Int = 0
        ): IntArray {
            val bw = texWidth.coerceAtLeast(4)
            val bh = texHeight.coerceAtLeast(4)
            val inferredGrid: Int = if (bw == bh) {
                Math.round(Math.pow(bw.toDouble(), 2.0 / 3.0)).toInt().coerceIn(4, 64)
            } else {
                minOf(bw, bh).coerceIn(4, 64)
            }
            val gridSize = if (gridSizeHint in 4..64 &&
                bw % gridSizeHint == 0 &&
                bh % gridSizeHint == 0 &&
                (bw / gridSizeHint) * (bh / gridSizeHint) >= gridSizeHint
            ) {
                gridSizeHint
            } else {
                inferredGrid
            }
            val tilesPerRow = (bw / gridSize).coerceAtLeast(1)
            return intArrayOf(gridSize, tilesPerRow, bw, bh)
        }

        /**
         * Computes the effective `post.grain@1` seed combining the node's `paramSeed` with [DeterministicFrameContext]
         * (`frameSeed` / `renderTimeSec`) so live preview grain animates organically while still photo capture and
         * deterministic tests remain 100% reproducible.
         */
        @JvmOverloads
        fun resolveEffectiveGrainSeed(
            paramSeed: Float,
            frameContext: DeterministicFrameContext = DeterministicFrameContext.DEFAULT,
            speedHz: Float = 24.0f
        ): Float {
            if (frameContext === DeterministicFrameContext.DEFAULT) {
                return paramSeed
            }
            val effSpeed = speedHz.coerceIn(0.0f, 120.0f)
            val seedMod = ((frameContext.frameSeed % 9973) + 9973) % 9973
            val timeStep = (frameContext.renderTimeSec * effSpeed).roundToInt() % 4096
            return paramSeed + seedMod * 0.173f + timeStep * 0.317f
        }

        /**
         * Evaluates a 3D color LUT (`post.lut@1`) with full trilinear interpolation for a single normalized RGB triplet.
         * Supports both square Hald LUTs (e.g. `512x512` for $64^3$, `64x64` for $16^3$) and horizontal strip LUTs
         * (`N^2 x N`, e.g. `256x16` or `1024x32`). When `lutBitmap == null` and `lutPixels == null`, applies a
         * deterministic filmic S-curve LUT.
         */
        @JvmOverloads
        fun evaluateLutRgb(
            r: Float,
            g: Float,
            b: Float,
            lutBitmap: Bitmap? = null,
            intensity: Float = 1.0f,
            outRgb: FloatArray = FloatArray(3),
            lutPixels: IntArray? = null,
            lutWidth: Int = lutBitmap?.width ?: 0,
            lutHeight: Int = lutBitmap?.height ?: 0,
            gridSizeHint: Int = 0
        ): FloatArray {
            val rc = r.coerceIn(0f, 1f)
            val gc = g.coerceIn(0f, 1f)
            val bc = b.coerceIn(0f, 1f)
            val mixW = intensity.coerceIn(0f, 1f)
            if (mixW <= 1e-5f) {
                outRgb[0] = rc
                outRgb[1] = gc
                outRgb[2] = bc
                return outRgb
            }

            var lutR: Float
            var lutG: Float
            var lutB: Float
            val hasValidLut = (lutWidth >= 4 && lutHeight >= 4) &&
                ((lutPixels != null && lutPixels.size >= lutWidth * lutHeight) || lutBitmap != null)
            if (hasValidLut) {
                val layout = resolveLutLayout(lutWidth, lutHeight, gridSizeHint)
                val gridSize = layout[0]
                val tilesPerRow = layout[1]
                val bw = layout[2]
                val bh = layout[3]
                val maxIdx = (gridSize - 1).coerceAtLeast(1)
                val rf = rc * maxIdx
                val gf = gc * maxIdx
                val bf = bc * maxIdx

                val r0 = rf.toInt().coerceIn(0, maxIdx)
                val r1 = (r0 + 1).coerceAtMost(maxIdx)
                val tr = (rf - r0).coerceIn(0f, 1f)

                val g0 = gf.toInt().coerceIn(0, maxIdx)
                val g1 = (g0 + 1).coerceAtMost(maxIdx)
                val tg = (gf - g0).coerceIn(0f, 1f)

                val b0 = bf.toInt().coerceIn(0, maxIdx)
                val b1 = (b0 + 1).coerceAtMost(maxIdx)
                val tb = (bf - b0).coerceIn(0f, 1f)

                fun sampleSlice(bSlice: Int, rx: Int, gy: Int): Int {
                    val tx = (bSlice % tilesPerRow) * gridSize
                    val ty = (bSlice / tilesPerRow) * gridSize
                    val px = (tx + rx).coerceIn(0, bw - 1)
                    val py = (ty + gy).coerceIn(0, bh - 1)
                    return if (lutPixels != null) {
                        lutPixels[py * bw + px]
                    } else {
                        lutBitmap!!.getPixel(px, py)
                    }
                }

                fun bilerpSlice(bSlice: Int, shift: Int): Float {
                    val c00 = ((sampleSlice(bSlice, r0, g0) ushr shift) and 0xFF) / 255f
                    val c10 = ((sampleSlice(bSlice, r1, g0) ushr shift) and 0xFF) / 255f
                    val c01 = ((sampleSlice(bSlice, r0, g1) ushr shift) and 0xFF) / 255f
                    val c11 = ((sampleSlice(bSlice, r1, g1) ushr shift) and 0xFF) / 255f
                    val top = c00 + (c10 - c00) * tr
                    val bot = c01 + (c11 - c01) * tr
                    return top + (bot - top) * tg
                }

                val rSlice0 = bilerpSlice(b0, 16)
                val rSlice1 = bilerpSlice(b1, 16)
                val gSlice0 = bilerpSlice(b0, 8)
                val gSlice1 = bilerpSlice(b1, 8)
                val bSlice0 = bilerpSlice(b0, 0)
                val bSlice1 = bilerpSlice(b1, 0)

                lutR = rSlice0 + (rSlice1 - rSlice0) * tb
                lutG = gSlice0 + (gSlice1 - gSlice0) * tb
                lutB = bSlice0 + (bSlice1 - bSlice0) * tb
            } else {
                // Snap-style non-linear skin-tone-safe grading (creamy highlights, teal shadows, rosy midtones)
                evaluateCinematicBeautyGrade(rc, gc, bc, outRgb)
                lutR = outRgb[0]
                lutG = outRgb[1]
                lutB = outRgb[2]
            }

            outRgb[0] = (rc + (lutR - rc) * mixW).coerceIn(0f, 1f)
            outRgb[1] = (gc + (lutG - gc) * mixW).coerceIn(0f, 1f)
            outRgb[2] = (bc + (lutB - bc) * mixW).coerceIn(0f, 1f)
            return outRgb
        }

        private const val VS_POST = """#version 300 es
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec2 aTexCoord;
            out vec2 vTexCoord;
            void main() {
                vTexCoord = aTexCoord;
                gl_Position = vec4(aPosition, 0.0, 1.0);
            }
        """

        internal const val FS_POST = """#version 300 es
            precision mediump float;
            in vec2 vTexCoord;
            uniform sampler2D uSceneTex;
            uniform sampler2D uSkinMaskTex;
            uniform int uHasSkinMask;
            uniform sampler2D uLutTex;
            uniform int uHasLutTex;
            uniform vec4 uLutParams; // x=gridSize, y=tilesPerRow, z=texWidth, w=texHeight
            uniform vec2 uTexelSize;
            uniform vec2 uRawTexelSize;
            uniform int uEffectType; // 0=colorGrade, 1=blur, 2=skinSoften/bilateral, 3=softFocus, 4=warp, 5=colorMatrix, 6=vignette, 7=lut, 8=grain, 9=sharpen
            uniform float uWarmth;
            uniform float uContrast;
            uniform float uSaturation;
            uniform float uExposure;
            uniform float uRadius;
            uniform float uAmount;
            uniform float uEdgePreserve;
            uniform float uDetailPreserve;
            uniform float uThreshold;
            uniform float uIntensity;
            uniform float uHueDeg;
            uniform float uTint;
            uniform float uSepia;
            uniform float uBw;
            uniform float uVigSoftness;
            uniform float uSeed;
            uniform float uLumaWeight;
            uniform vec2 uFaceCenter;
            uniform vec2 uFaceHalfExtents;
            uniform vec2 uLeftJaw;
            uniform vec2 uRightJaw;
            uniform vec2 uLeftEye;
            uniform vec2 uRightEye;
            uniform vec2 uLeftAla;
            uniform vec2 uRightAla;
            uniform vec2 uNoseTip;
            uniform vec2 uChinTip;
            uniform vec4 uWarpWeights;
            out vec4 fragColor;

            float luma(vec3 c) {
                return dot(c, vec3(0.299, 0.587, 0.114));
            }

            vec3 sampleLut3D(vec3 color) {
                vec3 c = clamp(color, 0.0, 1.0);
                float gridSize = max(4.0, uLutParams.x);
                float tilesPerRow = max(1.0, uLutParams.y);
                vec2 texSize = max(vec2(16.0), uLutParams.zw);
                float maxIdx = gridSize - 1.0;

                float blueColor = c.b * maxIdx;
                float b0 = floor(blueColor);
                float b1 = min(maxIdx, b0 + 1.0);
                float fb = blueColor - b0;

                vec2 quad0 = vec2(mod(b0, tilesPerRow), floor(b0 / tilesPerRow));
                vec2 quad1 = vec2(mod(b1, tilesPerRow), floor(b1 / tilesPerRow));

                // Clamp within [0.5, gridSize - 0.5] inside each slice tile so hardware bilinear filtering
                // never bleeds across adjacent blue slice boundaries.
                vec2 slicePos = clamp(vec2(0.5) + c.rg * maxIdx, vec2(0.5), vec2(gridSize - 0.5));
                vec2 uv0 = (quad0 * gridSize + slicePos) / texSize;
                vec2 uv1 = (quad1 * gridSize + slicePos) / texSize;

                vec3 s0 = texture(uLutTex, uv0).rgb;
                vec3 s1 = texture(uLutTex, uv1).rgb;
                return mix(s0, s1, fb);
            }

            void main() {
                vec4 baseSample = texture(uSceneTex, vTexCoord);
                vec3 orig = baseSample.rgb;
                vec3 res = orig;
                vec2 viewUv = vec2(vTexCoord.x, 1.0 - vTexCoord.y);
                float maskW = (uHasSkinMask == 1) ? texture(uSkinMaskTex, viewUv).r : 1.0;

                if (uEffectType == 0) {
                    vec3 c = orig * exp2(uExposure);
                    c.r += uWarmth * 0.08;
                    c.g += uTint * 0.08;
                    c.b -= uWarmth * 0.08;
                    c = (c - 0.5) * uContrast + 0.5;
                    float y = luma(c);
                    c = mix(vec3(y), c, uSaturation);
                    res = clamp(c, 0.0, 1.0);
                } else if (uEffectType == 1) {
                    vec2 stepUv = uTexelSize * max(uRadius, 0.0);
                    vec3 sum = vec3(0.0);
                    sum += texture(uSceneTex, vTexCoord + vec2(-stepUv.x, -stepUv.y)).rgb * 0.0625;
                    sum += texture(uSceneTex, vTexCoord + vec2( 0.0,      -stepUv.y)).rgb * 0.1250;
                    sum += texture(uSceneTex, vTexCoord + vec2( stepUv.x, -stepUv.y)).rgb * 0.0625;
                    sum += texture(uSceneTex, vTexCoord + vec2(-stepUv.x,  0.0     )).rgb * 0.1250;
                    sum += orig * 0.2500;
                    sum += texture(uSceneTex, vTexCoord + vec2( stepUv.x,  0.0     )).rgb * 0.1250;
                    sum += texture(uSceneTex, vTexCoord + vec2(-stepUv.x,  stepUv.y)).rgb * 0.0625;
                    sum += texture(uSceneTex, vTexCoord + vec2( 0.0,       stepUv.y)).rgb * 0.1250;
                    sum += texture(uSceneTex, vTexCoord + vec2( stepUv.x,  stepUv.y)).rgb * 0.0625;
                    res = mix(orig, sum, clamp(uAmount, 0.0, 1.0));
                } else if (uEffectType == 2) {
                    if (maskW > 0.001) {
                        vec2 stepUv = uTexelSize * max(uRadius, 1.0);
                        float centerY = luma(orig);
                        float sigmaRange = max(0.05, (1.0 - uEdgePreserve * 0.78) * 0.38);
                        vec3 accum = orig;
                        float wSum = 1.0;
                        vec3 tightSum = orig;
                        float tightWSum = 1.0;

                        vec2 offsets[12] = vec2[](
                            vec2(-stepUv.x, 0.0), vec2(stepUv.x, 0.0),
                            vec2(0.0, -stepUv.y), vec2(0.0, stepUv.y),
                            vec2(-stepUv.x, -stepUv.y), vec2(stepUv.x, -stepUv.y),
                            vec2(-stepUv.x, stepUv.y), vec2(stepUv.x, stepUv.y),
                            vec2(-stepUv.x * 2.0, 0.0), vec2(stepUv.x * 2.0, 0.0),
                            vec2(0.0, -stepUv.y * 2.0), vec2(0.0, stepUv.y * 2.0)
                        );
                        for (int i = 0; i < 12; i++) {
                            vec3 s = texture(uSceneTex, vTexCoord + offsets[i]).rgb;
                            float dy = abs(luma(s) - centerY);
                            float w = exp(-(dy * dy) / (2.0 * sigmaRange * sigmaRange));
                            accum += s * w;
                            wSum += w;
                            if (i < 8) {
                                vec3 st = texture(uSceneTex, vTexCoord + offsets[i] * 0.5).rgb;
                                tightSum += st;
                                tightWSum += 1.0;
                            }
                        }
                        vec3 smoothRgb = accum / wSum;
                        vec3 tightAvg = tightSum / tightWSum;
                        vec3 highFreqPores = orig - tightAvg;
                        vec3 detailedSmooth = clamp(smoothRgb + highFreqPores * (uDetailPreserve * 0.22), 0.0, 1.0);
                        float effAmount = clamp(uAmount * 1.45, 0.0, 1.0);
                        res = mix(orig, detailedSmooth, clamp(effAmount, 0.0, 1.0));
                    }
                } else if (uEffectType == 3) {
                    vec2 stepUv = uTexelSize * max(uRadius, 1.0);
                    vec3 bloom = vec3(0.0);
                    float wSum = 0.0;
                    vec2 kOff[8] = vec2[](
                        vec2(-stepUv.x, -stepUv.y), vec2(stepUv.x, -stepUv.y),
                        vec2(-stepUv.x,  stepUv.y), vec2(stepUv.x,  stepUv.y),
                        vec2(-stepUv.x * 2.0, 0.0), vec2(stepUv.x * 2.0, 0.0),
                        vec2(0.0, -stepUv.y * 2.0), vec2(0.0, stepUv.y * 2.0)
                    );
                    for (int i = 0; i < 8; i++) {
                        vec3 s = texture(uSceneTex, vTexCoord + kOff[i]).rgb;
                        float sy = luma(s);
                        float hw = smoothstep(uThreshold * 0.65, 1.0, sy);
                        bloom += s * (0.25 + 0.75 * hw);
                        wSum += 1.0;
                    }
                    bloom /= max(wSum, 1.0);
                    bloom.r += uWarmth * 0.06;
                    bloom.b -= uWarmth * 0.06;
                    bloom = clamp(bloom, 0.0, 1.0);
                    vec3 screened = 1.0 - (1.0 - orig) * (1.0 - bloom * 0.65);
                    float darkGuard = smoothstep(0.08, 0.35, luma(orig));
                    res = mix(orig, screened, clamp(uAmount, 0.0, 1.0) * (0.35 + 0.65 * darkGuard));
                } else if (uEffectType == 4) {
                    vec2 normOval = (viewUv - uFaceCenter) / max(uFaceHalfExtents * 1.05, vec2(0.05));
                    float ovalDistSq = dot(normOval, normOval);
                    if (ovalDistSq < 1.0) {
                        float env = 1.0 - smoothstep(0.72, 1.0, sqrt(ovalDistSq));
                        vec2 dUv = vec2(0.0);
                        float fw = uFaceHalfExtents.x * 2.0;

                        // 1. Jaw Slim (uWarpWeights.x)
                        if (abs(uWarpWeights.x) > 0.001) {
                            float jRad = max(0.03, fw * uRadius * 0.85);
                            float dL = distance(viewUv, uLeftJaw);
                            if (dL < jRad) {
                                float w = 1.0 - smoothstep(0.0, jRad, dL);
                                dUv.x += sign(uFaceCenter.x - uLeftJaw.x) * fw * 0.06 * uWarpWeights.x * w;
                                dUv.y -= fw * 0.06 * 0.18 * uWarpWeights.x * w;
                            }
                            float dR = distance(viewUv, uRightJaw);
                            if (dR < jRad) {
                                float w = 1.0 - smoothstep(0.0, jRad, dR);
                                dUv.x += sign(uFaceCenter.x - uRightJaw.x) * fw * 0.06 * uWarpWeights.x * w;
                                dUv.y -= fw * 0.06 * 0.18 * uWarpWeights.x * w;
                            }
                        }

                        // 2. Eye Enlarge (uWarpWeights.y)
                        if (abs(uWarpWeights.y) > 0.001) {
                            float eyeRad = max(0.025, fw * 0.22);
                            vec2 toLEye = viewUv - uLeftEye;
                            float dLEye = length(toLEye);
                            if (dLEye < eyeRad && dLEye > 0.0001) {
                                float t = dLEye / eyeRad;
                                float bulge = (1.0 - t * t) * uWarpWeights.y * 0.24;
                                dUv += toLEye * bulge;
                            }
                            vec2 toREye = viewUv - uRightEye;
                            float dREye = length(toREye);
                            if (dREye < eyeRad && dREye > 0.0001) {
                                float t = dREye / eyeRad;
                                float bulge = (1.0 - t * t) * uWarpWeights.y * 0.24;
                                dUv += toREye * bulge;
                            }
                        }

                        // 3. Nose Slim (uWarpWeights.z)
                        if (abs(uWarpWeights.z) > 0.001) {
                            float noseRad = max(0.02, fw * 0.20);
                            float dLAla = distance(viewUv, uLeftAla);
                            if (dLAla < noseRad) {
                                float w = 1.0 - smoothstep(0.0, noseRad, dLAla);
                                dUv.x += sign(uNoseTip.x - uLeftAla.x) * fw * 0.038 * uWarpWeights.z * w;
                            }
                            float dRAla = distance(viewUv, uRightAla);
                            if (dRAla < noseRad) {
                                float w = 1.0 - smoothstep(0.0, noseRad, dRAla);
                                dUv.x += sign(uNoseTip.x - uRightAla.x) * fw * 0.038 * uWarpWeights.z * w;
                            }
                        }

                        // 4. Chin V-Line (uWarpWeights.w)
                        if (abs(uWarpWeights.w) > 0.001) {
                            float chinRad = max(0.03, fw * 0.34);
                            float dChin = distance(viewUv, uChinTip);
                            if (dChin < chinRad) {
                                float w = 1.0 - smoothstep(0.0, chinRad, dChin);
                                dUv.y -= fw * 0.045 * uWarpWeights.w * w;
                                dUv.x += (uChinTip.x - viewUv.x) * 0.16 * uWarpWeights.w * w;
                            }
                        }

                        vec2 sampleUv = clamp(vTexCoord - vec2(dUv.x, -dUv.y) * env, vec2(0.0), vec2(1.0));
                        res = texture(uSceneTex, sampleUv).rgb;
                    }
                } else if (uEffectType == 5) {
                    vec3 c = orig + vec3(uExposure);
                    c = (c - 0.5) * uContrast + 0.5;
                    if (abs(uHueDeg) > 0.01) {
                        float rad = uHueDeg * 0.0174532925;
                        float cosH = cos(rad);
                        float sinH = sin(rad);
                        float yiqY = dot(c, vec3(0.299, 0.587, 0.114));
                        float yiqI = dot(c, vec3(0.596, -0.274, -0.322));
                        float yiqQ = dot(c, vec3(0.211, -0.523, 0.312));
                        float iRot = yiqI * cosH - yiqQ * sinH;
                        float qRot = yiqI * sinH + yiqQ * cosH;
                        c = vec3(
                            yiqY + 0.956 * iRot + 0.621 * qRot,
                            yiqY - 0.272 * iRot - 0.647 * qRot,
                            yiqY - 1.106 * iRot + 1.703 * qRot
                        );
                    }
                    c.r += uWarmth * 0.12;
                    c.b -= uWarmth * 0.12;
                    c.g += uTint * 0.10;
                    float y = luma(c);
                    c = mix(vec3(y), c, uSaturation);
                    vec3 sepiaRgb = vec3(y * 1.12, y * 0.95, y * 0.74);
                    c = mix(c, sepiaRgb, clamp(uSepia, 0.0, 1.0));
                    float bwLuma = luma(c);
                    c = mix(c, vec3(bwLuma), clamp(uBw, 0.0, 1.0));
                    res = clamp(c, 0.0, 1.0);
                } else if (uEffectType == 6) {
                    vec2 centered = (vTexCoord - vec2(0.5)) * 2.0;
                    float dist = length(centered);
                    float vig = 1.0 - smoothstep(uRadius, uRadius + max(uVigSoftness, 0.01), dist) * clamp(uAmount, 0.0, 1.0);
                    res = clamp(orig * vig, 0.0, 1.0);
                } else if (uEffectType == 7) {
                    vec3 lutColor;
                    if (uHasLutTex == 1) {
                        lutColor = sampleLut3D(orig);
                    } else {
                        vec3 sc = orig * orig * (3.0 - 2.0 * orig);
                        vec3 baseC = mix(orig, sc, 0.35);
                        float y = clamp(luma(baseC), 0.0, 1.0);
                        float shadowW = (1.0 - y) * (1.0 - y);
                        float highlightW = y * y;
                        float midtoneW = clamp(4.0 * y * (1.0 - y), 0.0, 1.0);
                        float warmExcess = clamp(baseC.r - baseC.b, 0.0, 1.0) * clamp(1.0 - abs(baseC.g - (baseC.r * 0.62 + baseC.b * 0.38)), 0.0, 1.0);
                        float skinW = clamp(warmExcess * 1.8, 0.0, 1.0) * midtoneW;
                        float shadowProtect = 1.0 - skinW * 0.65;
                        vec3 tealShadows = vec3(-0.028, 0.014, 0.036) * (shadowW * shadowProtect);
                        vec3 creamyHighlights = vec3(0.034, 0.022, -0.018) * highlightW;
                        vec3 rosyMidtones = vec3(0.026, 0.006, 0.010) * midtoneW + vec3(0.018, 0.004, 0.008) * skinW;
                        lutColor = clamp(baseC + tealShadows + creamyHighlights + rosyMidtones, 0.0, 1.0);
                    }
                    res = mix(orig, lutColor, clamp(uAmount, 0.0, 1.0));
                } else if (uEffectType == 8) {
                    vec2 rawStep = max(uRawTexelSize, vec2(1.0 / 4096.0));
                    vec2 grainCell = floor(viewUv / max(rawStep * max(uRadius, 0.25), vec2(0.0001)));
                    vec2 seedVec = grainCell + vec2(uSeed * 0.137, uSeed * 0.293);
                    float n = fract(sin(dot(seedVec, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
                    float y = luma(orig);
                    float lw = clamp(uLumaWeight, 0.0, 1.0);
                    float midtoneWeight = (1.0 - lw) + lw * (4.0 * y * (1.0 - y));
                    float delta = n * clamp(uAmount, 0.0, 1.0) * 0.32 * midtoneWeight;
                    res = clamp(orig + vec3(delta), 0.0, 1.0);
                } else if (uEffectType == 9) {
                    // Use true 1-pixel unscaled texel step so micro-contrast sharpening matches CPU stepPx without coarse halos
                    vec2 rawStep = max(uRawTexelSize, vec2(1.0 / 4096.0));
                    vec2 stepUv = rawStep * max(uRadius, 0.5);
                    vec3 nL = texture(uSceneTex, vTexCoord + vec2(-stepUv.x, 0.0)).rgb;
                    vec3 nR = texture(uSceneTex, vTexCoord + vec2( stepUv.x, 0.0)).rgb;
                    vec3 nU = texture(uSceneTex, vTexCoord + vec2(0.0, -stepUv.y)).rgb;
                    vec3 nD = texture(uSceneTex, vTexCoord + vec2(0.0,  stepUv.y)).rgb;
                    vec3 blurCross = 0.25 * (nL + nR + nU + nD);
                    vec3 highPass = orig - blurCross;
                    float hpMag = length(highPass);
                    float coring = smoothstep(uThreshold, uThreshold + 0.04, hpMag);
                    res = clamp(orig + highPass * (clamp(uAmount, 0.0, 1.0) * 1.6 * coring), 0.0, 1.0);
                }

                float effMix = clamp(uIntensity * maskW, 0.0, 1.0);
                fragColor = vec4(mix(orig, res, effMix), baseSample.a);
            }
        """
    }

    private var program = 0
    private var quadVao = 0
    private var quadVbo = 0
    private var pingPongFbo = 0
    private var pingPongTex = 0
    private var fboWidth = 0
    private var fboHeight = 0

    private val executedPassesScratch = ArrayList<ExecutedPostPass>(8)
    val lastExecutedPasses: List<ExecutedPostPass>
        get() = executedPassesScratch

    var lastExecutedPostCount: Int = 0
        private set

    var lastUsedHalfResTarget: Boolean = false
        private set

    var lastUsedQuarterResTarget: Boolean = false
        private set

    var lastBoundLutTexId: Int = 0
        private set

    var lastEffectiveGrainSeed: Float = 0f
        private set

    @JvmOverloads
    fun init(resourceOwner: GlResourceOwner? = null) {
        program = 0
        quadVao = 0
        quadVbo = 0
        pingPongFbo = 0
        pingPongTex = 0
        fboWidth = 0
        fboHeight = 0
        if (!hasActiveGlContext()) return
        try {
            program = createProgram(VS_POST, FS_POST)
            if (program != 0) resourceOwner?.trackProgram(program)

            val quadVertices = floatArrayOf(
                // x, y,    u, v
                -1f, -1f,  0f, 0f,
                 1f, -1f,  1f, 0f,
                 1f,  1f,  1f, 1f,
                -1f, -1f,  0f, 0f,
                 1f,  1f,  1f, 1f,
                -1f,  1f,  0f, 1f
            )

            val vaos = IntArray(1)
            val vbos = IntArray(1)
            GLES30.glGenVertexArrays(1, vaos, 0)
            GLES30.glGenBuffers(1, vbos, 0)
            quadVao = vaos[0]
            quadVbo = vbos[0]
            if (quadVao != 0) resourceOwner?.trackVao(quadVao)
            if (quadVbo != 0) resourceOwner?.trackBuffer(quadVbo)

            GLES30.glBindVertexArray(quadVao)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVbo)
            val fb = ByteBuffer.allocateDirect(quadVertices.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            fb.put(quadVertices).position(0)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, quadVertices.size * 4, fb, GLES30.GL_STATIC_DRAW)

            val stride = 4 * 4
            GLES30.glEnableVertexAttribArray(0)
            GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, stride, 0)
            GLES30.glEnableVertexAttribArray(1)
            GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, stride, 2 * 4)
            GLES30.glBindVertexArray(0)
        } catch (_: Throwable) {
        }
    }

    @JvmOverloads
    fun resize(width: Int, height: Int, resourceOwner: GlResourceOwner? = null) {
        val w = width.coerceAtLeast(1)
        val h = height.coerceAtLeast(1)
        if (pingPongFbo != 0 && fboWidth == w && fboHeight == h) return

        releasePingPongFbo(resourceOwner)
        fboWidth = w
        fboHeight = h
        if (!hasActiveGlContext()) return

        try {
            val prevDrawFbo = IntArray(1)
            val prevReadFbo = IntArray(1)
            val prevTex2D = IntArray(1)
            GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevDrawFbo, 0)
            GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, prevReadFbo, 0)
            GLES30.glGetIntegerv(GLES30.GL_TEXTURE_BINDING_2D, prevTex2D, 0)

            val fbos = IntArray(1)
            val texs = IntArray(1)
            GLES30.glGenFramebuffers(1, fbos, 0)
            GLES30.glGenTextures(1, texs, 0)
            pingPongFbo = fbos[0]
            pingPongTex = texs[0]
            if (pingPongFbo != 0) resourceOwner?.trackFramebuffer(pingPongFbo)
            if (pingPongTex != 0) resourceOwner?.trackTexture(pingPongTex)

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, pingPongTex)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8,
                w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, pingPongFbo)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, pingPongTex, 0
            )
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, prevTex2D[0])
            GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, prevDrawFbo[0])
            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevReadFbo[0])
        } catch (_: Throwable) {
        }
    }

    /**
     * Executes active `post` nodes in deterministic `renderOrder` (`Pass 7` in `RenderGraph`).
     * Works both on live EGL contexts and in headless JVM tests (recording [lastExecutedPasses]
     * and leasing half-res / quarter-res [RtPool] targets).
     */
    @JvmOverloads
    fun renderPass(
        effect: CompiledEffect,
        postIndices: IntArray,
        postCount: Int,
        viewportWidth: Int,
        viewportHeight: Int,
        resourceOwner: GlResourceOwner? = null,
        faceState: FaceState = FaceState.EMPTY,
        regionMask: RegionMaskPass? = null,
        rtPool: RtPool? = null,
        textures: TextureCache? = null,
        frameContext: DeterministicFrameContext = DeterministicFrameContext.DEFAULT
    ): Int {
        executedPassesScratch.clear()
        lastExecutedPostCount = 0
        lastUsedHalfResTarget = false
        lastUsedQuarterResTarget = false
        lastBoundLutTexId = 0
        if (postCount <= 0 || viewportWidth <= 0 || viewportHeight <= 0) return 0

        val glActive = hasActiveGlContext() && program != 0 && quadVao != 0
        val prevFbo = IntArray(1)
        val attachedTex = IntArray(1)
        if (glActive) {
            GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevFbo, 0)
            if (prevFbo[0] != 0) {
                try {
                    GLES30.glGetFramebufferAttachmentParameteriv(
                        GLES30.GL_DRAW_FRAMEBUFFER,
                        GLES30.GL_COLOR_ATTACHMENT0,
                        GLES30.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
                        attachedTex,
                        0
                    )
                } catch (_: Throwable) {
                }
            }
            resize(viewportWidth, viewportHeight, resourceOwner)
            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            GLES30.glDepthMask(false)
            GLES30.glDisable(GLES30.GL_BLEND)
            GLES30.glDisable(GLES30.GL_CULL_FACE)
            GLES30.glColorMask(true, true, true, true)
            GLES30.glUseProgram(program)
            GLES30.glBindVertexArray(quadVao)
        }
        val targetFbo = prevFbo[0]
        val targetColorTex = attachedTex[0]
        val canPingPongDirect = glActive && targetFbo != 0 && targetColorTex != 0 && pingPongFbo != 0 && pingPongTex != 0
        var pingPongParity = 0

        var executed = 0
        for (k in 0 until postCount) {
            val nodeIdx = postIndices[k]
            if (nodeIdx !in effect.nodes.indices) continue
            val node = effect.nodes[nodeIdx]
            val nodeOpacity = (effect.propertySlots.opacity[nodeIdx] * effect.propertySlots.visible[nodeIdx]).coerceIn(0f, 1f)
            if (nodeOpacity <= 0.001f) continue

            val rawEffect = node.postEffect ?: continue
            val effectName = normalizeEffectName(rawEffect)
            val params = resolveDynamicPostParams(node, effect)
            val baseIntensity = (params["intensity"] ?: 1.0f).coerceIn(0f, 1f) * nodeOpacity
            if (baseIntensity <= 0.001f) continue

            // Lease multi-scale RtPool targets and optional region mask target ($mask:skin or node.faceMeshRegion)
            var skinMaskTarget: RtPool.PooledRenderTarget? = null
            if (effectName == "skinSoften" || effectName == "bilateral") {
                if (rtPool != null) {
                    val halfRt = rtPool.acquireTemp(
                        baseWidth = viewportWidth,
                        baseHeight = viewportHeight,
                        format = RtPool.RtFormat.RGBA8,
                        scaleFactor = RtPool.RtScale.HALF.factor,
                        debugLabel = "skinSoftenHalf",
                        resourceOwner = resourceOwner
                    )
                    lastUsedHalfResTarget = halfRt.width == rtPool.computeScaledDimension(viewportWidth, RtPool.RtScale.HALF.factor)
                }
            } else if (effectName == "softFocus") {
                if (rtPool != null) {
                    val quarterRt = rtPool.acquireTemp(
                        baseWidth = viewportWidth,
                        baseHeight = viewportHeight,
                        format = RtPool.RtFormat.RGBA8,
                        scaleFactor = RtPool.RtScale.QUARTER.factor,
                        debugLabel = "softFocusQuarter",
                        resourceOwner = resourceOwner
                    )
                    lastUsedQuarterResTarget = quarterRt.width == rtPool.computeScaledDimension(viewportWidth, RtPool.RtScale.QUARTER.factor)
                }
            }

            val wantsSkinMask = node.faceMeshRegion != null || effectName == "bilateral" || effectName == "skinSoften"
            if (wantsSkinMask && regionMask != null && rtPool != null && faceState.isVisible) {
                val maskIdStr = (node.faceMeshRegion ?: "skin").removePrefix("\$mask:").trim().lowercase().replace('-', '_')
                val maskExpr = effect.maskDefs[maskIdStr]
                    ?: RegionMaskPass.BUILTIN_MASKS[maskIdStr]
                    ?: RegionMaskPass.BUILTIN_SKIN_MASK
                skinMaskTarget = regionMask.evaluateMaskToRtPool(
                    maskId = maskIdStr,
                    expr = maskExpr,
                    faceState = faceState,
                    viewportWidth = viewportWidth,
                    viewportHeight = viewportHeight,
                    rtPool = rtPool,
                    propertySlots = effect.propertySlots,
                    scaleFactor = RtPool.RtScale.HALF.factor,
                    resourceOwner = resourceOwner
                )
            }

            // Resolve 3D LUT texture and layout dimensions when effectName == "lut"
            var lutTexId = 0
            var lutLayout = intArrayOf(64, 8, 512, 512)
            if (effectName == "lut" && textures != null) {
                val lutAsset = node.postAssetId?.let { effect.assets[it] }
                    ?: effect.assets.values.firstOrNull { it.type == "lut" }
                if (lutAsset != null) {
                    lutTexId = textures.getOrLoad(lutAsset.localPath, isLut = true)
                    lastBoundLutTexId = lutTexId
                    val entry = textures.getEntry(lutAsset.localPath, isLut = true)
                        ?: textures.getEntryForPath(lutAsset.localPath)
                    if (entry != null && entry.width >= 4 && entry.height >= 4) {
                        val gridHint = params["gridSize"]?.roundToInt() ?: 0
                        lutLayout = resolveLutLayout(entry.width, entry.height, gridHint)
                    }
                }
            }

            val effectiveParams = if (effectName == "grain") {
                val effSeed = resolveEffectiveGrainSeed(
                    paramSeed = params["seed"] ?: 0f,
                    frameContext = frameContext,
                    speedHz = params["speed"] ?: 24.0f
                )
                lastEffectiveGrainSeed = effSeed
                if (effSeed != (params["seed"] ?: 0f)) {
                    HashMap(params).apply { put("seed", effSeed) }
                } else {
                    params
                }
            } else {
                params
            }

            if (glActive && pingPongFbo != 0 && pingPongTex != 0) {
                GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
                GLES30.glDisable(GLES30.GL_DEPTH_TEST)
                GLES30.glDepthMask(false)
                GLES30.glDisable(GLES30.GL_BLEND)
                GLES30.glColorMask(true, true, true, true)
                GLES30.glUseProgram(program)
                GLES30.glBindVertexArray(quadVao)

                val srcTexId: Int
                val dstFboId: Int
                if (canPingPongDirect) {
                    if (pingPongParity == 0) {
                        srcTexId = targetColorTex
                        dstFboId = pingPongFbo
                    } else {
                        srcTexId = pingPongTex
                        dstFboId = targetFbo
                    }
                    pingPongParity = 1 - pingPongParity
                } else {
                    srcTexId = 0
                    dstFboId = targetFbo
                }

                drawGlPostNode(
                    effectName = effectName,
                    params = effectiveParams,
                    baseIntensity = baseIntensity,
                    viewportWidth = viewportWidth,
                    viewportHeight = viewportHeight,
                    targetFbo = dstFboId,
                    directInputTexId = srcTexId,
                    skinMaskTexId = skinMaskTarget?.texId ?: 0,
                    lutTexId = lutTexId,
                    lutLayout = lutLayout,
                    faceState = faceState,
                    regionMask = regionMask
                )
            }

            executedPassesScratch.add(
                ExecutedPostPass(
                    nodeId = node.id,
                    effect = effectName,
                    renderOrder = node.renderOrder,
                    params = effectiveParams
                )
            )
            executed++
        }

        if (glActive) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            if (canPingPongDirect && pingPongParity == 1) {
                // Final pass landed in pingPongFbo; hardware-blit back into targetFbo
                GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, pingPongFbo)
                GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, targetFbo)
                GLES30.glBlitFramebuffer(
                    0, 0, viewportWidth, viewportHeight,
                    0, 0, viewportWidth, viewportHeight,
                    GLES30.GL_COLOR_BUFFER_BIT,
                    GLES30.GL_NEAREST
                )
            }
            GLES30.glBindVertexArray(0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, targetFbo)
        }
        lastExecutedPostCount = executed
        return executed
    }

    /**
     * Executes a single data-driven custom pass (`PassStageKind.CUSTOM_PASS` in `RenderGraph`, e.g.
     * `pass.lut@1`, `pass.grain@1`, `pass.sharpen@1`, `pass.skinSoften@1`, `pass.softFocus@1`, `pass.warp@1`,
     * `pass.colorMatrix@1`, `pass.vignette@1`), reading from [passNode.parsedReads] (`$rt:<name>`, `$mask:<id>`,
     * or current framebuffer) and writing to [passNode.parsedWrites] (`$rt:<name>` or `$output`).
     */
    @JvmOverloads
    fun renderCustomPass(
        passNode: PassNode,
        resolvedParams: Map<String, Float>,
        effect: CompiledEffect,
        viewportWidth: Int,
        viewportHeight: Int,
        effectiveScale: Float = 1.0f,
        resourceOwner: GlResourceOwner? = null,
        faceState: FaceState = FaceState.EMPTY,
        regionMask: RegionMaskPass? = null,
        rtPool: RtPool? = null,
        textures: TextureCache? = null,
        frameContext: DeterministicFrameContext = DeterministicFrameContext.DEFAULT
    ): Boolean {
        val rawTemplate = passNode.templateId.removePrefix("pass.").removePrefix("post.")
        val effectName = normalizeEffectName(rawTemplate)
        if (!SUPPORTED_EFFECTS.contains(effectName)) return false

        val seededParams = LinkedHashMap<String, Float>(getDefaultParamsForEffect(effectName)).apply {
            putAll(resolvedParams)
        }
        val baseIntensity = (seededParams["intensity"] ?: 1.0f).coerceIn(0f, 1f)
        if (baseIntensity <= 0.001f) return false

        val effectiveParams = if (effectName == "grain") {
            val effSeed = resolveEffectiveGrainSeed(
                paramSeed = seededParams["seed"] ?: 0f,
                frameContext = frameContext,
                speedHz = seededParams["speed"] ?: 24.0f
            )
            lastEffectiveGrainSeed = effSeed
            if (effSeed != (seededParams["seed"] ?: 0f)) {
                HashMap(seededParams).apply { put("seed", effSeed) }
            } else {
                seededParams
            }
        } else {
            seededParams
        }

        // Resolve optional LUT texture if a custom pass references a LUT asset
        var lutTexId = 0
        var lutLayout = intArrayOf(64, 8, 512, 512)
        if (effectName == "lut" && textures != null) {
            val lutAsset = passNode.assetId?.let { effect.assets[it] }
                ?: effect.nodes.firstOrNull {
                    it.id == passNode.id || (it.type == "post" && normalizeEffectName(it.postEffect ?: "") == "lut")
                }?.postAssetId?.let { effect.assets[it] }
                ?: effect.assets.values.firstOrNull { it.type == "lut" }
            if (lutAsset != null) {
                lutTexId = textures.getOrLoad(lutAsset.localPath, isLut = true)
                lastBoundLutTexId = lutTexId
                val entry = textures.getEntry(lutAsset.localPath, isLut = true)
                    ?: textures.getEntryForPath(lutAsset.localPath)
                if (entry != null && entry.width >= 4 && entry.height >= 4) {
                    val gridHint = effectiveParams["gridSize"]?.roundToInt() ?: 0
                    lutLayout = resolveLutLayout(entry.width, entry.height, gridHint)
                }
            }
        }

        // Resolve input scene texture and mask texture from passNode.parsedReads
        var inputSceneTexId = 0
        var skinMaskTexId = 0
        if (rtPool != null) {
            for (readRef in passNode.parsedReads) {
                when (readRef.kind) {
                    RenderTargetRef.TargetKind.NAMED_RT -> {
                        val namedRt = rtPool.getNamedTarget(readRef.raw)
                        if (namedRt != null && inputSceneTexId == 0) {
                            inputSceneTexId = namedRt.texId
                        }
                    }
                    RenderTargetRef.TargetKind.MASK -> {
                        val maskRt = rtPool.getNamedTarget(readRef.raw)
                        if (maskRt != null) {
                            skinMaskTexId = maskRt.texId
                        }
                    }
                    else -> Unit
                }
            }
        }

        val glActive = hasActiveGlContext() && program != 0 && quadVao != 0
        if (glActive) {
            val prevFbo = IntArray(1)
            GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevFbo, 0)
            val writeRef = passNode.parsedWrites
            val writeRt = if (writeRef.kind == RenderTargetRef.TargetKind.NAMED_RT && rtPool != null) {
                rtPool.getNamedTarget(writeRef.raw)
            } else {
                null
            }
            val targetFbo = writeRt?.fboId ?: prevFbo[0]
            val targetW = writeRt?.width ?: (viewportWidth * effectiveScale).roundToInt().coerceAtLeast(1)
            val targetH = writeRt?.height ?: (viewportHeight * effectiveScale).roundToInt().coerceAtLeast(1)

            resize(targetW, targetH, resourceOwner)
            GLES30.glViewport(0, 0, targetW, targetH)
            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            GLES30.glDepthMask(false)
            GLES30.glDisable(GLES30.GL_BLEND)
            GLES30.glDisable(GLES30.GL_CULL_FACE)
            GLES30.glColorMask(true, true, true, true)
            GLES30.glUseProgram(program)
            GLES30.glBindVertexArray(quadVao)

            drawGlPostNode(
                effectName = effectName,
                params = effectiveParams,
                baseIntensity = baseIntensity,
                viewportWidth = targetW,
                viewportHeight = targetH,
                targetFbo = targetFbo,
                directInputTexId = inputSceneTexId,
                skinMaskTexId = skinMaskTexId,
                lutTexId = lutTexId,
                lutLayout = lutLayout,
                faceState = faceState,
                regionMask = regionMask
            )

            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glBindVertexArray(0)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, prevFbo[0])
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
        }

        executedPassesScratch.add(
            ExecutedPostPass(
                nodeId = passNode.id,
                effect = effectName,
                renderOrder = executedPassesScratch.size,
                params = effectiveParams
            )
        )
        lastExecutedPostCount = executedPassesScratch.size
        return true
    }

    private fun drawGlPostNode(
        effectName: String,
        params: Map<String, Float>,
        baseIntensity: Float,
        viewportWidth: Int,
        viewportHeight: Int,
        targetFbo: Int,
        directInputTexId: Int = 0,
        skinMaskTexId: Int,
        lutTexId: Int = 0,
        lutLayout: IntArray = intArrayOf(64, 8, 512, 512),
        faceState: FaceState,
        regionMask: RegionMaskPass? = null
    ) {
        val uSceneTex = GLES30.glGetUniformLocation(program, "uSceneTex")
        val uSkinMaskTex = GLES30.glGetUniformLocation(program, "uSkinMaskTex")
        val uHasSkinMask = GLES30.glGetUniformLocation(program, "uHasSkinMask")
        val uTexelSize = GLES30.glGetUniformLocation(program, "uTexelSize")
        val uRawTexelSize = GLES30.glGetUniformLocation(program, "uRawTexelSize")
        val uEffectType = GLES30.glGetUniformLocation(program, "uEffectType")
        val uWarmth = GLES30.glGetUniformLocation(program, "uWarmth")
        val uContrast = GLES30.glGetUniformLocation(program, "uContrast")
        val uSaturation = GLES30.glGetUniformLocation(program, "uSaturation")
        val uExposure = GLES30.glGetUniformLocation(program, "uExposure")
        val uRadius = GLES30.glGetUniformLocation(program, "uRadius")
        val uAmount = GLES30.glGetUniformLocation(program, "uAmount")
        val uEdgePreserve = GLES30.glGetUniformLocation(program, "uEdgePreserve")
        val uDetailPreserve = GLES30.glGetUniformLocation(program, "uDetailPreserve")
        val uThreshold = GLES30.glGetUniformLocation(program, "uThreshold")
        val uIntensity = GLES30.glGetUniformLocation(program, "uIntensity")
        val uHueDeg = GLES30.glGetUniformLocation(program, "uHueDeg")
        val uTint = GLES30.glGetUniformLocation(program, "uTint")
        val uSepia = GLES30.glGetUniformLocation(program, "uSepia")
        val uBw = GLES30.glGetUniformLocation(program, "uBw")
        val uVigSoftness = GLES30.glGetUniformLocation(program, "uVigSoftness")
        val uFaceCenter = GLES30.glGetUniformLocation(program, "uFaceCenter")
        val uFaceHalfExtents = GLES30.glGetUniformLocation(program, "uFaceHalfExtents")
        val uLeftJaw = GLES30.glGetUniformLocation(program, "uLeftJaw")
        val uRightJaw = GLES30.glGetUniformLocation(program, "uRightJaw")
        val uLeftEye = GLES30.glGetUniformLocation(program, "uLeftEye")
        val uRightEye = GLES30.glGetUniformLocation(program, "uRightEye")
        val uLeftAla = GLES30.glGetUniformLocation(program, "uLeftAla")
        val uRightAla = GLES30.glGetUniformLocation(program, "uRightAla")
        val uNoseTip = GLES30.glGetUniformLocation(program, "uNoseTip")
        val uChinTip = GLES30.glGetUniformLocation(program, "uChinTip")
        val uWarpWeights = GLES30.glGetUniformLocation(program, "uWarpWeights")

        if (directInputTexId != 0) {
            GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, targetFbo)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, directInputTexId)
        } else {
            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, targetFbo)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, pingPongTex)
            GLES30.glCopyTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, 0, 0, viewportWidth, viewportHeight)
            GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, targetFbo)
        }
        val resScale = max(1.0f, kotlin.math.min(viewportWidth, viewportHeight).toFloat() / 360.0f)
        GLES30.glUniform2f(
            uTexelSize,
            resScale / viewportWidth.toFloat(),
            resScale / viewportHeight.toFloat()
        )
        if (uRawTexelSize >= 0) {
            GLES30.glUniform2f(
                uRawTexelSize,
                1.0f / viewportWidth.toFloat(),
                1.0f / viewportHeight.toFloat()
            )
        }
        GLES30.glUniform1i(uSceneTex, 0)

        val uLutTex = GLES30.glGetUniformLocation(program, "uLutTex")
        val uHasLutTex = GLES30.glGetUniformLocation(program, "uHasLutTex")
        val uLutParams = GLES30.glGetUniformLocation(program, "uLutParams")
        val uSeed = GLES30.glGetUniformLocation(program, "uSeed")
        val uLumaWeight = GLES30.glGetUniformLocation(program, "uLumaWeight")

        if (skinMaskTexId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, skinMaskTexId)
            GLES30.glUniform1i(uSkinMaskTex, 1)
            GLES30.glUniform1i(uHasSkinMask, 1)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        } else {
            GLES30.glUniform1i(uHasSkinMask, 0)
        }

        if (lutTexId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, lutTexId)
            if (uLutTex >= 0) GLES30.glUniform1i(uLutTex, 2)
            if (uHasLutTex >= 0) GLES30.glUniform1i(uHasLutTex, 1)
            if (uLutParams >= 0 && lutLayout.size >= 4) {
                GLES30.glUniform4f(
                    uLutParams,
                    lutLayout[0].toFloat(),
                    lutLayout[1].toFloat(),
                    lutLayout[2].toFloat(),
                    lutLayout[3].toFloat()
                )
            }
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        } else {
            if (uHasLutTex >= 0) GLES30.glUniform1i(uHasLutTex, 0)
            if (uLutTex >= 0) GLES30.glUniform1i(uLutTex, 2)
            if (uLutParams >= 0) GLES30.glUniform4f(uLutParams, 64f, 8f, 512f, 512f)
        }

        val effectTypeInt = when (effectName) {
            "colorGrade" -> 0
            "blur" -> 1
            "skinSoften", "bilateral" -> 2
            "softFocus" -> 3
            "warp" -> 4
            "colorMatrix" -> 5
            "vignette" -> 6
            "lut" -> 7
            "grain" -> 8
            "sharpen" -> 9
            else -> 0
        }
        val specs = EFFECT_PARAM_SPECS[effectName]
        val defaultRadius = specs?.get("radius")?.defaultValue
            ?: specs?.get("radiusFW")?.defaultValue
            ?: specs?.get("size")?.defaultValue
            ?: 1.5f
        val defaultAmount = specs?.get("amount")?.defaultValue ?: 0.5f
        val defaultEdgePreserve = specs?.get("edgePreserve")?.defaultValue ?: 0.7f
        val defaultDetailPreserve = specs?.get("detailPreserve")?.defaultValue ?: 0.25f
        val defaultThreshold = specs?.get("threshold")?.defaultValue ?: 0.55f
        val defaultSoftness = specs?.get("softness")?.defaultValue ?: 0.45f
        val defaultLumaWeight = specs?.get("lumaWeight")?.defaultValue ?: 0.75f

        GLES30.glUniform1i(uEffectType, effectTypeInt)
        GLES30.glUniform1f(uWarmth, params["warmth"] ?: params["temperature"] ?: 0.0f)
        GLES30.glUniform1f(uContrast, params["contrast"] ?: 1.0f)
        GLES30.glUniform1f(uSaturation, params["saturation"] ?: 1.0f)
        val expOrBright = (params["exposure"] ?: 0.0f) + (params["brightness"] ?: 0.0f)
        GLES30.glUniform1f(uExposure, expOrBright)
        GLES30.glUniform1f(uRadius, params["radius"] ?: params["radiusFW"] ?: params["size"] ?: defaultRadius)
        GLES30.glUniform1f(uAmount, params["amount"] ?: defaultAmount)
        GLES30.glUniform1f(uEdgePreserve, params["edgePreserve"] ?: defaultEdgePreserve)
        GLES30.glUniform1f(uDetailPreserve, params["detailPreserve"] ?: defaultDetailPreserve)
        GLES30.glUniform1f(uThreshold, params["threshold"] ?: defaultThreshold)
        GLES30.glUniform1f(uIntensity, baseIntensity)
        GLES30.glUniform1f(uHueDeg, params["hue"] ?: 0.0f)
        GLES30.glUniform1f(uTint, params["tint"] ?: 0.0f)
        GLES30.glUniform1f(uSepia, params["sepia"] ?: 0.0f)
        GLES30.glUniform1f(uBw, params["bw"] ?: 0.0f)
        GLES30.glUniform1f(uVigSoftness, params["softness"] ?: defaultSoftness)
        if (uSeed >= 0) {
            GLES30.glUniform1f(uSeed, params["seed"] ?: 0.0f)
        }
        if (uLumaWeight >= 0) {
            GLES30.glUniform1f(uLumaWeight, params["lumaWeight"] ?: defaultLumaWeight)
        }

        if (effectTypeInt == 4 && faceState.isVisible) {
            val tsNs = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
            val smoothedLms = regionMask?.smoother?.smooth(faceState.landmarks, tsNs, faceState.isVisible)
                ?: faceState.landmarks
            val anchors = populateWarpAnchors(faceState, smoothedLms, warpAnchorScratch)

            GLES30.glUniform2f(uFaceCenter, anchors.faceCenterU, anchors.faceCenterV)
            GLES30.glUniform2f(uFaceHalfExtents, anchors.faceHalfW, anchors.faceHalfH)
            GLES30.glUniform2f(uLeftJaw, anchors.leftJaw[0], anchors.leftJaw[1])
            GLES30.glUniform2f(uRightJaw, anchors.rightJaw[0], anchors.rightJaw[1])
            GLES30.glUniform2f(uLeftEye, anchors.leftEyeCenter[0], anchors.leftEyeCenter[1])
            GLES30.glUniform2f(uRightEye, anchors.rightEyeCenter[0], anchors.rightEyeCenter[1])
            GLES30.glUniform2f(uLeftAla, anchors.leftAla[0], anchors.leftAla[1])
            GLES30.glUniform2f(uRightAla, anchors.rightAla[0], anchors.rightAla[1])
            GLES30.glUniform2f(uNoseTip, anchors.noseTip[0], anchors.noseTip[1])
            GLES30.glUniform2f(uChinTip, anchors.chinTip[0], anchors.chinTip[1])
            GLES30.glUniform4f(
                uWarpWeights,
                params["jawSlim"] ?: 0f,
                params["eyeEnlarge"] ?: 0f,
                params["noseSlim"] ?: 0f,
                params["chinVLine"] ?: 0f
            )
        }

        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 6)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }

    @JvmOverloads
    fun warmShaders(resourceOwner: GlResourceOwner? = null) {
        if (!hasActiveGlContext()) return
        if (program == 0) init(resourceOwner)
        if (program != 0) GLES30.glUseProgram(program)
    }

    fun clearExecutedPasses() {
        executedPassesScratch.clear()
        lastExecutedPostCount = 0
    }

    /**
     * Resolves any runtime `PropertySlots` dynamic parameter overrides (`pass:<id>.<param>` or `post:<id>.<param>`)
     * and active tier `paramOverrides` on top of the node's static `postParams`.
     */
    fun resolveDynamicPostParams(node: CompiledNode, effect: CompiledEffect): Map<String, Float> {
        val effectName = normalizeEffectName(node.postEffect ?: "")
        val defaults = getDefaultParamsForEffect(effectName)
        val base = if (defaults.isNotEmpty() && !node.postParams.keys.containsAll(defaults.keys)) {
            LinkedHashMap<String, Float>(defaults).apply { putAll(node.postParams) }
        } else {
            node.postParams
        }
        val slots = effect.propertySlots
        val indexMap = slots.dynamicParamIndexMap
        val tierParams = effect.tierOverrides[effect.activeTier]?.paramOverrides ?: emptyMap()
        var overrideMap: HashMap<String, Float>? = null
        for ((k, v) in base) {
            val passKey = "pass:${node.id}.$k"
            val postKey = "post:${node.id}.$k"
            val bareKey = "${node.id}.$k"
            var dynVal = when {
                indexMap.containsKey(passKey) -> slots.getDynamicParam(passKey, v)
                indexMap.containsKey(postKey) -> slots.getDynamicParam(postKey, v)
                else -> v
            }
            val tierVal = tierParams[postKey] ?: tierParams[passKey] ?: tierParams[bareKey]
            if (tierVal != null) {
                dynVal = tierVal
            }
            if (dynVal != v) {
                if (overrideMap == null) overrideMap = HashMap(base)
                overrideMap[k] = dynVal
            }
        }
        return overrideMap ?: base
    }

    /**
     * Applies all visible `post` nodes in deterministic `renderOrder` to a composed ARGB_8888 `Bitmap`
     * (used by CPU photo fallback and deterministic unit tests so photo output and preview share the
     * exact same grading/softening/softFocus/warp/colorMatrix/vignette/lut/grain/sharpen math).
     */
    @JvmOverloads
    fun applyPostNodesToBitmap(
        composedBitmap: Bitmap,
        effect: CompiledEffect,
        isNodeVisible: BooleanArray,
        faceState: FaceState? = null,
        skinMaskWeights: FloatArray? = null,
        regionMaskPass: RegionMaskPass? = null,
        frameContext: DeterministicFrameContext = DeterministicFrameContext.DEFAULT
    ): Int {
        val width = composedBitmap.width
        val height = composedBitmap.height
        if (width <= 0 || height <= 0) return 0

        val activeTierOverride = effect.tierOverrides[effect.activeTier]
        val postNodes = ArrayList<Pair<Int, CompiledNode>>()
        val includedNodeIds = HashSet<String>()
        for (i in effect.nodes.indices) {
            val node = effect.nodes[i]
            val visibleInEvaluator = (i >= isNodeVisible.size || isNodeVisible[i])
            val visibleInTier = com.example.engine.governor.TierPolicy.isNodeVisibleInTier(
                i, node.baseVisible, activeTierOverride
            )
            if (node.type == "post" && visibleInEvaluator && visibleInTier) {
                val matchingPassAsset = if (node.postAssetId == null && effect.passGraph.isCustomManifestGraph) {
                    effect.passGraph.passes.firstOrNull { it.id == node.id }?.assetId
                } else {
                    null
                }
                val effectiveNode = if (matchingPassAsset != null) {
                    node.copy(postAssetId = matchingPassAsset)
                } else {
                    node
                }
                postNodes.add(i to effectiveNode)
                includedNodeIds.add(node.id)
            }
        }
        if (effect.passGraph.isCustomManifestGraph) {
            var synthOrder = (postNodes.maxOfOrNull { it.second.renderOrder } ?: 0) + 1
            for ((pIdx, passNode) in effect.passGraph.passes.withIndex()) {
                if (passNode.stageKind != PassStageKind.CUSTOM_PASS) continue
                if (passNode.id in includedNodeIds) continue
                val enabledInTier = passNode.enabledByDefault &&
                    com.example.engine.governor.TierPolicy.isPassEnabledInTier(
                        passNode.id,
                        activeTierOverride
                    )
                if (!enabledInTier) continue
                val rawTemplate = passNode.templateId.removePrefix("pass.").removePrefix("post.")
                val effectName = normalizeEffectName(rawTemplate)
                if (!SUPPORTED_EFFECTS.contains(effectName)) continue
                val maskRead = passNode.parsedReads.firstOrNull {
                    it.kind == RenderTargetRef.TargetKind.MASK
                }?.identifier
                val resolvedAssetId = passNode.assetId
                    ?: if (effectName == "lut") effect.assets.values.firstOrNull { it.type == "lut" }?.id else null
                val seededPassParams = LinkedHashMap<String, Float>(getDefaultParamsForEffect(effectName)).apply {
                    putAll(passNode.params)
                }
                val synthNode = CompiledNode(
                    index = effect.nodes.size + pIdx,
                    id = passNode.id,
                    parentIndex = -1,
                    tags = emptyList(),
                    type = "post",
                    anchor = null,
                    basePosition = floatArrayOf(0f, 0f, 0f),
                    baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
                    baseScale = floatArrayOf(1f, 1f, 1f),
                    rotationFollow = "none",
                    materialId = null,
                    baseVisible = true,
                    baseOpacity = 1f,
                    renderOrder = synthOrder++,
                    faceMeshRegion = maskRead,
                    postEffect = effectName,
                    postParams = seededPassParams,
                    postAssetId = resolvedAssetId
                )
                postNodes.add(synthNode.index to synthNode)
                includedNodeIds.add(passNode.id)
            }
        }
        if (postNodes.isEmpty()) return 0

        postNodes.sortBy { it.second.renderOrder }

        val pixels = IntArray(width * height)
        val scratch = IntArray(width * height)
        composedBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val evaluatedMasksCache = HashMap<String, FloatArray>()
        if (skinMaskWeights != null) {
            evaluatedMasksCache["skin"] = skinMaskWeights
        }

        fun resolveNodeMask(node: CompiledNode, defaultSkin: Boolean): FloatArray? {
            if (defaultSkin && skinMaskWeights != null && node.faceMeshRegion == null) {
                return skinMaskWeights
            }
            val hasExplicitMask = node.faceMeshRegion != null
            if (!hasExplicitMask && !defaultSkin) return null
            if (faceState == null || !faceState.isVisible) return null
            val maskId = (node.faceMeshRegion ?: "skin").removePrefix("\$mask:").trim().lowercase().replace('-', '_')
            evaluatedMasksCache[maskId]?.let { return it }
            val rPass = regionMaskPass ?: RegionMaskPass()
            val expr = effect.maskDefs[maskId]
                ?: RegionMaskPass.BUILTIN_MASKS[maskId]
                ?: RegionMaskPass.BUILTIN_SKIN_MASK
            val maxSide = max(width, height).coerceAtLeast(1)
            val evalScale = if (maxSide > 256) 256.0f / maxSide.toFloat() else 1.0f
            val ew = max(1, (width * evalScale).roundToInt())
            val eh = max(1, (height * evalScale).roundToInt())
            val r8Bytes = rPass.evaluateMaskExprToHalfResR8(
                expr = expr,
                faceState = faceState,
                viewportWidth = ew * 2,
                viewportHeight = eh * 2,
                maskWeight = 1.0f,
                useSmoothedLandmarks = false
            )
            val maskArr = FloatArray(width * height)
            for (py in 0 until height) {
                val sy = ((py * eh) / height).coerceIn(0, eh - 1)
                val srcRow = sy * ew
                val dstRow = py * width
                for (px in 0 until width) {
                    val sx = ((px * ew) / width).coerceIn(0, ew - 1)
                    maskArr[dstRow + px] = (r8Bytes[srcRow + sx].toInt() and 0xFF) / 255.0f
                }
            }
            evaluatedMasksCache[maskId] = maskArr
            return maskArr
        }

        fun blendMaskedPassIfNeeded(beforePixels: IntArray, afterPixels: IntArray, maskWeights: FloatArray) {
            for (i in afterPixels.indices) {
                val mw = maskWeights[i].coerceIn(0f, 1f)
                if (mw >= 0.999f) continue
                if (mw <= 0.001f) {
                    afterPixels[i] = beforePixels[i]
                } else {
                    val bPx = beforePixels[i]
                    val aPx = afterPixels[i]
                    val a = (aPx ushr 24) and 0xFF
                    val br = (bPx ushr 16) and 0xFF
                    val bg = (bPx ushr 8) and 0xFF
                    val bb = bPx and 0xFF
                    val ar = (aPx ushr 16) and 0xFF
                    val ag = (aPx ushr 8) and 0xFF
                    val ab = aPx and 0xFF
                    val outR = (br + (ar - br) * mw).roundToInt().coerceIn(0, 255)
                    val outG = (bg + (ag - bg) * mw).roundToInt().coerceIn(0, 255)
                    val outB = (bb + (ab - bb) * mw).roundToInt().coerceIn(0, 255)
                    afterPixels[i] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
        }

        var appliedCount = 0
        for ((idx, node) in postNodes) {
            val nodeOpacity = if (idx in effect.propertySlots.opacity.indices && idx in effect.propertySlots.visible.indices) {
                (effect.propertySlots.opacity[idx] * effect.propertySlots.visible[idx]).coerceIn(0f, 1f)
            } else {
                node.baseOpacity.coerceIn(0f, 1f)
            }
            if (nodeOpacity <= 0.001f) continue

            val effectName = normalizeEffectName(node.postEffect ?: continue)
            val params = resolveDynamicPostParams(node, effect)
            val intensity = ((params["intensity"] ?: 1.0f) * nodeOpacity).coerceIn(0f, 1f)
            if (intensity <= 0.001f) continue

            val nodeMask = if (effectName == "skinSoften" || effectName == "bilateral") {
                resolveNodeMask(node, defaultSkin = true)
            } else {
                resolveNodeMask(node, defaultSkin = false)
            }
            val preMaskBackup = if (nodeMask != null && effectName != "skinSoften" && effectName != "bilateral" && effectName != "blur") {
                pixels.clone()
            } else {
                null
            }

            when (effectName) {
                "colorGrade" -> {
                    val expOrBright = (params["exposure"] ?: 0.0f) + (params["brightness"] ?: 0.0f)
                    applyColorGrade(
                        pixels = pixels,
                        warmth = params["warmth"] ?: 0.0f,
                        tint = params["tint"] ?: 0.0f,
                        contrast = params["contrast"] ?: 1.0f,
                        saturation = params["saturation"] ?: 1.0f,
                        exposure = expOrBright,
                        intensity = intensity
                    )
                    appliedCount++
                }
                "blur" -> {
                    applyBlurOrBilateral(
                        pixels = pixels,
                        scratch = scratch,
                        width = width,
                        height = height,
                        radius = params["radius"] ?: 1.5f,
                        amount = params["amount"] ?: 0.5f,
                        edgePreserve = 0.0f,
                        detailPreserve = 0.0f,
                        bilateral = false,
                        intensity = intensity,
                        skinMask = nodeMask
                    )
                    appliedCount++
                }
                "skinSoften", "bilateral" -> {
                    applyBlurOrBilateral(
                        pixels = pixels,
                        scratch = scratch,
                        width = width,
                        height = height,
                        radius = params["radius"] ?: 2.0f,
                        amount = params["amount"] ?: 0.35f,
                        edgePreserve = params["edgePreserve"] ?: 0.7f,
                        detailPreserve = params["detailPreserve"] ?: 0.25f,
                        bilateral = true,
                        intensity = intensity,
                        skinMask = nodeMask
                    )
                    appliedCount++
                }
                "softFocus" -> {
                    applySoftFocus(
                        pixels = pixels,
                        scratch = scratch,
                        width = width,
                        height = height,
                        amount = params["amount"] ?: 0.35f,
                        radius = params["radius"] ?: 4.0f,
                        threshold = params["threshold"] ?: 0.55f,
                        warmth = params["warmth"] ?: 0.0f,
                        intensity = intensity
                    )
                    appliedCount++
                }
                "warp" -> {
                    if (faceState != null && faceState.isVisible) {
                        applyWarp(
                            pixels = pixels,
                            scratch = scratch,
                            width = width,
                            height = height,
                            faceState = faceState,
                            params = params,
                            intensity = intensity,
                            regionMaskPass = regionMaskPass
                        )
                        appliedCount++
                    }
                }
                "colorMatrix" -> {
                    applyColorMatrix(
                        pixels = pixels,
                        brightness = params["brightness"] ?: 0.0f,
                        contrast = params["contrast"] ?: 1.0f,
                        saturation = params["saturation"] ?: 1.0f,
                        hueDeg = params["hue"] ?: 0.0f,
                        temperature = params["temperature"] ?: 0.0f,
                        tint = params["tint"] ?: 0.0f,
                        sepia = params["sepia"] ?: 0.0f,
                        bw = params["bw"] ?: 0.0f,
                        intensity = intensity
                    )
                    appliedCount++
                }
                "vignette" -> {
                    applyVignette(
                        pixels = pixels,
                        width = width,
                        height = height,
                        radius = params["radius"] ?: 0.75f,
                        softness = params["softness"] ?: 0.45f,
                        amount = params["amount"] ?: 0.40f,
                        intensity = intensity
                    )
                    appliedCount++
                }
                "lut" -> {
                    val lutAsset = node.postAssetId?.let { effect.assets[it] }
                        ?: effect.assets.values.firstOrNull { it.type == "lut" }
                    val lutBitmap = lutAsset?.let { asset ->
                        val file = java.io.File(asset.localPath)
                        if (file.exists()) {
                            val opts = android.graphics.BitmapFactory.Options().apply {
                                inScaled = false
                                inPremultiplied = false
                                inPreferredConfig = Bitmap.Config.ARGB_8888
                            }
                            android.graphics.BitmapFactory.decodeFile(file.absolutePath, opts)
                        } else {
                            null
                        }
                    }
                    val effLutAmount = ((params["amount"] ?: 1.0f).coerceIn(0f, 1f) * intensity).coerceIn(0f, 1f)
                    val gridHint = params["gridSize"]?.roundToInt() ?: 0
                    applyLut(
                        pixels = pixels,
                        lutBitmap = lutBitmap,
                        intensity = effLutAmount,
                        gridSizeHint = gridHint
                    )
                    if (lutBitmap != null && !lutBitmap.isRecycled) {
                        lutBitmap.recycle()
                    }
                    appliedCount++
                }
                "grain" -> {
                    val effSeed = resolveEffectiveGrainSeed(
                        paramSeed = params["seed"] ?: 0.0f,
                        frameContext = frameContext,
                        speedHz = params["speed"] ?: 24.0f
                    )
                    lastEffectiveGrainSeed = effSeed
                    applyGrain(
                        pixels = pixels,
                        width = width,
                        height = height,
                        amount = params["amount"] ?: 0.15f,
                        size = params["size"] ?: 1.0f,
                        seed = effSeed,
                        intensity = intensity,
                        lumaWeight = params["lumaWeight"] ?: 0.75f
                    )
                    appliedCount++
                }
                "sharpen" -> {
                    applySharpen(
                        pixels = pixels,
                        scratch = scratch,
                        width = width,
                        height = height,
                        amount = params["amount"] ?: 0.35f,
                        radius = params["radius"] ?: 1.0f,
                        threshold = params["threshold"] ?: 0.02f,
                        intensity = intensity
                    )
                    appliedCount++
                }
            }
            if (preMaskBackup != null && nodeMask != null) {
                blendMaskedPassIfNeeded(preMaskBackup, pixels, nodeMask)
            }
        }

        if (appliedCount > 0) {
            composedBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        }
        lastExecutedPostCount = appliedCount
        return appliedCount
    }

    private fun applyColorGrade(
        pixels: IntArray,
        warmth: Float,
        tint: Float = 0.0f,
        contrast: Float,
        saturation: Float,
        exposure: Float,
        intensity: Float
    ) {
        val expScale = Math.pow(2.0, exposure.toDouble()).toFloat()
        val warmOffset = warmth * 0.08f
        val tintOffset = tint * 0.08f
        val mixW = intensity.coerceIn(0f, 1f)

        for (i in pixels.indices) {
            val c = pixels[i]
            val a = (c ushr 24) and 0xFF
            val origR = ((c ushr 16) and 0xFF) / 255f
            val origG = ((c ushr 8) and 0xFF) / 255f
            val origB = (c and 0xFF) / 255f

            var r = origR * expScale + warmOffset
            var g = origG * expScale + tintOffset
            var b = origB * expScale - warmOffset

            r = (r - 0.5f) * contrast + 0.5f
            g = (g - 0.5f) * contrast + 0.5f
            b = (b - 0.5f) * contrast + 0.5f

            val y = 0.299f * r + 0.587f * g + 0.114f * b
            r = y + (r - y) * saturation
            g = y + (g - y) * saturation
            b = y + (b - y) * saturation

            r = (origR + (r.coerceIn(0f, 1f) - origR) * mixW).coerceIn(0f, 1f)
            g = (origG + (g.coerceIn(0f, 1f) - origG) * mixW).coerceIn(0f, 1f)
            b = (origB + (b.coerceIn(0f, 1f) - origB) * mixW).coerceIn(0f, 1f)

            val outR = (r * 255f).roundToInt().coerceIn(0, 255)
            val outG = (g * 255f).roundToInt().coerceIn(0, 255)
            val outB = (b * 255f).roundToInt().coerceIn(0, 255)
            pixels[i] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
        }
    }

    private fun applyBlurOrBilateral(
        pixels: IntArray,
        scratch: IntArray,
        width: Int,
        height: Int,
        radius: Float,
        amount: Float,
        edgePreserve: Float,
        detailPreserve: Float,
        bilateral: Boolean,
        intensity: Float,
        skinMask: FloatArray?
    ) {
        val baseBlend = (amount.coerceIn(0f, 1f) * intensity.coerceIn(0f, 1f)).coerceIn(0f, 1f)
        if (baseBlend <= 0.001f || radius <= 0.05f) return

        System.arraycopy(pixels, 0, scratch, 0, pixels.size)
        val stepPx = max(1, radius.roundToInt().coerceIn(1, 8))
        val sigmaRange = max(0.04f, (1.0f - edgePreserve.coerceIn(0f, 1f) * 0.85f) * 0.35f)
        val invTwoSigmaSq = 1.0f / (2.0f * sigmaRange * sigmaRange)
        val detailGain = detailPreserve.coerceIn(0f, 1f) * 0.45f

        val offsetsX = intArrayOf(-stepPx, 0, stepPx, -stepPx, 0, stepPx, -stepPx, 0, stepPx)
        val offsetsY = intArrayOf(-stepPx, -stepPx, -stepPx, 0, 0, 0, stepPx, stepPx, stepPx)
        val spatialWeights = floatArrayOf(0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f)

        for (y in 0 until height) {
            val rowOffset = y * width
            for (x in 0 until width) {
                val idx = rowOffset + x
                val maskW = if (skinMask != null && idx < skinMask.size) {
                    skinMask[idx].coerceIn(0f, 1f)
                } else {
                    1f
                }
                if (maskW <= 1e-4f) continue

                val effBlend = (baseBlend * maskW).coerceIn(0f, 1f)
                if (effBlend <= 1e-4f) continue

                val centerPx = scratch[idx]
                val a = (centerPx ushr 24) and 0xFF
                val cR = ((centerPx ushr 16) and 0xFF) / 255f
                val cG = ((centerPx ushr 8) and 0xFF) / 255f
                val cB = (centerPx and 0xFF) / 255f
                val cLuma = 0.299f * cR + 0.587f * cG + 0.114f * cB

                var sumR = 0f
                var sumG = 0f
                var sumB = 0f
                var wSum = 0f

                var tightR = 0f
                var tightG = 0f
                var tightB = 0f
                var tightCount = 0f

                for (k in 0 until 9) {
                    val sx = (x + offsetsX[k]).coerceIn(0, width - 1)
                    val sy = (y + offsetsY[k]).coerceIn(0, height - 1)
                    val sPx = scratch[sy * width + sx]
                    val sR = ((sPx ushr 16) and 0xFF) / 255f
                    val sG = ((sPx ushr 8) and 0xFF) / 255f
                    val sB = (sPx and 0xFF) / 255f

                    var w = spatialWeights[k]
                    if (bilateral && k != 4) {
                        val sLuma = 0.299f * sR + 0.587f * sG + 0.114f * sB
                        val dy = sLuma - cLuma
                        w *= exp(-(dy * dy) * invTwoSigmaSq)
                    }
                    sumR += sR * w
                    sumG += sG * w
                    sumB += sB * w
                    wSum += w

                    if (bilateral && detailGain > 0.001f) {
                        val tx = (x + offsetsX[k].coerceIn(-1, 1)).coerceIn(0, width - 1)
                        val ty = (y + offsetsY[k].coerceIn(-1, 1)).coerceIn(0, height - 1)
                        val tPx = scratch[ty * width + tx]
                        tightR += ((tPx ushr 16) and 0xFF) / 255f
                        tightG += ((tPx ushr 8) and 0xFF) / 255f
                        tightB += (tPx and 0xFF) / 255f
                        tightCount += 1f
                    }
                }

                var filtR = sumR / wSum
                var filtG = sumG / wSum
                var filtB = sumB / wSum

                if (bilateral && detailGain > 0.001f && tightCount > 0f) {
                    val avgTR = tightR / tightCount
                    val avgTG = tightG / tightCount
                    val avgTB = tightB / tightCount
                    filtR = (filtR + (cR - avgTR) * detailGain).coerceIn(0f, 1f)
                    filtG = (filtG + (cG - avgTG) * detailGain).coerceIn(0f, 1f)
                    filtB = (filtB + (cB - avgTB) * detailGain).coerceIn(0f, 1f)
                }

                val outR = ((cR + (filtR - cR) * effBlend).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outG = ((cG + (filtG - cG) * effBlend).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outB = ((cB + (filtB - cB) * effBlend).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                pixels[idx] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    private fun applySoftFocus(
        pixels: IntArray,
        scratch: IntArray,
        width: Int,
        height: Int,
        amount: Float,
        radius: Float,
        threshold: Float,
        warmth: Float,
        intensity: Float
    ) {
        val effAmount = (amount.coerceIn(0f, 1f) * intensity.coerceIn(0f, 1f)).coerceIn(0f, 1f)
        if (effAmount <= 0.001f) return

        System.arraycopy(pixels, 0, scratch, 0, pixels.size)
        val stepPx = max(1, (radius * 0.75f).roundToInt().coerceIn(1, 12))
        val threshLow = (threshold.coerceIn(0f, 1f) * 0.65f).coerceIn(0f, 0.95f)
        val warmOffset = warmth.coerceIn(-1f, 1f) * 0.06f

        val kOffX = intArrayOf(-stepPx, stepPx, -stepPx, stepPx, -stepPx * 2, stepPx * 2, 0, 0)
        val kOffY = intArrayOf(-stepPx, -stepPx, stepPx, stepPx, 0, 0, -stepPx * 2, stepPx * 2)

        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                val idx = row + x
                val cPx = scratch[idx]
                val a = (cPx ushr 24) and 0xFF
                val origR = ((cPx ushr 16) and 0xFF) / 255f
                val origG = ((cPx ushr 8) and 0xFF) / 255f
                val origB = (cPx and 0xFF) / 255f
                val origLuma = 0.299f * origR + 0.587f * origG + 0.114f * origB

                var bloomR = 0f
                var bloomG = 0f
                var bloomB = 0f
                for (k in 0 until 8) {
                    val sx = (x + kOffX[k]).coerceIn(0, width - 1)
                    val sy = (y + kOffY[k]).coerceIn(0, height - 1)
                    val sPx = scratch[sy * width + sx]
                    val sR = ((sPx ushr 16) and 0xFF) / 255f
                    val sG = ((sPx ushr 8) and 0xFF) / 255f
                    val sB = (sPx and 0xFF) / 255f
                    val sLuma = 0.299f * sR + 0.587f * sG + 0.114f * sB
                    val hw = smoothstep(threshLow, 1.0f, sLuma)
                    val w = 0.25f + 0.75f * hw
                    bloomR += sR * w
                    bloomG += sG * w
                    bloomB += sB * w
                }
                bloomR = (bloomR / 8f + warmOffset).coerceIn(0f, 1f)
                bloomG = (bloomG / 8f).coerceIn(0f, 1f)
                bloomB = (bloomB / 8f - warmOffset).coerceIn(0f, 1f)

                val screenR = 1f - (1f - origR) * (1f - bloomR * 0.65f)
                val screenG = 1f - (1f - origG) * (1f - bloomG * 0.65f)
                val screenB = 1f - (1f - origB) * (1f - bloomB * 0.65f)

                val darkGuard = smoothstep(0.08f, 0.35f, origLuma)
                val mixW = (effAmount * (0.35f + 0.65f * darkGuard)).coerceIn(0f, 1f)

                val outR = ((origR + (screenR - origR) * mixW).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outG = ((origG + (screenG - origG) * mixW).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outB = ((origB + (screenB - origB) * mixW).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                pixels[idx] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    private val warpAnchorScratch = WarpAnchorCache()

    private fun applyWarp(
        pixels: IntArray,
        scratch: IntArray,
        width: Int,
        height: Int,
        faceState: FaceState,
        params: Map<String, Float>,
        intensity: Float,
        regionMaskPass: RegionMaskPass? = null
    ) {
        System.arraycopy(pixels, 0, scratch, 0, pixels.size)
        val aspect = width.toFloat() / height.toFloat().coerceAtLeast(1f)
        val effParams = HashMap(params)
        effParams["intensity"] = intensity

        val deltaUv = FloatArray(2)
        val maxX = (width - 1).coerceAtLeast(1)
        val maxY = (height - 1).coerceAtLeast(1)

        // Smooth landmarks once per frame via RegionMaskPass.smoother and hoist all warp anchor projections
        val tsNs = if (faceState.poseTimestampNs > 0L) faceState.poseTimestampNs else faceState.timestampNs
        val smoothedLms = regionMaskPass?.smoother?.smooth(faceState.landmarks, tsNs, faceState.isVisible)
            ?: faceState.landmarks
        val anchors = populateWarpAnchors(faceState, smoothedLms, warpAnchorScratch)
        val faceCenterU = anchors.faceCenterU
        val faceCenterV = anchors.faceCenterV
        val faceHalfW = anchors.faceHalfW * 1.08f
        val faceHalfH = anchors.faceHalfH * 1.08f

        val xStart = ((faceCenterU - faceHalfW) * width).toInt().coerceIn(0, maxX)
        val xEnd = ((faceCenterU + faceHalfW) * width).toInt().coerceIn(0, maxX)
        val yStart = ((faceCenterV - faceHalfH) * height).toInt().coerceIn(0, maxY)
        val yEnd = ((faceCenterV + faceHalfH) * height).toInt().coerceIn(0, maxY)

        for (y in yStart..yEnd) {
            val v = (y + 0.5f) / height.toFloat()
            val row = y * width
            for (x in xStart..xEnd) {
                val u = (x + 0.5f) / width.toFloat()
                computeWarpForwardDisplacementUv(
                    u = u,
                    v = v,
                    faceState = faceState,
                    params = effParams,
                    aspect = aspect,
                    outDeltaUv = deltaUv,
                    landmarksOverride = smoothedLms,
                    anchorCache = anchors
                )
                if (abs(deltaUv[0]) <= 1e-6f && abs(deltaUv[1]) <= 1e-6f) {
                    continue
                }
                val sampleU = (u - deltaUv[0]).coerceIn(0f, 1f)
                val sampleV = (v - deltaUv[1]).coerceIn(0f, 1f)
                val fx = (sampleU * width.toFloat() - 0.5f).coerceIn(0f, maxX.toFloat())
                val fy = (sampleV * height.toFloat() - 0.5f).coerceIn(0f, maxY.toFloat())

                val x0 = fx.toInt().coerceIn(0, maxX)
                val y0 = fy.toInt().coerceIn(0, maxY)
                val x1 = (x0 + 1).coerceAtMost(maxX)
                val y1 = (y0 + 1).coerceAtMost(maxY)
                val tx = fx - x0
                val ty = fy - y0

                val p00 = scratch[y0 * width + x0]
                val p10 = scratch[y0 * width + x1]
                val p01 = scratch[y1 * width + x0]
                val p11 = scratch[y1 * width + x1]

                val a = (scratch[row + x] ushr 24) and 0xFF
                val r = bilinearChannel(p00, p10, p01, p01.let { p11 }, 16, tx, ty)
                val g = bilinearChannel(p00, p10, p01, p11, 8, tx, ty)
                val b = bilinearChannel(p00, p10, p01, p11, 0, tx, ty)
                pixels[row + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    private fun bilinearChannel(p00: Int, p10: Int, p01: Int, p11: Int, shift: Int, tx: Float, ty: Float): Int {
        val c00 = ((p00 ushr shift) and 0xFF).toFloat()
        val c10 = ((p10 ushr shift) and 0xFF).toFloat()
        val c01 = ((p01 ushr shift) and 0xFF).toFloat()
        val c11 = ((p11 ushr shift) and 0xFF).toFloat()
        val top = c00 + (c10 - c00) * tx
        val bot = c01 + (c11 - c01) * tx
        return (top + (bot - top) * ty).roundToInt().coerceIn(0, 255)
    }

    private fun applyColorMatrix(
        pixels: IntArray,
        brightness: Float,
        contrast: Float,
        saturation: Float,
        hueDeg: Float,
        temperature: Float,
        tint: Float,
        sepia: Float,
        bw: Float,
        intensity: Float
    ) {
        val mixW = intensity.coerceIn(0f, 1f)
        val hueRad = Math.toRadians(hueDeg.toDouble()).toFloat()
        val cosH = cos(hueRad)
        val sinH = sin(hueRad)
        val sepiaW = sepia.coerceIn(0f, 1f)
        val bwW = bw.coerceIn(0f, 1f)

        for (i in pixels.indices) {
            val c = pixels[i]
            val a = (c ushr 24) and 0xFF
            val origR = ((c ushr 16) and 0xFF) / 255f
            val origG = ((c ushr 8) and 0xFF) / 255f
            val origB = (c and 0xFF) / 255f

            var r = origR + brightness
            var g = origG + brightness
            var b = origB + brightness

            r = (r - 0.5f) * contrast + 0.5f
            g = (g - 0.5f) * contrast + 0.5f
            b = (b - 0.5f) * contrast + 0.5f

            r += temperature * 0.12f
            b -= temperature * 0.12f
            g += tint * 0.10f

            if (abs(hueDeg) > 0.01f) {
                val y = 0.299f * r + 0.587f * g + 0.114f * b
                val iCh = 0.596f * r - 0.274f * g - 0.322f * b
                val qCh = 0.211f * r - 0.523f * g + 0.312f * b
                val iRot = iCh * cosH - qCh * sinH
                val qRot = iCh * sinH + qCh * cosH
                r = y + 0.956f * iRot + 0.621f * qRot
                g = y - 0.272f * iRot - 0.647f * qRot
                b = y - 1.106f * iRot + 1.703f * qRot
            }

            val luma = 0.299f * r + 0.587f * g + 0.114f * b
            r = luma + (r - luma) * saturation
            g = luma + (g - luma) * saturation
            b = luma + (b - luma) * saturation

            if (sepiaW > 0.001f) {
                val sR = (luma * 1.12f).coerceIn(0f, 1f)
                val sG = (luma * 0.95f).coerceIn(0f, 1f)
                val sB = (luma * 0.74f).coerceIn(0f, 1f)
                r += (sR - r) * sepiaW
                g += (sG - g) * sepiaW
                b += (sB - b) * sepiaW
            }

            if (bwW > 0.001f) {
                val bwLuma = (0.299f * r + 0.587f * g + 0.114f * b).coerceIn(0f, 1f)
                r += (bwLuma - r) * bwW
                g += (bwLuma - g) * bwW
                b += (bwLuma - b) * bwW
            }

            r = (origR + (r.coerceIn(0f, 1f) - origR) * mixW).coerceIn(0f, 1f)
            g = (origG + (g.coerceIn(0f, 1f) - origG) * mixW).coerceIn(0f, 1f)
            b = (origB + (b.coerceIn(0f, 1f) - origB) * mixW).coerceIn(0f, 1f)

            val outR = (r * 255f).roundToInt().coerceIn(0, 255)
            val outG = (g * 255f).roundToInt().coerceIn(0, 255)
            val outB = (b * 255f).roundToInt().coerceIn(0, 255)
            pixels[i] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
        }
    }

    private fun applyVignette(
        pixels: IntArray,
        width: Int,
        height: Int,
        radius: Float,
        softness: Float,
        amount: Float,
        intensity: Float
    ) {
        val effAmount = (amount.coerceIn(0f, 1f) * intensity.coerceIn(0f, 1f)).coerceIn(0f, 1f)
        if (effAmount <= 0.001f) return
        val edge0 = radius.coerceAtLeast(0f)
        val edge1 = edge0 + max(0.01f, softness)

        for (y in 0 until height) {
            val ny = ((y + 0.5f) / height.toFloat() - 0.5f) * 2.0f
            val row = y * width
            for (x in 0 until width) {
                val nx = ((x + 0.5f) / width.toFloat() - 0.5f) * 2.0f
                val dist = hypot(nx, ny)
                val vig = (1.0f - smoothstep(edge0, edge1, dist) * effAmount).coerceIn(0f, 1f)
                val idx = row + x
                val c = pixels[idx]
                val a = (c ushr 24) and 0xFF
                val r = (((c ushr 16) and 0xFF) * vig).roundToInt().coerceIn(0, 255)
                val g = (((c ushr 8) and 0xFF) * vig).roundToInt().coerceIn(0, 255)
                val b = ((c and 0xFF) * vig).roundToInt().coerceIn(0, 255)
                pixels[idx] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    private fun applyLut(
        pixels: IntArray,
        lutBitmap: Bitmap?,
        intensity: Float,
        gridSizeHint: Int = 0
    ) {
        val effIntensity = intensity.coerceIn(0f, 1f)
        if (effIntensity <= 0.001f) return
        val lutW = lutBitmap?.width ?: 0
        val lutH = lutBitmap?.height ?: 0
        val lutPixels = if (lutBitmap != null && lutW >= 4 && lutH >= 4) {
            IntArray(lutW * lutH).also { lutBitmap.getPixels(it, 0, lutW, 0, 0, lutW, lutH) }
        } else {
            null
        }
        val rgbScratch = FloatArray(3)
        for (i in pixels.indices) {
            val c = pixels[i]
            val a = (c ushr 24) and 0xFF
            val origR = ((c ushr 16) and 0xFF) / 255f
            val origG = ((c ushr 8) and 0xFF) / 255f
            val origB = (c and 0xFF) / 255f
            evaluateLutRgb(
                r = origR,
                g = origG,
                b = origB,
                lutBitmap = lutBitmap,
                intensity = effIntensity,
                outRgb = rgbScratch,
                lutPixels = lutPixels,
                lutWidth = lutW,
                lutHeight = lutH,
                gridSizeHint = gridSizeHint
            )
            val outR = (rgbScratch[0] * 255f).roundToInt().coerceIn(0, 255)
            val outG = (rgbScratch[1] * 255f).roundToInt().coerceIn(0, 255)
            val outB = (rgbScratch[2] * 255f).roundToInt().coerceIn(0, 255)
            pixels[i] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
        }
    }

    private fun applyGrain(
        pixels: IntArray,
        width: Int,
        height: Int,
        amount: Float,
        size: Float,
        seed: Float,
        intensity: Float,
        lumaWeight: Float = 0.75f
    ) {
        val effAmount = (amount.coerceIn(0f, 1f) * intensity.coerceIn(0f, 1f)).coerceIn(0f, 1f)
        if (effAmount <= 0.001f) return
        val cellPx = size.coerceIn(0.25f, 8.0f)
        val seedBits = (seed * 1337.0f).roundToInt()
        val lw = lumaWeight.coerceIn(0f, 1f)
        val invLw = 1.0f - lw

        for (y in 0 until height) {
            val cy = (y / cellPx).toInt()
            val row = y * width
            for (x in 0 until width) {
                val cx = (x / cellPx).toInt()
                var h = (cx * 374761393) xor (cy * 668265263) xor (seedBits * 1442695041)
                h = (h xor (h ushr 13)) * 1274126177
                h = h xor (h ushr 16)
                val noise01 = ((h and 0x7FFFFFFF).toFloat() / 2147483647.0f) - 0.5f

                val idx = row + x
                val c = pixels[idx]
                val a = (c ushr 24) and 0xFF
                val r = ((c ushr 16) and 0xFF) / 255f
                val g = ((c ushr 8) and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                val luma = 0.299f * r + 0.587f * g + 0.114f * b
                val midtoneWeight = invLw + lw * (4.0f * luma * (1.0f - luma))
                val delta = noise01 * effAmount * 0.32f * midtoneWeight

                val outR = ((r + delta).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outG = ((g + delta).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outB = ((b + delta).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                pixels[idx] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    private fun applySharpen(
        pixels: IntArray,
        scratch: IntArray,
        width: Int,
        height: Int,
        amount: Float,
        radius: Float,
        threshold: Float,
        intensity: Float
    ) {
        val effAmount = (amount.coerceIn(0f, 1f) * intensity.coerceIn(0f, 1f)).coerceIn(0f, 1f)
        if (effAmount <= 0.001f) return
        System.arraycopy(pixels, 0, scratch, 0, pixels.size)
        val stepPx = max(1, radius.roundToInt().coerceIn(1, 4))
        val thresh = threshold.coerceIn(0f, 1f)

        for (y in 0 until height) {
            val yUp = (y - stepPx).coerceAtLeast(0)
            val yDn = (y + stepPx).coerceAtMost(height - 1)
            val row = y * width
            for (x in 0 until width) {
                val xL = (x - stepPx).coerceAtLeast(0)
                val xR = (x + stepPx).coerceAtMost(width - 1)
                val idx = row + x
                val c = scratch[idx]
                val a = (c ushr 24) and 0xFF
                val cR = ((c ushr 16) and 0xFF) / 255f
                val cG = ((c ushr 8) and 0xFF) / 255f
                val cB = (c and 0xFF) / 255f

                val pL = scratch[row + xL]
                val pR = scratch[row + xR]
                val pU = scratch[yUp * width + x]
                val pD = scratch[yDn * width + x]

                val bR = 0.25f * (((pL ushr 16) and 0xFF) + ((pR ushr 16) and 0xFF) + ((pU ushr 16) and 0xFF) + ((pD ushr 16) and 0xFF)) / 255f
                val bG = 0.25f * (((pL ushr 8) and 0xFF) + ((pR ushr 8) and 0xFF) + ((pU ushr 8) and 0xFF) + ((pD ushr 8) and 0xFF)) / 255f
                val bB = 0.25f * ((pL and 0xFF) + (pR and 0xFF) + (pU and 0xFF) + (pD and 0xFF)) / 255f

                val hpR = cR - bR
                val hpG = cG - bG
                val hpB = cB - bB
                val hpMag = sqrt(hpR * hpR + hpG * hpG + hpB * hpB)
                val coring = smoothstep(thresh, thresh + 0.04f, hpMag)
                val gain = effAmount * 1.6f * coring

                val outR = ((cR + hpR * gain).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outG = ((cG + hpG * gain).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                val outB = ((cB + hpB * gain).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                pixels[idx] = (a shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    @JvmOverloads
    fun release(resourceOwner: GlResourceOwner? = null) {
        releasePingPongFbo(resourceOwner)
        if (resourceOwner != null) {
            if (quadVao != 0) resourceOwner.queueDeleteVao(quadVao)
            if (quadVbo != 0) resourceOwner.queueDeleteBuffer(quadVbo)
            if (program != 0) resourceOwner.queueDeleteProgram(program)
        } else if (hasActiveGlContext()) {
            try {
                if (quadVao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(quadVao), 0)
                if (quadVbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(quadVbo), 0)
                if (program != 0) GLES30.glDeleteProgram(program)
            } catch (_: Throwable) {
            }
        }
        quadVao = 0
        quadVbo = 0
        program = 0
    }

    private fun releasePingPongFbo(resourceOwner: GlResourceOwner? = null) {
        if (resourceOwner != null) {
            if (pingPongFbo != 0) resourceOwner.queueDeleteFramebuffer(pingPongFbo)
            if (pingPongTex != 0) resourceOwner.queueDeleteTexture(pingPongTex)
        } else if (hasActiveGlContext()) {
            try {
                if (pingPongFbo != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(pingPongFbo), 0)
                if (pingPongTex != 0) GLES30.glDeleteTextures(1, intArrayOf(pingPongTex), 0)
            } catch (_: Throwable) {
            }
        }
        pingPongFbo = 0
        pingPongTex = 0
        fboWidth = 0
        fboHeight = 0
    }

    private fun createProgram(vsSource: String, fsSource: String): Int {
        val vs = compileShader(GLES30.GL_VERTEX_SHADER, vsSource)
        if (vs == 0) return 0
        val fs = compileShader(GLES30.GL_FRAGMENT_SHADER, fsSource)
        if (fs == 0) {
            GLES30.glDeleteShader(vs)
            return 0
        }
        val prog = GLES30.glCreateProgram()
        GLES30.glAttachShader(prog, vs)
        GLES30.glAttachShader(prog, fs)
        GLES30.glLinkProgram(prog)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        return prog
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        if (shader == 0) return 0
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        return shader
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
