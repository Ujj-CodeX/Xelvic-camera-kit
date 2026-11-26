package com.example.engine.render

import com.example.engine.governor.TierPolicy
import com.example.engine.scene.PropertySlots
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToInt

/**
 * RenderTargetRef (Flow 2 — Chunk G0):
 *
 * Canonical named render target identifiers used by data-driven [PassNode] declarations:
 * - `$camera`: Input camera frame texture (Path B `CameraInputSurface` or photo base FBO).
 * - `$scene`: Main composed RGBA8 + Depth scene target.
 * - `$mask:<id>`: Half-resolution `R8` mask target produced by `RegionMaskPass` / mask algebra.
 * - `$rt:<name>`: Named intermediate `RGBA8` target allocated from [RtPool].
 * - `$output`: Final display surface or photo output FBO.
 */
data class RenderTargetRef(
    val raw: String,
    val kind: TargetKind,
    val identifier: String
) {
    enum class TargetKind {
        CAMERA,
        SCENE,
        MASK,
        NAMED_RT,
        OUTPUT
    }

    companion object {
        const val TARGET_CAMERA = "\$camera"
        const val TARGET_SCENE = "\$scene"
        const val TARGET_OUTPUT = "\$output"
        const val PREFIX_MASK = "\$mask:"
        const val PREFIX_RT = "\$rt:"

        fun parse(raw: String): RenderTargetRef {
            val trimmed = raw.trim()
            return when {
                trimmed == TARGET_CAMERA -> RenderTargetRef(TARGET_CAMERA, TargetKind.CAMERA, "camera")
                trimmed == TARGET_SCENE -> RenderTargetRef(TARGET_SCENE, TargetKind.SCENE, "scene")
                trimmed == TARGET_OUTPUT -> RenderTargetRef(TARGET_OUTPUT, TargetKind.OUTPUT, "output")
                trimmed.startsWith(PREFIX_MASK) -> {
                    val id = trimmed.removePrefix(PREFIX_MASK).ifEmpty { "default" }
                    RenderTargetRef("$PREFIX_MASK$id", TargetKind.MASK, id)
                }
                trimmed.startsWith(PREFIX_RT) -> {
                    val id = trimmed.removePrefix(PREFIX_RT).ifEmpty { "temp" }
                    RenderTargetRef("$PREFIX_RT$id", TargetKind.NAMED_RT, id)
                }
                else -> {
                    val id = trimmed.removePrefix("$").ifEmpty { "temp" }
                    RenderTargetRef("$PREFIX_RT$id", TargetKind.NAMED_RT, id)
                }
            }
        }
    }
}

/**
 * DeterministicFrameContext (Flow 2 — Chunk G0):
 *
 * Shared deterministic timing and pseudo-random seed context passed to both Live Preview
 * (`RenderGraph.render`) and Still Photo (`OffscreenRenderer` / `StillPipeline`).
 * Ensures time-varying or seeded effects (such as `grain@1`, shimmer, or procedural noise)
 * produce bit-identical sequences when evaluated with the same package and `renderTimeSec`.
 */
data class DeterministicFrameContext(
    val frameSeed: Int,
    val renderTimeSec: Float,
    val frameIndex: Long = 0L,
    val isPhotoPass: Boolean = false
) {
    /**
     * Generates a deterministic normalized float in `[0.0, 1.0]` for `(pixelX, pixelY)` using
     * [frameSeed], ensuring fused-shader grain and CPU fallback grain match between preview and photo.
     */
    fun seededNoise01(pixelX: Int, pixelY: Int, salt: Int = 0): Float {
        var h = frameSeed xor (pixelX * 374761393) xor (pixelY * 668265263) xor (salt * 1442695041)
        h = (h xor (h ushr 13)) * 1274126177
        h = h xor (h ushr 16)
        return ((h and 0x7FFFFFFF).toFloat() / Int.MAX_VALUE.toFloat()).coerceIn(0f, 1f)
    }

    companion object {
        val DEFAULT = DeterministicFrameContext(
            frameSeed = 0x58454C56, // "XELV"
            renderTimeSec = 0f,
            frameIndex = 0L,
            isPhotoPass = false
        )

        fun computeDeterministicSeed(
            packageId: String,
            renderTimeSec: Float,
            explicitSeed: Int? = null
        ): Int {
            if (explicitSeed != null) return explicitSeed
            val quantizedTimeMs = (renderTimeSec.coerceAtLeast(0f) * 1000f).roundToInt()
            var hash = -2128831035 // FNV-1a offset basis
            for (i in 0 until packageId.length) {
                hash = hash xor packageId[i].code
                hash *= 16777619
            }
            hash = hash xor quantizedTimeMs
            hash *= 16777619
            return if (hash == 0) 0x58454C56 else hash
        }

        fun forPreview(
            packageId: String,
            renderTimeSec: Float,
            frameIndex: Long,
            explicitSeed: Int? = null
        ): DeterministicFrameContext {
            return DeterministicFrameContext(
                frameSeed = computeDeterministicSeed(packageId, renderTimeSec, explicitSeed),
                renderTimeSec = renderTimeSec,
                frameIndex = frameIndex,
                isPhotoPass = false
            )
        }

        fun forPhoto(
            packageId: String,
            photoTimeSec: Float,
            explicitSeed: Int? = null
        ): DeterministicFrameContext {
            return DeterministicFrameContext(
                frameSeed = computeDeterministicSeed(packageId, photoTimeSec, explicitSeed),
                renderTimeSec = photoTimeSec,
                frameIndex = 0L,
                isPhotoPass = true
            )
        }
    }
}

/**
 * PassStageKind (Flow 2 — Chunk G0):
 *
 * Identifies either one of the 9 canonical engine stages or a custom data-driven pass node.
 */
enum class PassStageKind {
    CAMERA_FRAME,
    CAMERA_PIXEL_PASSES,
    CLEAR_DEPTH,
    OCCLUDERS,
    OPAQUE_NODES,
    TRANSPARENT_NODES,
    POST_NODES,
    OVERLAY_NODES,
    FINAL_OUTPUT,
    CUSTOM_PASS;

    fun toLegacyRenderPass(): RenderGraph.RenderPass? {
        return when (this) {
            CAMERA_FRAME -> RenderGraph.RenderPass.CAMERA_FRAME
            CAMERA_PIXEL_PASSES -> RenderGraph.RenderPass.CAMERA_PIXEL_PASSES
            CLEAR_DEPTH -> RenderGraph.RenderPass.CLEAR_DEPTH
            OCCLUDERS -> RenderGraph.RenderPass.OCCLUDERS
            OPAQUE_NODES -> RenderGraph.RenderPass.OPAQUE_NODES
            TRANSPARENT_NODES -> RenderGraph.RenderPass.TRANSPARENT_NODES
            POST_NODES -> RenderGraph.RenderPass.POST_NODES
            OVERLAY_NODES -> RenderGraph.RenderPass.OVERLAY_NODES
            FINAL_OUTPUT -> RenderGraph.RenderPass.FINAL_OUTPUT
            CUSTOM_PASS -> null
        }
    }
}

/**
 * PassNode (Flow 2 — Chunk G0):
 *
 * A single node in the data-driven [PassGraph].
 */
data class PassNode(
    val id: String,
    val stageKind: PassStageKind,
    val templateId: String = stageKind.name,
    val reads: List<String> = emptyList(),
    val writes: String = RenderTargetRef.TARGET_SCENE,
    val scaleFactor: Float = 1.0f,
    val blendMode: String = "alpha",
    val params: Map<String, Float> = emptyMap(),
    val enabledByDefault: Boolean = true,
    val assetId: String? = null
) {
    val parsedReads: List<RenderTargetRef> = reads.map { RenderTargetRef.parse(it) }
    val parsedWrites: RenderTargetRef = RenderTargetRef.parse(writes)
}

/**
 * Per-pass execution record captured during `RenderGraph.render(...)` for HUD diagnostics and tests.
 */
data class PassExecutionRecord(
    val passId: String,
    val stageKind: PassStageKind,
    val templateId: String,
    val reads: List<String>,
    val writes: String,
    val effectiveScaleFactor: Float,
    val durationMs: Float,
    val resolvedParams: Map<String, Float>
)

/**
 * PassGraph (Flow 2 — Chunk G0):
 *
 * Data-driven ordered pass graph executor.
 * Expresses the canonical 9-stage pipeline (`camera_frame -> camera_pixel_passes -> clear_depth ->
 * occluders -> opaque_nodes -> transparent_nodes -> post_nodes -> overlay_nodes -> final_output`)
 * in the exact same data model as custom manifest-declared `"passes"`, guaranteeing a zero-regression
 * gate for all existing packages while enabling arbitrary multi-pass effects, named render targets,
 * data-driven tier pass disabling (`tiers[tier].disablePasses`), scale overrides (`passScaleOverrides`),
 * and universal signal-to-parameter bindings (`pass:<passId>.<paramName>`).
 */
class PassGraph(
    val passes: List<PassNode>,
    val isCustomManifestGraph: Boolean = false
) {

    /**
     * Resolves whether `passNode` is active in the current tier (respecting `tierConfig.disabledPasses`).
     */
    fun isPassEnabledInTier(
        passNode: PassNode,
        tierConfig: TierPolicy.EffectiveTierConfig?
    ): Boolean {
        if (!passNode.enabledByDefault) return false
        if (tierConfig == null) return true
        return passNode.id !in tierConfig.disabledPasses &&
            passNode.templateId !in tierConfig.disabledPasses
    }

    /**
     * Resolves the effective resolution scale (`1.0`, `0.5`, `0.25`) for `passNode` in the active tier.
     */
    fun resolveEffectiveScale(
        passNode: PassNode,
        tierConfig: TierPolicy.EffectiveTierConfig?
    ): Float {
        val overrideScale = tierConfig?.passScaleOverrides?.get(passNode.id)
            ?: tierConfig?.passScaleOverrides?.get(passNode.templateId)
        return (overrideScale ?: passNode.scaleFactor).coerceIn(0.125f, 1.0f)
    }

    /**
     * Resolves the runtime parameter map for `passNode`, combining:
     * 1. Base compiled `passNode.params`
     * 2. Tier parameter overrides (`tierConfig.paramOverrides["pass:<id>.<param>"]` or `["<id>.<param>"]`)
     * 3. Dynamic `PropertySlots` signal bindings (`pass:<id>.<param>`) written by `BehaviorRunner`
     */
    fun resolvePassParams(
        passNode: PassNode,
        slots: PropertySlots?,
        tierConfig: TierPolicy.EffectiveTierConfig?
    ): Map<String, Float> {
        if (passNode.params.isEmpty() &&
            (tierConfig == null || tierConfig.paramOverrides.isEmpty()) &&
            (slots == null || slots.dynamicParamCount == 0)
        ) {
            return emptyMap()
        }

        val out = LinkedHashMap<String, Float>(passNode.params.size + 4)
        for ((k, v) in passNode.params) {
            var resolved = v
            if (tierConfig != null && tierConfig.paramOverrides.isNotEmpty()) {
                val tierVal = tierConfig.paramOverrides["pass:${passNode.id}.$k"]
                    ?: tierConfig.paramOverrides["${passNode.id}.$k"]
                    ?: tierConfig.paramOverrides[k]
                if (tierVal != null) {
                    resolved = tierVal
                }
            }
            if (slots != null) {
                resolved = slots.getDynamicParam("pass:${passNode.id}.$k", resolved)
            }
            out[k] = resolved
        }

        // Also include any dynamically bound `pass:<id>.<param>` slots not in static `passNode.params`
        if (slots != null && slots.dynamicParamCount > 0) {
            val prefix = "pass:${passNode.id}."
            for ((fullKey, slotIdx) in slots.dynamicParamIndexMap) {
                if (fullKey.startsWith(prefix)) {
                    val paramName = fullKey.substring(prefix.length)
                    out[paramName] = slots.dynamicParams[slotIdx]
                }
            }
        }
        return out
    }

    companion object {
        const val PASS_ID_CAMERA_FRAME = "camera_frame"
        const val PASS_ID_CAMERA_PIXEL = "camera_pixel_passes"
        const val PASS_ID_CLEAR_DEPTH = "clear_depth"
        const val PASS_ID_OCCLUDERS = "occluders"
        const val PASS_ID_OPAQUE = "opaque_nodes"
        const val PASS_ID_TRANSPARENT = "transparent_nodes"
        const val PASS_ID_POST = "post_nodes"
        const val PASS_ID_OVERLAY = "overlay_nodes"
        const val PASS_ID_FINAL_OUTPUT = "final_output"

        private val CANONICAL_9_STAGE_PASSES: List<PassNode> = listOf(
            PassNode(
                id = PASS_ID_CAMERA_FRAME,
                stageKind = PassStageKind.CAMERA_FRAME,
                templateId = "camera.input@1",
                reads = listOf(RenderTargetRef.TARGET_CAMERA),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "none"
            ),
            PassNode(
                id = PASS_ID_CAMERA_PIXEL,
                stageKind = PassStageKind.CAMERA_PIXEL_PASSES,
                templateId = "camera.pixelPasses@1",
                reads = listOf(RenderTargetRef.TARGET_CAMERA, "${RenderTargetRef.PREFIX_MASK}regions"),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "alpha"
            ),
            PassNode(
                id = PASS_ID_CLEAR_DEPTH,
                stageKind = PassStageKind.CLEAR_DEPTH,
                templateId = "scene.clearDepth@1",
                reads = emptyList(),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "none"
            ),
            PassNode(
                id = PASS_ID_OCCLUDERS,
                stageKind = PassStageKind.OCCLUDERS,
                templateId = "scene.occluders@1",
                reads = listOf(RenderTargetRef.TARGET_SCENE),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "none"
            ),
            PassNode(
                id = PASS_ID_OPAQUE,
                stageKind = PassStageKind.OPAQUE_NODES,
                templateId = "scene.opaque@1",
                reads = listOf(RenderTargetRef.TARGET_SCENE),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "none"
            ),
            PassNode(
                id = PASS_ID_TRANSPARENT,
                stageKind = PassStageKind.TRANSPARENT_NODES,
                templateId = "scene.transparent@1",
                reads = listOf(RenderTargetRef.TARGET_SCENE),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "alpha"
            ),
            PassNode(
                id = PASS_ID_POST,
                stageKind = PassStageKind.POST_NODES,
                templateId = "post.chain@1",
                reads = listOf(RenderTargetRef.TARGET_SCENE),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "none"
            ),
            PassNode(
                id = PASS_ID_OVERLAY,
                stageKind = PassStageKind.OVERLAY_NODES,
                templateId = "scene.overlays@1",
                reads = listOf(RenderTargetRef.TARGET_SCENE),
                writes = RenderTargetRef.TARGET_SCENE,
                scaleFactor = 1.0f,
                blendMode = "alpha"
            ),
            PassNode(
                id = PASS_ID_FINAL_OUTPUT,
                stageKind = PassStageKind.FINAL_OUTPUT,
                templateId = "scene.present@1",
                reads = listOf(RenderTargetRef.TARGET_SCENE),
                writes = RenderTargetRef.TARGET_OUTPUT,
                scaleFactor = 1.0f,
                blendMode = "none"
            )
        )

        /**
         * Returns the canonical 9-stage [PassGraph] matching `RenderGraph.RenderPass` 1-for-1.
         */
        fun defaultCanonical9StageGraph(): PassGraph {
            return PassGraph(passes = CANONICAL_9_STAGE_PASSES, isCustomManifestGraph = false)
        }

        /**
         * Compiles a [PassGraph] from an optional manifest `"passes"` JSONArray.
         * If `passesArr` is null or empty, returns [defaultCanonical9StageGraph].
         */
        fun compileFromJson(
            passesArr: JSONArray?,
            resolvedManifestParams: Map<String, Any> = emptyMap()
        ): PassGraph {
            if (passesArr == null || passesArr.length() == 0) {
                return defaultCanonical9StageGraph()
            }

            val compiledNodes = ArrayList<PassNode>(passesArr.length())
            for (i in 0 until passesArr.length()) {
                val obj = passesArr.optJSONObject(i) ?: continue
                val id = obj.optString("id", "pass_$i")
                val stageStr = obj.optString("stage", obj.optString("kind", ""))
                val templateId = obj.optString("template", obj.optString("op", stageStr.ifEmpty { "custom.pass@1" }))
                val stageKind = parseStageKind(id, stageStr, templateId)

                val readsList = ArrayList<String>()
                val readsArr = obj.optJSONArray("reads")
                if (readsArr != null) {
                    for (r in 0 until readsArr.length()) {
                        val ref = readsArr.optString(r, "").trim()
                        if (ref.isNotEmpty()) readsList.add(ref)
                    }
                } else {
                    val singleRead = obj.optString("read", obj.optString("input", "")).trim()
                    if (singleRead.isNotEmpty()) readsList.add(singleRead)
                }

                val writes = obj.optString(
                    "writes",
                    obj.optString("write", obj.optString("output", RenderTargetRef.TARGET_SCENE))
                ).trim().ifEmpty { RenderTargetRef.TARGET_SCENE }

                val rawScaleObj = obj.opt("scale") ?: obj.opt("scaleFactor")
                val scaleFactor = when (rawScaleObj) {
                    is Number -> rawScaleObj.toFloat()
                    is String -> when (rawScaleObj.trim().uppercase()) {
                        "HALF", "0.5" -> 0.5f
                        "QUARTER", "0.25" -> 0.25f
                        "FULL", "1.0", "1" -> 1.0f
                        else -> rawScaleObj.toFloatOrNull() ?: 1.0f
                    }
                    else -> 1.0f
                }.coerceIn(0.125f, 1.0f)
                val blendMode = obj.optString("blend", obj.optString("blendMode", "alpha"))
                val enabled = obj.optBoolean("enabled", true)

                val paramsMap = LinkedHashMap<String, Float>()
                val paramsObj = obj.optJSONObject("params")
                var resolvedAssetId: String? = obj.optString(
                    "asset",
                    obj.optString("lut", obj.optString("map", ""))
                ).trim().ifEmpty { null }
                val isRecolorTemplate = templateId == "recolor@1" ||
                    templateId == "pass.recolor@1" ||
                    templateId == "pixel.recolor@1" ||
                    templateId == "eyeColor@1"
                if (stageKind == PassStageKind.CUSTOM_PASS) {
                    val cleanEffectName = templateId.removePrefix("pass.").removePrefix("post.")
                    paramsMap.putAll(PostRenderer.getDefaultParamsForEffect(cleanEffectName))
                } else if (stageKind == PassStageKind.CAMERA_PIXEL_PASSES && isRecolorTemplate) {
                    paramsMap.putAll(
                        RecolorPass.resolveRecolorParams(
                            nodeId = id,
                            paramsJson = paramsObj,
                            materialColor = null,
                            materialOpacity = null,
                            resolvedManifestParams = resolvedManifestParams
                        )
                    )
                }
                if (paramsObj != null) {
                    for (pKey in paramsObj.keys()) {
                        if (pKey == "asset" || pKey == "lut" || pKey == "map") {
                            val rawAsset = paramsObj.optString(pKey, "").trim()
                            if (rawAsset.isNotEmpty() && resolvedAssetId == null) {
                                resolvedAssetId = rawAsset
                            }
                            continue
                        }
                        if (isRecolorTemplate && (paramsMap.containsKey(pKey) ||
                                pKey == "color" || pKey == "targetColor" ||
                                pKey == "luminancePreserve" || pKey == "opacity")
                        ) {
                            continue
                        }
                        val rawVal = paramsObj.opt(pKey)
                        val floatVal = resolveParamFloat(rawVal, resolvedManifestParams)
                        if (floatVal != null) {
                            paramsMap[pKey] = floatVal
                        }
                    }
                }

                compiledNodes.add(
                    PassNode(
                        id = id,
                        stageKind = stageKind,
                        templateId = templateId,
                        reads = readsList,
                        writes = writes,
                        scaleFactor = scaleFactor,
                        blendMode = blendMode,
                        params = paramsMap,
                        enabledByDefault = enabled,
                        assetId = resolvedAssetId
                    )
                )
            }

            if (compiledNodes.isEmpty()) {
                return defaultCanonical9StageGraph()
            }
            return PassGraph(passes = compiledNodes, isCustomManifestGraph = true)
        }

        private fun parseStageKind(id: String, stageStr: String, templateId: String): PassStageKind {
            val token = (stageStr.ifEmpty { id }).lowercase()
            return when (token) {
                "camera_frame", "cameraframe", "camera" -> PassStageKind.CAMERA_FRAME
                "camera_pixel_passes", "camerapixelpasses", "camera_pixel", "facemesh", "eyecolor", "recolor" -> PassStageKind.CAMERA_PIXEL_PASSES
                "clear_depth", "cleardepth" -> PassStageKind.CLEAR_DEPTH
                "occluders", "occluder" -> PassStageKind.OCCLUDERS
                "opaque_nodes", "opaquenodes", "opaque" -> PassStageKind.OPAQUE_NODES
                "transparent_nodes", "transparentnodes", "transparent", "alphalenses" -> PassStageKind.TRANSPARENT_NODES
                "post_nodes", "postnodes", "post" -> PassStageKind.POST_NODES
                "overlay_nodes", "overlaynodes", "overlay", "overlays" -> PassStageKind.OVERLAY_NODES
                "final_output", "finaloutput", "present", "output" -> PassStageKind.FINAL_OUTPUT
                else -> when {
                    templateId.startsWith("pixel.") || templateId == "recolor@1" || templateId == "pass.recolor@1" || templateId == "eyeColor@1" -> PassStageKind.CAMERA_PIXEL_PASSES
                    templateId == "post.chain@1" -> PassStageKind.POST_NODES
                    templateId.startsWith("overlay") -> PassStageKind.OVERLAY_NODES
                    else -> PassStageKind.CUSTOM_PASS
                }
            }
        }

        private fun resolveParamFloat(rawVal: Any?, resolvedManifestParams: Map<String, Any>): Float? {
            if (rawVal is Number) return rawVal.toFloat()
            if (rawVal is String && rawVal.startsWith("$")) {
                val pName = rawVal.substring(1).removePrefix("param.")
                return (resolvedManifestParams[pName] as? Number)?.toFloat()
            }
            return null
        }
    }
}
