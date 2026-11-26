package com.example.engine.pkg

import com.example.engine.scene.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Compiler turns a validated xfilter/3 manifest into an immutable CompiledEffect (README Section 6 & 21 Step 3).
 * Resolves all selectors, paths, expressions, and parameters into fixed slot arrays.
 * Zero string lookups and zero JSON parsing during the render frame loop!
 */
object Compiler {

    fun compile(
        manifestJson: JSONObject,
        packageDir: File,
        paramOverrides: Map<String, Any> = emptyMap()
    ): CompiledEffect {
        // 1. Run legacy adapter to canonical xfilter/3 if needed
        val adapted = LegacyAdapters.adapt(manifestJson)

        val id = adapted.optString("id", "effect")
        val name = adapted.optString("name", "Filter")
        val schema = adapted.optString("schema", "xfilter/3")

        // 2. Feature negotiation against CapabilityRegistry
        val requires = adapted.optJSONObject("requires")
        if (requires != null) {
            val featuresArr = requires.optJSONArray("features")
            if (featuresArr != null) {
                val reqFeatures = ArrayList<String>()
                for (i in 0 until featuresArr.length()) {
                    reqFeatures.add(featuresArr.getString(i))
                }
                val (isCompatible, missing) = CapabilityRegistry.negotiate(reqFeatures)
                if (!isCompatible) {
                    throw IllegalArgumentException("Package requires unsupported engine features: ${missing.joinToString(", ")}")
                }
            }
        }

        // 3. Resolve Parameters with min/max bounds and overrides
        val resolvedParams = HashMap<String, Any>()
        val paramsObj = adapted.optJSONObject("params")
        if (paramsObj != null) {
            for (key in paramsObj.keys()) {
                val p = paramsObj.getJSONObject(key)
                val type = p.optString("type", "float")
                val defaultVal = p.opt("default")
                val overrideVal = paramOverrides[key] ?: defaultVal

                when (type) {
                    "vec3" -> {
                        val arr = overrideVal as? JSONArray ?: defaultVal as? JSONArray
                        if (arr != null && arr.length() == 3) {
                            val v0 = arr.getDouble(0).toFloat()
                            val v1 = arr.getDouble(1).toFloat()
                            val v2 = arr.getDouble(2).toFloat()
                            resolvedParams[key] = floatArrayOf(v0, v1, v2)
                        }
                    }
                    else -> {
                        val num = (overrideVal as? Number)?.toFloat()
                            ?: (defaultVal as? Number)?.toFloat()
                            ?: 0f
                        resolvedParams[key] = num
                    }
                }
            }
        }

        // 4. Resolve Assets
        val assetsMap = HashMap<String, CompiledAsset>()
        val assetsArr = adapted.optJSONArray("assets") ?: JSONArray()
        for (i in 0 until assetsArr.length()) {
            val a = assetsArr.getJSONObject(i)
            val assetId = a.getString("id")
            val path = a.optString("path", a.optString("uri", ""))
            val localFile = File(packageDir, path)
            assetsMap[assetId] = CompiledAsset(
                id = assetId,
                type = a.optString("type", "image"),
                localPath = localFile.absolutePath,
                sha256 = a.optString("sha256", ""),
                bytes = a.optLong("bytes", localFile.length())
            )
        }

        // 5. Resolve Materials
        val materialsMap = HashMap<String, CompiledMaterial>()
        val matObj = adapted.optJSONObject("materials") ?: JSONObject()
        for (mKey in matObj.keys()) {
            val m = matObj.getJSONObject(mKey)
            val template = m.optString("template", "unlit@1")
            val params = m.optJSONObject("params") ?: m
            val colArr = params.optJSONArray("color") ?: m.optJSONArray("color")
            val col = if (colArr != null && colArr.length() >= 3) {
                floatArrayOf(
                    colArr.getDouble(0).toFloat(),
                    colArr.getDouble(1).toFloat(),
                    colArr.getDouble(2).toFloat()
                )
            } else {
                floatArrayOf(1f, 1f, 1f)
            }
            val rawUvLayout = params.optString("uvLayout", UvRegistry.TOPOLOGY_ID).ifBlank { UvRegistry.TOPOLOGY_ID }
            if (template == "faceTexture@1" && !UvRegistry.isSupportedUvLayoutId(rawUvLayout)) {
                throw IllegalArgumentException("Material '$mKey' specifies unsupported or legacy UV layout '$rawUvLayout'")
            }
            val mapId = params.optString("map", "").takeIf { it.isNotBlank() }
            val maskId = params.optString("mask", "").takeIf { it.isNotBlank() }
            val normalMapId = params.optString("normalMap", "").takeIf { it.isNotBlank() }
            val mrMapId = params.optString("metallicRoughnessMap", "").takeIf { it.isNotBlank() }
            val occMapId = params.optString("occlusionMap", "").takeIf { it.isNotBlank() }
            val emissiveMapId = params.optString("emissiveMap", params.optString("emissiveTexture", "")).takeIf { it.isNotBlank() }
            val hasExplicitMetallic = params.has("metallic") || params.has("metallicFactor")
            val hasExplicitRoughness = params.has("roughness") || params.has("roughnessFactor")
            val hasExplicitEmissive = params.has("emissive") || params.has("emissiveFactor")
            val metallic = params.optDouble("metallic", params.optDouble("metallicFactor", 0.0)).toFloat().coerceIn(0f, 1f)
            val roughness = params.optDouble("roughness", params.optDouble("roughnessFactor", 0.45)).toFloat().coerceIn(0.04f, 1f)
            val normalScale = params.optDouble("normalScale", 1.0).toFloat().coerceIn(0f, 4f)
            val occlusionStrength = params.optDouble("occlusionStrength", 1.0).toFloat().coerceIn(0f, 1f)
            val fresnelSheen = params.optDouble("fresnelSheen", params.optDouble("sheen", 0.15)).toFloat().coerceIn(0f, 1f)
            val emissiveArr = params.optJSONArray("emissive") ?: params.optJSONArray("emissiveFactor")
            val emissive = if (emissiveArr != null && emissiveArr.length() >= 3) {
                floatArrayOf(
                    emissiveArr.getDouble(0).toFloat(),
                    emissiveArr.getDouble(1).toFloat(),
                    emissiveArr.getDouble(2).toFloat()
                )
            } else {
                floatArrayOf(0f, 0f, 0f)
            }
            val isFaceTex = (template == "faceTexture@1")
            materialsMap[mKey] = CompiledMaterial(
                id = mKey,
                template = template,
                color = col,
                mapAssetId = mapId,
                maskAssetId = maskId,
                normalMapAssetId = normalMapId,
                metallicRoughnessMapAssetId = mrMapId,
                occlusionMapAssetId = occMapId,
                emissiveMapAssetId = emissiveMapId,
                opacity = params.optDouble("opacity", 1.0).toFloat().coerceIn(0f, 1f),
                metallic = metallic,
                roughness = roughness,
                normalScale = normalScale,
                occlusionStrength = occlusionStrength,
                fresnelSheen = fresnelSheen,
                emissive = emissive,
                hasExplicitMetallic = hasExplicitMetallic,
                hasExplicitRoughness = hasExplicitRoughness,
                hasExplicitEmissive = hasExplicitEmissive,
                uvLayout = rawUvLayout,
                protectEyes = params.optBoolean("protectEyes", true),
                protectLips = params.optBoolean("protectLips", true),
                protectMouthInterior = params.optBoolean("protectMouthInterior", true),
                blend = m.optString("blend", params.optString("blend", "alpha")),
                doubleSided = m.optBoolean("doubleSided", isFaceTex),
                depthTest = m.optBoolean("depthTest", true),
                depthWrite = m.optBoolean("depthWrite", !isFaceTex),
                cull = m.optBoolean("cull", !isFaceTex)
            )
        }

        // 6. Resolve Custom Anchors
        val customAnchorsMap = HashMap<String, AnchorResolver.CustomAnchorDef>()
        val anchorsObj = adapted.optJSONObject("anchors")
        if (anchorsObj != null) {
            for (aKey in anchorsObj.keys()) {
                val aDef = anchorsObj.optJSONObject(aKey) ?: continue
                val lmArr = aDef.optJSONArray("landmarks") ?: JSONArray()
                val lmWeights = ArrayList<AnchorResolver.LandmarkWeight>()
                for (j in 0 until lmArr.length()) {
                    val item = lmArr.optJSONObject(j)
                    if (item != null) {
                        val idVal = item.getInt("id")
                        if (idVal !in 0 until 478) {
                            throw IllegalArgumentException("Custom anchor '$aKey' landmark ID $idVal out of range (0..477)")
                        }
                        val wVal = item.optDouble("weight", item.optDouble("w", 1.0)).toFloat()
                        lmWeights.add(AnchorResolver.LandmarkWeight(idVal, wVal))
                    } else {
                        val idVal = lmArr.optInt(j, -1)
                        if (idVal in 0 until 478) {
                            lmWeights.add(AnchorResolver.LandmarkWeight(idVal, 1.0f))
                        } else if (idVal != -1) {
                            throw IllegalArgumentException("Custom anchor '$aKey' landmark ID $idVal out of range (0..477)")
                        }
                    }
                }
                val offArr = aDef.optJSONArray("offsetFW")
                val off = if (offArr != null && offArr.length() >= 3) {
                    floatArrayOf(
                        offArr.getDouble(0).toFloat(),
                        offArr.getDouble(1).toFloat(),
                        offArr.getDouble(2).toFloat()
                    )
                } else if (aKey == "head_center" && lmWeights.isEmpty()) {
                    floatArrayOf(
                        AnchorResolver.DEFAULT_HEAD_CENTER_X_FW,
                        AnchorResolver.DEFAULT_HEAD_CENTER_Y_FW,
                        AnchorResolver.DEFAULT_HEAD_CENTER_Z_FW
                    )
                } else {
                    floatArrayOf(0f, 0f, 0f)
                }
                customAnchorsMap[aKey] = AnchorResolver.CustomAnchorDef(lmWeights, off)
            }
        }
        val anchorResolver = AnchorResolver(customAnchorsMap)

        // 7. Resolve Nodes
        val nodesList = ArrayList<CompiledNode>()
        val nodeNameToIndex = HashMap<String, Int>()
        val tagToIndices = HashMap<String, ArrayList<Int>>()
        val manifestParamBindings = HashMap<String, MutableList<Pair<String, String>>>()

        val nodesArr = adapted.optJSONArray("nodes") ?: JSONArray()
        for (i in 0 until nodesArr.length()) {
            val n = nodesArr.getJSONObject(i)
            val nodeId = n.getString("id")
            nodeNameToIndex[nodeId] = i

            val tags = ArrayList<String>()
            val tagsArr = n.optJSONArray("tags")
            if (tagsArr != null) {
                for (t in 0 until tagsArr.length()) {
                    val tag = tagsArr.getString(t)
                    tags.add(tag)
                    tagToIndices.getOrPut(tag) { ArrayList() }.add(i)
                }
            }

            // Transform
            val transform = n.optJSONObject("transform")
            var pos = floatArrayOf(0f, 0f, 0f)
            var rot = floatArrayOf(0f, 0f, 0f, 1f)
            var scale = floatArrayOf(1f, 1f, 1f)
            var rotFollow = "full"

            if (transform != null) {
                val posVal = transform.opt("position")
                if (posVal is String && posVal.startsWith("$")) {
                    val pName = posVal.substring(1)
                    val paramVal = resolvedParams[pName]
                    if (paramVal is FloatArray && paramVal.size >= 3) {
                        pos = paramVal.clone()
                    }
                } else if (posVal is JSONArray && posVal.length() >= 3) {
                    pos[0] = posVal.getDouble(0).toFloat()
                    pos[1] = posVal.getDouble(1).toFloat()
                    pos[2] = posVal.getDouble(2).toFloat()
                }

                val scVal = transform.opt("scale")
                if (scVal is Number) {
                    val s = scVal.toFloat()
                    scale[0] = s; scale[1] = s; scale[2] = s
                } else if (scVal is JSONArray && scVal.length() >= 3) {
                    scale[0] = scVal.getDouble(0).toFloat()
                    scale[1] = scVal.getDouble(1).toFloat()
                    scale[2] = scVal.getDouble(2).toFloat()
                }

                rotFollow = transform.optString("rotationFollow", "full")
            }

            val sizeObj = n.optJSONObject("size")
            val wFw = sizeObj?.optDouble("widthFW", 0.5)?.toFloat() ?: 0.5f
            val hFw = sizeObj?.optDouble("heightFW", 0.5)?.toFloat() ?: 0.5f
            val pivotArr = n.optJSONArray("pivot") ?: sizeObj?.optJSONArray("pivot")
            val spritePivot = if (pivotArr != null && pivotArr.length() >= 2) {
                floatArrayOf(
                    pivotArr.optDouble(0, 0.5).toFloat().coerceIn(0f, 1f),
                    pivotArr.optDouble(1, 0.5).toFloat().coerceIn(0f, 1f)
                )
            } else {
                floatArrayOf(0.5f, 0.5f)
            }

            val occluderShape = n.optString("shape", null)
            val pushBack = n.optDouble("pushBackFW", 0.015).toFloat()
            var radii: FloatArray? = null
            val radiiArr = n.optJSONArray("radiiFW")
            if (radiiArr != null && radiiArr.length() >= 3) {
                radii = floatArrayOf(
                    radiiArr.getDouble(0).toFloat(),
                    radiiArr.getDouble(1).toFloat(),
                    radiiArr.getDouble(2).toFloat()
                )
            }
            var segments: IntArray? = null
            val segArr = n.optJSONArray("segments")
            if (segArr != null && segArr.length() >= 2) {
                segments = intArrayOf(segArr.getInt(0), segArr.getInt(1))
            }

            val nodeType = n.optString("type", "group")
            val matId = n.optString("material", null)?.takeIf { it.isNotBlank() }
            val faceMeshRegion = n.optString("region", n.optString("mask", null))?.takeIf { it.isNotBlank() }

            if (nodeType == "faceMesh") {
                val mat = matId?.let { materialsMap[it] }
                    ?: throw IllegalArgumentException("faceMesh node '$nodeId' missing valid material reference")
                if (mat.template != "faceTexture@1") {
                    throw IllegalArgumentException(
                        "faceMesh node '$nodeId' requires material template 'faceTexture@1', got '${mat.template}'"
                    )
                }
                if (mat.mapAssetId.isNullOrBlank()) {
                    throw IllegalArgumentException(
                        "faceMesh node '$nodeId' material '${mat.id}' missing required 'map' texture parameter"
                    )
                }
                if (faceMeshRegion != null && mat.maskAssetId == null && assetsMap.containsKey(faceMeshRegion)) {
                    materialsMap[mat.id] = mat.copy(maskAssetId = faceMeshRegion)
                }
            }

            var postEffect: String? = null
            var postParams: Map<String, Float> = emptyMap()
            val nodeParamsObj = n.optJSONObject("params")
            if (nodeType == "post") {
                val postVal = com.example.engine.render.PostRenderer.validatePostNode(n, adapted.optJSONObject("params"))
                if (!postVal.isValid) {
                    throw IllegalArgumentException(postVal.errorReason ?: "Invalid post node '$nodeId'")
                }
                postEffect = n.getString("effect")
                postParams = com.example.engine.render.PostRenderer.resolvePostParams(
                    nodeId = nodeId,
                    effectName = postEffect,
                    paramsJson = nodeParamsObj,
                    resolvedManifestParams = resolvedParams
                )
            } else if (nodeType == "eyeColor" || nodeType == "recolor") {
                val mat = matId?.let { materialsMap[it] }
                postEffect = "recolor@1"
                postParams = com.example.engine.render.RecolorPass.resolveRecolorParams(
                    nodeId = nodeId,
                    paramsJson = nodeParamsObj ?: n,
                    materialColor = mat?.color,
                    materialOpacity = mat?.opacity,
                    resolvedManifestParams = resolvedParams
                )
            }
            if ((nodeType == "post" || nodeType == "eyeColor" || nodeType == "recolor") && nodeParamsObj != null) {
                for (pKey in nodeParamsObj.keys()) {
                    val rawVal = nodeParamsObj.opt(pKey)
                    if (rawVal is String && rawVal.startsWith("$") && rawVal.length > 1) {
                        val manifestParamName = rawVal.substring(1).removePrefix("param.")
                        val mappedKey = if (nodeType == "eyeColor" || nodeType == "recolor") {
                            when (pKey) {
                                "luminancePreserve" -> "preserveLuminance"
                                "opacity" -> "strength"
                                else -> pKey
                            }
                        } else {
                            pKey
                        }
                        manifestParamBindings
                            .getOrPut(manifestParamName) { ArrayList() }
                            .add(Pair(nodeId, mappedKey))
                    }
                }
            }

            val postAssetId = if (nodeType == "post") {
                (n.optString("asset", null)?.takeIf { it.isNotBlank() }
                    ?: n.optString("lut", null)?.takeIf { it.isNotBlank() }
                    ?: n.optString("map", null)?.takeIf { it.isNotBlank() }
                    ?: nodeParamsObj?.optString("lut", null)?.takeIf { it.isNotBlank() }
                    ?: nodeParamsObj?.optString("asset", null)?.takeIf { it.isNotBlank() })
            } else {
                null
            }
            val flipbookObj = n.optJSONObject("flipbook") ?: n.optJSONObject("spriteSheet")
            val spriteRows = (flipbookObj?.optInt("rows", n.optInt("rows", 1)) ?: n.optInt("rows", 1)).coerceIn(1, 64)
            val spriteCols = (flipbookObj?.optInt("cols", n.optInt("cols", 1)) ?: n.optInt("cols", 1)).coerceIn(1, 64)
            val spriteFps = (flipbookObj?.optDouble("fps", n.optDouble("fps", 0.0)) ?: n.optDouble("fps", 0.0)).toFloat().coerceIn(0f, 240f)
            val spriteLoop = flipbookObj?.optBoolean("loop", n.optBoolean("loop", true)) ?: n.optBoolean("loop", true)
            val maxGridFrames = (spriteRows * spriteCols).coerceAtLeast(1)
            val rawFrameCount = flipbookObj?.optInt("frameCount", n.optInt("frameCount", 0)) ?: n.optInt("frameCount", 0)
            val spriteFrameCount = if (rawFrameCount > 0) rawFrameCount.coerceIn(1, maxGridFrames) else 0
            val rawFrameIndex = when {
                flipbookObj?.has("frameIndex") == true -> flipbookObj.optInt("frameIndex", -1)
                n.has("frameIndex") -> n.optInt("frameIndex", -1)
                else -> -1
            }
            val spriteFrameIndex = if (rawFrameIndex >= 0) rawFrameIndex.coerceIn(0, maxGridFrames - 1) else -1
            val spriteStartFrame = (
                flipbookObj?.optInt("startFrame", n.optInt("startFrame", 0)) ?: n.optInt("startFrame", 0)
            ).coerceIn(0, maxGridFrames - 1)

            val particlesObj = n.optJSONObject("particles") ?: n.optJSONObject("sparkle") ?: n.optJSONObject("emitter")
            val particleCount = (
                particlesObj?.optInt("count", n.optInt("particleCount", 1)) ?: n.optInt("particleCount", 1)
            ).coerceIn(1, 32)
            val spreadArr = particlesObj?.optJSONArray("spread")
            val particleSpreadX = when {
                spreadArr != null && spreadArr.length() >= 1 -> spreadArr.optDouble(0, 0.35).toFloat()
                particlesObj != null -> particlesObj.optDouble("spreadX", particlesObj.optDouble("spread", 0.35)).toFloat()
                else -> 0.35f
            }.coerceIn(0f, 4f)
            val particleSpreadY = when {
                spreadArr != null && spreadArr.length() >= 2 -> spreadArr.optDouble(1, 0.35).toFloat()
                particlesObj != null -> particlesObj.optDouble("spreadY", particlesObj.optDouble("spread", 0.35)).toFloat()
                else -> 0.35f
            }.coerceIn(0f, 4f)
            val driftArr = particlesObj?.optJSONArray("drift") ?: particlesObj?.optJSONArray("velocity")
            val particleDriftX = when {
                driftArr != null && driftArr.length() >= 1 -> driftArr.optDouble(0, 0.0).toFloat()
                particlesObj != null -> particlesObj.optDouble("driftX", 0.0).toFloat()
                else -> 0f
            }.coerceIn(-5f, 5f)
            val particleDriftY = when {
                driftArr != null && driftArr.length() >= 2 -> driftArr.optDouble(1, 0.0).toFloat()
                particlesObj != null -> particlesObj.optDouble("driftY", 0.0).toFloat()
                else -> 0f
            }.coerceIn(-5f, 5f)
            val particleTwinkleSpeed = (
                particlesObj?.optDouble("twinkleSpeed", particlesObj.optDouble("twinkleHz", 0.0))?.toFloat() ?: 0f
            ).coerceIn(0f, 30f)
            val particleScaleMin = (
                particlesObj?.optDouble("scaleMin", 0.65)?.toFloat() ?: 0.65f
            ).coerceIn(0.05f, 4f)
            val particleScaleMax = (
                particlesObj?.optDouble("scaleMax", 1.15)?.toFloat() ?: 1.15f
            ).coerceIn(particleScaleMin, 4f)
            val particleSeed = particlesObj?.optInt("seed", i * 97 + 13) ?: (i * 97 + 13)

            nodesList.add(
                CompiledNode(
                    index = i,
                    id = nodeId,
                    parentIndex = -1, // resolved below
                    tags = tags,
                    type = nodeType,
                    anchor = n.optString("anchor", null)?.takeIf { it.isNotBlank() },
                    basePosition = pos,
                    baseRotation = rot,
                    baseScale = scale,
                    rotationFollow = rotFollow,
                    materialId = matId,
                    baseVisible = n.optBoolean("visible", true),
                    baseOpacity = n.optDouble("opacity", 1.0).toFloat(),
                    renderOrder = n.optInt("renderOrder", i),
                    spriteAssetId = n.optString("asset", null)?.takeIf { it.isNotBlank() },
                    spriteWidthFW = wFw,
                    spriteHeightFW = hFw,
                    spritePivot = spritePivot,
                    modelAssetId = n.optString("asset", null)?.takeIf { it.isNotBlank() },
                    occluderShape = occluderShape,
                    pushBackFW = pushBack,
                    radiiFW = radii,
                    occluderSegments = segments,
                    faceMeshRegion = faceMeshRegion,
                    postEffect = postEffect,
                    postParams = postParams,
                    postAssetId = postAssetId,
                    overlayFit = n.optString("fit", "cover").lowercase().let {
                        if (it == "contain") "contain" else "cover"
                    },
                    overlaySafeArea = n.optBoolean("safeArea", false),
                    spriteRows = spriteRows,
                    spriteCols = spriteCols,
                    spriteFps = spriteFps,
                    spriteLoop = spriteLoop,
                    spriteFrameCount = spriteFrameCount,
                    spriteFrameIndex = spriteFrameIndex,
                    spriteStartFrame = spriteStartFrame,
                    particleCount = particleCount,
                    particleSpreadX = particleSpreadX,
                    particleSpreadY = particleSpreadY,
                    particleDriftX = particleDriftX,
                    particleDriftY = particleDriftY,
                    particleTwinkleSpeed = particleTwinkleSpeed,
                    particleScaleMin = particleScaleMin,
                    particleScaleMax = particleScaleMax,
                    particleSeed = particleSeed
                )
            )
        }

        // Resolve parent indices
        for (i in 0 until nodesArr.length()) {
            val pName = nodesArr.getJSONObject(i).optString("parent", "")
            if (pName.isNotEmpty()) {
                val pIdx = nodeNameToIndex[pName] ?: -1
                nodesList[i] = nodesList[i].copy(parentIndex = pIdx)
            }
        }

        // 8. Resolve Signal Slots, Cycle Detection & Expressions
        val slotMap = HashMap<String, Int>()
        // Built-ins: 0..7
        slotMap["face.present"] = 0
        slotMap["face.fade"] = 1
        slotMap["face.pose.yaw"] = 2
        slotMap["face.pose.pitch"] = 3
        slotMap["face.pose.roll"] = 4
        slotMap["time.render"] = 5
        slotMap["time.sinceAcquired"] = 6
        slotMap["input.tap"] = 7

        var nextSlot = 8
        val requestedBlendshapes = HashSet<String>()

        // Blendshape slots
        for (bs in SignalSourceRegistry.STANDARD_BLENDSHAPES) {
            slotMap["face.blend.$bs"] = nextSlot++
        }

        // Derived signals & Cycle Detection
        val derivedSignalsList = ArrayList<Pair<Int, ExpressionVM.Program>>()
        val vm = ExpressionVM()
        val signalsObj = adapted.optJSONObject("signals")
        if (signalsObj != null) {
            val rawSignals = HashMap<String, String>()
            for (sigKey in signalsObj.keys()) {
                rawSignals[sigKey] = signalsObj.getString(sigKey)
            }

            // Topological sort and cycle check
            val orderedSignalKeys = vm.getTopologicalOrder(rawSignals)

            for (sigKey in orderedSignalKeys) {
                val expr = rawSignals[sigKey]!!
                val fullKey = "sig.$sigKey"
                val slotIdx = nextSlot++
                slotMap[fullKey] = slotIdx

                // Scan for blendshape dependencies
                for (dep in vm.extractSignalDependencies(expr)) {
                    if (dep.startsWith("face.blend.")) {
                        requestedBlendshapes.add(dep.removePrefix("face.blend."))
                    }
                }

                val prog = vm.compile(expr) { token ->
                    if (token.startsWith("face.blend.")) {
                        requestedBlendshapes.add(token.removePrefix("face.blend."))
                    }
                    slotMap.getOrPut(token) { nextSlot++ }
                }
                derivedSignalsList.add(Pair(slotIdx, prog))
            }
        }
        val signalBus = SignalBus(nextSlot + 32, derivedSignalsList, slotMap)

        // 8b. Compile Data-Driven PassGraph, Composable Masks & Initialize PropertySlots (Flow 2 — Chunk G0 & Flow 3 — Chunk R1)
        val passGraph = com.example.engine.render.PassGraph.compileFromJson(
            passesArr = adapted.optJSONArray("passes") ?: adapted.optJSONArray("passGraph"),
            resolvedManifestParams = resolvedParams
        )
        val maskDefs = com.example.engine.render.RegionMaskPass.compileMasksFromJson(
            masksObj = adapted.optJSONObject("masks")
        )
        val propertySlots = PropertySlots(nodesList.size)
        val staticParamDefaults = HashMap<String, Float>()
        for (pNode in passGraph.passes) {
            for ((pKey, pVal) in pNode.params) {
                staticParamDefaults["pass:${pNode.id}.$pKey"] = pVal
            }
        }
        for ((matId, mat) in materialsMap) {
            staticParamDefaults["mat:$matId.opacity"] = mat.opacity
        }
        for (maskId in maskDefs.keys) {
            staticParamDefaults["mask:$maskId.weight"] = 1f
        }
        for (node in nodesList) {
            if (node.type == "post" || node.type == "eyeColor" || node.type == "recolor") {
                for ((pKey, pVal) in node.postParams) {
                    staticParamDefaults["pass:${node.id}.$pKey"] = pVal
                    staticParamDefaults["post:${node.id}.$pKey"] = pVal
                }
            } else if (node.type == "sprite" || node.type == "overlay") {
                if (node.spriteFrameIndex >= 0) {
                    staticParamDefaults["pass:${node.id}.frameIndex"] = node.spriteFrameIndex.toFloat()
                    staticParamDefaults["sprite:${node.id}.frameIndex"] = node.spriteFrameIndex.toFloat()
                }
            }
        }

        // 9. Resolve Behaviors
        val clipsList = ArrayList<BehaviorRunner.CompiledClip>()
        val mapsList = ArrayList<BehaviorRunner.CompiledMap>()
        val triggersList = ArrayList<BehaviorRunner.CompiledTrigger>()
        val springsList = ArrayList<BehaviorRunner.CompiledSpring>()

        val behaviorsArr = adapted.optJSONArray("behaviors") ?: JSONArray()
        for (bIdx in 0 until behaviorsArr.length()) {
            val b = behaviorsArr.getJSONObject(bIdx)
            val bId = b.getString("id")
            val bType = b.getString("type")

            when (bType) {
                "clip" -> {
                    val targetSel = b.optString("target", "")
                    val targetIdx = resolveSelectorToFirstNode(targetSel, nodeNameToIndex, tagToIndices)
                    clipsList.add(
                        BehaviorRunner.CompiledClip(
                            id = bId,
                            targetNodeIndex = targetIdx,
                            animationName = b.optString("animation", bId),
                            loop = b.optBoolean("loop", false),
                            startOn = b.optString("startOn", null),
                            speed = b.optDouble("speed", 1.0).toFloat(),
                            holdLast = b.optBoolean("holdLast", false),
                            durationSec = b.optDouble("durationSec", 1.0).toFloat().coerceAtLeast(0.05f)
                        )
                    )
                }
                "map" -> {
                    val inSig = b.getString("input")
                    if (inSig.startsWith("face.blend.")) {
                        requestedBlendshapes.add(inSig.removePrefix("face.blend."))
                    }
                    val slotIdx = slotMap.getOrPut(inSig) { nextSlot++ }
                    val inRange = b.optJSONArray("inputRange") ?: JSONArray()
                    val outRange = b.optJSONArray("outputRange") ?: JSONArray()
                    val writesArr = b.optJSONArray("writes") ?: JSONArray()
                    val writes = ArrayList<BehaviorRunner.PropertyWrite>()

                    for (w in 0 until writesArr.length()) {
                        val wObj = writesArr.getJSONObject(w)
                        val path = wObj.getString("path") // e.g. "tag:petals.scale", "pass:grain.strength", "mat:paint.opacity", "mask:iris.weight"
                        val modeStr = wObj.optString("mode", "multiply").uppercase()
                        val writeMode = when (modeStr) {
                            "SET" -> BehaviorRunner.WriteMode.SET
                            "ADD" -> BehaviorRunner.WriteMode.ADD
                            else -> BehaviorRunner.WriteMode.MULTIPLY
                        }
                        val (sel, prop) = parsePath(path)
                        when {
                            sel.startsWith("pass:") || sel.startsWith("post:") -> {
                                val passId = sel.substringAfter(":")
                                val canonicalKey = "pass:$passId.$prop"
                                val defaultVal = staticParamDefaults[canonicalKey] ?: 1f
                                val dynIdx = propertySlots.registerDynamicParam(canonicalKey, defaultVal)
                                writes.add(
                                    BehaviorRunner.PropertyWrite(
                                        nodeIndex = -1,
                                        propertyType = BehaviorRunner.PropertyType.PASS_UNIFORM,
                                        componentIndex = 0,
                                        mode = writeMode,
                                        dynamicParamSlotIndex = dynIdx
                                    )
                                )
                            }
                            sel.startsWith("mat:") || sel.startsWith("material:") -> {
                                val matId = sel.substringAfter(":")
                                val canonicalKey = "mat:$matId.$prop"
                                val defaultVal = staticParamDefaults[canonicalKey] ?: 1f
                                val dynIdx = propertySlots.registerDynamicParam(canonicalKey, defaultVal)
                                writes.add(
                                    BehaviorRunner.PropertyWrite(
                                        nodeIndex = -1,
                                        propertyType = BehaviorRunner.PropertyType.MATERIAL_PARAM,
                                        componentIndex = 0,
                                        mode = writeMode,
                                        dynamicParamSlotIndex = dynIdx
                                    )
                                )
                            }
                            sel.startsWith("mask:") -> {
                                val maskId = sel.substringAfter(":")
                                val canonicalKey = "mask:$maskId.$prop"
                                val defaultVal = staticParamDefaults[canonicalKey] ?: 1f
                                val dynIdx = propertySlots.registerDynamicParam(canonicalKey, defaultVal)
                                writes.add(
                                    BehaviorRunner.PropertyWrite(
                                        nodeIndex = -1,
                                        propertyType = BehaviorRunner.PropertyType.MASK_WEIGHT,
                                        componentIndex = 0,
                                        mode = writeMode,
                                        dynamicParamSlotIndex = dynIdx
                                    )
                                )
                            }
                            else -> {
                                val cleanSel = sel.removePrefix("sprite:").removePrefix("overlay:")
                                val targetIndices = resolveSelector(cleanSel, nodeNameToIndex, tagToIndices)
                                val propType = when (prop) {
                                    "scale" -> BehaviorRunner.PropertyType.SCALE
                                    "position", "positionX", "positionY", "positionZ" -> BehaviorRunner.PropertyType.POSITION
                                    "rotation" -> BehaviorRunner.PropertyType.ROTATION
                                    "opacity" -> BehaviorRunner.PropertyType.OPACITY
                                    "visible" -> BehaviorRunner.PropertyType.VISIBLE
                                    else -> null
                                }
                                if (propType != null) {
                                    for (tIdx in targetIndices) {
                                        when (prop) {
                                            "positionX" -> writes.add(BehaviorRunner.PropertyWrite(tIdx, propType, 0, writeMode))
                                            "positionY" -> writes.add(BehaviorRunner.PropertyWrite(tIdx, propType, 1, writeMode))
                                            "positionZ" -> writes.add(BehaviorRunner.PropertyWrite(tIdx, propType, 2, writeMode))
                                            else -> {
                                                writes.add(BehaviorRunner.PropertyWrite(tIdx, propType, 0, writeMode))
                                                if (propType == BehaviorRunner.PropertyType.SCALE || propType == BehaviorRunner.PropertyType.POSITION) {
                                                    writes.add(BehaviorRunner.PropertyWrite(tIdx, propType, 1, writeMode))
                                                    writes.add(BehaviorRunner.PropertyWrite(tIdx, propType, 2, writeMode))
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    // Custom parameter on a node (e.g. "node:colorGrade.warmth" or "sprite:sparkle.frameIndex") -> bind as PASS_UNIFORM
                                    val cleanId = cleanSel.removePrefix("node:")
                                    val canonicalKey = "pass:$cleanId.$prop"
                                    val defaultVal = staticParamDefaults[canonicalKey] ?: 0f
                                    val dynIdx = propertySlots.registerDynamicParam(canonicalKey, defaultVal)
                                    writes.add(
                                        BehaviorRunner.PropertyWrite(
                                            nodeIndex = -1,
                                            propertyType = BehaviorRunner.PropertyType.PASS_UNIFORM,
                                            componentIndex = 0,
                                            mode = writeMode,
                                            dynamicParamSlotIndex = dynIdx
                                        )
                                    )
                                }
                            }
                        }
                    }

                    mapsList.add(
                        BehaviorRunner.CompiledMap(
                            id = bId,
                            inputSlotIndex = slotIdx,
                            inMin = if (inRange.length() > 0) inRange.getDouble(0).toFloat() else 0f,
                            inMax = if (inRange.length() > 1) inRange.getDouble(1).toFloat() else 1f,
                            outMin = if (outRange.length() > 0) outRange.getDouble(0).toFloat() else 0f,
                            outMax = if (outRange.length() > 1) outRange.getDouble(1).toFloat() else 1f,
                            easing = b.optString("easing", "smoothstep"),
                            clamp = b.optBoolean("clamp", true),
                            smoothingTau = b.optDouble("smoothingTau", 0.12).toFloat(),
                            initial = b.optDouble("initial", 1.0).toFloat(),
                            writes = writes
                        )
                    )
                }
                "trigger" -> {
                    val inSig = b.optString("input", null)
                    val onEvent = b.optString("on", null)
                    if (inSig != null && inSig.startsWith("face.blend.")) {
                        requestedBlendshapes.add(inSig.removePrefix("face.blend."))
                    }
                    val slotIdx = inSig?.let { slotMap.getOrPut(it) { nextSlot++ } }
                    val actions = ArrayList<BehaviorRunner.TriggerAction>()
                    val fireArr = b.optJSONArray("onFire") ?: JSONArray()
                    for (f in 0 until fireArr.length()) {
                        val fObj = fireArr.getJSONObject(f)
                        val actionType = fObj.getString("action")
                        val clipId = fObj.optString("clip", null)
                        var propWrite: BehaviorRunner.PropertyWrite? = null
                        val targetSel = fObj.optString("target", fObj.optString("path", ""))
                        if (targetSel.isNotEmpty()) {
                            val (sel, prop) = parsePath(targetSel)
                            when {
                                sel.startsWith("pass:") || sel.startsWith("post:") -> {
                                    val passId = sel.substringAfter(":")
                                    val dynIdx = propertySlots.registerDynamicParam("pass:$passId.$prop", 0f)
                                    propWrite = BehaviorRunner.PropertyWrite(
                                        nodeIndex = -1,
                                        propertyType = BehaviorRunner.PropertyType.PASS_UNIFORM,
                                        componentIndex = 0,
                                        mode = BehaviorRunner.WriteMode.SET,
                                        dynamicParamSlotIndex = dynIdx
                                    )
                                }
                                sel.startsWith("mat:") || sel.startsWith("material:") -> {
                                    val matId = sel.substringAfter(":")
                                    val dynIdx = propertySlots.registerDynamicParam("mat:$matId.$prop", 1f)
                                    propWrite = BehaviorRunner.PropertyWrite(
                                        nodeIndex = -1,
                                        propertyType = BehaviorRunner.PropertyType.MATERIAL_PARAM,
                                        componentIndex = 0,
                                        mode = BehaviorRunner.WriteMode.SET,
                                        dynamicParamSlotIndex = dynIdx
                                    )
                                }
                                sel.startsWith("mask:") -> {
                                    val maskId = sel.substringAfter(":")
                                    val dynIdx = propertySlots.registerDynamicParam("mask:$maskId.$prop", 1f)
                                    propWrite = BehaviorRunner.PropertyWrite(
                                        nodeIndex = -1,
                                        propertyType = BehaviorRunner.PropertyType.MASK_WEIGHT,
                                        componentIndex = 0,
                                        mode = BehaviorRunner.WriteMode.SET,
                                        dynamicParamSlotIndex = dynIdx
                                    )
                                }
                                else -> {
                                    val cleanSel = sel.removePrefix("sprite:").removePrefix("overlay:")
                                    val propType = when (prop) {
                                        "", "visible" -> BehaviorRunner.PropertyType.VISIBLE
                                        "opacity" -> BehaviorRunner.PropertyType.OPACITY
                                        "position" -> BehaviorRunner.PropertyType.POSITION
                                        "rotation" -> BehaviorRunner.PropertyType.ROTATION
                                        "scale" -> BehaviorRunner.PropertyType.SCALE
                                        else -> null
                                    }
                                    propWrite = if (propType != null) {
                                        val targetIdx = resolveSelectorToFirstNode(cleanSel, nodeNameToIndex, tagToIndices)
                                        BehaviorRunner.PropertyWrite(targetIdx, propType, 0, BehaviorRunner.WriteMode.SET)
                                    } else {
                                        val cleanId = cleanSel.removePrefix("node:")
                                        val canonicalKey = "pass:$cleanId.$prop"
                                        val defaultVal = staticParamDefaults[canonicalKey] ?: 0f
                                        val dynIdx = propertySlots.registerDynamicParam(canonicalKey, defaultVal)
                                        BehaviorRunner.PropertyWrite(
                                            nodeIndex = -1,
                                            propertyType = BehaviorRunner.PropertyType.PASS_UNIFORM,
                                            componentIndex = 0,
                                            mode = BehaviorRunner.WriteMode.SET,
                                            dynamicParamSlotIndex = dynIdx
                                        )
                                    }
                                }
                            }
                        }
                        val value = fObj.optDouble("value", 0.0).toFloat()
                        actions.add(
                            BehaviorRunner.TriggerAction(
                                action = actionType,
                                clipId = clipId,
                                propertyWrite = propWrite,
                                value = value
                            )
                        )
                    }

                    triggersList.add(
                        BehaviorRunner.CompiledTrigger(
                            id = bId,
                            inputSlotIndex = slotIdx,
                            onEvent = onEvent,
                            risingAbove = if (b.has("risingAbove")) b.getDouble("risingAbove").toFloat() else null,
                            fallingBelow = if (b.has("fallingBelow")) b.getDouble("fallingBelow").toFloat() else null,
                            cooldownMs = b.optLong("cooldownMs", 500L),
                            armedInitially = b.optBoolean("armedInitially", true),
                            onFire = actions
                        )
                    )
                }
                "spring" -> {
                    val targetSel = b.optString("target", "")
                    val targetIndices = resolveSelector(targetSel, nodeNameToIndex, tagToIndices)
                        .ifEmpty { listOf(0) }
                    for ((idxInGroup, targetIdx) in targetIndices.withIndex()) {
                        val springId = if (targetIndices.size == 1 || idxInGroup == 0) bId else "${bId}_$idxInGroup"
                        springsList.add(
                            BehaviorRunner.CompiledSpring(
                                id = springId,
                                targetNodeIndex = targetIdx,
                                mode = b.optString("mode", "rotationInertia"),
                                stiffness = b.optDouble("stiffness", 90.0).toFloat(),
                                dampingRatio = b.optDouble("dampingRatio", 0.3).toFloat(),
                                gain = b.optDouble("gain", 0.35).toFloat(),
                                maxAngleDeg = b.optDouble("maxAngleDeg", 8.0).toFloat(),
                                lowPassTau = b.optDouble("lowPassTau", 0.04).toFloat()
                            )
                        )
                    }
                }
            }
        }

        val behaviorRunner = BehaviorRunner(
            clips = clipsList,
            maps = mapsList,
            triggers = triggersList,
            springs = springsList,
            nodeCount = nodesList.size
        )

        // 10. Lighting (supports 0, 1, or up to MAX_LIGHTS=2 directional lights + ambient)
        var lighting: CompiledLighting? = null
        val lightObj = adapted.optJSONObject("lighting")
        if (lightObj != null) {
            lighting = compileLightingBlock(lightObj)
        }

        // 11. Tier Overrides (supports both JSONArray [{"id":"A",...}] and JSONObject {"A":{...},"B":{...}})
        val tierOverrides = HashMap<String, TierOverride>()
        fun parseTierEntry(tierId: String, t: JSONObject) {
            val showArr = t.optJSONArray("show") ?: t.optJSONArray("visibleNodes") ?: JSONArray()
            val hideArr = t.optJSONArray("hide") ?: t.optJSONArray("hideNodes") ?: JSONArray()
            val disArr = t.optJSONArray("disable") ?: t.optJSONArray("disableBehaviors") ?: JSONArray()
            val disPassArr = t.optJSONArray("disablePasses") ?: t.optJSONArray("disabledPasses") ?: JSONArray()
            val scaleObj = t.optJSONObject("passScaleOverrides") ?: t.optJSONObject("scaleOverrides")
            val paramObj = t.optJSONObject("paramOverrides")

            val showNodes = HashSet<Int>()
            for (s in 0 until showArr.length()) {
                showNodes.addAll(resolveSelector(showArr.getString(s), nodeNameToIndex, tagToIndices))
            }
            val hideNodes = HashSet<Int>()
            for (h in 0 until hideArr.length()) {
                hideNodes.addAll(resolveSelector(hideArr.getString(h), nodeNameToIndex, tagToIndices))
            }
            val disBehaviors = HashSet<String>()
            for (d in 0 until disArr.length()) {
                disBehaviors.add(disArr.getString(d))
            }
            val disPasses = HashSet<String>()
            for (p in 0 until disPassArr.length()) {
                disPasses.add(disPassArr.getString(p))
            }
            val scaleOverrides = LinkedHashMap<String, Float>()
            if (scaleObj != null) {
                for (k in scaleObj.keys()) {
                    scaleOverrides[k] = scaleObj.optDouble(k, 1.0).toFloat().coerceIn(0.125f, 1.0f)
                }
            }
            val paramOverridesMap = LinkedHashMap<String, Float>()
            if (paramObj != null) {
                for (k in paramObj.keys()) {
                    val v = paramObj.optDouble(k, 0.0).toFloat()
                    paramOverridesMap[k] = v
                    val cleanKey = k.removePrefix("$").removePrefix("param.")
                    val boundTargets = manifestParamBindings[cleanKey]
                    if (boundTargets != null) {
                        for ((bNodeId, bParamKey) in boundTargets) {
                            paramOverridesMap["pass:$bNodeId.$bParamKey"] = v
                            paramOverridesMap["post:$bNodeId.$bParamKey"] = v
                            paramOverridesMap["$bNodeId.$bParamKey"] = v
                        }
                    }
                }
            }

            tierOverrides[tierId] = TierOverride(
                tierId = tierId,
                showNodes = showNodes,
                hideNodes = hideNodes,
                disabledBehaviors = disBehaviors,
                textureMaxSize = t.optInt("textureMaxSize", 1024),
                disabledPasses = disPasses,
                passScaleOverrides = scaleOverrides,
                paramOverrides = paramOverridesMap
            )
        }

        val tiersArr = adapted.optJSONArray("tiers")
        if (tiersArr != null) {
            for (i in 0 until tiersArr.length()) {
                val t = tiersArr.getJSONObject(i)
                val tierId = t.getString("id")
                parseTierEntry(tierId, t)
            }
        } else {
            val tiersObj = adapted.optJSONObject("tiers")
            if (tiersObj != null) {
                for (tierId in tiersObj.keys()) {
                    val t = tiersObj.optJSONObject(tierId) ?: continue
                    parseTierEntry(tierId, t)
                }
            }
        }

        // 12. Lifecycle & Photo Policies
        val rawLifecycle = adapted.optJSONObject("lifecycle")
        val faceLost = rawLifecycle?.optJSONObject("faceLost")
        val reacquire = rawLifecycle?.optJSONObject("reacquire")
        val lifecyclePolicy = CompiledLifecyclePolicy(
            holdMs = faceLost?.optLong("holdMs", 150L) ?: 150L,
            fadeMs = faceLost?.optLong("fadeMs", 200L) ?: 200L,
            fadeCurve = faceLost?.optString("fadeCurve", "cubicOut") ?: "cubicOut",
            afterFade = faceLost?.optString("afterFade", "hideAndReset") ?: "hideAndReset",
            cancelFade = reacquire?.optBoolean("cancelFade", true) ?: true,
            easeInMs = reacquire?.optLong("easeInMs", 100L) ?: 100L,
            preserveState = reacquire?.optBoolean("preserveState", true) ?: true,
            replayStartOn = reacquire?.optBoolean("replayStartOn", false) ?: false
        )

        val rawPhoto = adapted.optJSONObject("photo")
        val photoPolicy = CompiledPhotoPolicy(
            clips = rawPhoto?.optString("clips", "settle") ?: "settle",
            triggers = rawPhoto?.optString("triggers", "disarmed") ?: "disarmed",
            timeSec = if (rawPhoto?.has("timeSec") == true && !rawPhoto.isNull("timeSec")) rawPhoto.getDouble("timeSec").toFloat() else null,
            seed = if (rawPhoto?.has("seed") == true && !rawPhoto.isNull("seed")) rawPhoto.getInt("seed") else null
        )

        // 13. Derive Tracker Requirements
        val hasAnchors = nodesList.any { it.anchor != null && it.anchor != "none" }
        val hasFaceMesh = nodesList.any { it.type == "faceMesh" || it.occluderShape == "faceMesh" }
        val hasSprings = springsList.isNotEmpty()
        val usesPoseSignals = slotMap.keys.any { it.startsWith("face.pose.") }
        val hasIrisFeature = run {
            val fArr = requires?.optJSONArray("features")
            var found = false
            if (fArr != null) {
                for (k in 0 until fArr.length()) {
                    val feat = fArr.optString(k, "")
                    if (feat == "tracker.iris@1" || feat == "node.eyeColor@1") {
                        found = true
                        break
                    }
                }
            }
            found
        }
        val hasIrisNodesOrPasses = nodesList.any {
            it.type == "eyeColor" ||
                (it.type == "recolor" && (
                    (it.faceMeshRegion ?: "iris").contains("iris", ignoreCase = true) ||
                        (it.faceMeshRegion ?: "").contains("sclera", ignoreCase = true)
                    )) ||
                (it.faceMeshRegion != null && (
                    it.faceMeshRegion.contains("iris", ignoreCase = true) ||
                        it.faceMeshRegion.contains("sclera", ignoreCase = true)
                    ))
        } || passGraph.passes.any {
            it.templateId.contains("eyeColor", ignoreCase = true) ||
                it.reads.any { r -> r.contains("iris", ignoreCase = true) || r.contains("sclera", ignoreCase = true) }
        }
        val requiresIris = hasIrisFeature || hasIrisNodesOrPasses
        val hasRecolorNodes = nodesList.any { it.type == "recolor" || it.type == "eyeColor" }
        val hasTeethMask = nodesList.any {
            it.faceMeshRegion != null && it.faceMeshRegion.contains("teeth", ignoreCase = true)
        } || passGraph.passes.any { p ->
            p.reads.any { r -> r.contains("teeth", ignoreCase = true) } ||
                p.writes.contains("teeth", ignoreCase = true)
        }
        val hasFaceTrackedPost = nodesList.any {
            it.type == "post" && (
                it.faceMeshRegion != null ||
                    it.postEffect == "warp" ||
                    it.postEffect == "bilateral" ||
                    it.postEffect == "skinSoften"
            )
        }
        val hasRegionMasks = adapted.optJSONObject("masks") != null ||
            nodesList.any { it.faceMeshRegion != null } ||
            passGraph.passes.any { p ->
                p.reads.any { it.startsWith("\$mask:") } ||
                    p.writes.startsWith("\$mask:")
            }
        val usesFaceSignals = slotMap.keys.any { it.startsWith("face.") }
        val requiresPose = hasAnchors || hasFaceMesh || hasSprings || usesPoseSignals ||
            requiresIris || hasRecolorNodes || hasFaceTrackedPost || hasRegionMasks || usesFaceSignals

        if (requiresIris || hasRecolorNodes) {
            requestedBlendshapes.add("eyeBlinkLeft")
            requestedBlendshapes.add("eyeBlinkRight")
        }
        if (hasTeethMask) {
            requestedBlendshapes.add("jawOpen")
        }
        if (hasFaceMesh) {
            requestedBlendshapes.add("jawOpen")
            requestedBlendshapes.add("eyeBlinkLeft")
            requestedBlendshapes.add("eyeBlinkRight")
        }
        val requiresBlendshapes = requestedBlendshapes.isNotEmpty()

        val minTier = adapted.optString("minTier", requires?.optString("minTier", null)).takeIf { !it.isNullOrBlank() }

        return CompiledEffect(
            packageId = id,
            packageName = name,
            schemaVersion = schema,
            nodes = nodesList,
            materials = materialsMap,
            assets = assetsMap,
            lighting = lighting,
            signalBus = signalBus,
            behaviorRunner = behaviorRunner,
            anchorResolver = anchorResolver,
            propertySlots = propertySlots,
            passGraph = passGraph,
            maskDefs = maskDefs,
            minTier = minTier,
            tierOverrides = tierOverrides,
            requiredLandmarkCount = 478,
            requiresPose = requiresPose,
            requiresBlendshapes = requiresBlendshapes,
            requiresIris = requiresIris,
            requestedBlendshapes = requestedBlendshapes,
            photoPolicy = photoPolicy,
            lifecyclePolicy = lifecyclePolicy
        )
    }

    private fun parsePath(path: String): Pair<String, String> {
        val lastDot = path.lastIndexOf('.')
        return if (lastDot != -1) {
            Pair(path.substring(0, lastDot), path.substring(lastDot + 1))
        } else {
            Pair(path, "scale")
        }
    }

    private fun resolveSelector(
        selector: String,
        nodeNames: Map<String, Int>,
        tags: Map<String, List<Int>>
    ): List<Int> {
        return when {
            selector.startsWith("node:") -> {
                val name = selector.substring(5)
                nodeNames[name]?.let { listOf(it) } ?: emptyList()
            }
            selector.startsWith("tag:") -> {
                val tag = selector.substring(4)
                tags[tag] ?: emptyList()
            }
            selector.startsWith("glb:") -> {
                // glb:nodeId/glbNodeName pattern
                val rest = selector.substring(4)
                val slash = rest.indexOf('/')
                val rootNode = if (slash != -1) rest.substring(0, slash) else rest
                nodeNames[rootNode]?.let { listOf(it) } ?: emptyList()
            }
            else -> {
                nodeNames[selector]?.let { listOf(it) } ?: emptyList()
            }
        }
    }

    fun compileLightingBlock(lightObj: JSONObject): CompiledLighting {
        val space = lightObj.optString("space", "camera")
        val ambArr = lightObj.optJSONArray("ambient")
        val amb = FloatArray(3) { 0.3f }
        if (ambArr != null && ambArr.length() >= 3) {
            amb[0] = ambArr.getDouble(0).toFloat()
            amb[1] = ambArr.getDouble(1).toFloat()
            amb[2] = ambArr.getDouble(2).toFloat()
        }

        val lightDirs = floatArrayOf(
            0f, 0.5f, 0.866f,
            0f, 0f, 1f
        )
        val lightColors = floatArrayOf(
            1f, 1f, 1f,
            0f, 0f, 0f
        )
        val lightIntensities = floatArrayOf(0f, 0f)
        var lightCount = 0

        val lightsArr = lightObj.optJSONArray("lights")
        if (lightsArr != null) {
            lightCount = minOf(lightsArr.length(), CompiledLighting.MAX_LIGHTS)
            for (k in 0 until lightCount) {
                val lk = lightsArr.optJSONObject(k) ?: continue
                val base3 = k * 3
                val dArr = lk.optJSONArray("dirToLight")
                if (dArr != null && dArr.length() >= 3) {
                    lightDirs[base3] = dArr.getDouble(0).toFloat()
                    lightDirs[base3 + 1] = dArr.getDouble(1).toFloat()
                    lightDirs[base3 + 2] = dArr.getDouble(2).toFloat()
                }
                val cArr = lk.optJSONArray("color")
                if (cArr != null && cArr.length() >= 3) {
                    lightColors[base3] = cArr.getDouble(0).toFloat()
                    lightColors[base3 + 1] = cArr.getDouble(1).toFloat()
                    lightColors[base3 + 2] = cArr.getDouble(2).toFloat()
                } else {
                    lightColors[base3] = 1f
                    lightColors[base3 + 1] = 1f
                    lightColors[base3 + 2] = 1f
                }
                lightIntensities[k] = lk.optDouble("intensity", 0.8).toFloat()
            }
        } else if (lightObj.has("dirToLight") || lightObj.has("intensity") || lightObj.has("color")) {
            lightCount = 1
            val dArr = lightObj.optJSONArray("dirToLight")
            if (dArr != null && dArr.length() >= 3) {
                lightDirs[0] = dArr.getDouble(0).toFloat()
                lightDirs[1] = dArr.getDouble(1).toFloat()
                lightDirs[2] = dArr.getDouble(2).toFloat()
            }
            val cArr = lightObj.optJSONArray("color")
            if (cArr != null && cArr.length() >= 3) {
                lightColors[0] = cArr.getDouble(0).toFloat()
                lightColors[1] = cArr.getDouble(1).toFloat()
                lightColors[2] = cArr.getDouble(2).toFloat()
            }
            lightIntensities[0] = lightObj.optDouble("intensity", 0.8).toFloat()
        }

        val dirToLight0 = floatArrayOf(lightDirs[0], lightDirs[1], lightDirs[2])
        val lightColor0 = if (lightCount > 0) {
            floatArrayOf(lightColors[0], lightColors[1], lightColors[2])
        } else {
            floatArrayOf(0f, 0f, 0f)
        }
        val lightIntensity0 = if (lightCount > 0) lightIntensities[0] else 0f

        return CompiledLighting(
            space = space,
            ambient = amb,
            dirToLight = dirToLight0,
            lightColor = lightColor0,
            lightIntensity = lightIntensity0,
            lightCount = lightCount,
            lightDirs = lightDirs,
            lightColors = lightColors,
            lightIntensities = lightIntensities
        )
    }

    /**
     * Compiles a `"lighting"` block from either a full manifest JSON string or a standalone
     * lighting JSON object string. Uses [compileLightingBlock] when `org.json.JSONObject` is
     * active (Android runtime) and falls back to Moshi [com.squareup.moshi.JsonReader] in
     * pure-JVM unit tests where `android.jar` `org.json` methods return default `null`.
     */
    fun compileLightingFromJson(jsonText: String): CompiledLighting {
        val rootObj = try {
            JSONObject(jsonText)
        } catch (_: Throwable) {
            null
        }
        val lightObj = rootObj?.optJSONObject("lighting") ?: rootObj
        if (lightObj != null && (lightObj.optJSONArray("ambient") != null || lightObj.optJSONArray("lights") != null)) {
            return compileLightingBlock(lightObj)
        }

        val moshi = com.squareup.moshi.Moshi.Builder().build()
        val adapter = moshi.adapter(Map::class.java)
        val rootMap = adapter.fromJson(jsonText) ?: emptyMap<Any?, Any?>()
        @Suppress("UNCHECKED_CAST")
        val lightingMap = (rootMap["lighting"] as? Map<String, Any?>)
            ?: (rootMap as? Map<String, Any?>)
            ?: emptyMap()

        val space = (lightingMap["space"] as? String) ?: "camera"
        val amb = FloatArray(3) { 0.3f }
        val ambList = lightingMap["ambient"] as? List<*>
        if (ambList != null && ambList.size >= 3) {
            amb[0] = (ambList[0] as? Number)?.toFloat() ?: 0.3f
            amb[1] = (ambList[1] as? Number)?.toFloat() ?: 0.3f
            amb[2] = (ambList[2] as? Number)?.toFloat() ?: 0.3f
        }

        val lightDirs = floatArrayOf(
            0f, 0.5f, 0.866f,
            0f, 0f, 1f
        )
        val lightColors = floatArrayOf(
            1f, 1f, 1f,
            0f, 0f, 0f
        )
        val lightIntensities = floatArrayOf(0f, 0f)
        var lightCount = 0

        val lightsList = lightingMap["lights"] as? List<*>
        if (lightsList != null) {
            lightCount = minOf(lightsList.size, CompiledLighting.MAX_LIGHTS)
            for (k in 0 until lightCount) {
                val lk = lightsList[k] as? Map<*, *> ?: continue
                val base3 = k * 3
                val dList = lk["dirToLight"] as? List<*>
                if (dList != null && dList.size >= 3) {
                    lightDirs[base3] = (dList[0] as? Number)?.toFloat() ?: 0f
                    lightDirs[base3 + 1] = (dList[1] as? Number)?.toFloat() ?: 0f
                    lightDirs[base3 + 2] = (dList[2] as? Number)?.toFloat() ?: 1f
                }
                val cList = lk["color"] as? List<*>
                if (cList != null && cList.size >= 3) {
                    lightColors[base3] = (cList[0] as? Number)?.toFloat() ?: 1f
                    lightColors[base3 + 1] = (cList[1] as? Number)?.toFloat() ?: 1f
                    lightColors[base3 + 2] = (cList[2] as? Number)?.toFloat() ?: 1f
                } else {
                    lightColors[base3] = 1f
                    lightColors[base3 + 1] = 1f
                    lightColors[base3 + 2] = 1f
                }
                lightIntensities[k] = (lk["intensity"] as? Number)?.toFloat() ?: 0.8f
            }
        } else if (lightingMap.containsKey("dirToLight") || lightingMap.containsKey("intensity") || lightingMap.containsKey("color")) {
            lightCount = 1
            val dList = lightingMap["dirToLight"] as? List<*>
            if (dList != null && dList.size >= 3) {
                lightDirs[0] = (dList[0] as? Number)?.toFloat() ?: 0f
                lightDirs[1] = (dList[1] as? Number)?.toFloat() ?: 0.5f
                lightDirs[2] = (dList[2] as? Number)?.toFloat() ?: 0.866f
            }
            val cList = lightingMap["color"] as? List<*>
            if (cList != null && cList.size >= 3) {
                lightColors[0] = (cList[0] as? Number)?.toFloat() ?: 1f
                lightColors[1] = (cList[1] as? Number)?.toFloat() ?: 1f
                lightColors[2] = (cList[2] as? Number)?.toFloat() ?: 1f
            }
            lightIntensities[0] = (lightingMap["intensity"] as? Number)?.toFloat() ?: 0.8f
        }

        val dirToLight0 = floatArrayOf(lightDirs[0], lightDirs[1], lightDirs[2])
        val lightColor0 = if (lightCount > 0) {
            floatArrayOf(lightColors[0], lightColors[1], lightColors[2])
        } else {
            floatArrayOf(0f, 0f, 0f)
        }
        val lightIntensity0 = if (lightCount > 0) lightIntensities[0] else 0f

        return CompiledLighting(
            space = space,
            ambient = amb,
            dirToLight = dirToLight0,
            lightColor = lightColor0,
            lightIntensity = lightIntensity0,
            lightCount = lightCount,
            lightDirs = lightDirs,
            lightColors = lightColors,
            lightIntensities = lightIntensities
        )
    }

    private fun resolveSelectorToFirstNode(
        selector: String,
        nodeNames: Map<String, Int>,
        tags: Map<String, List<Int>>
    ): Int {
        val list = resolveSelector(selector, nodeNames, tags)
        return if (list.isNotEmpty()) list[0] else 0
    }
}
