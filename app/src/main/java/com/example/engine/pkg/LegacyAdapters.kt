package com.example.engine.pkg

import org.json.JSONArray
import org.json.JSONObject

/**
 * LegacyAdapters translates legacy schema v1, schema v2 (FACE_MESH_TEXTURE),
 * and xfilter-face-3d/1 manifests into canonical xfilter/3 JSON AST before compilation (README Section 6).
 */
object LegacyAdapters {

    fun adapt(json: JSONObject): JSONObject {
        val explicitSchema = json.optString("schema", json.optString("engineContract", ""))
        if (explicitSchema == "xfilter/3") return json
        if ((json.has("nodes") || json.has("passes") || json.has("masks")) && !json.has("layer") && !json.has("layers")) {
            return json
        }
        val pkgFormat = json.optString("packageFormat", "")
        val firstLayerType = json.optJSONArray("layers")?.optJSONObject(0)?.optString("type", "") ?: ""
        if (pkgFormat == "xelvic-filter-rnd" || firstLayerType.equals("MODEL_3D", ignoreCase = true)) {
            return adaptRndModel3d(json)
        }
        val schema = explicitSchema.ifEmpty { if (json.has("schemaVersion")) "xfilter-face-3d/1" else "" }

        return when {
            schema == "xfilter/3" -> json
            schema == "xfilter-face-3d/1" || json.has("layer") || json.has("model") || json.has("model3d") -> adaptFace3d(json)
            schema == "v2" || json.optString("type") == "FACE_MESH_TEXTURE" -> adaptSchemaV2(json)
            schema == "v1" || json.has("layers") -> adaptSchemaV1(json)
            else -> json // Try direct compilation
        }
    }

    /**
     * Convert Schema v1 2D layer filters to xfilter/3 sprite nodes.
     */
    private fun adaptSchemaV1(json: JSONObject): JSONObject {
        val adapted = JSONObject()
        adapted.put("schema", "xfilter/3")
        adapted.put("id", json.optString("id", "legacy_v1"))
        adapted.put("name", json.optString("name", "Legacy 2D Filter"))
        adapted.put("version", json.optString("version", "1.0.0"))

        val requires = JSONObject()
        requires.put("runtime", 3)
        val features = JSONArray()
        features.put("node.sprite@1")
        features.put("material.unlit@1")
        requires.put("features", features)
        adapted.put("requires", requires)

        val assets = JSONArray()
        val materials = JSONObject()
        val nodes = JSONArray()

        val layers = json.optJSONArray("layers") ?: JSONArray()
        for (i in 0 until layers.length()) {
            val layer = layers.getJSONObject(i)
            val assetPath = layer.optString("asset", "")
            val assetId = "tex_$i"

            val assetObj = JSONObject()
            assetObj.put("id", assetId)
            assetObj.put("type", "image")
            assetObj.put("path", assetPath)
            assets.put(assetObj)

            val matId = "mat_$i"
            val matObj = JSONObject()
            matObj.put("template", "unlit@1")
            val matParams = JSONObject()
            matParams.put("map", assetId)
            matObj.put("params", matParams)
            matObj.put("blend", "alpha")
            materials.put(matId, matObj)

            val oldAnchor = layer.optString("anchor", "FOREHEAD").uppercase()
            val newAnchor = when (oldAnchor) {
                "FOREHEAD" -> "forehead"
                "NOSE" -> "nose_tip"
                "MOUTH" -> "mouth_center"
                "CHIN" -> "chin"
                "LEFT_CHEEK" -> "cheek_left"
                "RIGHT_CHEEK" -> "cheek_right"
                "LEFT_EYE" -> "eye_left"
                "RIGHT_EYE" -> "eye_right"
                else -> "head_center"
            }

            val nodeObj = JSONObject()
            nodeObj.put("id", "sprite_$i")
            nodeObj.put("type", "sprite")
            nodeObj.put("anchor", newAnchor)
            nodeObj.put("material", matId)
            nodeObj.put("asset", assetId)

            val sizeObj = JSONObject()
            sizeObj.put("widthFW", layer.optDouble("scale", 0.5))
            sizeObj.put("heightFW", layer.optDouble("scale", 0.5))
            nodeObj.put("size", sizeObj)

            val transform = JSONObject()
            val pos = JSONArray()
            pos.put(layer.optDouble("offsetX", 0.0))
            pos.put(layer.optDouble("offsetY", 0.0))
            pos.put(0.0)
            transform.put("position", pos)

            val rotMode = layer.optString("rotationMode", "HEAD").uppercase()
            transform.put("rotationFollow", if (rotMode == "NONE") "none" else "full")
            nodeObj.put("transform", transform)

            nodes.put(nodeObj)
        }

        adapted.put("assets", assets)
        adapted.put("materials", materials)
        adapted.put("nodes", nodes)
        return adapted
    }

    /**
     * Convert Schema v2 FACE_MESH_TEXTURE to xfilter/3 faceMesh node.
     */
    private fun adaptSchemaV2(json: JSONObject): JSONObject {
        val adapted = JSONObject()
        adapted.put("schema", "xfilter/3")
        adapted.put("id", json.optString("id", "legacy_mesh"))
        adapted.put("name", json.optString("name", "Face Paint"))
        adapted.put("version", "1.0.0")

        val requires = JSONObject()
        requires.put("runtime", 3)
        val features = JSONArray()
        features.put("node.faceMesh@1")
        features.put("material.faceTexture@1")
        requires.put("features", features)
        adapted.put("requires", requires)

        val assets = JSONArray()
        val texturePath = json.optString("texture", json.optString("asset", "assets/texture.png"))
        val assetObj = JSONObject()
        assetObj.put("id", "face_texture")
        assetObj.put("type", "image")
        assetObj.put("path", texturePath)
        assets.put(assetObj)

        val materials = JSONObject()
        val matObj = JSONObject()
        matObj.put("template", "faceTexture@1")
        val matParams = JSONObject()
        matParams.put("map", "face_texture")
        matObj.put("params", matParams)
        matObj.put("blend", "alpha")
        materials.put("mat_mesh", matObj)

        val nodes = JSONArray()
        val nodeObj = JSONObject()
        nodeObj.put("id", "face_paint")
        nodeObj.put("type", "faceMesh")
        nodeObj.put("material", "mat_mesh")
        nodes.put(nodeObj)

        adapted.put("assets", assets)
        adapted.put("materials", materials)
        adapted.put("nodes", nodes)
        return adapted
    }

    /**
     * Convert xfilter-face-3d/1 to xfilter/3 model nodes, occluders, behaviors, tiers, and lifecycle (README Section 6).
     */
    private fun adaptFace3d(json: JSONObject): JSONObject {
        val adapted = JSONObject()
        adapted.put("schema", "xfilter/3")
        val id = json.optString("id", "legacy_face_3d")
        adapted.put("id", id)
        adapted.put("name", json.optString("name", "Legacy Face 3D"))
        adapted.put("version", json.optString("version", "1.0.0"))
        adapted.put("description", json.optString("description", ""))

        // Requires
        val requires = JSONObject()
        requires.put("runtime", 3)
        val features = JSONArray()
        features.put("node.model@1")
        features.put("material.lambert@1")
        features.put("behavior.clip@1")
        features.put("behavior.map@1")
        features.put("behavior.trigger@1")
        features.put("behavior.spring@1")
        features.put("occluder.faceMesh@1")
        features.put("occluder.ellipsoid@1")
        features.put("signal.blendshapes@1")
        requires.put("features", features)
        adapted.put("requires", requires)

        // Assets
        val assets = JSONArray()
        val rawAssets = json.optJSONArray("assets")
        if (rawAssets != null) {
            for (i in 0 until rawAssets.length()) {
                val a = rawAssets.getJSONObject(i)
                val assetObj = JSONObject()
                assetObj.put("id", a.optString("id", "model_asset"))
                val mediaType = a.optString("mediaType", "")
                val path = a.optString("path", "")
                assetObj.put("type", if (mediaType.contains("model") || path.endsWith(".glb")) "model" else "image")
                assetObj.put("path", path)
                assetObj.put("sha256", a.optString("sha256", ""))
                assetObj.put("bytes", a.optLong("bytes", 0L))
                assets.put(assetObj)
            }
        }
        adapted.put("assets", assets)

        // Materials
        val rawMat = json.optJSONObject("materials")
        val rawLighting = json.optJSONObject("lighting")
        val rawWorkflow = rawMat?.optString("workflow", "") ?: ""
        val rawShading = rawLighting?.optString("shading", "") ?: ""
        val usePbrTemplate = rawWorkflow.contains("pbr", ignoreCase = true) || rawShading.equals("pbr", ignoreCase = true)
        if (usePbrTemplate) {
            features.put("material.pbr@1")
        }
        val materials = JSONObject()
        val matObj = JSONObject()
        matObj.put("template", if (usePbrTemplate) "pbr@1" else "lambert@1")
        matObj.put("doubleSided", rawMat?.optBoolean("doubleSided", true) ?: true)
        val rawAlphaMode = rawMat?.optString("alphaMode", "OPAQUE") ?: "OPAQUE"
        matObj.put("blend", if (rawAlphaMode.equals("BLEND", ignoreCase = true)) "alpha" else "opaque")
        matObj.put("depthTest", true)
        matObj.put("depthWrite", !rawAlphaMode.equals("BLEND", ignoreCase = true))
        if (rawMat != null) {
            val matParams = JSONObject()
            if (rawMat.has("metallicFactor") || rawMat.has("metallic")) {
                matParams.put("metallic", rawMat.optDouble("metallicFactor", rawMat.optDouble("metallic", 0.0)))
            }
            if (rawMat.has("roughnessFactor") || rawMat.has("roughness")) {
                matParams.put("roughness", rawMat.optDouble("roughnessFactor", rawMat.optDouble("roughness", 0.5)))
            }
            if (matParams.length() > 0) {
                matObj.put("params", matParams)
            }
        }
        materials.put("mat_lambert", matObj)
        adapted.put("materials", materials)

        // Lighting
        if (json.has("lighting")) {
            adapted.put("lighting", json.getJSONObject("lighting"))
        }

        // Nodes
        val nodes = JSONArray()
        val layer = json.optJSONObject("layer")
        val occlusion = json.optJSONObject("occlusion")
        val tunableFields = json.optJSONObject("calibration")?.optJSONObject("tunableFields")

        // Forward manifest "anchors" block (and any tunableFields "head_center.offsetFW" override)
        val rawAnchors = json.optJSONObject("anchors")
        val tunableHeadCenterOffset = tunableFields?.optJSONArray("head_center.offsetFW")
        if (rawAnchors != null || tunableHeadCenterOffset != null) {
            val adaptedAnchors = JSONObject()
            if (rawAnchors != null) {
                val anchorKeys = rawAnchors.keys()
                while (anchorKeys.hasNext()) {
                    val k = anchorKeys.next()
                    val aObj = rawAnchors.optJSONObject(k) ?: continue
                    val copyObj = JSONObject()
                    if (aObj.has("offsetFW")) {
                        copyObj.put("offsetFW", aObj.optJSONArray("offsetFW"))
                    }
                    if (aObj.has("landmarks")) {
                        copyObj.put("landmarks", aObj.optJSONArray("landmarks"))
                    }
                    adaptedAnchors.put(k, copyObj)
                }
            }
            if (tunableHeadCenterOffset != null) {
                val hcObj = adaptedAnchors.optJSONObject("head_center") ?: JSONObject()
                hcObj.put("offsetFW", tunableHeadCenterOffset)
                adaptedAnchors.put("head_center", hcObj)
            }
            if (adaptedAnchors.length() > 0) {
                adapted.put("anchors", adaptedAnchors)
            }
        }

        // 1. Occluder: Head Proxy (ellipsoid)
        val headProxy = occlusion?.optJSONObject("headProxy")
        if (headProxy != null && headProxy.optBoolean("enabledByLayer", true)) {
            val nodeHead = JSONObject()
            nodeHead.put("id", "head_proxy_occluder")
            nodeHead.put("type", "occluder")
            nodeHead.put("shape", "ellipsoid")
            nodeHead.put("anchor", headProxy.optString("center", "head_center").ifEmpty { "head_center" })
            val radii = headProxy.optJSONArray("radiiFW")
            if (radii != null) {
                nodeHead.put("radiiFW", radii)
            } else {
                val defaultRadii = JSONArray()
                defaultRadii.put(0.56); defaultRadii.put(0.66); defaultRadii.put(0.68)
                nodeHead.put("radiiFW", defaultRadii)
            }
            val tags = JSONArray()
            tags.put("occluder")
            tags.put("headProxy")
            nodeHead.put("tags", tags)
            nodeHead.put("renderOrder", 0)
            nodes.put(nodeHead)
        }

        // 2. Occluder: Face Mesh
        val faceMesh = occlusion?.optJSONObject("faceMesh")
        if (faceMesh != null && faceMesh.optBoolean("enabledByLayer", true)) {
            val nodeMesh = JSONObject()
            nodeMesh.put("id", "face_mesh_occluder")
            nodeMesh.put("type", "occluder")
            nodeMesh.put("shape", "faceMesh")
            nodeMesh.put("pushBackFW", faceMesh.optDouble("pushBackFW", 0.015))
            val tags = JSONArray()
            tags.put("occluder")
            tags.put("faceMesh")
            nodeMesh.put("tags", tags)
            nodeMesh.put("renderOrder", 1)
            nodes.put(nodeMesh)
        }

        // 3. Model Node (from layer)
        val modelAssetId = layer?.optString("asset", "model_3d_asset") ?: "model_3d_asset"
        val modelNode = JSONObject()
        val defaultModelId = layer?.optString("nodeId", "model_3d") ?: "model_3d"
        modelNode.put("id", defaultModelId)
        modelNode.put("type", "model")
        modelNode.put("anchor", layer?.optString("anchor", "head_center") ?: "head_center")
        modelNode.put("asset", modelAssetId)
        modelNode.put("material", "mat_lambert")

        val modelTags = JSONArray()
        modelTags.put("prop")
        modelTags.put("node:prop")
        modelTags.put("tag:model")
        modelNode.put("tags", modelTags)

        val transform = JSONObject()
        val offsetFW = tunableFields?.optJSONArray("layer.offsetFW") ?: layer?.optJSONArray("offsetFW")
        if (offsetFW != null) {
            transform.put("position", offsetFW)
        } else {
            val defaultOff = JSONArray()
            defaultOff.put(-0.25); defaultOff.put(0.40); defaultOff.put(0.47)
            transform.put("position", defaultOff)
        }
        transform.put("scale", layer?.optDouble("scale", 1.0) ?: 1.0)
        transform.put("rotationFollow", "full")
        modelNode.put("transform", transform)
        modelNode.put("renderOrder", 2)
        nodes.put(modelNode)

        adapted.put("nodes", nodes)

        // Signals (smile driver expression)
        val signals = JSONObject()
        signals.put("sig.smile", "mean(face.blend.mouthSmileLeft, face.blend.mouthSmileRight)")
        adapted.put("signals", signals)

        // Behaviors
        val behaviors = JSONArray()
        val rawBehaviors = json.optJSONObject("behaviors")
        if (rawBehaviors != null) {
            // Clips
            val clipsArr = rawBehaviors.optJSONArray("clips")
            if (clipsArr != null) {
                for (i in 0 until clipsArr.length()) {
                    val c = clipsArr.getJSONObject(i)
                    val clipObj = JSONObject()
                    clipObj.put("id", c.optString("id", "clip_$i"))
                    clipObj.put("type", "clip")
                    clipObj.put("target", "node:$defaultModelId")
                    clipObj.put("animation", c.optString("glbAnimation", c.optString("id", "")))
                    clipObj.put("durationSec", c.optDouble("durationSec", 1.0))
                    clipObj.put("loop", c.optBoolean("loop", false))
                    clipObj.put("holdLast", c.optBoolean("holdLast", false))
                    clipObj.put("speed", c.optDouble("speed", 1.0))
                    clipObj.put("startOn", if (c.optString("startEvent") == "face_acquired") "faceAcquired" else "")
                    behaviors.put(clipObj)
                }
            }

            // Driver -> Map behavior
            val driver = rawBehaviors.optJSONObject("driver")
            if (driver != null) {
                val mapObj = JSONObject()
                mapObj.put("id", driver.optString("id", "smile_swell"))
                mapObj.put("type", "map")
                mapObj.put("input", "sig.smile")
                val norm = driver.optJSONObject("normalization")
                val inRange = norm?.optJSONArray("inputRange")
                if (inRange != null) {
                    mapObj.put("inputRange", inRange)
                } else {
                    val defaultIn = JSONArray(); defaultIn.put(0.25); defaultIn.put(0.75)
                    mapObj.put("inputRange", defaultIn)
                }
                val targetExpr = driver.optString("target", "1.0 + 0.12*u")
                val coeffMatch = Regex("""1\.0\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*\*\s*u""").find(targetExpr)
                val maxOut = if (coeffMatch != null) {
                    1.0 + (coeffMatch.groupValues[1].toDoubleOrNull() ?: 0.12)
                } else {
                    1.12
                }
                val outRange = JSONArray(); outRange.put(1.0); outRange.put(maxOut)
                mapObj.put("outputRange", outRange)
                mapObj.put("easing", norm?.optString("easing", "smoothstep") ?: "smoothstep")
                mapObj.put("clamp", true)
                val smoothing = driver.optJSONObject("smoothing")
                mapObj.put("smoothingTau", smoothing?.optDouble("timeConstantSec", 0.12) ?: 0.12)
                mapObj.put("initial", 1.0)

                val writes = JSONArray()
                val writeObj = JSONObject()
                writeObj.put("path", "node:$defaultModelId.scale")
                writeObj.put("mode", "multiply")
                writes.put(writeObj)
                mapObj.put("writes", writes)

                behaviors.put(mapObj)
            }

            // Trigger -> Trigger behavior
            val trigger = rawBehaviors.optJSONObject("trigger")
            if (trigger != null) {
                val trigObj = JSONObject()
                trigObj.put("id", trigger.optString("id", "burst_on_open"))
                trigObj.put("type", "trigger")
                val sig = trigger.optString("signal", "jawOpen")
                trigObj.put("input", if (sig.startsWith("face.blend.")) sig else "face.blend.$sig")
                trigObj.put("risingAbove", trigger.optDouble("risingAbove", 0.55))
                trigObj.put("fallingBelow", trigger.optDouble("fallingBelow", 0.35))
                trigObj.put("cooldownMs", trigger.optLong("cooldownMs", 900L))
                trigObj.put("armedInitially", trigger.optBoolean("armedInitially", true))

                val onFire = JSONArray()
                val actionObj = JSONObject()
                actionObj.put("action", "playClip")
                actionObj.put("clip", trigger.optString("clip", "burst"))
                actionObj.put("restart", trigger.optString("restart", "ignoreIfPlaying"))
                onFire.put(actionObj)
                trigObj.put("onFire", onFire)

                behaviors.put(trigObj)
            }

            // Spring -> Spring behavior
            val spring = rawBehaviors.optJSONObject("spring")
            if (spring != null) {
                val springObj = JSONObject()
                springObj.put("id", spring.optString("id", "head_inertia"))
                springObj.put("type", "spring")
                springObj.put("target", "node:$defaultModelId")
                springObj.put("mode", "rotationInertia")
                springObj.put("stiffness", spring.optDouble("stiffness", 90.0))
                springObj.put("dampingRatio", spring.optDouble("dampingRatio", 0.3))
                springObj.put("gain", spring.optDouble("gain", 0.35))
                springObj.put("maxAngleDeg", spring.optDouble("maxAngleDeg", 8.0))
                springObj.put("lowPassTau", spring.optDouble("lowPassTimeConstantSec", 0.04))
                springObj.put("maxAngularAccel", spring.optDouble("maxAngularAccelerationRadPerSec2", 80.0))
                springObj.put("substepMaxSec", spring.optDouble("substepMaxSec", 0.00833))
                behaviors.put(springObj)
            }
        }
        adapted.put("behaviors", behaviors)

        // Lifecycle
        val rawLifecycle = rawBehaviors?.optJSONObject("lifecycle")
        val lifecycle = JSONObject()
        val faceLost = JSONObject()
        val rawFaceLost = rawLifecycle?.optJSONObject("faceLost")
        faceLost.put("holdMs", rawFaceLost?.optLong("holdMs", 150L) ?: 150L)
        faceLost.put("fadeMs", rawFaceLost?.optLong("fadeMs", 200L) ?: 200L)
        faceLost.put("fadeCurve", "cubicOut")
        faceLost.put("afterFade", "hideAndReset")
        lifecycle.put("faceLost", faceLost)

        val reacquire = JSONObject()
        val rawReacquire = rawLifecycle?.optJSONObject("reacquiredDuringHoldOrFade")
        reacquire.put("cancelFade", rawReacquire?.optBoolean("cancelFade", true) ?: true)
        reacquire.put("easeInMs", rawReacquire?.optLong("easeRootScaleToOneMs", 100L) ?: 100L)
        reacquire.put("preserveState", rawReacquire?.optBoolean("preserveBehaviorState", true) ?: true)
        reacquire.put("replayStartOn", rawReacquire?.optBoolean("replayBloom", false) ?: false)
        lifecycle.put("reacquire", reacquire)
        adapted.put("lifecycle", lifecycle)

        // Tiers
        if (json.has("tiers")) {
            adapted.put("tiers", json.getJSONArray("tiers"))
        }

        return adapted
    }

    /**
     * Converts `xelvic-filter-rnd` / `MODEL_3D` multi-LOD layer manifests into canonical `xfilter/3`
     * nodes, occluders, tiers, lighting, signals, behaviors, and lifecycle policies.
     */
    private fun adaptRndModel3d(json: JSONObject): JSONObject {
        val adapted = JSONObject()
        adapted.put("schema", "xfilter/3")
        adapted.put("id", json.optString("id", "rnd_model_3d"))
        adapted.put("name", json.optString("name", "3D Filter"))
        adapted.put("version", json.optString("version", "1.0.0"))
        adapted.put("description", json.optString("description", ""))

        val requires = json.optJSONObject("requires") ?: JSONObject()
        adapted.put("requires", requires)

        // 1. Assets: assign deterministic IDs when asset entries omit "id"
        val assets = JSONArray()
        val pathToAssetId = LinkedHashMap<String, String>()
        val rawAssets = json.optJSONArray("assets")
        if (rawAssets != null) {
            for (i in 0 until rawAssets.length()) {
                val a = rawAssets.optJSONObject(i) ?: continue
                val path = a.optString("path", "")
                val explicitId = a.optString("id", "").trim()
                val derivedId = explicitId.ifEmpty {
                    path.substringAfterLast('/')
                        .substringBeforeLast('.')
                        .replace(Regex("[^a-zA-Z0-9_]"), "_")
                        .ifEmpty { "asset_$i" }
                }
                pathToAssetId[path] = derivedId
                val assetObj = JSONObject()
                assetObj.put("id", derivedId)
                val mediaType = a.optString("mediaType", "")
                assetObj.put("type", if (mediaType.contains("model") || path.endsWith(".glb", ignoreCase = true)) "model" else "image")
                assetObj.put("path", path)
                assetObj.put("sha256", a.optString("sha256", ""))
                assetObj.put("bytes", a.optLong("bytes", 0L))
                assets.put(assetObj)
            }
        }
        adapted.put("assets", assets)

        // 2. Materials (inspect layer material definitions for PBR metallic-roughness workflow)
        val firstLayerMat = json.optJSONArray("layers")?.optJSONObject(0)?.optJSONObject("material")
        val rndWorkflow = firstLayerMat?.optString("workflow", "") ?: ""
        val hasRndPbrFactors = firstLayerMat != null && (
            firstLayerMat.has("metallicFactor") ||
                firstLayerMat.has("roughnessFactor") ||
                firstLayerMat.has("metallic") ||
                firstLayerMat.has("roughness")
            )
        val isRndPbr = rndWorkflow.contains("pbr", ignoreCase = true) || hasRndPbrFactors
        val reqFeatures = requires.optJSONArray("features")
        if (isRndPbr && reqFeatures != null) {
            var hasPbrFeature = false
            for (fIdx in 0 until reqFeatures.length()) {
                if (reqFeatures.optString(fIdx) == "material.pbr@1") {
                    hasPbrFeature = true
                    break
                }
            }
            if (!hasPbrFeature) {
                reqFeatures.put("material.pbr@1")
            }
        }
        val materials = JSONObject()
        val matObj = JSONObject()
        matObj.put("template", if (isRndPbr) "pbr@1" else "lambert@1")
        matObj.put("doubleSided", firstLayerMat?.optBoolean("doubleSided", true) ?: true)
        val rndAlphaMode = firstLayerMat?.optString("alphaMode", "OPAQUE") ?: "OPAQUE"
        matObj.put("blend", if (rndAlphaMode.equals("BLEND", ignoreCase = true)) "alpha" else "opaque")
        matObj.put("depthTest", true)
        matObj.put("depthWrite", !rndAlphaMode.equals("BLEND", ignoreCase = true))
        if (firstLayerMat != null) {
            val matParams = JSONObject()
            if (firstLayerMat.has("metallicFactor") || firstLayerMat.has("metallic")) {
                matParams.put("metallic", firstLayerMat.optDouble("metallicFactor", firstLayerMat.optDouble("metallic", 0.0)))
            }
            if (firstLayerMat.has("roughnessFactor") || firstLayerMat.has("roughness")) {
                matParams.put("roughness", firstLayerMat.optDouble("roughnessFactor", firstLayerMat.optDouble("roughness", 0.45)))
            }
            if (matParams.length() > 0) {
                matObj.put("params", matParams)
            }
        }
        materials.put("mat_lambert", matObj)
        adapted.put("materials", materials)

        // 3. Lighting (preserve manifest lighting if declared, otherwise provide balanced soft studio lighting)
        val rawLighting = json.optJSONObject("lighting")
        if (rawLighting != null) {
            adapted.put("lighting", rawLighting)
        } else {
            val lighting = JSONObject()
            lighting.put("space", "camera")
            val amb = JSONArray().apply { put(0.42); put(0.39); put(0.40) }
            lighting.put("ambient", amb)
            val lights = JSONArray()
            val keyLight = JSONObject().apply {
                put("dirToLight", JSONArray().apply { put(-0.25); put(0.60); put(0.76) })
                put("color", JSONArray().apply { put(1.0); put(0.96); put(0.98) })
                put("intensity", 0.72)
            }
            val fillLight = JSONObject().apply {
                put("dirToLight", JSONArray().apply { put(0.45); put(0.20); put(0.87) })
                put("color", JSONArray().apply { put(1.0); put(0.90); put(0.94) })
                put("intensity", 0.25)
            }
            lights.put(keyLight)
            lights.put(fillLight)
            lighting.put("lights", lights)
            adapted.put("lighting", lighting)
        }

        // 4. Layers -> Occluders + Per-Tier LOD Model Nodes + Tiers + Behaviors + Lifecycle
        val nodes = JSONArray()
        val signals = JSONObject()
        val behaviors = JSONArray()
        val tierLodNodeMap = LinkedHashMap<String, MutableList<String>>()
        val allLodNodeSelectors = ArrayList<String>()
        var lifecycleObj: JSONObject? = null
        var renderOrderCounter = 0

        val layersArr = json.optJSONArray("layers") ?: JSONArray()
        for (lIdx in 0 until layersArr.length()) {
            val layer = layersArr.optJSONObject(lIdx) ?: continue
            val layerId = layer.optString("id", "layer_$lIdx")

            val anchorObj = layer.optJSONObject("anchor")
            val anchorName = (anchorObj?.optString("type", "")?.takeIf { it.isNotEmpty() }
                ?: layer.optString("anchor", "head_center")).ifEmpty { "head_center" }

            // Transform (parsed before occluders so temple proxy occluders can align relative to layer offset)
            val transObj = layer.optJSONObject("transform")
            val offsetObj = transObj?.optJSONObject("offsetFW")
            val offsetArr = transObj?.optJSONArray("offsetFW") ?: layer.optJSONArray("offsetFW")
            val posArr = JSONArray()
            if (offsetObj != null) {
                posArr.put(offsetObj.optDouble("x", 0.0))
                posArr.put(offsetObj.optDouble("y", 0.0))
                posArr.put(offsetObj.optDouble("z", 0.0))
            } else if (offsetArr != null && offsetArr.length() >= 3) {
                posArr.put(offsetArr.getDouble(0))
                posArr.put(offsetArr.getDouble(1))
                posArr.put(offsetArr.getDouble(2))
            } else {
                posArr.put(0.0); posArr.put(0.0); posArr.put(0.0)
            }

            val scaleFw = transObj?.optDouble("scaleFW", transObj.optDouble("scale", 1.0)) ?: 1.0
            val fitObj = layer.optJSONObject("fit")
            val fitTarget = fitObj?.optString("to", "faceWidth")?.trim() ?: "faceWidth"
            val fitRatio = if (fitTarget.equals("faceWidth", ignoreCase = true)) {
                fitObj?.optDouble("ratio", 1.0) ?: 1.0
            } else {
                1.0
            }
            val effectiveScale = scaleFw * fitRatio

            // Occlusion
            val occ = layer.optJSONObject("occlusion")
            val headProxy = occ?.optJSONObject("headProxyEllipsoid") ?: occ?.optJSONObject("headProxy")
            if (headProxy != null && headProxy.optBoolean("enabled", headProxy.optBoolean("enabledByLayer", true))) {
                val nodeHead = JSONObject()
                nodeHead.put("id", "${layerId}_head_proxy_occluder")
                nodeHead.put("type", "occluder")
                nodeHead.put("shape", "ellipsoid")
                nodeHead.put("anchor", anchorName)
                val radiiObj = headProxy.optJSONObject("radiiFW")
                val radiiArr = headProxy.optJSONArray("radiiFW")
                val outRadii = JSONArray()
                if (radiiObj != null) {
                    outRadii.put(radiiObj.optDouble("x", 0.62))
                    outRadii.put(radiiObj.optDouble("y", 0.82))
                    outRadii.put(radiiObj.optDouble("z", 0.35))
                } else if (radiiArr != null && radiiArr.length() >= 3) {
                    outRadii.put(radiiArr.getDouble(0))
                    outRadii.put(radiiArr.getDouble(1))
                    outRadii.put(radiiArr.getDouble(2))
                } else {
                    outRadii.put(0.62); outRadii.put(0.82); outRadii.put(0.35)
                }
                nodeHead.put("radiiFW", outRadii)
                val centerOffObj = headProxy.optJSONObject("centerOffsetFW")
                if (centerOffObj != null) {
                    nodeHead.put("transform", JSONObject().apply {
                        put("position", JSONArray().apply {
                            put(centerOffObj.optDouble("x", 0.0))
                            put(centerOffObj.optDouble("y", 0.0))
                            put(centerOffObj.optDouble("z", 0.0))
                        })
                        put("rotationFollow", "full")
                    })
                }
                nodeHead.put("renderOrder", renderOrderCounter++)
                nodes.put(nodeHead)
            }

            val faceMeshOcc = occ?.optJSONObject("faceMesh")
            if (faceMeshOcc != null && faceMeshOcc.optBoolean("enabled", faceMeshOcc.optBoolean("enabledByLayer", true))) {
                val nodeMesh = JSONObject()
                nodeMesh.put("id", "${layerId}_face_mesh_occluder")
                nodeMesh.put("type", "occluder")
                nodeMesh.put("shape", "faceMesh")
                nodeMesh.put("pushBackFW", faceMeshOcc.optDouble("pushBackFW", 0.015))
                nodeMesh.put("renderOrder", renderOrderCounter++)
                nodes.put(nodeMesh)
            }

            val earTempleProxy = occ?.optJSONObject("earTempleProxy")
            if (earTempleProxy != null && earTempleProxy.optBoolean("enabled", true)) {
                val radiusFw = earTempleProxy.optDouble("radiusFW", 0.042)
                var minAbsX = 0.465
                var minY = -0.10
                var maxY = 0.02
                var minZ = -0.35
                var maxZ = -0.04
                val pathsArr = earTempleProxy.optJSONArray("pathsFW")
                if (pathsArr != null && pathsArr.length() > 0) {
                    var foundPoint = false
                    for (pIdx in 0 until pathsArr.length()) {
                        val path = pathsArr.optJSONArray(pIdx) ?: continue
                        for (ptIdx in 0 until path.length()) {
                            val pt = path.optJSONArray(ptIdx) ?: continue
                            if (pt.length() >= 3) {
                                val px = kotlin.math.abs(pt.optDouble(0, 0.465))
                                val py = pt.optDouble(1, -0.05)
                                val pz = pt.optDouble(2, -0.18)
                                if (!foundPoint) {
                                    minAbsX = px
                                    minY = py; maxY = py
                                    minZ = pz; maxZ = pz
                                    foundPoint = true
                                } else {
                                    if (px < minAbsX) minAbsX = px
                                    if (py < minY) minY = py
                                    if (py > maxY) maxY = py
                                    if (pz < minZ) minZ = pz
                                    if (pz > maxZ) maxZ = pz
                                }
                            }
                        }
                    }
                }
                val medialHalfX = (minAbsX - radiusFw).coerceAtLeast(0.32)
                val boxWidthFw = medialHalfX * 2.0
                val boxHeightFw = ((maxY - minY) + radiusFw * 6.0).coerceAtLeast(0.32)
                val boxDepthFw = ((maxZ - minZ) + radiusFw * 2.5).coerceAtLeast(0.36)
                val centerX = posArr.optDouble(0, 0.0)
                val centerY = posArr.optDouble(1, 0.0) + 0.5 * (minY + maxY)
                val centerZ = posArr.optDouble(2, 0.0) + 0.5 * (minZ + maxZ) - radiusFw

                val nodeTemple = JSONObject()
                nodeTemple.put("id", "${layerId}_ear_temple_occluder")
                nodeTemple.put("type", "occluder")
                nodeTemple.put("shape", "box")
                nodeTemple.put("anchor", anchorName)
                nodeTemple.put("radiiFW", JSONArray().apply {
                    put(boxWidthFw)
                    put(boxHeightFw)
                    put(boxDepthFw)
                })
                nodeTemple.put("transform", JSONObject().apply {
                    put("position", JSONArray().apply {
                        put(centerX)
                        put(centerY)
                        put(centerZ)
                    })
                    put("rotationFollow", "full")
                })
                nodeTemple.put("renderOrder", renderOrderCounter++)
                nodes.put(nodeTemple)
            }

            // LODs
            val lodsArr = layer.optJSONArray("lods")
            if (lodsArr != null && lodsArr.length() > 0) {
                for (k in 0 until lodsArr.length()) {
                    val lodObj = lodsArr.optJSONObject(k) ?: continue
                    val tierId = lodObj.optString("tier", "A").trim().uppercase()
                    val assetPath = lodObj.optString("asset", "")
                    val assetId = pathToAssetId[assetPath] ?: assetPath
                    val nodeId = "${layerId}_lod_${tierId.lowercase()}"
                    val nodeSelector = "node:$nodeId"

                    val modelNode = JSONObject()
                    modelNode.put("id", nodeId)
                    modelNode.put("type", "model")
                    modelNode.put("anchor", anchorName)
                    modelNode.put("asset", assetId)
                    modelNode.put("material", "mat_lambert")
                    val tags = JSONArray().apply {
                        put("prop")
                        put(layerId)
                    }
                    modelNode.put("tags", tags)
                    val nodeTransform = JSONObject().apply {
                        put("position", JSONArray().apply {
                            put(posArr.getDouble(0))
                            put(posArr.getDouble(1))
                            put(posArr.getDouble(2))
                        })
                        put("scale", effectiveScale)
                        put("rotationFollow", "full")
                    }
                    modelNode.put("transform", nodeTransform)
                    modelNode.put("renderOrder", renderOrderCounter++)
                    nodes.put(modelNode)

                    tierLodNodeMap.getOrPut(tierId) { ArrayList() }.add(nodeSelector)
                    allLodNodeSelectors.add(nodeSelector)
                }
            }

            // Bounded Float & Spring Behaviors
            val layerBehaviors = layer.optJSONObject("behaviors")
            val gentleFloat = layerBehaviors?.optJSONObject("gentleFloat")
            if (gentleFloat != null) {
                val maxAngleDeg = gentleFloat.optDouble("maxAngleDeg", 2.2)
                val maxTransFw = gentleFloat.optDouble("maxTranslationFW", 0.006)
                val periodMs = gentleFloat.optDouble("periodMs", 3200.0).coerceAtLeast(200.0)
                val omega = (2.0 * Math.PI * 1000.0) / periodMs
                val sigName = "${layerId}_float_wave"
                signals.put(sigName, "sin(time.render * ${"%.6f".format(java.util.Locale.US, omega)})")

                if (maxTransFw > 0.0) {
                    val mapObj = JSONObject()
                    mapObj.put("id", "${layerId}_gentle_float")
                    mapObj.put("type", "map")
                    mapObj.put("input", "sig.$sigName")
                    mapObj.put("inputRange", JSONArray().apply { put(-1.0); put(1.0) })
                    mapObj.put("outputRange", JSONArray().apply { put(-maxTransFw); put(maxTransFw) })
                    mapObj.put("easing", "linear")
                    mapObj.put("clamp", true)
                    mapObj.put("smoothingTau", 0.08)
                    mapObj.put("initial", 0.0)
                    val writes = JSONArray()
                    writes.put(JSONObject().apply {
                        put("path", "tag:$layerId.positionY")
                        put("mode", "add")
                    })
                    mapObj.put("writes", writes)
                    behaviors.put(mapObj)
                }

                if (maxAngleDeg > 0.0) {
                    val springObj = JSONObject()
                    springObj.put("id", "${layerId}_gentle_inertia")
                    springObj.put("type", "spring")
                    springObj.put("target", "tag:$layerId")
                    springObj.put("mode", "rotationInertia")
                    springObj.put("stiffness", 70.0)
                    springObj.put("dampingRatio", 0.6)
                    springObj.put("gain", 0.18)
                    springObj.put("maxAngleDeg", maxAngleDeg)
                    springObj.put("lowPassTau", 0.04)
                    behaviors.put(springObj)
                }
            }

            // Lifecycle
            val rawLife = layer.optJSONObject("lifecycle")?.optJSONObject("faceLost")
            if (rawLife != null && lifecycleObj == null) {
                val lc = JSONObject()
                val fl = JSONObject()
                fl.put("holdMs", rawLife.optLong("holdMs", 160L))
                fl.put("fadeMs", rawLife.optLong("fadeMs", 220L))
                fl.put("fadeCurve", "cubicOut")
                fl.put("afterFade", "hideAndReset")
                lc.put("faceLost", fl)
                val reaq = JSONObject()
                reaq.put("cancelFade", true)
                reaq.put("easeInMs", 100L)
                reaq.put("preserveState", true)
                reaq.put("replayStartOn", false)
                lc.put("reacquire", reaq)
                lifecycleObj = lc
            }
        }

        adapted.put("nodes", nodes)
        if (signals.length() > 0) {
            adapted.put("signals", signals)
        }
        if (behaviors.length() > 0) {
            adapted.put("behaviors", behaviors)
        }
        if (lifecycleObj != null) {
            adapted.put("lifecycle", lifecycleObj)
        }

        // 5. Tiers (show matching tier LOD node, hide other tier LOD nodes)
        if (tierLodNodeMap.isNotEmpty()) {
            val tiers = JSONArray()
            for (tierId in listOf("A", "B", "C")) {
                val showList = tierLodNodeMap[tierId] ?: continue
                val hideList = allLodNodeSelectors.filter { it !in showList }
                val tierObj = JSONObject()
                tierObj.put("id", tierId)
                val showArr = JSONArray().apply { showList.forEach { put(it) } }
                val hideArr = JSONArray().apply { hideList.forEach { put(it) } }
                tierObj.put("show", showArr)
                tierObj.put("hide", hideArr)
                tierObj.put("textureMaxSize", when (tierId) {
                    "A" -> 1024
                    "B" -> 512
                    else -> 256
                })
                tiers.put(tierObj)
            }
            adapted.put("tiers", tiers)
        }

        return adapted
    }
}
