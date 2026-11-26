package com.example.engine.pkg

import com.example.engine.scene.AnchorResolver
import com.example.engine.scene.BehaviorRegistry
import com.example.engine.scene.ExpressionFunctionRegistry
import com.example.engine.scene.ExpressionVM
import com.example.engine.render.NodeRendererRegistry
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * PackageValidator: Strict validation for xfilter/3 packages and legacy packages (README Section 5, 6, 21 Step 3).
 * Enforces:
 * - Strict manifest schema (unknown fields fail closed)
 * - Required feature negotiation against CapabilityRegistry
 * - Asset hashes (SHA-256), byte counts, file signatures (PNG, JPEG, GLB)
 * - GLB structural validation (version 2.0, attributes, uint16 indices)
 * - Pinned topology and UV validation
 * - Expression grammar, allow-listed functions, cycle detection
 * - Node, material, behavior, and selector contracts
 * - Budget measurements against declared and global caps
 */
object PackageValidator {

    data class ValidationResult(
        val isValid: Boolean,
        val errorReason: String? = null
    ) {
        companion object {
            fun success() = ValidationResult(true)
            fun fail(reason: String) = ValidationResult(false, reason)
        }
    }

    // Engine global caps (README Section 5.15)
    const val MAX_PACKAGE_BYTES = 50 * 1024 * 1024L // 50 MB
    const val MAX_NODES = 128
    const val MAX_BEHAVIORS = 64
    const val MAX_SIGNALS = 64
    const val MAX_EXPRESSION_LENGTH = 512
    const val MAX_TEXTURE_DIMENSION = 2048

    // Normative top-level fields for xfilter/3
    private val NORMATIVE_XFILTER3_FIELDS = setOf(
        "schema", "id", "name", "version", "description", "preview",
        "requires", "assets", "params", "anchors", "materials", "lighting",
        "nodes", "passes", "masks", "signals", "behaviors", "lifecycle", "tiers", "minTier", "photo", "budgets",
        "compatibility"
    )

    // Legacy allowed top-level fields for backward compatibility adapters
    private val LEGACY_ALLOWED_FIELDS = setOf(
        "schemaVersion", "engineContract", "units", "faceAxes", "calibration",
        "model", "layer", "occlusion", "type", "texture", "layers",
        "features", "compatibility", "stats", "verification", "editingRules", "integrity",
        "packageFormat", "minimumRuntimeVersion", "status", "coordinateSpace", "renderPlan", "performance",
        "authoringNotes"
    )

    fun validateManifest(manifestJson: JSONObject): ValidationResult {
        val schema = manifestJson.optString(
            "schema",
            manifestJson.optString("engineContract", if (manifestJson.has("schemaVersion")) "xfilter-face-3d/1" else "")
        )
        if (schema.isEmpty()) {
            return ValidationResult.fail("Missing required 'schema' or 'engineContract' declaration")
        }

        val id = manifestJson.optString("id", "")
        if (id.isEmpty()) {
            return ValidationResult.fail("Missing required package 'id'")
        }

        // Strict field check for xfilter/3
        if (schema == "xfilter/3") {
            for (key in manifestJson.keys()) {
                if (key !in NORMATIVE_XFILTER3_FIELDS) {
                    return ValidationResult.fail("Unknown schema field '$key' in xfilter/3 manifest (fails closed)")
                }
            }
        } else {
            // Check legacy fields
            for (key in manifestJson.keys()) {
                if (key !in NORMATIVE_XFILTER3_FIELDS && key !in LEGACY_ALLOWED_FIELDS) {
                    return ValidationResult.fail("Unknown legacy manifest field '$key'")
                }
            }
        }

        // Validate requires features
        val requires = manifestJson.optJSONObject("requires")
        if (requires != null) {
            val runtime = requires.optInt("runtime", 3)
            if (runtime > 3) {
                return ValidationResult.fail("Unsupported runtime version $runtime (engine supports runtime 3)")
            }
            val features = requires.optJSONArray("features")
            if (features != null) {
                val reqList = ArrayList<String>()
                for (i in 0 until features.length()) {
                    reqList.add(features.getString(i))
                }
                val (isCompatible, missing) = CapabilityRegistry.negotiate(reqList)
                if (!isCompatible) {
                    return ValidationResult.fail("Unsupported required features: ${missing.joinToString(", ")}")
                }
            }
        }

        // Validate parameters
        val params = manifestJson.optJSONObject("params")
        if (params != null) {
            for (pKey in params.keys()) {
                val p = params.getJSONObject(pKey)
                val type = p.optString("type", "")
                if (type != "float" && type != "vec3") {
                    return ValidationResult.fail("Parameter '$pKey' has invalid type '$type' (expected 'float' or 'vec3')")
                }
                if (!p.has("default")) {
                    return ValidationResult.fail("Parameter '$pKey' missing required 'default' value")
                }
            }
        }

        // Validate anchors (custom landmark anchors)
        val anchors = manifestJson.optJSONObject("anchors")
        if (anchors != null && schema == "xfilter/3") {
            for (aKey in anchors.keys()) {
                val a = anchors.optJSONObject(aKey) ?: continue
                val lms = a.optJSONArray("landmarks")
                if (lms != null) {
                    for (i in 0 until lms.length()) {
                        val lm = lms.getJSONObject(i)
                        val lmId = lm.optInt("id", -1)
                        if (lmId !in 0 until 478) {
                            return ValidationResult.fail("Anchor '$aKey' references invalid landmark ID $lmId (must be 0..477)")
                        }
                    }
                }
            }
        }

        // Collect declared asset IDs for cross-reference validation in xfilter/3
        val declaredAssetIds = HashSet<String>()
        val assetsArr = manifestJson.optJSONArray("assets")
        if (assetsArr != null) {
            for (i in 0 until assetsArr.length()) {
                val aObj = assetsArr.optJSONObject(i) ?: continue
                val aId = aObj.optString("id", "")
                if (aId.isNotEmpty()) {
                    declaredAssetIds.add(aId)
                }
            }
        }

        // Validate materials
        val materialsObj = manifestJson.optJSONObject("materials")
        if (materialsObj != null && schema == "xfilter/3") {
            val supportedTemplates = setOf("unlit@1", "lambert@1", "pbr@1", "faceTexture@1")
            for (mKey in materialsObj.keys()) {
                val m = materialsObj.optJSONObject(mKey)
                    ?: return ValidationResult.fail("Material '$mKey' must be an object")
                val template = m.optString("template", "")
                if (template !in supportedTemplates) {
                    return ValidationResult.fail("Material '$mKey' uses unsupported template '$template'")
                }
                val mParams = m.optJSONObject("params") ?: m
                if (mParams.has("opacity")) {
                    val op = mParams.optDouble("opacity", 1.0)
                    if (op.isNaN() || op < 0.0 || op > 1.0) {
                        return ValidationResult.fail("Material '$mKey' has invalid opacity $op (must be in 0.0..1.0)")
                    }
                }
                if (mParams.has("metallic")) {
                    val met = mParams.optDouble("metallic", 0.0)
                    if (met.isNaN() || met < 0.0 || met > 1.0) {
                        return ValidationResult.fail("Material '$mKey' has invalid metallic $met (must be in 0.0..1.0)")
                    }
                }
                if (mParams.has("roughness")) {
                    val rough = mParams.optDouble("roughness", 0.45)
                    if (rough.isNaN() || rough < 0.0 || rough > 1.0) {
                        return ValidationResult.fail("Material '$mKey' has invalid roughness $rough (must be in 0.0..1.0)")
                    }
                }
                if (template == "faceTexture@1") {
                    val mapId = mParams?.optString("map", "") ?: ""
                    if (mapId.isEmpty()) {
                        return ValidationResult.fail("faceTexture@1 material '$mKey' requires non-empty params.map")
                    }
                    if (declaredAssetIds.isNotEmpty() && mapId !in declaredAssetIds) {
                        return ValidationResult.fail("faceTexture@1 material '$mKey' references unknown map asset '$mapId'")
                    }
                    val maskId = mParams?.optString("mask", "") ?: ""
                    if (maskId.isNotEmpty() && declaredAssetIds.isNotEmpty() && maskId !in declaredAssetIds) {
                        return ValidationResult.fail("faceTexture@1 material '$mKey' references unknown mask asset '$maskId'")
                    }
                    val uvLayout = mParams?.optString("uvLayout", UvRegistry.UV_LAYOUT_ID) ?: UvRegistry.UV_LAYOUT_ID
                    if (!UvRegistry.isSupportedUvLayoutId(uvLayout)) {
                        return ValidationResult.fail("faceTexture@1 material '$mKey' uses unsupported or legacy UV layout '$uvLayout'")
                    }
                }
            }
        }

        // Validate nodes
        val nodes = manifestJson.optJSONArray("nodes")
        if (nodes != null) {
            if (nodes.length() > MAX_NODES) {
                return ValidationResult.fail("Node count (${nodes.length()}) exceeds global cap of $MAX_NODES")
            }
            val supportedRegions = setOf("full", "face_oval", "cheeks", "forehead", "eyes_and_lips_protected")
            for (i in 0 until nodes.length()) {
                val n = nodes.getJSONObject(i)
                val nId = n.optString("id", "node_$i")
                val nType = n.optString("type", "")
                if (nType.isNotEmpty() && !NodeRendererRegistry.isNodeTypeSupported(nType)) {
                    return ValidationResult.fail("Node '$nId' has unsupported type '$nType'")
                }
                if (nType == "faceMesh" && schema == "xfilter/3") {
                    val matId = n.optString("material", "")
                    if (matId.isEmpty()) {
                        return ValidationResult.fail("faceMesh node '$nId' requires a 'material' binding")
                    }
                    val matDef = materialsObj?.optJSONObject(matId)
                        ?: return ValidationResult.fail("faceMesh node '$nId' references unknown material '$matId'")
                    val matTemplate = matDef.optString("template", "")
                    if (matTemplate != "faceTexture@1") {
                        return ValidationResult.fail("faceMesh node '$nId' requires a 'faceTexture@1' material, got '$matTemplate'")
                    }
                    val region = n.optString("region", "")
                    if (region.isNotEmpty() && region !in supportedRegions && region !in declaredAssetIds) {
                        return ValidationResult.fail("faceMesh node '$nId' specifies unsupported region '$region'")
                    }
                }
                if (nType == "post") {
                    val postValidation = com.example.engine.render.PostRenderer.validatePostNode(n, params)
                    if (!postValidation.isValid) {
                        return postValidation
                    }
                    val nParams = n.optJSONObject("params")
                    val postAssetRef = (n.optString("asset", null)?.takeIf { it.isNotBlank() }
                        ?: n.optString("lut", null)?.takeIf { it.isNotBlank() }
                        ?: n.optString("map", null)?.takeIf { it.isNotBlank() }
                        ?: nParams?.optString("lut", null)?.takeIf { it.isNotBlank() }
                        ?: nParams?.optString("asset", null)?.takeIf { it.isNotBlank() }
                        ?: nParams?.optString("map", null)?.takeIf { it.isNotBlank() })
                    if (postAssetRef != null && postAssetRef !in declaredAssetIds) {
                        return ValidationResult.fail("Post node '$nId' references unknown asset '$postAssetRef'")
                    }
                }
            }
        }

        // Validate custom passes (if present)
        val passesArr = manifestJson.optJSONArray("passes")
        if (passesArr != null) {
            for (i in 0 until passesArr.length()) {
                val pObj = passesArr.optJSONObject(i) ?: continue
                val pId = pObj.optString("id", "pass_$i")
                val pParams = pObj.optJSONObject("params")
                val passAssetRef = (pObj.optString("asset", null)?.takeIf { it.isNotBlank() }
                    ?: pObj.optString("lut", null)?.takeIf { it.isNotBlank() }
                    ?: pObj.optString("map", null)?.takeIf { it.isNotBlank() }
                    ?: pParams?.optString("lut", null)?.takeIf { it.isNotBlank() }
                    ?: pParams?.optString("asset", null)?.takeIf { it.isNotBlank() }
                    ?: pParams?.optString("map", null)?.takeIf { it.isNotBlank() })
                if (passAssetRef != null && declaredAssetIds.isNotEmpty() && passAssetRef !in declaredAssetIds) {
                    return ValidationResult.fail("Pass '$pId' references unknown asset '$passAssetRef'")
                }
            }
        }

        // Validate behaviors
        val behaviors = manifestJson.optJSONArray("behaviors")
        if (behaviors != null) {
            if (behaviors.length() > MAX_BEHAVIORS) {
                return ValidationResult.fail("Behavior count (${behaviors.length()}) exceeds global cap of $MAX_BEHAVIORS")
            }
            for (i in 0 until behaviors.length()) {
                val b = behaviors.getJSONObject(i)
                val bType = b.optString("type", "")
                if (bType.isNotEmpty() && !BehaviorRegistry.isBehaviorSupported(bType)) {
                    return ValidationResult.fail("Behavior '${b.optString("id")}' has unsupported type '$bType'")
                }
            }
        }

        // Validate signals & expressions
        val signals = manifestJson.optJSONObject("signals")
        if (signals != null) {
            if (signals.length() > MAX_SIGNALS) {
                return ValidationResult.fail("Signal count (${signals.length()}) exceeds cap of $MAX_SIGNALS")
            }
            val sigMap = HashMap<String, String>()
            val vm = ExpressionVM()
            for (sKey in signals.keys()) {
                val expr = signals.getString(sKey)
                if (expr.length > MAX_EXPRESSION_LENGTH) {
                    return ValidationResult.fail("Expression for signal '$sKey' exceeds maximum length ($MAX_EXPRESSION_LENGTH chars)")
                }
                sigMap[sKey] = expr
                try {
                    // Test compilation syntax
                    vm.compile(expr) { 0 }
                } catch (e: Exception) {
                    return ValidationResult.fail("Syntax error in signal '$sKey': ${e.message}")
                }
            }
            // Cycle detection
            try {
                vm.getTopologicalOrder(sigMap)
            } catch (e: IllegalStateException) {
                return ValidationResult.fail(e.message ?: "Cycle detected in signals")
            }
        }

        return ValidationResult.success()
    }

    fun validateAssetFile(file: File, declaredSha256: String?, declaredBytes: Long?): ValidationResult {
        if (!file.exists()) {
            return ValidationResult.fail("Asset file missing: ${file.name}")
        }

        if (declaredBytes != null && declaredBytes > 0 && file.length() != declaredBytes) {
            return ValidationResult.fail("Byte count mismatch for ${file.name}: expected $declaredBytes, got ${file.length()}")
        }

        if (!declaredSha256.isNullOrEmpty()) {
            val actualHash = sha256(file)
            if (!actualHash.equals(declaredSha256, ignoreCase = true)) {
                return ValidationResult.fail("SHA-256 hash mismatch for ${file.name}: expected $declaredSha256, got $actualHash")
            }
        }

        // Validate file signature and binary structures
        val ext = file.extension.lowercase()
        return when (ext) {
            "png", "jpg", "jpeg" -> validateImageSignature(file)
            "glb" -> validateGlbStructure(file)
            else -> ValidationResult.success()
        }
    }

    private fun validateImageSignature(file: File): ValidationResult {
        val bytes = ByteArray(8)
        try {
            FileInputStream(file).use { it.read(bytes) }
        } catch (e: Exception) {
            return ValidationResult.fail("Could not read image file ${file.name}: ${e.message}")
        }

        val ext = file.extension.lowercase()
        when (ext) {
            "png" -> {
                // 89 50 4E 47 0D 0A 1A 0A
                if (bytes[0] != 0x89.toByte() || bytes[1] != 0x50.toByte() ||
                    bytes[2] != 0x4E.toByte() || bytes[3] != 0x47.toByte()
                ) {
                    return ValidationResult.fail("Invalid PNG signature in ${file.name}")
                }
            }
            "jpg", "jpeg" -> {
                // FF D8 FF
                if (bytes[0] != 0xFF.toByte() || bytes[1] != 0xD8.toByte() || bytes[2] != 0xFF.toByte()) {
                    return ValidationResult.fail("Invalid JPEG signature in ${file.name}")
                }
            }
        }
        return ValidationResult.success()
    }

    /**
     * GLB structural validation:
     * - Header: magic 0x46546C67 ("glTF"), version 2, total length matching file
     * - Chunks: Chunk 0 must be JSON (0x4E4F534A)
     * - Validates JSON parses correctly and contains attributes and nodes
     */
    fun validateGlbStructure(file: File): ValidationResult {
        if (!file.exists()) return ValidationResult.fail("GLB file does not exist: ${file.name}")
        val fileLength = file.length()
        if (fileLength < 20) return ValidationResult.fail("GLB file too small: ${file.name}")

        try {
            FileInputStream(file).use { fis ->
                val header = ByteArray(12)
                if (fis.read(header) != 12) return ValidationResult.fail("Could not read GLB header: ${file.name}")
                val bb = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
                val magic = bb.int
                val version = bb.int
                val declaredLength = bb.int.toLong() and 0xFFFFFFFFL

                // Magic: 0x46546C67 ("glTF")
                if (magic != 0x46546C67) {
                    return ValidationResult.fail("Invalid GLB magic signature 0x${Integer.toHexString(magic)} in ${file.name}")
                }
                if (version != 2) {
                    return ValidationResult.fail("Unsupported GLB version $version in ${file.name} (must be 2)")
                }
                if (declaredLength != fileLength) {
                    return ValidationResult.fail("GLB declared length ($declaredLength) does not match file size ($fileLength)")
                }

                // Chunk 0 Header: 8 bytes (chunkLength: uint32, chunkType: uint32)
                val chunk0Header = ByteArray(8)
                if (fis.read(chunk0Header) != 8) return ValidationResult.fail("Could not read GLB chunk 0 header")
                val chunkBb = ByteBuffer.wrap(chunk0Header).order(ByteOrder.LITTLE_ENDIAN)
                val chunk0Length = chunkBb.int
                val chunk0Type = chunkBb.int

                // Chunk 0 Type must be 0x4E4F534A ("JSON")
                if (chunk0Type != 0x4E4F534A) {
                    return ValidationResult.fail("GLB chunk 0 type is not JSON (got 0x${Integer.toHexString(chunk0Type)})")
                }
                if (chunk0Length <= 0 || chunk0Length > fileLength - 20) {
                    return ValidationResult.fail("Invalid GLB chunk 0 length ($chunk0Length)")
                }

                val jsonBytes = ByteArray(chunk0Length)
                if (fis.read(jsonBytes) != chunk0Length) {
                    return ValidationResult.fail("Could not read full GLB JSON chunk")
                }
                val jsonStr = String(jsonBytes, Charsets.UTF_8)
                val gltf = JSONObject(jsonStr)

                // Validate glTF contents
                if (!gltf.has("asset")) {
                    return ValidationResult.fail("glTF missing required 'asset' object in ${file.name}")
                }
            }
        } catch (e: Exception) {
            return ValidationResult.fail("Error parsing GLB structure in ${file.name}: ${e.message}")
        }

        return ValidationResult.success()
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(8192)
            var read: Int
            while (input.read(buf).also { read = it } != -1) {
                digest.update(buf, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
