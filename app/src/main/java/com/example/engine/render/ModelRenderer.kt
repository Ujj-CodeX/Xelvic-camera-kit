package com.example.engine.render

import android.opengl.GLES30
import android.opengl.Matrix
import com.example.engine.scene.*
import com.example.engine.signal.FacePoseAdapter
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * ModelRenderer: Renders 3D glTF models with materials, camera-space lighting,
 * and skeletal/TRS animation playback (LINEAR & SLERP interpolation) (README Section 5.9, 5.11 & 21 Step 4).
 */
class ModelRenderer {

    private val tempMvp = FloatArray(16)
    private val tempModelView = FloatArray(16)
    private val scratchAnimQ = FloatArray(4)
    private val defaultAmbient = floatArrayOf(0.4f, 0.4f, 0.4f)
    private val defaultLightDirs = floatArrayOf(0f, 0.6f, 0.8f, 0f, 0f, 1f)
    private val defaultLightColors = floatArrayOf(1f, 1f, 1f, 0f, 0f, 0f)
    private val defaultLightIntensities = floatArrayOf(0.7f, 0f)

    var lastFrontFaceMode: Int = GLES30.GL_CCW
        private set

    var lastBoundTemplate: String = "unlit@1"
        private set

    var lastBoundMetallic: Float = 0.0f
        private set

    var lastBoundRoughness: Float = 0.45f
        private set

    val lastBoundColor: FloatArray = floatArrayOf(1f, 1f, 1f, 1f)

    var lastDrawnPrimitiveCount: Int = 0
        private set

    var lastBoundAlphaMode: Int = 0
        private set

    var lastBoundAlphaCutoff: Float = 0.5f
        private set

    var lastBoundEmissiveMapTexId: Int = 0
        private set

    val lastModelViewMatrix: FloatArray = FloatArray(16)

    val lastResolvedSubnodeMaterials: MutableMap<String, String> = LinkedHashMap()

    companion object {
        fun resolvePrimaryModelMaterial(
            node: CompiledNode,
            materials: Map<String, CompiledMaterial>
        ): CompiledMaterial? {
            node.materialId?.let { materials[it] }?.let { return it }
            node.tags.firstNotNullOfOrNull { materials[it] }?.let { return it }
            if (node.type != "model" || materials.isEmpty()) return null

            val nodeIdNorm = node.id.lowercase()
            val assetIdNorm = (node.modelAssetId ?: "").lowercase()
            for ((matKey, matVal) in materials) {
                if (matVal.template == "faceTexture@1") continue
                val cleanKey = matKey.lowercase().removePrefix("mat_").removeSuffix("_mat")
                if (cleanKey.length >= 3 && (nodeIdNorm.contains(cleanKey) || assetIdNorm.contains(cleanKey))) {
                    return matVal
                }
            }
            val lit3dCandidates = materials.values.filter {
                it.template == "pbr@1" || it.template == "lambert@1"
            }
            if (lit3dCandidates.size == 1) {
                return lit3dCandidates[0]
            }
            return null
        }

        fun resolveSubnodeMaterial(
            node: CompiledNode,
            subnodeName: String?,
            glbMatName: String?,
            primaryMaterial: CompiledMaterial?,
            materials: Map<String, CompiledMaterial>
        ): CompiledMaterial? {
            if (materials.isEmpty()) return primaryMaterial
            val subNorm = subnodeName?.lowercase() ?: ""
            val matNorm = glbMatName?.lowercase() ?: ""
            if (subNorm.isNotEmpty() || matNorm.isNotEmpty()) {
                for (tag in node.tags) {
                    val tagMat = materials[tag] ?: continue
                    val tagLower = tag.lowercase()
                    if (tagLower.isNotEmpty() && (subNorm.contains(tagLower) || matNorm.contains(tagLower))) {
                        return tagMat
                    }
                }
                for ((key, candidate) in materials) {
                    if (candidate.template == "faceTexture@1") continue
                    val cleanKey = key.lowercase().removePrefix("mat_").removeSuffix("_mat")
                    if (cleanKey.length >= 3 && (subNorm.contains(cleanKey) || matNorm.contains(cleanKey))) {
                        return candidate
                    }
                }
            }
            return primaryMaterial
        }

        fun hasActiveGlbPbr(
            model: GlbLoader.GlbModel,
            activeMat: CompiledMaterial?,
            glbMat: GlbLoader.GlbMaterial? = null
        ): Boolean {
            if (activeMat?.template == "unlit@1") return false
            if (activeMat?.template == "pbr@1") return true
            if (model.materials.isEmpty()) return false
            if (activeMat != null && activeMat.template == "lambert@1" &&
                (activeMat.hasExplicitRoughness || activeMat.hasExplicitMetallic)
            ) {
                val matsToCheck = if (glbMat != null) listOf(glbMat) else model.materials
                val hasNonMattePbr = matsToCheck.any {
                    it.metallicFactor > 0.001f ||
                        it.roughnessFactor < 0.99f ||
                        it.normalTextureIndex >= 0 ||
                        it.metallicRoughnessTextureIndex >= 0 ||
                        it.occlusionTextureIndex >= 0 ||
                        it.emissiveTextureIndex >= 0
                }
                if (!hasNonMattePbr) return false
            }
            return true
        }
    }

    @JvmOverloads
    fun render(
        model: GlbLoader.GlbModel,
        node: CompiledNode,
        worldMatrix: FloatArray,
        viewProjectionMatrix: FloatArray,
        material: CompiledMaterial?,
        lighting: CompiledLighting?,
        opacity: Float,
        programs: MaterialRegistry,
        textureId: Int,
        clips: List<BehaviorRunner.CompiledClip> = emptyList(),
        activeTier: String = "A",
        frontFacing: Boolean = false,
        normalMapTexId: Int = 0,
        metallicRoughnessMapTexId: Int = 0,
        occlusionMapTexId: Int = 0,
        emissiveMapTexId: Int = 0,
        allMaterials: Map<String, CompiledMaterial> = emptyMap()
    ) {
        lastDrawnPrimitiveCount = 0
        lastResolvedSubnodeMaterials.clear()
        val primaryMaterial = material ?: resolvePrimaryModelMaterial(node, allMaterials)
        // Front-camera mirror via negative proj[0] flips triangle winding in NDC (CCW -> CW);
        // set glFrontFace accordingly so backface culling culls true backfaces on both cameras.
        val frontFaceMode = FacePoseAdapter.glFrontFaceForCamera(frontFacing)
        lastFrontFaceMode = frontFaceMode
        try {
            GLES30.glFrontFace(frontFaceMode)
        } catch (_: Throwable) {
        }

        // Reset node transforms to base
        for (glbNode in model.nodes) {
            System.arraycopy(glbNode.baseTranslation, 0, glbNode.currentTranslation, 0, 3)
            System.arraycopy(glbNode.baseRotation, 0, glbNode.currentRotation, 0, 4)
            System.arraycopy(glbNode.baseScale, 0, glbNode.currentScale, 0, 3)
        }

        // Apply active animation clips
        for (clip in clips) {
            val anim = model.animations.find { it.name == clip.animationName || it.name == clip.id } ?: continue
            if (anim.durationSec > 0.01f) {
                clip.durationSec = anim.durationSec
            }
            if (!clip.isPlaying) continue
            val time = clip.currentTimeSec

            for (channel in anim.channels) {
                if (channel.targetNodeIndex in model.nodes.indices) {
                    val targetGlbNode = model.nodes[channel.targetNodeIndex]
                    sampleChannel(channel, time, targetGlbNode)
                }
            }
        }

        // Evaluate local matrices and combine with parent or root worldMatrix
        for (glbNode in model.nodes) {
            computeNodeLocalMatrix(glbNode)
            val pIdx = glbNode.parentIndex
            if (pIdx in model.nodes.indices) {
                val parentMat = model.nodes[pIdx].worldMatrix
                Matrix.multiplyMM(glbNode.worldMatrix, 0, parentMat, 0, glbNode.localMatrix, 0)
            } else {
                Matrix.multiplyMM(glbNode.worldMatrix, 0, worldMatrix, 0, glbNode.localMatrix, 0)
            }
        }

        fun prepareModelViewMatrix(srcWorld: FloatArray) {
            System.arraycopy(srcWorld, 0, tempModelView, 0, 16)
            if (frontFacing) {
                // Mirror X in view space to match the front-camera X mirror in viewProjectionMatrix
                tempModelView[0] = -tempModelView[0]
                tempModelView[4] = -tempModelView[4]
                tempModelView[8] = -tempModelView[8]
                tempModelView[12] = -tempModelView[12]
            }
            System.arraycopy(tempModelView, 0, lastModelViewMatrix, 0, 16)
        }

        fun drawPrimitiveWithMaterial(
            mesh: GlbLoader.GlbMesh,
            activeMat: CompiledMaterial?,
            mvpMat: FloatArray,
            mvMat: FloatArray
        ) {
            val glbMat = if (mesh.materialIndex in model.materials.indices) model.materials[mesh.materialIndex] else null
            val useGlbPbr = hasActiveGlbPbr(model, activeMat, glbMat)
            val effectiveTemplate = when {
                activeMat?.template == "pbr@1" || useGlbPbr -> "pbr@1"
                activeMat?.template == "lambert@1" -> "lambert@1"
                else -> "unlit@1"
            }
            lastBoundTemplate = effectiveTemplate

            val prog = when (effectiveTemplate) {
                "pbr@1" -> if (programs.programPbr != 0) programs.programPbr else programs.programLambert
                "lambert@1" -> programs.programLambert
                else -> programs.programUnlit
            }
            try {
                GLES30.glUseProgram(prog)
            } catch (_: Throwable) {
            }

            val matCol = activeMat?.color
            val glbCol = glbMat?.baseColorFactor
            val baseR = (matCol?.getOrElse(0) { 1f } ?: 1f) * (glbCol?.getOrElse(0) { 1f } ?: 1f)
            val baseG = (matCol?.getOrElse(1) { 1f } ?: 1f) * (glbCol?.getOrElse(1) { 1f } ?: 1f)
            val baseB = (matCol?.getOrElse(2) { 1f } ?: 1f) * (glbCol?.getOrElse(2) { 1f } ?: 1f)
            val matOpacity = activeMat?.opacity ?: 1f
            val nodeBaseOpacity = (node.baseOpacity.coerceIn(0f, 1f)).let { if (it > 0f) opacity / (material?.opacity ?: 1f).coerceAtLeast(0.001f) else opacity }
            val effectiveNodeOpacity = if (activeMat != null && activeMat !== material) {
                (nodeBaseOpacity * matOpacity).coerceIn(0f, 1f)
            } else {
                opacity.coerceIn(0f, 1f)
            }
            val baseA = ((glbCol?.getOrElse(3) { 1f } ?: 1f) * effectiveNodeOpacity).coerceIn(0f, 1f)
            lastBoundColor[0] = baseR
            lastBoundColor[1] = baseG
            lastBoundColor[2] = baseB
            lastBoundColor[3] = baseA

            val rawGlbAlphaMode = glbMat?.alphaMode?.uppercase() ?: "OPAQUE"
            val rawMatBlend = activeMat?.blend?.lowercase() ?: "opaque"
            val alphaModeInt = when {
                rawGlbAlphaMode == "MASK" || rawMatBlend == "mask" -> 1
                rawGlbAlphaMode == "BLEND" || rawMatBlend == "alpha" || baseA < 0.999f -> 2
                else -> 0
            }
            val alphaCutoff = (glbMat?.alphaCutoff ?: 0.5f).coerceIn(0.01f, 0.99f)
            lastBoundAlphaMode = alphaModeInt
            lastBoundAlphaCutoff = alphaCutoff

            // Configure blend and depth state per primitive
            when {
                rawMatBlend == "additive" -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE)
                }
                rawMatBlend == "multiply" -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_DST_COLOR, GLES30.GL_ONE_MINUS_SRC_ALPHA)
                }
                alphaModeInt == 2 -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)
                }
                else -> {
                    GLES30.glDisable(GLES30.GL_BLEND)
                }
            }

            val depthTest = activeMat?.depthTest ?: true
            if (depthTest) GLES30.glEnable(GLES30.GL_DEPTH_TEST) else GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            val depthWrite = if (alphaModeInt == 2 && rawGlbAlphaMode == "BLEND" && activeMat == null) {
                false
            } else {
                activeMat?.depthWrite ?: (alphaModeInt != 2)
            }
            GLES30.glDepthMask(depthWrite)

            val isDoubleSided = (activeMat?.doubleSided ?: false) || (glbMat?.doubleSided ?: (activeMat == null))
            if (isDoubleSided) {
                GLES30.glDisable(GLES30.GL_CULL_FACE)
            } else if (activeMat?.cull != false) {
                GLES30.glEnable(GLES30.GL_CULL_FACE)
                GLES30.glCullFace(GLES30.GL_BACK)
            }

            val isLit = (effectiveTemplate == "lambert@1" || effectiveTemplate == "pbr@1")
            val dblLoc = GLES30.glGetUniformLocation(prog, "uDoubleSided")
            if (isLit) {
                val ambLoc = GLES30.glGetUniformLocation(prog, "uAmbient")
                val lightCountLoc = GLES30.glGetUniformLocation(prog, "uLightCount")
                val lightDirLoc = GLES30.glGetUniformLocation(prog, "uLightDir[0]")
                    .let { if (it != -1) it else GLES30.glGetUniformLocation(prog, "uLightDir") }
                val lightColLoc = GLES30.glGetUniformLocation(prog, "uLightColor[0]")
                    .let { if (it != -1) it else GLES30.glGetUniformLocation(prog, "uLightColor") }
                val lightIntLoc = GLES30.glGetUniformLocation(prog, "uLightIntensity[0]")
                    .let { if (it != -1) it else GLES30.glGetUniformLocation(prog, "uLightIntensity") }
                val sheenLoc = GLES30.glGetUniformLocation(prog, "uFresnelSheen")

                val lightCount = lighting?.lightCount?.coerceIn(0, CompiledLighting.MAX_LIGHTS) ?: 1
                val amb = lighting?.ambient ?: defaultAmbient
                val lDirs = lighting?.lightDirs ?: defaultLightDirs
                val lCols = lighting?.lightColors ?: defaultLightColors
                val lInts = lighting?.lightIntensities ?: defaultLightIntensities

                if (ambLoc != -1) GLES30.glUniform3fv(ambLoc, 1, amb, 0)
                if (lightCountLoc != -1) GLES30.glUniform1i(lightCountLoc, lightCount)
                if (lightDirLoc != -1) GLES30.glUniform3fv(lightDirLoc, CompiledLighting.MAX_LIGHTS, lDirs, 0)
                if (lightColLoc != -1) GLES30.glUniform3fv(lightColLoc, CompiledLighting.MAX_LIGHTS, lCols, 0)
                if (lightIntLoc != -1) GLES30.glUniform1fv(lightIntLoc, CompiledLighting.MAX_LIGHTS, lInts, 0)
                if (dblLoc != -1) GLES30.glUniform1i(dblLoc, if (isDoubleSided) 1 else 0)
                if (sheenLoc != -1) GLES30.glUniform1f(sheenLoc, activeMat?.fresnelSheen ?: 0.15f)
            }

            val uColorLoc = GLES30.glGetUniformLocation(prog, "uColor")
            val uHasTexLoc = GLES30.glGetUniformLocation(prog, "uHasTexture")
            val uMvpLoc = GLES30.glGetUniformLocation(prog, "uMVP")
            val uMvLoc = GLES30.glGetUniformLocation(prog, "uModelView")
            val uTexLoc = GLES30.glGetUniformLocation(prog, "uTexture")
            val uUvRectLoc = GLES30.glGetUniformLocation(prog, "uUvRect")
            val uAlphaModeLoc = GLES30.glGetUniformLocation(prog, "uAlphaMode")
            val uAlphaCutoffLoc = GLES30.glGetUniformLocation(prog, "uAlphaCutoff")

            if (uUvRectLoc != -1) GLES30.glUniform4f(uUvRectLoc, 0f, 0f, 1f, 1f)
            if (uColorLoc != -1) GLES30.glUniform4f(uColorLoc, baseR, baseG, baseB, baseA)
            if (uAlphaModeLoc != -1) GLES30.glUniform1i(uAlphaModeLoc, alphaModeInt)
            if (uAlphaCutoffLoc != -1) GLES30.glUniform1f(uAlphaCutoffLoc, alphaCutoff)
            if (uMvpLoc != -1) GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, mvpMat, 0)
            if (uMvLoc != -1) GLES30.glUniformMatrix4fv(uMvLoc, 1, false, mvMat, 0)

            if (effectiveTemplate == "pbr@1") {
                val uMetallicLoc = GLES30.glGetUniformLocation(prog, "uMetallic")
                val uRoughnessLoc = GLES30.glGetUniformLocation(prog, "uRoughness")
                val uNormScaleLoc = GLES30.glGetUniformLocation(prog, "uNormalScale")
                val uOccStrengthLoc = GLES30.glGetUniformLocation(prog, "uOcclusionStrength")
                val uEmissiveLoc = GLES30.glGetUniformLocation(prog, "uEmissive")
                val uHasMrMapLoc = GLES30.glGetUniformLocation(prog, "uHasMetallicRoughnessMap")
                val uMrMapLoc = GLES30.glGetUniformLocation(prog, "uMetallicRoughnessMap")
                val uHasNormMapLoc = GLES30.glGetUniformLocation(prog, "uHasNormalMap")
                val uNormMapLoc = GLES30.glGetUniformLocation(prog, "uNormalMap")
                val uHasOccMapLoc = GLES30.glGetUniformLocation(prog, "uHasOcclusionMap")
                val uOccMapLoc = GLES30.glGetUniformLocation(prog, "uOcclusionMap")
                val uHasEmMapLoc = GLES30.glGetUniformLocation(prog, "uHasEmissiveMap")
                val uEmMapLoc = GLES30.glGetUniformLocation(prog, "uEmissiveMap")

                val met = when {
                    activeMat != null && activeMat.hasExplicitMetallic -> activeMat.metallic
                    glbMat != null -> glbMat.metallicFactor
                    activeMat != null -> activeMat.metallic
                    else -> 0.0f
                }
                val rough = when {
                    activeMat != null && activeMat.hasExplicitRoughness -> activeMat.roughness
                    glbMat != null -> glbMat.roughnessFactor
                    activeMat != null -> activeMat.roughness
                    else -> 0.45f
                }
                val em = when {
                    activeMat != null && activeMat.hasExplicitEmissive -> activeMat.emissive
                    glbMat != null -> glbMat.emissiveFactor
                    activeMat != null -> activeMat.emissive
                    else -> floatArrayOf(0f, 0f, 0f)
                }
                val normScale = glbMat?.normalScale ?: activeMat?.normalScale ?: 1.0f
                val occStrength = glbMat?.occlusionStrength ?: activeMat?.occlusionStrength ?: 1.0f

                lastBoundMetallic = met
                lastBoundRoughness = rough
                if (uMetallicLoc != -1) GLES30.glUniform1f(uMetallicLoc, met)
                if (uRoughnessLoc != -1) GLES30.glUniform1f(uRoughnessLoc, rough)
                if (uNormScaleLoc != -1) GLES30.glUniform1f(uNormScaleLoc, normScale)
                if (uOccStrengthLoc != -1) GLES30.glUniform1f(uOccStrengthLoc, occStrength)
                if (uEmissiveLoc != -1) GLES30.glUniform3f(uEmissiveLoc, em[0], em[1], em[2])

                val mrTex = when {
                    metallicRoughnessMapTexId != 0 -> metallicRoughnessMapTexId
                    glbMat != null && glbMat.metallicRoughnessTextureIndex in model.textures.indices ->
                        model.textures[glbMat.metallicRoughnessTextureIndex]
                    else -> 0
                }
                if (uHasMrMapLoc != -1) GLES30.glUniform1i(uHasMrMapLoc, if (mrTex != 0) 1 else 0)
                if (mrTex != 0 && uMrMapLoc != -1) {
                    GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
                    GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, mrTex)
                    GLES30.glUniform1i(uMrMapLoc, 1)
                }

                val normTex = when {
                    normalMapTexId != 0 -> normalMapTexId
                    glbMat != null && glbMat.normalTextureIndex in model.textures.indices ->
                        model.textures[glbMat.normalTextureIndex]
                    else -> 0
                }
                if (uHasNormMapLoc != -1) GLES30.glUniform1i(uHasNormMapLoc, if (normTex != 0) 1 else 0)
                if (normTex != 0 && uNormMapLoc != -1) {
                    GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
                    GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, normTex)
                    GLES30.glUniform1i(uNormMapLoc, 2)
                }

                val occTex = when {
                    occlusionMapTexId != 0 -> occlusionMapTexId
                    glbMat != null && glbMat.occlusionTextureIndex in model.textures.indices ->
                        model.textures[glbMat.occlusionTextureIndex]
                    else -> 0
                }
                if (uHasOccMapLoc != -1) GLES30.glUniform1i(uHasOccMapLoc, if (occTex != 0) 1 else 0)
                if (occTex != 0 && uOccMapLoc != -1) {
                    GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
                    GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, occTex)
                    GLES30.glUniform1i(uOccMapLoc, 3)
                }

                val emTex = when {
                    emissiveMapTexId != 0 -> emissiveMapTexId
                    glbMat != null && glbMat.emissiveTextureIndex in model.textures.indices ->
                        model.textures[glbMat.emissiveTextureIndex]
                    else -> 0
                }
                lastBoundEmissiveMapTexId = emTex
                if (uHasEmMapLoc != -1) GLES30.glUniform1i(uHasEmMapLoc, if (emTex != 0) 1 else 0)
                if (emTex != 0 && uEmMapLoc != -1) {
                    GLES30.glActiveTexture(GLES30.GL_TEXTURE4)
                    GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, emTex)
                    GLES30.glUniform1i(uEmMapLoc, 4)
                }
            }

            val meshTex = when {
                textureId != 0 -> textureId
                glbMat != null && glbMat.baseColorTextureIndex in model.textures.indices ->
                    model.textures[glbMat.baseColorTextureIndex]
                mesh.materialIndex in model.textures.indices -> model.textures[mesh.materialIndex]
                model.textures.isNotEmpty() -> model.textures[0]
                else -> 0
            }
            if (uHasTexLoc != -1) GLES30.glUniform1i(uHasTexLoc, if (meshTex != 0) 1 else 0)
            if (meshTex != 0 && uTexLoc != -1) {
                GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, meshTex)
                GLES30.glUniform1i(uTexLoc, 0)
            }

            GLES30.glBindVertexArray(mesh.vao)
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, mesh.indexCount, GLES30.GL_UNSIGNED_SHORT, 0)
            GLES30.glBindVertexArray(0)
            lastDrawnPrimitiveCount++
        }

        // Draw meshes with LOD tier filtering, multi-primitive glTF mesh mapping, per-subnode material matching,
        // and two-pass ordering (opaque primitives first, alpha-blended glass/lenses second)
        if (model.nodes.isNotEmpty()) {
            for (blendPass in 0..1) {
                for (glbNode in model.nodes) {
                    if (!isGlbNodeVisibleForTier(glbNode.name, activeTier)) {
                        continue
                    }
                    val primitives = model.meshesForGltfMeshIndex(glbNode.meshIndex)
                    if (primitives.isEmpty()) continue

                    for (mesh in primitives) {
                        val glbMat = if (mesh.materialIndex in model.materials.indices) model.materials[mesh.materialIndex] else null
                        val subMat = resolveSubnodeMaterial(node, glbNode.name, glbMat?.name, primaryMaterial, allMaterials)
                        val isTranslucentPrim = (subMat?.blend == "alpha" || subMat?.blend == "additive" || subMat?.blend == "multiply") ||
                            glbMat?.alphaMode.equals("BLEND", ignoreCase = true) ||
                            ((glbMat?.baseColorFactor?.getOrElse(3) { 1f } ?: 1f) * (subMat?.opacity ?: 1f) < 0.999f)
                        if ((blendPass == 0 && isTranslucentPrim) || (blendPass == 1 && !isTranslucentPrim)) {
                            continue
                        }
                        if (glbNode.name.isNotEmpty() && subMat != null) {
                            lastResolvedSubnodeMaterials[glbNode.name] = subMat.id
                        }
                        Matrix.multiplyMM(tempMvp, 0, viewProjectionMatrix, 0, glbNode.worldMatrix, 0)
                        prepareModelViewMatrix(glbNode.worldMatrix)
                        drawPrimitiveWithMaterial(mesh, subMat, tempMvp, tempModelView)
                    }
                }
            }
        } else {
            Matrix.multiplyMM(tempMvp, 0, viewProjectionMatrix, 0, worldMatrix, 0)
            prepareModelViewMatrix(worldMatrix)
            for (blendPass in 0..1) {
                for (mesh in model.meshes) {
                    val glbMat = if (mesh.materialIndex in model.materials.indices) model.materials[mesh.materialIndex] else null
                    val subMat = resolveSubnodeMaterial(node, null, glbMat?.name, primaryMaterial, allMaterials)
                    val isTranslucentPrim = (subMat?.blend == "alpha" || subMat?.blend == "additive" || subMat?.blend == "multiply") ||
                        glbMat?.alphaMode.equals("BLEND", ignoreCase = true) ||
                        ((glbMat?.baseColorFactor?.getOrElse(3) { 1f } ?: 1f) * (subMat?.opacity ?: 1f) < 0.999f)
                    if ((blendPass == 0 && isTranslucentPrim) || (blendPass == 1 && !isTranslucentPrim)) {
                        continue
                    }
                    drawPrimitiveWithMaterial(mesh, subMat, tempMvp, tempModelView)
                }
            }
        }
        try {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE4)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        } catch (_: Throwable) {
        }
    }

    private fun isGlbNodeVisibleForTier(nodeName: String, tier: String): Boolean {
        val lower = nodeName.lowercase()
        return when (tier.uppercase()) {
            "A" -> !lower.contains("lod_b") && !lower.contains("lod_c")
            "B" -> !lower.contains("lod_a") && !lower.contains("lod_c")
            "C" -> !lower.contains("lod_a") && !lower.contains("lod_b")
            else -> true
        }
    }

    private fun sampleChannel(channel: GlbLoader.AnimationChannel, time: Float, node: GlbLoader.GlbNode) {
        val times = channel.timestamps
        val vals = channel.keyframeValues
        if (times.isEmpty() || vals.isEmpty()) return

        if (time <= times[0]) {
            applyKeyframe(channel.targetPath, vals, 0, node)
            return
        }
        if (time >= times[times.size - 1]) {
            val lastIdx = times.size - 1
            val compSize = if (channel.targetPath == "rotation") 4 else 3
            applyKeyframe(channel.targetPath, vals, lastIdx * compSize, node)
            return
        }

        // Binary search or linear scan for segment
        var k = 0
        while (k < times.size - 1 && times[k + 1] < time) {
            k++
        }

        val t0 = times[k]
        val t1 = times[k + 1]
        val factor = if (t1 > t0) ((time - t0) / (t1 - t0)).coerceIn(0f, 1f) else 0f

        when (channel.targetPath) {
            "translation" -> {
                val idx0 = k * 3; val idx1 = (k + 1) * 3
                node.currentTranslation[0] = vals[idx0] + (vals[idx1] - vals[idx0]) * factor
                node.currentTranslation[1] = vals[idx0 + 1] + (vals[idx1 + 1] - vals[idx0 + 1]) * factor
                node.currentTranslation[2] = vals[idx0 + 2] + (vals[idx1 + 2] - vals[idx0 + 2]) * factor
            }
            "scale" -> {
                val idx0 = k * 3; val idx1 = (k + 1) * 3
                node.currentScale[0] = (vals[idx0] + (vals[idx1] - vals[idx0]) * factor).coerceAtLeast(0.001f)
                node.currentScale[1] = (vals[idx0 + 1] + (vals[idx1 + 1] - vals[idx0 + 1]) * factor).coerceAtLeast(0.001f)
                node.currentScale[2] = (vals[idx0 + 2] + (vals[idx1 + 2] - vals[idx0 + 2]) * factor).coerceAtLeast(0.001f)
            }
            "rotation" -> {
                val idx0 = k * 4; val idx1 = (k + 1) * 4
                slerp(vals, idx0, vals, idx1, factor, node.currentRotation)
            }
        }
    }

    private fun applyKeyframe(path: String, vals: FloatArray, offset: Int, node: GlbLoader.GlbNode) {
        when (path) {
            "translation" -> {
                node.currentTranslation[0] = vals[offset]
                node.currentTranslation[1] = vals[offset + 1]
                node.currentTranslation[2] = vals[offset + 2]
            }
            "scale" -> {
                node.currentScale[0] = vals[offset].coerceAtLeast(0.001f)
                node.currentScale[1] = vals[offset + 1].coerceAtLeast(0.001f)
                node.currentScale[2] = vals[offset + 2].coerceAtLeast(0.001f)
            }
            "rotation" -> {
                node.currentRotation[0] = vals[offset]
                node.currentRotation[1] = vals[offset + 1]
                node.currentRotation[2] = vals[offset + 2]
                node.currentRotation[3] = vals[offset + 3]
            }
        }
    }

    private fun slerp(q0: FloatArray, off0: Int, q1: FloatArray, off1: Int, t: Float, out: FloatArray) {
        var cosHalfTheta = q0[off0] * q1[off1] + q0[off0 + 1] * q1[off1 + 1] + q0[off0 + 2] * q1[off1 + 2] + q0[off0 + 3] * q1[off1 + 3]
        var sign = 1.0f
        if (cosHalfTheta < 0f) {
            cosHalfTheta = -cosHalfTheta
            sign = -1.0f
        }
        if (cosHalfTheta >= 1.0f - 1e-5f) {
            out[0] = q0[off0] + (q1[off1] * sign - q0[off0]) * t
            out[1] = q0[off0 + 1] + (q1[off1 + 1] * sign - q0[off0 + 1]) * t
            out[2] = q0[off0 + 2] + (q1[off1 + 2] * sign - q0[off0 + 2]) * t
            out[3] = q0[off0 + 3] + (q1[off1 + 3] * sign - q0[off0 + 3]) * t
        } else {
            val halfTheta = acos(cosHalfTheta.coerceIn(-1f, 1f))
            val sinHalfTheta = sqrt(1.0f - cosHalfTheta * cosHalfTheta)
            val ratioA = sin((1f - t) * halfTheta) / sinHalfTheta
            val ratioB = sin(t * halfTheta) / sinHalfTheta * sign
            out[0] = q0[off0] * ratioA + q1[off1] * ratioB
            out[1] = q0[off0 + 1] * ratioA + q1[off1 + 1] * ratioB
            out[2] = q0[off0 + 2] * ratioA + q1[off1 + 2] * ratioB
            out[3] = q0[off0 + 3] * ratioA + q1[off1 + 3] * ratioB
        }
        // Normalize
        val len = sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
        if (len > 1e-6f) {
            out[0] /= len; out[1] /= len; out[2] /= len; out[3] /= len
        }
    }

    private fun computeNodeLocalMatrix(node: GlbLoader.GlbNode) {
        val t = node.currentTranslation
        val q = node.currentRotation
        val s = node.currentScale
        val m = node.localMatrix

        val qx = q[0]; val qy = q[1]; val qz = q[2]; val qw = q[3]
        val sx = s[0].coerceAtLeast(0.001f)
        val sy = s[1].coerceAtLeast(0.001f)
        val sz = s[2].coerceAtLeast(0.001f)

        val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
        val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
        val wx = qw * qx; val wy = qw * qy; val wz = qw * qz

        m[0] = (1f - 2f * (yy + zz)) * sx
        m[1] = (2f * (xy + wz)) * sx
        m[2] = (2f * (xz - wy)) * sx
        m[3] = 0f

        m[4] = (2f * (xy - wz)) * sy
        m[5] = (1f - 2f * (xx + zz)) * sy
        m[6] = (2f * (yz + wx)) * sy
        m[7] = 0f

        m[8] = (2f * (xz + wy)) * sz
        m[9] = (2f * (yz - wx)) * sz
        m[10] = (1f - 2f * (xx + yy)) * sz
        m[11] = 0f

        m[12] = t[0]
        m[13] = t[1]
        m[14] = t[2]
        m[15] = 1f
    }
}
