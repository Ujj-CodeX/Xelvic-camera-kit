package com.example.engine.render

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES30
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * GlbLoader: Robust parser and GPU buffer builder for glTF 2.0 binary (.glb) files.
 * Supports POSITION, NORMAL, TEXCOORD_0, uint16 indices, node hierarchy, TRS,
 * embedded textures, and skeletal/TRS animation clips with LINEAR interpolation and SLERP.
 */
class GlbLoader {

    companion object {
        private const val TAG = "GlbLoader"
        private const val GLB_MAGIC = 0x46546C67 // "glTF"
        private const val CHUNK_JSON = 0x4E4F534A // "JSON"
        private const val CHUNK_BIN = 0x004E4942  // "BIN\0"
    }

    data class GlbMaterial(
        val name: String = "",
        val baseColorFactor: FloatArray = floatArrayOf(1f, 1f, 1f, 1f),
        val metallicFactor: Float = 0.0f,
        val roughnessFactor: Float = 0.5f,
        val baseColorTextureIndex: Int = -1,
        val metallicRoughnessTextureIndex: Int = -1,
        val normalTextureIndex: Int = -1,
        val normalScale: Float = 1.0f,
        val occlusionTextureIndex: Int = -1,
        val occlusionStrength: Float = 1.0f,
        val emissiveTextureIndex: Int = -1,
        val emissiveFactor: FloatArray = floatArrayOf(0f, 0f, 0f),
        val alphaMode: String = "OPAQUE",
        val alphaCutoff: Float = 0.5f,
        val doubleSided: Boolean = false
    )

    data class GlbModel(
        val meshes: List<GlbMesh>,
        val nodes: List<GlbNode>,
        val animations: List<GlbAnimation>,
        val textures: List<Int>, // GL texture IDs
        val materials: List<GlbMaterial> = emptyList(),
        val cpuTextureBitmaps: List<Bitmap?> = emptyList()
    ) {
        fun meshesForGltfMeshIndex(gltfMeshIdx: Int): List<GlbMesh> {
            if (gltfMeshIdx < 0) return emptyList()
            val hasExplicitGltfIndices = meshes.any { it.gltfMeshIndex >= 0 }
            if (hasExplicitGltfIndices) {
                val matched = meshes.filter { it.gltfMeshIndex == gltfMeshIdx }
                if (matched.isNotEmpty()) return matched
            }
            return if (gltfMeshIdx in meshes.indices) listOf(meshes[gltfMeshIdx]) else emptyList()
        }

        fun release() {
            for (m in meshes) m.release()
            for (t in textures) {
                if (t != 0) {
                    try {
                        GLES30.glDeleteTextures(1, intArrayOf(t), 0)
                    } catch (_: Throwable) {
                    }
                }
            }
            for (bmp in cpuTextureBitmaps) {
                try {
                    if (bmp != null && !bmp.isRecycled) {
                        bmp.recycle()
                    }
                } catch (_: Throwable) {
                }
            }
        }
    }

    data class GlbMesh(
        val vao: Int,
        val vbo: Int,
        val ibo: Int,
        val indexCount: Int,
        val materialIndex: Int,
        val hasNormals: Boolean,
        val hasTexCoords: Boolean,
        val cpuPositions: FloatArray = FloatArray(0),
        val cpuIndices: ShortArray? = null,
        val cpuNormals: FloatArray = FloatArray(0),
        val cpuTexCoords: FloatArray = FloatArray(0),
        val gltfMeshIndex: Int = -1,
        val primitiveIndex: Int = 0
    ) {
        fun release() {
            try {
                if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
                if (vbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
                if (ibo != 0) GLES30.glDeleteBuffers(1, intArrayOf(ibo), 0)
            } catch (_: Throwable) {
            }
        }
    }

    data class GlbNode(
        val name: String,
        val meshIndex: Int,
        val parentIndex: Int,
        val baseTranslation: FloatArray,
        val baseRotation: FloatArray, // [qx, qy, qz, qw]
        val baseScale: FloatArray,
        // Runtime evaluated TRS
        val currentTranslation: FloatArray,
        val currentRotation: FloatArray,
        val currentScale: FloatArray,
        val localMatrix: FloatArray = FloatArray(16),
        val worldMatrix: FloatArray = FloatArray(16)
    )

    data class AnimationChannel(
        val targetNodeIndex: Int,
        val targetPath: String, // "translation", "rotation", "scale"
        val timestamps: FloatArray,
        val keyframeValues: FloatArray // 3 floats per key for trans/scale, 4 for rot
    )

    data class GlbAnimation(
        val name: String,
        val durationSec: Float,
        val channels: List<AnimationChannel>
    )

    fun load(file: File): GlbModel? {
        if (!file.exists()) return null

        try {
            val bytes = file.readBytes()
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            // Read header
            val magic = buf.int
            val version = buf.int
            val length = buf.int
            if (magic != GLB_MAGIC || version != 2) {
                Log.e(TAG, "Invalid GLB header in ${file.name}")
                return null
            }

            // Read JSON chunk
            val jsonLength = buf.int
            val jsonType = buf.int
            if (jsonType != CHUNK_JSON) {
                Log.e(TAG, "Missing JSON chunk in ${file.name}")
                return null
            }
            val jsonBytes = ByteArray(jsonLength)
            buf.get(jsonBytes)
            val jsonStr = String(jsonBytes, Charsets.UTF_8)
            val gltfJson = JSONObject(jsonStr)

            // Read BIN chunk
            var binBuffer: ByteBuffer? = null
            if (buf.hasRemaining()) {
                val binLength = buf.int
                val binType = buf.int
                if (binType == CHUNK_BIN) {
                    val binBytes = ByteArray(binLength)
                    buf.get(binBytes)
                    binBuffer = ByteBuffer.wrap(binBytes).order(ByteOrder.LITTLE_ENDIAN)
                }
            }

            return buildModel(gltfJson, binBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load GLB ${file.name}: ${e.message}", e)
            return null
        }
    }

    private fun buildModel(json: JSONObject, binBuffer: ByteBuffer?): GlbModel? {
        val bufferViews = json.optJSONArray("bufferViews")
        val accessors = json.optJSONArray("accessors")
        val meshesArr = json.optJSONArray("meshes")
        val nodesArr = json.optJSONArray("nodes")
        val imagesArr = json.optJSONArray("images")
        val gltfTexturesArr = json.optJSONArray("textures")
        val materialsArr = json.optJSONArray("materials")

        // 1. Load embedded images
        val imageTexIds = ArrayList<Int>()
        val imageBitmaps = ArrayList<Bitmap?>()
        if (imagesArr != null && binBuffer != null && bufferViews != null) {
            for (i in 0 until imagesArr.length()) {
                val imgObj = imagesArr.getJSONObject(i)
                val bvIdx = imgObj.optInt("bufferView", -1)
                if (bvIdx >= 0 && bvIdx < bufferViews.length()) {
                    val bv = bufferViews.getJSONObject(bvIdx)
                    val offset = bv.optInt("byteOffset", 0)
                    val byteLength = bv.getInt("byteLength")

                    val imgBytes = ByteArray(byteLength)
                    binBuffer.position(offset)
                    binBuffer.get(imgBytes)

                    val bmp = BitmapFactory.decodeByteArray(imgBytes, 0, byteLength)
                    if (bmp != null) {
                        val texId = uploadGlbTexture(bmp)
                        imageTexIds.add(texId)
                        imageBitmaps.add(bmp)
                    } else {
                        imageTexIds.add(0)
                        imageBitmaps.add(null)
                    }
                } else {
                    imageTexIds.add(0)
                    imageBitmaps.add(null)
                }
            }
        }

        // Map glTF textures array (which points to image `source`) or fallback to 1:1 imageTexIds
        val textureIds = ArrayList<Int>()
        val cpuTextureBitmaps = ArrayList<Bitmap?>()
        if (gltfTexturesArr != null && gltfTexturesArr.length() > 0) {
            for (t in 0 until gltfTexturesArr.length()) {
                val tObj = gltfTexturesArr.optJSONObject(t)
                val srcImgIdx = tObj?.optInt("source", t) ?: t
                val glTexId = if (srcImgIdx in imageTexIds.indices) imageTexIds[srcImgIdx] else 0
                val cpuBmp = if (srcImgIdx in imageBitmaps.indices) imageBitmaps[srcImgIdx] else null
                textureIds.add(glTexId)
                cpuTextureBitmaps.add(cpuBmp)
            }
        } else {
            textureIds.addAll(imageTexIds)
            cpuTextureBitmaps.addAll(imageBitmaps)
        }

        // 1b. Parse glTF 2.0 materials (pbrMetallicRoughness, normalTexture, occlusionTexture, emissiveTexture, emissiveFactor, doubleSided)
        val materials = ArrayList<GlbMaterial>()
        if (materialsArr != null) {
            for (m in 0 until materialsArr.length()) {
                val matObj = materialsArr.optJSONObject(m) ?: continue
                val name = matObj.optString("name", "mat_$m")
                val pbr = matObj.optJSONObject("pbrMetallicRoughness")
                val baseColor = floatArrayOf(1f, 1f, 1f, 1f)
                val bcArr = pbr?.optJSONArray("baseColorFactor")
                if (bcArr != null && bcArr.length() >= 4) {
                    baseColor[0] = bcArr.getDouble(0).toFloat().coerceIn(0f, 1f)
                    baseColor[1] = bcArr.getDouble(1).toFloat().coerceIn(0f, 1f)
                    baseColor[2] = bcArr.getDouble(2).toFloat().coerceIn(0f, 1f)
                    baseColor[3] = bcArr.getDouble(3).toFloat().coerceIn(0f, 1f)
                }
                val metallic = (pbr?.optDouble("metallicFactor", 0.0)?.toFloat() ?: 0.0f).coerceIn(0f, 1f)
                val roughness = (pbr?.optDouble("roughnessFactor", 0.5)?.toFloat() ?: 0.5f).coerceIn(0.04f, 1f)
                val baseColorTexIdx = pbr?.optJSONObject("baseColorTexture")?.optInt("index", -1) ?: -1
                val mrTexIdx = pbr?.optJSONObject("metallicRoughnessTexture")?.optInt("index", -1) ?: -1
                val normTexObj = matObj.optJSONObject("normalTexture")
                val normTexIdx = normTexObj?.optInt("index", -1) ?: -1
                val normScale = (normTexObj?.optDouble("scale", 1.0)?.toFloat() ?: 1.0f).coerceIn(0f, 4f)
                val occTexObj = matObj.optJSONObject("occlusionTexture")
                val occTexIdx = occTexObj?.optInt("index", -1) ?: -1
                val occStrength = (occTexObj?.optDouble("strength", 1.0)?.toFloat() ?: 1.0f).coerceIn(0f, 1f)
                val emTexIdx = matObj.optJSONObject("emissiveTexture")?.optInt("index", -1) ?: -1
                val emissive = floatArrayOf(0f, 0f, 0f)
                val emArr = matObj.optJSONArray("emissiveFactor")
                if (emArr != null && emArr.length() >= 3) {
                    emissive[0] = emArr.getDouble(0).toFloat().coerceAtLeast(0f)
                    emissive[1] = emArr.getDouble(1).toFloat().coerceAtLeast(0f)
                    emissive[2] = emArr.getDouble(2).toFloat().coerceAtLeast(0f)
                }
                val alphaMode = matObj.optString("alphaMode", "OPAQUE")
                val alphaCutoff = matObj.optDouble("alphaCutoff", 0.5).toFloat().coerceIn(0f, 1f)
                val doubleSided = matObj.optBoolean("doubleSided", false)
                materials.add(
                    GlbMaterial(
                        name = name,
                        baseColorFactor = baseColor,
                        metallicFactor = metallic,
                        roughnessFactor = roughness,
                        baseColorTextureIndex = baseColorTexIdx,
                        metallicRoughnessTextureIndex = mrTexIdx,
                        normalTextureIndex = normTexIdx,
                        normalScale = normScale,
                        occlusionTextureIndex = occTexIdx,
                        occlusionStrength = occStrength,
                        emissiveTextureIndex = emTexIdx,
                        emissiveFactor = emissive,
                        alphaMode = alphaMode,
                        alphaCutoff = alphaCutoff,
                        doubleSided = doubleSided
                    )
                )
            }
        }

        // 2. Build Meshes and GPU buffers
        val meshes = ArrayList<GlbMesh>()
        if (meshesArr != null && binBuffer != null && accessors != null && bufferViews != null) {
            for (m in 0 until meshesArr.length()) {
                val meshObj = meshesArr.getJSONObject(m)
                val prims = meshObj.optJSONArray("primitives") ?: continue
                for (p in 0 until prims.length()) {
                    val prim = prims.getJSONObject(p)
                    val attrs = prim.getJSONObject("attributes")

                    val posAccIdx = attrs.getInt("POSITION")
                    val posData = extractAccessorFloats(posAccIdx, accessors, bufferViews, binBuffer)

                    val normAccIdx = attrs.optInt("NORMAL", -1)
                    val normData = if (normAccIdx >= 0) extractAccessorFloats(normAccIdx, accessors, bufferViews, binBuffer) else null

                    val uvAccIdx = attrs.optInt("TEXCOORD_0", -1)
                    val uvData = if (uvAccIdx >= 0) extractAccessorFloats(uvAccIdx, accessors, bufferViews, binBuffer) else null

                    val idxAccIdx = prim.optInt("indices", -1)
                    val indicesData = if (idxAccIdx >= 0) extractAccessorIndices(idxAccIdx, accessors, bufferViews, binBuffer) else null

                    val gpuMesh = uploadGpuMesh(
                        positions = posData,
                        normals = normData,
                        uvs = uvData,
                        indices = indicesData,
                        matIdx = prim.optInt("material", 0),
                        gltfMeshIdx = m,
                        primIdx = p
                    )
                    if (gpuMesh != null) {
                        meshes.add(gpuMesh)
                    }
                }
            }
        }

        // 3. Build Nodes
        val nodes = ArrayList<GlbNode>()
        if (nodesArr != null) {
            for (n in 0 until nodesArr.length()) {
                val nodeObj = nodesArr.getJSONObject(n)
                val nodeName = nodeObj.optString("name", "node_$n")
                val meshIdx = nodeObj.optInt("mesh", -1)

                val trans = floatArrayOf(0f, 0f, 0f)
                val tArr = nodeObj.optJSONArray("translation")
                if (tArr != null && tArr.length() >= 3) {
                    trans[0] = tArr.getDouble(0).toFloat()
                    trans[1] = tArr.getDouble(1).toFloat()
                    trans[2] = tArr.getDouble(2).toFloat()
                }

                val rot = floatArrayOf(0f, 0f, 0f, 1f)
                val rArr = nodeObj.optJSONArray("rotation")
                if (rArr != null && rArr.length() >= 4) {
                    rot[0] = rArr.getDouble(0).toFloat()
                    rot[1] = rArr.getDouble(1).toFloat()
                    rot[2] = rArr.getDouble(2).toFloat()
                    rot[3] = rArr.getDouble(3).toFloat()
                }

                val scale = floatArrayOf(1f, 1f, 1f)
                val sArr = nodeObj.optJSONArray("scale")
                if (sArr != null && sArr.length() >= 3) {
                    scale[0] = sArr.getDouble(0).toFloat()
                    scale[1] = sArr.getDouble(1).toFloat()
                    scale[2] = sArr.getDouble(2).toFloat()
                }

                nodes.add(
                    GlbNode(
                        name = nodeName,
                        meshIndex = meshIdx,
                        parentIndex = -1,
                        baseTranslation = trans,
                        baseRotation = rot,
                        baseScale = scale,
                        currentTranslation = trans.clone(),
                        currentRotation = rot.clone(),
                        currentScale = scale.clone()
                    )
                )
            }
            // Resolve parentIndex from children arrays
            for (n in 0 until nodesArr.length()) {
                val childrenArr = nodesArr.getJSONObject(n).optJSONArray("children") ?: continue
                for (c in 0 until childrenArr.length()) {
                    val childIdx = childrenArr.getInt(c)
                    if (childIdx in nodes.indices) {
                        nodes[childIdx] = nodes[childIdx].copy(parentIndex = n)
                    }
                }
            }
        }

        // 4. Build Animations
        val animations = ArrayList<GlbAnimation>()
        val animsArr = json.optJSONArray("animations")
        if (animsArr != null && binBuffer != null && accessors != null && bufferViews != null) {
            for (a in 0 until animsArr.length()) {
                val animObj = animsArr.getJSONObject(a)
                val animName = animObj.optString("name", "anim_$a")
                val samplers = animObj.getJSONArray("samplers")
                val channels = animObj.getJSONArray("channels")
                val compiledChannels = ArrayList<AnimationChannel>()
                var maxDuration = 1.0f

                for (c in 0 until channels.length()) {
                    val ch = channels.getJSONObject(c)
                    val samplerIdx = ch.getInt("sampler")
                    val sampler = samplers.getJSONObject(samplerIdx)
                    val target = ch.getJSONObject("target")
                    val targetNode = target.getInt("node")
                    val targetPath = target.getString("path") // "translation", "rotation", "scale"

                    val inputAcc = sampler.getInt("input")
                    val outputAcc = sampler.getInt("output")

                    val times = extractAccessorFloats(inputAcc, accessors, bufferViews, binBuffer)
                    val values = extractAccessorFloats(outputAcc, accessors, bufferViews, binBuffer)

                    if (times.isNotEmpty()) {
                        val dur = times.last()
                        if (dur > maxDuration) maxDuration = dur
                    }

                    compiledChannels.add(AnimationChannel(targetNode, targetPath, times, values))
                }

                animations.add(GlbAnimation(animName, maxDuration, compiledChannels))
            }
        }

        return GlbModel(meshes, nodes, animations, textureIds, materials, cpuTextureBitmaps)
    }

    private fun extractAccessorFloats(accIdx: Int, accessors: org.json.JSONArray, bufferViews: org.json.JSONArray, binBuffer: ByteBuffer): FloatArray {
        val acc = accessors.getJSONObject(accIdx)
        val count = acc.getInt("count")
        val type = acc.getString("type")
        val compCount = when (type) {
            "SCALAR" -> 1
            "VEC2" -> 2
            "VEC3" -> 3
            "VEC4" -> 4
            else -> 1
        }
        val bvIdx = acc.getInt("bufferView")
        val bv = bufferViews.getJSONObject(bvIdx)
        val offset = bv.optInt("byteOffset", 0) + acc.optInt("byteOffset", 0)
        val byteStride = bv.optInt("byteStride", 0)
        val elemBytes = compCount * 4

        val totalFloats = count * compCount
        val out = FloatArray(totalFloats)
        if (byteStride > elemBytes) {
            for (elem in 0 until count) {
                binBuffer.position(offset + elem * byteStride)
                val base = elem * compCount
                for (c in 0 until compCount) {
                    out[base + c] = binBuffer.float
                }
            }
        } else {
            binBuffer.position(offset)
            for (i in 0 until totalFloats) {
                out[i] = binBuffer.float
            }
        }
        return out
    }

    private fun extractAccessorIndices(accIdx: Int, accessors: org.json.JSONArray, bufferViews: org.json.JSONArray, binBuffer: ByteBuffer): ShortArray {
        val acc = accessors.getJSONObject(accIdx)
        val count = acc.getInt("count")
        val compType = acc.getInt("componentType") // 5121 = UNSIGNED_BYTE, 5123 = UNSIGNED_SHORT, 5125 = UNSIGNED_INT
        val bvIdx = acc.getInt("bufferView")
        val bv = bufferViews.getJSONObject(bvIdx)
        val offset = bv.optInt("byteOffset", 0) + acc.optInt("byteOffset", 0)
        val byteStride = bv.optInt("byteStride", 0)
        val elemBytes = when (compType) {
            5121 -> 1
            5125 -> 4
            else -> 2
        }

        val out = ShortArray(count)
        if (byteStride > elemBytes) {
            for (i in 0 until count) {
                binBuffer.position(offset + i * byteStride)
                out[i] = when (compType) {
                    5121 -> (binBuffer.get().toInt() and 0xFF).toShort()
                    5125 -> binBuffer.int.toShort()
                    else -> binBuffer.short
                }
            }
        } else {
            binBuffer.position(offset)
            for (i in 0 until count) {
                out[i] = when (compType) {
                    5121 -> (binBuffer.get().toInt() and 0xFF).toShort()
                    5125 -> binBuffer.int.toShort()
                    else -> binBuffer.short
                }
            }
        }
        return out
    }

    private fun uploadGpuMesh(
        positions: FloatArray,
        normals: FloatArray?,
        uvs: FloatArray?,
        indices: ShortArray?,
        matIdx: Int,
        gltfMeshIdx: Int = -1,
        primIdx: Int = 0
    ): GlbMesh? {
        val vertCount = positions.size / 3
        if (vertCount == 0) return null

        // Always interleave 8 floats per vertex: Position (3) + Normal (3) + UV (2)
        val strideFloats = 8
        val interleaved = FloatArray(vertCount * strideFloats)
        var pIdx = 0; var nIdx = 0; var uIdx = 0; var outIdx = 0

        for (v in 0 until vertCount) {
            interleaved[outIdx++] = positions[pIdx++]
            interleaved[outIdx++] = positions[pIdx++]
            interleaved[outIdx++] = positions[pIdx++]
            if (normals != null && nIdx + 2 < normals.size) {
                interleaved[outIdx++] = normals[nIdx++]
                interleaved[outIdx++] = normals[nIdx++]
                interleaved[outIdx++] = normals[nIdx++]
            } else {
                interleaved[outIdx++] = 0f
                interleaved[outIdx++] = 0f
                interleaved[outIdx++] = 1f
            }
            if (uvs != null && uIdx + 1 < uvs.size) {
                interleaved[outIdx++] = uvs[uIdx++]
                interleaved[outIdx++] = uvs[uIdx++]
            } else {
                interleaved[outIdx++] = 0f
                interleaved[outIdx++] = 0f
            }
        }

        val indexCount = indices?.size ?: vertCount
        val vaos = IntArray(1)
        val vbos = IntArray(1)
        val ibos = IntArray(1)
        try {
            GLES30.glGenVertexArrays(1, vaos, 0)
            GLES30.glGenBuffers(1, vbos, 0)
            GLES30.glGenBuffers(1, ibos, 0)

            if (vaos[0] != 0 && vbos[0] != 0) {
                GLES30.glBindVertexArray(vaos[0])

                // Upload vertex buffer
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbos[0])
                val fb = ByteBuffer.allocateDirect(interleaved.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                fb.put(interleaved).position(0)
                GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, interleaved.size * 4, fb, GLES30.GL_STATIC_DRAW)

                val strideBytes = strideFloats * 4

                // Location 0: Position (offset 0)
                GLES30.glEnableVertexAttribArray(0)
                GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, strideBytes, 0)

                // Location 1: Normal (offset 12 bytes)
                GLES30.glEnableVertexAttribArray(1)
                GLES30.glVertexAttribPointer(1, 3, GLES30.GL_FLOAT, false, strideBytes, 3 * 4)

                // Location 2: UV (offset 24 bytes)
                GLES30.glEnableVertexAttribArray(2)
                GLES30.glVertexAttribPointer(2, 2, GLES30.GL_FLOAT, false, strideBytes, 6 * 4)

                if (indices != null && ibos[0] != 0) {
                    GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, ibos[0])
                    val sb = ByteBuffer.allocateDirect(indices.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
                    sb.put(indices).position(0)
                    GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, indices.size * 2, sb, GLES30.GL_STATIC_DRAW)
                }

                GLES30.glBindVertexArray(0)
            }
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
        return GlbMesh(
            vao = vaos[0],
            vbo = vbos[0],
            ibo = ibos[0],
            indexCount = indexCount,
            materialIndex = matIdx,
            hasNormals = normals != null,
            hasTexCoords = uvs != null,
            cpuPositions = positions,
            cpuIndices = indices,
            cpuNormals = normals ?: FloatArray(0),
            cpuTexCoords = uvs ?: FloatArray(0),
            gltfMeshIndex = gltfMeshIdx,
            primitiveIndex = primIdx
        )
    }

    private fun uploadGlbTexture(bitmap: Bitmap): Int {
        return try {
            val textures = IntArray(1)
            GLES30.glGenTextures(1, textures, 0)
            val texId = textures[0]
            if (texId == 0) return 0

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texId)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR_MIPMAP_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_REPEAT)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_REPEAT)

            android.opengl.GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
            GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
            texId
        } catch (_: Throwable) {
            0
        }
    }
}
