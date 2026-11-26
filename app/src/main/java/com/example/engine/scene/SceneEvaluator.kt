package com.example.engine.scene

import android.opengl.Matrix
import com.example.engine.signal.FaceState
import kotlin.math.cos
import kotlin.math.sin

/**
 * SceneEvaluator computes the 4x4 model-view matrices for all nodes in the scene.
 * Resolves anchors, parent transforms, rotationFollow, and scales by FW.
 * Preallocates memory to guarantee zero frame-loop allocations.
 */
class SceneEvaluator(val maxNodes: Int = 64) {

    // 16 floats per node matrix
    val worldMatrices = FloatArray(maxNodes * 16)
    val isNodeVisible = BooleanArray(maxNodes)

    // Temporary scratch buffers
    private val anchorPos = FloatArray(3)
    private val localMatrix = FloatArray(16)
    private val trsMatrix = FloatArray(16)
    private val tempMat = FloatArray(16)
    private val headRotationMat = FloatArray(16)

    fun evaluate(
        effect: CompiledEffect,
        faceState: FaceState,
        slots: PropertySlots
    ) {
        val nodes = effect.nodes
        val count = nodes.size.coerceAtMost(maxNodes)
        val fw = faceState.faceWidth.coerceAtLeast(0.01f)
        val pose = faceState.poseMatrix

        // Extract pure head rotation matrix from pose matrix
        headRotationMat[0] = pose[0]; headRotationMat[1] = pose[1]; headRotationMat[2] = pose[2]; headRotationMat[3] = 0f
        headRotationMat[4] = pose[4]; headRotationMat[5] = pose[5]; headRotationMat[6] = pose[6]; headRotationMat[7] = 0f
        headRotationMat[8] = pose[8]; headRotationMat[9] = pose[9]; headRotationMat[10] = pose[10]; headRotationMat[11] = 0f
        headRotationMat[12] = 0f; headRotationMat[13] = 0f; headRotationMat[14] = 0f; headRotationMat[15] = 1f

        val tierOverride = effect.tierOverrides[effect.activeTier]

        for (i in 0 until count) {
            val node = nodes[i]

            // Check visibility including active tier show/hide overrides
            val baseVisInTier = com.example.engine.governor.TierPolicy.isNodeVisibleInTier(i, node.baseVisible, tierOverride)
            val slotVis = slots.visible[i]
            val isVis = baseVisInTier && (slotVis > 0.5f) && (slots.opacity[i] > 0.001f)
            isNodeVisible[i] = isVis
            if (!isVis) continue

            // 1. Resolve Anchor
            if (node.anchor != null) {
                effect.anchorResolver.resolveAnchor(node.anchor, faceState, anchorPos)
            } else {
                anchorPos[0] = 0f; anchorPos[1] = 0f; anchorPos[2] = 0f
            }

            // 2. Compute local TRS with property slots (position converted via FacePoseAdapter; rotation/scale unmirrored)
            val localX = node.basePosition[0] + anchorPos[0] + slots.position[i * 3]
            val localY = node.basePosition[1] + anchorPos[1] + slots.position[i * 3 + 1]
            val localZ = node.basePosition[2] + anchorPos[2] + slots.position[i * 3 + 2]
            val px = com.example.engine.signal.FacePoseAdapter.faceLocalXToPoseMeters(localX, fw)
            val py = localY * fw
            val pz = localZ * fw

            val rx = when (node.type) {
                "occluder" -> if (node.radiiFW != null && node.radiiFW.size >= 3) node.radiiFW[0] else 1f
                "sprite" -> node.spriteWidthFW
                else -> 1f
            }
            val ry = when (node.type) {
                "occluder" -> if (node.radiiFW != null && node.radiiFW.size >= 3) node.radiiFW[1] else 1f
                "sprite" -> node.spriteHeightFW
                else -> 1f
            }
            val rz = if (node.type == "occluder" && node.radiiFW != null && node.radiiFW.size >= 3) node.radiiFW[2] else 1f

            val sx = node.baseScale[0] * slots.scale[i * 3] * rx * fw
            val sy = node.baseScale[1] * slots.scale[i * 3 + 1] * ry * fw
            val sz = node.baseScale[2] * slots.scale[i * 3 + 2] * rz * fw

            // Rotation quaternion: combine baseRotation and animated slots
            val bqx = node.baseRotation.getOrElse(0) { 0f }
            val bqy = node.baseRotation.getOrElse(1) { 0f }
            val bqz = node.baseRotation.getOrElse(2) { 0f }
            val bqw = node.baseRotation.getOrElse(3) { 1f }
            val sqx = slots.rotation[i * 4]
            val sqy = slots.rotation[i * 4 + 1]
            val sqz = slots.rotation[i * 4 + 2]
            val sqw = slots.rotation[i * 4 + 3]

            val qx = bqw * sqx + bqx * sqw + bqy * sqz - bqz * sqy
            val qy = bqw * sqy - bqx * sqz + bqy * sqw + bqz * sqx
            val qz = bqw * sqz + bqx * sqy - bqy * sqx + bqz * sqw
            val qw = bqw * sqw - bqx * sqx - bqy * sqy - bqz * sqz

            quaternionToMatrix(qx, qy, qz, qw, localMatrix)

            // Scale columns
            localMatrix[0] *= sx; localMatrix[1] *= sx; localMatrix[2] *= sx
            localMatrix[4] *= sy; localMatrix[5] *= sy; localMatrix[6] *= sy
            localMatrix[8] *= sz; localMatrix[9] *= sz; localMatrix[10] *= sz

            // Translate (applying spritePivot offset in scaled local axes when pivot != [0.5, 0.5])
            if (node.type == "sprite" && node.spritePivot.size >= 2) {
                val pivDx = 0.5f - node.spritePivot[0]
                val pivDy = 0.5f - node.spritePivot[1]
                localMatrix[12] = px + localMatrix[0] * pivDx + localMatrix[4] * pivDy
                localMatrix[13] = py + localMatrix[1] * pivDx + localMatrix[5] * pivDy
                localMatrix[14] = pz + localMatrix[2] * pivDx + localMatrix[6] * pivDy
            } else {
                localMatrix[12] = px
                localMatrix[13] = py
                localMatrix[14] = pz
            }
            localMatrix[15] = 1f

            val offset = i * 16

            if (node.parentIndex >= 0 && node.parentIndex < i) {
                // Child node: multiply parent * local
                val parentOffset = node.parentIndex * 16
                Matrix.multiplyMM(worldMatrices, offset, worldMatrices, parentOffset, localMatrix, 0)
            } else {
                // Root node: apply rotationFollow and camera pose M
                when (node.rotationFollow) {
                    "full" -> {
                        // Apply full head pose matrix M
                        Matrix.multiplyMM(worldMatrices, offset, pose, 0, localMatrix, 0)
                    }
                    "roll" -> {
                        // Only in-plane roll: create Z rotation matrix
                        val rollRad = Math.toRadians(faceState.rollDeg.toDouble()).toFloat()
                        val s = sin(rollRad); val c = cos(rollRad)
                        Matrix.setIdentityM(tempMat, 0)
                        tempMat[0] = c; tempMat[1] = s
                        tempMat[4] = -s; tempMat[5] = c
                        tempMat[12] = pose[12]; tempMat[13] = pose[13]; tempMat[14] = pose[14]
                        Matrix.multiplyMM(worldMatrices, offset, tempMat, 0, localMatrix, 0)
                    }
                    "none" -> {
                        // Fixed screen orientation at face position
                        Matrix.setIdentityM(tempMat, 0)
                        tempMat[12] = pose[12]; tempMat[13] = pose[13]; tempMat[14] = pose[14]
                        Matrix.multiplyMM(worldMatrices, offset, tempMat, 0, localMatrix, 0)
                    }
                    else -> {
                        Matrix.multiplyMM(worldMatrices, offset, pose, 0, localMatrix, 0)
                    }
                }
            }
        }
    }

    private fun quaternionToMatrix(qx: Float, qy: Float, qz: Float, qw: Float, out: FloatArray) {
        val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
        val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
        val wx = qw * qx; val wy = qw * qy; val wz = qw * qz

        out[0] = 1f - 2f * (yy + zz)
        out[1] = 2f * (xy + wz)
        out[2] = 2f * (xz - wy)
        out[3] = 0f

        out[4] = 2f * (xy - wz)
        out[5] = 1f - 2f * (xx + zz)
        out[6] = 2f * (yz + wx)
        out[7] = 0f

        out[8] = 2f * (xz + wy)
        out[9] = 2f * (yz - wx)
        out[10] = 1f - 2f * (xx + yy)
        out[11] = 0f

        out[12] = 0f; out[13] = 0f; out[14] = 0f; out[15] = 1f
    }
}
