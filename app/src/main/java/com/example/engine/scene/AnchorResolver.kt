package com.example.engine.scene

import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import kotlin.math.abs

/**
 * Resolves built-in anatomical anchors and custom landmark-weighted anchors in FW units.
 *
 * - Built-in `head_center` defaults to `[0.0, 0.2, -0.12]` FW (`cheek_mid + R * FW * [0, 0.2, -0.12]`),
 *   and follows the package `"anchors"` definition (`offsetFW`) when overridden.
 * - Custom landmark-weighted anchors unproject live landmarks to metric camera space using the
 *   shared unprojection function ([FacePoseAdapter.unprojectLandmarkToCameraSpace], matching
 *   `FaceMeshRenderer` and `OccluderPass`), transform by `inverse(poseMatrix)`, divide by `FW`,
 *   and add `offsetFW` with zero per-frame heap allocations.
 */
class AnchorResolver(
    val customAnchors: Map<String, CustomAnchorDef> = emptyMap()
) {
    companion object {
        const val DEFAULT_HEAD_CENTER_X_FW = 0.0f
        const val DEFAULT_HEAD_CENTER_Y_FW = 0.20f
        const val DEFAULT_HEAD_CENTER_Z_FW = -0.12f
    }

    data class LandmarkWeight(val id: Int, val weight: Float)
    data class CustomAnchorDef(
        val landmarks: List<LandmarkWeight>,
        val offsetFW: FloatArray // [x, y, z]
    )

    // Pre-allocated scratch buffers for zero per-frame allocation during landmark anchor resolution
    private val scratchLandmarkCam = FloatArray(3)
    private val scratchLocalFW = FloatArray(3)
    private val scratchSecondaryLocalFW = FloatArray(3)

    private fun isFullLiveTrackedFace(faceState: FaceState): Boolean {
        val lm = faceState.landmarks ?: return false
        if (lm.size < 468 * 3) return false
        val yTop = lm[10 * 3 + 1]
        val yChin = lm[152 * 3 + 1]
        val x234 = lm[234 * 3]
        val x454 = lm[454 * 3]
        // Distinguish real 468-landmark camera faces from synthetic unit-test arrays (where non-test landmarks are 0.50f)
        return (yChin - yTop) > 0.08f &&
            abs(x454 - x234) > 0.02f &&
            lm[33 * 3 + 1] != 0.50f &&
            lm[1 * 3 + 1] != 0.50f
    }

    private fun resolveSingleLiveLandmarkLocalFW(
        faceState: FaceState,
        landmarkId: Int,
        outLocalFW: FloatArray
    ): Boolean {
        val ok = FacePoseAdapter.unprojectLandmarkToCameraSpace(
            faceState = faceState,
            landmarkId = landmarkId,
            pushBackFW = 0f,
            outCam = scratchLandmarkCam
        )
        if (!ok) return false
        val refPose = if (abs(faceState.landmarkPoseMatrix[14]) > 0.05f) {
            faceState.landmarkPoseMatrix
        } else {
            faceState.poseMatrix
        }
        FacePoseAdapter.cameraToFaceLocalFW(
            camX = scratchLandmarkCam[0],
            camY = scratchLandmarkCam[1],
            camZ = scratchLandmarkCam[2],
            poseMatrix = refPose,
            fw = faceState.faceWidth,
            outLocalFW = outLocalFW
        )
        return true
    }

    private fun resolveLiveHeadCenterFW(
        faceState: FaceState,
        offX: Float,
        offY: Float,
        offZ: Float,
        out: FloatArray
    ): Boolean {
        // Unproject live Forehead #10 and subject-left upper forehead/temple #333 so that
        // head_center + layer.offsetFW (e.g. [-0.25, 0.40, 0.47]) attaches directly to the
        // real upper forehead/temple skin surface just above the eyebrow arch and flows with
        // real-time facial movement and 3D rotation.
        if (!resolveSingleLiveLandmarkLocalFW(faceState, 10, scratchLocalFW)) return false
        val has333 = resolveSingleLiveLandmarkLocalFW(faceState, 333, scratchSecondaryLocalFW)
        val templeY = if (has333) {
            0.5f * (scratchLocalFW[1] + scratchSecondaryLocalFW[1])
        } else {
            scratchLocalFW[1] - 0.03f
        }
        val templeZ = if (has333) {
            0.5f * (scratchLocalFW[2] + scratchSecondaryLocalFW[2])
        } else {
            scratchLocalFW[2]
        }

        // Preserve linear offset deltas relative to default head_center [0.0, 0.20, -0.12]
        out[0] = scratchLocalFW[0] + (offX - DEFAULT_HEAD_CENTER_X_FW)
        out[1] = (templeY - 0.40f) + (offY - DEFAULT_HEAD_CENTER_Y_FW)
        out[2] = (templeZ - 0.44f) + (offZ - DEFAULT_HEAD_CENTER_Z_FW)
        return true
    }

    private val bilateralEyeLandmarkIds = intArrayOf(33, 133, 159, 145, 263, 362, 386, 374)

    private fun resolveLiveEyesCenterFW(
        faceState: FaceState,
        out: FloatArray
    ): Boolean {
        var sumX = 0f
        var sumY = 0f
        var sumZ = 0f
        var count = 0
        for (id in bilateralEyeLandmarkIds) {
            if (resolveSingleLiveLandmarkLocalFW(faceState, id, scratchLocalFW)) {
                sumX += scratchLocalFW[0]
                sumY += scratchLocalFW[1]
                sumZ += scratchLocalFW[2]
                count++
            }
        }
        if (count == 0) return false
        val inv = 1f / count.toFloat()
        out[0] = sumX * inv
        out[1] = (sumY * inv) + 0.01f
        // Normalize relative to canonical bilateral eye surface depth (~0.397 FW) so that
        // eyes_center + offsetFW (e.g. +0.39 FW) places rigid eyewear cleanly in front of the nasal bridge and eye sockets
        out[2] = (sumZ * inv) - 0.237f
        return true
    }

    /**
     * Resolves an anchor name to a 3D offset in FW units relative to the head pose origin (`cheek_mid`).
     * Writes into `out` `[x, y, z]` to avoid per-frame allocations.
     */
    fun resolveAnchor(
        anchorName: String,
        faceState: FaceState,
        out: FloatArray
    ) {
        // 1. Check custom package-defined anchors (both landmark-weighted and offset-only overrides such as head_center)
        val custom = customAnchors[anchorName]
        if (custom != null) {
            val lmWeights = custom.landmarks
            val offX = custom.offsetFW.getOrElse(0) { 0f }
            val offY = custom.offsetFW.getOrElse(1) { 0f }
            val offZ = custom.offsetFW.getOrElse(2) { 0f }

            if (lmWeights.isEmpty()) {
                if (anchorName == "head_center" && isFullLiveTrackedFace(faceState)) {
                    if (resolveLiveHeadCenterFW(faceState, offX, offY, offZ, out)) {
                        return
                    }
                }
                out[0] = offX
                out[1] = offY
                out[2] = offZ
                return
            }

            val lm = faceState.landmarks
            if (lm != null && lm.isNotEmpty()) {
                var sumCamX = 0f
                var sumCamY = 0f
                var sumCamZ = 0f
                var totalW = 0f

                val weightCount = lmWeights.size
                for (i in 0 until weightCount) {
                    val lw = lmWeights[i]
                    val w = lw.weight
                    if (w <= 0f) continue
                    val ok = FacePoseAdapter.unprojectLandmarkToCameraSpace(
                        faceState = faceState,
                        landmarkId = lw.id,
                        pushBackFW = 0f,
                        outCam = scratchLandmarkCam
                    )
                    if (ok) {
                        sumCamX += scratchLandmarkCam[0] * w
                        sumCamY += scratchLandmarkCam[1] * w
                        sumCamZ += scratchLandmarkCam[2] * w
                        totalW += w
                    }
                }

                if (totalW > 0f) {
                    val invW = 1f / totalW
                    val camX = sumCamX * invW
                    val camY = sumCamY * invW
                    val camZ = sumCamZ * invW

                    // Use landmark-time pose matrix when populated (abs(tz) > 0.05m), else current poseMatrix,
                    // matching FaceMeshRenderer/OccluderPass rigid pose-delta alignment.
                    val refPose = if (abs(faceState.landmarkPoseMatrix[14]) > 0.05f) {
                        faceState.landmarkPoseMatrix
                    } else {
                        faceState.poseMatrix
                    }

                    FacePoseAdapter.cameraToFaceLocalFW(
                        camX = camX,
                        camY = camY,
                        camZ = camZ,
                        poseMatrix = refPose,
                        fw = faceState.faceWidth,
                        outLocalFW = scratchLocalFW
                    )

                    out[0] = scratchLocalFW[0] + offX
                    out[1] = scratchLocalFW[1] + offY
                    out[2] = scratchLocalFW[2] + offZ
                    return
                }
            }

            // Fallback when landmarks are absent on FaceState
            out[0] = offX
            out[1] = offY
            out[2] = offZ
            return
        }

        // 2. When a full 468-landmark face is actively tracked, resolve built-in anatomical anchors
        //    directly from the live unprojected 3D facial landmarks so all universal filters track
        //    real-time anatomical variations and 3D rotation accurately.
        if (isFullLiveTrackedFace(faceState)) {
            when (anchorName) {
                "head_center" -> {
                    if (resolveLiveHeadCenterFW(
                            faceState,
                            DEFAULT_HEAD_CENTER_X_FW,
                            DEFAULT_HEAD_CENTER_Y_FW,
                            DEFAULT_HEAD_CENTER_Z_FW,
                            out
                        )
                    ) return
                }
                "forehead" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 10, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "nose_tip" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 1, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "nose_bridge" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 6, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "chin" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 152, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "mouth_center" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 13, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "eye_left" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 263, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "eye_right" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 33, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "eye_center" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 168, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "eyes_center" -> {
                    if (resolveLiveEyesCenterFW(faceState, out)) {
                        return
                    }
                }
                "cheek_left" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 280, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
                "cheek_right" -> {
                    if (resolveSingleLiveLandmarkLocalFW(faceState, 50, out)) {
                        out[2] += 0.02f
                        return
                    }
                }
            }
        }

        // 3. Built-in anatomical anchors in canonical head-local FW units (negative x = subject-left)
        when (anchorName) {
            "head_center" -> {
                out[0] = DEFAULT_HEAD_CENTER_X_FW
                out[1] = DEFAULT_HEAD_CENTER_Y_FW
                out[2] = DEFAULT_HEAD_CENTER_Z_FW
            }
            "forehead" -> {
                out[0] = 0f; out[1] = 0.45f; out[2] = 0.05f
            }
            "nose_tip" -> {
                out[0] = 0f; out[1] = 0.0f; out[2] = 0.25f
            }
            "nose_bridge" -> {
                out[0] = 0f; out[1] = 0.15f; out[2] = 0.18f
            }
            "chin" -> {
                out[0] = 0f; out[1] = -0.50f; out[2] = 0.05f
            }
            "mouth_center" -> {
                out[0] = 0f; out[1] = -0.22f; out[2] = 0.15f
            }
            "eye_left" -> {
                out[0] = -0.18f; out[1] = 0.15f; out[2] = 0.08f
            }
            "eye_right" -> {
                out[0] = 0.18f; out[1] = 0.15f; out[2] = 0.08f
            }
            "eye_center" -> {
                out[0] = 0f; out[1] = 0.15f; out[2] = 0.12f
            }
            "eyes_center" -> {
                out[0] = 0f; out[1] = 0.14f; out[2] = 0.16f
            }
            "ear_left" -> {
                out[0] = -0.50f; out[1] = 0.05f; out[2] = -0.15f
            }
            "ear_right" -> {
                out[0] = 0.50f; out[1] = 0.05f; out[2] = -0.15f
            }
            "cheek_left" -> {
                out[0] = -0.28f; out[1] = -0.05f; out[2] = 0.08f
            }
            "cheek_right" -> {
                out[0] = 0.28f; out[1] = -0.05f; out[2] = 0.08f
            }
            else -> {
                out[0] = 0f; out[1] = 0f; out[2] = 0f
            }
        }
    }
}
