package com.example.engine.render

import android.graphics.Bitmap
import android.graphics.Color
import android.opengl.EGL14
import android.opengl.GLES30
import com.example.engine.pkg.FaceRegions
import com.example.engine.pkg.UvRegistry
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.CompiledNode
import com.example.engine.scene.PropertySlots
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * RecolorPass (Flow 4 — Chunk Primitives-1 `EC1` / `recolor@1`):
 *
 * Reusable camera-pixel color-transfer template (`pixel.recolor@1`, `pass.recolor@1`, `node.eyeColor@1`):
 * - Consumes any composable `$mask:<id>` target (defaulting to `$mask:iris` = `(left_iris ∩ left_eye) ∪ (right_iris ∩ right_eye)`).
 * - Applies luminance-preserving color transfer (`color = [r, g, b]`, `strength`) that retains natural
 *   fibrous iris texture contrast ($\Delta Y$) instead of painting a flat disc.
 * - Enforces default `strength <= 0.70f` cap (`MAX_DEFAULT_STRENGTH`).
 * - Attenuates per-eye weight by blink blendshapes (`1.0 - eyeBlinkLeft`, `1.0 - eyeBlinkRight`).
 * - Clips off-center gaze strictly by the eyelid contour (`irisCircle ∩ eyeContour`).
 * - Protects strong corneal reflections and eyeglasses glare via a luminance-key roll-off
 *   (`glareLowLuma = 0.78`, `glareHighLuma = 0.92`) so bright specular highlights remain uncolored.
 * - Executes in `CAMERA_PIXEL_PASSES` (Stage 2, before `CLEAR_DEPTH`, `OCCLUDERS`, and 3D props)
 *   in both Live Preview (`PATH:B`) and Offscreen Still Photo rendering.
 */
class RecolorPass {

    companion object {
        const val TEMPLATE_ID = "recolor@1"
        const val PIXEL_TEMPLATE_ID = "pixel.recolor@1"
        const val EYE_COLOR_NODE_TYPE = "eyeColor"
        const val RECOLOR_NODE_TYPE = "recolor"

        /**
         * Default maximum recolor strength (`0.70f`) so natural fibrous iris detail is preserved.
         */
        const val MAX_DEFAULT_STRENGTH = 0.70f
        const val DEFAULT_STRENGTH = 0.55f
        const val DEFAULT_PRESERVE_LUMINANCE = 0.85f
        const val DEFAULT_GLARE_LOW_LUMA = 0.78f
        const val DEFAULT_GLARE_HIGH_LUMA = 0.92f
        const val DEFAULT_PUPIL_LOW_LUMA = 0.03f
        const val DEFAULT_PUPIL_HIGH_LUMA = 0.11f
        const val DEFAULT_FEATHER_PX = 2.0f

        private const val VS_RECOLOR = """#version 300 es
            layout(location = 0) in vec2 aPosition;
            layout(location = 1) in vec2 aTexCoord;
            out vec2 vTexCoord;
            void main() {
                vTexCoord = aTexCoord;
                gl_Position = vec4(aPosition, 0.0, 1.0);
            }
        """

        private const val FS_RECOLOR = """#version 300 es
            precision mediump float;
            in vec2 vTexCoord;
            uniform sampler2D uCameraTexture;
            uniform sampler2D uMaskTexture;
            uniform int uHasCameraTexture;
            uniform int uHasMaskTexture;
            uniform int uIsIrisMask;
            uniform int uRecolorMode; // 0 = tint, 1 = iris, 2 = teeth whitening, 3 = sclera brightening
            uniform vec3 uTargetColor;
            uniform float uStrength;
            uniform float uPreserveLuminance;
            uniform vec2 uGlareRange; // x = low (0.78), y = high (0.92)
            uniform vec2 uPupilRange; // x = low (0.03), y = high (0.11)
            uniform vec4 uRightIrisCirclePx; // xy = centerPx, z = radiusPx, w = 1.0 - eyeBlinkRight
            uniform vec4 uLeftIrisCirclePx;  // xy = centerPx, z = radiusPx, w = 1.0 - eyeBlinkLeft
            uniform vec2 uViewportSize;
            uniform float uFeatherPx;
            uniform float uMouthOpenWeight;
            uniform float uMinCavityLuma;
            uniform float uDesaturateYellow;
            uniform vec2 uEyeOpenWeights; // x = rightEyeOpen, y = leftEyeOpen
            uniform float uFaceMidX;
            out vec4 fragColor;

            float circleMask(vec2 fragPx, vec4 irisCircle, float featherPx) {
                if (irisCircle.z <= 0.5 || irisCircle.w <= 0.001) return 0.0;
                float d = length(fragPx - irisCircle.xy);
                float f = max(featherPx, 0.75);
                float w = 1.0 - smoothstep(max(0.0, irisCircle.z - f), irisCircle.z + f * 0.5, d);
                return w * clamp(irisCircle.w, 0.0, 1.0);
            }

            void main() {
                vec2 viewUv = vec2(vTexCoord.x, 1.0 - vTexCoord.y);
                vec2 fragPx = vec2(viewUv.x * uViewportSize.x, viewUv.y * uViewportSize.y);
                float maskW = 0.0;
                if (uHasMaskTexture == 1) {
                    maskW = texture(uMaskTexture, viewUv).r;
                } else {
                    float wR = circleMask(fragPx, uRightIrisCirclePx, uFeatherPx);
                    float wL = circleMask(fragPx, uLeftIrisCirclePx, uFeatherPx);
                    maskW = max(wR, wL);
                }

                float clampedStrength = clamp(uStrength, 0.0, 0.70);
                if (uRecolorMode == 2 || uRecolorMode == 3) {
                    float modeGate = (uRecolorMode == 2)
                        ? clamp(uMouthOpenWeight, 0.0, 1.0)
                        : clamp((viewUv.x < uFaceMidX) ? uEyeOpenWeights.x : uEyeOpenWeights.y, 0.0, 1.0);
                    float effW0 = clamp(maskW, 0.0, 1.0) * clampedStrength * modeGate;
                    if (effW0 <= 0.001) {
                        discard;
                    }
                    if (uHasCameraTexture == 1) {
                        vec4 baseTex = texture(uCameraTexture, vTexCoord);
                        vec3 baseRgb = baseTex.rgb;
                        float luma = dot(baseRgb, vec3(0.2126, 0.7152, 0.0722));
                        float cavityGuard = smoothstep(uMinCavityLuma, min(0.95, uMinCavityLuma + 0.12), luma);
                        float glareGuard = 1.0 - smoothstep(uGlareRange.x, uGlareRange.y, luma);
                        float mucosaGuard = 1.0;
                        if (uRecolorMode == 2) {
                            float redExcess = max(0.0, (baseRgb.r - baseRgb.g) - 0.14);
                            mucosaGuard = 1.0 - smoothstep(0.0, 0.16, redExcess);
                        }
                        float weight = clamp(effW0 * cavityGuard * glareGuard * mucosaGuard, 0.0, 0.70);
                        if (weight <= 0.001) {
                            discard;
                        }
                        float liftedLuma = clamp(luma + (1.0 - luma) * 0.28 * luma, 0.0, 1.0);
                        float yellowExcess = clamp(max(0.0, (baseRgb.r + baseRgb.g) * 0.5 - baseRgb.b) * clamp(uDesaturateYellow, 0.0, 1.0), 0.0, 0.35);
                        vec3 whitened = clamp(vec3(
                            baseRgb.r + (liftedLuma - baseRgb.r) * 0.55 - yellowExcess * 0.20,
                            baseRgb.g + (liftedLuma - baseRgb.g) * 0.55 - yellowExcess * 0.10,
                            baseRgb.b + (liftedLuma - baseRgb.b) * 0.65 + yellowExcess * 0.55
                        ), 0.0, 1.0);
                        fragColor = vec4(whitened * weight, weight);
                    } else {
                        float weight = effW0 * 0.45;
                        fragColor = vec4(vec3(1.0) * weight, weight);
                    }
                    return;
                }

                // Perceptual chroma response boost for subtle lip/cheek beauty tints while keeping iris linear
                float perceptualStrength = (uIsIrisMask == 1)
                    ? clampedStrength
                    : clamp(clampedStrength * 1.85, 0.0, 0.70);
                float effMask = clamp(maskW, 0.0, 1.0) * perceptualStrength;
                if (effMask <= 0.001) {
                    discard;
                }

                if (uHasCameraTexture == 1) {
                    vec4 baseTex = texture(uCameraTexture, vTexCoord);
                    vec3 baseRgb = baseTex.rgb;
                    float luma = dot(baseRgb, vec3(0.2126, 0.7152, 0.0722));
                    float glareProtection = 1.0 - smoothstep(uGlareRange.x, uGlareRange.y, luma);
                    float pupilProtection = (uIsIrisMask == 1) ? smoothstep(uPupilRange.x, uPupilRange.y, luma) : 1.0;
                    float weight = effMask * glareProtection * pupilProtection;

                    float targetLuma = max(dot(uTargetColor, vec3(0.2126, 0.7152, 0.0722)), 0.08);
                    vec3 lumaScaledColor = clamp(uTargetColor * (luma / targetLuma), 0.0, 1.0);
                    vec3 directTint = clamp(uTargetColor * (0.35 + 0.65 * luma), 0.0, 1.0);
                    vec3 recolored = mix(directTint, lumaScaledColor, clamp(uPreserveLuminance, 0.0, 1.0));
                    fragColor = vec4(recolored * weight, weight);
                } else {
                    // Fallback translucent tint when camera pixels are not bound
                    float weight = effMask * 0.65;
                    fragColor = vec4(uTargetColor * weight, weight);
                }
            }
        """

        /**
         * Smoothstep utility in `[0, 1]`.
         */
        fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
            if (edge1 <= edge0) return if (x >= edge1) 1f else 0f
            val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }

        /**
         * Relative luminance ($Y = 0.2126 R + 0.7152 G + 0.0722 B$) in `[0, 1]`.
         */
        fun luminance(r: Float, g: Float, b: Float): Float {
            return (0.2126f * r + 0.7152f * g + 0.0722f * b).coerceIn(0f, 1f)
        }

        /**
         * Computes the specular highlight / eyeglasses glare protection factor in `[0.0, 1.0]`:
         * - Returns `1.0` for normal iris luminance ($Y \le 0.78$).
         * - Smoothly rolls off to `0.0` as $Y$ approaches `glareHighLuma` ($0.92$) so strong
         *   corneal reflections and glasses glare stay 100% uncolored.
         */
        @JvmOverloads
        fun computeGlareProtectionFactor(
            luma: Float,
            glareLowLuma: Float = DEFAULT_GLARE_LOW_LUMA,
            glareHighLuma: Float = DEFAULT_GLARE_HIGH_LUMA
        ): Float {
            return (1f - smoothstep(glareLowLuma, glareHighLuma, luma)).coerceIn(0f, 1f)
        }

        /**
         * Computes pupil center protection factor in `[0.0, 1.0]` so the dark pupil core
         * ($Y \le 0.03$) is not washed out by iris recoloring.
         */
        @JvmOverloads
        fun computePupilProtectionFactor(
            luma: Float,
            pupilLowLuma: Float = DEFAULT_PUPIL_LOW_LUMA,
            pupilHighLuma: Float = DEFAULT_PUPIL_HIGH_LUMA
        ): Float {
            return smoothstep(pupilLowLuma, pupilHighLuma, luma).coerceIn(0f, 1f)
        }

        /**
         * Pure, allocation-free luminance-preserving pixel recolor function used by CPU photo fallback
         * and primitive conformance tests.
         *
         * @return Effective blend weight applied in `[0.0, 0.70]`. Writes resulting `[r, g, b]` into [outRgb].
         */
        @JvmOverloads
        fun recolorPixelRgb(
            baseR: Float,
            baseG: Float,
            baseB: Float,
            targetR: Float,
            targetG: Float,
            targetB: Float,
            maskWeight: Float,
            strength: Float = DEFAULT_STRENGTH,
            preserveLuminance: Float = DEFAULT_PRESERVE_LUMINANCE,
            glareLowLuma: Float = DEFAULT_GLARE_LOW_LUMA,
            glareHighLuma: Float = DEFAULT_GLARE_HIGH_LUMA,
            allowOverStrength: Boolean = false,
            outRgb: FloatArray,
            protectPupil: Boolean = true
        ): Float {
            val bR = baseR.coerceIn(0f, 1f)
            val bG = baseG.coerceIn(0f, 1f)
            val bB = baseB.coerceIn(0f, 1f)
            val maxStr = if (allowOverStrength) 1.0f else MAX_DEFAULT_STRENGTH
            val clampedStrength = strength.coerceIn(0f, maxStr)
            val mW = maskWeight.coerceIn(0f, 1f)
            if (mW <= 0.0001f || clampedStrength <= 0.0001f) {
                outRgb[0] = bR
                outRgb[1] = bG
                outRgb[2] = bB
                return 0f
            }

            val luma = (0.2126f * bR + 0.7152f * bG + 0.0722f * bB).coerceIn(0f, 1f)
            val glareFactor = computeGlareProtectionFactor(luma, glareLowLuma, glareHighLuma)
            val pupilFactor = if (protectPupil) computePupilProtectionFactor(luma) else 1f
            val effectiveWeight = (mW * clampedStrength * glareFactor * pupilFactor).coerceIn(0f, maxStr)

            if (effectiveWeight <= 0.0001f) {
                outRgb[0] = bR
                outRgb[1] = bG
                outRgb[2] = bB
                return 0f
            }

            val tR = targetR.coerceIn(0f, 1f)
            val tG = targetG.coerceIn(0f, 1f)
            val tB = targetB.coerceIn(0f, 1f)
            val targetLuma = (0.2126f * tR + 0.7152f * tG + 0.0722f * tB).coerceAtLeast(0.08f)

            // Luminance-scaled color transfer preserves radial fibrous texture variations (Delta Y)
            val lumaRatio = luma / targetLuma
            val scaledR = (tR * lumaRatio).coerceIn(0f, 1f)
            val scaledG = (tG * lumaRatio).coerceIn(0f, 1f)
            val scaledB = (tB * lumaRatio).coerceIn(0f, 1f)

            val directScale = 0.35f + 0.65f * luma
            val directR = (tR * directScale).coerceIn(0f, 1f)
            val directG = (tG * directScale).coerceIn(0f, 1f)
            val directB = (tB * directScale).coerceIn(0f, 1f)

            val pLuma = preserveLuminance.coerceIn(0f, 1f)
            val recolorR = directR + (scaledR - directR) * pLuma
            val recolorG = directG + (scaledG - directG) * pLuma
            val recolorB = directB + (scaledB - directB) * pLuma

            outRgb[0] = (bR + (recolorR - bR) * effectiveWeight).coerceIn(0f, 1f)
            outRgb[1] = (bG + (recolorG - bG) * effectiveWeight).coerceIn(0f, 1f)
            outRgb[2] = (bB + (recolorB - bB) * effectiveWeight).coerceIn(0f, 1f)
            return effectiveWeight
        }

        /**
         * Computes the smooth mouth-open gate in `[0.0, 1.0]` used to gate `$mask:teeth` whitening
         * so closed or nearly closed lips never trigger inner-lip whitening.
         */
        fun computeMouthOpenGate(faceState: FaceState): Float {
            val ratio = FaceRegions.computeMouthOpeningRatio(faceState.landmarks)
            val jawOpen = (faceState.blendshapes["jawOpen"] ?: 0f).coerceIn(0f, 1f)
            return smoothstep(0.008f, 0.035f, max(ratio, jawOpen * 0.25f)).coerceIn(0f, 1f)
        }

        /**
         * Natural teeth whitening and sclera brightening transfer (`$mask:teeth`, `$mask:sclera`):
         * - Neutralizes yellow/warm cast (boosting blue relative to excess red/green) while preserving
         *   natural enamel/sclera luminance detail and specular reflections.
         * - Protects dark oral cavity / shadow pixels below [minCavityLuma] (`0.22f` for teeth) so the
         *   inside of the mouth is never artificially lightened.
         * - When [protectMucosaRedness] is `true` (default for teeth), rejects pink/red tongue and gum
         *   mucosa pixels where `R - G > 0.14f` so only teeth enamel is whitened.
         *
         * @return Effective whitening weight applied in `[0.0, 0.70]`. Writes `[r, g, b]` into [outRgb].
         */
        @JvmOverloads
        fun whitenTeethOrScleraPixelRgb(
            baseR: Float,
            baseG: Float,
            baseB: Float,
            maskWeight: Float,
            whiteningStrength: Float = DEFAULT_STRENGTH,
            desaturateYellow: Float = 0.65f,
            minCavityLuma: Float = 0.22f,
            glareLowLuma: Float = 0.86f,
            glareHighLuma: Float = 0.97f,
            outRgb: FloatArray,
            protectMucosaRedness: Boolean = true
        ): Float {
            val bR = baseR.coerceIn(0f, 1f)
            val bG = baseG.coerceIn(0f, 1f)
            val bB = baseB.coerceIn(0f, 1f)
            val mW = maskWeight.coerceIn(0f, 1f)
            val clampedStr = whiteningStrength.coerceIn(0f, MAX_DEFAULT_STRENGTH)
            if (mW <= 0.0001f || clampedStr <= 0.0001f) {
                outRgb[0] = bR
                outRgb[1] = bG
                outRgb[2] = bB
                return 0f
            }

            val luma = luminance(bR, bG, bB)
            val cavityGuard = smoothstep(minCavityLuma, (minCavityLuma + 0.12f).coerceAtMost(0.95f), luma)
            val glareGuard = computeGlareProtectionFactor(luma, glareLowLuma, glareHighLuma)
            val mucosaGuard = if (protectMucosaRedness) {
                val redExcess = max(0f, (bR - bG) - 0.14f)
                (1f - smoothstep(0f, 0.16f, redExcess)).coerceIn(0f, 1f)
            } else {
                1f
            }
            val effW = (mW * clampedStr * cavityGuard * glareGuard * mucosaGuard).coerceIn(0f, MAX_DEFAULT_STRENGTH)
            if (effW <= 0.0001f) {
                outRgb[0] = bR
                outRgb[1] = bG
                outRgb[2] = bB
                return 0f
            }

            // Lift midtone luminance with soft-knee curve and neutralize warm/yellow tint (RG > B)
            val liftedLuma = (luma + (1f - luma) * 0.28f * luma).coerceIn(0f, 1f)
            val yellowExcess = (max(0f, (bR + bG) * 0.5f - bB) * desaturateYellow.coerceIn(0f, 1f)).coerceIn(0f, 0.35f)
            val neutralR = (bR + (liftedLuma - bR) * 0.55f - yellowExcess * 0.20f).coerceIn(0f, 1f)
            val neutralG = (bG + (liftedLuma - bG) * 0.55f - yellowExcess * 0.10f).coerceIn(0f, 1f)
            val neutralB = (bB + (liftedLuma - bB) * 0.65f + yellowExcess * 0.55f).coerceIn(0f, 1f)

            outRgb[0] = (bR + (neutralR - bR) * effW).coerceIn(0f, 1f)
            outRgb[1] = (bG + (neutralG - bG) * effW).coerceIn(0f, 1f)
            outRgb[2] = (bB + (neutralB - bB) * effW).coerceIn(0f, 1f)
            return effW
        }

        /**
         * Evaluates the gaze-contained, blink-attenuated iris mask weight at `(xPx, yPx)` in screen pixels:
         * `weight = max((rightIrisCircle ∩ rightEyeContour) * (1 - eyeBlinkRight), (leftIrisCircle ∩ leftEyeContour) * (1 - eyeBlinkLeft))`.
         *
         * Guarantees that:
         * 1. Off-center gaze is clipped by the eyelid polygon (`right_eye` / `left_eye`), never spilling onto skin.
         * 2. Blinking (`eyeBlinkLeft / eyeBlinkRight -> 1.0`) smoothly fades the iris recolor weight to `0.0`.
         */
        @JvmOverloads
        fun evaluateIrisMaskWeightAtPixel(
            xPx: Float,
            yPx: Float,
            faceState: FaceState,
            viewportWidth: Int,
            viewportHeight: Int,
            featherPx: Float = DEFAULT_FEATHER_PX
        ): Float {
            val lm = faceState.landmarks ?: return 0f
            if (lm.size < 478 * 3) return 0f
            val vw = viewportWidth.coerceAtLeast(1).toFloat()
            val vh = viewportHeight.coerceAtLeast(1).toFloat()

            val blinkR = (faceState.blendshapes["eyeBlinkRight"] ?: 0f).coerceIn(0f, 1f)
            val blinkL = (faceState.blendshapes["eyeBlinkLeft"] ?: 0f).coerceIn(0f, 1f)
            val openR = (1f - blinkR).coerceIn(0f, 1f)
            val openL = (1f - blinkL).coerceIn(0f, 1f)

            val rightW = if (openR > 0.001f) {
                val irisW = evaluateSingleIrisCircleWeightPx(
                    xPx = xPx,
                    yPx = yPx,
                    centerIndex = FaceRegions.RIGHT_IRIS_CENTER,
                    ringIndices = FaceRegions.RIGHT_IRIS_RING,
                    landmarks = lm,
                    faceState = faceState,
                    viewW = vw,
                    viewH = vh,
                    featherPx = featherPx
                )
                if (irisW > 0.001f) {
                    val eyeClipW = evaluatePolygonContainmentWeightPx(
                        xPx = xPx,
                        yPx = yPx,
                        loopIndices = FaceRegions.RIGHT_EYE_CONTOUR,
                        landmarks = lm,
                        faceState = faceState,
                        viewW = vw,
                        viewH = vh,
                        featherPx = featherPx
                    )
                    min(irisW, eyeClipW) * openR
                } else {
                    0f
                }
            } else {
                0f
            }

            val leftW = if (openL > 0.001f) {
                val irisW = evaluateSingleIrisCircleWeightPx(
                    xPx = xPx,
                    yPx = yPx,
                    centerIndex = FaceRegions.LEFT_IRIS_CENTER,
                    ringIndices = FaceRegions.LEFT_IRIS_RING,
                    landmarks = lm,
                    faceState = faceState,
                    viewW = vw,
                    viewH = vh,
                    featherPx = featherPx
                )
                if (irisW > 0.001f) {
                    val eyeClipW = evaluatePolygonContainmentWeightPx(
                        xPx = xPx,
                        yPx = yPx,
                        loopIndices = FaceRegions.LEFT_EYE_CONTOUR,
                        landmarks = lm,
                        faceState = faceState,
                        viewW = vw,
                        viewH = vh,
                        featherPx = featherPx
                    )
                    min(irisW, eyeClipW) * openL
                } else {
                    0f
                }
            } else {
                0f
            }

            return max(rightW, leftW).coerceIn(0f, 1f)
        }

        private fun evaluateSingleIrisCircleWeightPx(
            xPx: Float,
            yPx: Float,
            centerIndex: Int,
            ringIndices: IntArray,
            landmarks: FloatArray,
            faceState: FaceState,
            viewW: Float,
            viewH: Float,
            featherPx: Float
        ): Float {
            val cBase = centerIndex * 3
            if (cBase + 1 >= landmarks.size) return 0f
            val ptScratch = FloatArray(2)
            FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                u = landmarks[cBase],
                v = landmarks[cBase + 1],
                z = if (cBase + 2 < landmarks.size) landmarks[cBase + 2] else 0f,
                faceState = faceState,
                viewWidth = viewW,
                viewHeight = viewH,
                outPt = ptScratch
            )
            val cx = ptScratch[0]
            val cy = ptScratch[1]

            var radiusSum = 0f
            var ringCount = 0
            for (rIdx in ringIndices) {
                val rBase = rIdx * 3
                if (rBase + 1 < landmarks.size) {
                    FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                        u = landmarks[rBase],
                        v = landmarks[rBase + 1],
                        z = if (rBase + 2 < landmarks.size) landmarks[rBase + 2] else 0f,
                        faceState = faceState,
                        viewWidth = viewW,
                        viewHeight = viewH,
                        outPt = ptScratch
                    )
                    radiusSum += hypot(ptScratch[0] - cx, ptScratch[1] - cy)
                    ringCount++
                }
            }
            val radiusPx = if (ringCount > 0) (radiusSum / ringCount.toFloat()) else 0f
            if (radiusPx <= 0.5f) return 0f

            val dist = hypot(xPx - cx, yPx - cy)
            val f = featherPx.coerceAtLeast(0.75f)
            return (1f - smoothstep((radiusPx - f).coerceAtLeast(0f), radiusPx + f * 0.5f, dist)).coerceIn(0f, 1f)
        }

        internal fun evaluatePolygonContainmentWeightPx(
            xPx: Float,
            yPx: Float,
            loopIndices: IntArray,
            landmarks: FloatArray,
            faceState: FaceState,
            viewW: Float,
            viewH: Float,
            featherPx: Float
        ): Float {
            val n = loopIndices.size
            if (n < 3) return 0f
            val polyX = FloatArray(n)
            val polyY = FloatArray(n)
            val pt = FloatArray(2)
            for (i in 0 until n) {
                val lmIdx = loopIndices[i]
                val base = lmIdx * 3
                if (base + 1 >= landmarks.size) return 0f
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = landmarks[base],
                    v = landmarks[base + 1],
                    z = if (base + 2 < landmarks.size) landmarks[base + 2] else 0f,
                    faceState = faceState,
                    viewWidth = viewW,
                    viewHeight = viewH,
                    outPt = pt
                )
                polyX[i] = pt[0]
                polyY[i] = pt[1]
            }

            var inside = false
            var minDistSq = Float.MAX_VALUE
            var j = n - 1
            for (i in 0 until n) {
                val xi = polyX[i]
                val yi = polyY[i]
                val xj = polyX[j]
                val yj = polyY[j]

                val intersect = ((yi > yPx) != (yj > yPx)) &&
                    (xPx < (xj - xi) * (yPx - yi) / ((yj - yi).let { if (kotlin.math.abs(it) < 1e-5f) 1e-5f else it }) + xi)
                if (intersect) inside = !inside

                val dx = xj - xi
                val dy = yj - yi
                val lenSq = dx * dx + dy * dy
                val t = if (lenSq > 1e-5f) {
                    (((xPx - xi) * dx + (yPx - yi) * dy) / lenSq).coerceIn(0f, 1f)
                } else {
                    0f
                }
                val projX = xi + t * dx
                val projY = yi + t * dy
                val distSq = (xPx - projX) * (xPx - projX) + (yPx - projY) * (yPx - projY)
                if (distSq < minDistSq) minDistSq = distSq
                j = i
            }

            val dist = kotlin.math.sqrt(minDistSq)
            val f = featherPx.coerceAtLeast(0.5f)
            return if (inside) {
                if (dist >= f) 1f else (0.5f + 0.5f * (dist / f)).coerceIn(0f, 1f)
            } else {
                if (dist >= f) 0f else (0.5f * (1f - dist / f)).coerceIn(0f, 1f)
            }
        }

        /**
         * Resolves node/pass parameters for `eyeColor` / `recolor@1` from JSON and manifest params.
         */
        fun resolveRecolorParams(
            nodeId: String,
            paramsJson: JSONObject?,
            materialColor: FloatArray?,
            materialOpacity: Float?,
            resolvedManifestParams: Map<String, Any> = emptyMap()
        ): Map<String, Float> {
            val out = LinkedHashMap<String, Float>()
            var r = materialColor?.getOrNull(0) ?: 0.22f
            var g = materialColor?.getOrNull(1) ?: 0.58f
            var b = materialColor?.getOrNull(2) ?: 0.88f

            if (paramsJson != null) {
                val colRaw = paramsJson.opt("color") ?: paramsJson.opt("targetColor")
                when (colRaw) {
                    is JSONArray -> if (colRaw.length() >= 3) {
                        r = colRaw.optDouble(0, r.toDouble()).toFloat()
                        g = colRaw.optDouble(1, g.toDouble()).toFloat()
                        b = colRaw.optDouble(2, b.toDouble()).toFloat()
                    }
                    is String -> {
                        val trimmed = colRaw.trim()
                        if (trimmed.startsWith("$")) {
                            val key = trimmed.substring(1).removePrefix("param.")
                            val pVal = resolvedManifestParams[key]
                            if (pVal is FloatArray && pVal.size >= 3) {
                                r = pVal[0]
                                g = pVal[1]
                                b = pVal[2]
                            } else if (pVal is String && pVal.trim().startsWith("#")) {
                                val parsed = parseHexColorRgb(pVal.trim())
                                if (parsed != null) {
                                    r = parsed[0]; g = parsed[1]; b = parsed[2]
                                }
                            }
                        } else if (trimmed.startsWith("#")) {
                            val parsed = parseHexColorRgb(trimmed)
                            if (parsed != null) {
                                r = parsed[0]; g = parsed[1]; b = parsed[2]
                            }
                        }
                    }
                }
            }

            val rawStrength = resolveFloatParam(
                paramsJson = paramsJson,
                key = "strength",
                fallback = resolveFloatParam(paramsJson, "opacity", materialOpacity ?: DEFAULT_STRENGTH, resolvedManifestParams),
                resolvedManifestParams = resolvedManifestParams
            ).coerceIn(0f, MAX_DEFAULT_STRENGTH)
            val intensity = resolveFloatParam(paramsJson, "intensity", 1.0f, resolvedManifestParams).coerceIn(0f, 1f)
            val preserveLumaFallback = resolveFloatParam(
                paramsJson = paramsJson,
                key = "luminancePreserve",
                fallback = DEFAULT_PRESERVE_LUMINANCE,
                resolvedManifestParams = resolvedManifestParams
            )
            val preserveLuma = resolveFloatParam(paramsJson, "preserveLuminance", preserveLumaFallback, resolvedManifestParams)
                .coerceIn(0f, 1f)
            val glareLow = resolveFloatParam(paramsJson, "glareLowLuma", DEFAULT_GLARE_LOW_LUMA, resolvedManifestParams)
                .coerceIn(0.4f, 0.98f)
            val glareHigh = resolveFloatParam(paramsJson, "glareHighLuma", DEFAULT_GLARE_HIGH_LUMA, resolvedManifestParams)
                .coerceIn(glareLow + 0.02f, 1.0f)
            val featherPx = resolveFloatParam(paramsJson, "featherPx", DEFAULT_FEATHER_PX, resolvedManifestParams)
                .coerceIn(0.5f, 12.0f)
            val desaturateYellow = resolveFloatParam(paramsJson, "desaturateYellow", 0.65f, resolvedManifestParams)
                .coerceIn(0f, 1f)

            out["colorR"] = r.coerceIn(0f, 1f)
            out["colorG"] = g.coerceIn(0f, 1f)
            out["colorB"] = b.coerceIn(0f, 1f)
            out["strength"] = rawStrength
            out["intensity"] = intensity
            out["preserveLuminance"] = preserveLuma
            out["glareLowLuma"] = glareLow
            out["glareHighLuma"] = glareHigh
            out["featherPx"] = featherPx
            out["desaturateYellow"] = desaturateYellow
            if (paramsJson != null && paramsJson.has("minCavityLuma")) {
                out["minCavityLuma"] = resolveFloatParam(paramsJson, "minCavityLuma", 0.22f, resolvedManifestParams)
                    .coerceIn(0.02f, 0.85f)
            }
            if (paramsJson != null && paramsJson.has("mode")) {
                val rawMode = paramsJson.opt("mode")
                val modeFloat = when (rawMode) {
                    is Number -> rawMode.toFloat()
                    is String -> when (rawMode.trim().lowercase()) {
                        "teethwhiten", "teeth_whiten", "teeth" -> 2.0f
                        "sclerabrighten", "sclera_brighten", "sclera" -> 3.0f
                        "iris", "eyecolor", "eye_color" -> 1.0f
                        else -> rawMode.trim().toFloatOrNull() ?: 0.0f
                    }
                    else -> 0.0f
                }
                out["mode"] = modeFloat
            }
            return out
        }

        private fun parseHexColorRgb(hexStr: String): FloatArray? {
            val clean = hexStr.removePrefix("#").trim()
            return try {
                when (clean.length) {
                    6 -> {
                        val rgb = clean.toInt(16)
                        floatArrayOf(
                            ((rgb ushr 16) and 0xFF) / 255f,
                            ((rgb ushr 8) and 0xFF) / 255f,
                            (rgb and 0xFF) / 255f
                        )
                    }
                    8 -> {
                        val argb = clean.toLong(16).toInt()
                        floatArrayOf(
                            ((argb ushr 16) and 0xFF) / 255f,
                            ((argb ushr 8) and 0xFF) / 255f,
                            (argb and 0xFF) / 255f
                        )
                    }
                    else -> null
                }
            } catch (_: Throwable) {
                null
            }
        }

        private fun resolveFloatParam(
            paramsJson: JSONObject?,
            key: String,
            fallback: Float,
            resolvedManifestParams: Map<String, Any>
        ): Float {
            if (paramsJson == null || !paramsJson.has(key)) return fallback
            val raw = paramsJson.opt(key)
            if (raw is Number) return raw.toFloat()
            if (raw is String && raw.startsWith("$")) {
                val pName = raw.substring(1).removePrefix("param.")
                return (resolvedManifestParams[pName] as? Number)?.toFloat() ?: fallback
            }
            return fallback
        }
    }

    data class ExecutedRecolorRecord(
        val id: String,
        val maskId: String,
        val targetColorR: Float,
        val targetColorG: Float,
        val targetColorB: Float,
        val effectiveStrength: Float,
        val preserveLuminance: Float,
        val glareLowLuma: Float,
        val glareHighLuma: Float,
        val rightEyeOpenWeight: Float,
        val leftEyeOpenWeight: Float
    )

    private var programId: Int = 0
    private var vaoId: Int = 0
    private var vboId: Int = 0
    private var quadBuffer: FloatBuffer? = null
    private var sceneCopyTexId: Int = 0
    private var sceneCopyWidth: Int = 0
    private var sceneCopyHeight: Int = 0
    private var sceneCopyCapturedThisFrame: Boolean = false
    private var boundResourceOwner: GlResourceOwner? = null

    private val rightIrisCircleScratch = FloatArray(4)
    private val leftIrisCircleScratch = FloatArray(4)
    private val ptScratch = FloatArray(2)
    private val rgbScratch = FloatArray(3)
    private val regionMaskHelper by lazy { RegionMaskPass() }

    private val executedRecordsScratch = ArrayList<ExecutedRecolorRecord>(4)
    val lastExecutedRecords: List<ExecutedRecolorRecord>
        get() = executedRecordsScratch

    fun clearExecutedRecords() {
        executedRecordsScratch.clear()
        sceneCopyCapturedThisFrame = false
    }

    fun init(resourceOwner: GlResourceOwner? = null) {
        boundResourceOwner = resourceOwner
        programId = 0
        vaoId = 0
        vboId = 0
        sceneCopyTexId = 0
        sceneCopyWidth = 0
        sceneCopyHeight = 0
        sceneCopyCapturedThisFrame = false
        val quadData = floatArrayOf(
            // x, y, u, v
            -1f, -1f, 0f, 0f,
             1f, -1f, 1f, 0f,
            -1f,  1f, 0f, 1f,
             1f,  1f, 1f, 1f
        )
        quadBuffer = ByteBuffer.allocateDirect(quadData.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(quadData)
                position(0)
            }

        try {
            programId = compileRecolorProgram(VS_RECOLOR, FS_RECOLOR)
            if (programId != 0) {
                resourceOwner?.trackProgram(programId)
            }

            val vaos = IntArray(1)
            val vbos = IntArray(1)
            GLES30.glGenVertexArrays(1, vaos, 0)
            GLES30.glGenBuffers(1, vbos, 0)
            vaoId = vaos[0]
            vboId = vbos[0]
            if (vaoId != 0) resourceOwner?.trackVao(vaoId)
            if (vboId != 0) resourceOwner?.trackBuffer(vboId)

            if (vaoId != 0 && vboId != 0) {
                GLES30.glBindVertexArray(vaoId)
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboId)
                GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, quadData.size * 4, quadBuffer, GLES30.GL_STATIC_DRAW)
                GLES30.glEnableVertexAttribArray(0)
                GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, 0)
                GLES30.glEnableVertexAttribArray(1)
                GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 16, 8)
                GLES30.glBindVertexArray(0)
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
            }
        } catch (_: Throwable) {
            // Headless unit test safety
        }
    }

    private fun compileRecolorProgram(vsSource: String, fsSource: String): Int {
        val vs = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER)
        if (vs == 0) return 0
        GLES30.glShaderSource(vs, vsSource)
        GLES30.glCompileShader(vs)
        val status = IntArray(1)
        GLES30.glGetShaderiv(vs, GLES30.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            GLES30.glDeleteShader(vs)
            return 0
        }

        val fs = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER)
        if (fs == 0) {
            GLES30.glDeleteShader(vs)
            return 0
        }
        GLES30.glShaderSource(fs, fsSource)
        GLES30.glCompileShader(fs)
        GLES30.glGetShaderiv(fs, GLES30.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            GLES30.glDeleteShader(vs)
            GLES30.glDeleteShader(fs)
            return 0
        }

        val prog = GLES30.glCreateProgram()
        if (prog == 0) {
            GLES30.glDeleteShader(vs)
            GLES30.glDeleteShader(fs)
            return 0
        }
        GLES30.glAttachShader(prog, vs)
        GLES30.glAttachShader(prog, fs)
        GLES30.glLinkProgram(prog)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, status, 0)
        if (status[0] == 0) {
            GLES30.glDeleteProgram(prog)
            return 0
        }
        return prog
    }

    fun warmShaders() {
        if (programId != 0) {
            GLES30.glUseProgram(programId)
        }
    }

    /**
     * Renders a `recolor@1` / `eyeColor` node in Stage 2 (`CAMERA_PIXEL_PASSES`).
     * Supports dynamic parameter bindings via `PropertySlots` (`pass:<nodeId>.strength`, etc.)
     * and records an [ExecutedRecolorRecord] for telemetry/test inspection.
     */
    @JvmOverloads
    fun renderNode(
        node: CompiledNode,
        effect: CompiledEffect,
        faceState: FaceState,
        opacity: Float,
        viewportWidth: Int,
        viewportHeight: Int,
        cameraTextureId: Int = 0,
        maskTextureId: Int = 0
    ) {
        val mat = node.materialId?.let { effect.materials[it] }
        val slots = effect.propertySlots
        val tierParams = effect.tierOverrides[effect.activeTier]?.paramOverrides ?: emptyMap()
        val baseParams = node.postParams

        val r = resolveDynamic(slots, tierParams, node.id, "colorR", baseParams["colorR"] ?: mat?.color?.getOrNull(0) ?: 0.22f).coerceIn(0f, 1f)
        val g = resolveDynamic(slots, tierParams, node.id, "colorG", baseParams["colorG"] ?: mat?.color?.getOrNull(1) ?: 0.58f).coerceIn(0f, 1f)
        val b = resolveDynamic(slots, tierParams, node.id, "colorB", baseParams["colorB"] ?: mat?.color?.getOrNull(2) ?: 0.88f).coerceIn(0f, 1f)
        val rawStrength = resolveDynamic(slots, tierParams, node.id, "strength", baseParams["strength"] ?: mat?.opacity ?: DEFAULT_STRENGTH)
        val intensity = resolveDynamic(slots, tierParams, node.id, "intensity", baseParams["intensity"] ?: 1.0f).coerceIn(0f, 1f)
        val effectiveStrength = (rawStrength * intensity * opacity).coerceIn(0f, MAX_DEFAULT_STRENGTH)
        val preserveLuma = resolveDynamic(slots, tierParams, node.id, "preserveLuminance", baseParams["preserveLuminance"] ?: DEFAULT_PRESERVE_LUMINANCE)
            .coerceIn(0f, 1f)
        val glareLow = resolveDynamic(slots, tierParams, node.id, "glareLowLuma", baseParams["glareLowLuma"] ?: DEFAULT_GLARE_LOW_LUMA)
            .coerceIn(0.4f, 0.98f)
        val glareHigh = resolveDynamic(slots, tierParams, node.id, "glareHighLuma", baseParams["glareHighLuma"] ?: DEFAULT_GLARE_HIGH_LUMA)
            .coerceIn(glareLow + 0.02f, 1.0f)
        val featherPx = resolveDynamic(slots, tierParams, node.id, "featherPx", baseParams["featherPx"] ?: DEFAULT_FEATHER_PX)
            .coerceIn(0.5f, 12.0f)

        val blinkR = (faceState.blendshapes["eyeBlinkRight"] ?: 0f).coerceIn(0f, 1f)
        val blinkL = (faceState.blendshapes["eyeBlinkLeft"] ?: 0f).coerceIn(0f, 1f)
        val openR = (1f - blinkR).coerceIn(0f, 1f)
        val openL = (1f - blinkL).coerceIn(0f, 1f)

        val maskId = node.faceMeshRegion ?: "iris"
        val cleanMaskId = maskId.removePrefix("\$mask:").trim().lowercase().replace('-', '_')
        val isIrisMask = maskId.contains("iris", ignoreCase = true)
        val isTeethMask = cleanMaskId == "teeth"
        val isScleraMask = cleanMaskId == "sclera"
        val mouthOpenW = if (isTeethMask) computeMouthOpenGate(faceState) else 1f
        val desaturateYellow = resolveDynamic(slots, tierParams, node.id, "desaturateYellow", baseParams["desaturateYellow"] ?: 0.65f)
            .coerceIn(0f, 1f)
        val minCavityLuma = resolveDynamic(
            slots,
            tierParams,
            node.id,
            "minCavityLuma",
            baseParams["minCavityLuma"] ?: if (isTeethMask) 0.22f else 0.16f
        ).coerceIn(0.02f, 0.85f)

        executedRecordsScratch.add(
            ExecutedRecolorRecord(
                id = node.id,
                maskId = maskId,
                targetColorR = r,
                targetColorG = g,
                targetColorB = b,
                effectiveStrength = effectiveStrength,
                preserveLuminance = preserveLuma,
                glareLowLuma = glareLow,
                glareHighLuma = glareHigh,
                rightEyeOpenWeight = openR,
                leftEyeOpenWeight = openL
            )
        )

        if (effectiveStrength <= 0.001f ||
            ((isIrisMask || isScleraMask) && openR <= 0.001f && openL <= 0.001f) ||
            (isTeethMask && mouthOpenW <= 0.001f)
        ) {
            return
        }

        if (isIrisMask) {
            computeIrisCircleScreenPx(
                faceState = faceState,
                centerIndex = FaceRegions.RIGHT_IRIS_CENTER,
                ringIndices = FaceRegions.RIGHT_IRIS_RING,
                openWeight = openR,
                viewportWidth = viewportWidth,
                viewportHeight = viewportHeight,
                outCircle4 = rightIrisCircleScratch
            )
            computeIrisCircleScreenPx(
                faceState = faceState,
                centerIndex = FaceRegions.LEFT_IRIS_CENTER,
                ringIndices = FaceRegions.LEFT_IRIS_RING,
                openWeight = openL,
                viewportWidth = viewportWidth,
                viewportHeight = viewportHeight,
                outCircle4 = leftIrisCircleScratch
            )
        } else {
            rightIrisCircleScratch[0] = 0f; rightIrisCircleScratch[1] = 0f; rightIrisCircleScratch[2] = 0f; rightIrisCircleScratch[3] = 0f
            leftIrisCircleScratch[0] = 0f; leftIrisCircleScratch[1] = 0f; leftIrisCircleScratch[2] = 0f; leftIrisCircleScratch[3] = 0f
        }

        if (programId == 0) return

        // If cameraTextureId == 0 on an active GL context (e.g. SINGLE_SCENE_GL Path B after Pass 1),
        // copy the current framebuffer into sceneCopyTexId so luminance-preserving recoloring reads the real camera pixels
        var effectiveCamTexId = cameraTextureId
        if (effectiveCamTexId == 0 && viewportWidth > 0 && viewportHeight > 0 && hasActiveGlContext()) {
            effectiveCamTexId = captureCurrentFramebufferToTex(viewportWidth, viewportHeight)
        }

        val faceMidX = run {
            val lms = faceState.landmarks
            val base168 = 168 * 3
            if (lms != null && base168 + 1 < lms.size) {
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = lms[base168],
                    v = lms[base168 + 1],
                    z = if (base168 + 2 < lms.size) lms[base168 + 2] else 0f,
                    faceState = faceState,
                    viewWidth = 1.0f,
                    viewHeight = 1.0f,
                    outPt = ptScratch
                )
                ptScratch[0].coerceIn(0.1f, 0.9f)
            } else {
                0.5f
            }
        }
        val recolorMode = when {
            isTeethMask -> 2
            isScleraMask -> 3
            isIrisMask -> 1
            else -> 0
        }

        GLES30.glUseProgram(programId)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(false)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        val uCamLoc = GLES30.glGetUniformLocation(programId, "uCameraTexture")
        val uMaskLoc = GLES30.glGetUniformLocation(programId, "uMaskTexture")
        val uHasCamLoc = GLES30.glGetUniformLocation(programId, "uHasCameraTexture")
        val uHasMaskLoc = GLES30.glGetUniformLocation(programId, "uHasMaskTexture")
        val uIsIrisLoc = GLES30.glGetUniformLocation(programId, "uIsIrisMask")
        val uModeLoc = GLES30.glGetUniformLocation(programId, "uRecolorMode")
        val uColLoc = GLES30.glGetUniformLocation(programId, "uTargetColor")
        val uStrLoc = GLES30.glGetUniformLocation(programId, "uStrength")
        val uPresLoc = GLES30.glGetUniformLocation(programId, "uPreserveLuminance")
        val uGlareLoc = GLES30.glGetUniformLocation(programId, "uGlareRange")
        val uPupilLoc = GLES30.glGetUniformLocation(programId, "uPupilRange")
        val uRightLoc = GLES30.glGetUniformLocation(programId, "uRightIrisCirclePx")
        val uLeftLoc = GLES30.glGetUniformLocation(programId, "uLeftIrisCirclePx")
        val uVpLoc = GLES30.glGetUniformLocation(programId, "uViewportSize")
        val uFeatherLoc = GLES30.glGetUniformLocation(programId, "uFeatherPx")
        val uMouthOpenLoc = GLES30.glGetUniformLocation(programId, "uMouthOpenWeight")
        val uMinCavityLoc = GLES30.glGetUniformLocation(programId, "uMinCavityLuma")
        val uDesatYellowLoc = GLES30.glGetUniformLocation(programId, "uDesaturateYellow")
        val uEyeOpenLoc = GLES30.glGetUniformLocation(programId, "uEyeOpenWeights")
        val uFaceMidXLoc = GLES30.glGetUniformLocation(programId, "uFaceMidX")

        if (effectiveCamTexId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, effectiveCamTexId)
            GLES30.glUniform1i(uCamLoc, 0)
            GLES30.glUniform1i(uHasCamLoc, 1)
        } else {
            GLES30.glUniform1i(uHasCamLoc, 0)
        }

        if (maskTextureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTextureId)
            GLES30.glUniform1i(uMaskLoc, 1)
            GLES30.glUniform1i(uHasMaskLoc, 1)
        } else {
            GLES30.glUniform1i(uHasMaskLoc, 0)
        }

        GLES30.glUniform1i(uIsIrisLoc, if (isIrisMask) 1 else 0)
        GLES30.glUniform1i(uModeLoc, recolorMode)
        GLES30.glUniform3f(uColLoc, r, g, b)
        GLES30.glUniform1f(uStrLoc, effectiveStrength)
        GLES30.glUniform1f(uPresLoc, preserveLuma)
        GLES30.glUniform2f(uGlareLoc, glareLow, glareHigh)
        GLES30.glUniform2f(uPupilLoc, DEFAULT_PUPIL_LOW_LUMA, DEFAULT_PUPIL_HIGH_LUMA)
        GLES30.glUniform4fv(uRightLoc, 1, rightIrisCircleScratch, 0)
        GLES30.glUniform4fv(uLeftLoc, 1, leftIrisCircleScratch, 0)
        GLES30.glUniform2f(uVpLoc, viewportWidth.toFloat(), viewportHeight.toFloat())
        GLES30.glUniform1f(uFeatherLoc, featherPx)
        GLES30.glUniform1f(uMouthOpenLoc, mouthOpenW)
        GLES30.glUniform1f(uMinCavityLoc, minCavityLuma)
        GLES30.glUniform1f(uDesatYellowLoc, desaturateYellow)
        GLES30.glUniform2f(uEyeOpenLoc, openR, openL)
        GLES30.glUniform1f(uFaceMidXLoc, faceMidX)

        if (vaoId != 0) {
            GLES30.glBindVertexArray(vaoId)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
            GLES30.glBindVertexArray(0)
        }
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
    }

    private fun captureCurrentFramebufferToTex(width: Int, height: Int): Int {
        return try {
            val w = width.coerceAtLeast(1)
            val h = height.coerceAtLeast(1)
            if (sceneCopyCapturedThisFrame && sceneCopyTexId != 0 && sceneCopyWidth == w && sceneCopyHeight == h) {
                return sceneCopyTexId
            }
            if (sceneCopyTexId == 0) {
                val texs = IntArray(1)
                GLES30.glGenTextures(1, texs, 0)
                sceneCopyTexId = texs[0]
                if (sceneCopyTexId != 0) {
                    boundResourceOwner?.trackTexture(sceneCopyTexId)
                }
            }
            if (sceneCopyTexId == 0) return 0
            val prevState = IntArray(3)
            GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevState, 0)
            GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, prevState, 1)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glGetIntegerv(GLES30.GL_TEXTURE_BINDING_2D, prevState, 2)
            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevState[0])
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, sceneCopyTexId)
            if (sceneCopyWidth != w || sceneCopyHeight != h) {
                sceneCopyWidth = w
                sceneCopyHeight = h
                GLES30.glTexImage2D(
                    GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8,
                    w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
                )
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            }
            GLES30.glCopyTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, prevState[2])
            GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevState[1])
            sceneCopyCapturedThisFrame = true
            sceneCopyTexId
        } catch (_: Throwable) {
            0
        }
    }

    private fun hasActiveGlContext(): Boolean {
        return try {
            val ctx = EGL14.eglGetCurrentContext()
            ctx != null && ctx != EGL14.EGL_NO_CONTEXT
        } catch (_: Throwable) {
            false
        }
    }

    /**
     * Applies a `recolor@1` / `eyeColor` node directly to `bitmap` on the CPU (used by
     * `OffscreenRenderer.renderCpuFallback` and deterministic conformance tests).
     * Supports both iris masks (`$mask:iris`) and general region masks (`$mask:lips`, `$mask:cheeks`, etc.).
     *
     * @return Number of modified pixels.
     */
    fun applyNodeToBitmap(
        bitmap: Bitmap,
        node: CompiledNode,
        effect: CompiledEffect,
        faceState: FaceState,
        opacity: Float
    ): Int {
        val lm = faceState.landmarks ?: return 0
        val maskId = node.faceMeshRegion ?: "iris"
        val isIrisMask = maskId.contains("iris", ignoreCase = true)
        val minRequiredLandmarks = if (isIrisMask) 478 * 3 else UvRegistry.VERTEX_COUNT * 3
        if (lm.size < minRequiredLandmarks) return 0
        val width = bitmap.width
        val height = bitmap.height
        if (width <= 0 || height <= 0) return 0

        val mat = node.materialId?.let { effect.materials[it] }
        val slots = effect.propertySlots
        val tierParams = effect.tierOverrides[effect.activeTier]?.paramOverrides ?: emptyMap()
        val baseParams = node.postParams

        val targetR = resolveDynamic(slots, tierParams, node.id, "colorR", baseParams["colorR"] ?: mat?.color?.getOrNull(0) ?: 0.22f).coerceIn(0f, 1f)
        val targetG = resolveDynamic(slots, tierParams, node.id, "colorG", baseParams["colorG"] ?: mat?.color?.getOrNull(1) ?: 0.58f).coerceIn(0f, 1f)
        val targetB = resolveDynamic(slots, tierParams, node.id, "colorB", baseParams["colorB"] ?: mat?.color?.getOrNull(2) ?: 0.88f).coerceIn(0f, 1f)
        val rawStrength = resolveDynamic(slots, tierParams, node.id, "strength", baseParams["strength"] ?: mat?.opacity ?: DEFAULT_STRENGTH)
        val intensity = resolveDynamic(slots, tierParams, node.id, "intensity", baseParams["intensity"] ?: 1.0f).coerceIn(0f, 1f)
        val effectiveStrength = (rawStrength * intensity * opacity).coerceIn(0f, MAX_DEFAULT_STRENGTH)
        val preserveLuma = resolveDynamic(slots, tierParams, node.id, "preserveLuminance", baseParams["preserveLuminance"] ?: DEFAULT_PRESERVE_LUMINANCE)
            .coerceIn(0f, 1f)
        val glareLow = resolveDynamic(slots, tierParams, node.id, "glareLowLuma", baseParams["glareLowLuma"] ?: DEFAULT_GLARE_LOW_LUMA)
            .coerceIn(0.4f, 0.98f)
        val glareHigh = resolveDynamic(slots, tierParams, node.id, "glareHighLuma", baseParams["glareHighLuma"] ?: DEFAULT_GLARE_HIGH_LUMA)
            .coerceIn(glareLow + 0.02f, 1.0f)
        val featherPx = resolveDynamic(slots, tierParams, node.id, "featherPx", baseParams["featherPx"] ?: DEFAULT_FEATHER_PX)
            .coerceIn(0.5f, 12.0f)

        val blinkR = (faceState.blendshapes["eyeBlinkRight"] ?: 0f).coerceIn(0f, 1f)
        val blinkL = (faceState.blendshapes["eyeBlinkLeft"] ?: 0f).coerceIn(0f, 1f)
        val openR = (1f - blinkR).coerceIn(0f, 1f)
        val openL = (1f - blinkL).coerceIn(0f, 1f)

        executedRecordsScratch.add(
            ExecutedRecolorRecord(
                id = node.id,
                maskId = maskId,
                targetColorR = targetR,
                targetColorG = targetG,
                targetColorB = targetB,
                effectiveStrength = effectiveStrength,
                preserveLuminance = preserveLuma,
                glareLowLuma = glareLow,
                glareHighLuma = glareHigh,
                rightEyeOpenWeight = openR,
                leftEyeOpenWeight = openL
            )
        )

        val cleanMaskId = maskId.removePrefix("\$mask:").trim().lowercase().replace('-', '_')
        val isTeethMask = cleanMaskId == "teeth"
        val isScleraMask = cleanMaskId == "sclera"
        val mouthOpenW = if (isTeethMask) computeMouthOpenGate(faceState) else 1f
        val desaturateYellow = resolveDynamic(slots, tierParams, node.id, "desaturateYellow", baseParams["desaturateYellow"] ?: 0.65f)
            .coerceIn(0f, 1f)
        val minCavityLuma = resolveDynamic(
            slots,
            tierParams,
            node.id,
            "minCavityLuma",
            baseParams["minCavityLuma"] ?: if (isTeethMask) 0.22f else 0.16f
        ).coerceIn(0.02f, 0.85f)

        if (effectiveStrength <= 0.001f ||
            ((isIrisMask || isScleraMask) && openR <= 0.001f && openL <= 0.001f) ||
            (isTeethMask && mouthOpenW <= 0.001f)
        ) {
            return 0
        }

        if (!isIrisMask) {
            // General region mask ($mask:lips, $mask:cheeks, $mask:teeth, $mask:sclera, $mask:skin, etc.)
            var cranialDx = 0f
            var cranialDy = 0f
            if (lm.size >= 153 * 3) {
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = lm[10 * 3],
                    v = lm[10 * 3 + 1],
                    z = lm[10 * 3 + 2],
                    faceState = faceState,
                    viewWidth = 1.0f,
                    viewHeight = 1.0f,
                    outPt = ptScratch
                )
                val fX = ptScratch[0]
                val fY = ptScratch[1]
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = lm[152 * 3],
                    v = lm[152 * 3 + 1],
                    z = lm[152 * 3 + 2],
                    faceState = faceState,
                    viewWidth = 1.0f,
                    viewHeight = 1.0f,
                    outPt = ptScratch
                )
                cranialDx = fX - ptScratch[0]
                cranialDy = fY - ptScratch[1]
            }

            val boundLoop = if (isTeethMask) FaceRegions.MOUTH_INNER else FaceRegions.FACE_OVAL
            val isOvalLoop = boundLoop === FaceRegions.FACE_OVAL
            var minNormX = 1f
            var minNormY = 1f
            var maxNormX = 0f
            var maxNormY = 0f
            for (idx in boundLoop) {
                val base = idx * 3
                if (base + 1 < lm.size) {
                    FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                        u = lm[base],
                        v = lm[base + 1],
                        z = if (base + 2 < lm.size) lm[base + 2] else 0f,
                        faceState = faceState,
                        viewWidth = 1.0f,
                        viewHeight = 1.0f,
                        outPt = ptScratch
                    )
                    val cranialW = if (isOvalLoop) RegionMaskPass.cranialForeheadWeightForLandmark(idx) else 0f
                    val pxNorm = (ptScratch[0] + cranialDx * cranialW).coerceIn(0f, 1f)
                    val pyNorm = (ptScratch[1] + cranialDy * cranialW).coerceIn(0f, 1f)
                    if (pxNorm < minNormX) minNormX = pxNorm
                    if (pyNorm < minNormY) minNormY = pyNorm
                    if (pxNorm > maxNormX) maxNormX = pxNorm
                    if (pyNorm > maxNormY) maxNormY = pyNorm
                }
            }
            val padNorm = 0.05f
            val minX = ((minNormX - padNorm) * width).roundToInt().coerceIn(0, width - 1)
            val maxX = ((maxNormX + padNorm) * width).roundToInt().coerceIn(0, width - 1)
            val minY = ((minNormY - padNorm) * height).roundToInt().coerceIn(0, height - 1)
            val maxY = ((maxNormY + padNorm) * height).roundToInt().coerceIn(0, height - 1)

            val faceMidX = run {
                val base168 = 168 * 3
                if (base168 + 1 < lm.size) {
                    FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                        u = lm[base168],
                        v = lm[base168 + 1],
                        z = if (base168 + 2 < lm.size) lm[base168 + 2] else 0f,
                        faceState = faceState,
                        viewWidth = 1.0f,
                        viewHeight = 1.0f,
                        outPt = ptScratch
                    )
                    ptScratch[0].coerceIn(0.1f, 0.9f)
                } else {
                    0.5f
                }
            }

            val expr = effect.maskDefs[cleanMaskId]
                ?: effect.maskDefs[maskId]
                ?: RegionMaskPass.BUILTIN_MASKS[cleanMaskId]
            val maxSide = kotlin.math.max(width, height).coerceAtLeast(1)
            val evalScale = if (maxSide > 256) 256.0f / maxSide.toFloat() else 1.0f
            val mw = kotlin.math.max(1, (width * evalScale).roundToInt())
            val mh = kotlin.math.max(1, (height * evalScale).roundToInt())
            val maskBytes: ByteArray? = if (expr != null && maxSide > 256 && !isTeethMask) {
                val lumaBuf = FloatArray(mw * mh)
                for (gy in 0 until mh) {
                    val sy = ((gy + 0.5f) / mh.toFloat() * height).toInt().coerceIn(0, height - 1)
                    val rowOff = gy * mw
                    for (gx in 0 until mw) {
                        val sx = ((gx + 0.5f) / mw.toFloat() * width).toInt().coerceIn(0, width - 1)
                        val sc = bitmap.getPixel(sx, sy)
                        lumaBuf[rowOff + gx] = luminance(
                            Color.red(sc) / 255f,
                            Color.green(sc) / 255f,
                            Color.blue(sc) / 255f
                        )
                    }
                }
                regionMaskHelper.evaluateMaskExprToHalfResR8(
                    expr = expr,
                    faceState = faceState,
                    viewportWidth = mw * 2,
                    viewportHeight = mh * 2,
                    lumaBuffer = lumaBuf,
                    maskWeight = 1.0f,
                    useSmoothedLandmarks = false
                )
            } else {
                null
            }

            var modifiedPixels = 0
            for (py in minY..maxY) {
                val ny = (py + 0.5f) / height.toFloat()
                for (px in minX..maxX) {
                    val nx = (px + 0.5f) / width.toFloat()
                    val c = bitmap.getPixel(px, py)
                    val a = Color.alpha(c)
                    val baseR = Color.red(c) / 255f
                    val baseG = Color.green(c) / 255f
                    val baseB = Color.blue(c) / 255f
                    val pixelLuma = luminance(baseR, baseG, baseB)

                    val rawMaskW = if (maskBytes != null && mw > 0 && mh > 0 && maskBytes.size >= mw * mh) {
                        val gx = (nx.coerceIn(0f, 1f) * (mw - 1)).coerceIn(0f, (mw - 1).toFloat())
                        val gy = (ny.coerceIn(0f, 1f) * (mh - 1)).coerceIn(0f, (mh - 1).toFloat())
                        val x0 = gx.toInt().coerceIn(0, mw - 1)
                        val y0 = gy.toInt().coerceIn(0, mh - 1)
                        val x1 = (x0 + 1).coerceAtMost(mw - 1)
                        val y1 = (y0 + 1).coerceAtMost(mh - 1)
                        val fx = gx - x0
                        val fy = gy - y0
                        val v00 = (maskBytes[y0 * mw + x0].toInt() and 0xFF) / 255f
                        val v10 = (maskBytes[y0 * mw + x1].toInt() and 0xFF) / 255f
                        val v01 = (maskBytes[y1 * mw + x0].toInt() and 0xFF) / 255f
                        val v11 = (maskBytes[y1 * mw + x1].toInt() and 0xFF) / 255f
                        val top = v00 + (v10 - v00) * fx
                        val bot = v01 + (v11 - v01) * fx
                        (top + (bot - top) * fy).coerceIn(0f, 1f)
                    } else {
                        regionMaskHelper.evaluateNamedMaskAtNormalizedViewPt(
                            maskId = maskId,
                            normX = nx,
                            normY = ny,
                            faceState = faceState,
                            customMasks = effect.maskDefs,
                            sampleLuma = pixelLuma,
                            useSmoothedLandmarks = false
                        )
                    }
                    val modeGate = when {
                        isTeethMask -> mouthOpenW
                        isScleraMask -> if (nx < faceMidX) openR else openL
                        else -> 1f
                    }
                    val maskW = (rawMaskW * modeGate).coerceIn(0f, 1f)
                    if (maskW <= 0.005f) continue

                    val appliedW = if (isTeethMask || isScleraMask) {
                        whitenTeethOrScleraPixelRgb(
                            baseR = baseR,
                            baseG = baseG,
                            baseB = baseB,
                            maskWeight = maskW,
                            whiteningStrength = effectiveStrength,
                            desaturateYellow = desaturateYellow,
                            minCavityLuma = minCavityLuma,
                            glareLowLuma = glareLow,
                            glareHighLuma = glareHigh,
                            outRgb = rgbScratch,
                            protectMucosaRedness = isTeethMask
                        )
                    } else {
                        recolorPixelRgb(
                            baseR = baseR,
                            baseG = baseG,
                            baseB = baseB,
                            targetR = targetR,
                            targetG = targetG,
                            targetB = targetB,
                            maskWeight = maskW,
                            strength = effectiveStrength,
                            preserveLuminance = preserveLuma,
                            glareLowLuma = glareLow,
                            glareHighLuma = glareHigh,
                            allowOverStrength = false,
                            outRgb = rgbScratch,
                            protectPupil = false
                        )
                    }
                    if (appliedW > 0.005f) {
                        val outColor = Color.argb(
                            a.coerceAtLeast(255),
                            (rgbScratch[0] * 255f).roundToInt().coerceIn(0, 255),
                            (rgbScratch[1] * 255f).roundToInt().coerceIn(0, 255),
                            (rgbScratch[2] * 255f).roundToInt().coerceIn(0, 255)
                        )
                        bitmap.setPixel(px, py, outColor)
                        modifiedPixels++
                    }
                }
            }
            return modifiedPixels
        }

        computeIrisCircleScreenPx(
            faceState = faceState,
            centerIndex = FaceRegions.RIGHT_IRIS_CENTER,
            ringIndices = FaceRegions.RIGHT_IRIS_RING,
            openWeight = openR,
            viewportWidth = width,
            viewportHeight = height,
            outCircle4 = rightIrisCircleScratch
        )
        computeIrisCircleScreenPx(
            faceState = faceState,
            centerIndex = FaceRegions.LEFT_IRIS_CENTER,
            ringIndices = FaceRegions.LEFT_IRIS_RING,
            openWeight = openL,
            viewportWidth = width,
            viewportHeight = height,
            outCircle4 = leftIrisCircleScratch
        )

        var modifiedPixels = 0
        val circles = arrayOf(rightIrisCircleScratch, leftIrisCircleScratch)
        for (circle in circles) {
            val cx = circle[0]
            val cy = circle[1]
            val rad = circle[2]
            val openW = circle[3]
            if (rad <= 0.5f || openW <= 0.001f) continue

            val pad = (rad + featherPx + 2f).roundToInt()
            val minX = (cx.roundToInt() - pad).coerceIn(0, width - 1)
            val maxX = (cx.roundToInt() + pad).coerceIn(0, width - 1)
            val minY = (cy.roundToInt() - pad).coerceIn(0, height - 1)
            val maxY = (cy.roundToInt() + pad).coerceIn(0, height - 1)

            for (py in minY..maxY) {
                for (px in minX..maxX) {
                    val maskW = evaluateIrisMaskWeightAtPixel(
                        xPx = px.toFloat(),
                        yPx = py.toFloat(),
                        faceState = faceState,
                        viewportWidth = width,
                        viewportHeight = height,
                        featherPx = featherPx
                    )
                    if (maskW <= 0.005f) continue

                    val c = bitmap.getPixel(px, py)
                    val a = Color.alpha(c)
                    val baseR = Color.red(c) / 255f
                    val baseG = Color.green(c) / 255f
                    val baseB = Color.blue(c) / 255f

                    val appliedW = recolorPixelRgb(
                        baseR = baseR,
                        baseG = baseG,
                        baseB = baseB,
                        targetR = targetR,
                        targetG = targetG,
                        targetB = targetB,
                        maskWeight = maskW,
                        strength = effectiveStrength,
                        preserveLuminance = preserveLuma,
                        glareLowLuma = glareLow,
                        glareHighLuma = glareHigh,
                        allowOverStrength = false,
                        outRgb = rgbScratch
                    )
                    if (appliedW > 0.005f) {
                        val outColor = Color.argb(
                            a.coerceAtLeast(255),
                            (rgbScratch[0] * 255f).roundToInt().coerceIn(0, 255),
                            (rgbScratch[1] * 255f).roundToInt().coerceIn(0, 255),
                            (rgbScratch[2] * 255f).roundToInt().coerceIn(0, 255)
                        )
                        bitmap.setPixel(px, py, outColor)
                        modifiedPixels++
                    }
                }
            }
        }
        return modifiedPixels
    }

    private fun resolveDynamic(
        slots: PropertySlots,
        tierParams: Map<String, Float>,
        nodeId: String,
        paramName: String,
        defaultVal: Float
    ): Float {
        val passKey = "pass:$nodeId.$paramName"
        val postKey = "post:$nodeId.$paramName"
        val bareKey = "$nodeId.$paramName"
        val tierVal = tierParams[passKey] ?: tierParams[postKey] ?: tierParams[bareKey]
        if (tierVal != null) return tierVal
        val byPass = slots.getDynamicParam(passKey, Float.NaN)
        if (!byPass.isNaN()) return byPass
        val byMat = slots.getDynamicParam("mat:$nodeId.$paramName", Float.NaN)
        if (!byMat.isNaN()) return byMat
        return defaultVal
    }

    private fun computeIrisCircleScreenPx(
        faceState: FaceState,
        centerIndex: Int,
        ringIndices: IntArray,
        openWeight: Float,
        viewportWidth: Int,
        viewportHeight: Int,
        outCircle4: FloatArray
    ) {
        val lm = faceState.landmarks
        if (lm == null || lm.size < 478 * 3) {
            outCircle4[0] = 0f; outCircle4[1] = 0f; outCircle4[2] = 0f; outCircle4[3] = 0f
            return
        }
        val vw = viewportWidth.coerceAtLeast(1).toFloat()
        val vh = viewportHeight.coerceAtLeast(1).toFloat()
        val cBase = centerIndex * 3
        FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
            u = lm[cBase],
            v = lm[cBase + 1],
            z = if (cBase + 2 < lm.size) lm[cBase + 2] else 0f,
            faceState = faceState,
            viewWidth = vw,
            viewHeight = vh,
            outPt = ptScratch
        )
        val cx = ptScratch[0]
        val cy = ptScratch[1]
        var sumR = 0f
        var count = 0
        for (idx in ringIndices) {
            val rBase = idx * 3
            if (rBase + 1 < lm.size) {
                FacePoseAdapter.mapLandmarkToViewWithPoseDelta(
                    u = lm[rBase],
                    v = lm[rBase + 1],
                    z = if (rBase + 2 < lm.size) lm[rBase + 2] else 0f,
                    faceState = faceState,
                    viewWidth = vw,
                    viewHeight = vh,
                    outPt = ptScratch
                )
                sumR += hypot(ptScratch[0] - cx, ptScratch[1] - cy)
                count++
            }
        }
        val radiusPx = if (count > 0) (sumR / count.toFloat()) else 0f
        outCircle4[0] = cx
        outCircle4[1] = cy
        outCircle4[2] = radiusPx
        outCircle4[3] = openWeight.coerceIn(0f, 1f)
    }

    fun release(resourceOwner: GlResourceOwner? = null) {
        if (sceneCopyTexId != 0) {
            if (resourceOwner != null) {
                resourceOwner.queueDeleteTexture(sceneCopyTexId)
            } else {
                try { GLES30.glDeleteTextures(1, intArrayOf(sceneCopyTexId), 0) } catch (_: Throwable) {}
            }
            sceneCopyTexId = 0
            sceneCopyWidth = 0
            sceneCopyHeight = 0
        }
        if (vaoId != 0) {
            if (resourceOwner != null) {
                resourceOwner.queueDeleteVao(vaoId)
            } else {
                try { GLES30.glDeleteVertexArrays(1, intArrayOf(vaoId), 0) } catch (_: Throwable) {}
            }
            vaoId = 0
        }
        if (vboId != 0) {
            if (resourceOwner != null) {
                resourceOwner.queueDeleteBuffer(vboId)
            } else {
                try { GLES30.glDeleteBuffers(1, intArrayOf(vboId), 0) } catch (_: Throwable) {}
            }
            vboId = 0
        }
        if (programId != 0) {
            if (resourceOwner != null) {
                resourceOwner.queueDeleteProgram(programId)
            } else {
                try { GLES30.glDeleteProgram(programId) } catch (_: Throwable) {}
            }
            programId = 0
        }
    }
}
