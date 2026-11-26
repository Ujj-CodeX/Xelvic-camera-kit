package com.example.engine.render

import android.opengl.GLES30
import android.util.Log

/**
 * MaterialRegistry: Owns compiled GL shader programs for engine material templates:
 * unlit@1, lambert@1, faceTexture@1.
 */
class MaterialRegistry {

    companion object {
        private const val TAG = "MaterialRegistry"

        // Unlit Shaders (with optional uUvRect flipbook window: (u0, v0, uScale, vScale))
        internal const val VS_UNLIT = """#version 300 es
            layout(location = 0) in vec3 aPosition;
            layout(location = 2) in vec2 aTexCoord;
            uniform mat4 uMVP;
            uniform vec4 uUvRect;
            out vec2 vTexCoord;
            void main() {
                vec2 uvScale = (uUvRect.z > 0.0001 && uUvRect.w > 0.0001) ? uUvRect.zw : vec2(1.0, 1.0);
                vTexCoord = uUvRect.xy + aTexCoord * uvScale;
                gl_Position = uMVP * vec4(aPosition, 1.0);
            }
        """

        internal const val FS_UNLIT = """#version 300 es
            precision mediump float;
            in vec2 vTexCoord;
            uniform sampler2D uTexture;
            uniform vec4 uColor;
            uniform int uHasTexture;
            out vec4 fragColor;
            void main() {
                vec4 tex = (uHasTexture == 1) ? texture(uTexture, vTexCoord) : vec4(1.0);
                // Premultiplied alpha
                vec4 col = tex * uColor;
                fragColor = vec4(col.rgb * col.a, col.a);
            }
        """

        // Lambert Shaders (with view-space vViewPos, normal matrix, and optional Blinn-Phong specular + Fresnel rim sheen)
        internal const val VS_LAMBERT = """#version 300 es
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec2 aTexCoord;
            uniform mat4 uMVP;
            uniform mat4 uModelView;
            out vec3 vNormal;
            out vec3 vViewPos;
            out vec2 vTexCoord;
            void main() {
                vec4 viewPos4 = uModelView * vec4(aPosition, 1.0);
                vViewPos = viewPos4.xyz;
                mat3 normalMat = transpose(inverse(mat3(uModelView)));
                vNormal = normalMat * aNormal;
                vTexCoord = aTexCoord;
                gl_Position = uMVP * vec4(aPosition, 1.0);
            }
        """

        internal const val FS_LAMBERT = """#version 300 es
            precision mediump float;
            in vec3 vNormal;
            in vec3 vViewPos;
            in vec2 vTexCoord;
            uniform sampler2D uTexture;
            uniform vec4 uColor;
            uniform int uHasTexture;
            uniform vec3 uAmbient;
            uniform int uLightCount;
            uniform vec3 uLightDir[2];
            uniform vec3 uLightColor[2];
            uniform float uLightIntensity[2];
            uniform int uDoubleSided;
            uniform float uFresnelSheen;
            uniform int uAlphaMode;
            uniform float uAlphaCutoff;
            out vec4 fragColor;
            void main() {
                vec4 tex = (uHasTexture == 1) ? texture(uTexture, vTexCoord) : vec4(1.0);
                vec4 col = tex * uColor;
                if (uAlphaMode == 1 && col.a < uAlphaCutoff) {
                    discard;
                }
                float nLen = length(vNormal);
                vec3 N = (nLen > 0.0001) ? (vNormal / nLen) : vec3(0.0, 0.0, 1.0);
                if (uDoubleSided == 1 && !gl_FrontFacing) {
                    N = -N;
                }
                float vLen = length(vViewPos);
                vec3 V = (vLen > 0.0001) ? (-vViewPos / vLen) : vec3(0.0, 0.0, 1.0);
                float NdotV = max(dot(N, V), 0.0);

                vec3 diffuse = vec3(0.0);
                vec3 specular = vec3(0.0);
                if (uLightCount >= 1) {
                    float lLen0 = length(uLightDir[0]);
                    vec3 L0 = (lLen0 > 0.0001) ? (uLightDir[0] / lLen0) : vec3(0.0, 0.6, 0.8);
                    float NdotL0 = max(dot(N, L0), 0.0);
                    diffuse = uLightColor[0] * (NdotL0 * uLightIntensity[0]);
                    if (NdotL0 > 0.0) {
                        vec3 H0 = normalize(L0 + V);
                        float spec0 = pow(max(dot(N, H0), 0.0), 28.0) * 0.14;
                        specular += uLightColor[0] * (spec0 * uLightIntensity[0]);
                    }
                }
                if (uLightCount >= 2) {
                    float lLen1 = length(uLightDir[1]);
                    vec3 L1 = (lLen1 > 0.0001) ? (uLightDir[1] / lLen1) : vec3(0.0, 0.6, 0.8);
                    float NdotL1 = max(dot(N, L1), 0.0);
                    diffuse += uLightColor[1] * (NdotL1 * uLightIntensity[1]);
                    if (NdotL1 > 0.0) {
                        vec3 H1 = normalize(L1 + V);
                        float spec1 = pow(max(dot(N, H1), 0.0), 28.0) * 0.10;
                        specular += uLightColor[1] * (spec1 * uLightIntensity[1]);
                    }
                }
                float fresnel = pow(1.0 - NdotV, 3.0) * clamp(uFresnelSheen, 0.0, 0.5);
                vec3 lighting = clamp(uAmbient + diffuse, 0.0, 1.0);
                vec3 specAndRim = specular + vec3(fresnel * 0.25);
                float baseA = (uAlphaMode == 1) ? 1.0 : clamp(col.a, 0.0, 1.0);
                float specPeak = max(specAndRim.r, max(specAndRim.g, specAndRim.b));
                float outAlpha = clamp(baseA + specPeak * (1.0 - baseA) * 0.85, 0.0, 1.0);
                vec3 premulRgb = clamp(col.rgb * lighting * baseA + specAndRim * mix(baseA, 1.0, 0.85), 0.0, 1.0);
                fragColor = vec4(premulRgb, outAlpha);
            }
        """

        // PBR Shaders (material.pbr@1 — Linear sRGB workflow, Cook-Torrance GGX, Smith-Schlick G, Roughness-Aware Schlick Fresnel, Split-Sum Studio IBL)
        internal const val VS_PBR = """#version 300 es
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec2 aTexCoord;
            uniform mat4 uMVP;
            uniform mat4 uModelView;
            out vec3 vNormal;
            out vec3 vViewPos;
            out vec2 vTexCoord;
            void main() {
                vec4 viewPos4 = uModelView * vec4(aPosition, 1.0);
                vViewPos = viewPos4.xyz;
                mat3 normalMat = transpose(inverse(mat3(uModelView)));
                vNormal = normalMat * aNormal;
                vTexCoord = aTexCoord;
                gl_Position = uMVP * vec4(aPosition, 1.0);
            }
        """

        internal const val FS_PBR = """#version 300 es
            precision highp float;
            in vec3 vNormal;
            in vec3 vViewPos;
            in vec2 vTexCoord;
            uniform sampler2D uTexture;
            uniform sampler2D uMetallicRoughnessMap;
            uniform sampler2D uNormalMap;
            uniform sampler2D uOcclusionMap;
            uniform sampler2D uEmissiveMap;
            uniform vec4 uColor;
            uniform int uHasTexture;
            uniform int uHasMetallicRoughnessMap;
            uniform int uHasNormalMap;
            uniform int uHasOcclusionMap;
            uniform int uHasEmissiveMap;
            uniform float uMetallic;
            uniform float uRoughness;
            uniform float uNormalScale;
            uniform float uOcclusionStrength;
            uniform float uFresnelSheen;
            uniform vec3 uEmissive;
            uniform vec3 uAmbient;
            uniform int uLightCount;
            uniform vec3 uLightDir[2];
            uniform vec3 uLightColor[2];
            uniform float uLightIntensity[2];
            uniform int uDoubleSided;
            uniform int uAlphaMode;
            uniform float uAlphaCutoff;
            out vec4 fragColor;

            const float PI = 3.14159265359;

            float distributionGGX(float NdotH, float rough) {
                float a = rough * rough;
                float a2 = a * a;
                float NdotH2 = NdotH * NdotH;
                float denom = (NdotH2 * (a2 - 1.0) + 1.0);
                return a2 / max(PI * denom * denom, 0.0001);
            }

            float geometrySchlickGGX(float NdotV, float rough) {
                float r = (rough + 1.0);
                float k = (r * r) / 8.0;
                return NdotV / max(NdotV * (1.0 - k) + k, 0.0001);
            }

            float geometrySmith(float NdotV, float NdotL, float rough) {
                return geometrySchlickGGX(NdotV, rough) * geometrySchlickGGX(NdotL, rough);
            }

            vec3 fresnelSchlick(float cosTheta, vec3 F0) {
                return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
            }

            vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float rough) {
                return F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
            }

            vec2 envBRDFApprox(float rough, float NdotV) {
                const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
                const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
                vec4 r = rough * c0 + c1;
                float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
                return clamp(vec2(-1.04, 1.04) * a004 + r.zw, 0.0, 1.0);
            }

            vec3 perturbNormalCotangent(vec3 N, vec3 p, vec2 uv, vec3 mapN) {
                vec3 dp1 = dFdx(p);
                vec3 dp2 = dFdy(p);
                vec2 duv1 = dFdx(uv);
                vec2 duv2 = dFdy(uv);
                vec3 dp2perp = cross(dp2, N);
                vec3 dp1perp = cross(N, dp1);
                vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
                vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
                float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1e-6);
                mat3 TBN = mat3(T * invmax, B * invmax, N);
                return normalize(TBN * mapN);
            }

            void main() {
                vec4 tex = (uHasTexture == 1) ? texture(uTexture, vTexCoord) : vec4(1.0);
                vec4 baseCol = tex * uColor;
                if (uAlphaMode == 1 && baseCol.a < uAlphaCutoff) {
                    discard;
                }
                vec3 srgbAlbedo = clamp(baseCol.rgb, 0.0, 1.0);
                // Decode sRGB baseColor into linear light space
                vec3 albedo = pow(srgbAlbedo, vec3(2.2));

                float metallic = clamp(uMetallic, 0.0, 1.0);
                float roughness = clamp(uRoughness, 0.04, 1.0);
                float ao = 1.0;
                float occStrength = (uOcclusionStrength > 0.0001) ? clamp(uOcclusionStrength, 0.0, 1.0) : 1.0;
                if (uHasMetallicRoughnessMap == 1) {
                    vec4 mrSample = texture(uMetallicRoughnessMap, vTexCoord);
                    roughness = clamp(roughness * mrSample.g, 0.04, 1.0);
                    metallic = clamp(metallic * mrSample.b, 0.0, 1.0);
                }
                if (uHasOcclusionMap == 1) {
                    float occSample = texture(uOcclusionMap, vTexCoord).r;
                    ao *= mix(1.0, occSample, occStrength);
                }

                float nLen = length(vNormal);
                vec3 N = (nLen > 0.0001) ? (vNormal / nLen) : vec3(0.0, 0.0, 1.0);
                if (uDoubleSided == 1 && !gl_FrontFacing) {
                    N = -N;
                }
                float vLen = length(vViewPos);
                vec3 V = (vLen > 0.0001) ? (-vViewPos / vLen) : vec3(0.0, 0.0, 1.0);

                if (uHasNormalMap == 1) {
                    vec3 tangentNormal = texture(uNormalMap, vTexCoord).xyz * 2.0 - 1.0;
                    float nScale = (uNormalScale > 0.0001) ? uNormalScale : 1.0;
                    tangentNormal.xy *= nScale;
                    N = perturbNormalCotangent(N, vViewPos, vTexCoord, normalize(tangentNormal));
                }

                float NdotV = max(dot(N, V), 0.001);
                vec3 F0 = mix(vec3(0.04), albedo, metallic);
                float wrapAmount = (1.0 - metallic) * 0.14;

                vec3 LoDiff = vec3(0.0);
                vec3 LoSpec = vec3(0.0);
                for (int i = 0; i < 2; i++) {
                    if (i >= uLightCount) break;
                    vec3 lDir = (i == 0) ? uLightDir[0] : uLightDir[1];
                    vec3 lCol = (i == 0) ? uLightColor[0] : uLightColor[1];
                    float lInt = (i == 0) ? uLightIntensity[0] : uLightIntensity[1];
                    float lLen = length(lDir);
                    vec3 L = (lLen > 0.0001) ? (lDir / lLen) : vec3(0.0, 0.6, 0.8);
                    vec3 H = normalize(V + L);
                    float rawDotNL = dot(N, L);
                    float NdotL = max(rawDotNL, 0.0);
                    float wrapNdotL = max((rawDotNL + wrapAmount) / (1.0 + wrapAmount), 0.0);
                    if (wrapNdotL > 0.0) {
                        float NdotH = max(dot(N, H), 0.0);
                        float HdotV = max(dot(H, V), 0.0);
                        vec3 F = fresnelSchlick(HdotV, F0);
                        vec3 spec = vec3(0.0);
                        if (NdotL > 0.0) {
                            float NDF = distributionGGX(NdotH, roughness);
                            float G = geometrySmith(NdotV, NdotL, roughness);
                            vec3 numerator = NDF * G * F;
                            float denominator = 4.0 * NdotV * NdotL + 0.0001;
                            spec = (numerator / denominator) * NdotL;
                        }

                        vec3 kS = F;
                        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
                        // Punctual lights authored in normalized unit scale (0.7..0.82) correspond to PI-scaled physical irradiance for diffuse
                        LoDiff += (kD * albedo * wrapNdotL) * (lCol * lInt);
                        LoSpec += spec * (lCol * (lInt * 2.2));
                    }
                }

                // Analytical multi-lobe studio hemisphere IBL with roughness-aware Fresnel and split-sum BRDF
                vec3 R = reflect(-V, N);
                float skyWeight = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
                vec3 ambLinear = pow(clamp(uAmbient, 0.0, 1.0), vec3(1.6));
                vec3 envDiffuse = mix(ambLinear * 0.65, ambLinear * 1.35 + vec3(0.05, 0.06, 0.08), skyWeight);
                float reflSky = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
                float studioKeyLobe = pow(max(dot(R, normalize(vec3(-0.25, 0.65, 0.72))), 0.0), mix(48.0, 3.0, roughness));
                float studioRimLobe = pow(max(dot(R, normalize(vec3(0.55, 0.35, -0.75))), 0.0), mix(24.0, 2.5, roughness)) * 0.35;
                vec3 envSpecRadiance = mix(vec3(0.12, 0.11, 0.10), vec3(0.90, 0.93, 0.98), reflSky)
                    + vec3((studioKeyLobe + studioRimLobe) * (1.0 - roughness * 0.72));
                vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
                vec2 envBrdf = envBRDFApprox(roughness, NdotV);
                vec3 specIbl = envSpecRadiance * (F_ibl * envBrdf.x + envBrdf.y);
                vec3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
                float velvetSheen = pow(1.0 - NdotV, 3.5) * clamp(uFresnelSheen, 0.0, 0.5) * (1.0 - metallic) * roughness * 0.18;
                vec3 ambientDiffIbl = (kD_ibl * albedo * envDiffuse + albedo * velvetSheen) * ao;
                vec3 ambientSpecIbl = (specIbl * 0.55) * ao;

                vec3 emissiveTerm = uEmissive;
                if (uHasEmissiveMap == 1) {
                    vec3 emSample = pow(clamp(texture(uEmissiveMap, vTexCoord).rgb, 0.0, 1.0), vec3(2.2));
                    vec3 emFactor = (dot(uEmissive, uEmissive) > 0.00001) ? uEmissive : vec3(1.0);
                    emissiveTerm = emSample * emFactor;
                }

                float baseA = (uAlphaMode == 1) ? 1.0 : clamp(baseCol.a, 0.0, 1.0);
                vec3 linearColor = ambientDiffIbl + ambientSpecIbl + LoDiff + LoSpec + emissiveTerm;
                // ACES filmic tone-mapping curve in linear space followed by sRGB gamma encoding
                vec3 mapped = (linearColor * (2.51 * linearColor + 0.03)) / (linearColor * (2.43 * linearColor + 0.59) + 0.14);
                vec3 srgbOut = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
                if (baseA < 0.999) {
                    vec3 specOnlyLin = ambientSpecIbl + LoSpec;
                    vec3 specMapped = (specOnlyLin * (2.51 * specOnlyLin + 0.03)) / (specOnlyLin * (2.43 * specOnlyLin + 0.59) + 0.14);
                    vec3 specSrgb = pow(clamp(specMapped, 0.0, 1.0), vec3(1.0 / 2.2));
                    float specPeak = max(specSrgb.r, max(specSrgb.g, specSrgb.b));
                    float outAlpha = clamp(baseA + specPeak * (1.0 - baseA) * 0.85, 0.0, 1.0);
                    vec3 premulRgb = clamp(srgbOut * baseA + specSrgb * (1.0 - baseA) * 0.85, 0.0, 1.0);
                    fragColor = vec4(premulRgb, outAlpha);
                } else {
                    fragColor = vec4(srgbOut, 1.0);
                }
            }
        """

        // Face Texture Shaders (faceTexture@1 — README Section 5.8, 11.4 & 21 Step 8)
        private const val VS_FACE_TEXTURE = """#version 300 es
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec2 aTexCoord;
            layout(location = 2) in float aRegionMask;
            uniform mat4 uMVP;
            out vec2 vTexCoord;
            out float vRegionMask;
            out vec2 vScreenUv;
            void main() {
                vTexCoord = aTexCoord;
                vRegionMask = aRegionMask;
                vec4 clipPos = uMVP * vec4(aPosition, 1.0);
                gl_Position = clipPos;
                vec2 ndc = clipPos.xy / max(abs(clipPos.w), 0.0001);
                vScreenUv = ndc * 0.5 + 0.5;
            }
        """

        private const val FS_FACE_TEXTURE = """#version 300 es
            precision mediump float;
            in vec2 vTexCoord;
            in float vRegionMask;
            in vec2 vScreenUv;
            uniform sampler2D uTexture;
            uniform sampler2D uMask;
            uniform sampler2D uCameraTexture;
            uniform int uHasMask;
            uniform int uHasCameraTexture;
            uniform int uBlendMode; // 0=alpha, 1=multiply, 2=softLight, 3=screen, 4=add
            uniform float uOpacity;
            uniform vec4 uTintColor;
            uniform int uProtectEyes;
            uniform int uProtectLips;
            uniform int uProtectMouthInterior;
            uniform int uTexturePremultiplied;
            uniform float uMouthOpen;
            uniform float uEyeBlink;
            out vec4 fragColor;

            float ellipseMask(vec2 uv, vec2 center, vec2 radii, float feather) {
                vec2 d = (uv - center) / radii;
                float dist = length(d);
                return smoothstep(1.0 - feather, 1.0 + feather, dist);
            }

            vec3 applyBlendMode(vec3 base, vec3 src, int mode) {
                if (mode == 1) {
                    // Multiply
                    return base * src;
                } else if (mode == 2) {
                    // SoftLight (Pegtop / W3C)
                    return (1.0 - 2.0 * src) * base * base + 2.0 * src * base;
                } else if (mode == 3) {
                    // Screen
                    return 1.0 - (1.0 - base) * (1.0 - src);
                } else if (mode == 4) {
                    // Additive
                    return clamp(base + src, 0.0, 1.0);
                }
                return src;
            }

            void main() {
                vec4 tex = texture(uTexture, vTexCoord);
                float maskVal = 1.0;
                if (uHasMask == 1) {
                    vec4 m = texture(uMask, vTexCoord);
                    maskVal = max(m.r, m.a);
                }

                float regionMask = clamp(vRegionMask, 0.0, 1.0);
                if (uProtectEyes == 1) {
                    float rEye = ellipseMask(vTexCoord, vec2(0.3496, 0.3780), vec2(0.076, 0.032), 0.15);
                    float lEye = ellipseMask(vTexCoord, vec2(0.6504, 0.3780), vec2(0.076, 0.032), 0.15);
                    float eyeCut = min(rEye, lEye);
                    if (uEyeBlink > 0.02 && eyeCut < 0.85) {
                        eyeCut *= clamp(1.0 - uEyeBlink, 0.0, 1.0);
                    }
                    regionMask = min(regionMask, eyeCut);
                }
                if (uProtectMouthInterior == 1) {
                    float mouthRadiiY = 0.025 + clamp(uMouthOpen, 0.0, 1.0) * 0.045;
                    float mouthIn = ellipseMask(vTexCoord, vec2(0.5000, 0.6939), vec2(0.098, mouthRadiiY), 0.15);
                    regionMask = min(regionMask, mouthIn);
                }
                if (uProtectLips == 1) {
                    float lipsOut = ellipseMask(vTexCoord, vec2(0.5000, 0.6937), vec2(0.130, 0.048), 0.15);
                    regionMask = min(regionMask, lipsOut);
                }

                float effectiveMask = maskVal * regionMask * clamp(uOpacity, 0.0, 1.0) * uTintColor.a;
                float outAlpha = clamp(tex.a * effectiveMask, 0.0, 1.0);
                float safeTexA = max(tex.a, 0.0001);
                vec3 unpremulRgb = (uTexturePremultiplied == 1) ? clamp(tex.rgb / safeTexA, 0.0, 1.0) : clamp(tex.rgb, 0.0, 1.0);
                vec3 srcRgb = clamp(unpremulRgb * uTintColor.rgb, 0.0, 1.0);

                if (uBlendMode >= 1 && uBlendMode <= 3 && uHasCameraTexture == 1) {
                    vec3 camRgb = texture(uCameraTexture, clamp(vScreenUv, 0.0, 1.0)).rgb;
                    vec3 blendedRgb = applyBlendMode(camRgb, srcRgb, uBlendMode);
                    vec3 mixedRgb = mix(camRgb, blendedRgb, outAlpha);
                    fragColor = vec4(mixedRgb * outAlpha, outAlpha);
                } else {
                    vec3 baseRgb = (uTexturePremultiplied == 1) ? tex.rgb : (tex.rgb * tex.a);
                    vec3 outRgb = baseRgb * uTintColor.rgb * effectiveMask;
                    fragColor = vec4(outRgb, outAlpha);
                }
            }
        """

        // Occluder (Depth-only) Shaders
        private const val VS_OCCLUDER = """#version 300 es
            layout(location = 0) in vec3 aPosition;
            uniform mat4 uMVP;
            void main() {
                gl_Position = uMVP * vec4(aPosition, 1.0);
            }
        """

        private const val FS_OCCLUDER = """#version 300 es
            precision lowp float;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(0.0);
            }
        """

        fun isTemplateSupported(template: String): Boolean {
            return template == "unlit@1" ||
                template == "lambert@1" ||
                template == "pbr@1" ||
                template == "faceTexture@1" ||
                template == "recolor@1" ||
                template == "pixel.recolor@1"
        }

        /**
         * Evaluates the Cook-Torrance GGX PBR shader formula (`FS_PBR`) on CPU for deterministic
         * unit testing and photo fallback parity.
         */
        @JvmOverloads
        fun evaluatePbrColor(
            albedoRgb: FloatArray,
            normal: FloatArray,
            viewDir: FloatArray = floatArrayOf(0f, 0f, 1f),
            metallic: Float = 0.0f,
            roughness: Float = 0.45f,
            emissiveRgb: FloatArray = floatArrayOf(0f, 0f, 0f),
            lighting: com.example.engine.scene.CompiledLighting = com.example.engine.scene.CompiledLighting(),
            outRgb: FloatArray = FloatArray(3),
            occlusion: Float = 1.0f,
            fresnelSheen: Float = 0.15f,
            emissiveMapRgb: FloatArray? = null
        ): FloatArray {
            val pi = Math.PI.toFloat()
            val srgbR = albedoRgb[0].coerceIn(0f, 1f)
            val srgbG = albedoRgb[1].coerceIn(0f, 1f)
            val srgbB = albedoRgb[2].coerceIn(0f, 1f)
            // Decode sRGB baseColor into linear light space
            val albR = Math.pow(srgbR.toDouble(), 2.2).toFloat()
            val albG = Math.pow(srgbG.toDouble(), 2.2).toFloat()
            val albB = Math.pow(srgbB.toDouble(), 2.2).toFloat()
            val met = metallic.coerceIn(0f, 1f)
            val rough = roughness.coerceIn(0.04f, 1f)
            val ao = occlusion.coerceIn(0f, 1f)

            val nLen = kotlin.math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
            val nx = if (nLen > 1e-4f) normal[0] / nLen else 0f
            val ny = if (nLen > 1e-4f) normal[1] / nLen else 0f
            val nz = if (nLen > 1e-4f) normal[2] / nLen else 1f

            val vLen = kotlin.math.sqrt(viewDir[0] * viewDir[0] + viewDir[1] * viewDir[1] + viewDir[2] * viewDir[2])
            val vx = if (vLen > 1e-4f) viewDir[0] / vLen else 0f
            val vy = if (vLen > 1e-4f) viewDir[1] / vLen else 0f
            val vz = if (vLen > 1e-4f) viewDir[2] / vLen else 1f

            val ndotv = maxOf(nx * vx + ny * vy + nz * vz, 0.001f)
            val f0R = 0.04f + (albR - 0.04f) * met
            val f0G = 0.04f + (albG - 0.04f) * met
            val f0B = 0.04f + (albB - 0.04f) * met
            val wrapAmount = (1f - met) * 0.14f

            var loR = 0f
            var loG = 0f
            var loB = 0f
            val count = lighting.lightCount.coerceIn(0, com.example.engine.scene.CompiledLighting.MAX_LIGHTS)

            for (i in 0 until count) {
                val base3 = i * 3
                val ldx = lighting.lightDirs[base3]
                val ldy = lighting.lightDirs[base3 + 1]
                val ldz = lighting.lightDirs[base3 + 2]
                val lLen = kotlin.math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz)
                val lx = if (lLen > 1e-4f) ldx / lLen else 0f
                val ly = if (lLen > 1e-4f) ldy / lLen else 0.6f
                val lz = if (lLen > 1e-4f) ldz / lLen else 0.8f

                val rawDotNL = nx * lx + ny * ly + nz * lz
                val ndotl = maxOf(rawDotNL, 0f)
                val wrapNdotL = maxOf((rawDotNL + wrapAmount) / (1f + wrapAmount), 0f)
                if (wrapNdotL > 0f) {
                    var hx = vx + lx
                    var hy = vy + ly
                    var hz = vz + lz
                    val hLen = kotlin.math.sqrt(hx * hx + hy * hy + hz * hz)
                    if (hLen > 1e-4f) {
                        hx /= hLen; hy /= hLen; hz /= hLen
                    }
                    val ndoth = maxOf(nx * hx + ny * hy + nz * hz, 0f)
                    val hdotv = maxOf(hx * vx + hy * vy + hz * vz, 0f)

                    val fPow = Math.pow((1f - hdotv).coerceIn(0f, 1f).toDouble(), 5.0).toFloat()
                    val fR = f0R + (1f - f0R) * fPow
                    val fG = f0G + (1f - f0G) * fPow
                    val fB = f0B + (1f - f0B) * fPow

                    var specR = 0f
                    var specG = 0f
                    var specB = 0f
                    if (ndotl > 0f) {
                        val a = rough * rough
                        val a2 = a * a
                        val denomNdf = ndoth * ndoth * (a2 - 1f) + 1f
                        val ndf = a2 / maxOf(pi * denomNdf * denomNdf, 0.0001f)

                        val r = rough + 1f
                        val k = (r * r) / 8f
                        val gV = ndotv / maxOf(ndotv * (1f - k) + k, 0.0001f)
                        val gL = ndotl / maxOf(ndotl * (1f - k) + k, 0.0001f)
                        val g = gV * gL

                        val denomSpec = 4f * ndotv * ndotl + 0.0001f
                        specR = ((ndf * g * fR) / denomSpec) * ndotl
                        specG = ((ndf * g * fG) / denomSpec) * ndotl
                        specB = ((ndf * g * fB) / denomSpec) * ndotl
                    }

                    val kdR = (1f - fR) * (1f - met)
                    val kdG = (1f - fG) * (1f - met)
                    val kdB = (1f - fB) * (1f - met)

                    val rawInt = lighting.lightIntensities[i]
                    val diffRadR = lighting.lightColors[base3] * rawInt
                    val diffRadG = lighting.lightColors[base3 + 1] * rawInt
                    val diffRadB = lighting.lightColors[base3 + 2] * rawInt
                    val specRadR = diffRadR * 2.2f
                    val specRadG = diffRadG * 2.2f
                    val specRadB = diffRadB * 2.2f

                    loR += (kdR * albR * wrapNdotL) * diffRadR + specR * specRadR
                    loG += (kdG * albG * wrapNdotL) * diffRadG + specG * specRadG
                    loB += (kdB * albB * wrapNdotL) * diffRadB + specB * specRadB
                }
            }

            // Analytical multi-lobe studio hemisphere IBL with roughness-aware Fresnel and split-sum BRDF
            val dotNV = nx * vx + ny * vy + nz * vz
            val rx = 2f * dotNV * nx - vx
            val ry = 2f * dotNV * ny - vy
            val rz = 2f * dotNV * nz - vz
            val skyW = (ny * 0.5f + 0.5f).coerceIn(0f, 1f)
            val amb0 = Math.pow(lighting.ambient[0].coerceIn(0f, 1f).toDouble(), 1.6).toFloat()
            val amb1 = Math.pow(lighting.ambient[1].coerceIn(0f, 1f).toDouble(), 1.6).toFloat()
            val amb2 = Math.pow(lighting.ambient[2].coerceIn(0f, 1f).toDouble(), 1.6).toFloat()
            val envDiffR = (amb0 * 0.65f) * (1f - skyW) + (amb0 * 1.35f + 0.05f) * skyW
            val envDiffG = (amb1 * 0.65f) * (1f - skyW) + (amb1 * 1.35f + 0.06f) * skyW
            val envDiffB = (amb2 * 0.65f) * (1f - skyW) + (amb2 * 1.35f + 0.08f) * skyW

            val reflSky = (ry * 0.5f + 0.5f).coerceIn(0f, 1f)
            val keyLen = kotlin.math.sqrt(0.25f * 0.25f + 0.65f * 0.65f + 0.72f * 0.72f)
            val keyDot = maxOf((rx * -0.25f + ry * 0.65f + rz * 0.72f) / keyLen, 0f)
            val specExp = 48f * (1f - rough) + 3f * rough
            val studioKeyLobe = Math.pow(keyDot.toDouble(), specExp.toDouble()).toFloat()

            val rimLen = kotlin.math.sqrt(0.55f * 0.55f + 0.35f * 0.35f + 0.75f * 0.75f)
            val rimDot = maxOf((rx * 0.55f + ry * 0.35f + rz * -0.75f) / rimLen, 0f)
            val rimExp = 24f * (1f - rough) + 2.5f * rough
            val studioRimLobe = Math.pow(rimDot.toDouble(), rimExp.toDouble()).toFloat() * 0.35f

            val lobeGain = (studioKeyLobe + studioRimLobe) * (1f - rough * 0.72f)
            val envSpecR = 0.12f + (0.90f - 0.12f) * reflSky + lobeGain
            val envSpecG = 0.11f + (0.93f - 0.11f) * reflSky + lobeGain
            val envSpecB = 0.10f + (0.98f - 0.10f) * reflSky + lobeGain

            // fresnelSchlickRoughness
            val iblFPow = Math.pow((1f - ndotv).coerceIn(0f, 1f).toDouble(), 5.0).toFloat()
            val maxRoughCompl = 1f - rough
            val fIblR = f0R + (maxOf(maxRoughCompl, f0R) - f0R) * iblFPow
            val fIblG = f0G + (maxOf(maxRoughCompl, f0G) - f0G) * iblFPow
            val fIblB = f0B + (maxOf(maxRoughCompl, f0B) - f0B) * iblFPow

            // envBRDFApprox
            val r0 = rough * -1.0f + 1.0f
            val r1 = rough * -0.0275f + 0.0425f
            val r2 = rough * -0.572f + 1.04f
            val r3 = rough * 0.022f - 0.04f
            val expTerm = Math.pow(2.0, (-9.28f * ndotv).toDouble()).toFloat()
            val a004 = minOf(r0 * r0, expTerm) * r0 + r1
            val brdfScale = (-1.04f * a004 + r2).coerceIn(0f, 1f)
            val brdfBias = (1.04f * a004 + r3).coerceIn(0f, 1f)

            val specIblR = envSpecR * (fIblR * brdfScale + brdfBias)
            val specIblG = envSpecG * (fIblG * brdfScale + brdfBias)
            val specIblB = envSpecB * (fIblB * brdfScale + brdfBias)

            val kdIblR = (1f - fIblR) * (1f - met)
            val kdIblG = (1f - fIblG) * (1f - met)
            val kdIblB = (1f - fIblB) * (1f - met)
            val velvetSheen = Math.pow((1f - ndotv).toDouble(), 3.5).toFloat() *
                fresnelSheen.coerceIn(0f, 0.5f) * (1f - met) * rough * 0.18f

            val ambIblR = (kdIblR * albR * envDiffR + specIblR * 0.55f + albR * velvetSheen) * ao
            val ambIblG = (kdIblG * albG * envDiffG + specIblG * 0.55f + albG * velvetSheen) * ao
            val ambIblB = (kdIblB * albB * envDiffB + specIblB * 0.55f + albB * velvetSheen) * ao

            val em0 = emissiveRgb.getOrElse(0) { 0f }
            val em1 = emissiveRgb.getOrElse(1) { 0f }
            val em2 = emissiveRgb.getOrElse(2) { 0f }
            val hasEmFactor = (em0 * em0 + em1 * em1 + em2 * em2) > 1e-5f
            val finalEmR: Float
            val finalEmG: Float
            val finalEmB: Float
            if (emissiveMapRgb != null && emissiveMapRgb.size >= 3) {
                val emMapR = Math.pow(emissiveMapRgb[0].coerceIn(0f, 1f).toDouble(), 2.2).toFloat()
                val emMapG = Math.pow(emissiveMapRgb[1].coerceIn(0f, 1f).toDouble(), 2.2).toFloat()
                val emMapB = Math.pow(emissiveMapRgb[2].coerceIn(0f, 1f).toDouble(), 2.2).toFloat()
                finalEmR = emMapR * (if (hasEmFactor) em0 else 1f)
                finalEmG = emMapG * (if (hasEmFactor) em1 else 1f)
                finalEmB = emMapB * (if (hasEmFactor) em2 else 1f)
            } else {
                finalEmR = em0
                finalEmG = em1
                finalEmB = em2
            }

            val linR = ambIblR + loR + finalEmR
            val linG = ambIblG + loG + finalEmG
            val linB = ambIblB + loB + finalEmB

            fun acesAndSrgb(x: Float): Float {
                val c = maxOf(x, 0f)
                val mapped = ((c * (2.51f * c + 0.03f)) / (c * (2.43f * c + 0.59f) + 0.14f)).coerceIn(0f, 1f)
                return Math.pow(mapped.toDouble(), 1.0 / 2.2).toFloat().coerceIn(0f, 1f)
            }

            outRgb[0] = acesAndSrgb(linR)
            outRgb[1] = acesAndSrgb(linG)
            outRgb[2] = acesAndSrgb(linB)
            return outRgb
        }

        /**
         * Evaluates the multi-light Lambert shader formula (`FS_LAMBERT`) for 0, 1, or 2 directional lights:
         * `C = tex_linear * clamp(ambient + sum_k(intensity_k * color_k * max(dot(N, L_k), 0)), 0, 1), L_k = normalize(dirToLight)`
         * When `includeSpecular = true` or `fresnelSheen > 0f`, also includes the Phase 1 Blinn-Phong specular
         * and Fresnel rim sheen terms matching `FS_LAMBERT`.
         */
        @JvmOverloads
        fun evaluateLambertColor(
            texRgb: FloatArray,
            normal: FloatArray,
            lighting: com.example.engine.scene.CompiledLighting,
            outRgb: FloatArray = FloatArray(3),
            viewDir: FloatArray = floatArrayOf(0f, 0f, 1f),
            fresnelSheen: Float = 0.0f,
            includeSpecular: Boolean = false
        ): FloatArray {
            val nLen = kotlin.math.sqrt(
                normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
            )
            val nx: Float
            val ny: Float
            val nz: Float
            if (nLen > 0.0001f) {
                nx = normal[0] / nLen
                ny = normal[1] / nLen
                nz = normal[2] / nLen
            } else {
                nx = 0f
                ny = 0f
                nz = 1f
            }

            val vLen = kotlin.math.sqrt(viewDir[0] * viewDir[0] + viewDir[1] * viewDir[1] + viewDir[2] * viewDir[2])
            val vx = if (vLen > 0.0001f) viewDir[0] / vLen else 0f
            val vy = if (vLen > 0.0001f) viewDir[1] / vLen else 0f
            val vz = if (vLen > 0.0001f) viewDir[2] / vLen else 1f
            val ndotv = maxOf(nx * vx + ny * vy + nz * vz, 0f)

            var diffR = 0f
            var diffG = 0f
            var diffB = 0f
            var specR = 0f
            var specG = 0f
            var specB = 0f
            val count = lighting.lightCount.coerceIn(0, com.example.engine.scene.CompiledLighting.MAX_LIGHTS)

            if (count >= 1) {
                val dx0 = lighting.lightDirs[0]
                val dy0 = lighting.lightDirs[1]
                val dz0 = lighting.lightDirs[2]
                val lLen0 = kotlin.math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
                val lx0 = if (lLen0 > 0.0001f) dx0 / lLen0 else 0f
                val ly0 = if (lLen0 > 0.0001f) dy0 / lLen0 else 0.6f
                val lz0 = if (lLen0 > 0.0001f) dz0 / lLen0 else 0.8f
                val ndotl0 = maxOf(nx * lx0 + ny * ly0 + nz * lz0, 0f)
                val scale0 = ndotl0 * lighting.lightIntensities[0]
                diffR = lighting.lightColors[0] * scale0
                diffG = lighting.lightColors[1] * scale0
                diffB = lighting.lightColors[2] * scale0
                if (includeSpecular && ndotl0 > 0f) {
                    var hx0 = lx0 + vx; var hy0 = ly0 + vy; var hz0 = lz0 + vz
                    val hLen0 = kotlin.math.sqrt(hx0 * hx0 + hy0 * hy0 + hz0 * hz0)
                    if (hLen0 > 0.0001f) { hx0 /= hLen0; hy0 /= hLen0; hz0 /= hLen0 }
                    val ndoth0 = maxOf(nx * hx0 + ny * hy0 + nz * hz0, 0f)
                    val s0 = Math.pow(ndoth0.toDouble(), 28.0).toFloat() * 0.14f * lighting.lightIntensities[0]
                    specR += lighting.lightColors[0] * s0
                    specG += lighting.lightColors[1] * s0
                    specB += lighting.lightColors[2] * s0
                }
            }
            if (count >= 2) {
                val dx1 = lighting.lightDirs[3]
                val dy1 = lighting.lightDirs[4]
                val dz1 = lighting.lightDirs[5]
                val lLen1 = kotlin.math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
                val lx1 = if (lLen1 > 0.0001f) dx1 / lLen1 else 0f
                val ly1 = if (lLen1 > 0.0001f) dy1 / lLen1 else 0.6f
                val lz1 = if (lLen1 > 0.0001f) dz1 / lLen1 else 0.8f
                val ndotl1 = maxOf(nx * lx1 + ny * ly1 + nz * lz1, 0f)
                val scale1 = ndotl1 * lighting.lightIntensities[1]
                diffR += lighting.lightColors[3] * scale1
                diffG += lighting.lightColors[4] * scale1
                diffB += lighting.lightColors[5] * scale1
                if (includeSpecular && ndotl1 > 0f) {
                    var hx1 = lx1 + vx; var hy1 = ly1 + vy; var hz1 = lz1 + vz
                    val hLen1 = kotlin.math.sqrt(hx1 * hx1 + hy1 * hy1 + hz1 * hz1)
                    if (hLen1 > 0.0001f) { hx1 /= hLen1; hy1 /= hLen1; hz1 /= hLen1 }
                    val ndoth1 = maxOf(nx * hx1 + ny * hy1 + nz * hz1, 0f)
                    val s1 = Math.pow(ndoth1.toDouble(), 28.0).toFloat() * 0.10f * lighting.lightIntensities[1]
                    specR += lighting.lightColors[3] * s1
                    specG += lighting.lightColors[4] * s1
                    specB += lighting.lightColors[5] * s1
                }
            }

            val litR = (lighting.ambient[0] + diffR).coerceIn(0f, 1f)
            val litG = (lighting.ambient[1] + diffG).coerceIn(0f, 1f)
            val litB = (lighting.ambient[2] + diffB).coerceIn(0f, 1f)

            if (!includeSpecular && fresnelSheen <= 0f) {
                outRgb[0] = texRgb[0] * litR
                outRgb[1] = texRgb[1] * litG
                outRgb[2] = texRgb[2] * litB
            } else {
                val rim = Math.pow((1f - ndotv).toDouble(), 3.0).toFloat() * fresnelSheen.coerceIn(0f, 0.5f) * 0.25f
                outRgb[0] = (texRgb[0] * litR + specR + rim).coerceIn(0f, 1f)
                outRgb[1] = (texRgb[1] * litG + specG + rim).coerceIn(0f, 1f)
                outRgb[2] = (texRgb[2] * litB + specB + rim).coerceIn(0f, 1f)
            }
            return outRgb
        }

        /**
         * Legacy 1-light Lambert formula prior to Chunk E1, used by golden unit tests to verify
         * bit-for-bit identical results for 1-light packages.
         */
        fun evaluateLegacySingleLightLambertColor(
            texRgb: FloatArray,
            normal: FloatArray,
            ambient: FloatArray,
            dirToLight: FloatArray,
            lightColor: FloatArray,
            lightIntensity: Float,
            outRgb: FloatArray = FloatArray(3)
        ): FloatArray {
            val nLen = kotlin.math.sqrt(
                normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
            )
            val nx = if (nLen > 0.0001f) normal[0] / nLen else 0f
            val ny = if (nLen > 0.0001f) normal[1] / nLen else 0f
            val nz = if (nLen > 0.0001f) normal[2] / nLen else 1f

            val lLen = kotlin.math.sqrt(
                dirToLight[0] * dirToLight[0] + dirToLight[1] * dirToLight[1] + dirToLight[2] * dirToLight[2]
            )
            val lx = if (lLen > 0.0001f) dirToLight[0] / lLen else 0f
            val ly = if (lLen > 0.0001f) dirToLight[1] / lLen else 0.6f
            val lz = if (lLen > 0.0001f) dirToLight[2] / lLen else 0.8f

            val ndotl = maxOf(nx * lx + ny * ly + nz * lz, 0f)
            val scale = ndotl * lightIntensity
            val diffR = lightColor[0] * scale
            val diffG = lightColor[1] * scale
            val diffB = lightColor[2] * scale

            val litR = (ambient[0] + diffR).coerceIn(0f, 1f)
            val litG = (ambient[1] + diffG).coerceIn(0f, 1f)
            val litB = (ambient[2] + diffB).coerceIn(0f, 1f)

            outRgb[0] = texRgb[0] * litR
            outRgb[1] = texRgb[1] * litG
            outRgb[2] = texRgb[2] * litB
            return outRgb
        }
    }

    var programUnlit = 0; private set
    var programLambert = 0; private set
    var programPbr = 0; private set
    var programFaceTexture = 0; private set
    var programOccluder = 0; private set

    fun init() {
        programUnlit = 0
        programLambert = 0
        programPbr = 0
        programFaceTexture = 0
        programOccluder = 0
        try {
            programUnlit = createProgram(VS_UNLIT, FS_UNLIT)
            programLambert = createProgram(VS_LAMBERT, FS_LAMBERT)
            programPbr = createProgram(VS_PBR, FS_PBR)
            programFaceTexture = createProgram(VS_FACE_TEXTURE, FS_FACE_TEXTURE)
            programOccluder = createProgram(VS_OCCLUDER, FS_OCCLUDER)
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
    }

    fun release() {
        try {
            val progs = intArrayOf(programUnlit, programLambert, programPbr, programFaceTexture, programOccluder)
            for (p in progs) {
                if (p != 0) GLES30.glDeleteProgram(p)
            }
        } catch (_: Throwable) {
        }
        programUnlit = 0
        programLambert = 0
        programPbr = 0
        programFaceTexture = 0
        programOccluder = 0
    }

    private fun createProgram(vsCode: String, fsCode: String): Int {
        val vs = compileShader(GLES30.GL_VERTEX_SHADER, vsCode)
        val fs = compileShader(GLES30.GL_FRAGMENT_SHADER, fsCode)
        if (vs == 0 || fs == 0) return 0

        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vs)
        GLES30.glAttachShader(program, fs)
        GLES30.glLinkProgram(program)

        val linkStatus = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] == 0) {
            val log = GLES30.glGetProgramInfoLog(program)
            Log.e(TAG, "Shader link error: $log")
            GLES30.glDeleteProgram(program)
            return 0
        }

        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        return program
    }

    private fun compileShader(type: Int, code: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, code)
        GLES30.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(shader)
            Log.e(TAG, "Shader compile error ($type): $log")
            GLES30.glDeleteShader(shader)
            return 0
        }
        return shader
    }
}
