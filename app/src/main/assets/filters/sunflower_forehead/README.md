# Xelvic Filter Package — sunflower_forehead

**Package version:** schemaVersion 1  
**Engine contract:** `xfilter-face-3d/1`  
**Asset:** `assets/sunflower_forehead.glb`

## Visual intent

A **single golden sunflower bloom** worn as a small face-attached accent on the subject-left upper forehead/temple, just above the eyebrow and slightly overlapping the hairline. It is not a crown and does not contain a multi-flower cluster. The catalogue preview shows subject-left on the viewer-right side of the proxy head; confirm the canonical side mapping on the target camera before shipping.

## Package contents

- `manifest.json` — complete layer, anchor, material, lighting, occlusion, behavior, LOD, compatibility and verification contract.
- `assets/sunflower_forehead.glb` — glTF 2.0 model, embedded PNG textures, three static LOD mesh sets and three animation clips.
- `preview.jpg` — catalogue proxy-head placement preview.
- `source_textures/` — editable texture masters; runtime copies are embedded inside the GLB.
- `schema/xfilter.schema.json` — strict JSON Schema for this package contract.
- `tools/validate_package.py` — schema, GLB, embedded image, animation, LOD, hash and rest-clearance validator.

## Render model

The GLB has one root (`sunflower_root`) and a shared pivot (`sunflower_pivot`). All A/B/C LOD meshes are identity-transform children of the pivot. The selected tier shows exactly one LOD set and hides the other sets. Mesh geometry is face-width normalized (`1 FW = cheek distance between canonical landmarks 234 and 454`).

Three embedded sRGB PNG textures are used (editable masters are also supplied under `source_textures/`): gold petal fibers, brown seed disk, and green foliage. Materials are opaque, double-sided, base-color-texture only, metallic 0 and roughness 1. The engine applies the camera-space Lambert lighting formula in the manifest.

## Placement

The flower's seed-disk centre is the GLB root origin. The layer uses `head_center` with offset `[-0.40, 0.42, 0.58] FW`. `head_center` is computed from the cheek midpoint plus `[0, 0.20, -0.12] FW` in the face-local frame. The layer scale is 1.0; the geometry itself is sized to a compact ~0.38 FW bloom.

The subject-left mapping is assumed to be negative X and is marked VERIFY. Do not compensate by mirroring individual model nodes. Mirror the final projection once, as defined by the contract.

## Behaviors

- `idle_sway` — 3.2 s closed rotation loop.
- `bloom_in` — 1.0 s acquisition bloom, ends at unit scale.
- `burst` — 0.9 s non-looping scale pulse, driven by `jawOpen` threshold/cooldown logic.
- `smile_swell` — smoothed 1.0 to 1.12 petal scale driven by the mean of `mouthSmileLeft` / `mouthSmileRight`.
- `head_inertia` — bounded critically underdamped rotational spring per the manifest parameters.
- Face-loss hold/fade/reset and re-acquisition preservation semantics are specified in `manifest.json`.

## LODs

| Tier | Visible mesh nodes | Draw calls | Triangles | Disabled behaviors |
|---|---:|---:|---:|---|
| A | 4 | 4 | 2204 | none |
| B | 4 | 4 | 840 | none |
| C | 3 | 3 | 400 | `smile_swell`, `head_inertia`, `burst_on_open` |

C keeps a simplified petal silhouette, centre disk and foliage, plus `idle_sway` and `bloom_in`.

## Validate

Requires Python 3 and `jsonschema` for Draft 2020-12 validation:

```bash
python -m pip install jsonschema
python tools/validate_package.py .
```

The validator checks manifest schema, asset byte count and SHA-256, GLB header/chunks, embedded textures, animation names/targets/endpoints, identity rest transforms, uint16 triangle geometry, tier draw-call/triangle budgets, and rest clearance against the specified head-proxy ellipsoid.

## Device verification (not claimed as complete)

The following are VERIFY assumptions and must be tested on the actual Xelvic SDK/device: face-pose axes, `-X` subject-left mapping under the final front-camera mirror policy, camera FOV/projection, canonical cheek ids, blendshape names, and actual placement above the eyebrow/at the hairline. The catalogue preview is an illustration, not a camera calibration or a guarantee of on-face fit. The package validator cannot replace this physical-device check.

## Editing rules

Keep animated nodes at rest scale 1 and identity rotation; scale clips must end at 1; the `idle_sway` loop must close; new clips must be added to `stats.clips`; recompute SHA-256 and byte counts after any asset edit; rerun the validator before publishing.
