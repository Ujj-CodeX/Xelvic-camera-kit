# Xelvic Filter Package — Periwinkle Forehead

**Package version:** schemaVersion 1 (same strict schema contract as the supplied `sunflower_forehead` package)  
**Engine contract:** `xfilter-face-3d/1`  
**Asset:** `assets/periwinkle_forehead.glb`

## Visual intent

A single compact, cute five-petaled Periwinkle bloom in soft lavender/periwinkle with pink gradients and a small magenta centre. It is attached as a small accent on the **subject-left upper forehead/temple**, above the eyebrow and near the hairline. It is intentionally much smaller than the sunflower reference (approximately 0.22 face-width across versus the reference bloom's approximately 0.38 face-width). It contains **only five petals, subtle petal-vein details, and a small flower centre**: no green leaves, foliage, stem, crown or flower cluster.

The catalogue proxy places subject-left on viewer-right. Confirm the canonical side mapping on the target camera before shipping; the package does not mirror individual model nodes.

## Package contents

- `manifest.json` — complete layer, anchor, material, lighting, occlusion, behavior, LOD, compatibility and verification contract.
- `assets/periwinkle_forehead.glb` — glTF 2.0 3D mesh model, embedded PNG textures, three static LOD mesh sets and three animation clips. The GLB is Blender-importable.
- `preview.jpg` — catalogue proxy-head placement preview (illustrative, not a device calibration).
- `source_textures/` — editable PNG texture masters and the generated flower-design reference.
- `schema/xfilter.schema.json` — strict JSON Schema for this package contract.
- `tools/validate_package.py` — schema, GLB, embedded image, animation, LOD, hash and rest-clearance validator.

## 3D model

The GLB scene contains one root (`periwinkle_root`) and one shared pivot (`periwinkle_pivot`). Three LOD tiers each contain:
- `periwinkle_petals_lod_*` — five rounded, gently curved petal surfaces.
- `periwinkle_veins_lod_*` — subtle raised petal centre-line details.
- `periwinkle_center_lod_*` — small magenta flower centre.

All meshes are children of the shared pivot. Tier A/B/C each show exactly one LOD set and hide the other two. Mesh geometry is face-width normalized (`1 FW = cheek distance between canonical landmarks 234 and 454`). The bloom is approximately 0.22 FW across.

The three embedded sRGB PNG textures are `petals_periwinkle`, `center_magenta`, and `petal_veins`. Materials follow the existing package contract: base-colour texture only, metallic 0, roughness 1, opaque and double-sided. The engine supplies camera-space Lambert lighting from the manifest.

## Placement

The flower centre is the GLB root origin. The layer uses the `head_center` anchor and offset `[-0.40, 0.42, 0.58] FW`. The subject-left mapping is assumed to be negative X and is marked VERIFY. Do not compensate by mirroring individual model nodes. Mirror the final projection once, as defined by the engine contract.

## Behaviours

- `idle_sway` — subtle 3.2 s closed rotation loop.
- `bloom_in` — 1.0 s gentle acquisition bloom, ends at unit scale.
- `burst` — 0.9 s small pulse, driven by the `jawOpen` threshold/cooldown logic.
- `soft_petal_smile` — subtle 1.0 to 1.06 petal scale response driven by mean `mouthSmileLeft` / `mouthSmileRight`.
- `subtle_head_inertia` — bounded rotational spring on the shared pivot.
- Face-loss hold/fade/reset and re-acquisition semantics are inherited from the package contract.

## LODs

| Tier | Visible mesh nodes | Draw calls | Triangles | Disabled behaviours |
|---|---:|---:|---:|---|
| A | 3 | 3 | 1,064 | none |
| B | 3 | 3 | 556 | none |
| C | 3 | 3 | 292 | `soft_petal_smile`, `subtle_head_inertia`, `tiny_bloom_pop_on_open` |

The reduced tiers retain the five-petal silhouette and centre; they reduce tessellation and detail geometry rather than removing petals.

## Validate

Requires Python 3 and `jsonschema` for Draft 2020-12 validation:

```bash
python -m pip install jsonschema
python tools/validate_package.py .
```

The validator checks manifest schema, asset byte count and SHA-256, GLB header/chunks, embedded PNG textures, animation names/targets/endpoints, identity rest transforms, uint16 triangle geometry, tier draw-call/triangle budgets, and rest clearance against the specified head-proxy ellipsoid.

## Device verification (not claimed as complete)

Verify on the actual Xelvic SDK/device: face-pose axes, negative-X subject-left mapping under the final front-camera mirror policy, camera FOV/projection, canonical cheek ids, blendshape names, and final size/placement above the eyebrow near the hairline. The preview is illustrative; it is not a guarantee of live camera placement. The package validator cannot replace this physical-device check.

## Editing rules

Keep animated nodes at rest scale 1 and identity rotation; scale clips must end at 1; the `idle_sway` loop must close; new clips must be added to `stats.clips`; recompute asset hashes and byte counts after any asset edit; rerun the validator before publishing.

## Editing in Blender

The GLB is glTF 2.0 and can be imported into Blender with **File → Import → glTF 2.0**. Edit the petals/centre as required, preserve the root/pivot and LOD node names, and export back to glTF Binary (`.glb`). This package contains the exported GLB rather than a native `.blend` project file.
