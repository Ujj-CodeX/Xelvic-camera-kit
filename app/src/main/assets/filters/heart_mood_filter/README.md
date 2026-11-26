# Heart Mood Filter — R&D Authoring Package

A pastel pink floating-heart crown matching the supplied reference composition. This package is authored for the planned Xelvic mesh/3D prop pipeline and aligned with the attached revised implementation plan.

## Important runtime compatibility

This is an **R&D authoring ZIP**, not a currently importable `.xfilter` for the existing app. The current repository `FilterPackageManager` allows `.json`, `.png`, and `.webp` archive entries only; GLB is rejected, and the active camera pipeline does not yet render 3D models. The package uses schemaVersion 3 / minimumRuntimeVersion 3 as a proposed model-prop contract. Do not rename this ZIP to `.xfilter` or import it into the current runtime until the package validator, model loader, renderer router, and GLB support are implemented.

## Contents

- `manifest.json` — effect definition, FACE_WIDTH coordinates, head-centre anchor, calibration tunables, face/head occlusion, bounded behaviour, lifecycle, tiers and metrics.
- `models/heart_mood_crown_lod_a.glb` — high-detail model.
- `models/heart_mood_crown_lod_b.glb` — medium detail, all nine hearts retained.
- `models/heart_mood_crown_lod_c.glb` — low detail, all nine hearts retained.
- `assets/heart_mood_atlas.png` — source preview of the embedded sRGB atlas; each GLB embeds the texture.
- `thumbnail.png` — package thumbnail.
- `validation/report.json` — hashes, GLB attributes and metric checks.
- `source/build_heart_mood_filter.py` — reproducible geometry/texture generator.

## Design and alignment

- 9 small hearts arranged in a gentle upper-hairline arc. The full crown spans about 0.93 face-width (FW); individual heart width varies around 0.073–0.101 FW to match the reference's delicate floating hearts. Each baked heart centre, width and rotation is also documented as a calibration field so the source mesh can be regenerated consistently.
- Model axes: +X subject-right, +Y up, +Z camera. No mirror/flip/side compensation is baked into the assets.
- Each LOD uses one combined mesh and one material (one draw call). LOD B/C reduce contour tessellation only; all nine hearts remain present.
- Model pivot is the centre of the combined model bounds; rest scale is 1 and rest rotation is identity. Layer placement is calibrated from `head_center` using FW units.
- Face-mesh occluder plus a head-proxy ellipsoid handle face/head depth. Ear/temple proxy is deliberately disabled because this effect remains within the upper forehead/hairline region.
- The model renders in Stage 5 `OPAQUE_MODELS`, after Stage 4 `CLEAR_DEPTH_AND_OCCLUDERS`, and before transparent props/post/overlay passes. No camera-pixel paint or eye-colour pass is required.
- Gentle float is bounded (2.2 degrees; 0.006 FW), rest pose returns to identity, and face loss uses hold → fade → reset.
- No blendshapes are required. Compatibility is one face, index 0.

## Calibration

Every placement, occlusion and behaviour tuning default is mirrored by a path in `calibration.tunableFields`. Tune using the intended engine's canonical head centre and face-width measurement; do not add package-side mirroring. The initial values are authoring estimates and must be validated with frontal, ±30-degree yaw, roll, near/far, glasses, low-light, pause/resume and rapid filter-switch tests.

## GLB technical contract

- glTF 2.0 binary (GLB).
- Vertex attributes: POSITION, NORMAL, TEXCOORD_0.
- Index accessor: UNSIGNED_SHORT.
- Single opaque sRGB base-colour texture, embedded in every GLB; enable mipmap generation at runtime.
- Front +Z, up +Y; centered pivot; rest scale 1 and identity rotation.
- Actual triangle and draw-call counts are recorded in manifest and validation report.

## Render/occlusion requirements

The renderer must draw the face mesh and head proxy in a depth-only occluder pass before opaque models. Both occluders and the heart model must use the same camera projection/depth convention. The `pushBackFW` value is a small calibration bias; if the renderer uses a different depth convention, it must map this value consistently rather than applying an arbitrary screen-space offset.

## Reproducibility

Run the generator with Python packages `numpy`, `Pillow`, `shapely`, and `trimesh`. The generator rebuilds the three LOD GLBs, thumbnail, atlas, manifest, and validation report.

## Verification status

Static file structure, hashes, GLB container fields, vertex attributes, index widths, and declared triangle/draw-call counts are validated by the included report. No Android runtime import or physical-device visual verification is claimed.
