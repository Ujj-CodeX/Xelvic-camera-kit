# Furry Glasses — Xelvic Mesh Engine R&D Authoring Package

An oversized plush-pink rounded-square eyewear prop inspired by the supplied reference. The model has two genuinely open eye apertures, a padded central bridge kept outside the eye-opening polygons, nose supports, and curved rear temple arms. The prop is designed to follow the bilateral eye-line and rigid head pose; it is not driven by smile/mouth expression deformation.

## Runtime compatibility

**This is an R&D authoring ZIP, not importable by the current Xelvic runtime.** The current `FilterPackageManager` only accepts `json/png/webp` archive entries, and the current app does not yet have the GLB loader, model material path, 3D render pass, or runtime face/head/temple occluder implementation. The package uses the proposed `xelvic-filter-rnd` schema v3 authoring contract. Do not rename it to `.xfilter` or import it into the production app.

## Package contents

- `manifest.json` — FACE_WIDTH units, eye-line anchor, rigid attachment, model LODs, face/head/temple occlusion metadata, lifecycle, tunable calibration values, and render-pass declaration.
- `models/furry_glasses_lod_a.glb` — highest contour detail.
- `models/furry_glasses_lod_b.glb` — medium contour detail.
- `models/furry_glasses_lod_c.glb` — low contour detail. All three preserve the two frames, two apertures, bridge, nose supports and both temples.
- `assets/furry_plush_pink_srgb.png` — embedded plush-fibre base-colour texture, also included separately for inspection.
- `thumbnail.png` — package preview.
- `source/build_furry_glasses.py` — reproducible mesh, texture, thumbnail, manifest and ZIP generator.
- `validation/report.json` — hashes, GLB accessor/material checks, actual triangle/draw-call counts and static requirements.

## Placement and geometry

- Coordinate convention: `+X subject-right`, `+Y up`, `+Z camera`; all geometry is in Face Width (`FW`) units.
- Main anchor: `eyes_center`, derived from bilateral eye landmarks. Rotation follows the eye-line / rigid head pose.
- Front-frame width is fit to the measured inter-pupillary distance (front-frame-width/IPD ratio `1.96`); the curved temples extend rearward beyond the front-frame bounds. Initial offset is `x=0`, `y=0`, `z=0.39 FW` relative to the engine's eye-centre anchor. These are calibrated starting values, not universal final numbers.
- No mirror/flip/side compensation is baked into the package. The engine applies front-camera preview mirroring exactly once.
- Eye openings are real holes. There are no dark lens discs or transparent lens quads covering the eyes.
- GLB uses glTF 2.0, `POSITION`, `NORMAL`, `TEXCOORD_0`, unsigned-short indices, one combined opaque mesh/material (one declared draw call), embedded sRGB base-colour texture, high roughness, and mipmaps expected at runtime.

## Occlusion and render order

This eyewear is an opaque prop in **Pass 5 `OPAQUE_MODELS`**. It must render after **Pass 4 `CLEAR_DEPTH_AND_OCCLUDERS`**. The face mesh, head ellipsoid and bilateral ear/temple proxies are depth-only (`colorWrite=false`, `depthWrite=true`). The temple arms curve rearward so the head/temple proxies can hide the arms during turns. No transparent lens pass is needed because the apertures are open.

The manifest contains `pushBackFW`, head-ellipsoid centre/radii and temple-proxy path/radius as tuning values. These values must be calibrated in the target renderer's canonical depth convention. A static package cannot guarantee perfect clipping at every yaw or every anatomy; verify at frontal, ±30° and ±40° yaw, plus near/far distance and face-size changes. Do not fix depth errors with arbitrary screen-space offsets.

## Calibration checklist

1. Frontal eye alignment: aperture centres cover both eyes symmetrically, bridge rests above nose bridge, frame top/bottom do not drift up/down.
2. Yaw: `-40°`, `-30°`, `0°`, `+30°`, `+40°`. Temple arms must disappear behind the head progressively; front frame must remain in front of the eye sockets.
3. Roll: follow eye-line; no independent asset rotation or mirror.
4. Distance: arm's length and approximately 1.5 m; face-width fitting should preserve proportions.
5. Expressions: blink, mouth open, smile. The rigid glasses must not jump when lower-face landmarks deform.
6. Front/rear cameras: exactly one engine-owned mirror transform.
7. Pause/resume, camera switch, filter switch and face lost/re-entry: hold → fade → reset; no tracker restart on ordinary filter switching.
8. Glasses already worn, low light and profile views: document known overlap/occlusion limitations.

## Validation status

The included report verifies archive structure, asset byte counts/SHA-256, GLB v2 container validity, required vertex attributes, unsigned-short indices, opaque embedded material/texture, non-zero triangle counts, one mesh/material primitive per LOD, and LOD component preservation by design. This is **static validation only**. No Xelvic GL runtime import, camera alignment, physical-device test, yaw occlusion, or performance measurement is claimed.
