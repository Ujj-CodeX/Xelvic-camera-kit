# Xelvic Camera Engine

Xelvic is a native camera and face-effects engine built as a **general, data-driven runtime**. A filter is a package of data (a scene description plus assets). The engine is an interpreter for that data. Adding a new filter never means writing renderer, tracker or behavior code.

One persistent face tracker feeds every effect: 2D stickers, face-surface textures, 3D props, multi-layer accessories, beauty and post effects, and later hands, segmentation and world effects.

The target is Snapchat-class smoothness on low-end Android phones without lowering preview quality. The same package format runs on Android and iOS.

This is the build specification. Build in the order of section 21. No performance or accuracy claim is made until it appears in measured results (section 22).

---

## 1. The core law

> **The engine knows capabilities. Packages know filters.**

- The engine implements a fixed set of **primitives**: node types, material templates, behavior types, signal sources, expression functions, tracker outputs.
- A package composes those primitives with data: which nodes, where they attach, how they react, how they degrade.
- Engine source code contains **no filter ids, no node names, no asset names**. A new filter that needs engine code means a missing primitive, and the fix is to add a versioned primitive, never a special case.
- New capability = register a new primitive once. Existing packages keep working untouched.

---

## 2. Architecture in one picture

```text
 INSTALL TIME (once per package)
 .xfilter archive
   → secure extract → schema + hash + budget validation
   → LegacyAdapter (v1 / v2 / face-3d/1 packages → scene model)
   → Compiler: resolve refs, params, selectors, expressions, property paths
   → CompiledEffect (immutable IR: slot tables, bytecode, render list)

 RUN TIME (every frame)
 Camera ─► analysis frame ─► Provider(s) ─► Snapshot ─► Signal layer ─► FaceState
                                                                        │ latest-value
 Render thread: read FaceState                                          ▼
   → SignalBus.evaluate()          (built-in + derived signals → float slots)
   → BehaviorRunner.step(dt)       (clips, maps, triggers, springs → property slots)
   → SceneEvaluator                (node transforms from anchors + slots)
   → RenderGraph                   (occluders → opaque → transparent → post)
   → Preview | Photo | Video

 PerformanceGovernor: tracker cadence, analysis size, tier A/B/C
```

At run time there are **no string lookups, no JSON, no parsing**. Everything was compiled to indexed slots at install.

---

## 3. Rules the whole design follows

1. Track once, render many. Changing a filter never restarts the camera or tracker.
2. The render thread never waits. Inference, decoding, IO and validation run elsewhere. Handoffs are latest-value, never queues.
3. One owner per convention. Axes, mirroring, left/right mapping and face-width normalisation live only in `FacePoseAdapter`. Packages carry no calibration and never compensate.
4. Packages are data. No scripts, no custom shaders, no network, no paths outside the package.
5. Fail closed. Missing feature, unknown node type, bad hash, over-budget asset: reject with a reason and keep the previous filter. Never crash, never black-frame.
6. Work only for what the filter needs. The compiler derives the tracker outputs from what the package references.
7. Preview quality is the last thing reduced.
8. Face data stays on device.
9. Everything is versioned: schema, features, templates. Old packages declare what they need and are matched against what the engine offers.

---

## 4. Module layout

Align names with the existing repository conventions.

```text
engine/
  capture/      CameraSession, AnalysisFrame, FrameClock
  perception/   ProviderRegistry, FaceGeometryProvider, MediaPipeFaceGeometryProvider,
                FaceGeometrySnapshot
  signal/       SignalPipeline, PoseFilter (One-Euro), LatencyCompensator,
                FaceLifecycle, FacePoseAdapter, FaceState
  package/      PackageInstaller, PackageValidator, LegacyAdapters, Compiler,
                CapabilityRegistry, PackagePreloader, UvRegistry
  scene/        CompiledEffect, SignalBus, ExpressionVM, BehaviorRunner,
                SceneEvaluator, PropertySlots, AnchorResolver
  render/       RenderGraph, NodeRendererRegistry, MaterialRegistry,
                SpriteRenderer, ModelRenderer, FaceMeshRenderer, OccluderPass,
                PostRenderer, GlbLoader, TextureCache, GlResourceOwner
  governor/     PerformanceGovernor, TierPolicy, ThermalMonitor
  photo/        StillPipeline, OffscreenRenderer
  debug/        DebugOverlay, ReplayHarness
```

Existing production classes (`MainActivity`, `CameraController`, `FilterRepository`, `InstalledFilter`, `MediaStoreSaver`, Canvas `StickerRenderer` for the fallback path) stay as they are.

---

## 5. The `xfilter` scene format (normative)

### 5.1 Package layout

```text
manifest.json
assets/      glb, png, webp, masks
preview.jpg
```

### 5.2 Manifest top level

| Field | Purpose |
|---|---|
| `schema` | `"xfilter/3"` |
| `id`, `name`, `version`, `description`, `preview` | Identity and catalogue data |
| `requires` | Engine features the package needs (5.3) |
| `assets[]` | Files with hashes and load parameters (5.4) |
| `params{}` | Named tunables with defaults and ranges (5.5) |
| `anchors{}` | Custom anchors from landmarks (5.6) |
| `materials{}` | Material template instances (5.7) |
| `lighting` | Scene lights (5.8) |
| `nodes[]` | The scene (5.9) |
| `signals{}` | Derived signals from expressions (5.10) |
| `behaviors[]` | Clips, maps, triggers, springs (5.11) |
| `lifecycle` | Face lost / re-acquire policy (5.12) |
| `tiers[]` | Per-tier overrides (5.13) |
| `photo` | Still-photo policy (5.14) |
| `budgets` | Declared cost, verified by the validator (5.15) |

### 5.3 `requires`: capability negotiation

```json
"requires": {
  "runtime": 3,
  "features": ["node.model@1", "material.lambert@1", "behavior.clip@1", "behavior.map@1",
               "behavior.trigger@1", "behavior.spring@1", "occluder.faceMesh@1",
               "occluder.ellipsoid@1", "signal.blendshapes@1"],
  "faces": 1
}
```

Features are `name@version`. The engine has a `CapabilityRegistry` listing everything it implements. If any required feature is missing, the package is rejected and the missing list is shown. An optional `optionalFeatures` list may be given; nodes may declare `fallback: "hide"` for graceful degradation. Tracker outputs (pose, blendshape names, landmarks) are not declared by hand: the compiler scans the package and derives them.

### 5.4 `assets[]`

```json
{ "id": "model", "type": "model", "path": "assets/prop.glb", "sha256": "...", "bytes": 507556 }
{ "id": "tex",   "type": "image", "path": "assets/tex.png", "sha256": "...", "bytes": 48918,
  "colorSpace": "sRGB", "mipmaps": true, "maxSize": 1024 }
```

Types: `image`, `model`, `mask`, `lut`. Every asset needs `sha256` and `bytes`. Validation checks file signature, dimensions, decoded pixel budget and total package size before any allocation.

### 5.5 `params`

```json
"params": { "offset": { "type": "vec3", "default": [-0.4, 0.42, 0.64],
                        "min": [-2, -2, -2], "max": [2, 2, 2] } }
```

Referenced anywhere as `"$offset"`. Placement almost always needs on-device calibration, so a backend can override param values per filter without re-exporting the package. Values are always range-checked.

### 5.6 Anchors

Built-in anchors, computed by `AnchorResolver` from the face state, in face-local FW space:

`head_center`, `forehead`, `nose_tip`, `eye_left`, `eye_right`, `eye_center`, `mouth_center`, `chin`, `ear_left`, `ear_right`, `cheek_left`, `cheek_right`.

Left/right are **subject-based** and resolved by `FacePoseAdapter`, never by the package.

Custom anchors:

```json
"anchors": {
  "brow_left": { "landmarks": [ { "id": 105, "w": 0.5 }, { "id": 66, "w": 0.5 } ],
                 "offsetFW": [0.0, 0.05, 0.0] }
}
```

Landmark ids are checked against the pinned topology at install. All distances in packages are in **face widths (FW)**: `FW = ‖M·c234 − M·c454‖`, so every filter scales with every face.

### 5.7 Materials

A material is an instance of an engine-owned **template**, with typed, range-checked parameters. Packages never provide shader code.

| Template | Parameters |
|---|---|
| `unlit@1` | `map`, `color`, `opacity` |
| `lambert@1` | `map`, `color`, `opacity`, uses scene `lighting` |
| `faceTexture@1` | `map`, `mask`, `uvLayout`, `opacity` |

Render state per material: `blend` (`opaque`, `alpha`, `additive`, `multiply`; alpha is premultiplied), `doubleSided`, `backFaceNormal` (`flip`, `keep`), `depthTest`, `depthWrite`, `cull`.

New templates (for example `pbrBasic@1`) are added to `MaterialRegistry` and versioned. A package that uses one lists it in `requires.features`.

### 5.8 Lighting

```json
"lighting": { "space": "camera", "ambient": [0.3, 0.27, 0.2],
              "lights": [ { "dirToLight": [-0.35, 0.65, 0.68], "color": [1, 0.91, 0.72], "intensity": 0.82 } ] }
```

### 5.9 Nodes: the scene

A flat list; hierarchy via `parent`. Every node:

| Field | Meaning |
|---|---|
| `id`, `parent`, `tags[]` | Identity; tags group nodes (`lod_a`, `petals`, `lens`) |
| `type` | `group`, `sprite`, `model`, `faceMesh`, `occluder`, `post` |
| `anchor` | Built-in or custom anchor id (for root-level nodes) |
| `transform` | `position` (FW), `rotationDeg`, `scale`, `rotationFollow` |
| `material` | Material id |
| `visible`, `opacity`, `renderOrder` | Base values |

`rotationFollow`: `full` (head rotation), `roll` (in-plane only), `eyeLine`, `none` (screen-fixed).

Type-specific fields:

| Type | Fields |
|---|---|
| `sprite` | `asset` (image), `size` (`{ "widthFW": 0.5 }` or `heightFW`), `pivot` |
| `model` | `asset` (glb), `rootNode`, `sceneNode`, `animations` resolved from the GLB |
| `faceMesh` | `material` (faceTexture), `region` (optional mask) |
| `occluder` | `shape`: `faceMesh` (`pushBackFW`) or `ellipsoid` (`radiiFW`, `segments`, `clearanceRatioMin`) or `box`; draws depth only |
| `post` | `effect` (allow-listed op, e.g. `colorGrade`, `blur`, `skinSoften`) with typed `params` |

A `model` node instantiates the GLB's node hierarchy. Mesh nodes inside are addressed from behaviors by selector (`model1/petals_outer`) or by tags declared in the manifest's `model.nodeTags` map (glob patterns on GLB node names, evaluated at compile time).

**Selectors** used everywhere: `node:<id>`, `tag:<tag>`, `glb:<nodeId>/<glbNodeName>`.

### 5.10 Signals and expressions

Built-in signal sources (provided by the engine, versioned):

- `face.present`, `face.fade`
- `face.pose.yaw|pitch|roll` (degrees, filtered)
- `face.blend.<name>` (the 52 blendshape scores, smoothed)
- `time.render`, `time.sinceAcquired`
- `input.tap` (optional feature)

Derived signals:

```json
"signals": {
  "smile": "mean(face.blend.mouthSmileLeft, face.blend.mouthSmileRight)",
  "openAmount": "smoothstep(0.25, 0.75, face.blend.jawOpen)"
}
```

**Expression grammar**: numbers, signal references, `param.*` references, unary minus, `+ - * /`, parentheses, and the allow-listed functions `min max clamp abs mean mix smoothstep step sin cos sqrt pow(x, constant)`. No variables, no loops, no recursion, no comparisons beyond `step`. The derived-signal graph must be acyclic (checked at compile). Expressions compile to a small bytecode run by `ExpressionVM` with fixed stack depth. NaN and infinity are clamped to 0. New functions are added to the `ExpressionFunctionRegistry` and versioned.

### 5.11 Behaviors

Four primitive types. Each has an `id`, can be disabled by tiers, and writes to **properties** by selector.

**Property paths:** `<selector>.<prop>` where `<prop>` is `position`, `rotation`, `scale`, `opacity`, `visible`, or `material.<param>`.

**`clip`**: plays a GLB animation.

```json
{ "id": "idle_sway", "type": "clip", "target": "node:flower", "animation": "idle_sway",
  "loop": true, "startOn": "faceAcquired", "speed": 1.0 }
```

**`map`**: shapes a signal into a property.

```json
{ "id": "smile_swell", "type": "map", "input": "sig.smile",
  "inputRange": [0.25, 0.75], "outputRange": [1.0, 1.12], "easing": "smoothstep",
  "clamp": true, "smoothingTau": 0.12, "initial": 1.0,
  "writes": [ { "path": "tag:petals.scale", "mode": "multiply" } ] }
```

Smoothing is frame-rate independent: `y += (target − y)·(1 − exp(−dt/τ))`.

**`trigger`**: edge detection with hysteresis and cooldown, firing actions.

```json
{ "id": "burst_on_open", "type": "trigger", "input": "face.blend.jawOpen",
  "risingAbove": 0.55, "fallingBelow": 0.35, "cooldownMs": 900, "armedInitially": true,
  "onFire": [ { "action": "playClip", "clip": "burst", "restart": "ignoreIfPlaying" } ] }
```

Actions: `playClip`, `stopClip`, `setProperty`, `toggleVisible`, `emit` (custom event). Events a trigger may use instead of a signal: `"on": "faceAcquired" | "faceLost" | "tap" | "<custom>"`.

**`spring`**: inertia that follows pose change.

```json
{ "id": "head_inertia", "type": "spring", "target": "node:flower", "mode": "rotationInertia",
  "stiffness": 90, "dampingRatio": 0.3, "gain": 0.35, "maxAngleDeg": 8,
  "lowPassTau": 0.04, "maxAngularAccel": 80, "substepMaxSec": 0.00833 }
```

Modes: `rotationInertia`, `positionInertia`.

**Composition order (fixed in the engine, identical for every filter)**

```text
base value
 → clips          (scale ×, rotation ×, translation +)
 → maps           (in declared order, by each write's mode: set | multiply | add)
 → springs        (rotation = spring_local · current)
 → lifecycle fade (scale ×, opacity ×)
```

So for any node: `scale = rest × Π clips × Π maps × fade` and `rotation = spring · clip · rest`. Declared order resolves ties.

### 5.12 Lifecycle

```json
"lifecycle": {
  "faceLost":  { "holdMs": 150, "fadeMs": 200, "fadeCurve": "cubicOut", "afterFade": "hideAndReset" },
  "reacquire": { "cancelFade": true, "easeInMs": 100, "preserveState": true, "replayStartOn": false }
}
```

`afterFade`: `hideAndReset` (hide, reset all behavior state: clip times 0, maps to `initial`, springs 0, triggers armed, last fire −∞) or `keep`. Defaults when omitted: hold 200 ms, fade 250 ms, `hideAndReset`. Selecting a new filter is not a face loss.

### 5.13 Tiers

```json
"tiers": [
  { "id": "A", "show": ["tag:lod_a"], "hide": ["tag:lod_b", "tag:lod_c"], "textureMaxSize": 1024 },
  { "id": "B", "show": ["tag:lod_b"], "hide": ["tag:lod_a", "tag:lod_c"], "textureMaxSize": 512 },
  { "id": "C", "show": ["tag:lod_c"], "hide": ["tag:lod_a", "tag:lod_b"], "textureMaxSize": 256,
    "disable": ["smile_swell", "head_inertia", "burst_on_open"] }
]
```

Hidden nodes are neither drawn nor animated. Disabled behaviors are skipped entirely. `minTier` may be set for a filter that cannot run low. A package with one tier is valid.

### 5.14 Still-photo policy

```json
"photo": { "clips": "settle", "triggers": "disarmed", "timeSec": null }
```

`settle`: clips jump to their hold-last or t=0 state; loops start at t=0. Deterministic and documented per filter.

### 5.15 Budgets and limits

`budgets` declares per-tier `drawCalls`, `triangles`, `textureBytes`. The validator measures actual values and rejects a mismatch or anything above the engine's global caps (maximum nodes, behaviors, signals, expression length, texture sizes, total package size). All caps are engine configuration, not package data.

---

## 6. Compile and install

```text
archive
 → secure extraction (no ZIP-slip, entry and size limits)
 → schema validation (strict; unknown fields rejected)
 → hashes, byte counts, file signatures, dimension and pixel budgets
 → LegacyAdapter if schema is older
 → feature negotiation against CapabilityRegistry
 → reference resolution (assets, anchors, materials, selectors, params)
 → mesh packages: UV layout pinned to the exact tracker model (UvRegistry)
 → GLB validation: nodes, attributes, uint16 indices, animations, rest transforms
 → expression compile, acyclic check, property-path to slot resolution
 → tracker output derivation (pose? blendshape names? landmarks? custom anchors?)
 → budget measurement against declared and global limits
 → CompiledEffect (immutable)
```

Install is transactional. The first failure stops installation and reports the reason. The package validator runs in CI for every shipped package.

**LegacyAdapters** (so nothing existing breaks):

| Old format | Mapped to |
|---|---|
| Schema v1 (2D layers) | `sprite` nodes; anchors map to built-ins; `scaleMode` becomes `size` in FW; `rotationMode` becomes `rotationFollow` |
| Schema v2 (`FACE_MESH_TEXTURE`) | one `faceMesh` node plus a `faceTexture` material |
| `xfilter-face-3d/1` (sunflower style) | `model` node, occluder nodes, behaviors, tiers, lifecycle |

New packages are authored directly in `xfilter/3`.

---

## 7. Engine capabilities and extension points

Everything a package can use comes from a registry. Each entry is versioned and listed in `CapabilityRegistry`.

| Registry | Adds |
|---|---|
| `ProviderRegistry` | Face, hands, segmentation, world providers |
| `NodeRendererRegistry` | New node types |
| `MaterialRegistry` | New material templates |
| `BehaviorRegistry` | New behavior types |
| `SignalSourceRegistry` | New signals (hands, audio, sensors, segmentation masks) |
| `ExpressionFunctionRegistry` | New expression functions |

**Rule:** to support a new kind of filter, add a primitive to a registry with a feature name and version. Never branch on a filter id, node name or asset name. Deferred capabilities (hand tracking, hair/background segmentation, world anchors, audio) are added here later; the manifest already has the `requires` and `signals` slots for them.

---

## 8. Perception

```kotlin
interface FaceGeometryProvider {
  fun start(config: ProviderConfig)
  fun stop()
  fun submit(frame: AnalysisFrame)        // never blocks
  val modelSha256: String
}
```

**Snapshot** (immutable, defensive copies): monotonic `timestampNs`, image size and rotation, 478 landmarks, 4×4 pose matrix (validated finite), requested blendshapes only, `modelSha256`, `topologyId`, the **exact 3×3 image-to-view transform captured at submit time for that frame**, tracking status.

**MediaPipe provider rules**

- `LIVE_STREAM`, monotonic timestamps, one inference in flight, newer frames replace pending ones.
- Pose matrix and blendshape output are switched on only when the compiled effect needs them. No face-dependent signal means the face provider can stop.
- Analysis frame is independent of preview. Start near 320 px on the long edge, tune by profiling.
- Tracker cadence ceiling 30 Hz, floor 15 Hz, owned by the governor.
- CPU delegate first. A runtime switch enables the GPU delegate per device class only where profiling shows a win.
- YUV to ARGB conversion is the first hotspot on low-end phones: reuse buffers, avoid per-frame allocation, compare output formats before other optimisation.
- One tracker only. ML Kit is used only by the fallback Canvas path for legacy sprite packages.

---

## 9. Signal layer

This is where smoothness comes from.

1. **Decompose** the pose matrix into translation, quaternion rotation and scale. Never low-pass raw 4×4 entries.
2. **Filter** each quantity separately with a One-Euro filter (quaternion smoothing with sign continuity). Parameters are per quantity and tuned on replay recordings.
3. **Blendshapes** are smoothed independently from rigid pose.
4. **Latency compensation:** the tracker result describes an older frame. Extrapolate the filtered pose to render time over a bounded, measured horizon. One shared clock; never mix camera time with wall-clock time. The render loop runs at display rate on the predicted pose even when the tracker is slower.
5. **FaceLifecycle:** `NO_FACE → ACQUIRING → TRACKING → HOLD → FADING → NO_FACE`, with re-acquisition returning to `TRACKING` as the package's `lifecycle` specifies.
6. **FacePoseAdapter** is the single owner of: canonical axes (x subject-right, y up, z toward camera), front-camera mirror (mirror the final projection once), subject-left/right mapping, FW normalisation and the face-local frame. Any left/right, axis or FOV error is fixed here only.

Output: one immutable `FaceState` (lifecycle, fade, rigid pose, FW, smoothed expressions, acquisition time, projection parameters), published to an atomic latest-value reference.

---

## 10. Coordinate and projection contract

Every API names its space.

1. Sensor / `ImageProxy` pixels
2. Rotation metadata and oriented analysis image
3. Tracker-normalised landmarks (and Z convention)
4. Canonical face geometry with UV and topology version
5. Runtime geometry and the facial transformation matrix
6. Camera projection (aspect, units, handedness, matrix convention matching the geometry)
7. `PreviewView` crop/scale/rotation, or the single GL scene projection
8. GL clip space / NDC and viewport
9. Front-camera mirror in exactly one stage
10. Still-photo space, computed independently
11. Face-local FW space (the space all package data uses)

The transformation matrix is not automatically a projection matrix: verify canonical-to-runtime and camera projection separately. The FOV used to compute the pose must equal the FOV used to render. Occlusion needs one shared depth convention. Use a shared `ViewPort`/`UseCaseGroup` where lifecycle allows. Never read `PreviewView` properties from the analysis executor.

---

## 11. Render engine

### 11.1 Pass order

```text
1 camera frame (no depth)
2 clear depth
3 occluder nodes (depth only; face mesh pushed back along view rays; ellipsoids)
4 opaque nodes (depth test/write per material)
5 transparent nodes (sorted by renderOrder, then depth)
6 post nodes
7 output → Preview | Photo | Video
```

### 11.2 Node renderers

| Node | Renderer notes |
|---|---|
| `sprite` | Textured quad in face-local space; sized in FW; Canvas fallback exists for the legacy path only |
| `model` | `GlbLoader`: glTF 2.0, embedded PNG, `POSITION`/`NORMAL`/`TEXCOORD_0`, uint16 indices, node hierarchy, TRS, animation sampler (LINEAR; SLERP for rotation). Static GPU buffers |
| `faceMesh` | Pinned 468-point tessellation, live vertices updated per frame, static topology and UVs reused, mask and alpha handling, region masks protect eyes, lips and mouth interior |
| `occluder` | Depth-only draw (colour mask off) |
| `post` | Allow-listed effect passes with typed params (needs single-scene composition) |

### 11.3 Materials and shading

Material templates are fixed engine shaders. `lambert@1`: `C = tex_linear · clamp(ambient + Σ intensity·color·max(dot(N,L),0), 0, 1)` with lights in camera space; textures decoded sRGB to linear; back-face normal flip for double-sided meshes; node scale floor to avoid degenerate normals. Premultiplied alpha throughout.

### 11.4 Engine rules

- GLES 3.0 minimum for the node renderers. The existing GLES 2.0 code remains only as the diagnostic wireframe.
- Zero allocation in the frame loop.
- Warm shaders and pipelines at startup and at effect preload, so a filter's first frame never hitches.
- All GL calls, uploads, deletes and buffer updates happen on the GL thread. Old effect resources are released only after queued GL work referencing them drains.
- `TextureCache`: bounded LRU, background decode, GL-thread upload and delete. Honor each tier's `textureMaxSize`.
- A thin backend interface (GLES / Metal) sits under the render graph so scenes stay platform-neutral.

### 11.5 Composition

Ship with a transparent GL surface over `PreviewView`, behind a `CompositionSurface` interface. Move to a single GL scene that imports the camera texture when post effects or video begin. Nothing in the scene model or compiled effect changes when the surface changes. Before moving, run the offset test: a marker on a face, high-speed recording of preview vs overlay under fast head motion, decision by measured lag. Handle pause/resume, surface recreation and context loss by recreating and re-uploading all GPU resources.

---

## 12. Per-frame runtime order

```text
1  read latest FaceState (one atomic read)
2  SignalBus: built-in signals → derived signals (topological order) into float slots
3  BehaviorRunner: events (faceAcquired, faceLost, tap) → triggers → clips → maps → springs
4  Lifecycle fade
5  SceneEvaluator: anchors → node world transforms from property slots
6  RenderGraph draws
```

`dt` is clamped (0.001 to 0.05 s). Fixed-size slot arrays and a preallocated expression stack mean this loop allocates nothing.

---

## 13. Performance governor

**Tiers** are package-defined (section 5.13). The governor picks which tier is active.

**Inputs:** frame time p50/p95, tracker rate, result age at render, dropped frames, thermal status, memory pressure.

**Order of degradation**

1. Lower tracker cadence (not below 15 Hz), then analysis size.
2. Move to a lower tier: its `disable` list turns off optional behaviors.
3. Lower tier LOD and texture size.
4. Reduce occluder segment counts.
5. Preview quality only with explicit product approval.

**Starting hysteresis values** (tune on devices): demote when p95 frame time exceeds 33 ms for 1.5 s; promote only after 10 s of headroom; minimum 5 s dwell per tier; no changes inside the dwell window. Low-end devices start at the lowest tier and promote; high-end start at the highest.

**Targets to validate, not promises:** steady 30 fps render, effective tracking near 20 Hz or better on low-end devices, camera-to-screen latency near 100 ms, no visible heating or throttling collapse in 5 minutes.

---

## 14. Threads and resource ownership

| Work | Owner |
|---|---|
| UI, `PreviewView`, permissions | Main thread |
| CameraX analysis delivery | Analysis executor; close every `ImageProxy` exactly once |
| Inference | Provider thread; monotonic timestamps; one in flight |
| Signal pipeline | Provider callback or dedicated signal thread, never the GL thread |
| `FaceState` handoff | Atomic latest-value reference |
| Scene evaluation, GL calls, uploads, deletes | GL render thread only |
| Decode, archive validation, hashing, compile | Background IO workers |
| Photo inference and offscreen composite | Background worker with an owned offscreen context |

Handle rotation, lens switch, background/foreground and context loss by re-deriving transforms through `FacePoseAdapter` and recreating GPU resources, never by patching inside effects.

---

## 15. Effect switching and preloading

- Switching replaces the active `CompiledEffect` and its GPU state only. Camera and tracker keep running. If the new effect needs different tracker outputs, only the provider output flags change.
- `PackagePreloader` prepares the next filters in the carousel in the background: validate, compile, decode, build GPU resources, warm shaders; large uploads are spread over frames.
- Any failure leaves the previous filter active and flags the failed one with a reason.
- Static topology, UV layouts and recent textures sit in a bounded LRU.

---

## 16. Still photo (video later)

The shutter happens at a different time and resolution, so preview pose is not authoritative.

1. Capture with `ImageCapture`.
2. Run fresh still-image face inference on a background worker. A downsampled copy may be used if the transform back to full resolution is kept.
3. Evaluate the same `CompiledEffect` on the new geometry, using the manifest's `photo` policy.
4. Render offscreen through an explicitly owned EGL context (sharing defined and tested), with a documented CPU fallback for simple sprite effects or incompatible devices.
5. No-face policy: save without the effect and tell the user, with a one-tap retake. Never undefined.
6. Match crop, aspect and mirror deliberately; bound bitmap and texture memory and check peak memory on real devices.

Video comes after the single-scene composition and lifecycle are stable. The same render graph output feeds the encoder.

---

## 17. Platforms

**Android** (this repo): Kotlin, CameraX, OpenGL ES 3.0, MediaPipe Tasks.

**iOS:** Swift, AVFoundation, Metal, MediaPipe Tasks (iOS). Zero-copy camera via `CVMetalTextureCache`. The schema, capability names, expression grammar, composition order, `FacePoseAdapter` conventions and GLB assets are shared. Both platforms must reproduce a shared set of recorded pose/landmark inputs with expected anchor positions and signal values within tolerance, and match golden images within a perceptual tolerance.

**React Native:** camera, tracker, signal, scene and renderer live in a native view/module. Frames, landmarks and poses never cross the JS bridge. JS sends commands (select filter, switch camera, capture, tier override, param override) and receives low-rate events (filter ready/failed with reason, capture done, permission state). Catalogue download and caching sit in native code so preloading does not depend on JS timing.

---

## 18. Examples (all data, no engine code)

### 18.1 2D sticker

```json
{ "schema": "xfilter/3", "id": "star_cheek", "name": "Star Cheek", "version": "1.0.0",
  "requires": { "runtime": 3, "features": ["node.sprite@1", "material.unlit@1"], "faces": 1 },
  "assets": [ { "id": "star", "type": "image", "path": "assets/star.png", "sha256": "...", "bytes": 10240 } ],
  "materials": { "m": { "template": "unlit@1", "params": { "map": "star" }, "blend": "alpha" } },
  "nodes": [ { "id": "star", "type": "sprite", "anchor": "cheek_left", "material": "m",
               "size": { "widthFW": 0.18 }, "transform": { "position": [0, 0, 0.02], "rotationFollow": "full" } } ] }
```

### 18.2 Glasses (two anchors, transparent lens, head occluder, no behaviors)

```json
{ "schema": "xfilter/3", "id": "round_glasses", "name": "Round Glasses", "version": "1.0.0",
  "requires": { "runtime": 3, "features": ["node.model@1", "material.lambert@1", "material.unlit@1",
                "blend.alpha@1", "occluder.ellipsoid@1"], "faces": 1 },
  "assets": [ { "id": "g", "type": "model", "path": "assets/glasses.glb", "sha256": "...", "bytes": 300000 } ],
  "materials": { "frame": { "template": "lambert@1", "params": { "color": [0.1, 0.1, 0.1] } },
                 "lens":  { "template": "unlit@1", "params": { "color": [0.6, 0.8, 1.0], "opacity": 0.25 },
                            "blend": "alpha", "depthWrite": false } },
  "lighting": { "space": "camera", "ambient": [0.4, 0.4, 0.4],
                "lights": [ { "dirToLight": [0, 0.6, 0.8], "color": [1, 1, 1], "intensity": 0.7 } ] },
  "nodes": [
    { "id": "head", "type": "occluder", "anchor": "head_center", "shape": "ellipsoid",
      "radiiFW": [0.56, 0.66, 0.68], "segments": [16, 24] },
    { "id": "glasses", "type": "model", "anchor": "eye_center", "asset": "g",
      "transform": { "position": [0, 0, 0.1], "scale": 1.0 }, "tags": ["frame"] } ] }
```

### 18.3 3D prop with behaviors (the sunflower, condensed)

```json
{ "schema": "xfilter/3", "id": "sunflower_forehead", "name": "Sunflower Forehead", "version": "1.0.0",
  "requires": { "runtime": 3, "features": ["node.model@1", "material.lambert@1", "behavior.clip@1",
                "behavior.map@1", "behavior.trigger@1", "behavior.spring@1",
                "occluder.faceMesh@1", "occluder.ellipsoid@1", "signal.blendshapes@1"], "faces": 1 },
  "params": { "offset": { "type": "vec3", "default": [-0.4, 0.42, 0.64], "min": [-2,-2,-2], "max": [2,2,2] } },
  "signals": { "smile": "mean(face.blend.mouthSmileLeft, face.blend.mouthSmileRight)" },
  "nodes": [
    { "id": "face_occ", "type": "occluder", "shape": "faceMesh", "pushBackFW": 0.015 },
    { "id": "head_occ", "type": "occluder", "anchor": "head_center", "shape": "ellipsoid",
      "radiiFW": [0.56, 0.66, 0.68], "segments": [16, 24] },
    { "id": "flower", "type": "model", "anchor": "head_center", "asset": "model",
      "transform": { "position": "$offset" },
      "nodeTags": { "petals": "sunflower_petals_*", "lod_a": "*_lod_a", "lod_b": "*_lod_b", "lod_c": "*_lod_c" } } ],
  "behaviors": [
    { "id": "bloom_in", "type": "clip", "target": "node:flower", "animation": "bloom_in", "startOn": "faceAcquired", "holdLast": true },
    { "id": "idle_sway", "type": "clip", "target": "node:flower", "animation": "idle_sway", "loop": true, "startOn": "faceAcquired" },
    { "id": "smile_swell", "type": "map", "input": "sig.smile", "inputRange": [0.25, 0.75], "outputRange": [1.0, 1.12],
      "easing": "smoothstep", "smoothingTau": 0.12, "initial": 1.0, "writes": [ { "path": "tag:petals.scale", "mode": "multiply" } ] },
    { "id": "burst_on_open", "type": "trigger", "input": "face.blend.jawOpen", "risingAbove": 0.55, "fallingBelow": 0.35,
      "cooldownMs": 900, "armedInitially": true, "onFire": [ { "action": "playClip", "clip": "burst", "restart": "ignoreIfPlaying" } ] },
    { "id": "head_inertia", "type": "spring", "target": "node:flower", "mode": "rotationInertia",
      "stiffness": 90, "dampingRatio": 0.3, "gain": 0.35, "maxAngleDeg": 8 } ],
  "lifecycle": { "faceLost": { "holdMs": 150, "fadeMs": 200, "afterFade": "hideAndReset" },
                 "reacquire": { "cancelFade": true, "easeInMs": 100, "preserveState": true, "replayStartOn": false } },
  "tiers": [ { "id": "A", "show": ["tag:lod_a"], "hide": ["tag:lod_b", "tag:lod_c"] },
             { "id": "B", "show": ["tag:lod_b"], "hide": ["tag:lod_a", "tag:lod_c"] },
             { "id": "C", "show": ["tag:lod_c"], "hide": ["tag:lod_a", "tag:lod_b"],
               "disable": ["smile_swell", "head_inertia", "burst_on_open"] } ] }
```

(Assets, hashes, budgets and some fields omitted for length. The existing `sunflower_forehead` package converts through the legacy adapter and is the first conformance package.)

---

## 19. Keeping it general (enforced, not hoped)

1. **No filter strings in engine code.** CI greps engine sources for known filter ids, node names and asset names and fails the build on a hit.
2. **Data-only conformance suite.** Reference packages built only from manifests and assets:
   - `ref_sticker_2d` (sprite on an anchor)
   - `ref_face_paint` (faceMesh with mask and UV layout)
   - `ref_sunflower` (model, clips, map, trigger, spring, occluders, tiers, lifecycle)
   - `ref_glasses` (two anchors, alpha blend, occluder, no behaviors)
   - `ref_multi_prop` (several nodes, custom anchor, tap trigger, custom event)
   - `ref_beauty` (post node, once single-scene composition exists)
3. **Zero-diff rule.** After the engine passes the first reference package, every later reference package must pass with **no engine source change**. A needed code change means a missing primitive: add it to a registry with a feature name and version, update `CapabilityRegistry`, and add a conformance package for it.
4. **Golden images and signal vectors** per reference package on every platform.
5. **Unknown means reject.** Unknown schema, node type, material template, behavior type, function or feature is rejected with a clear reason, never ignored.

---

## 20. Testing and device verification

**Replay harness:** test-only playback of consented local frames and/or timestamped geometry results, so changes to coordinates, filtering, behaviors, rendering and occlusion are compared on identical input. No private footage is committed.

**Unit tests:** snapshot validation; pose decomposition and filters; lifecycle transitions; expression VM (grammar, limits, NaN handling); compiler (selector, param and path resolution, cycle detection); behavior runner (each primitive, composition order, reset semantics); capability negotiation; GLB validation; governor hysteresis; legacy adapter conversions.

**Golden images:** frontal, left/right yaw, roll, near/far; smile, mouth open, blink, eyebrows; front/rear and mirroring; crop, aspect, rotation, display sizes; face loss and re-entry; rapid filter switching; texture seams and alpha; occlusion edges; high-resolution photo; no-face capture.

**On-device checks (before any face-anchored package ships)**

| Check | Pass condition |
|---|---|
| Axes | Gizmo at `head_center`: X red right, Y green to top of head, Z blue toward camera |
| FOV | Canonical wireframe tracks the live face at yaw 0 and about 40° |
| Side | Negative X is the subject's anatomical left under the active camera and mirror policy |
| Cheek ids | Vertices 234 and 454 are the cheek-side points used for FW |
| Blendshapes | Required names present in tracker output |
| Anchors | Built-in anchors sit on the expected facial features |
| Head proxy | Top, temple width and rest clearance confirmed with the wireframe |
| Latency | Fast head turn shows no visible detachment of a prop |

**Device matrix:** two low-end Android, one mid-range, one flagship, one iPhone (when iOS starts). Record model, OS, build variant, model asset version, camera, lighting, pose cases.

**CI:** GitHub Actions runs unit tests, the package validator and the conformance suite on the research branch and on pull requests into `main`.

---

## 21. Build order

Each step has an exit gate. Do not start the next until the gate is met.

1. **Alignment.** Wireframe and axes gizmo aligned on a physical device, front and rear, with rotation. Pose matrix and blendshapes become requestable outputs. *Gate: the axes, FOV, side, cheek id, blendshape and anchor checks pass.*
2. **Signal layer.** `SignalPipeline`, One-Euro, quaternion smoothing, latency compensation, `FaceLifecycle`, `FacePoseAdapter`, `FaceState`. *Gate: replay shows lower jitter than raw tracker; fast motion shows no detachment; lifecycle matches section 9.*
3. **Package pipeline.** `CapabilityRegistry`, schema, `Compiler`, `ExpressionVM`, `SignalBus`, `LegacyAdapters`, validator. *Gate: legacy v1, v2 and the sunflower package compile to `CompiledEffect`; invalid packages fail closed with reasons.*
4. **Scene runtime and renderers.** `BehaviorRunner`, `SceneEvaluator`, `AnchorResolver`, sprite/model/occluder renderers, `GlbLoader`, material templates. *Gate: `ref_sunflower` passes Appendix A on device.*
5. **Generality gate.** Add `ref_glasses` and `ref_multi_prop` as data only. *Gate: both pass with zero engine source diff, and the CI hardcoding check is green.*
6. **Governor and preload.** `PerformanceGovernor`, `ThermalMonitor`, `PackagePreloader`, `TextureCache`. *Gate: a low-end device holds target frame time over 5 minutes with no tier flip-flop or switch hitches.*
7. **Composition and lifecycle.** `CompositionSurface`, offset test, context loss and resume, background/foreground. *Gate: recorded surface decision; no crash on context loss.*
8. **Face mesh textures.** Verify a UV layout against the pinned task bundle, pin its SHA-256 in `UvRegistry`, enable `faceMesh` and masks. Never use legacy canonical OBJ UVs unverified: MediaPipe issue #5760 documents ordering incompatibility with newer landmark output. *Gate: `ref_face_paint` stays attached through smile, blink and turn; no topology mismatch; bounded memory.*
9. **Photo parity.** `StillPipeline`, offscreen rendering, no-face policy. *Gate: photo matches preview intent on the test set; bounded peak memory.*
10. **Post and beauty** on the single-scene composition (`ref_beauty`).
11. **iOS engine** against the shared schema, vectors and goldens.
12. **Later:** hands, segmentation, world anchors, audio signals, video, optical flow and learned-rigidity research, each added through the registries (section 7).

---

## 22. Measured results (fill only from real runs)

| Metric | Result |
|---|---|
| Alignment, axes, FOV, side mapping | NOT MEASURED |
| Pose jitter raw vs filtered | NOT MEASURED |
| Tracker inference p50/p95 per device | NOT MEASURED |
| Render GPU time per tier | NOT MEASURED |
| Result age at render | NOT MEASURED |
| Dropped frames, filter-switch latency | NOT MEASURED |
| 5-minute thermal behavior | NOT MEASURED |
| Peak Java/native/GPU memory | NOT MEASURED |
| Preview vs photo consistency | NOT MEASURED |
| Context loss and resume | NOT MEASURED |

---

## 23. Privacy

Frames, landmarks, blendshapes and geometry are processed locally. Raw frames and face geometry are never uploaded or stored as telemetry. A user-consented debug clip is test data, stored separately and removed per policy. Production telemetry is aggregate only: frame time, tier, filter id, error codes, and download/use counts per filter.

---

## 24. Technology baseline

Verify checked-out files before changing any version. Do not upgrade dependencies as a side effect of feature work.

| Component | Version / setting |
|---|---|
| Android Gradle Plugin | 9.1.1 |
| Kotlin | 2.2.10 |
| CameraX (core, camera2, lifecycle, view) | 1.5.0 |
| ML Kit Face Detection (legacy sprite path) | 16.1.7 |
| MediaPipe Tasks Vision | 1.1.1 |
| compileSdk / minSdk | API 36 / API 24 |
| Application ID / Namespace | `com.aistudio.xelvic.krvpmz` / `com.example` |

Model asset: `app/src/main/assets/face_landmarker.task` (binary, downloaded locally; its SHA-256 is attached to every snapshot).

---

## 25. Build, run and branch workflow

Requirements: Android Studio matching the AGP/Kotlin versions, Android SDK and build-tools, Git, Python 3 with `jsonschema`, and **a physical Android device** for camera, timing, GPU, thermal and lifecycle validation.

```powershell
cd "$env:USERPROFILE\StudioProjects"
git clone https://github.com/Ujj-CodeX/XelvicFilterbase.git
cd XelvicFilterbase
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c research/face-mesh-texture-occlusion
```

If the branch already exists: `git switch research/face-mesh-texture-occlusion`, then `git status`. Do not discard uncommitted changes blindly. If `--ff-only` fails, inspect instead of forcing.

```powershell
.\gradlew.bat testDebugUnitTest
.\gradlew.bat assembleDebug
python -m pip install jsonschema
python tools/validate_package.py path\to\package
```

Change workflow: confirm branch and tree; one scoped change set; run tests, build, the package validator and the conformance suite; test on a physical device before claiming runtime validation; commit only intended source, tests and docs (never signing files, secrets, `.env`, APKs or personal footage); push the research branch when ready; never push unreviewed work to `main`. `main` stays the production baseline.

---

## Appendix A. Acceptance test: `ref_sunflower`

The scene runtime is correct when all of these hold on a physical device:

1. The package installs; schema, hashes, GLB structure, tiers and budgets validate.
2. One bloom on the subject-left upper forehead/temple above the eyebrow, slight hairline overlap, eye uncovered, correct side on front and rear camera.
3. At large yaw the bloom hides correctly behind the face and never pops through the head.
4. `bloom_in` plays on acquisition and ends at unit scale; `idle_sway` loops seamlessly.
5. `burst_on_open` fires on the `jawOpen` threshold with cooldown; `smile_swell` scales smoothly within its range.
6. `head_inertia` follows its parameters and stays within the angle bound.
7. Face loss follows hold, fade and reset; re-acquisition preserves state as specified.
8. Tier C disables the three listed behaviors and renders its reduced draw calls.
9. Tier switching under load shows no flip-flop.
10. Switching to or from the filter causes no visible hitch.

Package fix required before shipping: the package README lists the layer offset as `[-0.40, 0.42, 0.58]`, the manifest has `[-0.4, 0.42, 0.64]`. The manifest is the source of truth.

## Appendix B. Glossary

| Term | Meaning |
|---|---|
| FW | Face width, `‖M·c234 − M·c454‖`, the unit for all package distances |
| M | Facial transformation matrix from the tracker |
| IR / CompiledEffect | Immutable compiled form of a package: slot tables, bytecode, render list |
| Primitive | An engine-implemented node type, material, behavior, signal or function |
| Feature | A versioned primitive name (`behavior.spring@1`) used for negotiation |
| Selector | `node:`, `tag:` or `glb:` reference to scene elements |
| One-Euro filter | Adaptive low-pass filter: smooth at rest, responsive in motion |
| Occluder | Geometry drawn depth-only to hide virtual objects behind real surfaces |
| Tier | Quality level selecting visible nodes, enabled behaviors and texture sizes |

## References

- MediaPipe Face Landmarker (Android): <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/android>
- MediaPipe Face Landmarker overview: <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker>
- MediaPipe face geometry proto: <https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/protos/face_geometry.proto>
- MediaPipe issue on legacy canonical OBJ ordering: <https://github.com/google-ai-edge/mediapipe/issues/5760>
- CameraX image analysis: <https://developer.android.com/media/camera/camerax/analyze>
- CameraX `ImageAnalysis.Builder`: <https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Builder>
- Snap Research, *Stabilized Real-time Face Tracking via a Learned Dynamic Rigidity Prior* (SIGGRAPH Asia 2018): <https://research.snap.com/publications/stabilized-real-time-face-tracking-via-a-learned-dynamic-rigidity-prior.html>
