# Xelvic Engine: Filter-Agnostic Pass Graph, Generalized Beauty/Soft-Focus & Multi-Effect Pipeline Plan

This comprehensive architectural plan unifies the **9-Stage Data-Driven Pass Graph**, **Composable Region Mask Algebra**, **Zero-Lag Snapchat-Level Beauty / Soft-Focus / Subtle Face Deform Primitives**, **Classic Camera Grading**, and **3D Try-On** into a single **100% package-driven (`manifest.json`) engine** with zero filter-specific hardcoding.

---

## Where to Start Right Now (Execution Status & Immediate Next Step)

> [!IMPORTANT]
> **Completed Foundation (Flows 1–4 are DONE & Verified in Repo)**:
> - **Flow 1 (`Chunk B0`)** — *Completed*: Direct GPU camera input (`CameraInputSurface` + `FrameRing` with `delayDepth = 0`), `PATH:A` default with runtime `PATH:A / PATH:B` HUD toggle, camera-pixel passes ordered before 3D occluders/models, and single-pass photo parity.
> - **Flow 2 (`Chunk G0`)** — *Completed*: Data-driven `PassGraph` executor, pooled `RtPool` (`FULL 1.0x`, `HALF 0.5x`, `QUARTER 0.25x` FBOs), `DeterministicFrameContext`, data-driven tier overrides, and universal `SignalBus` $\rightarrow$ pass/mask parameter bindings.
> - **Flow 3 (`Chunk R1 Generalized`)** — *Completed*: Verified `FaceRegions` loop table, composable half-res `GL_R8` masks (`$mask:skin`, `$mask:lips`, `$mask:cheeks`, `$mask:eyes`, `$mask:iris`, `luminanceKey`), 1-Euro `ControlPointSmoother`, and fast-motion mask-offset HUD telemetry.
> - **Flow 4 (`Chunk Primitives-1: FM1 + EC1`)** — *Completed*: `faceTexture@1` blend modes (`alpha`, `multiply`, `softLight`, `screen`), dynamic open-mouth cutout, `recolor@1` (`strength <= 0.70`, corneal/glasses glare protection), `320/480/640` stationary iris jitter HUD, and `PerformanceGovernor` `320 <-> 480` scoping.
> - **3D Try-On Packages** — *Completed*: `sunflower_forehead`, `periwinkle_forehead`, `heart_mood_filter`, `furry_glasses_filter`, `round_glasses`, and `multi_prop` validated and integrated with zero engine hardcoding.
>
> **START HERE NEXT $\rightarrow$ Flow 5 (`Chunk Primitives-2: Snap-Level Beauty, Soft Focus, Subtle Face Morph & CF1`)**:
> Upon approval of this plan, we start immediately at **Flow 5**, followed sequentially by **Flow 6**, **Flow 7**, and **Flow 8** (one chunk per turn, with unit/golden test verification at every gate).

---

## 1. Overview & Core Concept

- **What It Does**: Enables any Beauty, Soft-Focus Lens, Subtle Face Reshaping (Liquify/Morph), Makeup/Retouch, Classic Color/LUT Grading, or 3D AR Try-On effect to be authored purely as a declarative `manifest.json` package without touching engine code.
- **How Zero-Lag Snap-Level Beauty Works**:
  1. **Direct GPU OES Camera Stream (`PATH:B`)**: Camera frames stay on the GPU (`$camera`), avoiding CPU bitmap copies. When a package requires camera-pixel beauty/warp/post ops, the engine can auto-select `PATH:B` (while preserving manual HUD override and `PATH:A` startup default).
  2. **Stage 3 Masked Half-Res Skin Smoothing (`post.skinSoften@1` / `post.bilateral@1` + `$mask:skin`)**: Downsamples `$camera` into a `0.5x` `RtPool` FBO, runs a fast edge-preserving surface blur, re-injects high-frequency pore detail (`detailPreserve`), and masks it strictly by `$mask:skin` (`face_oval - eyes - eyebrows - lips`) **before** 3D props are drawn. Eyes, eyelashes, eyebrows, lips, hair, and 3D glasses stay 100% razor-sharp.
  3. **Stage 3 1-Euro Smoothed Landmark Face Morph (`post.warp@1`)**: Evaluates subtle, scale-invariant (`FaceWidth` normalized) radial and directional UV warps (`jawSlim`, `eyeEnlarge`, `noseSlim`, `chinVLine`) driven by 1-Euro smoothed landmarks (`ControlPointSmoother`) so stationary faces never wobble and background lines outside `face_oval` never bend.
  4. **Stage 7 Fused Quarter-Res Soft-Focus & Grading Uber-Pass (`post.softFocus@1` + `colorMatrix@1` + `lut@1` + `vignette@1` + `grain@1`)**: Computes dreamy highlight halation in a `0.25x` `RtPool` FBO and fuses `softFocus`, `colorMatrix`, `3D LUT`, `vignette`, and deterministic `seeded grain` into a single fullscreen composite draw call ($< 0.4\text{ ms}$ GPU cost, zero redundant full-res `glCopyTexSubImage2D` passes).

---

## 2. Unified 9-Stage Filter-Agnostic Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                     CameraX UseCaseGroup & PerceptionProviderRegistry                    │
├────────────────────────────────────────────┬─────────────────────────────────────────────┤
│ Camera Frame Input                         │ Dynamic Perception Providers                │
│ • Path A (Default): PreviewView + GL       │ • Default: 320p (SMALL)                     │
│ • Path B: CameraInputSurface + FrameRing   │ • tracker.iris@1: 480p (configurable        │
│   (delayDepth = 0, sensorTimestampNs,      │   320/480/640, restores 320p on exit;       │
│   single-owner mirror/crop/rot matrix)     │   result age alone never demotes tier)      │
└─────────────────────┬──────────────────────┴──────────────────────┬──────────────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│          Universal SignalBus + BehaviorRunner + DeterministicFrameContext (G0)           │
│ • Evaluates signals (blendshapes, head pose/velocity, tap, timeSec, deterministicSeed)   │
│ • Binds float slots to: Node Transforms │ Material Params │ Mask Weights │ Pass Uniforms │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│             Data-Driven PassGraph Executor & Named Render Target Pool (RtPool)           │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Named RTs: $camera (OES/2D) │ $mask:<id> (Half/Quarter R8) │ $rt:<id> (RGBA8) │ $output  │
│ Data-Driven Tier Policy: tiers[activeTier].disablePasses / passScaleOverrides            │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Canonical 9-Stage Pass Graph (Compiled Automatically & Overridable per Package):         │
│                                                                                          │
│  [1] CAMERA_FRAME        -> Writes $camera / $scene (single mirror/crop/rot)             │
│  [2] REGION_MASK (R1)    -> Evaluates 1-Euro smoothed landmark loops & algebra into      │
│                             half-res R8 targets ($mask:skin, $mask:lips, $mask:iris...)  │
│  [3] CAMERA_PIXEL_PASSES -> Executes on $camera BEFORE 3D Occluders & 3D Props:          │
│      (Before 3D Props!)     • warp@1: 1-Euro smoothed subtle face morph                  │
│                               (jawSlim, eyeEnlarge, noseSlim, chinVLine; zero bg warp)   │
│                             • skinSoften@1 / bilateral@1: Half-res (0.5x RtPool)         │
│                               surface blur masked by $mask:skin + pore detailPreserve    │
│                             • recolor@1: Lip tint ($mask:lips), Cheek blush              │
│                               ($mask:cheeks), Eye color ($mask:iris, glare-protected)    │
│                             • faceTexture@1: Face paint (alpha/multiply/softLight/screen)│
│  [4] CLEAR_DEPTH &       -> Clears depth; draws depth-only FaceMesh + Head Ellipsoid +   │
│      OCCLUDERS (TO1)        Ear/Temple Proxies (colorMask = false)                       │
│  [5] OPAQUE_NODES (TO1)  -> Renders 3D Props (Glasses, Crowns, Flowers, Hats)            │
│                             sharp & unblurred on top of the beautified camera layer      │
│  [6] TRANSPARENT_NODES   -> Renders sorted alpha 3D submeshes / tinted eyewear lenses    │
│  [7] POST_NODES          -> Fused GPU Post Uber-Pass (runs once on composed $scene):     │
│      (CF1 / CF2 / Beauty)   • softFocus@1: Quarter-res (0.25x RtPool) Kawase halation    │
│                             • colorMatrix@1: Brightness/Contrast/Sat/Temp/Tint/B&W/Sepia │
│                             • lut@1: 2D-strip trilinear 3D LUT with strength             │
│                             • vignette@1 + deterministic seeded grain@1 (uGrainSeed)     │
│  [8] OVERLAY_NODES (CF1) -> Screen-space 2D frames/stickers/text (cover/contain,         │
│                             safe-area insets, unmirrored text on front camera)           │
│  [9] FINAL_OUTPUT        -> Presents once to GLSurfaceView or Offscreen Photo FBO        │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Declarative Package Schema (`manifest.json`) for Beauty & Soft Focus

Every Beauty, Soft-Focus, or Face-Morph filter is defined purely through `manifest.json` using generalized engine capabilities (zero filter-specific code in `com.example.engine`):

```json
{
  "schema": "xfilter/3",
  "id": "soft_focus_beauty",
  "name": "Soft Focus & Beauty",
  "version": "1.0.0",
  "requires": {
    "runtime": 3,
    "features": [
      "node.post@1",
      "post.warp@1",
      "post.skinSoften@1",
      "post.softFocus@1",
      "post.colorMatrix@1",
      "post.vignette@1",
      "post.grain@1",
      "pixel.recolor@1"
    ]
  },
  "params": {
    "smoothAmount": { "type": "float", "default": 0.55, "min": 0.0, "max": 1.0 },
    "softGlow":     { "type": "float", "default": 0.35, "min": 0.0, "max": 1.0 },
    "jawSlim":      { "type": "float", "default": 0.22, "min": 0.0, "max": 1.0 },
    "eyeEnlarge":   { "type": "float", "default": 0.18, "min": 0.0, "max": 1.0 }
  },
  "nodes": [
    {
      "id": "subtle_face_morph",
      "type": "post",
      "effect": "warp",
      "params": {
        "jawSlim": "$jawSlim",
        "eyeEnlarge": "$eyeEnlarge",
        "noseSlim": 0.12,
        "chinVLine": 0.15
      },
      "renderOrder": 10
    },
    {
      "id": "masked_skin_smooth",
      "type": "post",
      "effect": "skinSoften",
      "params": {
        "amount": "$smoothAmount",
        "radius": 3.0,
        "detailPreserve": 0.65,
        "warmth": 0.08,
        "brightness": 0.03
      },
      "renderOrder": 20
    },
    {
      "id": "natural_lip_tint",
      "type": "recolor",
      "mask": "$mask:lips",
      "params": {
        "color": [0.86, 0.35, 0.43],
        "strength": 0.22,
        "preserveLuminance": 0.90
      },
      "renderOrder": 30
    },
    {
      "id": "dreamy_soft_focus",
      "type": "post",
      "effect": "softFocus",
      "params": {
        "amount": "$softGlow",
        "radius": 4.0,
        "threshold": 0.55
      },
      "renderOrder": 100
    },
    {
      "id": "warm_grade",
      "type": "post",
      "effect": "colorMatrix",
      "params": {
        "brightness": 0.02,
        "contrast": 1.04,
        "saturation": 1.05,
        "temperature": 0.10
      },
      "renderOrder": 101
    }
  ]
}
```

---

## 4. Complete Step-by-Step Chunk Roadmap & Verification Gates

### **Flow 1 — Chunk B0: Path B Camera Input, `FrameRing` (Depth 0), Pass Order & Photo Single-Pass Parity** *(COMPLETED)*
- **Status**: Done & verified (`ChunkB0PathBAndPassOrderUnitTest`).

### **Flow 2 — Chunk G0: Data-Driven `PassGraph` Executor, `RtPool`, Universal Signal Binding & Data-Driven Tiers** *(COMPLETED)*
- **Status**: Done & verified (`ChunkG0PassGraphRtPoolAndBindingUnitTest`).

### **Flow 3 — Chunk R1 (Generalized): Data-Table `FaceRegions`, Composable Mask Algebra (`$mask:skin`, `$mask:lips`, `$mask:iris`) & Fast-Motion Mask-Offset HUD** *(COMPLETED)*
- **Status**: Done & verified (`ChunkR1RegionMaskAndTelemetryUnitTest`).

### **Flow 4 — Chunk Primitives-1 (`FM1 + EC1`): `faceTexture@1` Blend Modes, `recolor@1`, Specular Glare Protection & `320/480/640` Iris Jitter HUD** *(COMPLETED)*
- **Status**: Done & verified (`ChunkPrimitives1Fm1Ec1UnitTest`).

---

### **👉 START HERE: Flow 5 — Chunk Primitives-2 (`CF1 + Snap-Level Beauty, Soft Focus & Subtle Face Morph`)**
1. **Masked Half-Res Skin Smoothing (`post.skinSoften@1` / `post.bilateral@1` + `$mask:skin`)**:
   - Upgrade `skinSoften` / `bilateral` to read `$mask:skin` (`face_oval - eyes - eyebrows - lips`) from `RegionMaskPass` and use `RtPool` `HALF` (`0.5x`) FBO downsampling + high-frequency pore `detailPreserve` (`sharp - smooth` re-injection).
   - Guarantee that eyes, eyelashes, eyebrows, lips, and 3D props remain 100% unblurred while skin blemishes/pores are smoothed.
2. **Quarter-Res Dreamy Soft-Focus Lens (`post.softFocus@1` / `post.bloom@1`)**:
   - Use `RtPool` `QUARTER` (`0.25x`) FBO to compute a fast 4-tap Dual-Kawase blur with luminance highlight weighting (`threshold`) and blend via `screen`/`softLight` halation over the sharp base frame.
3. **1-Euro Smoothed Landmark Subtle Face Morph (`post.warp@1`)**:
   - Implement `warp` (`post.warp@1`) driven by 1-Euro smoothed landmarks (`ControlPointSmoother`) and scaled by `faceWidth` (`FW`):
     - `jawSlim` (`0.0..1.0`: pulls lower cheek/jaw contour `#234/#454/#172/#397` inward toward nasal midline, clamped `<= 6% FW`)
     - `eyeEnlarge` (`0.0..1.0`: smooth radial magnification around left/right eye centers `#468/#473`, clamped `<= 8% FW`)
     - `noseSlim` (`0.0..1.0`: subtle horizontal compression across alar wings `#129/#358`)
     - `chinVLine` (`-1.0..1.0`: subtle vertical/inward V-taper at `#152`)
   - Smooth radial falloff reaches `0.0` inside `face_oval` so background geometry never distorts.
4. **Classic Camera Filters Part 1 (`post.colorMatrix@1`, `post.vignette@1`, `node.overlay@1`)**:
   - Implement `colorMatrix` (brightness, contrast, saturation, hue, temperature, tint, B&W, sepia) and `vignette` (radius, softness, intensity).
   - Implement `node.overlay@1` in Stage 8 (`OVERLAY_NODES`, after post-processing) with `fit: cover | contain`, safe-area insets, and single-mirror protection so text is never reversed on the front camera.
- **Flow 5 Gate & Tests (`ChunkPrimitives2Cf1BeautyUnitTest`)**:
  - Unit & golden tests verifying:
    1. `skinSoften` with `$mask:skin` smooths interior cheek skin while preserving 100% sharpness on eye/lip/background pixels and preserving micro-contrast via `detailPreserve`.
    2. `softFocus` quarter-res halation boosts soft highlight glow without blurring dark edge contrast.
    3. `warp` (`jawSlim`, `eyeEnlarge`, `noseSlim`, `chinVLine`) displaces target facial landmarks smoothly, exhibits zero jitter on stationary 1-Euro landmarks, and produces 0.0px displacement on background pixels outside `face_oval`.
    4. `colorMatrix`, `vignette`, and unmirrored `overlay@1` placement across aspect ratios.

---

### **Flow 6 — Chunk Primitives-3 (`CF2 + Fused Post Uber-Pass, 3D LUT, Seeded Grain & Bundled Soft-Focus Beauty Package`)**
1. **2D-Strip 3D `lut` (`post.lut@1`) & Deterministic Seeded `grain` (`post.grain@1`)**:
   - Add trilinear 3D LUT sampling from a 2D strip texture with bindable `strength`.
   - Add deterministic seeded film grain (`uGrainSeed` from `DeterministicFrameContext`) so live preview and captured still photos match identically.
2. **Single-Draw Fused Post Uber-Pass & Tier C Auto-Degradation**:
   - Fuse active Stage 7 ops (`softFocus` + `colorMatrix` + `lut` + `vignette` + `grain`) so they execute without redundant full-resolution `glCopyTexSubImage2D` passes.
   - Automatically disable heavy passes (`grain`, `softFocus`, `warp`) on Tier C via data-driven tier rules while preserving `lut` and `colorMatrix`.
3. **Upgrade `ref_beauty` & Bundle `soft_focus_beauty` Package**:
   - Upgrade `ref_beauty` and add a complete Snapchat-style `soft_focus_beauty` package combining subtle `warp`, `$mask:skin` `skinSoften`, `$mask:lips` `recolor`, `softFocus`, and warm `colorMatrix`/`lut`.
- **Flow 6 Gate & Tests (`ChunkPrimitives3Cf2FusedBeautyUnitTest`)**:
  - Golden unit tests verifying 3D LUT trilinear interpolation, deterministic seeded grain parity between live preview and offscreen photo passes, single-pass fused execution count, Tier C degradation, and end-to-end compilation/rendering of `soft_focus_beauty` with zero engine hardcoding.

---

### **Flow 7 — Chunk G1 + TO1: Unified `AnchorSource` / `FitPolicy` Interface & Generalized Geometry Sources (`instancer@1`, `path@1`)**
1. **Unified `AnchorSource` & `FitPolicy` Interface**:
   - Consolidate all anchor resolution behind `AnchorSource` (`RigidPoseAnchor`, `LandmarkWeightedAnchor`, `EarProxyAnchor`, `IrisCenterAnchor`, `CrownAnchor`) and `FitPolicy` (`fit.to = "fw" | "ipd" | "eye_width" | "mouth_width"`, `ratio`).
2. **Generalized Geometry Sources (`node.instancer@1` & `node.path@1`)**:
   - `node.instancer@1`: Multi-anchor / procedural mesh or sprite instancing (e.g., bilateral earrings, star/freckle clusters).
   - `node.path@1`: Dynamic triangle-strip ribbon along any named landmark chain from `FaceRegions` (e.g., procedural eyeliner, brow stroke, neon contour).
- **Flow 7 Gate & Tests**:
  - Unit tests for `AnchorSource` & `FitPolicy` scaling invariance, `instancer@1` multi-anchor placement, and `path@1` ribbon generation along landmark chains.

---

### **Flow 8 — Chunk V1: Registry-Generated Schema/Validator & Primitive-Wise Golden Conformance Suite**
1. **Single-Source `TemplateAndCapabilityRegistry` & Auto-Derived Schema Validator**:
   - Register all node types, pass templates, mask ops, beauty/post ops (`post.skinSoften@1`, `post.softFocus@1`, `post.warp@1`, `post.colorMatrix@1`, `post.lut@1`, `post.vignette@1`, `post.grain@1`, `node.overlay@1`, `node.instancer@1`, `node.path@1`), and capabilities in `TemplateAndCapabilityRegistry`.
2. **Complete Primitive-Wise & Reference Package Conformance Suite**:
   - End-to-end golden suite verifying every primitive in isolation plus all bundled packages (`sunflower_forehead`, `periwinkle_forehead`, `heart_mood_filter`, `furry_glasses_filter`, `soft_focus_beauty`, `ref_beauty`, `ref_face_paint`, `round_glasses`, `multi_prop`) and `tools/check_hardcoding.py`.
