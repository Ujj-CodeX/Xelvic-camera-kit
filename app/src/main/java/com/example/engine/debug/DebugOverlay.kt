package com.example.engine.debug

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import com.example.engine.capture.CameraSession
import com.example.engine.capture.FrameClock
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.AnalysisResolutionMode
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.pkg.FaceRegions
import com.example.engine.render.RegionMaskPass
import com.example.engine.scene.AnchorResolver
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import java.util.Locale
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Diagnostic visualizer implementing all README Section 7, 10, 20 & 21 alignment requirements:
 * 1. 3D Canonical Axes Gizmo at `head_center` projected through `FacePoseAdapter.createCameraProjection`
 *    (using the visible-view FOV after `FILL_CENTER`) and the 4x4 `faceState.poseMatrix`
 *    (replacing the legacy 2D trig approximation):
 *    - X axis: RED, pointing subject-right (+X).
 *    - Y axis: GREEN, pointing toward top of head (+Y).
 *    - Z axis: BLUE, pointing toward camera (+Z).
 * 2. Anatomical Left/Right side mapping confirmation.
 * 3. 3D Head-Proxy Ellipsoid Wireframe (radii `[0.56, 0.66, 0.68]` in FW) projected through
 *    `FacePoseAdapter.createCameraProjection` and `faceState.poseMatrix`.
 * 4. Pinned cheek measurement points (landmarks 234 and 454) and built-in anchor verification
 *    markers (forehead 10, nose 1, chin 152, eyes 33 & 263) mapped via `FacePoseAdapter.mapLandmarkToView`
 *    using the per-frame `imageToViewTransform`.
 * 5. Camera intrinsics & FOV telemetry (`FOV source: intrinsics | fallback`, `LENS_FOCAL_LENGTH`,
 *    `SCALER_CROP_REGION`) refreshed at ~2 Hz with zero per-frame allocations.
 */
class DebugOverlay @JvmOverloads constructor(
    var latencyProbe: LatencyProbe = LatencyProbe.shared,
    var governor: PerformanceGovernor? = null,
    var faceProvider: MediaPipeFaceGeometryProvider? = null,
    var cameraSession: CameraSession? = null
) {
    companion object {
        const val GIZMO_PLUS_X_AXIS_LABEL = "+X (Subject-R)"
        private const val GIZMO_AXIS_LEN_FW = 0.45f
        private const val HEAD_PROXY_RX_FW = 0.56f
        private const val HEAD_PROXY_RY_FW = 0.66f
        private const val HEAD_PROXY_RZ_FW = 0.68f
        private const val PROXY_RING_SEGMENTS = 16

        // Precomputed unit-circle tables for zero-allocation 3D head-proxy ring projection
        private val COS_TABLE = FloatArray(PROXY_RING_SEGMENTS + 1) { i ->
            cos((2.0 * PI * i) / PROXY_RING_SEGMENTS).toFloat()
        }
        private val SIN_TABLE = FloatArray(PROXY_RING_SEGMENTS + 1) { i ->
            sin((2.0 * PI * i) / PROXY_RING_SEGMENTS).toFloat()
        }
    }

    var isEnabled: Boolean = false

    /**
     * Controls whether resolved 3D anchor markers (`head_center`, `forehead`, and a custom landmark
     * anchor) are drawn on the debug overlay canvas. Defaults to `false`, toggleable from the debug HUD.
     */
    var showResolvedAnchorMarkers: Boolean = false

    /**
     * Controls whether `FaceRegions` contour loops (`FACE_OVAL`, eyes, lips, inner mouth)
     * are drawn on the debug overlay canvas for region mask verification.
     */
    var showRegionMaskWireframe: Boolean = false

    /**
     * Active generic mask viewer mode (`MASK:OFF/EYES/MOUTH/IRIS/SKIN`) on the Debug HUD & Canvas (Flow 3 — Chunk R1).
     */
    var debugMaskMode: RegionMaskPass.DebugMaskMode = RegionMaskPass.DebugMaskMode.OFF

    /**
     * Optional inspected render target name (e.g. `"$mask:skin"`, `"$rt:blurTemp"`) for HUD display.
     */
    var inspectedRtName: String? = null

    /**
     * Active [AnchorResolver] used to resolve `head_center`, `forehead`, and custom landmark anchors.
     */
    var anchorResolver: AnchorResolver = AnchorResolver()

    private val fallbackCustomLandmarkResolver = AnchorResolver(
        mapOf(
            "custom_lm" to AnchorResolver.CustomAnchorDef(
                landmarks = listOf(
                    AnchorResolver.LandmarkWeight(70, 0.5f),
                    AnchorResolver.LandmarkWeight(105, 0.5f)
                ),
                offsetFW = floatArrayOf(0f, 0.05f, 0.02f)
            )
        )
    )

    /**
     * When `false` (default), `draw()` renders only the 3D spatial axes gizmo, head-proxy
     * ellipsoid wireframe, and landmark verification markers on the Canvas while the text HUD card
     * is rendered by Compose below the top bar and debug toggle buttons (preventing UI overlap).
     */
    var drawHudCardOnCanvas: Boolean = false

    // Pre-allocated projection buffers for zero per-frame heap allocation in draw()
    private val cardRect = RectF()
    private val viewProjMatrix = FloatArray(16)
    private val scratchCam = FloatArray(3)
    private val gizmoOriginPx = FloatArray(2)
    private val gizmoXPx = FloatArray(2)
    private val gizmoYPx = FloatArray(2)
    private val gizmoZPx = FloatArray(2)
    private val ringPrevPx = FloatArray(2)
    private val ringCurrPx = FloatArray(2)
    private val landmark234Px = FloatArray(2)
    private val landmark454Px = FloatArray(2)
    private val landmarkPtPx = FloatArray(2)

    // Pre-allocated resolved anchor FW & pixel buffers (zero per-frame heap allocation)
    val lastResolvedHeadCenterFW = FloatArray(3)
    val lastResolvedForeheadFW = FloatArray(3)
    val lastResolvedCustomLandmarkFW = FloatArray(3)
    val lastProjectedHeadCenterAnchorPx = FloatArray(2)
    val lastProjectedForeheadAnchorPx = FloatArray(2)
    val lastProjectedCustomLandmarkAnchorPx = FloatArray(2)
    var lastResolvedCustomLandmarkAnchorName: String = "custom_lm"
        private set

    // Exposed projection telemetry for unit-test verification
    var lastUsedFovYCropDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
        private set
    var lastUsedFovYViewDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
        private set
    val lastProjectedGizmoOriginPx = FloatArray(2)
    val lastProjectedGizmoXPx = FloatArray(2)
    val lastProjectedGizmoYPx = FloatArray(2)
    val lastProjectedGizmoZPx = FloatArray(2)
    val lastProjectedHeadTopPx = FloatArray(2)
    val lastProjectedTempleRightPx = FloatArray(2)
    val lastProjectedTempleLeftPx = FloatArray(2)

    private val paintRed = Paint().apply {
        color = Color.rgb(239, 68, 68) // #EF4444 Red
        strokeWidth = 6f
        isAntiAlias = true
        style = Paint.Style.STROKE
    }

    private val paintGreen = Paint().apply {
        color = Color.rgb(34, 197, 94) // #22C55E Green
        strokeWidth = 6f
        isAntiAlias = true
        style = Paint.Style.STROKE
    }

    private val paintBlue = Paint().apply {
        color = Color.rgb(56, 189, 248) // #38BDF8 Cyan/Blue
        strokeWidth = 6f
        isAntiAlias = true
        style = Paint.Style.STROKE
    }

    private val paintWireframe = Paint().apply {
        color = Color.argb(160, 56, 189, 248)
        strokeWidth = 2.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
        pathEffect = DashPathEffect(floatArrayOf(12f, 8f), 0f)
    }

    private val paintFwLine = Paint().apply {
        color = Color.rgb(251, 191, 36) // Yellow/Amber
        strokeWidth = 3f
        isAntiAlias = true
        style = Paint.Style.STROKE
    }

    private val paintLandmark = Paint().apply {
        color = Color.rgb(251, 191, 36)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val paintAnchor = Paint().apply {
        color = Color.rgb(168, 85, 247) // Purple
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val paintResolvedAnchor = Paint().apply {
        color = Color.rgb(244, 114, 182) // Pink/Magenta for resolved 3D anchor markers
        strokeWidth = 3.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val paintRegionMask = Paint().apply {
        color = Color.argb(190, 52, 211, 153) // Emerald for region contour loops
        strokeWidth = 2.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val paintHudBg = Paint().apply {
        color = Color.argb(200, 15, 23, 42) // Dark Slate
        style = Paint.Style.FILL
    }

    private val paintHudBorder = Paint().apply {
        color = Color.argb(80, 255, 255, 255)
        strokeWidth = 1.5f
        style = Paint.Style.STROKE
    }

    private val paintText = Paint().apply {
        color = Color.WHITE
        textSize = 25f
        isAntiAlias = true
    }

    private val paintProbeText = Paint().apply {
        color = Color.rgb(251, 191, 36) // Amber highlight for LatencyProbe line
        textSize = 23f
        isFakeBoldText = true
        isAntiAlias = true
    }

    private val paintTextBold = Paint().apply {
        color = Color.rgb(56, 189, 248)
        textSize = 28f
        isFakeBoldText = true
        isAntiAlias = true
    }

    private val paintGaugeBg = Paint().apply {
        color = Color.argb(100, 255, 255, 255)
        style = Paint.Style.FILL
    }

    private val paintGaugeFill = Paint().apply {
        color = Color.rgb(34, 197, 94)
        style = Paint.Style.FILL
    }

    // Cached HUD strings refreshed at ~2 Hz (every 500 ms) to avoid per-frame String.format
    private var lastHudTextRefreshNs: Long = 0L

    @Volatile
    var hudTextRevision: Int = 0
        private set

    var cachedLatencyLine: String =
        "age p50/p95: --/--ms | trk: 0.0fps | cam: 0.0fps | rnd: 0.0fps | wait: 0.0ms | inf: 0.0ms | gc: +0"
        private set

    var cachedStageLine: String = "Stages: conv 0.0ms • wait 0.0ms • inf 0.0ms • q 0.0ms • pred:ON (0.0mm/0.0°)"
        private set

    var cachedConvBreakdownLine: String =
        "Conv[FAST]: tot 0.0ms (unpk 0.0 + yuv 0.0 + px 0.0 + rot 0.0 + mp 0.0)"
        private set

    var cachedPumpAndIdleLine: String =
        "Pump: 0.0/0.0ms (mean/max) • IdleGap: pend 0.0ms / noPend 0.0ms"
        private set

    var cachedBlendAndPrioLine: String =
        "Blend: ON • inf(ON/OFF): 0.0/0.0ms • Prio A:NORM R:NORM"
        private set

    var cachedGovWarmupLine: String =
        "Therm: NONE(0) hdrm:n/a • Gov: 30Hz@320px (none) • Warm-up: steady"
        private set

    var cachedCountersLine: String =
        "Counters: drop=0 repl=0 wdog=0 • ovlp=0 cadDrop=0/0 • PATH:A • cam texture: missing • preview: 0.0fps"
        private set

    var cachedMaskOffsetLine: String =
        "MaskOffset: dt=0.0ms (fast: 0.0ms) • shift=0.0px (fast p95/max: 0.0/0.0px) • motion: STILL"
        private set

    var cachedMaskViewerLine: String =
        "MaskView: MASK:OFF • target: none • mouthOpen: 0.00"
        private set

    var cachedIrisJitterLine: String =
        "IrisJitter[480px]: 320:--px • 480:--px • 640:--px"
        private set

    var cachedCameraLine: String = "Camera: FRONT (Mirrored) • PATH:A • cam texture: missing • Transform: Fallback FILL_CENTER • Tier A"
        private set

    var cachedFovLine: String = "Analysis: SMALL • YUV_420_888 • CPU • 640x360"
        private set

    var cachedProjectionHudLine: String =
        "FOV: 60.0° (view 60.0°) • src: fallback • f=0.00mm • crop=none • depth=24b"
        private set

    var cachedLifecycleLine: String = "Lifecycle: NO_FACE • Fade: 0.00 • 30.0 FPS"
        private set

    var cachedPoseLine: String = "Pose (Deg): Y:0.0 P:0.0 R:0.0"
        private set

    var cachedFwLine: String = "FW: 0.140m (140mm)"
        private set

    var cachedFwMidLabel: String = "FW: 140mm (#234-#454)"
        private set

    var cachedJawOpen: Float = 0f
        private set

    var cachedSmile: Float = 0f
        private set

    var cachedJawOpenPercentText: String = "0%"
        private set

    var cachedSmilePercentText: String = "0%"
        private set

    /**
     * Computes the visible-view FOV, 3D camera projection matrix, 3D-projected axes gizmo,
     * 3D-projected head-proxy clearance markers, and ~2 Hz cached HUD strings for [faceState]
     * at `[viewWidth, viewHeight]` with zero heap allocations between HUD refreshes.
     */
    @JvmOverloads
    fun projectOverlayGeometry(
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float,
        fps: Float = 30f,
        tier: String = "A"
    ): Boolean {
        val safeW = viewWidth.coerceAtLeast(1f)
        val safeH = viewHeight.coerceAtLeast(1f)
        val viewAspect = (safeW / safeH).coerceAtLeast(0.01f)
        val fovYCropDeg = faceState.projectionParams.fovYDeg
        val fovYViewDeg = faceState.projectionParams.visibleViewFovYDeg(viewAspect)
        lastUsedFovYCropDeg = fovYCropDeg
        lastUsedFovYViewDeg = fovYViewDeg

        refreshCachedHudStringsIfNeeded(faceState, fps, tier, fovYCropDeg, fovYViewDeg)

        if (!faceState.isVisible) return false

        // Build 3D camera perspective projection matrix using VISIBLE-VIEW FOV after FILL_CENTER
        FacePoseAdapter.createCameraProjection(
            fovYDeg = fovYViewDeg,
            aspectRatio = viewAspect,
            near = faceState.projectionParams.near,
            far = faceState.projectionParams.far,
            frontFacing = faceState.isFrontCamera,
            out = viewProjMatrix
        )

        // Resolve head_center (package override or built-in default [0, 0.2, -0.12])
        anchorResolver.resolveAnchor("head_center", faceState, lastResolvedHeadCenterFW)
        val hx = lastResolvedHeadCenterFW[0]
        val hy = lastResolvedHeadCenterFW[1]
        val hz = lastResolvedHeadCenterFW[2]

        // Project 3D Axes Gizmo at head_center through 4x4 poseMatrix and viewProjMatrix
        projectLocalToPixel(hx, hy, hz, faceState, safeW, safeH, gizmoOriginPx)
        projectLocalToPixel(hx + GIZMO_AXIS_LEN_FW, hy, hz, faceState, safeW, safeH, gizmoXPx)
        projectLocalToPixel(hx, hy + GIZMO_AXIS_LEN_FW, hz, faceState, safeW, safeH, gizmoYPx)
        projectLocalToPixel(hx, hy, hz + GIZMO_AXIS_LEN_FW, faceState, safeW, safeH, gizmoZPx)

        lastProjectedGizmoOriginPx[0] = gizmoOriginPx[0]; lastProjectedGizmoOriginPx[1] = gizmoOriginPx[1]
        lastProjectedGizmoXPx[0] = gizmoXPx[0]; lastProjectedGizmoXPx[1] = gizmoXPx[1]
        lastProjectedGizmoYPx[0] = gizmoYPx[0]; lastProjectedGizmoYPx[1] = gizmoYPx[1]
        lastProjectedGizmoZPx[0] = gizmoZPx[0]; lastProjectedGizmoZPx[1] = gizmoZPx[1]

        // Project 3D Head-Proxy Temple width & Head Top clearance markers centered at head_center
        projectLocalToPixel(hx + HEAD_PROXY_RX_FW, hy, hz, faceState, safeW, safeH, lastProjectedTempleRightPx)
        projectLocalToPixel(hx - HEAD_PROXY_RX_FW, hy, hz, faceState, safeW, safeH, lastProjectedTempleLeftPx)
        projectLocalToPixel(hx, hy + HEAD_PROXY_RY_FW, hz, faceState, safeW, safeH, lastProjectedHeadTopPx)

        // Project resolved anchors (head_center, forehead, and custom landmark anchor) when enabled
        lastProjectedHeadCenterAnchorPx[0] = gizmoOriginPx[0]
        lastProjectedHeadCenterAnchorPx[1] = gizmoOriginPx[1]

        if (showResolvedAnchorMarkers) {
            anchorResolver.resolveAnchor("forehead", faceState, lastResolvedForeheadFW)
            projectLocalToPixel(
                lastResolvedForeheadFW[0],
                lastResolvedForeheadFW[1],
                lastResolvedForeheadFW[2],
                faceState,
                safeW,
                safeH,
                lastProjectedForeheadAnchorPx
            )

            var customName: String? = null
            for ((k, v) in anchorResolver.customAnchors) {
                if (v.landmarks.isNotEmpty()) {
                    customName = k
                    break
                }
            }
            if (customName != null) {
                lastResolvedCustomLandmarkAnchorName = customName
                anchorResolver.resolveAnchor(customName, faceState, lastResolvedCustomLandmarkFW)
            } else {
                lastResolvedCustomLandmarkAnchorName = "custom_lm"
                fallbackCustomLandmarkResolver.resolveAnchor("custom_lm", faceState, lastResolvedCustomLandmarkFW)
            }
            projectLocalToPixel(
                lastResolvedCustomLandmarkFW[0],
                lastResolvedCustomLandmarkFW[1],
                lastResolvedCustomLandmarkFW[2],
                faceState,
                safeW,
                safeH,
                lastProjectedCustomLandmarkAnchorPx
            )
        }
        return true
    }

    fun draw(
        canvas: Canvas,
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float,
        fps: Float = 30f,
        tier: String = "A"
    ) {
        if (!isEnabled) return

        val safeW = viewWidth.coerceAtLeast(1f)
        val safeH = viewHeight.coerceAtLeast(1f)
        val hasVisibleGeometry = projectOverlayGeometry(faceState, safeW, safeH, fps, tier)

        // 1. Optional legacy Canvas HUD card (disabled by default; Compose renders the non-overlapping HUD card)
        if (drawHudCardOnCanvas) {
            drawTelemetryCard(canvas, faceState, fps, tier, lastUsedFovYCropDeg, lastUsedFovYViewDeg)
        }

        if (!hasVisibleGeometry) return

        val ox = gizmoOriginPx[0]
        val oy = gizmoOriginPx[1]

        // +Z Axis (Toward Camera): Blue
        canvas.drawLine(ox, oy, gizmoZPx[0], gizmoZPx[1], paintBlue)
        canvas.drawCircle(gizmoZPx[0], gizmoZPx[1], 6f, paintBlue)
        canvas.drawText("+Z (Camera)", gizmoZPx[0] + 8f, gizmoZPx[1] - 8f, paintBlue)

        // +X Axis (Subject-Right): Red
        canvas.drawLine(ox, oy, gizmoXPx[0], gizmoXPx[1], paintRed)
        canvas.drawCircle(gizmoXPx[0], gizmoXPx[1], 6f, paintRed)
        canvas.drawText(GIZMO_PLUS_X_AXIS_LABEL, gizmoXPx[0] - 60f, gizmoXPx[1] - 12f, paintRed)

        // +Y Axis (Top of Head / Up): Green
        canvas.drawLine(ox, oy, gizmoYPx[0], gizmoYPx[1], paintGreen)
        canvas.drawCircle(gizmoYPx[0], gizmoYPx[1], 6f, paintGreen)
        canvas.drawText("+Y (Up)", gizmoYPx[0] + 8f, gizmoYPx[1] - 12f, paintGreen)

        // Origin Dot
        canvas.drawCircle(ox, oy, 8f, paintLandmark)

        // 4. Project 3D Head-Proxy Ellipsoid Wireframe (radii [0.56, 0.66, 0.68] in FW) through poseMatrix & viewProjMatrix
        // 4a. Coronal ring (XY plane at z = 0)
        drawProjectedEllipsoidRing(canvas, faceState, safeW, safeH, planeMode = 0)
        // 4b. Transverse ring (XZ plane at y = 0)
        drawProjectedEllipsoidRing(canvas, faceState, safeW, safeH, planeMode = 1)
        // 4c. Sagittal ring (YZ plane at x = 0)
        drawProjectedEllipsoidRing(canvas, faceState, safeW, safeH, planeMode = 2)

        canvas.drawCircle(lastProjectedTempleRightPx[0], lastProjectedTempleRightPx[1], 5f, paintGreen)
        canvas.drawCircle(lastProjectedTempleLeftPx[0], lastProjectedTempleLeftPx[1], 5f, paintGreen)
        canvas.drawText("Temple", lastProjectedTempleRightPx[0] + 8f, lastProjectedTempleRightPx[1] + 10f, paintText)

        canvas.drawLine(
            lastProjectedHeadTopPx[0] - 20f,
            lastProjectedHeadTopPx[1],
            lastProjectedHeadTopPx[0] + 20f,
            lastProjectedHeadTopPx[1],
            paintGreen
        )
        canvas.drawText("Head Top", lastProjectedHeadTopPx[0] - 40f, lastProjectedHeadTopPx[1] - 12f, paintText)

        // 5. Draw Landmark Alignment Verification & FW Line mapped via per-frame imageToViewTransform
        val lm = faceState.landmarks
        if (lm != null && lm.size >= 455 * 3) {
            val i234 = FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3
            val i454 = FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3

            mapLandmarkToPixel(lm[i234], lm[i234 + 1], faceState, safeW, safeH, landmark234Px)
            mapLandmarkToPixel(lm[i454], lm[i454 + 1], faceState, safeW, safeH, landmark454Px)

            val x234 = landmark234Px[0]
            val y234 = landmark234Px[1]
            val x454 = landmark454Px[0]
            val y454 = landmark454Px[1]

            // Draw line between 234 and 454 (FW measurement baseline)
            canvas.drawLine(x234, y234, x454, y454, paintFwLine)
            val midX = (x234 + x454) * 0.5f
            val midY = (y234 + y454) * 0.5f
            canvas.drawText(cachedFwMidLabel, midX - 80f, midY - 14f, paintTextBold)

            // Cheek 234 (Anatomical Right)
            canvas.drawCircle(x234, y234, 7f, paintLandmark)
            canvas.drawText("#234 R", x234 + 10f, y234 - 10f, paintText)

            // Cheek 454 (Anatomical Left)
            canvas.drawCircle(x454, y454, 7f, paintLandmark)
            canvas.drawText("#454 L", x454 + 10f, y454 - 10f, paintText)

            // Forehead #10
            val i10 = FacePoseAdapter.LANDMARK_FOREHEAD * 3
            mapLandmarkToPixel(lm[i10], lm[i10 + 1], faceState, safeW, safeH, landmarkPtPx)
            canvas.drawCircle(landmarkPtPx[0], landmarkPtPx[1], 6f, paintAnchor)
            canvas.drawText("Forehead #10", landmarkPtPx[0] + 10f, landmarkPtPx[1], paintText)

            // Nose Tip #1
            val i1 = FacePoseAdapter.LANDMARK_NOSE_TIP * 3
            mapLandmarkToPixel(lm[i1], lm[i1 + 1], faceState, safeW, safeH, landmarkPtPx)
            canvas.drawCircle(landmarkPtPx[0], landmarkPtPx[1], 6f, paintAnchor)
            canvas.drawText("Nose #1", landmarkPtPx[0] + 10f, landmarkPtPx[1], paintText)

            // Chin #152
            val i152 = FacePoseAdapter.LANDMARK_CHIN * 3
            mapLandmarkToPixel(lm[i152], lm[i152 + 1], faceState, safeW, safeH, landmarkPtPx)
            canvas.drawCircle(landmarkPtPx[0], landmarkPtPx[1], 6f, paintAnchor)
            canvas.drawText("Chin #152", landmarkPtPx[0] + 10f, landmarkPtPx[1] + 10f, paintText)

            // Eyes #33 (R) & #263 (L)
            val i33 = FacePoseAdapter.LANDMARK_EYE_RIGHT * 3
            mapLandmarkToPixel(lm[i33], lm[i33 + 1], faceState, safeW, safeH, landmarkPtPx)
            canvas.drawCircle(landmarkPtPx[0], landmarkPtPx[1], 5f, paintAnchor)

            val i263 = FacePoseAdapter.LANDMARK_EYE_LEFT * 3
            mapLandmarkToPixel(lm[i263], lm[i263 + 1], faceState, safeW, safeH, landmarkPtPx)
            canvas.drawCircle(landmarkPtPx[0], landmarkPtPx[1], 5f, paintAnchor)
        }

        // 6. Optional resolved anchor markers (head_center, forehead, custom landmark anchor)
        if (showResolvedAnchorMarkers) {
            drawResolvedAnchorMarker(
                canvas = canvas,
                px = lastProjectedHeadCenterAnchorPx[0],
                py = lastProjectedHeadCenterAnchorPx[1],
                label = "ANC:head_center"
            )
            drawResolvedAnchorMarker(
                canvas = canvas,
                px = lastProjectedForeheadAnchorPx[0],
                py = lastProjectedForeheadAnchorPx[1],
                label = "ANC:forehead"
            )
            drawResolvedAnchorMarker(
                canvas = canvas,
                px = lastProjectedCustomLandmarkAnchorPx[0],
                py = lastProjectedCustomLandmarkAnchorPx[1],
                label = "ANC:$lastResolvedCustomLandmarkAnchorName"
            )
        }

        // 7. Optional FaceRegions contour loops for Chunk R1 region-mask visual verification
        if ((showRegionMaskWireframe || debugMaskMode != RegionMaskPass.DebugMaskMode.OFF) && lm != null && lm.size >= 455 * 3) {
            when (debugMaskMode) {
                RegionMaskPass.DebugMaskMode.EYES -> {
                    drawRegionContourLoop(canvas, lm, FaceRegions.RIGHT_EYE_CONTOUR, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.LEFT_EYE_CONTOUR, faceState, safeW, safeH)
                }
                RegionMaskPass.DebugMaskMode.MOUTH -> {
                    drawRegionContourLoop(canvas, lm, FaceRegions.LIPS_OUTER, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.MOUTH_INNER, faceState, safeW, safeH)
                }
                RegionMaskPass.DebugMaskMode.IRIS -> {
                    drawRegionContourLoop(canvas, lm, FaceRegions.RIGHT_EYE_CONTOUR, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.LEFT_EYE_CONTOUR, faceState, safeW, safeH)
                    if (lm.size >= 478 * 3) {
                        drawRegionContourLoop(canvas, lm, FaceRegions.RIGHT_IRIS_RING, faceState, safeW, safeH)
                        drawRegionContourLoop(canvas, lm, FaceRegions.LEFT_IRIS_RING, faceState, safeW, safeH)
                    }
                }
                RegionMaskPass.DebugMaskMode.SKIN, RegionMaskPass.DebugMaskMode.OFF -> {
                    drawRegionContourLoop(canvas, lm, FaceRegions.FACE_OVAL, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.RIGHT_EYE_CONTOUR, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.LEFT_EYE_CONTOUR, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.LIPS_OUTER, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.MOUTH_INNER, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.RIGHT_EYEBROW, faceState, safeW, safeH)
                    drawRegionContourLoop(canvas, lm, FaceRegions.LEFT_EYEBROW, faceState, safeW, safeH)
                }
            }
        }
    }

    private fun drawRegionContourLoop(
        canvas: Canvas,
        landmarks: FloatArray,
        loopIndices: IntArray,
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float
    ) {
        val count = loopIndices.size
        if (count < 2) return
        val isFaceOval = RegionMaskPass.isFaceOvalLoop(loopIndices)
        var cranialDxPx = 0f
        var cranialDyPx = 0f
        if (isFaceOval && landmarks.size >= 153 * 3) {
            mapLandmarkToPixel(landmarks[10 * 3], landmarks[10 * 3 + 1], faceState, viewWidth, viewHeight, ringPrevPx)
            val fPxX = ringPrevPx[0]
            val fPxY = ringPrevPx[1]
            mapLandmarkToPixel(landmarks[152 * 3], landmarks[152 * 3 + 1], faceState, viewWidth, viewHeight, ringCurrPx)
            cranialDxPx = fPxX - ringCurrPx[0]
            cranialDyPx = fPxY - ringCurrPx[1]
        }
        for (i in 0 until count) {
            val lmA = loopIndices[i]
            val lmB = loopIndices[(i + 1) % count]
            val idxA = lmA * 3
            val idxB = lmB * 3
            if (idxA + 1 >= landmarks.size || idxB + 1 >= landmarks.size) continue
            mapLandmarkToPixel(landmarks[idxA], landmarks[idxA + 1], faceState, viewWidth, viewHeight, ringPrevPx)
            mapLandmarkToPixel(landmarks[idxB], landmarks[idxB + 1], faceState, viewWidth, viewHeight, ringCurrPx)
            val wA = if (isFaceOval) RegionMaskPass.cranialForeheadWeightForLandmark(lmA) else 0f
            val wB = if (isFaceOval) RegionMaskPass.cranialForeheadWeightForLandmark(lmB) else 0f
            val x0 = ringPrevPx[0] + cranialDxPx * wA
            val y0 = ringPrevPx[1] + cranialDyPx * wA
            val x1 = ringCurrPx[0] + cranialDxPx * wB
            val y1 = ringCurrPx[1] + cranialDyPx * wB
            canvas.drawLine(x0, y0, x1, y1, paintRegionMask)
        }
    }

    private fun drawResolvedAnchorMarker(
        canvas: Canvas,
        px: Float,
        py: Float,
        label: String
    ) {
        val r = 10f
        canvas.drawCircle(px, py, r, paintResolvedAnchor)
        canvas.drawLine(px - 14f, py, px + 14f, py, paintResolvedAnchor)
        canvas.drawLine(px, py - 14f, px, py + 14f, paintResolvedAnchor)
        canvas.drawText(label, px + 14f, py - 8f, paintProbeText)
    }

    private fun projectLocalToPixel(
        localX: Float,
        localY: Float,
        localZ: Float,
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float,
        outPx: FloatArray
    ) {
        FacePoseAdapter.projectFaceLocalToView(
            localX = localX,
            localY = localY,
            localZ = localZ,
            poseMatrix = faceState.poseMatrix,
            fw = faceState.faceWidth,
            projectionMatrix = viewProjMatrix,
            viewWidth = viewWidth,
            viewHeight = viewHeight,
            outScreenPt = outPx,
            scratchCam = scratchCam
        )
    }

    private fun drawProjectedEllipsoidRing(
        canvas: Canvas,
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float,
        planeMode: Int
    ) {
        val hx = lastResolvedHeadCenterFW[0]
        val hy = lastResolvedHeadCenterFW[1]
        val hz = lastResolvedHeadCenterFW[2]
        for (i in 0..PROXY_RING_SEGMENTS) {
            val c = COS_TABLE[i]
            val s = SIN_TABLE[i]
            val lx: Float
            val ly: Float
            val lz: Float
            when (planeMode) {
                0 -> { // XY coronal ring
                    lx = HEAD_PROXY_RX_FW * c
                    ly = HEAD_PROXY_RY_FW * s
                    lz = 0f
                }
                1 -> { // XZ transverse ring
                    lx = HEAD_PROXY_RX_FW * c
                    ly = 0f
                    lz = HEAD_PROXY_RZ_FW * s
                }
                else -> { // YZ sagittal ring
                    lx = 0f
                    ly = HEAD_PROXY_RY_FW * c
                    lz = HEAD_PROXY_RZ_FW * s
                }
            }
            projectLocalToPixel(hx + lx, hy + ly, hz + lz, faceState, viewWidth, viewHeight, ringCurrPx)
            if (i > 0) {
                canvas.drawLine(ringPrevPx[0], ringPrevPx[1], ringCurrPx[0], ringCurrPx[1], paintWireframe)
            }
            ringPrevPx[0] = ringCurrPx[0]
            ringPrevPx[1] = ringCurrPx[1]
        }
    }

    private fun mapLandmarkToPixel(
        u: Float,
        v: Float,
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float,
        outPx: FloatArray
    ) {
        FacePoseAdapter.mapLandmarkToView(
            u = u,
            v = v,
            imageToViewTransform = faceState.imageToViewTransform,
            isFrontCamera = faceState.isFrontCamera,
            viewWidth = viewWidth,
            viewHeight = viewHeight,
            outPt = outPx
        )
    }

    @JvmOverloads
    fun refreshCachedHudStringsIfNeeded(
        faceState: FaceState,
        fps: Float,
        tier: String,
        fovYCropDeg: Float = faceState.projectionParams.fovYDeg,
        fovYViewDeg: Float = faceState.projectionParams.fovYDeg,
        force: Boolean = false
    ): Boolean {
        val nowNs = FrameClock.elapsedRealtimeNs()
        if (!force && lastHudTextRefreshNs != 0L &&
            (nowNs - lastHudTextRefreshNs) < LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        ) {
            return false
        }
        lastHudTextRefreshNs = nowNs

        cachedLatencyLine = latencyProbe.cachedHudLine

        val cam = if (faceState.isFrontCamera) "FRONT (Mirrored)" else "REAR"
        val rotLabel = if (FacePoseAdapter.isNormalizedImageToViewTransform(faceState.imageToViewTransform)) {
            "ImageToView OK"
        } else if (faceState.imageToViewTransform != null) {
            "Matrix OK"
        } else {
            "Fallback FILL_CENTER"
        }
        val pathLabel = latencyProbe.pathModeLabel.ifEmpty { "A" }
        val camTexStatus = if (latencyProbe.cameraTextureOk) "OK" else "missing"
        cachedCameraLine =
            "Camera: $cam • PATH:$pathLabel • cam texture: $camTexStatus • Transform: $rotLabel • Tier $tier"
        val prov = faceProvider
        val sess = cameraSession
        val statusSuffix = if (latencyProbe.trackerStatusLabel.isNotEmpty()) {
            " • ${latencyProbe.trackerStatusLabel}"
        } else {
            ""
        }
        val srcW = prov?.lastSourceWidth?.takeIf { it > 0 } ?: latencyProbe.lastSourceWidth
        val srcH = prov?.lastSourceHeight?.takeIf { it > 0 } ?: latencyProbe.lastSourceHeight
        val cropW = prov?.lastCropWidth?.takeIf { it > 0 } ?: latencyProbe.lastCropWidth
        val cropH = prov?.lastCropHeight?.takeIf { it > 0 } ?: latencyProbe.lastCropHeight
        val rotDeg = prov?.lastRotationDegrees ?: latencyProbe.lastSourceRotationDeg
        val srcInfoSuffix = if (srcW > 0 && srcH > 0) {
            val smallSum = prov?.formatModeSourceSummary(AnalysisResolutionMode.SMALL) ?: "n/a"
            val legacySum = prov?.formatModeSourceSummary(AnalysisResolutionMode.LEGACY) ?: "n/a"
            " • src:${srcW}x${srcH}(crop ${cropW}x${cropH})@${rotDeg}° [S:$smallSum L:$legacySum]"
        } else {
            ""
        }
        cachedFovLine =
            "Analysis: ${latencyProbe.resolutionModeLabel} • ${latencyProbe.analysisFormatLabel} • ${latencyProbe.delegateLabel} • ${faceState.imageWidth}x${faceState.imageHeight}$srcInfoSuffix$statusSuffix"

        val projParams = faceState.projectionParams
        val fovSrc = projParams.fovSource
        val focalMm = projParams.focalLengthMm
        val cropDesc = projParams.scalerCropDescriptor
        cachedProjectionHudLine = String.format(
            Locale.US,
            "FOV: %.1f° (view %.1f°) • FOV source: %s • f=%.2fmm • crop=%s • depth=%db",
            fovYCropDeg,
            fovYViewDeg,
            fovSrc,
            focalMm,
            cropDesc,
            latencyProbe.lastDepthBits
        )

        cachedLifecycleLine = String.format(
            Locale.US,
            "Lifecycle: %s • Fade: %.2f • %.1f FPS",
            faceState.lifecycle.name,
            faceState.fadeWeight,
            fps
        )
        cachedPoseLine = String.format(
            Locale.US,
            "Pose (Deg): Y:%.1f P:%.1f R:%.1f",
            faceState.yawDeg,
            faceState.pitchDeg,
            faceState.rollDeg
        )
        val fwMm = (faceState.faceWidth * 1000f).toInt()
        cachedFwLine = String.format(
            Locale.US,
            "FW: %.3fm (%dmm)",
            faceState.faceWidth,
            fwMm
        )
        cachedFwMidLabel = "FW: ${fwMm}mm (#234-#454)"
        val predLabel = if (latencyProbe.lastPredictionOn) "ON" else "OFF"
        cachedStageLine = String.format(
            Locale.US,
            "Stages: conv %.1fms • wait %.1fms • inf %.1fms • q %.1fms • pred:%s (%.1fmm/%.1f°)",
            latencyProbe.rollingConversionMs,
            latencyProbe.rollingPendingWaitMs,
            latencyProbe.rollingInferenceMs,
            latencyProbe.lastQueueMs,
            predLabel,
            latencyProbe.rollingPredErrPosMm,
            latencyProbe.rollingPredErrRotDeg
        )

        val convModeStr = (prov?.conversionMode?.name ?: latencyProbe.conversionModeLabel).ifEmpty { "FAST" }
        cachedConvBreakdownLine = String.format(
            Locale.US,
            "Conv[%s]: tot %.1fms (unpk %.1f + yuv %.1f + px %.1f + rot %.1f + mp %.1f)",
            convModeStr,
            latencyProbe.rollingConversionMs,
            latencyProbe.rollingUnpackMs,
            latencyProbe.rollingYuvToRgbMs,
            latencyProbe.rollingBitmapCopyMs,
            latencyProbe.rollingRotateScaleMs,
            latencyProbe.rollingMpImageBuildMs
        )

        cachedPumpAndIdleLine = String.format(
            Locale.US,
            "Pump: %.1f/%.1fms (mean/max) • IdleGap: pend %.1fms / noPend %.1fms",
            latencyProbe.rollingPumpDelayMeanMs,
            latencyProbe.rollingPumpDelayMaxMs,
            latencyProbe.rollingIdleGapPendingMs,
            latencyProbe.rollingIdleGapNoPendingMs
        )

        val isBlendOn = (cameraSession?.isDebugBlendshapesEnabled ?: true) &&
            !latencyProbe.blendshapesEnabledLabel.startsWith("OFF")
        val blendStatusPrefix = if (isBlendOn) {
            "Blend: ON"
        } else {
            "blendshapes: held (stale)"
        }
        val swapMsVal = prov?.lastSwapMs?.takeIf { it > 0f } ?: latencyProbe.lastSwapMs
        cachedBlendAndPrioLine = String.format(
            Locale.US,
            "%s • inf(ON/OFF): %.1f/%.1fms • swap: %.1fms • Prio A:%s R:%s",
            blendStatusPrefix,
            latencyProbe.rollingInferenceBlendOnMs,
            latencyProbe.rollingInferenceBlendOffMs,
            swapMsVal,
            latencyProbe.analysisThreadPriorityLabel,
            latencyProbe.renderThreadPriorityLabel
        )

        val warmupState = if (latencyProbe.isInWarmup(nowNs) || latencyProbe.lastSampleWasWarmup) {
            "ACTIVE"
        } else {
            "steady"
        }
        val gov = governor
        val nowMs = nowNs / 1_000_000L
        val thermMon = gov?.thermalMonitor
        val thermLabel = thermMon?.formatThermalStatusLabel() ?: "NONE(0)"
        val headroom = thermMon?.queryThermalHeadroomIfNeeded(nowMs) ?: Float.NaN
        val headroomStr = if (headroom.isFinite() && headroom >= 0f) {
            String.format(Locale.US, "%.2f", headroom)
        } else {
            "n/a"
        }
        val targetHz = gov?.targetTrackerCadenceHz ?: PerformanceGovernor.MAX_TRACKER_CADENCE_HZ
        val targetSizePx = gov?.targetAnalysisSizePx ?: PerformanceGovernor.ANALYSIS_SIZE_HIGH_PX
        val govSummary = gov?.formatHudGovernorSummary(nowMs, latencyProbe.governorReasonLabel)
            ?: "Gov: ${targetHz}Hz@${targetSizePx}px (${latencyProbe.governorReasonLabel}, lowEnd=false, backoff=${PerformanceGovernor.PROMOTE_HEADROOM_MS / 1000L}s, promo=max)"
        cachedGovWarmupLine =
            "Therm: $thermLabel hdrm:$headroomStr • $govSummary • Warm-up: $warmupState"

        val dropped = prov?.framesDroppedCount ?: 0
        val replaced = prov?.pendingReplacedCount ?: 0
        val wdog = prov?.watchdogResetCount ?: 0
        val ovlp = prov?.overlappedConversionCount ?: 0
        val sessCadDrop = sess?.sessionCadenceThrottledCount ?: 0
        val provCadDefer = prov?.providerCadenceDeferredCount ?: 0
        val skipFreshEnabled = prov?.skipConversionIfPendingFresh ?: (sess?.skipConversionIfPendingFresh ?: false)
        val skippedBeforeConv = prov?.framesSkippedBeforeConversionCount ?: 0
        val previewFpsVal = if (latencyProbe.renderFps > 0f) latencyProbe.renderFps else fps
        val previewFpsStr = String.format(Locale.US, "%.1f", previewFpsVal)
        cachedCountersLine =
            "Counters: drop=$dropped repl=$replaced wdog=$wdog • ovlp=$ovlp cadDrop=$sessCadDrop/$provCadDefer • skipFresh:${if (skipFreshEnabled) "ON" else "OFF"}($skippedBeforeConv) • PATH:$pathLabel • cam texture: $camTexStatus • preview: ${previewFpsStr}fps"

        val motionLabel = if (latencyProbe.lastIsFastHeadMotion) {
            String.format(
                Locale.US,
                "FAST(%.2fm/s, %.0f°/s)",
                latencyProbe.lastHeadSpeedMps,
                latencyProbe.lastHeadAngularSpeedDegPerSec
            )
        } else {
            "STILL"
        }
        val dtMsVal = if (latencyProbe.rollingMaskTimeDeltaMs > 0f) {
            latencyProbe.rollingMaskTimeDeltaMs
        } else {
            latencyProbe.lastMaskTimeDeltaMs
        }
        cachedMaskOffsetLine = String.format(
            Locale.US,
            "MaskOffset: dt=%.1fms (fast: %.1fms) • shift=%.1fpx (fast p95/max: %.1f/%.1fpx) • motion: %s",
            dtMsVal,
            latencyProbe.rollingFastMotionMaskDeltaMs,
            latencyProbe.lastMaskOffsetPx,
            latencyProbe.fastMotionMaskOffsetP95Px,
            latencyProbe.maxFastMotionMaskOffsetPx,
            motionLabel
        )

        val maskTargetStr = inspectedRtName
            ?: debugMaskMode.maskId?.let { "\$mask:$it" }
            ?: "none"
        val mouthOpenRatio = FaceRegions.computeMouthOpeningRatio(faceState.landmarks)
        cachedMaskViewerLine = String.format(
            Locale.US,
            "MaskView: MASK:%s • target: %s • mouthOpen: %.2f",
            debugMaskMode.name,
            maskTargetStr,
            mouthOpenRatio
        )

        cachedIrisJitterLine = latencyProbe.formatIrisJitterSummaryLine()

        val jaw = (faceState.blendshapes["jawOpen"] ?: 0f).coerceIn(0f, 1f)
        val smileL = faceState.blendshapes["mouthSmileLeft"] ?: 0f
        val smileR = faceState.blendshapes["mouthSmileRight"] ?: 0f
        val smile = ((smileL + smileR) * 0.5f).coerceIn(0f, 1f)
        cachedJawOpen = jaw
        cachedSmile = smile
        cachedJawOpenPercentText = "${(jaw * 100f).toInt()}%"
        cachedSmilePercentText = "${(smile * 100f).toInt()}%"

        hudTextRevision++
        return true
    }

    private fun drawTelemetryCard(
        canvas: Canvas,
        faceState: FaceState,
        fps: Float,
        tier: String,
        fovYCropDeg: Float,
        fovYViewDeg: Float
    ) {
        refreshCachedHudStringsIfNeeded(faceState, fps, tier, fovYCropDeg, fovYViewDeg)

        val left = 24f
        val top = 130f
        val cardWidth = 720f
        val cardHeight = 450f

        cardRect.set(left, top, left + cardWidth, top + cardHeight)
        canvas.drawRoundRect(cardRect, 16f, 16f, paintHudBg)
        canvas.drawRoundRect(cardRect, 16f, 16f, paintHudBorder)

        var y = top + 34f
        canvas.drawText("SIGNAL & ALIGNMENT HUD", left + 16f, y, paintTextBold)

        // LatencyProbe 5s rolling window summary line (cached at ~2 Hz, zero per-frame formatting)
        y += 32f
        canvas.drawText(latencyProbe.cachedHudLine, left + 16f, y, paintProbeText)

        y += 30f
        canvas.drawText(cachedStageLine, left + 16f, y, paintText)

        y += 30f
        canvas.drawText(cachedCameraLine, left + 16f, y, paintText)

        y += 30f
        canvas.drawText(cachedFovLine, left + 16f, y, paintText)

        y += 30f
        canvas.drawText(cachedProjectionHudLine, left + 16f, y, paintProbeText)

        y += 30f
        canvas.drawText(cachedLifecycleLine, left + 16f, y, paintText)

        y += 30f
        canvas.drawText(cachedPoseLine, left + 16f, y, paintText)

        y += 30f
        canvas.drawText(cachedFwLine, left + 16f, y, paintText)

        y += 34f
        canvas.drawText("Blendshapes:", left + 16f, y, paintTextBold)

        // Draw live blendshape gauges: jawOpen and mouthSmile
        y += 28f
        drawGauge(canvas, "jawOpen", cachedJawOpen, left + 16f, y, 220f)

        y += 30f
        drawGauge(canvas, "smile", cachedSmile, left + 16f, y, 220f)
    }

    private fun drawGauge(canvas: Canvas, label: String, value: Float, x: Float, y: Float, width: Float) {
        canvas.drawText(label, x, y + 10f, paintText)
        val barLeft = x + 150f
        val barTop = y - 10f
        val barRight = barLeft + width
        val barBottom = y + 12f

        canvas.drawRect(barLeft, barTop, barRight, barBottom, paintGaugeBg)
        val fillRight = barLeft + (width * value.coerceIn(0f, 1f))
        canvas.drawRect(barLeft, barTop, fillRight, barBottom, paintGaugeFill)
        canvas.drawText("${(value * 100).toInt()}%", barRight + 12f, y + 10f, paintText)
    }
}

