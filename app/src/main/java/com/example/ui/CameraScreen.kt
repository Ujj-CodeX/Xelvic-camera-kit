package com.example.ui

import android.graphics.Bitmap
import android.graphics.RectF
import android.widget.FrameLayout
import android.widget.Toast
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import coil.compose.rememberAsyncImagePainter
import com.example.camera.CameraController
import com.example.engine.debug.DebugOverlay
import com.example.engine.render.XelvicGlSurfaceView
import com.example.repository.FilterRepository
import com.example.util.MediaStoreSaver
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.ceil

/**
 * Collapsible section state for the Debug HUD card:
 * - `"Latency"` defaults to **expanded** (`true`).
 * - `"Projection"` defaults to **collapsed** (`false`).
 * - `"Pose and blendshapes"` defaults to **collapsed** (`false`).
 */
data class DebugHudSectionState(
    val isLatencyExpanded: Boolean = true,
    val isProjectionExpanded: Boolean = false,
    val isPoseExpanded: Boolean = false
) {
    fun toggleLatency(): DebugHudSectionState = copy(isLatencyExpanded = !isLatencyExpanded)
    fun toggleProjection(): DebugHudSectionState = copy(isProjectionExpanded = !isProjectionExpanded)
    fun togglePose(): DebugHudSectionState = copy(isPoseExpanded = !isPoseExpanded)
}

/**
 * Pure, allocation-light layout bounds solver for the Debug HUD card and surrounding controls.
 *
 * Enforces:
 * 1. Strict vertical order below status bar / display cutout:
 *    `(a)` top bar -> `(b)` debug toggle row (`RES / FMT / DEL / PRED`) -> `(c)` HUD text card.
 * 2. The HUD card starts below the measured bottom of `(a)` and `(b)`, and its height is capped
 *    to the free vertical space above the bottom controls (`bottomControlsBounds.top - spacingPx`)
 *    so it never overlaps the filter carousel, shutter button, or bottom navigation area.
 */
object DebugHudLayoutCalculator {

    data class HudRect(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    ) {
        val width: Float get() = (right - left).coerceAtLeast(0f)
        val height: Float get() = (bottom - top).coerceAtLeast(0f)

        /**
         * Returns `true` if this rectangle intersects [other] (with a `0.5 px` sub-pixel epsilon).
         */
        fun intersects(other: HudRect): Boolean {
            if (width <= 0f || height <= 0f || other.width <= 0f || other.height <= 0f) {
                return false
            }
            return left < other.right - 0.5f &&
                other.left < right - 0.5f &&
                top < other.bottom - 0.5f &&
                other.top < bottom - 0.5f
        }
    }

    data class LayoutBoundsResult(
        val topBarBounds: HudRect,
        val debugToggleRowBounds: HudRect,
        val hudCardBounds: HudRect,
        val bottomControlsBounds: HudRect,
        val maxAvailableHudHeightPx: Float,
        val isHudScrollable: Boolean
    ) {
        fun intersects(a: HudRect, b: HudRect): Boolean = a.intersects(b)
    }

    /**
     * Computes the exact `maxAvailableHudHeightPx` and bounded `hudCardBounds` from the measured
     * root-space bounds of `(a)` the top bar, `(b)` the debug toggle row, and the bottom controls.
     */
    fun computeHudBounds(
        screenWidthPx: Float,
        screenHeightPx: Float,
        topBarBounds: HudRect,
        debugToggleRowBounds: HudRect,
        bottomControlsBounds: HudRect,
        horizontalPaddingPx: Float,
        verticalSpacingPx: Float,
        desiredHudContentHeightPx: Float
    ): LayoutBoundsResult {
        val controlsBottomPx = maxOf(topBarBounds.bottom, debugToggleRowBounds.bottom)
        val hudTopPx = controlsBottomPx + verticalSpacingPx
        val bottomLimitPx = (if (bottomControlsBounds.top > 0f) bottomControlsBounds.top else screenHeightPx) - verticalSpacingPx
        val maxAvailableHeightPx = (bottomLimitPx - hudTopPx).coerceAtLeast(0f)
        val clampedHudHeightPx = desiredHudContentHeightPx.coerceAtLeast(0f).coerceAtMost(maxAvailableHeightPx)
        val hudLeftPx = horizontalPaddingPx.coerceAtLeast(0f)
        val hudRightPx = (screenWidthPx - horizontalPaddingPx).coerceAtLeast(hudLeftPx)

        val hudBounds = HudRect(
            left = hudLeftPx,
            top = hudTopPx,
            right = hudRightPx,
            bottom = hudTopPx + clampedHudHeightPx
        )

        return LayoutBoundsResult(
            topBarBounds = topBarBounds,
            debugToggleRowBounds = debugToggleRowBounds,
            hudCardBounds = hudBounds,
            bottomControlsBounds = bottomControlsBounds,
            maxAvailableHudHeightPx = maxAvailableHeightPx,
            isHudScrollable = desiredHudContentHeightPx > maxAvailableHeightPx + 0.5f
        )
    }

    /**
     * Computes measured control bounds for a given screen configuration (`widthDp x heightDp`,
     * `density`, `fontScale`) and section expansion state (`sectionState`), accounting for
     * multi-line text wrapping at `fontScale` (`1.0`, `1.3`, etc.) and wrapped toggle
     * pills on narrow screens.
     */
    fun measureScreenLayout(
        widthDp: Float,
        heightDp: Float,
        density: Float = 2.625f,
        fontScale: Float = 1.0f,
        statusBarInsetDp: Float = 28f,
        navBarInsetDp: Float = 24f,
        isDebugEnabled: Boolean = true,
        sectionState: DebugHudSectionState = DebugHudSectionState()
    ): LayoutBoundsResult {
        val screenW = widthDp * density
        val screenH = heightDp * density
        val padH = 16f * density
        val padV = 8f * density
        val spacingPx = 6f * density
        val hudTopGapPx = 8f * density

        // (a) Top bar: min 44.dp icon buttons or scaled pill height
        val topBarTop = statusBarInsetDp * density + padV
        val pillTextHeight = (13f * fontScale * density) + (12f * density)
        val topBarHeight = maxOf(44f * density, pillTextHeight)
        val topBarBounds = HudRect(padH, topBarTop, screenW - padH, topBarTop + topBarHeight)

        // (b) Debug toggle row (10 pills: RES / FMT / DEL / PRED / BLEND / PRIO / RPRIO / CONV / ANC / PATH)
        // Wrapped via FlowRow so on 360dp screens all 10 pills wrap into 2+ rows without horizontal clipping,
        // and the HUD card starts below the measured bottom of the last wrapped row.
        val debugRowBounds = if (isDebugEnabled) {
            val rowTop = topBarBounds.bottom + spacingPx
            val pillHeight = (11f * fontScale * density) + (14f * density)
            val pillCount = 10
            val singlePillEstW = (62f * fontScale * density) + (16f * density)
            val availW = (screenW - 2f * padH).coerceAtLeast(1f)
            val pillsPerRow = ((availW + spacingPx) / (singlePillEstW + spacingPx)).toInt().coerceAtLeast(1)
            val rowLines = ceil(pillCount.toFloat() / pillsPerRow.toFloat()).toInt().coerceAtLeast(1)
            val rowHeight = rowLines * pillHeight + (rowLines - 1) * spacingPx
            HudRect(padH, rowTop, screenW - padH, rowTop + rowHeight)
        } else {
            HudRect(padH, topBarBounds.bottom, screenW - padH, topBarBounds.bottom)
        }

        // Bottom controls: Filter badge + LazyRow carousel (90.dp) + spacer (16.dp) + shutter (76.dp) + bottom padding
        val filterBadgeH = (14f * fontScale * density) + (24f * density)
        val carouselH = (64f * density) + (4f * density) + (14f * fontScale * density) + (16f * density)
        val shutterH = 76f * density
        val bottomPad = (navBarInsetDp + 16f) * density
        val bottomTotalH = filterBadgeH + carouselH + (16f * density) + shutterH + bottomPad
        val bottomTop = (screenH - bottomTotalH).coerceAtLeast(debugRowBounds.bottom + 80f * density)
        val bottomControlsBounds = HudRect(0f, bottomTop, screenW, screenH)

        // (c) Desired HUD card height based on expanded sections & wrapped line count
        val desiredHudHeightPx = if (!isDebugEnabled) {
            0f
        } else {
            val headerH = (12f * fontScale * density) + (16f * density)
            val lineH = (15f * fontScale * density) + (4f * density)
            val cardInnerPad = 20f * density
            var h = cardInnerPad + 3f * headerH
            // Estimate line wrapping factor for long telemetry lines (~85 chars)
            val availTextW = (screenW - 2f * padH - 24f * density).coerceAtLeast(100f)
            val charW = 6.5f * fontScale * density
            val wrapLinesLong = ceil((88f * charW) / availTextW).toInt().coerceAtLeast(1)
            val wrapLinesMed = ceil((64f * charW) / availTextW).toInt().coerceAtLeast(1)

            if (sectionState.isLatencyExpanded) {
                h += (3 * wrapLinesLong + 4 * wrapLinesMed) * lineH + (10f * density)
            }
            if (sectionState.isProjectionExpanded) {
                h += (wrapLinesMed + wrapLinesMed + wrapLinesLong) * lineH + (8f * density)
            }
            if (sectionState.isPoseExpanded) {
                val gaugeH = maxOf(22f * density, 14f * fontScale * density + 8f * density)
                h += (wrapLinesMed + wrapLinesMed) * lineH + 2f * gaugeH + (10f * density)
            }
            h
        }

        return computeHudBounds(
            screenWidthPx = screenW,
            screenHeightPx = screenH,
            topBarBounds = topBarBounds,
            debugToggleRowBounds = debugRowBounds,
            bottomControlsBounds = bottomControlsBounds,
            horizontalPaddingPx = padH,
            verticalSpacingPx = hudTopGapPx,
            desiredHudContentHeightPx = desiredHudHeightPx
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun CameraScreen(
    filterRepository: FilterRepository,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val density = LocalDensity.current
    val coroutineScope = rememberCoroutineScope()

    val cameraController = remember { CameraController(context, lifecycleOwner) }
    val installedFilters by filterRepository.installedFilters.collectAsState()
    val activeEffect by filterRepository.activeEffect.collectAsState()
    val selectedId by filterRepository.selectedFilterId.collectAsState()

    var glSurfaceView by remember { mutableStateOf<XelvicGlSurfaceView?>(null) }
    var previewViewRef by remember { mutableStateOf<PreviewView?>(null) }
    var diagnosticViewRef by remember { mutableStateOf<android.view.View?>(null) }

    var isDebugEnabled by remember { mutableStateOf(false) }
    var isPredictionEnabled by remember { mutableStateOf(true) }
    var resolutionMode by remember { mutableStateOf(com.example.engine.perception.AnalysisResolutionMode.SMALL) }
    var useRgbaFormat by remember { mutableStateOf(false) }
    var useGpuDelegate by remember { mutableStateOf(false) }
    var isBlendshapesDebugEnabled by remember { mutableStateOf(true) }
    var useLowAnalysisPriority by remember { mutableStateOf(false) }
    var useHighRenderPriority by remember { mutableStateOf(false) }
    var conversionMode by remember { mutableStateOf(com.example.engine.perception.ConversionMode.FAST) }
    var showResolvedAnchorMarkers by remember { mutableStateOf(false) }
    var pathMode by remember { mutableStateOf(com.example.engine.render.CompositionMode.SINGLE_SCENE_GL) }
    var maskDebugMode by remember { mutableStateOf(com.example.engine.render.RegionMaskPass.DebugMaskMode.OFF) }
    var irisAnalysisSizePx by remember { mutableIntStateOf(480) }
    var activeDelegateStatus by remember { mutableStateOf("CPU") }
    val debugOverlay = remember(cameraController) {
        DebugOverlay(
            latencyProbe = cameraController.cameraSession.latencyProbe,
            governor = cameraController.governor,
            faceProvider = cameraController.faceProvider,
            cameraSession = cameraController.cameraSession
        ).apply {
            // Compose renders the HUD text card below the top bar and debug toggle row
            drawHudCardOnCanvas = false
        }
    }
    var currentFps by remember { mutableFloatStateOf(30f) }
    var currentTier by remember { mutableStateOf("A") }
    var isFaceTracking by remember { mutableStateOf(false) }

    // ~2 Hz HUD text revision counter and collapsible section state (Latency expanded by default)
    var hudTextRevision by remember { mutableIntStateOf(0) }
    var hudSectionState by remember { mutableStateOf(DebugHudSectionState()) }

    // Measured root-space vertical coordinates (in px) to cap HUD card height above bottom controls
    var rootHeightPx by remember { mutableFloatStateOf(0f) }
    var topControlsBottomPx by remember { mutableFloatStateOf(0f) }
    var bottomControlsTopPx by remember { mutableFloatStateOf(0f) }

    var lastCapturedPhoto by remember { mutableStateOf<Bitmap?>(null) }
    var isShowingPhotoPreview by remember { mutableStateOf(false) }
    var isCapturing by remember { mutableStateOf(false) }
    var noFaceRetakeMessage by remember { mutableStateOf<String?>(null) }
    var isRecordingVideo by remember { mutableStateOf(false) }
    var recordingElapsedSec by remember { mutableIntStateOf(0) }

    val startVideoRecordingWithAudio: (Boolean) -> Unit = { enableAudio ->
        if (!isRecordingVideo && !isCapturing) {
            val outDir = File(context.cacheDir, "recordings").apply { mkdirs() }
            val outMp4 = File(outDir, "xelvic_clip_${System.currentTimeMillis()}.mp4")
            val started = cameraController.startVideoRecording(
                outputFile = outMp4,
                enableAudio = enableAudio,
                previewView = previewViewRef,
                glView = glSurfaceView
            )
            if (started) {
                recordingElapsedSec = 0
                isRecordingVideo = true
            } else {
                Toast.makeText(context, "Unable to start video recording", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val stopAndSaveVideoRecording: () -> Unit = {
        if (isRecordingVideo) {
            isRecordingVideo = false
            val res = cameraController.stopVideoRecording(
                previewView = previewViewRef,
                glView = glSurfaceView
            )
            if (res != null) {
                coroutineScope.launch {
                    val uri = MediaStoreSaver.saveVideo(context, res.outputFile)
                    val msg = if (uri != null) {
                        "Video saved to Movies/Xelvic (${res.recordedFrameCount} frames)"
                    } else {
                        "Video recorded (${res.recordedFrameCount} frames)"
                    }
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    val audioPermissionLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { granted ->
        startVideoRecordingWithAudio(granted)
    }

    val triggerVideoRecordToggle: () -> Unit = {
        if (isRecordingVideo) {
            stopAndSaveVideoRecording()
        } else if (!isCapturing) {
            val hasAudioPerm = androidx.core.content.ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.RECORD_AUDIO
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
            if (hasAudioPerm) {
                startVideoRecordingWithAudio(true)
            } else {
                audioPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    LaunchedEffect(isRecordingVideo) {
        if (isRecordingVideo) {
            recordingElapsedSec = 0
            while (isRecordingVideo) {
                kotlinx.coroutines.delay(1000L)
                if (isRecordingVideo) {
                    recordingElapsedSec++
                }
            }
        }
    }

    val triggerPhotoCapture: () -> Unit = {
        if (isRecordingVideo) {
            stopAndSaveVideoRecording()
        } else if (!isCapturing) {
            isCapturing = true
            noFaceRetakeMessage = null
            coroutineScope.launch {
                try {
                    val currentEffect = filterRepository.activeEffect.value ?: activeEffect
                    glSurfaceView?.activeEffect = currentEffect

                    val pv = previewViewRef
                    val previewBmp = pv?.bitmap
                    val previewAspect = if (pv != null && pv.width > 0 && pv.height > 0) {
                        pv.width.toFloat() / pv.height.toFloat()
                    } else {
                        9f / 16f
                    }
                    val result = cameraController.capturePhoto(
                        activeEffect = currentEffect,
                        previewBitmap = previewBmp,
                        targetAspectRatio = previewAspect
                    )
                    if (result != null) {
                        lastCapturedPhoto = result.bitmap
                        val uri = MediaStoreSaver.saveBitmap(context, result.bitmap)
                        if (result.canOneTapRetake && result.userNotificationMessage != null) {
                            noFaceRetakeMessage = result.userNotificationMessage
                            Toast.makeText(context, result.userNotificationMessage, Toast.LENGTH_LONG).show()
                        } else {
                            Toast.makeText(
                                context,
                                if (uri != null) "Photo saved to Pictures/Xelvic" else "Photo captured",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                } finally {
                    isCapturing = false
                }
            }
        }
    }

    // Launcher to allow importing any standalone .xfilter / .zip filter package at runtime
    val importFilterLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            coroutineScope.launch {
                val res = filterRepository.installFromUri(uri)
                when (res) {
                    is com.example.engine.pkg.PackageInstaller.InstallResult.Success -> {
                        Toast.makeText(context, "Loaded filter: ${res.effect.packageName}", Toast.LENGTH_SHORT).show()
                    }
                    is com.example.engine.pkg.PackageInstaller.InstallResult.Failure -> {
                        Toast.makeText(context, "Filter error: ${res.reason}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    val lifecycleObserver = remember(cameraController, filterRepository) {
        com.example.camera.CameraLifecycleObserver.forCameraController(
            cameraController = cameraController,
            glSurfaceViewProvider = { glSurfaceView },
            preloader = filterRepository.preloader
        )
    }

    DisposableEffect(lifecycleOwner, lifecycleObserver) {
        lifecycleOwner.lifecycle.addObserver(lifecycleObserver)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(lifecycleObserver)
            lifecycleObserver.dispose()
        }
    }

    // Update active effect in GL view and dynamically adjust tracker output flags without restarting camera/tracker.
    // Unified SINGLE_SCENE_GL pipeline handles 3D AR/VR, Beauty, and 3D+Beauty combo filters seamlessly with zero surface hot-swap.
    LaunchedEffect(activeEffect, previewViewRef, glSurfaceView) {
        glSurfaceView?.activeEffect = activeEffect
        debugOverlay.anchorResolver = com.example.engine.scene.AnchorResolver(
            activeEffect?.anchorResolver?.customAnchors ?: emptyMap()
        )
        cameraController.applyEffectRequirements(activeEffect)
        val needsSingleSceneGl = activeEffect?.nodes?.any {
            it.type == "post" || it.type == "recolor" || it.type == "eyeColor"
        } == true
        if (needsSingleSceneGl &&
            pathMode != com.example.engine.render.CompositionMode.SINGLE_SCENE_GL &&
            previewViewRef != null &&
            glSurfaceView != null
        ) {
            pathMode = com.example.engine.render.CompositionMode.SINGLE_SCENE_GL
            cameraController.setPathMode(
                com.example.engine.render.CompositionMode.SINGLE_SCENE_GL,
                previewViewRef,
                glSurfaceView
            )
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
            .onGloballyPositioned { coords ->
                rootHeightPx = coords.size.height.toFloat()
            }
            .pointerInput(Unit) {
                detectTapGestures {
                    // Tap event sent to signal bus for tap-reactive filters
                    activeEffect?.signalBus?.triggerTap()
                }
            }
    ) {
        // 1. CameraX PreviewView and Transparent GLSurfaceView Overlay
        AndroidView(
            factory = { ctx ->
                val container = FrameLayout(ctx)

                val preview = PreviewView(ctx).apply {
                    // Use PERFORMANCE (SurfaceView at layer -2) underneath XelvicGlSurfaceView
                    // (setZOrderMediaOverlay at layer -1), keeping all Jetpack Compose UI at layer 0 on top.
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    setBackgroundColor(android.graphics.Color.TRANSPARENT)
                }
                previewViewRef = preview
                container.addView(preview)

                val diagnosticView = object : android.view.View(ctx) {
                    init {
                        isClickable = false
                        isFocusable = false
                    }

                    override fun onDraw(canvas: android.graphics.Canvas) {
                        super.onDraw(canvas)
                        val state = cameraController.signalPipeline.getLatestState()
                        debugOverlay.draw(
                            canvas = canvas,
                            faceState = state,
                            viewWidth = width.toFloat(),
                            viewHeight = height.toFloat(),
                            fps = currentFps,
                            tier = currentTier
                        )
                    }
                }
                diagnosticViewRef = diagnosticView

                val glView = XelvicGlSurfaceView(ctx).apply {
                    governor = cameraController.governor
                    signalPipeline = cameraController.signalPipeline
                    cameraSession = cameraController.cameraSession
                    preloader = filterRepository.preloader
                    bindCameraInputSurface(cameraController.cameraInputSurface)
                    setRequestedPathMode(pathMode)
                    this.isPredictionEnabled = isPredictionEnabled
                    this.activeEffect = filterRepository.activeEffect.value
                    onVideoRecordingAutoStopped = { autoResult ->
                        isRecordingVideo = false
                        coroutineScope.launch {
                            MediaStoreSaver.saveVideo(ctx, autoResult.outputFile)
                        }
                    }
                    onFpsCalculated = { fps, tier ->
                        currentFps = fps
                        currentTier = tier
                        isFaceTracking = cameraController.signalPipeline.lifecycle.isVisible
                        activeDelegateStatus = cameraController.faceProvider.activeDelegateLabel
                        if (debugOverlay.isEnabled) {
                            val state = cameraController.signalPipeline.getLatestState()
                            val vw = diagnosticView.width.toFloat().coerceAtLeast(1f)
                            val vh = diagnosticView.height.toFloat().coerceAtLeast(1f)
                            val viewAspect = (vw / vh).coerceAtLeast(0.01f)
                            val fovCrop = state.projectionParams.fovYDeg
                            val fovView = state.projectionParams.visibleViewFovYDeg(viewAspect)
                            debugOverlay.refreshCachedHudStringsIfNeeded(
                                faceState = state,
                                fps = fps,
                                tier = tier,
                                fovYCropDeg = fovCrop,
                                fovYViewDeg = fovView
                            )
                            hudTextRevision = debugOverlay.hudTextRevision
                        }
                        diagnosticView.postInvalidate()
                    }
                }
                glSurfaceView = glView
                container.addView(glView)
                container.addView(diagnosticView)

                cameraController.setPathMode(pathMode, preview, glView)
                cameraController.start(preview)
                cameraController.applyEffectRequirements(filterRepository.activeEffect.value)
                lifecycleObserver.syncGlTargetWithCurrentState(lifecycleOwner.lifecycle.currentState)
                container
            },
            update = {
                glSurfaceView?.activeEffect = activeEffect
                glSurfaceView?.isPredictionEnabled = isPredictionEnabled
            },
            modifier = Modifier.fillMaxSize()
        )

        // 2. Top Controls & Telemetry Bar:
        //    Strict vertical order below status bar / display cutout:
        //    (a) Top bar -> (b) Debug toggle row (RES / FMT / DEL / PRED) -> (c) Collapsible HUD card.
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp)
        ) {
            // (a) Top Bar: XELVIC Tier • FPS pill + Import (+), Info (Debug toggle), Switch Camera
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("top_bar_row")
                    .onGloballyPositioned { coords ->
                        val b = coords.boundsInRoot()
                        if (!isDebugEnabled) {
                            topControlsBottomPx = b.bottom
                        }
                    },
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // App Branding & Tier Badge
                Surface(
                    color = Color.Black.copy(alpha = 0.6f),
                    shape = RoundedCornerShape(20.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, Color.White.copy(alpha = 0.2f))
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(
                                    if (isFaceTracking) Color(0xFF10B981) else Color(0xFFF59E0B)
                                )
                        )
                        Text(
                            text = "XELVIC",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            letterSpacing = 1.sp
                        )
                        Text(
                            text = "Tier $currentTier • ${"%.0f".format(currentFps)} fps",
                            color = Color(0xFF38BDF8),
                            fontSize = 11.sp
                        )
                    }
                }

                // Top action buttons: Import Filter, Gizmo debug & Camera flip
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Import .xfilter / .zip package button
                    IconButton(
                        onClick = {
                            importFilterLauncher.launch("*/*")
                        },
                        modifier = Modifier
                            .size(44.dp)
                            .background(Color.Black.copy(alpha = 0.5f), CircleShape)
                            .testTag("import_filter_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = "Import Filter Package",
                            tint = Color.White
                        )
                    }

                    // Debug toggle button (toggles HUD card + debug button row + 3D gizmo cleanly on/off)
                    IconButton(
                        onClick = {
                            val nextEnabled = !isDebugEnabled
                            isDebugEnabled = nextEnabled
                            debugOverlay.isEnabled = nextEnabled
                            if (nextEnabled) {
                                val state = cameraController.signalPipeline.getLatestState()
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = state,
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                            diagnosticViewRef?.postInvalidate()
                        },
                        modifier = Modifier
                            .size(44.dp)
                            .background(if (isDebugEnabled) Color(0xFF0284C7) else Color.Black.copy(alpha = 0.5f), CircleShape)
                            .testTag("debug_info_toggle_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = "Toggle Debug Gizmo",
                            tint = Color.White
                        )
                    }

                    // Camera switch button
                    IconButton(
                        onClick = {
                            previewViewRef?.let {
                                cameraController.toggleCamera(it)
                                glSurfaceView?.onCameraConfigChanged(
                                    isFrontCamera = cameraController.isFrontCamera.value,
                                    rotationDegrees = cameraController.cameraSession.displayRotationDegrees
                                )
                            }
                        },
                        modifier = Modifier
                            .size(44.dp)
                            .background(Color.Black.copy(alpha = 0.5f), CircleShape)
                            .testTag("switch_camera_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.FlipCameraAndroid,
                            contentDescription = "Switch Camera",
                            tint = Color.White
                        )
                    }
                }
            }

            // (b) Debug A/B switches row (RES / FMT / DEL / PRED / BLEND / PRIO / RPRIO) + (c) Collapsible HUD Card
            if (isDebugEnabled) {
                Spacer(modifier = Modifier.height(6.dp))
                FlowRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("debug_toggle_row")
                        .onGloballyPositioned { coords ->
                            topControlsBottomPx = coords.boundsInRoot().bottom
                        },
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    // 1. Resolution Mode toggle (SMALL default vs LEGACY)
                    Surface(
                        color = if (resolutionMode == com.example.engine.perception.AnalysisResolutionMode.SMALL) {
                            Color(0xFF0284C7).copy(alpha = 0.88f)
                        } else {
                            Color(0xFF475569).copy(alpha = 0.88f)
                        },
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("analysis_res_mode_toggle")
                            .clickable {
                                val next = if (resolutionMode == com.example.engine.perception.AnalysisResolutionMode.LEGACY) {
                                    com.example.engine.perception.AnalysisResolutionMode.SMALL
                                } else {
                                    com.example.engine.perception.AnalysisResolutionMode.LEGACY
                                }
                                resolutionMode = next
                                cameraController.setAnalysisResolutionMode(next, previewViewRef)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = "RES:${resolutionMode.name}",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 2. Output Format toggle (YUV_420_888 default vs RGBA_8888)
                    Surface(
                        color = if (useRgbaFormat) Color(0xFF8B5CF6).copy(alpha = 0.88f) else Color(0xFF475569).copy(alpha = 0.88f),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("analysis_format_toggle")
                            .clickable {
                                useRgbaFormat = !useRgbaFormat
                                cameraController.setUseRgbaOutputFormat(useRgbaFormat, previewViewRef)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = if (useRgbaFormat) "FMT:RGBA" else "FMT:YUV",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 3. MediaPipe Delegate toggle (CPU default vs GPU, with GPU->CPU fallback indicator)
                    Surface(
                        color = if (useGpuDelegate) Color(0xFFD97706).copy(alpha = 0.88f) else Color(0xFF475569).copy(alpha = 0.88f),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("mediapipe_delegate_toggle")
                            .clickable {
                                useGpuDelegate = !useGpuDelegate
                                cameraController.setUseGpuDelegate(useGpuDelegate)
                                activeDelegateStatus = cameraController.faceProvider.activeDelegateLabel
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = "DEL:$activeDelegateStatus",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 4. Prediction ON/OFF toggle
                    Surface(
                        color = if (isPredictionEnabled) Color(0xFF10B981).copy(alpha = 0.88f) else Color(0xFF64748B).copy(alpha = 0.88f),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("prediction_ab_toggle")
                            .clickable {
                                isPredictionEnabled = !isPredictionEnabled
                                cameraController.signalPipeline.isPredictionEnabled = isPredictionEnabled
                                glSurfaceView?.isPredictionEnabled = isPredictionEnabled
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = if (isPredictionEnabled) "PRED:ON" else "PRED:OFF",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 5. Face Blendshapes ON/OFF toggle (recreates landmarker on analysisExecutor with 3s warm-up)
                    Surface(
                        color = if (isBlendshapesDebugEnabled) Color(0xFF0D9488).copy(alpha = 0.88f) else Color(0xFF64748B).copy(alpha = 0.88f),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("blendshapes_debug_toggle")
                            .clickable {
                                isBlendshapesDebugEnabled = !isBlendshapesDebugEnabled
                                cameraController.setDebugBlendshapesEnabled(isBlendshapesDebugEnabled)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = if (isBlendshapesDebugEnabled) "BLEND:ON" else "BLEND:OFF",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 6. Analysis/Conversion Thread Priority toggle (NORM default vs LOW)
                    Surface(
                        color = if (useLowAnalysisPriority) Color(0xFFEC4899).copy(alpha = 0.88f) else Color(0xFF475569).copy(alpha = 0.88f),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("analysis_priority_toggle")
                            .clickable {
                                useLowAnalysisPriority = !useLowAnalysisPriority
                                cameraController.setUseLowAnalysisThreadPriority(useLowAnalysisPriority)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = if (useLowAnalysisPriority) "PRIO:LOW" else "PRIO:NORM",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 7. GL Render Thread Priority toggle (NORM default vs HIGH)
                    Surface(
                        color = if (useHighRenderPriority) Color(0xFFF59E0B).copy(alpha = 0.88f) else Color(0xFF475569).copy(alpha = 0.88f),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("render_priority_toggle")
                            .clickable {
                                useHighRenderPriority = !useHighRenderPriority
                                cameraController.setUseHighRenderThreadPriority(useHighRenderPriority, glSurfaceView)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = if (useHighRenderPriority) "RPRIO:HIGH" else "RPRIO:NORM",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 8. Conversion Fast/Old Path toggle (CONV:FAST default vs CONV:OLD)
                    Surface(
                        color = if (conversionMode == com.example.engine.perception.ConversionMode.FAST) {
                            Color(0xFF10B981).copy(alpha = 0.88f)
                        } else {
                            Color(0xFF475569).copy(alpha = 0.88f)
                        },
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("conversion_mode_toggle")
                            .clickable {
                                val next = if (conversionMode == com.example.engine.perception.ConversionMode.FAST) {
                                    com.example.engine.perception.ConversionMode.OLD
                                } else {
                                    com.example.engine.perception.ConversionMode.FAST
                                }
                                conversionMode = next
                                cameraController.setConversionMode(next)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = "CONV:${conversionMode.name}",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 9. Resolved Anchor Markers toggle (ANC:OFF default vs ANC:ON)
                    Surface(
                        color = if (showResolvedAnchorMarkers) {
                            Color(0xFFDB2777).copy(alpha = 0.88f)
                        } else {
                            Color(0xFF475569).copy(alpha = 0.88f)
                        },
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("resolved_anchors_debug_toggle")
                            .clickable {
                                val next = !showResolvedAnchorMarkers
                                showResolvedAnchorMarkers = next
                                debugOverlay.showResolvedAnchorMarkers = next
                                diagnosticViewRef?.postInvalidate()
                            }
                    ) {
                        Text(
                            text = if (showResolvedAnchorMarkers) "ANC:ON" else "ANC:OFF",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 10. Path Mode toggle (PATH:A default vs PATH:B single-scene GL)
                    Surface(
                        color = if (pathMode == com.example.engine.render.CompositionMode.SINGLE_SCENE_GL) {
                            Color(0xFF0EA5E9).copy(alpha = 0.88f)
                        } else {
                            Color(0xFF475569).copy(alpha = 0.88f)
                        },
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("path_mode_debug_toggle")
                            .clickable {
                                val next = if (pathMode == com.example.engine.render.CompositionMode.TRANSPARENT_OVERLAY) {
                                    com.example.engine.render.CompositionMode.SINGLE_SCENE_GL
                                } else {
                                    com.example.engine.render.CompositionMode.TRANSPARENT_OVERLAY
                                }
                                pathMode = next
                                cameraController.setPathMode(next, previewViewRef, glSurfaceView)
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = if (pathMode == com.example.engine.render.CompositionMode.SINGLE_SCENE_GL) "PATH:B" else "PATH:A",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 11. Region Mask Viewer toggle (MASK:OFF -> EYES -> MOUTH -> IRIS -> SKIN)
                    Surface(
                        color = if (maskDebugMode != com.example.engine.render.RegionMaskPass.DebugMaskMode.OFF) {
                            Color(0xFF059669).copy(alpha = 0.88f)
                        } else {
                            Color(0xFF475569).copy(alpha = 0.88f)
                        },
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("mask_debug_mode_toggle")
                            .clickable {
                                val next = maskDebugMode.next()
                                maskDebugMode = next
                                debugOverlay.debugMaskMode = next
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                                diagnosticViewRef?.postInvalidate()
                            }
                    ) {
                        Text(
                            text = "MASK:${maskDebugMode.name}",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }

                    // 12. Iris Analysis Size toggle (IRIS:480 default -> 640 -> 320 -> 480)
                    Surface(
                        color = if (cameraController.governor.isIrisTrackingRequired) {
                            Color(0xFF7C3AED).copy(alpha = 0.88f)
                        } else {
                            Color(0xFF475569).copy(alpha = 0.88f)
                        },
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .testTag("iris_size_debug_toggle")
                            .clickable {
                                irisAnalysisSizePx = cameraController.cameraSession.cycleConfiguredIrisAnalysisSizePx()
                                debugOverlay.refreshCachedHudStringsIfNeeded(
                                    faceState = cameraController.signalPipeline.getLatestState(),
                                    fps = currentFps,
                                    tier = currentTier,
                                    force = true
                                )
                                hudTextRevision = debugOverlay.hudTextRevision
                            }
                    ) {
                        Text(
                            text = "IRIS:${irisAnalysisSizePx}",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp)
                        )
                    }
                }

                // (c) Debug HUD Text Card: placed directly below (b) and constrained above bottom controls
                Spacer(modifier = Modifier.height(8.dp))
                val maxHudHeightDp: Dp = remember(topControlsBottomPx, bottomControlsTopPx, rootHeightPx, density) {
                    val gapPx = with(density) { 12.dp.toPx() }
                    val bottomLimit = if (bottomControlsTopPx > 0f) {
                        bottomControlsTopPx
                    } else if (rootHeightPx > 0f) {
                        rootHeightPx * 0.65f
                    } else {
                        1000f
                    }
                    val availPx = (bottomLimit - topControlsBottomPx - gapPx).coerceAtLeast(with(density) { 96.dp.toPx() })
                    with(density) { availPx.toDp() }
                }

                DebugHudCard(
                    debugOverlay = debugOverlay,
                    hudTextRevision = hudTextRevision,
                    sectionState = hudSectionState,
                    onSectionStateChange = { hudSectionState = it },
                    maxHeightDp = maxHudHeightDp
                )
            }
        }

        // 3. Bottom Controls: Filter Carousel + Shutter
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(bottom = 16.dp)
                .testTag("bottom_controls_column")
                .onGloballyPositioned { coords ->
                    bottomControlsTopPx = coords.boundsInRoot().top
                },
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Filter info label
            activeEffect?.let { effect ->
                val needsCameraSceneInLive = effect.nodes.any { it.type == "post" } &&
                    pathMode == com.example.engine.render.CompositionMode.TRANSPARENT_OVERLAY &&
                    (glSurfaceView?.isPostSkippedInLivePreview(effect) == true)
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(bottom = 12.dp)
                ) {
                    Surface(
                        color = Color.Black.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = effect.packageName,
                            color = Color.White,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 6.dp)
                        )
                    }
                    if (needsCameraSceneInLive) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Surface(
                            color = Color(0xCC1E293B),
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier.testTag("needs_camera_scene_badge")
                        ) {
                            Text(
                                text = "Needs camera scene (post active in photo)",
                                color = Color(0xFFFBBF24),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Medium,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }
                }
            }

            // Filter Carousel
            LazyRow(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                contentPadding = PaddingValues(horizontal = 24.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // "None" option
                item {
                    FilterItem(
                        title = "Normal",
                        previewPath = null,
                        isSelected = selectedId == null,
                        onClick = { filterRepository.deselectFilter() }
                    )
                }

                // Installed reference filters
                items(installedFilters) { filter ->
                    FilterItem(
                        title = filter.name,
                        previewPath = filter.previewPath,
                        isSelected = filter.id == selectedId,
                        onClick = { filterRepository.selectFilter(filter.id) }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // No-Face Policy Banner with One-Tap Retake (README Section 16 point 5)
            AnimatedVisibility(
                visible = noFaceRetakeMessage != null,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically()
            ) {
                noFaceRetakeMessage?.let { msg ->
                    Surface(
                        color = Color(0xE61E293B),
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .padding(horizontal = 24.dp, vertical = 8.dp)
                            .testTag("no_face_retake_banner")
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Text(
                                text = msg,
                                color = Color.White,
                                fontSize = 12.sp,
                                modifier = Modifier.weight(1f)
                            )
                            Button(
                                onClick = { triggerPhotoCapture() },
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0284C7)),
                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                                modifier = Modifier.testTag("one_tap_retake_button")
                            ) {
                                Text("Retake", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                }
            }

            AnimatedVisibility(
                visible = isRecordingVideo,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically()
            ) {
                val mins = recordingElapsedSec / 60
                val secs = recordingElapsedSec % 60
                Surface(
                    color = Color(0xD9DC2626),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier
                        .padding(bottom = 8.dp)
                        .testTag("video_recording_status_badge")
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(Color.White)
                        )
                        Text(
                            text = "REC %02d:%02d".format(mins, secs),
                            color = Color.White,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // Shutter & Video Record Row
            Box(
                modifier = Modifier.fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                // Primary Shutter trigger button (Tap = Photo, or Stop if recording video)
                Box(
                    modifier = Modifier
                        .size(76.dp)
                        .border(4.dp, if (isRecordingVideo) Color(0xFFEF4444) else Color.White, CircleShape)
                        .padding(6.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                isRecordingVideo -> Color(0xFFDC2626)
                                isCapturing -> Color.LightGray
                                else -> Color.White
                            }
                        )
                        .testTag("shutter_capture_button")
                        .clickable(enabled = !isCapturing) {
                            triggerPhotoCapture()
                        },
                    contentAlignment = Alignment.Center
                ) {
                    if (isCapturing) {
                        CircularProgressIndicator(
                            color = Color(0xFF0284C7),
                            modifier = Modifier.size(28.dp),
                            strokeWidth = 3.dp
                        )
                    } else if (isRecordingVideo) {
                        Box(
                            modifier = Modifier
                                .size(24.dp)
                                .clip(RoundedCornerShape(6.dp))
                                .background(Color.White)
                        )
                    }
                }

                // Dedicated Real-Time GPU Video Record / Stop Button (MP4 + AAC Audio)
                IconButton(
                    onClick = { triggerVideoRecordToggle() },
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .padding(end = 36.dp)
                        .size(52.dp)
                        .border(
                            width = 2.dp,
                            color = if (isRecordingVideo) Color(0xFFEF4444) else Color.White.copy(alpha = 0.85f),
                            shape = CircleShape
                        )
                        .background(
                            if (isRecordingVideo) Color(0xCCDC2626) else Color.Black.copy(alpha = 0.55f),
                            CircleShape
                        )
                        .testTag("video_record_toggle_button")
                ) {
                    Icon(
                        imageVector = if (isRecordingVideo) Icons.Default.Stop else Icons.Default.Videocam,
                        contentDescription = if (isRecordingVideo) "Stop Video Recording" else "Start Video Recording",
                        tint = Color.White
                    )
                }
            }
        }

        // Captured Photo Thumbnail Preview
        AnimatedVisibility(
            visible = lastCapturedPhoto != null,
            enter = fadeIn() + scaleIn(),
            exit = fadeOut() + scaleOut(),
            modifier = Modifier
                .align(Alignment.BottomStart)
                .navigationBarsPadding()
                .padding(start = 24.dp, bottom = 24.dp)
        ) {
            lastCapturedPhoto?.let { bmp ->
                Box(
                    modifier = Modifier
                        .size(60.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .border(2.dp, Color.White, RoundedCornerShape(8.dp))
                        .clickable {
                            isShowingPhotoPreview = true
                        }
                ) {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = "Last Photo",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
        }

        // Full-Screen Captured Photo Preview Modal
        if (isShowingPhotoPreview && lastCapturedPhoto != null) {
            androidx.activity.compose.BackHandler {
                isShowingPhotoPreview = false
            }
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.92f))
                    .clickable { isShowingPhotoPreview = false },
                contentAlignment = Alignment.Center
            ) {
                Image(
                    bitmap = lastCapturedPhoto!!.asImageBitmap(),
                    contentDescription = "Captured Photo Preview",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize()
                )
                IconButton(
                    onClick = { isShowingPhotoPreview = false },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .statusBarsPadding()
                        .padding(16.dp)
                        .size(44.dp)
                        .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Close Photo Preview",
                        tint = Color.White
                    )
                }
            }
        }
    }
}

/**
 * Non-overlapping Debug HUD Card rendered below `(a)` the top bar and `(b)` the debug toggle row,
 * bounded in height by `maxHeightDp` so it never covers the bottom carousel or shutter button.
 *
 * Uses the `~2 Hz` cached strings from [debugOverlay] (keyed by [hudTextRevision]) so layout and
 * text measurement do not run on every render frame.
 */
@Composable
fun DebugHudCard(
    debugOverlay: DebugOverlay,
    hudTextRevision: Int,
    sectionState: DebugHudSectionState,
    onSectionStateChange: (DebugHudSectionState) -> Unit,
    maxHeightDp: Dp,
    modifier: Modifier = Modifier
) {
    // Read ~2 Hz cached strings keyed by hudTextRevision
    val latencyLine = remember(hudTextRevision) { debugOverlay.cachedLatencyLine }
    val stageLine = remember(hudTextRevision) { debugOverlay.cachedStageLine }
    val convBreakdownLine = remember(hudTextRevision) { debugOverlay.cachedConvBreakdownLine }
    val pumpAndIdleLine = remember(hudTextRevision) { debugOverlay.cachedPumpAndIdleLine }
    val blendAndPrioLine = remember(hudTextRevision) { debugOverlay.cachedBlendAndPrioLine }
    val govWarmupLine = remember(hudTextRevision) { debugOverlay.cachedGovWarmupLine }
    val countersLine = remember(hudTextRevision) { debugOverlay.cachedCountersLine }
    val maskOffsetLine = remember(hudTextRevision) { debugOverlay.cachedMaskOffsetLine }
    val maskViewerLine = remember(hudTextRevision) { debugOverlay.cachedMaskViewerLine }
    val irisJitterLine = remember(hudTextRevision) { debugOverlay.cachedIrisJitterLine }
    val cameraLine = remember(hudTextRevision) { debugOverlay.cachedCameraLine }
    val analysisLine = remember(hudTextRevision) { debugOverlay.cachedFovLine }
    val projectionLine = remember(hudTextRevision) { debugOverlay.cachedProjectionHudLine }
    val lifecycleLine = remember(hudTextRevision) { debugOverlay.cachedLifecycleLine }
    val poseLine = remember(hudTextRevision) { debugOverlay.cachedPoseLine }
    val fwLine = remember(hudTextRevision) { debugOverlay.cachedFwLine }
    val jawOpen = remember(hudTextRevision) { debugOverlay.cachedJawOpen }
    val smile = remember(hudTextRevision) { debugOverlay.cachedSmile }
    val jawPercentText = remember(hudTextRevision) { debugOverlay.cachedJawOpenPercentText }
    val smilePercentText = remember(hudTextRevision) { debugOverlay.cachedSmilePercentText }

    val scrollState = rememberScrollState()

    Surface(
        color = Color(0xD90F172A), // Semi-opaque dark slate background for crisp text contrast
        shape = RoundedCornerShape(12.dp),
        border = androidx.compose.foundation.BorderStroke(1.dp, Color.White.copy(alpha = 0.22f)),
        modifier = modifier
            .fillMaxWidth()
            .heightIn(max = maxHeightDp)
            .testTag("debug_hud_card")
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(scrollState)
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Section 1: Latency (Default: Expanded)
            DebugHudSectionHeader(
                title = "Latency",
                isExpanded = sectionState.isLatencyExpanded,
                testTag = "hud_section_latency_header",
                onClick = { onSectionStateChange(sectionState.toggleLatency()) }
            )
            if (sectionState.isLatencyExpanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(start = 4.dp, bottom = 4.dp)
                        .testTag("hud_section_latency_content"),
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    Text(
                        text = latencyLine,
                        color = Color(0xFFFBBF24), // Amber highlight
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontWeight = FontWeight.SemiBold,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = stageLine,
                        color = Color.White,
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = convBreakdownLine,
                        color = Color(0xFF93C5FD),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = pumpAndIdleLine,
                        color = Color.White,
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = blendAndPrioLine,
                        color = Color(0xFFA7F3D0),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = govWarmupLine,
                        color = Color(0xFF38BDF8),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = countersLine,
                        color = Color(0xFFE2E8F0),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = maskOffsetLine,
                        color = Color(0xFFFCD34D),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = maskViewerLine,
                        color = Color(0xFF6EE7B7),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = irisJitterLine,
                        color = Color(0xFFC4B5FD),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                }
            }

            HorizontalDivider(color = Color.White.copy(alpha = 0.12f))

            // Section 2: Projection (Default: Collapsed)
            DebugHudSectionHeader(
                title = "Projection",
                isExpanded = sectionState.isProjectionExpanded,
                testTag = "hud_section_projection_header",
                onClick = { onSectionStateChange(sectionState.toggleProjection()) }
            )
            if (sectionState.isProjectionExpanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(start = 4.dp, bottom = 4.dp)
                        .testTag("hud_section_projection_content"),
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    Text(
                        text = cameraLine,
                        color = Color.White,
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = analysisLine,
                        color = Color.White,
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = projectionLine,
                        color = Color(0xFFFBBF24),
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontWeight = FontWeight.SemiBold,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                }
            }

            HorizontalDivider(color = Color.White.copy(alpha = 0.12f))

            // Section 3: Pose and blendshapes (Default: Collapsed)
            DebugHudSectionHeader(
                title = "Pose and blendshapes",
                isExpanded = sectionState.isPoseExpanded,
                testTag = "hud_section_pose_header",
                onClick = { onSectionStateChange(sectionState.togglePose()) }
            )
            if (sectionState.isPoseExpanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(start = 4.dp, bottom = 4.dp)
                        .testTag("hud_section_pose_content"),
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Text(
                        text = lifecycleLine,
                        color = Color.White,
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    Text(
                        text = "$poseLine • $fwLine",
                        color = Color.White,
                        fontSize = 11.sp,
                        lineHeight = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        softWrap = true
                    )
                    BlendshapeBarRow(label = "jawOpen", fraction = jawOpen, percentText = jawPercentText)
                    BlendshapeBarRow(label = "smile", fraction = smile, percentText = smilePercentText)
                }
            }
        }
    }
}

@Composable
private fun DebugHudSectionHeader(
    title: String,
    isExpanded: Boolean,
    testTag: String,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .clickable { onClick() }
            .padding(vertical = 4.dp, horizontal = 2.dp)
            .testTag(testTag),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            color = Color(0xFF38BDF8),
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = if (isExpanded) "▼" else "▶",
            color = Color.White.copy(alpha = 0.8f),
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun BlendshapeBarRow(
    label: String,
    fraction: Float,
    percentText: String
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = label,
            color = Color.White,
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(64.dp)
        )
        Box(
            modifier = Modifier
                .weight(1f)
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(Color.White.copy(alpha = 0.25f))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(fraction.coerceIn(0f, 1f))
                    .fillMaxHeight()
                    .background(Color(0xFF22C55E))
            )
        }
        Text(
            text = percentText,
            color = Color.White,
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(36.dp)
        )
    }
}

@Composable
fun FilterItem(
    title: String,
    previewPath: String?,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.clickable { onClick() }
    ) {
        Box(
            modifier = Modifier
                .size(64.dp)
                .clip(CircleShape)
                .background(Color(0xFF1E293B))
                .border(
                    width = if (isSelected) 3.dp else 1.5.dp,
                    color = if (isSelected) Color(0xFF38BDF8) else Color.White.copy(alpha = 0.4f),
                    shape = CircleShape
                ),
            contentAlignment = Alignment.Center
        ) {
            if (previewPath != null) {
                Image(
                    painter = rememberAsyncImagePainter(File(previewPath)),
                    contentDescription = title,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "No Filter",
                    tint = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.size(24.dp)
                )
            }
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = title,
            color = if (isSelected) Color(0xFF38BDF8) else Color.White.copy(alpha = 0.8f),
            fontSize = 11.sp,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
            maxLines = 1
        )
    }
}
