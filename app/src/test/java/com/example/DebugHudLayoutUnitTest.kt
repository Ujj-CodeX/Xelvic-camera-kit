package com.example

import com.example.engine.capture.FrameClock
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import com.example.ui.DebugHudLayoutCalculator
import com.example.ui.DebugHudSectionState
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for Chunk: Debug HUD Layout Fix (UI & Overlay only):
 * 1. Verifies that the HUD card bounds NEVER intersect the top bar (`XELVIC Tier X • NN fps` pill,
 *    `+`, `Info`, `FlipCamera`), the debug toggle row (`RES / FMT / DEL / PRED`), or the bottom
 *    controls (filter carousel + shutter + nav bar) across:
 *    - `360x800 dp` at `fontScale = 1.0f` and `fontScale = 1.3f`
 *    - `411x891 dp` at `fontScale = 1.0f` and `fontScale = 1.3f`
 *    - `842x1920 px` (`~320.76x731.43 dp` at `2.625` density) at `fontScale = 1.0f` and `1.3f`
 *    both in the default section state (`Latency` expanded, others collapsed) and when ALL 3
 *    sections (`Latency`, `Projection`, `Pose and blendshapes`) are expanded simultaneously.
 * 2. Verifies collapsible section defaults (`Latency = true`, `Projection = false`,
 *    `Pose and blendshapes = false`), independent tap toggling, and clean hide when debug is toggled off.
 * 3. Verifies `DebugOverlay` has `drawHudCardOnCanvas == false` by default (preventing the legacy
 *    `top = 130f` Canvas card overlap) and refreshes its cached HUD strings at `~2 Hz` (`500 ms`
 *    interval) with zero per-frame string formatting between refreshes.
 */
class DebugHudLayoutUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testHudBoundsDoNotIntersectTopBarDebugButtonsOrBottomControlsAcrossScreenSizesAndFontScales() {
        val screenConfigs = listOf(
            // (widthDp, heightDp, density, label)
            Triple(360f, 800f, 2.5f),
            Triple(411f, 891f, 2.625f),
            // Exact 842x1920 px phone from the screen recording (at 2.625x density)
            Triple(842f / 2.625f, 1920f / 2.625f, 2.625f)
        )
        val fontScales = listOf(1.0f, 1.3f)
        val sectionStates = listOf(
            // Default: Latency expanded, Projection & Pose collapsed
            DebugHudSectionState(isLatencyExpanded = true, isProjectionExpanded = false, isPoseExpanded = false),
            // All 3 sections expanded simultaneously (tests verticalScroll height clamping)
            DebugHudSectionState(isLatencyExpanded = true, isProjectionExpanded = true, isPoseExpanded = true)
        )

        for ((widthDp, heightDp, density) in screenConfigs) {
            for (fontScale in fontScales) {
                for (sections in sectionStates) {
                    val layout = DebugHudLayoutCalculator.measureScreenLayout(
                        widthDp = widthDp,
                        heightDp = heightDp,
                        density = density,
                        fontScale = fontScale,
                        isDebugEnabled = true,
                        sectionState = sections
                    )

                    val contextDesc = "screen=${widthDp}x${heightDp}dp, fontScale=$fontScale, sections=$sections"

                    // 1. Strict top-to-bottom vertical ordering: (a) topBar -> (b) debugToggleRow -> (c) hudCard -> bottomControls
                    assertTrue(
                        "debugToggleRow must start below topBar ($contextDesc)",
                        layout.debugToggleRowBounds.top >= layout.topBarBounds.bottom
                    )
                    assertTrue(
                        "hudCard must start below both topBar and debugToggleRow ($contextDesc)",
                        layout.hudCardBounds.top >= layout.topBarBounds.bottom &&
                            layout.hudCardBounds.top >= layout.debugToggleRowBounds.bottom
                    )
                    assertTrue(
                        "hudCard bottom (${layout.hudCardBounds.bottom}) must stay above bottomControls top (${layout.bottomControlsBounds.top}) ($contextDesc)",
                        layout.hudCardBounds.bottom <= layout.bottomControlsBounds.top
                    )

                    // 2. Zero bounding-box intersection between HUD card and any control region
                    assertFalse(
                        "HUD card must NOT intersect top bar ($contextDesc)",
                        layout.hudCardBounds.intersects(layout.topBarBounds)
                    )
                    assertFalse(
                        "HUD card must NOT intersect debug toggle button row ($contextDesc)",
                        layout.hudCardBounds.intersects(layout.debugToggleRowBounds)
                    )
                    assertFalse(
                        "HUD card must NOT intersect bottom filter carousel / shutter controls ($contextDesc)",
                        layout.hudCardBounds.intersects(layout.bottomControlsBounds)
                    )
                    assertFalse(
                        "Top bar and debug toggle button row must NOT intersect ($contextDesc)",
                        layout.topBarBounds.intersects(layout.debugToggleRowBounds)
                    )

                    // 3. HUD card must remain horizontally inside the screen viewport (no right-edge overflow)
                    val screenWidthPx = widthDp * density
                    assertTrue(layout.hudCardBounds.left >= 0f)
                    assertTrue(
                        "HUD card right edge (${layout.hudCardBounds.right}) must fit within screenWidthPx ($screenWidthPx) ($contextDesc)",
                        layout.hudCardBounds.right <= screenWidthPx + 0.5f
                    )
                    assertTrue(
                        "HUD card height must be positive when debug is enabled ($contextDesc)",
                        layout.hudCardBounds.height > 0f
                    )
                }
            }
        }
    }

    @Test
    fun testMeasuredBoundsClampingAndScrollableFallbackOn842x1920Phone() {
        // Simulate measured onGloballyPositioned coordinates from an 842x1920 px phone
        val screenW = 842f
        val screenH = 1920f
        val topBar = DebugHudLayoutCalculator.HudRect(left = 42f, top = 96f, right = 800f, bottom = 212f)
        val debugRow = DebugHudLayoutCalculator.HudRect(left = 42f, top = 228f, right = 800f, bottom = 312f)
        val bottomControls = DebugHudLayoutCalculator.HudRect(left = 0f, top = 1320f, right = 842f, bottom = 1920f)

        // Case A: Default collapsed sections (desired content height = 240 px < free space)
        val defaultLayout = DebugHudLayoutCalculator.computeHudBounds(
            screenWidthPx = screenW,
            screenHeightPx = screenH,
            topBarBounds = topBar,
            debugToggleRowBounds = debugRow,
            bottomControlsBounds = bottomControls,
            horizontalPaddingPx = 42f,
            verticalSpacingPx = 21f,
            desiredHudContentHeightPx = 240f
        )
        assertEquals(312f + 21f, defaultLayout.hudCardBounds.top, 0.01f)
        assertEquals(333f + 240f, defaultLayout.hudCardBounds.bottom, 0.01f)
        assertFalse(defaultLayout.isHudScrollable)
        assertFalse(defaultLayout.hudCardBounds.intersects(topBar))
        assertFalse(defaultLayout.hudCardBounds.intersects(debugRow))
        assertFalse(defaultLayout.hudCardBounds.intersects(bottomControls))

        // Case B: Tall content (desired content height = 1400 px > free space between 333 px and 1299 px)
        val tallLayout = DebugHudLayoutCalculator.computeHudBounds(
            screenWidthPx = screenW,
            screenHeightPx = screenH,
            topBarBounds = topBar,
            debugToggleRowBounds = debugRow,
            bottomControlsBounds = bottomControls,
            horizontalPaddingPx = 42f,
            verticalSpacingPx = 21f,
            desiredHudContentHeightPx = 1400f
        )
        assertEquals(333f, tallLayout.hudCardBounds.top, 0.01f)
        assertEquals("HUD bottom must be clamped to bottomControls.top - verticalSpacingPx", 1320f - 21f, tallLayout.hudCardBounds.bottom, 0.01f)
        assertTrue("Tall HUD content must enable vertical scrolling when clamped", tallLayout.isHudScrollable)
        assertFalse(tallLayout.hudCardBounds.intersects(bottomControls))
    }

    @Test
    fun testCollapsibleSectionsDefaultStateAndCleanInfoButtonToggle() {
        // 1. Default section state: Latency expanded, Projection and Pose/Blendshapes collapsed
        var state = DebugHudSectionState()
        assertTrue("Latency section must default to expanded", state.isLatencyExpanded)
        assertFalse("Projection section must default to collapsed", state.isProjectionExpanded)
        assertFalse("Pose and blendshapes section must default to collapsed", state.isPoseExpanded)

        // 2. Tap Projection header -> expands Projection without collapsing Latency
        state = state.toggleProjection()
        assertTrue(state.isLatencyExpanded)
        assertTrue(state.isProjectionExpanded)
        assertFalse(state.isPoseExpanded)

        // 3. Tap Pose header -> expands Pose
        state = state.togglePose()
        assertTrue(state.isPoseExpanded)

        // 4. Tap Latency header -> collapses Latency
        state = state.toggleLatency()
        assertFalse(state.isLatencyExpanded)

        // 5. Info button toggle off -> HUD card height becomes 0 and does not overlap anything
        val disabledLayout = DebugHudLayoutCalculator.measureScreenLayout(
            widthDp = 360f,
            heightDp = 800f,
            density = 2.5f,
            fontScale = 1.3f,
            isDebugEnabled = false,
            sectionState = state
        )
        assertEquals("When debug is toggled off, HUD card height must be 0", 0f, disabledLayout.hudCardBounds.height, 0.001f)
        assertEquals("When debug is toggled off, debug toggle row height must be 0", 0f, disabledLayout.debugToggleRowBounds.height, 0.001f)
    }

    @Test
    fun testDebugOverlay2HzCachedStringsAndNoCanvasCardByDefault() {
        var fakeNowNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeNowNs }

        val probe = LatencyProbe()
        probe.recordDepthBits(24)
        probe.recordGovernorTransition("demote:frameP95=38.2ms")
        val overlay = DebugOverlay(latencyProbe = probe)

        assertFalse(
            "DebugOverlay must default to drawHudCardOnCanvas = false so the Canvas card does not overlap top controls",
            overlay.drawHudCardOnCanvas
        )

        val faceState = FaceState(
            timestampNs = fakeNowNs,
            poseTimestampNs = fakeNowNs - 30_000_000L,
            lifecycle = FaceLifecycle.State.TRACKING,
            isVisible = true,
            fadeWeight = 1.0f,
            faceWidth = 0.145f,
            yawDeg = 5.2f,
            pitchDeg = -3.1f,
            rollDeg = 1.4f,
            imageWidth = 640,
            imageHeight = 480,
            blendshapes = mapOf("jawOpen" to 0.42f, "mouthSmileLeft" to 0.60f, "mouthSmileRight" to 0.80f),
            projectionParams = ProjectionParams(
                fovYDeg = 54.0f,
                fovXDeg = 42.0f,
                cropAspectRatio = 480f / 640f,
                focalLengthMm = 3.85f,
                scalerCropLeft = 0,
                scalerCropTop = 0,
                scalerCropWidth = 4000,
                scalerCropHeight = 3000,
                isCalibratedFromIntrinsics = true
            )
        )

        // First call at t = 1.0s refreshes all cached strings and increments hudTextRevision to 1
        val refreshed1 = overlay.refreshCachedHudStringsIfNeeded(faceState, fps = 59.8f, tier = "A")
        assertTrue(refreshed1)
        assertEquals(1, overlay.hudTextRevision)
        assertTrue(overlay.cachedGovWarmupLine.contains("demote:frameP95=38.2ms"))
        assertTrue(overlay.cachedProjectionHudLine.contains("depth=24b"))
        assertEquals("42%", overlay.cachedJawOpenPercentText)
        assertEquals("70%", overlay.cachedSmilePercentText)

        // Subsequent frame 16ms later (< 500ms refresh interval) must NOT reformat strings or increment hudTextRevision
        fakeNowNs += 16_666_667L
        val modifiedState = faceState.copy(yawDeg = 25.0f)
        val refreshedEarly = overlay.refreshCachedHudStringsIfNeeded(modifiedState, fps = 45.0f, tier = "B")
        assertFalse("HUD strings must not reformat inside 500ms (~2 Hz) window", refreshedEarly)
        assertEquals(1, overlay.hudTextRevision)
        assertTrue("Pose line must remain cached until 500ms expires", overlay.cachedPoseLine.contains("Y:5.2"))

        // Advance past 500ms (~2 Hz interval) -> refreshes cached strings and increments hudTextRevision to 2
        fakeNowNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        val refreshedAfter500ms = overlay.refreshCachedHudStringsIfNeeded(modifiedState, fps = 45.0f, tier = "B")
        assertTrue(refreshedAfter500ms)
        assertEquals(2, overlay.hudTextRevision)
        assertTrue(overlay.cachedPoseLine.contains("Y:25.0"))
        assertTrue(overlay.cachedCameraLine.contains("Tier B"))
    }

    @Test
    fun testSevenDebugTogglesWrapOn360x800DpAndHudStartsBelowLastRow() {
        val density = 2.5f
        val widthDp = 360f
        val heightDp = 800f
        val screenWidthPx = widthDp * density

        for (fontScale in listOf(1.0f, 1.3f)) {
            val layout = DebugHudLayoutCalculator.measureScreenLayout(
                widthDp = widthDp,
                heightDp = heightDp,
                density = density,
                fontScale = fontScale,
                isDebugEnabled = true,
                sectionState = DebugHudSectionState(
                    isLatencyExpanded = true,
                    isProjectionExpanded = false,
                    isPoseExpanded = false
                )
            )

            val singleRowPillHeightPx = 30f * density * fontScale.coerceAtLeast(1f)
            assertTrue(
                "7 debug toggles (RES FMT DEL PRED BLEND PRIO RPRIO) must wrap onto >= 2 rows on 360dp (fontScale=$fontScale), got height=${layout.debugToggleRowBounds.height} vs singleRow=$singleRowPillHeightPx",
                layout.debugToggleRowBounds.height > singleRowPillHeightPx * 1.5f
            )
            assertTrue(
                "Wrapped debug toggle row must not clip horizontally on 360dp (right=${layout.debugToggleRowBounds.right} <= $screenWidthPx)",
                layout.debugToggleRowBounds.right <= screenWidthPx + 0.5f
            )
            assertTrue(
                "HUD card top (${layout.hudCardBounds.top}) must start below the last wrapped row of debug toggles (${layout.debugToggleRowBounds.bottom}) on 360x800dp (fontScale=$fontScale)",
                layout.hudCardBounds.top >= layout.debugToggleRowBounds.bottom
            )
            assertFalse(
                "HUD card must not intersect wrapped debug toggle row on 360x800dp",
                layout.hudCardBounds.intersects(layout.debugToggleRowBounds)
            )
            assertFalse(
                "HUD card must not intersect bottom controls on 360x800dp",
                layout.hudCardBounds.intersects(layout.bottomControlsBounds)
            )
        }
    }
}

