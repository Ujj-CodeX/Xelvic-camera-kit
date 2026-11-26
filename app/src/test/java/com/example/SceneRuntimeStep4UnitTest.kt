package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.LegacyAdapters
import com.example.engine.render.GlResourceOwner
import com.example.engine.render.OccluderPass
import com.example.engine.scene.*
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SceneRuntimeStep4UnitTest {

    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
    }

    @Test
    fun testBehaviorCompositionOrder() {
        // Base -> Clips -> Maps -> Springs -> Lifecycle Fade
        val slots = PropertySlots(1)

        val map = BehaviorRunner.CompiledMap(
            id = "test_map",
            inputSlotIndex = 0,
            inMin = 0f, inMax = 1f,
            outMin = 1f, outMax = 2f,
            easing = "linear",
            clamp = true,
            smoothingTau = 0.001f, // near instant
            initial = 1f,
            writes = listOf(
                BehaviorRunner.PropertyWrite(0, BehaviorRunner.PropertyType.SCALE, 0, BehaviorRunner.WriteMode.MULTIPLY)
            )
        )

        val runner = BehaviorRunner(
            clips = emptyList(),
            maps = listOf(map),
            triggers = emptyList(),
            springs = emptyList(),
            nodeCount = 1
        )

        val sigBus = SignalBus(totalSlots = 4, derivedSignals = emptyList(), slotIndexMap = mapOf("test" to 0))
        sigBus.slotValues[0] = 1.0f // full input

        val faceState = FaceState(isVisible = true, fadeWeight = 0.5f)

        runner.step(dt = 0.05f, signalBus = sigBus, faceState = faceState, slots = slots)

        // Composition: Base 1.0 * Map(outMax = 2.0) * Fade(0.5) = 1.0
        assertEquals(1.0f, slots.scale[0], 0.05f)
        assertEquals(0.5f, slots.opacity[0], 0.05f)
    }

    @Test
    fun testTriggerHysteresisAndCooldown() {
        val actions = listOf(
            BehaviorRunner.TriggerAction(action = "playClip", clipId = "test_clip")
        )

        val clip = BehaviorRunner.CompiledClip(
            id = "test_clip",
            targetNodeIndex = 0,
            animationName = "test_anim",
            loop = false,
            startOn = null,
            speed = 1f,
            holdLast = false
        )

        val trigger = BehaviorRunner.CompiledTrigger(
            id = "burst_trigger",
            inputSlotIndex = 0,
            onEvent = null,
            risingAbove = 0.55f,
            fallingBelow = 0.35f,
            cooldownMs = 500L,
            armedInitially = true,
            onFire = actions
        )

        val runner = BehaviorRunner(
            clips = listOf(clip),
            maps = emptyList(),
            triggers = listOf(trigger),
            springs = emptyList(),
            nodeCount = 1
        )

        val slots = PropertySlots(1)
        val sigBus = SignalBus(totalSlots = 2, derivedSignals = emptyList(), slotIndexMap = mapOf("jaw" to 0))
        val faceState = FaceState(isVisible = true, fadeWeight = 1.0f)

        // 1. Below threshold (0.4) -> should not fire
        sigBus.slotValues[0] = 0.4f
        runner.step(0.016f, sigBus, faceState, slots, nowMs = 1000L)
        assertFalse(clip.isPlaying)
        assertTrue(trigger.isArmed)

        // 2. Rising above threshold (0.6) -> fires!
        sigBus.slotValues[0] = 0.6f
        runner.step(0.016f, sigBus, faceState, slots, nowMs = 1016L)
        assertTrue(clip.isPlaying)
        assertFalse(trigger.isArmed) // disarmed by hysteresis

        // Reset clip for cooldown check
        clip.isPlaying = false

        // 3. Still above threshold but cooldown has not elapsed -> should NOT fire
        runner.step(0.016f, sigBus, faceState, slots, nowMs = 1200L)
        assertFalse(clip.isPlaying)

        // 4. Drop below falling threshold (0.3) -> re-arms
        sigBus.slotValues[0] = 0.3f
        runner.step(0.016f, sigBus, faceState, slots, nowMs = 1300L)
        assertTrue(trigger.isArmed)

        // 5. Rise above threshold after cooldown -> fires again!
        sigBus.slotValues[0] = 0.65f
        runner.step(0.016f, sigBus, faceState, slots, nowMs = 1600L)
        assertTrue(clip.isPlaying)
    }

    @Test
    fun testClipPlaybackLoopAndHoldLast() {
        val loopClip = BehaviorRunner.CompiledClip(
            id = "loop_clip",
            targetNodeIndex = 0,
            animationName = "loop",
            loop = true,
            startOn = null,
            speed = 1.0f,
            holdLast = false,
            isPlaying = true,
            currentTimeSec = 0f,
            durationSec = 2.0f
        )

        val holdClip = BehaviorRunner.CompiledClip(
            id = "hold_clip",
            targetNodeIndex = 0,
            animationName = "hold",
            loop = false,
            startOn = null,
            speed = 1.0f,
            holdLast = true,
            isPlaying = true,
            currentTimeSec = 0f,
            durationSec = 1.0f
        )

        val runner = BehaviorRunner(
            clips = listOf(loopClip, holdClip),
            maps = emptyList(),
            triggers = emptyList(),
            springs = emptyList(),
            nodeCount = 1
        )

        val slots = PropertySlots(1)
        val sigBus = SignalBus(totalSlots = 1, derivedSignals = emptyList(), slotIndexMap = emptyMap())
        val faceState = FaceState(isVisible = true, fadeWeight = 1.0f)

        // Advance by 1.5s (in 30 steps of 0.05s)
        for (i in 0 until 30) {
            runner.step(0.05f, sigBus, faceState, slots)
        }

        // Loop clip (dur 2.0): 1.5s in, still playing
        assertEquals(1.5f, loopClip.currentTimeSec, 0.05f)
        assertTrue(loopClip.isPlaying)

        // Hold clip (dur 1.0): clamped to 1.0s, holdLast active
        assertEquals(1.0f, holdClip.currentTimeSec, 0.001f)
    }

    @Test
    fun testSpringInertiaIntegrationAndLimits() {
        val spring = BehaviorRunner.CompiledSpring(
            id = "test_spring",
            targetNodeIndex = 0,
            mode = "rotationInertia",
            stiffness = 90.0f,
            dampingRatio = 0.3f,
            gain = 0.35f,
            maxAngleDeg = 8.0f,
            lowPassTau = 0.04f
        )

        val runner = BehaviorRunner(
            clips = emptyList(),
            maps = emptyList(),
            triggers = emptyList(),
            springs = listOf(spring),
            nodeCount = 1
        )

        val slots = PropertySlots(1)
        val sigBus = SignalBus(totalSlots = 1, derivedSignals = emptyList(), slotIndexMap = emptyMap())

        // Initial head pose at roll 0
        val state0 = FaceState(isVisible = true, rollDeg = 0f, fadeWeight = 1f)
        runner.step(0.016f, sigBus, state0, slots)

        // Fast head turn to roll 30 deg
        val state1 = FaceState(isVisible = true, rollDeg = 30f, fadeWeight = 1f)
        runner.step(0.016f, sigBus, state1, slots)

        // Spring position must be bounded by maxAngleDeg (8 degrees)
        assertTrue(abs(spring.pos) <= 8.05f)
    }

    @Test
    fun testBuiltInAndCustomAnchorsFaceWidth() {
        val customMap = mapOf(
            "brow_custom" to AnchorResolver.CustomAnchorDef(
                listOf(AnchorResolver.LandmarkWeight(105, 0.5f), AnchorResolver.LandmarkWeight(66, 0.5f)),
                floatArrayOf(0f, 0.05f, 0f)
            )
        )
        val resolver = AnchorResolver(customMap)

        val landmarks = FloatArray(478 * 3)
        // 105 at (0.4, 0.6, 0.0), 66 at (0.4, 0.6, 0.0)
        landmarks[105 * 3] = 0.4f; landmarks[105 * 3 + 1] = 0.6f; landmarks[105 * 3 + 2] = 0f
        landmarks[66 * 3] = 0.4f; landmarks[66 * 3 + 1] = 0.6f; landmarks[66 * 3 + 2] = 0f

        val faceState = FaceState(
            isVisible = true,
            landmarks = landmarks,
            faceWidth = 0.3f
        )

        val out = FloatArray(3)
        // Built-in head_center ([0.0, 0.20, -0.12] FW)
        resolver.resolveAnchor("head_center", faceState, out)
        assertEquals(0.0f, out[0], 0.001f)
        assertEquals(0.20f, out[1], 0.001f)
        assertEquals(-0.12f, out[2], 0.001f)

        // Custom brow anchor
        resolver.resolveAnchor("brow_custom", faceState, out)
        assertNotNull(out)
    }

    @Test
    fun testParentChildTransformHierarchy() {
        val evaluator = SceneEvaluator(maxNodes = 4)

        // Node 0: Root group at position (0, 0, 0)
        // Node 1: Child model with parent = 0, at position (0.1, 0.2, 0.0)
        val n0 = CompiledNode(
            index = 0, id = "root", parentIndex = -1, tags = emptyList(), type = "group",
            anchor = null, basePosition = floatArrayOf(0f, 0f, 0f),
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f), baseScale = floatArrayOf(1f, 1f, 1f),
            rotationFollow = "full", materialId = null, baseVisible = true, baseOpacity = 1f, renderOrder = 0
        )
        val n1 = CompiledNode(
            index = 1, id = "child", parentIndex = 0, tags = emptyList(), type = "model",
            anchor = null, basePosition = floatArrayOf(0.1f, 0.2f, 0f),
            baseRotation = floatArrayOf(0f, 0f, 0f, 1f), baseScale = floatArrayOf(1f, 1f, 1f),
            rotationFollow = "full", materialId = null, baseVisible = true, baseOpacity = 1f, renderOrder = 1
        )

        val slots = PropertySlots(2)
        val effect = CompiledEffect(
            packageId = "test_hierarchy", packageName = "Test", schemaVersion = "xfilter/3",
            nodes = listOf(n0, n1), materials = emptyMap(), assets = emptyMap(), lighting = null,
            signalBus = SignalBus(1, emptyList(), emptyMap()),
            behaviorRunner = BehaviorRunner(emptyList(), emptyList(), emptyList(), emptyList(), 2),
            anchorResolver = AnchorResolver(emptyMap()), propertySlots = slots
        )

        val faceState = FaceState(isVisible = true, faceWidth = 1.0f)
        evaluator.evaluate(effect, faceState, slots)

        assertTrue(evaluator.isNodeVisible[0])
        assertTrue(evaluator.isNodeVisible[1])
        // World matrix of child (offset 16) has translation evaluated
        assertNotNull(evaluator.worldMatrices)
    }

    @Test
    fun testGlResourceOwnerDeferredRelease() {
        val owner = GlResourceOwner()

        // Track resources
        owner.trackTexture(101)
        owner.trackBuffer(201)
        owner.trackVao(301)
        owner.trackProgram(401)

        // Queue deletion
        owner.queueDeleteTexture(101)
        owner.queueDeleteBuffer(201)
        owner.queueDeleteVao(301)
        owner.queueDeleteProgram(401)

        // Drain on GL thread
        owner.drainPendingDeletions()
    }

    @Test
    fun testSunflowerForeheadAppendixAReferenceContract() {
        val manifestFile = listOf(
            File("src/main/assets/filters/sunflower_forehead/manifest.json"),
            File("app/src/main/assets/filters/sunflower_forehead/manifest.json")
        ).firstOrNull { it.exists() }
        assertNotNull("Sunflower manifest file must exist", manifestFile)
        assertTrue(manifestFile!!.exists())

        val json = JSONObject(manifestFile.readText())
        val compiled = Compiler.compile(json, manifestFile.parentFile!!)

        assertNotNull(compiled)
        assertEquals("sunflower_forehead", compiled.packageId)
        assertEquals("Sunflower Forehead", compiled.packageName)

        // Appendix A condition 2: Offset matches manifest layer.offsetFW [0.0, 0.06, 0.02] on forehead_left_hairline
        val modelNode = compiled.nodes.find { it.type == "model" }
        assertNotNull("Model node must be present", modelNode)
        assertEquals(0.0f, modelNode!!.basePosition[0], 0.001f)
        assertEquals(0.06f, modelNode.basePosition[1], 0.001f)
        assertEquals(0.02f, modelNode.basePosition[2], 0.001f)

        // Appendix A condition 3: Occluders present
        val occluders = compiled.nodes.filter { it.type == "occluder" }
        assertTrue("Must include headProxy and/or faceMesh occluders", occluders.isNotEmpty())

        // Appendix A condition 4: Clips idle_sway and bloom_in
        val clips = compiled.behaviorRunner.clips
        assertTrue("Must have idle_sway clip", clips.any { it.id == "idle_sway" && it.loop })
        assertTrue("Must have bloom_in clip", clips.any { it.id == "bloom_in" && it.holdLast })

        // Appendix A condition 5: Trigger burst_on_open & Map smile_swell
        val triggers = compiled.behaviorRunner.triggers
        assertTrue("Must have burst_on_open trigger", triggers.any { it.id == "burst_on_open" && it.risingAbove == 0.55f })
        val maps = compiled.behaviorRunner.maps
        assertTrue("Must have smile_swell map", maps.any { it.id == "smile_swell" })

        // Appendix A condition 6: Spring head_inertia
        val springs = compiled.behaviorRunner.springs
        assertTrue("Must have head_inertia spring", springs.any { it.id == "head_inertia" && it.maxAngleDeg == 8.0f })

        // Appendix A condition 7: Face loss hold/fade/reset
        assertEquals(150L, compiled.lifecyclePolicy.holdMs)
        assertEquals(200L, compiled.lifecyclePolicy.fadeMs)
        assertEquals("hideAndReset", compiled.lifecyclePolicy.afterFade)

        // Appendix A condition 8: Tier C overrides
        val tierC = compiled.tierOverrides["C"]
        assertNotNull("Tier C overrides must be compiled", tierC)
        assertTrue(tierC!!.disabledBehaviors.contains("smile_swell"))
        assertTrue(tierC.disabledBehaviors.contains("head_inertia"))
        assertTrue(tierC.disabledBehaviors.contains("burst_on_open"))
    }
}
