package com.example

import com.example.engine.governor.PerformanceGovernor
import com.example.engine.pkg.CapabilityRegistry
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.LegacyAdapters
import com.example.engine.pkg.PackageValidator
import com.example.engine.scene.*
import com.example.engine.signal.*
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class XelvicEngineUnitTest {

    @Test
    fun testPoseDecompositionAndRecomposition() {
        val t = floatArrayOf(0.1f, -0.2f, 0.5f)
        val q = floatArrayOf(0f, 0f, 0.7071f, 0.7071f) // 90-degree Z rotation
        val s = floatArrayOf(1.0f, 1.0f, 1.0f)

        val matrix = PoseDecomposer.recompose(t, q, s)
        assertNotNull(matrix)
        assertEquals(16, matrix.size)

        val decomposed = PoseDecomposer.decompose(matrix)
        assertEquals(t[0], decomposed.translation[0], 0.001f)
        assertEquals(t[1], decomposed.translation[1], 0.001f)
        assertEquals(t[2], decomposed.translation[2], 0.001f)

        assertEquals(s[0], decomposed.scale[0], 0.01f)
        assertEquals(s[1], decomposed.scale[1], 0.01f)
        assertEquals(s[2], decomposed.scale[2], 0.01f)
    }

    @Test
    fun testOneEuroFilterAndQuaternionSignContinuity() {
        val qFilter = OneEuroFilterQuaternion()
        val t0 = 1_000_000_000L
        val t1 = 1_033_000_000L

        // Initial orientation
        val out1 = qFilter.filter(0f, 0f, 0f, 1f, t0)
        assertEquals(1f, out1[3], 0.01f)

        // Opposite hemisphere representation of same orientation: [0, 0, 0, -1]
        // Sign continuity must flip it to maintain smooth continuity
        val out2 = qFilter.filter(0f, 0f, 0f, -1f, t1)
        assertTrue("Quaternion sign continuity should preserve positive w", out2[3] > 0.5f)
    }

    @Test
    fun testFaceLifecycleTransitions() {
        val lifecycle = FaceLifecycle(holdMs = 150L, fadeMs = 200L, easeInMs = 50L)

        // Initial state
        assertEquals(FaceLifecycle.State.NO_FACE, lifecycle.currentState)
        assertEquals(0f, lifecycle.fadeWeight, 0.001f)

        // 1. Face detected
        val ev1 = lifecycle.update(hasFace = true, nowNs = 1_000_000_000L)
        assertEquals("faceAcquired", ev1)
        assertEquals(FaceLifecycle.State.ACQUIRING, lifecycle.currentState)

        // After easeInMs (50ms = 50,000,000ns)
        lifecycle.update(hasFace = true, nowNs = 1_060_000_000L)
        assertEquals(FaceLifecycle.State.TRACKING, lifecycle.currentState)
        assertEquals(1.0f, lifecycle.fadeWeight, 0.001f)

        // 2. Face lost -> enters HOLD
        lifecycle.update(hasFace = false, nowNs = 2_000_000_000L)
        assertEquals(FaceLifecycle.State.HOLD, lifecycle.currentState)
        assertEquals(1.0f, lifecycle.fadeWeight, 0.001f)

        // After holdMs (150ms = 150,000,000ns) -> enters FADING
        lifecycle.update(hasFace = false, nowNs = 2_160_000_000L)
        assertEquals(FaceLifecycle.State.FADING, lifecycle.currentState)
        assertTrue(lifecycle.fadeWeight > 0f)

        // 3. Re-acquisition cancels fade!
        lifecycle.update(hasFace = true, nowNs = 2_200_000_000L)
        assertEquals(FaceLifecycle.State.TRACKING, lifecycle.currentState)
        assertEquals(1.0f, lifecycle.fadeWeight, 0.001f)
    }

    @Test
    fun testFacePoseAdapterFaceWidth() {
        // Landmarks buffer with cheek points 234 and 454
        val lm = FloatArray(478 * 3)
        // 234 (cheek right) at x = 0.35, 454 (cheek left) at x = 0.65
        lm[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3] = 0.35f
        lm[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3] = 0.65f

        val fw = FacePoseAdapter.computeFaceWidth(lm)
        assertEquals(0.30f, fw, 0.001f)
    }

    @Test
    fun testExpressionVMMathAndClamping() {
        val vm = ExpressionVM()
        val slots = floatArrayOf(0.4f, 0.8f, 10f)

        // Test smoothstep(0.25, 0.75, slot[0])
        val prog1 = vm.compile("smoothstep(0.25, 0.75, a)") { 0 }
        val res1 = vm.execute(prog1, slots)
        assertTrue(res1 in 0f..1f)

        // Test division by zero clamps to 0 (no NaN/Infinity)
        val prog2 = vm.compile("10 / 0") { 0 }
        val res2 = vm.execute(prog2, slots)
        assertEquals(0f, res2, 0.0001f)

        // Test mean(a, b)
        val prog3 = vm.compile("mean(a, b)") { if (it == "a") 0 else 1 }
        val res3 = vm.execute(prog3, slots)
        assertEquals(0.6f, res3, 0.001f)
    }

    @Test
    fun testBehaviorRunnerCompositionOrder() {
        // Compose Order: Base -> Clips -> Maps -> Springs -> Lifecycle fade
        val slots = PropertySlots(1)
        slots.scale[0] = 1f; slots.scale[1] = 1f; slots.scale[2] = 1f

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

        // Base 1.0 * Map(outMax = 2.0) * Fade(0.5) = 1.0
        assertEquals(1.0f, slots.scale[0], 0.05f)
    }

    @Test
    fun testCapabilityNegotiation() {
        val supported = listOf("node.sprite@1", "node.model@1", "material.lambert@1")
        val (ok, missing) = CapabilityRegistry.negotiate(supported)
        assertTrue(ok)
        assertTrue(missing.isEmpty())

        val unsupported = listOf("node.sprite@1", "unknown_future_node@99")
        val (ok2, missing2) = CapabilityRegistry.negotiate(unsupported)
        assertFalse(ok2)
        assertEquals(1, missing2.size)
        assertEquals("unknown_future_node@99", missing2[0])
    }

    @Test
    fun testLegacyAdapterV1ToXfilter3() {
        val v1 = JSONObject()
        v1.put("id", "vintage_crown")
        v1.put("schema", "v1")
        val layers = JSONArray()
        val layer0 = JSONObject()
        layer0.put("asset", "crown.png")
        layer0.put("anchor", "FOREHEAD")
        layer0.put("scale", 0.6)
        layers.put(layer0)
        v1.put("layers", layers)

        val adapted = LegacyAdapters.adapt(v1)
        assertEquals("xfilter/3", adapted.getString("schema"))
        val nodes = adapted.getJSONArray("nodes")
        assertEquals(1, nodes.length())
        val node0 = nodes.getJSONObject(0)
        assertEquals("sprite", node0.getString("type"))
        assertEquals("forehead", node0.getString("anchor"))
    }

    @Test
    fun testLegacyAdapterFace3dToXfilter3() {
        val face3d = JSONObject()
        face3d.put("schemaVersion", 1)
        face3d.put("id", "sunflower_forehead")
        face3d.put("name", "Sunflower Forehead")
        face3d.put("engineContract", "xfilter-face-3d/1")

        val assets = JSONArray()
        val a0 = JSONObject()
        a0.put("id", "sunflower_forehead_model")
        a0.put("path", "assets/sunflower_forehead.glb")
        a0.put("mediaType", "model/gltf-binary")
        assets.put(a0)
        face3d.put("assets", assets)

        val layer = JSONObject()
        layer.put("id", "sunflower_forehead")
        layer.put("asset", "sunflower_forehead_model")
        layer.put("anchor", "head_center")
        val off = JSONArray()
        off.put(-0.4); off.put(0.42); off.put(0.64)
        layer.put("offsetFW", off)
        layer.put("scale", 1.0)
        face3d.put("layer", layer)

        val behaviors = JSONObject()
        val clips = JSONArray()
        val c0 = JSONObject()
        c0.put("id", "idle_sway")
        c0.put("loop", true)
        clips.put(c0)
        behaviors.put("clips", clips)
        face3d.put("behaviors", behaviors)

        val adapted = LegacyAdapters.adapt(face3d)
        assertEquals("xfilter/3", adapted.getString("schema"))
        assertEquals("sunflower_forehead", adapted.getString("id"))

        // Validate nodes
        val nodes = adapted.getJSONArray("nodes")
        assertTrue("Should contain model node", nodes.length() >= 1)
        var foundModel = false
        for (i in 0 until nodes.length()) {
            val node = nodes.getJSONObject(i)
            if (node.getString("type") == "model") {
                foundModel = true
                assertEquals("head_center", node.getString("anchor"))
                assertEquals("sunflower_forehead_model", node.getString("asset"))
            }
        }
        assertTrue("Model node must be present", foundModel)

        // Validate compilation through Compiler
        val compiled = Compiler.compile(adapted, File("app/src/main/assets/filters/sunflower_forehead"))
        assertNotNull(compiled)
        assertEquals("sunflower_forehead", compiled.packageId)
        assertTrue(compiled.nodes.isNotEmpty())
    }
}
