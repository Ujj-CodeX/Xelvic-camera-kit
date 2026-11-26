package com.example.camera

import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import com.example.engine.capture.CameraSession
import com.example.engine.capture.FrameClock
import com.example.engine.pkg.PackagePreloader
import com.example.engine.render.BaseCompositionSurface
import com.example.engine.render.XelvicGlSurfaceView

/**
 * Coordinates Activity / Compose lifecycle events (`ON_START`, `ON_RESUME`, `ON_PAUSE`, `ON_STOP`,
 * `ON_DESTROY`) and `DisposableEffect` disposal across `CameraController` / `CameraSession`,
 * `XelvicGlSurfaceView` / `BaseCompositionSurface`, and `PackagePreloader`.
 *
 * Lifecycle policy enforced:
 * 1. `ON_PAUSE` (short pause: dialog, multi-window, notification shade):
 *    - Stops `Choreographer` vsync callback and frame processing only (`CameraSession.isPaused = true`,
 *      `FaceGeometryProvider` enters IDLE, pending frames closed, in-flight results dropped).
 *    - Suspends `PackagePreloader` and `TextureCache` staging.
 *    - Does NOT tear down GL resources or pause the `GLSurfaceView` render thread (no black frame).
 * 2. `ON_STOP` (backgrounded):
 *    - Records stop timestamp; if `ON_STOP` lasts `>= 45 s`, closes `FaceGeometryProvider` and
 *      restarts it lazily on resume (`trackerStartCount` becomes 2 in that long-stop case).
 *    - Queues GL resource release on the GL thread BEFORE `GLSurfaceView.onPause()` (guarded by
 *      generation/allocated flag so double-free is impossible).
 * 3. `ON_START` / `ON_RESUME`:
 *    - Resets `FaceLifecycle` to `NO_FACE`, resets active `CompiledEffect` behavior/clip/trigger state,
 *      resets `SignalPipeline` filters and velocity, and resets `ClockOffsetCalibrator` + `LatencyProbe`.
 *    - Restores GL resources ONCE via the single idempotent entry point (`onSurfaceCreated` or lazy
 *      first draw in `renderFrame`) — never from both `onResume` and `onSurfaceCreated`.
 * 4. `ON_DESTROY` / `dispose()`:
 *    - Executes strict 5-step disposal order:
 *      `STOP_FRAME_DELIVERY` -> `UNBIND_CAMERAX` -> `RELEASE_GL_ON_GL_THREAD` -> `CLOSE_TRACKER` -> `SHUTDOWN_EXECUTORS`.
 */
class CameraLifecycleObserver(
    private val controllerTarget: ControllerLifecycleTarget,
    private val glTargetProvider: () -> GlLifecycleTarget?,
    private val preloader: PackagePreloader? = null,
    private val clockElapsedMs: () -> Long = { FrameClock.elapsedRealtimeNs() / 1_000_000L },
    private val actionLog: MutableList<String>? = null
) : LifecycleEventObserver {

    enum class DisposeStep {
        STOP_FRAME_DELIVERY,
        UNBIND_CAMERAX,
        RELEASE_GL_ON_GL_THREAD,
        CLOSE_TRACKER,
        SHUTDOWN_EXECUTORS
    }

    interface ControllerLifecycleTarget {
        fun onPause()
        fun onStop(nowElapsedMs: Long)
        fun onStart(nowElapsedMs: Long)
        fun onResume(nowElapsedMs: Long)
        fun stopFrameDelivery()
        fun unbindCameraX()
        fun closeTracker()
        fun shutdownExecutors()
    }

    interface GlLifecycleTarget {
        fun onPause()
        fun onStop()
        fun onStart()
        fun onResume()
        fun stopFrameDelivery()
        fun releaseGlOnGlThread()
    }

    private val _executedDisposeSteps = ArrayList<DisposeStep>(5)
    val executedDisposeSteps: List<DisposeStep>
        get() = _executedDisposeSteps

    @Volatile
    var isDisposed: Boolean = false
        private set

    override fun onStateChanged(source: LifecycleOwner, event: Lifecycle.Event) {
        if (isDisposed) return
        val nowMs = clockElapsedMs()
        when (event) {
            Lifecycle.Event.ON_START -> {
                actionLog?.add("ON_START")
                controllerTarget.onStart(nowMs)
                glTargetProvider()?.onStart()
            }
            Lifecycle.Event.ON_RESUME -> {
                actionLog?.add("ON_RESUME")
                preloader?.onResume()
                controllerTarget.onResume(nowMs)
                glTargetProvider()?.onResume()
            }
            Lifecycle.Event.ON_PAUSE -> {
                actionLog?.add("ON_PAUSE")
                preloader?.onPause()
                glTargetProvider()?.onPause()
                controllerTarget.onPause()
            }
            Lifecycle.Event.ON_STOP -> {
                actionLog?.add("ON_STOP")
                preloader?.onPause()
                // Release GL resources on GL thread before GLSurfaceView.onPause()
                glTargetProvider()?.onStop()
                controllerTarget.onStop(nowMs)
            }
            Lifecycle.Event.ON_DESTROY -> {
                dispose()
            }
            else -> Unit
        }
    }

    /**
     * Synchronizes a newly created `GlLifecycleTarget` with the current `Lifecycle.State` of the owner.
     */
    fun syncGlTargetWithCurrentState(state: Lifecycle.State) {
        if (isDisposed) return
        val gl = glTargetProvider() ?: return
        when {
            state.isAtLeast(Lifecycle.State.RESUMED) -> gl.onResume()
            state.isAtLeast(Lifecycle.State.STARTED) -> {
                gl.onStart()
                gl.onPause()
            }
            state == Lifecycle.State.CREATED -> gl.onStop()
            else -> Unit
        }
    }

    /**
     * Executes the mandatory 5-step disposal order exactly once:
     * 1. Stop frame delivery
     * 2. Unbind CameraX
     * 3. Release GL on the GL thread
     * 4. Close tracker
     * 5. Shut down executors
     */
    fun dispose() {
        if (isDisposed) return
        isDisposed = true

        // 1. Stop frame delivery
        _executedDisposeSteps.add(DisposeStep.STOP_FRAME_DELIVERY)
        actionLog?.add("stop_frame_delivery")
        preloader?.onPause()
        glTargetProvider()?.stopFrameDelivery()
        controllerTarget.stopFrameDelivery()

        // 2. Unbind CameraX
        _executedDisposeSteps.add(DisposeStep.UNBIND_CAMERAX)
        actionLog?.add("unbind_camerax")
        controllerTarget.unbindCameraX()

        // 3. Release GL on the GL thread
        _executedDisposeSteps.add(DisposeStep.RELEASE_GL_ON_GL_THREAD)
        actionLog?.add("release_gl_on_gl_thread")
        glTargetProvider()?.releaseGlOnGlThread()

        // 4. Close tracker
        _executedDisposeSteps.add(DisposeStep.CLOSE_TRACKER)
        actionLog?.add("close_tracker")
        controllerTarget.closeTracker()

        // 5. Shut down executors
        _executedDisposeSteps.add(DisposeStep.SHUTDOWN_EXECUTORS)
        actionLog?.add("shutdown_executors")
        controllerTarget.shutdownExecutors()
        preloader?.shutdown()
    }

    companion object {
        fun forCameraController(
            cameraController: CameraController,
            glSurfaceViewProvider: () -> XelvicGlSurfaceView?,
            preloader: PackagePreloader? = null,
            clockElapsedMs: () -> Long = { FrameClock.elapsedRealtimeNs() / 1_000_000L },
            actionLog: MutableList<String>? = null
        ): CameraLifecycleObserver {
            val ctrlTarget = object : ControllerLifecycleTarget {
                override fun onPause() = cameraController.onPause()
                override fun onStop(nowElapsedMs: Long) = cameraController.onStop(nowElapsedMs)
                override fun onStart(nowElapsedMs: Long) = cameraController.onStart(nowElapsedMs)
                override fun onResume(nowElapsedMs: Long) = cameraController.onResume(nowElapsedMs)
                override fun stopFrameDelivery() = cameraController.stopFrameDelivery()
                override fun unbindCameraX() = cameraController.unbindCameraX()
                override fun closeTracker() = cameraController.closeTracker()
                override fun shutdownExecutors() = cameraController.shutdownExecutors()
            }
            val glTarget = {
                glSurfaceViewProvider()?.let { view ->
                    object : GlLifecycleTarget {
                        override fun onPause() = view.onPause()
                        override fun onStop() = view.onStop()
                        override fun onStart() = view.onStart()
                        override fun onResume() = view.onResume()
                        override fun stopFrameDelivery() = view.stopFrameDelivery()
                        override fun releaseGlOnGlThread() = view.release()
                    }
                }
            }
            return CameraLifecycleObserver(
                controllerTarget = ctrlTarget,
                glTargetProvider = glTarget,
                preloader = preloader,
                clockElapsedMs = clockElapsedMs,
                actionLog = actionLog
            )
        }

        fun forSessionAndSurface(
            cameraSession: CameraSession,
            surface: BaseCompositionSurface,
            preloader: PackagePreloader? = null,
            onUnbindCameraX: (() -> Unit)? = null,
            onShutdownExecutors: (() -> Unit)? = null,
            clockElapsedMs: () -> Long = { FrameClock.elapsedRealtimeNs() / 1_000_000L },
            actionLog: MutableList<String>? = null
        ): CameraLifecycleObserver {
            val ctrlTarget = object : ControllerLifecycleTarget {
                override fun onPause() = cameraSession.onPause()
                override fun onStop(nowElapsedMs: Long) = cameraSession.onStop(nowElapsedMs)
                override fun onStart(nowElapsedMs: Long) = cameraSession.onStart(nowElapsedMs)
                override fun onResume(nowElapsedMs: Long) = cameraSession.onResume(nowElapsedMs)
                override fun stopFrameDelivery() = cameraSession.onPause()
                override fun unbindCameraX() {
                    onUnbindCameraX?.invoke()
                }
                override fun closeTracker() = cameraSession.stop()
                override fun shutdownExecutors() {
                    onShutdownExecutors?.invoke()
                }
            }
            val glTarget = object : GlLifecycleTarget {
                override fun onPause() {
                    preloader?.onPause(surface.renderGraph.textures)
                    surface.renderGraph.textures.onPause()
                    surface.onPause()
                }
                override fun onStop() {
                    preloader?.onPause(surface.renderGraph.textures)
                    surface.renderGraph.textures.onPause()
                    surface.onStop()
                }
                override fun onStart() = surface.onStart()
                override fun onResume() {
                    preloader?.onResume(surface.renderGraph.textures)
                    surface.renderGraph.textures.onResume()
                    surface.onResume()
                }
                override fun stopFrameDelivery() {
                    preloader?.onPause(surface.renderGraph.textures)
                    surface.renderGraph.textures.onPause()
                    surface.onPause()
                }
                override fun releaseGlOnGlThread() = surface.release()
            }
            return CameraLifecycleObserver(
                controllerTarget = ctrlTarget,
                glTargetProvider = { glTarget },
                preloader = preloader,
                clockElapsedMs = clockElapsedMs,
                actionLog = actionLog
            )
        }
    }
}
