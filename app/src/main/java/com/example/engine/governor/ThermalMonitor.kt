package com.example.engine.governor

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executor

/**
 * ThermalMonitor: Integrates Android platform thermal status (PowerManager) and memory
 * pressure (ComponentCallbacks2 / ActivityManager) with PerformanceGovernor and TextureCache
 * (README Section 4, 13 & 21 Step 6).
 *
 * Uses official Android PowerManager.THERMAL_STATUS_* and ComponentCallbacks2.TRIM_MEMORY_*
 * constants without inventing non-standard classifications.
 */
class ThermalMonitor(private val context: Context? = null) : ComponentCallbacks2 {

    companion object {
        private const val TAG = "ThermalMonitor"

        // Mirror standard Android PowerManager thermal constants for API-independent access
        const val THERMAL_STATUS_NONE = PowerManager.THERMAL_STATUS_NONE
        const val THERMAL_STATUS_LIGHT = PowerManager.THERMAL_STATUS_LIGHT
        const val THERMAL_STATUS_MODERATE = PowerManager.THERMAL_STATUS_MODERATE
        const val THERMAL_STATUS_SEVERE = PowerManager.THERMAL_STATUS_SEVERE
        const val THERMAL_STATUS_CRITICAL = PowerManager.THERMAL_STATUS_CRITICAL
        const val THERMAL_STATUS_EMERGENCY = PowerManager.THERMAL_STATUS_EMERGENCY
        const val THERMAL_STATUS_SHUTDOWN = PowerManager.THERMAL_STATUS_SHUTDOWN

        /**
         * Classifies whether the current physical device is low-end based on RAM and CPU cores.
         * Low-end devices start at the lowest tier (C) and promote after headroom;
         * high-end devices start at the highest tier (A).
         */
        fun detectLowEndDevice(context: Context?): Boolean {
            if (context == null) return false
            return try {
                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
                if (am != null) {
                    if (am.isLowRamDevice) return true
                    val memInfo = ActivityManager.MemoryInfo()
                    am.getMemoryInfo(memInfo)
                    val totalRamBytes = memInfo.totalMem
                    // Treat devices with <= 3.5 GB RAM or <= 4 cores as low-end
                    if (totalRamBytes in 1..(3_600_000_000L)) return true
                }
                val cores = Runtime.getRuntime().availableProcessors()
                cores <= 4
            } catch (e: Exception) {
                false
            }
        }
    }

    interface Listener {
        fun onThermalStatusChanged(thermalStatus: Int)
        fun onMemoryPressure(trimLevel: Int, isLowMemory: Boolean)
    }

    @Volatile
    var currentThermalStatus: Int = THERMAL_STATUS_NONE
        private set

    @Volatile
    var lastTrimMemoryLevel: Int = 0
        private set

    @Volatile
    var isMemoryPressured: Boolean = false
        private set

    @Volatile
    var lastThermalHeadroom: Float = Float.NaN
        private set

    private var lastHeadroomQueryElapsedMs: Long = 0L

    private val listeners = CopyOnWriteArrayList<Listener>()
    private var thermalListenerObj: Any? = null
    private var isRegistered = false

    fun addListener(listener: Listener) {
        if (!listeners.contains(listener)) {
            listeners.add(listener)
        }
    }

    fun removeListener(listener: Listener) {
        listeners.remove(listener)
    }

    /**
     * Formats [status] as a compact human-readable label with numeric code for the debug HUD.
     */
    @JvmOverloads
    fun formatThermalStatusLabel(status: Int = currentThermalStatus): String {
        return when (status) {
            THERMAL_STATUS_NONE -> "NONE(0)"
            THERMAL_STATUS_LIGHT -> "LIGHT(1)"
            THERMAL_STATUS_MODERATE -> "MODERATE(2)"
            THERMAL_STATUS_SEVERE -> "SEVERE(3)"
            THERMAL_STATUS_CRITICAL -> "CRITICAL(4)"
            THERMAL_STATUS_EMERGENCY -> "EMERGENCY(5)"
            THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN(6)"
            else -> "UNKNOWN($status)"
        }
    }

    /**
     * Updates cached thermal headroom directly (used by deterministic unit tests).
     */
    fun reportThermalHeadroom(headroom: Float, nowElapsedMs: Long = System.currentTimeMillis()) {
        lastThermalHeadroom = headroom
        lastHeadroomQueryElapsedMs = nowElapsedMs
    }

    /**
     * Queries `PowerManager.getThermalHeadroom(forecastSeconds)` on API 30+ at most once per
     * `1,000 ms` (Android platform rate limit for `getThermalHeadroom`), caching and returning
     * [lastThermalHeadroom] (`Float.NaN` if unsupported or unavailable).
     */
    @JvmOverloads
    fun queryThermalHeadroomIfNeeded(
        nowElapsedMs: Long = System.currentTimeMillis(),
        forecastSeconds: Int = 1
    ): Float {
        if (lastHeadroomQueryElapsedMs > 0L && (nowElapsedMs - lastHeadroomQueryElapsedMs) < 1_000L) {
            return lastThermalHeadroom
        }
        lastHeadroomQueryElapsedMs = nowElapsedMs
        val ctx = context ?: return lastThermalHeadroom
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                val pm = ctx.getSystemService(Context.POWER_SERVICE) as? PowerManager
                if (pm != null) {
                    val hr = pm.getThermalHeadroom(forecastSeconds)
                    if (hr.isFinite() && hr >= 0f) {
                        lastThermalHeadroom = hr
                    }
                }
            } catch (_: Throwable) {
            }
        }
        return lastThermalHeadroom
    }

    /**
     * Starts listening to Android PowerManager thermal events and ComponentCallbacks2 memory callbacks.
     */
    fun start() {
        if (isRegistered || context == null) return
        try {
            context.applicationContext?.registerComponentCallbacks(this) ?: context.registerComponentCallbacks(this)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
                if (pm != null) {
                    currentThermalStatus = pm.currentThermalStatus
                    val listener = PowerManager.OnThermalStatusChangedListener { status ->
                        reportThermalStatus(status)
                    }
                    val directExecutor = Executor { command -> command.run() }
                    pm.addThermalStatusListener(directExecutor, listener)
                    thermalListenerObj = listener
                }
            }
            isRegistered = true
        } catch (e: Exception) {
            Log.w(TAG, "Could not register platform thermal/memory listeners: ${e.message}")
        }
    }

    /**
     * Unregisters platform thermal and memory listeners.
     */
    fun stop() {
        if (!isRegistered || context == null) return
        try {
            context.applicationContext?.unregisterComponentCallbacks(this) ?: context.unregisterComponentCallbacks(this)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
                val listener = thermalListenerObj as? PowerManager.OnThermalStatusChangedListener
                if (pm != null && listener != null) {
                    pm.removeThermalStatusListener(listener)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error unregistering thermal/memory listeners: ${e.message}")
        } finally {
            thermalListenerObj = null
            isRegistered = false
        }
    }

    /**
     * Updates thermal status (invoked by Android PowerManager or deterministic tests).
     */
    fun reportThermalStatus(status: Int) {
        val clamped = status.coerceIn(THERMAL_STATUS_NONE, THERMAL_STATUS_SHUTDOWN)
        currentThermalStatus = clamped
        for (l in listeners) {
            l.onThermalStatusChanged(clamped)
        }
    }

    /**
     * Updates memory pressure state (invoked by ComponentCallbacks2 or deterministic tests).
     */
    fun reportMemoryTrim(level: Int, lowMemory: Boolean = false) {
        lastTrimMemoryLevel = level
        val pressured = lowMemory ||
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW ||
            level == ComponentCallbacks2.TRIM_MEMORY_COMPLETE ||
            level == ComponentCallbacks2.TRIM_MEMORY_MODERATE
        isMemoryPressured = pressured
        for (l in listeners) {
            l.onMemoryPressure(level, pressured)
        }
    }

    fun clearMemoryPressure() {
        lastTrimMemoryLevel = 0
        isMemoryPressured = false
    }

    /**
     * Returns true when platform thermal status indicates throttling (>= THERMAL_STATUS_MODERATE).
     */
    fun isThermalElevated(): Boolean {
        return currentThermalStatus >= THERMAL_STATUS_MODERATE
    }

    /**
     * Returns true when platform thermal status is severe or critical (>= THERMAL_STATUS_SEVERE).
     */
    fun isThermalSevere(): Boolean {
        return currentThermalStatus >= THERMAL_STATUS_SEVERE
    }

    override fun onTrimMemory(level: Int) {
        reportMemoryTrim(level, lowMemory = false)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        // No-op
    }

    override fun onLowMemory() {
        reportMemoryTrim(ComponentCallbacks2.TRIM_MEMORY_COMPLETE, lowMemory = true)
    }
}
