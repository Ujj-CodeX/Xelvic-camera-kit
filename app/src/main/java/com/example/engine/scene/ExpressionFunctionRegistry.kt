package com.example.engine.scene

/**
 * ExpressionFunctionRegistry: Single source of truth for allowable expression functions.
 * Section 5.10 & 7: min, max, clamp, abs, mean, mix, smoothstep, step, sin, cos, sqrt, pow.
 * Unknown functions are rejected and fail closed.
 */
object ExpressionFunctionRegistry {

    data class FunctionDef(
        val name: String,
        val argCount: Int,
        val description: String
    )

    private val supportedFunctions = hashMapOf(
        "min" to FunctionDef("min", 2, "Minimum of two values"),
        "max" to FunctionDef("max", 2, "Maximum of two values"),
        "clamp" to FunctionDef("clamp", 3, "Clamp value to [min, max] range: clamp(val, min, max)"),
        "abs" to FunctionDef("abs", 1, "Absolute value"),
        "mean" to FunctionDef("mean", 2, "Arithmetic mean of two values: (a + b) / 2"),
        "mix" to FunctionDef("mix", 3, "Linear interpolation: mix(x, y, t)"),
        "smoothstep" to FunctionDef("smoothstep", 3, "Hermite interpolation between edge0 and edge1: smoothstep(edge0, edge1, x)"),
        "step" to FunctionDef("step", 2, "Step function: step(edge, x) returns 1 if x >= edge, else 0"),
        "sin" to FunctionDef("sin", 1, "Trigonometric sine (radians)"),
        "cos" to FunctionDef("cos", 1, "Trigonometric cosine (radians)"),
        "sqrt" to FunctionDef("sqrt", 1, "Square root (clamped to 0 for negative input)"),
        "pow" to FunctionDef("pow", 2, "Power: pow(base, exponent)")
    )

    fun isFunctionSupported(funcName: String): Boolean {
        return supportedFunctions.containsKey(funcName)
    }

    fun getFunction(funcName: String): FunctionDef? {
        return supportedFunctions[funcName]
    }

    fun getSupportedFunctionNames(): Set<String> {
        return supportedFunctions.keys
    }
}
