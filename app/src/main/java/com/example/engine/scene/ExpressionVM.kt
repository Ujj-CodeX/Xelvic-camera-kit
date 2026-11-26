package com.example.engine.scene

import kotlin.math.*

/**
 * Expression bytecode compiler and fixed-stack VM (README Section 5.10 & 21 Step 3).
 * Supports:
 * - Numbers, signal references (sig.xxx, face.xxx, time.xxx), params ($param, param.xxx)
 * - Operators: unary minus, +, -, *, /
 * - Functions from ExpressionFunctionRegistry: min, max, clamp, abs, mean, mix, smoothstep, step, sin, cos, sqrt, pow
 * - Cycle detection and topological dependency ordering
 * - Fixed-stack execution (32 slots)
 * - NaN and infinity clamped to 0
 */
class ExpressionVM {

    enum class OpCode {
        PUSH_CONST,
        PUSH_SLOT,
        NEG,
        ADD,
        SUB,
        MUL,
        DIV,
        FUNC_MIN,
        FUNC_MAX,
        FUNC_CLAMP,
        FUNC_ABS,
        FUNC_MEAN,
        FUNC_MIX,
        FUNC_SMOOTHSTEP,
        FUNC_STEP,
        FUNC_SIN,
        FUNC_COS,
        FUNC_SQRT,
        FUNC_POW
    }

    data class Instruction(val op: OpCode, val floatVal: Float = 0f, val intVal: Int = 0)

    data class Program(val instructions: List<Instruction>, val referencedSlots: List<Int>)

    private val stack = FloatArray(32)
    private var sp = 0

    fun execute(program: Program, slots: FloatArray): Float {
        sp = 0
        val insts = program.instructions
        val len = insts.size
        for (i in 0 until len) {
            val inst = insts[i]
            when (inst.op) {
                OpCode.PUSH_CONST -> {
                    if (sp < stack.size) stack[sp++] = inst.floatVal
                }
                OpCode.PUSH_SLOT -> {
                    val idx = inst.intVal
                    val v = if (idx >= 0 && idx < slots.size) slots[idx] else 0f
                    if (sp < stack.size) stack[sp++] = v
                }
                OpCode.NEG -> {
                    if (sp > 0) stack[sp - 1] = -stack[sp - 1]
                }
                OpCode.ADD -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = a + b
                    }
                }
                OpCode.SUB -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = a - b
                    }
                }
                OpCode.MUL -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = a * b
                    }
                }
                OpCode.DIV -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = if (abs(b) > 1e-7f) a / b else 0f
                    }
                }
                OpCode.FUNC_MIN -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = min(a, b)
                    }
                }
                OpCode.FUNC_MAX -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = max(a, b)
                    }
                }
                OpCode.FUNC_CLAMP -> {
                    if (sp >= 3) {
                        val hi = stack[--sp]; val lo = stack[--sp]; val x = stack[--sp]
                        stack[sp++] = x.coerceIn(lo, hi)
                    }
                }
                OpCode.FUNC_ABS -> {
                    if (sp >= 1) stack[sp - 1] = abs(stack[sp - 1])
                }
                OpCode.FUNC_MEAN -> {
                    if (sp >= 2) {
                        val b = stack[--sp]; val a = stack[--sp]
                        stack[sp++] = (a + b) * 0.5f
                    }
                }
                OpCode.FUNC_MIX -> {
                    if (sp >= 3) {
                        val t = stack[--sp]; val y = stack[--sp]; val x = stack[--sp]
                        stack[sp++] = x + (y - x) * t
                    }
                }
                OpCode.FUNC_SMOOTHSTEP -> {
                    if (sp >= 3) {
                        val x = stack[--sp]; val edge1 = stack[--sp]; val edge0 = stack[--sp]
                        val t = ((x - edge0) / (edge1 - edge0).coerceAtLeast(1e-6f)).coerceIn(0f, 1f)
                        stack[sp++] = t * t * (3f - 2f * t)
                    }
                }
                OpCode.FUNC_STEP -> {
                    if (sp >= 2) {
                        val x = stack[--sp]; val edge = stack[--sp]
                        stack[sp++] = if (x >= edge) 1f else 0f
                    }
                }
                OpCode.FUNC_SIN -> {
                    if (sp >= 1) stack[sp - 1] = sin(stack[sp - 1])
                }
                OpCode.FUNC_COS -> {
                    if (sp >= 1) stack[sp - 1] = cos(stack[sp - 1])
                }
                OpCode.FUNC_SQRT -> {
                    if (sp >= 1) stack[sp - 1] = sqrt(max(0f, stack[sp - 1]))
                }
                OpCode.FUNC_POW -> {
                    if (sp >= 2) {
                        val exp = stack[--sp]; val base = stack[--sp]
                        stack[sp++] = base.toDouble().pow(exp.toDouble()).toFloat()
                    }
                }
            }
        }

        val res = if (sp > 0) stack[--sp] else 0f
        return if (res.isNaN() || res.isInfinite()) 0f else res
    }

    /**
     * Parse and compile a mathematical expression string into bytecode.
     */
    fun compile(expr: String, slotResolver: (String) -> Int): Program {
        val tokens = tokenize(expr)
        val instructions = ArrayList<Instruction>()
        val referencedSlots = ArrayList<Int>()

        // Shunting-yard algorithm
        val opStack = ArrayList<String>()
        var i = 0
        var prevToken: String? = null

        while (i < tokens.size) {
            val token = tokens[i]
            when {
                token.toFloatOrNull() != null -> {
                    instructions.add(Instruction(OpCode.PUSH_CONST, floatVal = token.toFloat()))
                }
                token.startsWith("func:") -> {
                    val fn = token.substring(5)
                    if (!ExpressionFunctionRegistry.isFunctionSupported(fn)) {
                        throw IllegalArgumentException("Unknown expression function '$fn'")
                    }
                    opStack.add(token)
                }
                token == "(" -> {
                    opStack.add("(")
                }
                token == ")" -> {
                    while (opStack.isNotEmpty() && opStack.last() != "(") {
                        emitOp(opStack.removeAt(opStack.size - 1), instructions)
                    }
                    if (opStack.isNotEmpty() && opStack.last() == "(") {
                        opStack.removeAt(opStack.size - 1)
                    }
                    if (opStack.isNotEmpty() && opStack.last().startsWith("func:")) {
                        emitOp(opStack.removeAt(opStack.size - 1), instructions)
                    }
                }
                token == "," -> {
                    while (opStack.isNotEmpty() && opStack.last() != "(") {
                        emitOp(opStack.removeAt(opStack.size - 1), instructions)
                    }
                }
                token == "-" && (prevToken == null || prevToken == "(" || prevToken == "," || isOperator(prevToken)) -> {
                    // Unary minus
                    opStack.add("u-")
                }
                isOperator(token) -> {
                    while (opStack.isNotEmpty() && (isOperator(opStack.last()) || opStack.last() == "u-") &&
                        precedence(opStack.last()) >= precedence(token)
                    ) {
                        emitOp(opStack.removeAt(opStack.size - 1), instructions)
                    }
                    opStack.add(token)
                }
                else -> {
                    // Variable / signal name
                    val slot = slotResolver(token)
                    referencedSlots.add(slot)
                    instructions.add(Instruction(OpCode.PUSH_SLOT, intVal = slot))
                }
            }
            prevToken = token
            i++
        }

        while (opStack.isNotEmpty()) {
            val top = opStack.removeAt(opStack.size - 1)
            if (top != "(") {
                emitOp(top, instructions)
            }
        }

        return Program(instructions, referencedSlots)
    }

    private fun isOperator(t: String): Boolean = t in listOf("+", "-", "*", "/")
    private fun precedence(op: String): Int = when (op) {
        "u-" -> 3
        "*", "/" -> 2
        "+", "-" -> 1
        else -> 0
    }

    private fun emitOp(op: String, instructions: ArrayList<Instruction>) {
        when (op) {
            "u-" -> instructions.add(Instruction(OpCode.NEG))
            "+" -> instructions.add(Instruction(OpCode.ADD))
            "-" -> instructions.add(Instruction(OpCode.SUB))
            "*" -> instructions.add(Instruction(OpCode.MUL))
            "/" -> instructions.add(Instruction(OpCode.DIV))
            "func:min" -> instructions.add(Instruction(OpCode.FUNC_MIN))
            "func:max" -> instructions.add(Instruction(OpCode.FUNC_MAX))
            "func:clamp" -> instructions.add(Instruction(OpCode.FUNC_CLAMP))
            "func:abs" -> instructions.add(Instruction(OpCode.FUNC_ABS))
            "func:mean" -> instructions.add(Instruction(OpCode.FUNC_MEAN))
            "func:mix" -> instructions.add(Instruction(OpCode.FUNC_MIX))
            "func:smoothstep" -> instructions.add(Instruction(OpCode.FUNC_SMOOTHSTEP))
            "func:step" -> instructions.add(Instruction(OpCode.FUNC_STEP))
            "func:sin" -> instructions.add(Instruction(OpCode.FUNC_SIN))
            "func:cos" -> instructions.add(Instruction(OpCode.FUNC_COS))
            "func:sqrt" -> instructions.add(Instruction(OpCode.FUNC_SQRT))
            "func:pow" -> instructions.add(Instruction(OpCode.FUNC_POW))
        }
    }

    fun extractSignalDependencies(expr: String): List<String> {
        val tokens = tokenize(expr)
        val deps = ArrayList<String>()
        val skip = setOf("(", ")", ",", "+", "-", "*", "/", "u-")
        for (t in tokens) {
            if (t !in skip && !t.startsWith("func:") && t.toFloatOrNull() == null) {
                deps.add(t)
            }
        }
        return deps
    }

    /**
     * Cycle detection and topological sort of derived signal definitions.
     * @return Ordered list of signal keys to evaluate sequentially, or throws IllegalStateException if a cycle is found.
     */
    fun getTopologicalOrder(signals: Map<String, String>): List<String> {
        val graph = HashMap<String, List<String>>()
        for ((k, expr) in signals) {
            val deps = extractSignalDependencies(expr).filter { it in signals || it.removePrefix("sig.") in signals }
                .map { if (signals.containsKey(it)) it else it.removePrefix("sig.") }
            graph[k] = deps
        }

        val visited = HashMap<String, Int>() // 0 = unvisited, 1 = visiting, 2 = visited
        val result = ArrayList<String>()

        fun dfs(node: String, path: ArrayList<String>) {
            val state = visited[node] ?: 0
            if (state == 1) {
                path.add(node)
                val cycleStr = path.joinToString(" -> ")
                throw IllegalStateException("Cycle detected in signals: $cycleStr")
            }
            if (state == 2) return

            visited[node] = 1
            path.add(node)
            val deps = graph[node] ?: emptyList()
            for (dep in deps) {
                dfs(dep, path)
            }
            path.removeAt(path.size - 1)
            visited[node] = 2
            result.add(node)
        }

        for (k in signals.keys) {
            if (visited[k] == null || visited[k] == 0) {
                dfs(k, ArrayList())
            }
        }

        return result
    }

    private fun tokenize(expr: String): List<String> {
        val list = ArrayList<String>()
        var cur = StringBuilder()
        var j = 0

        while (j < expr.length) {
            val c = expr[j]
            if (c.isWhitespace()) {
                if (cur.isNotEmpty()) {
                    val s = cur.toString()
                    if (ExpressionFunctionRegistry.isFunctionSupported(s)) list.add("func:$s") else list.add(s)
                    cur = StringBuilder()
                }
            } else if (c in listOf('(', ')', ',', '+', '-', '*', '/')) {
                if (cur.isNotEmpty()) {
                    val s = cur.toString()
                    if (ExpressionFunctionRegistry.isFunctionSupported(s)) list.add("func:$s") else list.add(s)
                    cur = StringBuilder()
                }
                list.add(c.toString())
            } else {
                cur.append(c)
            }
            j++
        }
        if (cur.isNotEmpty()) {
            val s = cur.toString()
            if (ExpressionFunctionRegistry.isFunctionSupported(s)) list.add("func:$s") else list.add(s)
        }
        return list
    }
}
