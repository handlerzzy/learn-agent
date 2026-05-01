import math
import os
from serpapi import SerpApiClient


# ============================================================
# 工具1: 网页搜索
# ============================================================
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("SERPAPI_API_KEY必须在环境变量中定义。")
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-CN",
        }
        client = SerpApiClient(params)
        results = client.get_dict()
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


# ============================================================
# 工具2: 计算器
# ============================================================
def calculate(expression: str) -> str:
    """
    【计算器工具】安全地执行数学表达式计算，支持加减乘除幂等多种运算。

    —— 功能说明 ——
    Agent在推理过程中若遇到需要数学计算的步骤（例如统计数据、比较数值、
    计算比例等），可以使用此工具代替自行推导，以获得精确结果。

    —— 支持的操作符 ——
    操作符   │ 说明                  │ 示例
    ─────────┼───────────────────────┼─────────────
    +        │ 加法                  │ 3 + 5 = 8
    -        │ 减法                  │ 10 - 4 = 6
    *        │ 乘法                  │ 6 * 7 = 42
    /        │ 除法（返回浮点数）     │ 15 / 4 = 3.75
    //       │ 整除（向下取整）       │ 15 // 4 = 3
    %        │ 取模（求余数）         │ 17 % 5 = 2
    **       │ 幂运算                │ 2 ** 10 = 1024
    ()       │ 括号（改变优先级）     │ (2 + 3) * 4 = 20

    —— 内置数学函数与常量 ——
    ��� 可直接在表达式中使用的函数:
        abs(), round(), int(), float(), max(), min(), pow()
        sqrt(), sin(), cos(), tan(), log(), log10(), log2(),
        ceil(), floor(), factorial(), pi（圆周率）, e（自然常数）

    💡 使用示例:
        "3 + 5 * 2"
        "(8 + 2) ** 3"
        "sqrt(144) + 100"
        "500 * (1 + 0.05) ** 3"    ← 复利计算
        "pi * 5 ** 2"              ← 圆的面积

    参数:
        expression: 数学表达式字符串。
                    例如: "3 + 5", "2 ** 10", "(8 + 2) * 3"

    返回:
        格式化字符串: "【表达式】 = 【结果】"
        例如: "3 + 5 * 2 = 13"

    异常处理:
        - 除零错误       → "计算错误: 除数不能为0"
        - 无效表达式     → "计算错误: 表达式语法有误，请检查后重试"
        - 使用了危险代码 → "计算错误: 表达式包含不允许的操作"
    """
    print(f"[计算器] 正在计算: {expression}")

    try:
        # ── 构建安全的执行环境 ──
        # 只允许访问 math 模块中的函数和常量，以及少量安全的内置函数
        # 禁止 __builtins__ 可以防止执行危险代码（如 open, exec, import 等）
        safe_globals = {
            "__builtins__": {},          # 禁用所有内置函数，只开放下面几个
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "max": max,
            "min": min,
            "pow": pow,
            "sum": sum,
        }

        # ── 注入 math 模块的所有函数和常量 ──
        # 这样用户可以直接使用 sqrt(16)、sin(pi/2) 等
        # 注意: vars(math) 返回 math 模块的所有属性（函数+常量）
        safe_globals.update(
            {name: getattr(math, name) for name in vars(math) if not name.startswith("_")}
        )

        # ── 执行计算 ──
        # eval() 在 safe_globals 的限制下执行，无法访问文件系统或执行系统命令
        result = eval(expression.strip(), safe_globals)

        # ── 格式化输出 ──
        # 如果结果是浮点数且包含很多小数位，保留 10 位有效数字
        if isinstance(result, float):
            # 对于整数结果（如 4.0），显示为整数更清爽
            if result == int(result):
                return f"{expression} = {int(result)}"
            # 否则保留 10 位小数，并去掉末尾多余的 0
            return f"{expression} = {round(result, 10)}"
        return f"{expression} = {result}"

    except ZeroDivisionError:
        # 除零错误: 当用户尝试除以 0 时触发
        return "计算错误: 除数不能为0，请检查表达式"

    except SyntaxError:
        # 语法错误: 当表达式不符合 Python 语法时触发
        # 例如: "3 ++ 5"、"hello" 等
        return "计算错误: 表达式语法有误，请检查后重试"

    except NameError:
        # 名称错误: 当表达式中使用了未定义的名称时触发
        # 例如: "foo + 1"（foo 未定义）
        return "计算错误: 表达式中包含未识别的名称，请检查拼写"

    except TypeError:
        # 类型错误: 当操作数类型不匹配时触发
        # 例如: "sqrt('hello')"（字符串不能开平方）
        return "计算错误: 表达式中存在类型不匹配的操作"

    except Exception as e:
        # 兜底捕获: 处理其他所有未预见的异常
        return f"计算错误: {e}"


