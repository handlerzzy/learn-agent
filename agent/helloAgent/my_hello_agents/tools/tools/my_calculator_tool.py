import ast # 用于解析表达式
import operator # 用于执行基本的算术运算
import math
from hello_agents import ToolRegistry

def my_calculate(expression: str) -> str:
    """
    简单的计算器函数，支持基本的算术运算和一些数学函数
    """
    if not expression.strip():
        return "请输入一个有效的表达式"
    
    # 支持的基本运算
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    # 支持的基本函数
    functions = {
        "sqrt": math.sqrt,
        "pi": math.pi,
    }

    try:
        node = ast.parse(expression, mode='eval')
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

def _eval_node(node, operators, functions):
    """
    简化的表达式求值
    """
    if isinstance(node, ast.Constant):
        return node.value # 直接返回常量值
    
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        return op(left, right) if op else None
    
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func_name](*args)
    
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]
        
def create_calculator_registry():
    """
    创建一个工具注册表，并注册计算器函数
    """
    registry = ToolRegistry()
    registry.register_function(
        name="my_calculator",
        description="简单的数学计算工具，支持基本的数学运算(+, -, *, /)和一些数学函数(sqrt, pi)",
        func=my_calculate
    )
    return registry

