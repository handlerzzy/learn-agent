from abc import ABC,abstractmethod
from typing import Any
from pydantic import BaseModel

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        """执行工具的逻辑"""
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """返回工具需要的参数信息"""
        pass

class ToolRegistry:
    """
    工具注册表，管理所有可用的工具
    """
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """注册一个工具"""
        if tool.name in self._tools:
            print(f"⚠️ 工具 '{tool.name}' 已经注册，覆盖原有工具")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 注册成功")
    
    def register_function(self, name: str, description: str, func: callable[[str],str]):
        """
        直接注册函数作为工具
        Args:
            name: 工具名称
            description: 工具描述
            func: 可调用对象，接受一个字符串参数，返回一个字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 函数工具 '{name}' 已经注册，覆盖原有工具")
        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 函数工具 '{name}' 注册成功")
    
    def get_tools_description(self) -> str:
        """返回所有工具的描述信息"""
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(f"-{tool.name}: {tool.description}")
        for name, info in self._functions.items():
            descriptions.append(f"-{name}: {info['description']}")
        return "\n".join(descriptions) if descriptions else "没有可用的工具"
    
    def to_openai_schema(self) -> dict[str, Any]:
        """将工具信息转换为OpenAI API所需的格式"""
        parameters = self.get_parameters()

        # 构建 properties 字段
        properties = {}
        required = []
        for param in parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                prop['description'] = f"{param.description} (默认值: {param.default})"

            # 如果是数组类型，添加items定义
            if param.type == "array":
                prop["items"] = {"type": "string"}

            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }