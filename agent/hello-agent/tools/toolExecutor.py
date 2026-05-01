from typing import Any # 用于类型提示，any表示任意类型
class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具
    """
    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {} # 存储工具的字典，键是工具名称，值是工具的描述和函数

    def registerTool(self, name: str, description: str, func: callable): 
        """
        向工具箱中注册一个新工具
        """
        if name in self.tools:
            print(f"⚠️ 工具 '{name}' 已经存在，正在覆盖...")
        self.tools[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 注册成功！")
    
    def getTool(self, name: str) -> callable:
        """
        根据工具名称获取工具函数
        """
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"工具 '{name}' 未找到！")
        return tool["func"]
    
    def getAvailbleTools(self) -> str:
        """
        获取所有可用工具的描述，供大语言模型参考
        """
        return "\n".join(
            [f"-{name}: {info['description']}" for name, info in self.tools.items()]
        )