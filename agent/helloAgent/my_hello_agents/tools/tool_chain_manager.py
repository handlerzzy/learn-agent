from typing import List, dict, Any, Optional
from hello_agents import ToolRegistry

class ToolChain:

    def __init__(self, name: str, description: str):
        self.name = name 
        self.description = description
        self.steps: List[dict[str, Any]] = []
    
    def add_step(self, tool_name: str,input_template: str, output_key: str = None):

        """添加工具链步骤"""
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key
        })

    def execute(self, registry: ToolRegistry, initial_input: str,context: dict[str, Any]=None) -> str:
        """执行工具链"""
        context = context or {}
        context['input'] = initial_input
        print(f"🔧 执行工具链 '{self.name}'，初始输入: {initial_input}")
        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]
            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                print(f"❌ 工具链 '{self.name}' 第 {i} 步输入模板格式错误，缺少参数: {e}")
                return f"工具链执行失败，缺少参数: {e}"
            print(f"➡️ 第 {i} 步: 使用工具 '{tool_name}'，输入: {tool_input}")
            result = registry.execute_tool(tool_name, tool_input)
            context[output_key or f"step_{i}_output"] = result
            print(f"✅ 第 {i} 步完成，输出: {result}")

        final_result = context[self.steps[-1]["output_key"]]
        print(f"🎉 工具链 '{self.name}' 执行完成，最终结果: {final_result}")
        return final_result
    
class ToolChainManager:


    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}
    
    def register_chain(self, chain: ToolChain):
        """注册工具链"""

        self.chains[chain.name] = chain
    def execute_chain(self, chain_name: str, input_data: str, context: dict[str, Any] = None) -> str:
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 未找到"
        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)
    
    def list_chains(self) ->  list[str]:
        return list(self.chains.keys())