from typing import Optional, Iterator
from ..core.my_agent import Agent
from ..core.my_llm import HelloAgentsLLM
from ..core.my_config import Config
from ..core.my_messages import Message
import re

class MySimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry=None,
        enable_tool_calling: bool = False,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        运行简单Agent
        
        Args:
            input_text: 用户输入
            max_tool_iterations: 最大工具调用次数
            **kwargs: 其他参数
            
        Returns:
            Agent响应
        """

        print(f"[{self.name}] 正在处理输入: {input_text}")
        # 构建消息列表
        messages = []

        # 添加系统消息(可能包含工具信息)
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"[{self.name}] 响应完成")
            return response
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """
        获取增强的系统提示，包含工具信息
        """
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt
        
        tools_description = self.tool_registry.get_tools_description()

        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt
        
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + '\n'

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当你需要使用工具时，请按照以下格式调用:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]`或`[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
        return base_prompt + tools_section
    
    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        """
        支持工具调用的运行逻辑
        """
        current_iteration = 0
        final_response = ""
        while current_iteration < max_tool_iterations:
            # 调用LLM获取响应
            response = self.llm.invoke(messages, **kwargs)

            # 检擦是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"[{self.name}] 检测到{len(tool_calls)}个工具调用")
                tool_results = []
                clean_response = response
                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append((result))
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")
                # 将工具结果插入到消息中
                messages.append({"role": "assistant", "content": clean_response})

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role":"user","content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})
                current_iteration += 1
                continue
            final_response = response
            break

        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"[{self.name}] 响应完成")
        return final_response
    
    def _parse_tool_calls(self, text: str) -> list[dict]:

        """
        解析文本中的工具调用
        """
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                "original": f'[TOOL_CALL:{tool_name}:{parameters}]'
            })
        return tool_calls
    
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """
        执行工具调用
        """
        if not self.tool_registry:
            return f"[错误] 未配置工具注册表"
        
        try:
            if tool_name == 'calculator':
                # 计算器工具需要特殊处理，直接返回结果
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                param_dict = self._parse_tool_parameters(tool_name,parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"[错误] 未找到工具 {tool_name}"
                result = tool.run(param_dict)
            return f"[工具] {tool_name} 执行结果:\n{result}"
        
        except Exception as e:
            return f"[错误] 工具 {tool_name} 执行失败: {str(e)}"
        
    def _parse_tool_parameters(self, tool_name: str,parameters: str) -> dict:
        """
        智能解析工具参数
        """
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if "," in parameters:
                # 多个参数:
                pairs = parameters.split(",")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        param_dict[key.strip()] = value.strip()
                else:
                    # 单个参数
                    key, value = parameters.split("=", 1)
                    param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {"query": parameters.strip()}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters.strip()}
            else:
                param_dict = {"input": parameters.strip()}
        return param_dict
    
    def stream_run(self, input_text: str, **kwargs)-> Iterator[str]:
        """
        流式运行Agent，逐步生成响应
        """
        print(f"[{self.name}] 正在处理输入: {input_text}")
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})

        full_response = ""
        print(f"[{self.name}] 正在生成响应...")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        print()
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"[{self.name}] 响应完成")
    
    def add_tool(self, tool):
        """
        添加工具并更新系统提示
        """
        if not self.tool_registry:
            from hello_agents.tools import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True
        self.tool_registry.register_tool(tool)
        print(f"[工具] {tool.name} 已添加到 {self.name}")
    
    def has_tools(self) -> bool:
        """
        检查是否有工具可用
        """
        return self.tool_registry and self.tool_registry is not None
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具并更新系统提示
        """
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False
    
    def list_tools(self) -> list:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []
    
    