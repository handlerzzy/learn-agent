import json
from enum import Enum, auto
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError

from customModel.model import HelloAgentsLLM
from tools.toolExecutor import ToolExecutor

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格只输出一个 JSON 对象，不要输出 Markdown、代码块或额外解释。

输出格式如下:
{{
    "thought": "你的思考过程，用于分析问题、拆解任务和规划下一步行动",
    "action": {{
        "type": "tool",
        "name": "工具名称",
        "input": "工具输入"
    }}
}}

当你已经获得最终答案时，请输出:
{{
    "thought": "你的思考过程",
    "action": {{
        "type": "finish",
        "answer": "最终答案"
    }}
}}

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class ToolAction(BaseModel):
    type: Literal["tool"]
    name: Annotated[str, Field(min_length=1,description="工具名称，必须是可用工具列表中的一个")]
    input: Annotated[str, Field(min_length=1,description="工具输入，必须是一个非空字符串")]


class FinishAction(BaseModel):
    type: Literal["finish"]
    answer: Annotated[str, Field(min_length=1)]


class ReActStep(BaseModel):
    thought: Annotated[str, Field(default="",description="思考过程，可以包含分析、拆解和规划等内容")]
    action: Annotated[ToolAction | FinishAction, Field(discriminator="type",description="动作类型，决定是调用工具还是输出最终答案")] 


class FailureType(Enum):
    """工具调用失败类型。"""
    TOOL_NOT_FOUND = auto()          # 调用了不存在的工具
    TOOL_EXECUTION_ERROR = auto()     # 工具执行时返回错误
    INVALID_ACTION = auto()           # LLM输出动作结构解析失败


def _is_tool_error(observation: str) -> bool:
    """判断工具返回结果是否包含错误信息。"""
    error_prefixes = ["错误:", "计算错误:", "搜索时发生错误"]
    return any(observation.startswith(prefix) for prefix in error_prefixes)


class ToolFailureManager:
    """
    工具选择失败处理机制——追踪连续的工具调用错误。

    当 agent 多次调用错误的工具或提供错误的参数时，
    生成逐步升级的纠正引导，帮助 agent 自我修正。
    连续失败达到阈值时终止运行，避免无限浪费 token。
    """

    def __init__(self, max_consecutive_failures: int = 3):
        self.consecutive_failures = 0
        self.total_failures = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_history: list[dict] = []

    def record_success(self):
        """工具调用成功时重置连续失败计数。"""
        self.consecutive_failures = 0

    def record_failure(self, failure_type: FailureType, tool_name: str, error_msg: str) -> str:
        """
        记录一次失败，返回逐步升级的纠正引导文本。

        Args:
            failure_type: 失败类型（工具不存在/执行错误/动作格式错误）
            tool_name: 尝试调用的工具名称（INVALID_ACTION 时传空字符串）
            error_msg: 具体的错误信息

        Returns:
            纠正引导文本，将注入到 history 中供 LLM 下次迭代参考
        """
        self.consecutive_failures += 1
        self.total_failures += 1
        self.failure_history.append({
            "type": failure_type.name,
            "tool_name": tool_name,
            "error": error_msg,
            "consecutive": self.consecutive_failures,
        })
        return self._build_guidance()

    def should_abort(self) -> bool:
        """连续失败达到阈值时建议终止。"""
        return self.consecutive_failures >= self.max_consecutive_failures

    def _build_guidance(self) -> str:
        """根据连续失败次数构建不同力度的纠正引导（3级递增）。"""
        count = self.consecutive_failures
        last = self.failure_history[-1]

        if count == 1:
            return (
                f"[系统纠正] 工具调用出错（{last['type']}）: {last['error']}。"
                f"请检查工具名称和参数格式，从可用工具列表中重新选择。"
            )
        elif count == 2:
            return (
                f"[系统纠正] 你连续 2 次工具调用失败。"
                f"最近错误: {last['error']}。"
                f"请仔细阅读可用工具列表及其参数说明，确保工具名和输入完全正确。"
            )
        else:
            # count >= 3: 达到阈值，最明确的纠正
            return (
                f"[严重纠正] 你已连续 {count} 次工具调用失败！"
                f"最后一次错误: {last['error']}。\n"
                f"请立即暂停当前思路，重新分析问题。"
                f"列出全部可用工具，逐个判断哪个最适合当前步骤，确保参数格式准确无误。"
            )


class ReActAgent:
    def __init__(self, llm_client:HelloAgentsLLM, tool_executor:ToolExecutor,max_steps:int=5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []  # 用于记录对话历史
    
    def run(self, question: str):
        """
        运行ReAct Agent来回答用户的问题。
        集成了工具选择失败处理机制，在连续错误时引导并纠正 agent。
        """
        self.history = []
        failure_mgr = ToolFailureManager(max_consecutive_failures=3)
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"---第{current_step}步---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailbleTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
            )

            # 2. 调用LLM获取响应
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages)
            if not response_text:
                print("LLM没有返回任何内容，结束对话。")
                break

            # 3. 解析LLM的输出
            try:
                step = ReActStep.model_validate_json(response_text)
            except ValidationError as exc:
                print(f"警告：LLM输出不符合ReActStep结构。\n{exc}")
                guidance = failure_mgr.record_failure(
                    FailureType.INVALID_ACTION, "", str(exc)
                )
                self.history.append(f"System: {guidance}")
                print(f"纠正引导: {guidance}")
                if failure_mgr.should_abort():
                    print(
                        f"连续 {failure_mgr.consecutive_failures} 次输出格式错误，终止对话。"
                    )
                    return None
                continue

            if step.thought:
                print(f"Thought: {step.thought}")

            # 4. 执行Action
            if step.action.type == "finish":
                final_answer = step.action.answer.strip()
                print(f"最终答案：{final_answer}")
                return final_answer

            tool_name = step.action.name.strip()
            tool_input = step.action.input.strip()
            print(f"执行工具: {tool_name}，输入: {tool_input}")

            try:
                tool_func = self.tool_executor.getTool(tool_name)
                observation = tool_func(tool_input)
                if _is_tool_error(observation):
                    guidance = failure_mgr.record_failure(
                        FailureType.TOOL_EXECUTION_ERROR, tool_name, observation
                    )
                else:
                    failure_mgr.record_success()
                    guidance = None
            except ValueError as exc:
                observation = f"错误: {exc}"
                guidance = failure_mgr.record_failure(
                    FailureType.TOOL_NOT_FOUND, tool_name, str(exc)
                )

            print(f"观察: {observation}")
            self.history.append(
                f"Action: {json.dumps(step.action.model_dump(), ensure_ascii=False)}"
            )
            self.history.append(f"Observation: {observation}")
            if guidance:
                self.history.append(f"System: {guidance}")
                print(f"纠正引导: {guidance}")

            if failure_mgr.should_abort():
                print(
                    f"连续 {failure_mgr.consecutive_failures} 次工具调用失败，终止对话。"
                )
                return None

        print("达到最大步骤限制，结束对话。")
        return None



    