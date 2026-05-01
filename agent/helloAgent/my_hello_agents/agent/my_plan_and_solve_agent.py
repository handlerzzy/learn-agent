# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""
from typing import Optional
from hello_agents import PlanAndSolveAgent, HelloAgentsLLM, Config, Message
class MyPlanAndSolveAgent(PlanAndSolveAgent):
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            planner_prompt: Optional[str] = None,
            executor_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.planner_prompt = planner_prompt if planner_prompt else DEFAULT_PLANNER_PROMPT
        self.executor_prompt = executor_prompt if executor_prompt else DEFAULT_EXECUTOR_PROMPT
        
    def _planner(self, question: str) -> list[str]:
        prompt = self.planner_prompt.format(question=question)
        self.add_message(Message(role="user", content=prompt))
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        try:
            clean_response = response.strip().strip("```python").strip("```").strip()
            plan = eval(clean_response)  # 注意：eval存在安全风险，实际使用时请确保输入可信
            if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                print(f"生成的计划: {plan}")
                return plan
            else:
                raise ValueError("计划必须是一个字符串列表")
        except Exception as e:
            print(f"ai的回答{response}")
            print(f"解析计划失败: {e}")
            return []
    
    def _executor(self, question: str, plan: list[str]) -> str:
        print("开始执行计划...\n")
        history = []
        for step in plan:
            print(f"当前步骤: {step}")
            prompt = self.executor_prompt.format(
                question=question,
                plan=plan,
                history="\n".join([f"{h['step']}: {h['result']}" for h in history]),
                current_step=step
            )
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            print(f"步骤结果: {response}\n")
            history.append({"step": step, "result": response})
        final_answer = history[-1]["result"] if history else "没有执行任何步骤"
        self.add_message(Message(role="assistant", content=final_answer))
        return final_answer
    
    def run(self, input_text, **kwargs) -> str:
        question = input_text
        plan = self._planner(question)
        if not plan:
            return "无法生成有效的计划"
        answer = self._executor(question, plan)
        return answer
