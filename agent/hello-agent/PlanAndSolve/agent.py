from customModel.model import HelloAgentsLLM
import ast 

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的，可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题：{question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的：
```python
["步骤1", "步骤2", "步骤3", ...]
```

"""

REPLANNER_PROMPT_TEMPLATE = """
你是一位顶级的AI规划专家。你之前的计划在执行过程中遇到了问题，需要你重新规划剩余步骤。

# 原始问题：
{question}

# 原始计划：
{original_plan}

# 已成功完成的步骤与结果：
{completed_history}

# 执行失败的步骤及原因：
{failed_step_info}

请根据当前状态，重新规划后续需要执行的步骤（只包含还未完成的步骤）。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的：
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

STEP_VALIDATOR_PROMPT_TEMPLATE = """
你是一位严格的质量评估专家。请评估一个任务步骤的执行结果是否令人满意。

# 原始问题：
{question}

# 当前步骤：
{step}

# 执行结果：
{result}

该步骤是否成功完成了它的目标？如果结果中包含错误信息、内容为空、或没有实质性回答步骤的要求，则判定为失败。
请只输出一个词:"成功"或"失败"。
"""

class Planner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
    
    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        # 为了生成计划，我们构建一个简单的消息列表
        messages = [
            {"role": "user", "content": prompt}
        ]
        print("🧩 正在生成行动计划...")
        response_text = self.llm_client.think(messages)
        print(f"✅ 行动计划生成完成: {response_text}")

        # 解析LLM输出的列表字符串
        try:
            # 找到 ```python``` 标签之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()

            # 使用ast.literal_eval安全地解析字符串为列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析行动计划时发生错误: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 处理行动计划时发生未知错误: {e}")
            return []

class Replanner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def replan(self, question: str, original_plan: list[str],
               completed_history: str, failed_step_info: str) -> list[str]:
        """
        根据执行失败的信息重新规划剩余步骤。
        """
        prompt = REPLANNER_PROMPT_TEMPLATE.format(
            question=question,
            original_plan=original_plan,
            completed_history=completed_history,
            failed_step_info=failed_step_info
        )

        messages = [
            {"role": "user", "content": prompt}
        ]
        print("  🔄 正在重新规划剩余步骤...")
        response_text = self.llm_client.think(messages)
        print(f"  ✅ 重规划完成: {response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"  ❌ 解析重规划结果时发生错误: {e}")
            return []
        except Exception as e:
            print(f"  ❌ 处理重规划结果时发生未知错误: {e}")
            return []

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题，完整的计划，以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤",并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题：
{question}

# 完整计划：
{plan}

# 历史步骤与结果：
{history}

# 当前步骤：
{current_step}

请仅输出针对"当前步骤"的回答:
"""   

class Executor:
    def __init__(self,llm_client):
        self.llm_client = llm_client

    def execute_step(self, question: str, plan: list[str], history: str, step: str) -> str:
        """执行单个步骤并返回结果。"""
        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            question=question,
            plan=plan,
            history=history,
            current_step=step
        )
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages) or ""
        return response_text

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = ""
        print("\n🚀 正在执行行动计划...")
        for i, step in enumerate(plan):
            print(f"\n🔹 当前步骤 {i+1}/{len(plan)}: {step}")

            response_text = self.execute_step(question, plan, history, step)

            # 更新历史记录
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"

            print(f"✅ 步骤 {i+1} 完成，结果: {response_text}")

        print("\n🎉 所有步骤执行完成！")
        final_answer = response_text
        return final_answer

class StepValidator:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def validate(self, question: str, step: str, result: str) -> bool:
        """检查步骤执行结果是否达到预期。"""
        prompt = STEP_VALIDATOR_PROMPT_TEMPLATE.format(
            question=question,
            step=step,
            result=result
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.think(messages) or ""
        return "成功" in response

class PlanAndSolveAgent:
    def __init__(self, llm_client, max_replans=3):
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)
        self.replanner = Replanner(llm_client)
        self.validator = StepValidator(llm_client)
        self.max_replans = max_replans

    def run(self, question: str) -> str:
        """
        运行agent的完整流程：先规划，后动态执行（支持重规划）。
        当某一步骤执行结果未通过验证时，自动触发重规划调整剩余步骤。
        """
        print(f"\n🤖 用户问题: {question}")

        # 1. 调用规划器生成初始计划
        plan = self.planner.plan(question)
        if not plan:
            print("❌ 没有生成有效的计划，无法继续执行。")
            return ""

        print(f"📋 初始计划 ({len(plan)} 步): {plan}")

        # 2. 动态执行循环（支持重规划）
        history = ""
        replan_count = 0
        step_index = 0
        final_answer = ""

        print("\n🚀 开始动态执行（支持重规划）...")

        while step_index < len(plan):
            current_step = plan[step_index]
            print(f"\n🔹 步骤 {step_index + 1}/{len(plan)}: {current_step}")

            # 执行当前步骤
            try:
                response_text = self.executor.execute_step(question, plan, history, current_step)
            except Exception as e:
                print(f"  ❌ 执行异常: {e}")
                response_text = ""

            # 截断显示过长结果
            display = response_text[:200] + "..." if len(response_text) > 200 else response_text
            print(f"  📝 结果: {display}")

            # 验证步骤结果是否达到预期
            is_ok = self.validator.validate(question, current_step, response_text)

            if is_ok:
                print(f"  ✅ 步骤通过验证")
                history += f"步骤 {step_index + 1}: {current_step}\n结果: {response_text}\n\n"
                step_index += 1
                final_answer = response_text
            else:
                print(f"  ⚠️ 步骤未通过验证")
                if replan_count < self.max_replans:
                    replan_count += 1
                    print(f"  🔄 触发第 {replan_count}/{self.max_replans} 次重规划...")

                    new_plan = self.replanner.replan(
                        question=question,
                        original_plan=plan,
                        completed_history=history,
                        failed_step_info=(
                            f"步骤: {current_step}\n"
                            f"执行结果: {response_text}\n"
                            f"原因: 结果未能满足步骤目标"
                        )
                    )

                    if new_plan and len(new_plan) > 0:
                        print(f"  📋 重规划后的新计划 ({len(new_plan)} 步): {new_plan}")
                        plan = new_plan
                        step_index = 0  # 从新计划的第1步开始
                    else:
                        print("  ❌ 重规划未生成有效计划，跳过当前步骤")
                        history += (
                            f"步骤 {step_index + 1}: {current_step}\n"
                            f"结果: {response_text}（未验证通过，重规划失败）\n\n"
                        )
                        step_index += 1
                else:
                    print(f"  ⏭️ 已达最大重规划次数 ({self.max_replans})，跳过当前步骤")
                    history += (
                        f"步骤 {step_index + 1}: {current_step}\n"
                        f"结果: {response_text}（未验证通过）\n\n"
                    )
                    step_index += 1

        print(f"\n🎯 最终答案: {final_answer}")
        return final_answer
            