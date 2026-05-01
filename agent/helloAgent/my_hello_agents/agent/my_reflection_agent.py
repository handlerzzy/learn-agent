DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "rank": """
你是一个专业的ai评审员，负责评估回答的质量。请根据以下标准对回答进行评分:
- 准确性: 回答是否正确地解决了问题。
- 完整性: 回答是否涵盖了问题的所有方面。
- 清晰度: 回答是否表达清晰，易于理解。

原始任务: {task}

回答: {content}

你需要根据上述标准针对这个对于原始任务的回答进行评分，满分为10分。

输出格式:
评分：你的评分

请严格按照输出格式进行回答。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}   
from typing import Optional
from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message

class MyReflectionAgent(ReflectionAgent):
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            custom_prompts: Optional[dict[str, str]] = None,
            max_iterations: int = 3,
            min_rank_score: int = 6
    ):
        super().__init__(name, llm, system_prompt, config)
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
        self.max_iterations = max_iterations
        self.min_rank_score = min_rank_score
        print(f"{name}初始化完成，使用自定义提示词: {bool(custom_prompts)}")
    
    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        # 1.初始回答
        initial_prompt = self.prompts["initial"].format(task=input_text)
        current_response = self.llm.invoke([{"role": "user", "content": initial_prompt}], **kwargs)
        print(f"初始回答: {current_response}\n")
        iteration = 0
        while iteration < self.max_iterations:
            rank_prompt = self.prompts['rank'].format(
                task=input_text,
                content=current_response
            )
            rank_response = self.llm.invoke([{"role": "user", "content": rank_prompt}], **kwargs)

            if "评分：" in rank_response:
                try:
                    score_str = rank_response.split("评分：")[1].strip()
                    score = int(score_str)
                    print(f"第 {iteration+1} 轮评分: {score}\n")
                    if score >= self.min_rank_score:
                        print(f"第 {iteration+1} 轮评分达到要求，结束迭代")
                        final_response = current_response
                        break
                except Exception as e:
                    print(f"解析评分失败: {e}")
            else:
                print(f"第 {iteration+1} 轮评分格式不正确，继续迭代\n")

            
            reflect_prompt = self.prompts['reflect'].format(
                task=input_text,
                content=current_response
            )
            feedback = self.llm.invoke([{"role": "user", "content": reflect_prompt}], **kwargs)
            print(f"第 {iteration+1} 轮反思反馈: {feedback}\n")
            if "无需改进" in feedback:
                print(f"第 {iteration+1} 轮反馈: 无需改进，结束迭代")
                final_response = current_response
                break
            else:
                refine_prompt = self.prompts['refine'].format(
                    task=input_text,
                    last_attempt=current_response,
                    feedback=feedback
                )
                current_response = self.llm.invoke([{"role": "user", "content": refine_prompt}], **kwargs)
                print(f"第 {iteration+1} 轮改进回答: {current_response}\n")
                iteration += 1
        if iteration >= self.max_iterations:
            print(f"达到最大迭代次数 {self.max_iterations}，结束迭代")
            final_response = current_response
        self.add_message(Message(role="user", content=input_text))
        self.add_message(Message(role="assistant", content=final_response))
        return final_response