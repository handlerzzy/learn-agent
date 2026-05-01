from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_hello_agents.agent.my_plan_and_solve_agent import MyPlanAndSolveAgent
import os
load_dotenv()
llm = HelloAgentsLLM(
    model=os.getenv("MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 创建自定义PlanAndSolveAgent
agent = MyPlanAndSolveAgent(
    name="我的规划执行助手",
    llm=llm
)

# 测试复杂问题
question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"

result = agent.run(question)
print(f"\n最终结果: {result}")

# 查看对话历史
print(f"对话历史: {len(agent.get_history())} 条消息")

