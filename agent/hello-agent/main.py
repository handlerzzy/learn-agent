from dotenv import load_dotenv
from tools.toolExecutor import ToolExecutor
from tools.tools import calculate, search
from ReAct.agent import ReActAgent
from customModel.model import HelloAgentsLLM
from PlanAndSolve.agent import PlanAndSolveAgent
from Reflection.agent import ReflectionAgent
load_dotenv()




if __name__ == '__main__':
    # 初始化工具执行器并注册工具
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        name="search",
        description="一个基于SerpApi的网页搜索工具，输入是搜索查询，输出是搜索结果的摘要。",
        func=search
    )
    tool_executor.registerTool(
        name="calculate",
        description="一个数学计算器工具，支持加减乘除幂运算，输入是一个数学表达式字符串(如'3+5'或'2**10')，输出是计算结果。",
        func=calculate
    )

    # 初始化大语言模型客户端
    llm_client = HelloAgentsLLM()

    # 创建ReAct Agent并运行
    react_agent = ReActAgent(llm_client, tool_executor)
    question = "帮我计算：(37.85 × 16.4 - 289.32) ÷ 7.23 + 根号 1849 - 13 的立方 ÷ 41"  
    print("\n=== 运行ReAct Agent ===")
    react_agent.run(question)

    