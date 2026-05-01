"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息
3. 生成基于搜索结果的回答
"""

import sys # 确保输出使用UTF-8编码，避免中文乱码问题
import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from tavily import TavilyClient


# 确保输出使用UTF-8编码，避免中文乱码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 加载环境变量
load_dotenv()

# 定义一个函数来清理文本中的无效字符，避免UTF-8编码错误
def sanitize(text):
    """清理字串中的无效 surrogate 字符，避免 UTF-8 编码错误"""
    return text.encode("utf-8", errors="replace").decode("utf-8")

# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str        # 用户查询
    search_query: str      # 优化后的搜索查询
    search_results: str    # Tavily搜索结果
    final_answer: str      # 最终答案
    step: str             # 当前步骤
    reflect_count: int         # 反思次数，防止无限循环

# 初始化模型和Tavily客户端

llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.7,
    request_timeout=30,
    max_retries=2, # 增加重试次数以提高稳定性
)


# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 理解节点
def understand_query_node(state: SearchState) -> SearchState:
    """步骤1：理解用户查询并生成搜索关键词"""
    
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    try:
        response = llm.invoke([SystemMessage(content=understand_prompt)])
        response_text = sanitize(response.content)
    except Exception as e:
        return {
            "user_query": f"用户想了解: {user_message}",
            "search_query": sanitize(user_message[:400]),
            "step": "understood",
            "messages": [AIMessage(content="理解阶段出错，直接用原始查询搜索")],
            
        }

    search_query = user_message  # 默认使用原始查询

    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()

    # Tavily 限制查询最长 400 字符
    search_query = search_query[:400]

    return {
        "user_query": sanitize(response_text),
        "search_query": sanitize(search_query),
        "step": "understood",
        "messages": [AIMessage(content=sanitize(f"我理解您的需求：{response_text}"))]
    }

# 搜索节点
def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2：使用Tavily API进行真实搜索"""
    
    search_query = state["search_query"]
    
    try:
        print(f"[搜索] 正在搜索: {sanitize(search_query)}")

        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5
        )

        search_results = ""

        if response.get("answer"):
            search_results = f"综合答案：\n{sanitize(response['answer'])}\n\n"

        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = sanitize(result.get("title", ""))
                content = sanitize(result.get("content", ""))
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"

        if not search_results:
            search_results = "抱歉，没有找到相关信息。"

        return {
            "search_results": sanitize(search_results),
            "step": "searched",
            "messages": [AIMessage(content="[OK] 搜索完成，正在为您整理答案...")]
        }

    except Exception as e:
        print(f"[ERROR] 搜索时发生错误: {sanitize(str(e))}")

        return {
            "search_results": "[搜索失败]",
            "step": "search_failed",
            "messages": [AIMessage(content="[WARN] 搜索遇到问题，我将基于已有知识为您回答")]
        }
    
# 生成节点
def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3：基于搜索结果生成最终答案"""
    
    # 检查是否有搜索结果
    if state["step"] == "search_failed":
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
        try:
            response = llm.invoke([SystemMessage(content=fallback_prompt)])
            answer = sanitize(response.content)
        except Exception:
            answer = "抱歉，AI 服务暂时不可用"

        return {
            "final_answer": answer,
            "step": "completed",
            "messages": [AIMessage(content=answer)]
        }

    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    try:
        response = llm.invoke([SystemMessage(content=answer_prompt)])
        answer = sanitize(response.content)
    except Exception:
        answer = "抱歉，AI 服务暂时不可用"

    return {
        "final_answer": answer,
        "step": "completed",
        "messages": [AIMessage(content=answer)]
    }

# 反思节点
def reflection_node(state: SearchState) -> SearchState:

    reflection_prompt = f"""请评估最终生成的答案的质量。
最终生成的答案:{state['final_answer']}

输出：
1.如果你觉得答案足以回答用户的问题时，请严格只输出[满意]。
2.如果你觉得答案不够好时，如回答过于简短，缺乏细节等等,请回顾用户的问题{state['user_query']} 重新生成更好的
搜索关键词。

请严格按照以下格式输出:
1.如果满意,输出:[满意]
2.如果不满意,输出:新的搜索关键词

输出示例：
1.[满意]
2.python异步编程

**注意：请严格按照要求输出，不要添加任何多余的解释或文本**
"""
    reflect_count = state.get("reflect_count", 0) + 1
    try:
        response = llm.invoke([SystemMessage(content=reflection_prompt)])
        reflection_result = sanitize(response.content)
    except Exception:
        reflection_result = "抱歉，AI 服务暂时不可用"
    if "[满意]" in reflection_result:
        return {
            "final_answer": state["final_answer"],
            "step": "satisfied",
            "messages": [AIMessage(content=f"反思结果: {reflection_result}")],
            "reflect_count": reflect_count
        }
    
    else:
        return {
            "search_query": sanitize(reflection_result),
            "step": "reflected",
            "messages": [AIMessage(content=f"重新生成搜索关键词: {reflection_result}")],
            "reflect_count": reflect_count
        }

# 设置条件边函数
def should_reflect(state: SearchState) -> str:
    if state.get("reflect_count", 0) >= 3:
        return END

    if state["step"] == "reflected":
        return "search"
    return END
    

# 构建搜索工作流
def create_search_assistant():
    workflow = StateGraph(SearchState)
    
    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    
    workflow.add_node("reflect", reflection_node)
    
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", "reflect")
    workflow.add_conditional_edges("reflect", should_reflect,["search", END])
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

async def main():
    """主函数：运行智能搜索助手"""
    
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return

    app = create_search_assistant()

    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0
    
    while True:
        user_input = sanitize(input("[问题] 您想了解什么: ").strip())
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start",
            "reflect_count": 0
        }
        
        try:
            print("\n" + "="*60)
            
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"[理解阶段] {sanitize(latest_message.content)}")
                            elif node_name == "search":
                                print(f"[搜索阶段] {sanitize(latest_message.content)}")
                            elif node_name == "answer":
                                print(f"\n[最终回答]\n{sanitize(latest_message.content)}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
           
            print(f"❌ 发生错误: {e}")
            
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    asyncio.run(main())

