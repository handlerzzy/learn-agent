import asyncio
import concurrent.futures # 用于线程池执行工具
from typing import Any, Callable
from hello_agents import ToolRegistry

class AsyncToolExecutor:
    def __init__(self, registry: ToolRegistry,max_workers: int=4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) # 创建线程池

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行工具"""
        loop = asyncio.get_event_loop() # 获取事件循环
        def _execute():
            return self.registry.execute_tool(tool_name, input_data) # 在线程池中执行工具
        result = await loop.run_in_executor(self.executor, _execute) # 在事件循环中等待工具执行完成
        return result
    
    async def execute_tools_parallel(self, tasks: list[dict[str, str]]) -> list[str]:
        """
        并行执行多个工具
        """
        print(f"🔧 并行执行工具任务: {tasks}")
        # 创建异步任务
        async_tasks = []
        for task in tasks:
            tool_name = task["tool_name"]
            input_data = task["input_data"]
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append(async_task) # 将异步任务添加到列表中
        results = await asyncio.gather(*async_tasks) # 等待所有异步任务完成并获取结果
        print(f"✅ 并行工具任务完成，结果: {results}")
        return results
    
    def __del__(self):
        """清理线程池资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)