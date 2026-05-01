"""配置管理"""

import os
from typing import Optional, Any
from pydantic import BaseModel
class Config(BaseModel):
    """
    HelloAgents的配置类，包含LLM相关的配置项
    """
    # LLM配置
    default_model: str = "deepseek-v4-flash"
    default_provider: str = "deepseek"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置
        """
        return cls(
            debug = os.getenv("DEBUG", "false").lower() == "true",
            log_level = os.getenv("LOG_LEVEL", "INFO"),
            temperature = float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens = int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典
        """
        return self.model_dump() # 使用pydantic的model_dump方法转换为字典
