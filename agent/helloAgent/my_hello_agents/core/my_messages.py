"""消息系统"""

from typing import Optional, Any, Literal
from datetime import datetime
from  pydantic import BaseModel

# 定义消息角色的类型，限制其角色
MessageRole = Literal["user","assistant","system","tool"]

class Message(BaseModel):
    """
    消息类
    """
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {})
        )
    def to_dict(self) -> dict[str, Any]:
        """
        将消息转换为字典格式，适用于LLM输入
        """
        return {
            "role": self.role,
            "content": self.content,
        }
    
    def __str__(self) -> str:
        """
        消息的字符串表示，包含角色和内容
        """
        return f"[{self.role}] {self.content}"
    