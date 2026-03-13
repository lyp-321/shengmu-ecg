"""
Pydantic 数据模型
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class ECGTaskResponse(BaseModel):
    """ECG 任务响应"""
    id: int
    filename: str
    status: str
    result: Optional[Dict[str, Any]] = None  # 添加result字段，用于前端过滤和显示
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
        from_attributes = True


class ECGResultResponse(BaseModel):
    """ECG 结果响应"""
    id: int
    filename: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
