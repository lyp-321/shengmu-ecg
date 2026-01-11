"""
ECG 业务服务层 - Tortoise ORM
"""
import os
from datetime import datetime
from typing import Any
from fastapi import UploadFile

from app.models.task import ECGTask
from app.models.user import User
from app.algorithms.reader import ECGReader
from app.algorithms.preprocess import ECGPreprocessor
from app.algorithms.features import ECGFeatureExtractor
from app.algorithms.inference import ECGInference


class ECGService:
    """ECG 分析服务"""
    
    def __init__(self):
        self.reader = ECGReader()
        self.preprocessor = ECGPreprocessor()
        self.feature_extractor = ECGFeatureExtractor()
        self.inference = ECGInference()
    
    async def create_task(self, file: UploadFile, user_id: int, background_tasks: Any):
        """创建分析任务"""
        # 保存上传文件
        file_path = f"data/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        os.makedirs("data", exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 创建数据库记录（Tortoise ORM 异步创建）
        user = await User.get(id=user_id)
        task = await ECGTask.create(
            filename=file.filename,
            file_path=file_path,
            status="pending",
            user=user
        )
        
        # 使用后台任务执行分析
        background_tasks.add_task(self._process_task, task.id, file_path)
        
        return task
    
    async def _process_task(self, task_id: int, file_path: str):
        """处理分析任务"""
        try:
            # 1. 读取数据
            ecg_data = self.reader.read(file_path)
            
            # 2. 预处理
            processed_data = self.preprocessor.process(ecg_data)
            
            # 3. 特征提取
            features = self.feature_extractor.extract(processed_data)
            
            # 4. 推理/结果计算
            result = self.inference.predict(features)
            
            # 更新任务状态（Tortoise ORM 异步更新）
            task = await ECGTask.get(id=task_id)
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now()
            await task.save()
            
        except Exception as e:
            # 标记任务失败
            task = await ECGTask.get(id=task_id)
            task.status = "failed"
            task.error_message = str(e)
            await task.save()
    
    async def get_result(self, task_id: int):
        """获取任务结果"""
        try:
            task = await ECGTask.get(id=task_id)
            return task
        except:
            return None
    
    async def list_tasks(self, skip: int = 0, limit: int = 10, current_user: User = None):
        """获取任务列表"""
        if current_user.role == "admin":
            # 管理员可以看到所有任务
            tasks = await ECGTask.all().offset(skip).limit(limit)
        else:
            # 普通用户只能看到自己的任务
            tasks = await ECGTask.filter(user_id=current_user.id).offset(skip).limit(limit)
        return tasks
