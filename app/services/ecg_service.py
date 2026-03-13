"""
ECG 业务服务层 - Tortoise ORM - 优化版
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
from app.core.logger import logger
from app.core.validators import FileValidator
from app.core.exceptions import DataProcessError


class ECGService:
    """ECG 分析服务 - 优化版"""
    
    def __init__(self):
        """初始化服务（使用单例模式的推理引擎）"""
        self.reader = ECGReader()
        self.preprocessor = ECGPreprocessor()
        self.feature_extractor = ECGFeatureExtractor()
        self.inference = ECGInference()  # 单例模式，不会重复加载模型
        logger.info("ECG服务初始化完成")
    
    async def create_task(self, file: UploadFile, user_id: int, background_tasks: Any, algo_mode: str = "fusion"):
        """
        创建分析任务
        
        Args:
            file: 上传的文件
            user_id: 用户ID
            background_tasks: 后台任务管理器
            algo_mode: 算法模式 (fusion/dual/ml_ensemble/dl_advanced/graph)
            
        Returns:
            创建的任务对象
        """
        logger.info(f"用户 {user_id} 上传文件: {file.filename}, 算法模式: {algo_mode}")
        
        # 1. 验证文件
        await FileValidator.validate_upload_file(file)
        
        # 2. 清理文件名
        clean_filename = FileValidator.sanitize_filename(file.filename)
        
        # 3. 保存上传文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"data/{timestamp}_{clean_filename}"
        os.makedirs("data", exist_ok=True)
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.info(f"文件保存成功: {file_path}")
        except Exception as e:
            logger.error(f"文件保存失败: {e}")
            raise DataProcessError(f"文件保存失败: {str(e)}")
        
        # 4. 创建数据库记录
        try:
            user = await User.get(id=user_id)
            task = await ECGTask.create(
                filename=clean_filename,
                file_path=file_path,
                status="pending",
                user=user
            )
            logger.info(f"任务创建成功: ID={task.id}")
        except Exception as e:
            logger.error(f"任务创建失败: {e}")
            # 清理已保存的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            raise DataProcessError(f"任务创建失败: {str(e)}")
        
        # 5. 添加后台任务
        background_tasks.add_task(self._process_task, task.id, file_path, algo_mode)
        logger.info(f"后台任务已添加: task_id={task.id}, algo_mode={algo_mode}")
        
        return task
    
    async def _process_task(self, task_id: int, file_path: str, algo_mode: str = "fusion"):
        """
        处理分析任务（后台执行）
        
        Args:
            task_id: 任务ID
            file_path: 文件路径
            algo_mode: 算法模式
        """
        logger.info(f"开始处理任务: task_id={task_id}, algo_mode={algo_mode}")
        start_time = datetime.now()
        
        try:
            # 1. 读取数据
            logger.info(f"[Task {task_id}] 步骤1: 读取数据")
            ecg_data = self.reader.read(file_path)
            
            # 2. 预处理
            logger.info(f"[Task {task_id}] 步骤2: 预处理")
            processed_data = self.preprocessor.process(ecg_data)
            
            # 3. 特征提取
            logger.info(f"[Task {task_id}] 步骤3: 特征提取")
            features = self.feature_extractor.extract(processed_data)
            
            # 4. 推理/结果计算（根据算法模式）
            logger.info(f"[Task {task_id}] 步骤4: 推理 (模式: {algo_mode})")
            use_fusion = (algo_mode == "fusion")
            result = self.inference.predict(features, use_fusion=use_fusion)
            
            # 添加算法模式信息到结果中
            result['algo_mode'] = algo_mode
            result['algo_mode_display'] = self._get_algo_mode_display(algo_mode)
            
            # 5. 生成 PDF 报告
            logger.info(f"[Task {task_id}] 步骤5: 生成报告")
            from app.services.report_service import ECGReportService
            report_service = ECGReportService()
            task_data = {
                "id": task_id,
                "filename": os.path.basename(file_path),
                "result": result
            }
            report_path = report_service.generate_report(task_data)
            
            # 6. 更新任务状态
            task = await ECGTask.get(id=task_id)
            task.status = "completed"
            task.result = result
            task.report_path = report_path
            task.completed_at = datetime.now()
            await task.save()
            
            # 计算处理时间
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"[Task {task_id}] 处理完成，耗时: {elapsed:.2f}秒")
            logger.info(f"[Task {task_id}] 诊断结果: {result.get('diagnosis')}, 风险: {result.get('risk_level')}")
            
        except Exception as e:
            # 标记任务失败
            logger.error(f"[Task {task_id}] 处理失败: {str(e)}", exc_info=True)
            try:
                task = await ECGTask.get(id=task_id)
                task.status = "failed"
                task.error_message = str(e)
                await task.save()
            except Exception as e2:
                logger.error(f"[Task {task_id}] 更新失败状态时出错: {e2}")
    
    async def get_result(self, task_id: int):
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象或None
        """
        try:
            task = await ECGTask.get(id=task_id)
            logger.debug(f"获取任务结果: task_id={task_id}, status={task.status}")
            return task
        except Exception as e:
            logger.warning(f"任务不存在: task_id={task_id}")
            return None
    
    async def list_tasks(self, skip: int = 0, limit: int = 10, current_user: User = None):
        """
        获取任务列表
        
        Args:
            skip: 跳过数量
            limit: 限制数量
            current_user: 当前用户
            
        Returns:
            任务列表
        """
        logger.debug(f"获取任务列表: user={current_user.username}, role={current_user.role}")
        
        if current_user.role == "admin":
            # 管理员可以看到所有任务
            tasks = await ECGTask.all().offset(skip).limit(limit).order_by('-created_at')
        else:
            # 普通用户只能看到自己的任务
            tasks = await ECGTask.filter(user_id=current_user.id).offset(skip).limit(limit).order_by('-created_at')
        
        logger.debug(f"返回 {len(tasks)} 个任务")
        return tasks

    async def get_signal(self, task_id: int):
        """
        获取任务关联的原始信号数据（自动归一化以适配前端显示）
        
        Args:
            task_id: 任务ID
            
        Returns:
            信号数据字典或None
        """
        try:
            task = await ECGTask.get(id=task_id)
            
            if not os.path.exists(task.file_path):
                logger.warning(f"信号文件不存在: {task.file_path}")
                return None
            
            # 使用 reader 读取信号
            ecg_data = self.reader.read(task.file_path)
            signal = ecg_data['signal']
            
            # 自动归一化信号到 [-1, 1] 范围，以适配前端固定的 scaleY 参数
            import numpy as np
            signal_array = np.array(signal)
            
            # 去除均值
            signal_centered = signal_array - np.mean(signal_array)
            
            # 归一化到 [-1, 1]
            max_abs = np.max(np.abs(signal_centered))
            if max_abs > 0:
                signal_normalized = signal_centered / max_abs
            else:
                signal_normalized = signal_centered
            
            # 转换为列表以便 JSON 序列化
            signal_list = signal_normalized.tolist()
            
            logger.debug(f"获取信号数据: task_id={task_id}, 信号长度={len(signal_list)}, 原始范围=[{signal_array.min():.2f}, {signal_array.max():.2f}], 归一化后范围=[{signal_normalized.min():.2f}, {signal_normalized.max():.2f}]")
            
            return {
                "id": task_id,
                "filename": task.filename,
                "signal": signal_list,
                "sampling_rate": ecg_data.get('sampling_rate', 360)
            }
        except Exception as e:
            logger.error(f"获取信号数据失败: task_id={task_id}, error={e}")
            return None
    
    def _get_algo_mode_display(self, algo_mode: str) -> str:
        """获取算法模式的显示名称"""
        mode_names = {
            'fusion': '多模态融合引擎',
            'dual': '双驱动 (CNN+RF)',
            'ml_ensemble': 'ML集成 (5模型投票)',
            'dl_advanced': '深度学习 (ResNet/Transformer)',
            'graph': '图神经网络 (GCN/GAT)'
        }
        return mode_names.get(algo_mode, '多模态融合引擎')
