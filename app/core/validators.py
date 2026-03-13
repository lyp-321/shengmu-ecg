"""
数据验证工具
"""
import os
from typing import List
from fastapi import UploadFile

from app.core.exceptions import FileValidationError
from app.core.logger import logger


class FileValidator:
    """文件验证器"""
    
    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {'.csv', '.dat', '.txt', '.edf', '.wfdb'}
    
    # 最大文件大小（50MB）
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # 危险的文件扩展名
    DANGEROUS_EXTENSIONS = {'.exe', '.sh', '.bat', '.cmd', '.com', '.pif', '.scr'}
    
    @classmethod
    async def validate_upload_file(cls, file: UploadFile) -> None:
        """
        验证上传的文件
        
        Args:
            file: 上传的文件对象
            
        Raises:
            FileValidationError: 文件验证失败
        """
        # 1. 检查文件名
        if not file.filename:
            raise FileValidationError("文件名不能为空")
        
        # 2. 检查文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext in cls.DANGEROUS_EXTENSIONS:
            logger.warning(f"检测到危险文件类型: {file.filename}")
            raise FileValidationError(f"不允许上传 {file_ext} 类型的文件")
        
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"不支持的文件类型: {file_ext}。"
                f"支持的类型: {', '.join(cls.ALLOWED_EXTENSIONS)}"
            )
        
        # 3. 检查文件大小
        # 读取文件内容以检查大小
        content = await file.read()
        file_size = len(content)
        
        # 重置文件指针，以便后续读取
        await file.seek(0)
        
        if file_size == 0:
            raise FileValidationError("文件内容为空")
        
        if file_size > cls.MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_mb = cls.MAX_FILE_SIZE / (1024 * 1024)
            raise FileValidationError(
                f"文件过大: {size_mb:.2f}MB，最大允许 {max_mb:.0f}MB"
            )
        
        # 4. 基本内容验证（针对CSV文件）
        if file_ext == '.csv':
            await file.seek(0)
            first_line = (await file.read(1024)).decode('utf-8', errors='ignore')
            await file.seek(0)
            
            # 检查是否包含数字（ECG数据应该包含数值）
            if not any(char.isdigit() for char in first_line):
                raise FileValidationError("CSV文件格式异常，未检测到数值数据")
        
        logger.info(f"文件验证通过: {file.filename} ({file_size / 1024:.2f}KB)")
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        清理文件名，移除危险字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            清理后的文件名
        """
        # 移除路径分隔符和其他危险字符
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        clean_name = filename
        
        for char in dangerous_chars:
            clean_name = clean_name.replace(char, '_')
        
        # 限制文件名长度
        name, ext = os.path.splitext(clean_name)
        if len(name) > 100:
            name = name[:100]
        
        return name + ext


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_heart_rate(hr: float) -> bool:
        """验证心率是否在合理范围内"""
        return 20 <= hr <= 300
    
    @staticmethod
    def validate_signal_length(signal_length: int) -> bool:
        """验证信号长度是否合理"""
        return 100 <= signal_length <= 1000000  # 100点到100万点
    
    @staticmethod
    def validate_sampling_rate(fs: float) -> bool:
        """验证采样率是否合理"""
        return 100 <= fs <= 10000  # 100Hz到10kHz
