"""
自定义异常类和全局异常处理器
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logger import logger


class ECGException(Exception):
    """ECG系统基础异常类"""
    def __init__(self, message: str, code: str = "ECG_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class FileValidationError(ECGException):
    """文件验证异常"""
    def __init__(self, message: str):
        super().__init__(message, "FILE_VALIDATION_ERROR")


class ModelLoadError(ECGException):
    """模型加载异常"""
    def __init__(self, message: str):
        super().__init__(message, "MODEL_LOAD_ERROR")


class DataProcessError(ECGException):
    """数据处理异常"""
    def __init__(self, message: str):
        super().__init__(message, "DATA_PROCESS_ERROR")


class InferenceError(ECGException):
    """推理异常"""
    def __init__(self, message: str):
        super().__init__(message, "INFERENCE_ERROR")


async def ecg_exception_handler(request: Request, exc: ECGException):
    """ECG自定义异常处理器"""
    logger.error(f"ECG异常: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.code,
            "message": exc.message,
            "path": str(request.url)
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    logger.warning(f"请求验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "请求参数验证失败",
            "details": exc.errors()
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未捕获的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "服务器内部错误，请稍后重试",
            "detail": str(exc) if __debug__ else None  # 开发环境显示详情
        }
    )


def register_exception_handlers(app):
    """注册所有异常处理器"""
    app.add_exception_handler(ECGException, ecg_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
