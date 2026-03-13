"""
FastAPI 应用入口 - 优化版
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise

from app.api import ecg, auth, users
from app.db.session import TORTOISE_ORM
from app.core.logger import logger
from app.core.exceptions import register_exception_handlers

# 创建应用实例
app = FastAPI(
    title="ECG Analysis System",
    description="智能心电图分析系统 API - 基于多模态深度学习",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 注册异常处理器
register_exception_handlers(app)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(users.router, prefix="/api/users", tags=["用户管理"])
app.include_router(ecg.router, prefix="/api/ecg", tags=["ECG分析"])

# 注册 Tortoise ORM
register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,  # 自动创建表
    add_exception_handlers=True,
)


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("=" * 60)
    logger.info("ECG智能心电图分析系统启动中...")
    logger.info("版本: 2.0.0")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("ECG系统正在关闭...")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "service": "ECG Analysis System"
    }


@app.get("/debug/config")
async def debug_config():
    """调试配置接口（仅开发环境）"""
    from app.config import settings
    return {
        "secret_key_len": len(settings.SECRET_KEY),
        "secret_key_start": settings.SECRET_KEY[:5],
        "algorithm": settings.ALGORITHM
    }


# 挂载静态文件（前端）
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
