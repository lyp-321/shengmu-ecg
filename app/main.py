"""
FastAPI 应用入口
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tortoise.contrib.fastapi import register_tortoise

from app.api import ecg, auth, users
from app.db.session import TORTOISE_ORM

app = FastAPI(
    title="ECG Analysis System",
    description="心电图分析系统 API",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(users.router, prefix="/api/users", tags=["用户管理"])
app.include_router(ecg.router, prefix="/api/ecg", tags=["ECG"])


# 注册 Tortoise ORM
register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=True,  # 自动创建表
    add_exception_handlers=True,
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}

@app.get("/debug/config")
async def debug_config():
    from app.config import settings
    return {
        "secret_key_len": len(settings.SECRET_KEY),
        "secret_key_start": settings.SECRET_KEY[:5],
        "algorithm": settings.ALGORITHM
    }

# 挂载静态文件（前端）
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
