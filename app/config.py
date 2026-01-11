"""
数据库配置文件
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    """应用配置"""
    
    # MySQL 数据库配置
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ecg_system")
    
    @property
    def database_url(self) -> str:
        """生成数据库连接 URL"""
        return f"mysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    # JWT 认证配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 天
    
    # 应用配置
    APP_TITLE: str = "ECG Analysis System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"


# 全局配置实例
settings = Settings()


