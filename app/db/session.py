"""
Tortoise ORM 配置
"""
from app.config import settings

# Tortoise ORM 配置
TORTOISE_ORM = {
    "connections": {
        "default": settings.database_url
    },
    "apps": {
        "models": {
            "models": ["app.models.task", "app.models.user", "aerich.models"],
            "default_connection": "default",
        }
    },
    "use_tz": False,
    "timezone": "Asia/Shanghai"
}
