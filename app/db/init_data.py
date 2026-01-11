"""
初始化默认数据
"""
from app.models.user import User
from app.core.security import get_password_hash


async def init_default_admin():
    """创建默认管理员账号"""
    # 检查是否已存在管理员
    admin = await User.get_or_none(username="admin")
    if not admin:
        # 创建默认管理员
        await User.create(
            username="admin",
            email="admin@ecg-system.com",
            hashed_password=get_password_hash("admin123"),
            full_name="系统管理员",
            role="admin",
            is_active=True
        )
        print("✅ 默认管理员账号已创建")
        print("   用户名: admin")
        print("   密码: admin123")
        print("   ⚠️  请尽快修改默认密码！")
    else:
        print("✅ 管理员账号已存在")
