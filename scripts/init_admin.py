"""
创建默认管理员账号的脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from tortoise import Tortoise
from app.db.session import TORTOISE_ORM
from app.db.init_data import init_default_admin


async def main():
    """主函数"""
    # 初始化 Tortoise ORM
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()
    
    # 创建默认管理员
    await init_default_admin()
    
    # 关闭连接
    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
