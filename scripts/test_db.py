"""
测试数据库连接
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from tortoise import Tortoise
from app.db.session import TORTOISE_ORM


async def test_connection():
    """测试 MySQL 连接"""
    try:
        # 初始化 Tortoise ORM
        await Tortoise.init(config=TORTOISE_ORM)
        
        # 生成数据库表
        await Tortoise.generate_schemas()
        
        print("✅ 数据库连接成功！")
        print(f"✅ 数据库表已创建")
        
        # 关闭连接
        await Tortoise.close_connections()
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print("\n请检查：")
        print("1. MySQL 服务是否启动")
        print("2. 数据库 'ecg_system' 是否已创建")
        print("3. app/config.py 中的连接配置是否正确")


if __name__ == "__main__":
    asyncio.run(test_connection())
