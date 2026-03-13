import asyncio
from tortoise import Tortoise
from app.db.session import TORTOISE_ORM

async def migrate():
    await Tortoise.init(config=TORTOISE_ORM)
    conn = Tortoise.get_connection('default')
    
    # 修正后的表名探测
    table_name = 'ecg_tasks'
    try:
        await conn.execute_script(f'ALTER TABLE {table_name} ADD COLUMN report_path VARCHAR(512) NULL;')
        print(f'Migration successful: report_path added to {table_name}')
    except Exception as e:
        print(f'Migration skipped or failed for {table_name}: {e}')

    await Tortoise.close_connections()

if __name__ == "__main__":
    asyncio.run(migrate())
