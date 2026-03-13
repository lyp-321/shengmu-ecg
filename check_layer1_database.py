#!/usr/bin/env python
"""
第1层检查：数据库层
"""
import sqlite3
import os
from datetime import datetime

def check_database():
    """检查数据库层"""
    
    print("=" * 80)
    print("第1层检查：数据库层")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    db_path = "ecg_system.db"
    issues = []
    
    # 1. 检查数据库文件是否存在
    print("【1】检查数据库文件...")
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"   ✅ 数据库文件存在: {db_path}")
        print(f"   文件大小: {size / 1024:.2f} KB")
    else:
        print(f"   ❌ 数据库文件不存在: {db_path}")
        issues.append("数据库文件不存在")
        return issues
    
    print()
    
    # 连接数据库
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"   ❌ 无法连接数据库: {e}")
        issues.append(f"数据库连接失败: {e}")
        return issues
    
    # 2. 检查表结构
    print("【2】检查表结构...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = ['users', 'ecg_tasks']
    for table in required_tables:
        if table in tables:
            print(f"   ✅ 表 '{table}' 存在")
        else:
            print(f"   ❌ 表 '{table}' 不存在")
            issues.append(f"缺少表: {table}")
    
    print()
    
    # 3. 检查users表结构
    if 'users' in tables:
        print("【3】检查users表结构...")
        cursor.execute("PRAGMA table_info(users);")
        columns = cursor.fetchall()
        
        required_columns = ['id', 'username', 'email', 'hashed_password', 'full_name', 'role', 'is_active']
        existing_columns = [col[1] for col in columns]
        
        for col in required_columns:
            if col in existing_columns:
                print(f"   ✅ 字段 '{col}' 存在")
            else:
                print(f"   ❌ 字段 '{col}' 不存在")
                issues.append(f"users表缺少字段: {col}")
        
        # 检查数据
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        print(f"   用户总数: {user_count}")
        
        if user_count == 0:
            print(f"   ⚠️  警告: 没有用户数据")
            issues.append("没有用户数据")
        else:
            # 显示用户列表
            cursor.execute("SELECT id, username, role, is_active FROM users;")
            users = cursor.fetchall()
            print(f"   用户列表:")
            for user in users:
                status = "激活" if user[3] else "禁用"
                print(f"      - ID:{user[0]}, 用户名:{user[1]}, 角色:{user[2]}, 状态:{status}")
    
    print()
    
    # 4. 检查ecg_tasks表结构
    if 'ecg_tasks' in tables:
        print("【4】检查ecg_tasks表结构...")
        cursor.execute("PRAGMA table_info(ecg_tasks);")
        columns = cursor.fetchall()
        
        required_columns = ['id', 'filename', 'file_path', 'status', 'result', 'user_id']
        existing_columns = [col[1] for col in columns]
        
        for col in required_columns:
            if col in existing_columns:
                print(f"   ✅ 字段 '{col}' 存在")
            else:
                print(f"   ❌ 字段 '{col}' 不存在")
                issues.append(f"ecg_tasks表缺少字段: {col}")
        
        # 检查数据
        cursor.execute("SELECT COUNT(*) FROM ecg_tasks;")
        task_count = cursor.fetchone()[0]
        print(f"   任务总数: {task_count}")
        
        if task_count > 0:
            # 统计各状态任务数
            cursor.execute("SELECT status, COUNT(*) FROM ecg_tasks GROUP BY status;")
            status_counts = cursor.fetchall()
            print(f"   任务状态统计:")
            for status, count in status_counts:
                print(f"      - {status}: {count}")
            
            # 显示最近5个任务
            cursor.execute("""
                SELECT id, filename, status, created_at 
                FROM ecg_tasks 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_tasks = cursor.fetchall()
            print(f"   最近5个任务:")
            for task in recent_tasks:
                print(f"      - ID:{task[0]}, 文件:{task[1]}, 状态:{task[2]}, 时间:{task[3]}")
    
    print()
    
    # 5. 检查外键关系
    print("【5】检查外键关系...")
    if 'ecg_tasks' in tables:
        cursor.execute("PRAGMA foreign_key_list(ecg_tasks);")
        foreign_keys = cursor.fetchall()
        
        if foreign_keys:
            print(f"   ✅ 外键关系已定义")
            for fk in foreign_keys:
                print(f"      - {fk[2]}.{fk[3]} → {fk[4]}")
        else:
            print(f"   ⚠️  警告: 没有外键约束")
    
    print()
    
    # 6. 检查索引
    print("【6】检查索引...")
    cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index';")
    indexes = cursor.fetchall()
    
    if indexes:
        print(f"   索引总数: {len(indexes)}")
        for idx in indexes:
            if not idx[0].startswith('sqlite_'):  # 跳过系统索引
                print(f"      - {idx[0]} on {idx[1]}")
    else:
        print(f"   ⚠️  警告: 没有自定义索引")
    
    conn.close()
    
    # 总结
    print()
    print("=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if not issues:
        print("✅ 数据库层检查通过，没有发现问题")
        return True
    else:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print()
        print("建议修复方案:")
        if "数据库文件不存在" in issues or "缺少表" in str(issues):
            print("   → 运行: python scripts/init_admin.py")
        if "没有用户数据" in issues:
            print("   → 运行: python scripts/init_admin.py")
        return False

if __name__ == '__main__':
    import sys
    success = check_database()
    sys.exit(0 if success else 1)
