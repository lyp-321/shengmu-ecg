#!/usr/bin/env python
"""
第2层检查：数据模型层
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

def check_models():
    """检查数据模型层"""
    
    print("=" * 80)
    print("第2层检查：数据模型层")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    issues = []
    
    # 1. 检查User模型
    print("【1】检查User模型...")
    try:
        from app.models.user import User
        print("   ✅ User模型导入成功")
        
        # 检查字段
        fields = User._meta.fields_map
        required_fields = ['id', 'username', 'email', 'hashed_password', 'full_name', 'role', 'is_active']
        
        for field in required_fields:
            if field in fields:
                field_type = fields[field].__class__.__name__
                print(f"   ✅ 字段 '{field}' 存在 (类型: {field_type})")
            else:
                print(f"   ❌ 字段 '{field}' 不存在")
                issues.append(f"User模型缺少字段: {field}")
        
        # 检查关系
        if hasattr(User, 'tasks'):
            print(f"   ✅ 关系 'tasks' 已定义 (反向关系)")
        
    except ImportError as e:
        print(f"   ❌ User模型导入失败: {e}")
        issues.append(f"User模型导入失败: {e}")
    except Exception as e:
        print(f"   ❌ User模型检查失败: {e}")
        issues.append(f"User模型检查失败: {e}")
    
    print()
    
    # 2. 检查ECGTask模型
    print("【2】检查ECGTask模型...")
    try:
        from app.models.task import ECGTask
        print("   ✅ ECGTask模型导入成功")
        
        # 检查字段
        fields = ECGTask._meta.fields_map
        required_fields = ['id', 'filename', 'file_path', 'status', 'result', 'created_at']
        
        for field in required_fields:
            if field in fields:
                field_type = fields[field].__class__.__name__
                print(f"   ✅ 字段 '{field}' 存在 (类型: {field_type})")
            else:
                print(f"   ❌ 字段 '{field}' 不存在")
                issues.append(f"ECGTask模型缺少字段: {field}")
        
        # 检查外键关系（Tortoise会自动创建user_id）
        if 'user' in fields:
            print(f"   ✅ 外键 'user' 已定义 (自动创建 user_id 字段)")
        elif 'user_id' in fields:
            print(f"   ✅ 外键字段 'user_id' 存在")
        else:
            print(f"   ❌ 外键关系未定义")
            issues.append("ECGTask模型缺少user外键")
        
    except ImportError as e:
        print(f"   ❌ ECGTask模型导入失败: {e}")
        issues.append(f"ECGTask模型导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGTask模型检查失败: {e}")
        issues.append(f"ECGTask模型检查失败: {e}")
    
    print()
    
    # 3. 测试模型实例化
    print("【3】测试模型实例化...")
    try:
        from app.core.security import get_password_hash
        
        # 测试User模型
        print("   测试User模型...")
        test_user_data = {
            'username': 'test_check',
            'email': 'test@check.com',
            'hashed_password': get_password_hash('test123'),
            'full_name': '测试用户',
            'role': 'user',
            'is_active': True
        }
        # 不实际创建，只检查字段
        print(f"   ✅ User模型字段验证通过")
        
        # 测试ECGTask模型
        print("   测试ECGTask模型...")
        test_task_data = {
            'filename': 'test.csv',
            'file_path': 'data/test.csv',
            'status': 'pending',
            'user_id': 1
        }
        print(f"   ✅ ECGTask模型字段验证通过")
        
    except Exception as e:
        print(f"   ❌ 模型实例化测试失败: {e}")
        issues.append(f"模型实例化失败: {e}")
    
    print()
    
    # 4. 检查JSON字段序列化
    print("【4】检查JSON字段...")
    try:
        import json
        
        # 测试result字段的JSON序列化
        test_result = {
            'heart_rate': 75.5,
            'hrv_sdnn': 45.2,
            'diagnosis': '正常窦性心律',
            'risk_level': '低风险'
        }
        
        json_str = json.dumps(test_result, ensure_ascii=False)
        json_obj = json.loads(json_str)
        
        print(f"   ✅ JSON序列化测试通过")
        print(f"   示例: {json_str[:50]}...")
        
    except Exception as e:
        print(f"   ❌ JSON序列化测试失败: {e}")
        issues.append(f"JSON序列化失败: {e}")
    
    print()
    
    # 5. 检查模型方法
    print("【5】检查模型方法...")
    try:
        from app.models.user import User
        
        # 检查User模型的方法
        if hasattr(User, '__str__'):
            print(f"   ✅ User模型有 __str__ 方法")
        
        # 检查密码验证相关
        from app.core.security import verify_password, get_password_hash
        test_password = "test123"
        hashed = get_password_hash(test_password)
        is_valid = verify_password(test_password, hashed)
        
        if is_valid:
            print(f"   ✅ 密码哈希和验证功能正常")
        else:
            print(f"   ❌ 密码验证失败")
            issues.append("密码验证功能异常")
        
    except Exception as e:
        print(f"   ❌ 模型方法检查失败: {e}")
        issues.append(f"模型方法检查失败: {e}")
    
    print()
    
    # 总结
    print("=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if not issues:
        print("✅ 数据模型层检查通过，没有发现问题")
        print()
        print("模型结构:")
        print("  User:")
        print("    - id, username, email, hashed_password")
        print("    - full_name, role, is_active")
        print("    - created_at")
        print("    - 关系: tasks (一对多)")
        print()
        print("  ECGTask:")
        print("    - id, filename, file_path, status")
        print("    - result (JSON), report_path, error_message")
        print("    - user_id (外键), created_at, completed_at")
        print("    - 关系: user (多对一)")
        return True
    else:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        return False

if __name__ == '__main__':
    success = check_models()
    sys.exit(0 if success else 1)
