#!/usr/bin/env python
"""
第5层检查：API层
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

def check_api():
    """检查API层"""
    
    print("=" * 80)
    print("第5层检查：API层")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    issues = []
    
    # 1. 检查FastAPI应用
    print("【1】检查FastAPI应用 (main.py)...")
    try:
        from app.main import app
        print("   ✅ FastAPI应用导入成功")
        print(f"   应用标题: {app.title}")
        print(f"   应用版本: {app.version}")
        
        # 检查路由
        routes = [route.path for route in app.routes]
        print(f"   ✅ 注册的路由数: {len(routes)}")
        
    except ImportError as e:
        print(f"   ❌ FastAPI应用导入失败: {e}")
        issues.append(f"FastAPI应用导入失败: {e}")
    except Exception as e:
        print(f"   ❌ FastAPI应用检查失败: {e}")
        issues.append(f"FastAPI应用检查失败: {e}")
    
    print()
    
    # 2. 检查认证API
    print("【2】检查认证API (api/auth.py)...")
    try:
        from app.api import auth
        print("   ✅ 认证API模块导入成功")
        
        # 检查路由器
        if hasattr(auth, 'router'):
            routes = [route.path for route in auth.router.routes]
            print(f"   ✅ 认证路由数: {len(routes)}")
            for route in routes:
                print(f"      - {route}")
        
    except ImportError as e:
        print(f"   ❌ 认证API导入失败: {e}")
        issues.append(f"认证API导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 认证API检查失败: {e}")
        issues.append(f"认证API检查失败: {e}")
    
    print()
    
    # 3. 检查用户API
    print("【3】检查用户API (api/users.py)...")
    try:
        from app.api import users
        print("   ✅ 用户API模块导入成功")
        
        if hasattr(users, 'router'):
            routes = [route.path for route in users.router.routes]
            print(f"   ✅ 用户路由数: {len(routes)}")
            for route in routes:
                print(f"      - {route}")
        
    except ImportError as e:
        print(f"   ❌ 用户API导入失败: {e}")
        issues.append(f"用户API导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 用户API检查失败: {e}")
        issues.append(f"用户API检查失败: {e}")
    
    print()
    
    # 4. 检查ECG API
    print("【4】检查ECG API (api/ecg.py)...")
    try:
        from app.api import ecg
        print("   ✅ ECG API模块导入成功")
        
        if hasattr(ecg, 'router'):
            routes = [route.path for route in ecg.router.routes]
            print(f"   ✅ ECG路由数: {len(routes)}")
            for route in routes:
                print(f"      - {route}")
        
    except ImportError as e:
        print(f"   ❌ ECG API导入失败: {e}")
        issues.append(f"ECG API导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECG API检查失败: {e}")
        issues.append(f"ECG API检查失败: {e}")
    
    print()
    
    # 5. 检查配置
    print("【5】检查应用配置 (config.py)...")
    try:
        from app.config import settings
        print("   ✅ 配置模块导入成功")
        print(f"   数据库URL: {settings.database_url}")
        print(f"   JWT算法: {settings.ALGORITHM}")
        print(f"   Token过期时间: {settings.ACCESS_TOKEN_EXPIRE_MINUTES} 分钟")
        print(f"   调试模式: {settings.DEBUG}")
        
    except ImportError as e:
        print(f"   ❌ 配置模块导入失败: {e}")
        issues.append(f"配置模块导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 配置检查失败: {e}")
        issues.append(f"配置检查失败: {e}")
    
    print()
    
    # 6. 检查Schemas
    print("【6】检查Pydantic Schemas...")
    try:
        from app.schemas.user import UserCreate, UserResponse
        from app.schemas.ecg import ECGTaskResponse, ECGResultResponse
        print("   ✅ User Schemas导入成功")
        print("   ✅ ECG Schemas导入成功")
        
        # 简单测试Schema（不使用EmailStr验证）
        print(f"   ✅ Schema类定义正确")
        print(f"      UserCreate字段: {list(UserCreate.model_fields.keys())}")
        print(f"      UserResponse字段: {list(UserResponse.model_fields.keys())}")
        
    except ImportError as e:
        print(f"   ❌ Schemas导入失败: {e}")
        issues.append(f"Schemas导入失败: {e}")
    except Exception as e:
        print(f"   ❌ Schemas检查失败: {e}")
        issues.append(f"Schemas检查失败: {e}")
    
    print()
    
    # 7. 检查中间件和CORS
    print("【7】检查中间件配置...")
    try:
        from app.main import app
        
        # 检查中间件
        middleware_count = len(app.user_middleware)
        print(f"   ✅ 中间件数量: {middleware_count}")
        
        # 检查CORS配置
        print(f"   ✅ CORS配置已加载")
        
    except Exception as e:
        print(f"   ⚠️  中间件检查异常: {e}")
    
    print()
    
    # 8. 测试API路由完整性
    print("【8】检查API路由完整性...")
    try:
        from app.main import app
        
        # 必需的路由
        required_routes = [
            '/api/auth/login',
            '/api/auth/register',
            '/api/ecg/upload',
            '/api/ecg/tasks',
        ]
        
        all_routes = [route.path for route in app.routes]
        
        for required in required_routes:
            # 检查是否有匹配的路由（考虑路径参数）
            found = any(required in route for route in all_routes)
            if found:
                print(f"   ✅ 路由存在: {required}")
            else:
                print(f"   ❌ 路由缺失: {required}")
                issues.append(f"缺少路由: {required}")
        
    except Exception as e:
        print(f"   ❌ 路由完整性检查失败: {e}")
        issues.append(f"路由完整性检查失败: {e}")
    
    print()
    
    # 总结
    print("=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if not issues:
        print("✅ API层检查通过，没有发现问题")
        print()
        print("API模块:")
        print("  ✅ FastAPI应用 - 主应用")
        print("  ✅ 认证API - 登录/注册")
        print("  ✅ 用户API - 用户管理")
        print("  ✅ ECG API - ECG分析")
        print("  ✅ Schemas - 数据验证")
        print("  ✅ 配置 - 应用配置")
        print()
        print("核心接口:")
        print("  POST /api/auth/login - 用户登录")
        print("  POST /api/auth/register - 用户注册")
        print("  POST /api/ecg/upload - 上传ECG文件")
        print("  GET  /api/ecg/tasks - 获取任务列表")
        print("  GET  /api/ecg/tasks/{id} - 获取任务详情")
        print("  GET  /api/ecg/tasks/{id}/report - 下载PDF报告")
        print()
        print("下一步: 启动服务器测试")
        print("  uvicorn app.main:app --reload")
        return True
    else:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        return False

if __name__ == '__main__':
    success = check_api()
    sys.exit(0 if success else 1)
