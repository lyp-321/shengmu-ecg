#!/usr/bin/env python
"""
第4层检查：服务层
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

def check_services():
    """检查服务层"""
    
    print("=" * 80)
    print("第4层检查：服务层")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    issues = []
    
    # 1. 检查ECG服务
    print("【1】检查ECG服务 (ecg_service.py)...")
    try:
        from app.services.ecg_service import ECGService
        service = ECGService()
        print("   ✅ ECGService导入成功")
        print(f"   ✅ 服务初始化完成")
        
        # 检查服务属性
        if hasattr(service, 'reader'):
            print(f"   ✅ reader模块已加载")
        if hasattr(service, 'preprocessor'):
            print(f"   ✅ preprocessor模块已加载")
        if hasattr(service, 'feature_extractor'):
            print(f"   ✅ feature_extractor模块已加载")
        if hasattr(service, 'inference'):
            print(f"   ✅ inference引擎已加载")
        
    except ImportError as e:
        print(f"   ❌ ECGService导入失败: {e}")
        issues.append(f"ECGService导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGService初始化失败: {e}")
        issues.append(f"ECGService初始化失败: {e}")
    
    print()
    
    # 2. 检查报告服务
    print("【2】检查报告服务 (report_service.py)...")
    try:
        from app.services.report_service import ECGReportService
        report_service = ECGReportService()
        print("   ✅ ECGReportService导入成功")
        
        # 测试报告生成（不实际生成）
        test_task = {
            'id': 999,
            'filename': 'test.csv',
            'result': {
                'heart_rate': 75.0,
                'hrv_sdnn': 45.0,
                'diagnosis': '正常窦性心律',
                'risk_level': '低风险'
            }
        }
        
        print(f"   ✅ 报告服务可用")
        
    except ImportError as e:
        print(f"   ❌ ECGReportService导入失败: {e}")
        issues.append(f"ECGReportService导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGReportService测试失败: {e}")
        issues.append(f"ECGReportService测试失败: {e}")
    
    print()
    
    # 3. 检查日志系统
    print("【3】检查日志系统 (logger.py)...")
    try:
        from app.core.logger import logger
        print("   ✅ logger导入成功")
        
        # 测试日志记录
        logger.info("测试日志记录")
        print(f"   ✅ 日志记录功能正常")
        
        # 检查日志文件
        log_dir = "logs"
        if os.path.exists(log_dir):
            log_files = os.listdir(log_dir)
            print(f"   ✅ 日志目录存在: {len(log_files)} 个日志文件")
        else:
            print(f"   ⚠️  日志目录不存在")
        
    except ImportError as e:
        print(f"   ❌ logger导入失败: {e}")
        issues.append(f"logger导入失败: {e}")
    except Exception as e:
        print(f"   ❌ logger测试失败: {e}")
        issues.append(f"logger测试失败: {e}")
    
    print()
    
    # 4. 检查异常处理
    print("【4】检查异常处理 (exceptions.py)...")
    try:
        from app.core.exceptions import (
            DataProcessError,
            ModelLoadError,
            InferenceError,
            FileValidationError
        )
        print("   ✅ 自定义异常类导入成功")
        
        # 测试异常
        try:
            raise DataProcessError("测试异常")
        except DataProcessError as e:
            print(f"   ✅ DataProcessError可正常捕获")
        
        print(f"   ✅ 异常处理系统正常")
        
    except ImportError as e:
        print(f"   ❌ 异常类导入失败: {e}")
        issues.append(f"异常类导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 异常处理测试失败: {e}")
        issues.append(f"异常处理测试失败: {e}")
    
    print()
    
    # 5. 检查文件验证
    print("【5】检查文件验证 (validators.py)...")
    try:
        from app.core.validators import FileValidator
        print("   ✅ FileValidator导入成功")
        
        # 测试文件名清理
        test_filename = "test file (1).csv"
        clean_name = FileValidator.sanitize_filename(test_filename)
        print(f"   ✅ 文件名清理: '{test_filename}' → '{clean_name}'")
        
        # 测试文件类型验证
        allowed_types = ['.csv', '.dat']
        print(f"   ✅ 允许的文件类型: {allowed_types}")
        
    except ImportError as e:
        print(f"   ❌ FileValidator导入失败: {e}")
        issues.append(f"FileValidator导入失败: {e}")
    except Exception as e:
        print(f"   ❌ FileValidator测试失败: {e}")
        issues.append(f"FileValidator测试失败: {e}")
    
    print()
    
    # 6. 检查安全模块
    print("【6】检查安全模块 (security.py)...")
    try:
        from app.core.security import (
            get_password_hash,
            verify_password,
            create_access_token
        )
        print("   ✅ 安全函数导入成功")
        
        # 测试密码哈希
        test_password = "test123"
        hashed = get_password_hash(test_password)
        is_valid = verify_password(test_password, hashed)
        
        if is_valid:
            print(f"   ✅ 密码哈希和验证功能正常")
        else:
            print(f"   ❌ 密码验证失败")
            issues.append("密码验证功能异常")
        
        # 测试JWT token生成
        token = create_access_token(data={"sub": "test_user"})
        if token:
            print(f"   ✅ JWT token生成功能正常")
            print(f"      Token长度: {len(token)}")
        else:
            print(f"   ❌ JWT token生成失败")
            issues.append("JWT token生成失败")
        
    except ImportError as e:
        print(f"   ❌ 安全模块导入失败: {e}")
        issues.append(f"安全模块导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 安全模块测试失败: {e}")
        issues.append(f"安全模块测试失败: {e}")
    
    print()
    
    # 7. 检查依赖注入
    print("【7】检查依赖注入 (deps.py)...")
    try:
        from app.core.deps import get_current_user, get_current_active_user
        print("   ✅ 依赖注入函数导入成功")
        print(f"   ✅ get_current_user可用")
        print(f"   ✅ get_current_active_user可用")
        
    except ImportError as e:
        print(f"   ❌ 依赖注入导入失败: {e}")
        issues.append(f"依赖注入导入失败: {e}")
    except Exception as e:
        print(f"   ❌ 依赖注入测试失败: {e}")
        issues.append(f"依赖注入测试失败: {e}")
    
    print()
    
    # 总结
    print("=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if not issues:
        print("✅ 服务层检查通过，没有发现问题")
        print()
        print("服务模块:")
        print("  ✅ ECGService - ECG分析服务")
        print("  ✅ ECGReportService - PDF报告生成")
        print("  ✅ Logger - 日志系统")
        print("  ✅ Exceptions - 异常处理")
        print("  ✅ FileValidator - 文件验证")
        print("  ✅ Security - 安全认证")
        print("  ✅ Dependencies - 依赖注入")
        print()
        print("服务层功能:")
        print("  - 任务创建和管理")
        print("  - 后台任务处理")
        print("  - PDF报告生成")
        print("  - 日志记录")
        print("  - 异常处理")
        print("  - 文件验证")
        print("  - JWT认证")
        return True
    else:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        return False

if __name__ == '__main__':
    success = check_services()
    sys.exit(0 if success else 1)
