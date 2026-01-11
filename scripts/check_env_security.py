#!/usr/bin/env python3
"""
环境变量安全检查脚本
"""
import os
from pathlib import Path


def check_env_security():
    """检查环境变量配置的安全性"""
    
    print("🔒 环境变量安全检查\n")
    print("=" * 50)
    
    # 检查 .env 文件
    base_dir = Path(__file__).parent.parent
    env_file = base_dir / ".env"
    env_example = base_dir / ".env.example"
    gitignore = base_dir / ".gitignore"
    
    checks = []
    
    # 1. 检查 .env 文件是否存在
    if env_file.exists():
        checks.append(("✅", ".env 文件存在"))
    else:
        checks.append(("❌", ".env 文件不存在，请创建"))
    
    # 2. 检查 .env.example 文件
    if env_example.exists():
        checks.append(("✅", ".env.example 模板文件存在"))
    else:
        checks.append(("⚠️", ".env.example 模板文件不存在"))
    
    # 3. 检查 .gitignore
    if gitignore.exists():
        with open(gitignore, 'r') as f:
            content = f.read()
            if '.env' in content:
                checks.append(("✅", ".env 已在 .gitignore 中"))
            else:
                checks.append(("❌", ".env 未在 .gitignore 中，有泄露风险！"))
    else:
        checks.append(("❌", ".gitignore 文件不存在"))
    
    # 4. 检查环境变量是否加载
    try:
        from dotenv import load_dotenv
        load_dotenv()
        checks.append(("✅", "python-dotenv 已安装"))
    except ImportError:
        checks.append(("❌", "python-dotenv 未安装"))
    
    # 5. 检查关键环境变量
    required_vars = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_DATABASE']
    for var in required_vars:
        if os.getenv(var):
            checks.append(("✅", f"环境变量 {var} 已设置"))
        else:
            checks.append(("⚠️", f"环境变量 {var} 未设置"))
    
    # 6. 检查密码是否为默认值
    password = os.getenv('MYSQL_PASSWORD', '')
    if password and password != 'your_password_here':
        checks.append(("✅", "MYSQL_PASSWORD 已设置为实际密码"))
    else:
        checks.append(("⚠️", "MYSQL_PASSWORD 仍为默认值或未设置"))
    
    # 输出检查结果
    print("\n检查结果：\n")
    for status, message in checks:
        print(f"{status} {message}")
    
    # 统计
    success = sum(1 for s, _ in checks if s == "✅")
    total = len(checks)
    
    print("\n" + "=" * 50)
    print(f"\n通过检查：{success}/{total}")
    
    if success == total:
        print("\n🎉 恭喜！您的环境变量配置完全安全！")
    elif success >= total * 0.7:
        print("\n✅ 配置基本安全，但有一些建议需要改进")
    else:
        print("\n⚠️ 请修复上述问题以确保安全")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    check_env_security()
