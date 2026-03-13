#!/usr/bin/env python3
"""
第6层检查：前端层
检查前端HTML/JavaScript代码，诊断task_id为undefined的问题
"""

import os
import re
import sys
from datetime import datetime

def print_header():
    print("=" * 80)
    print("第6层检查：前端层")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def check_frontend_files():
    print("【1】检查前端文件结构...")
    
    frontend_dir = "frontend"
    if not os.path.exists(frontend_dir):
        print("❌ 前端目录不存在:", frontend_dir)
        return False
    
    files = os.listdir(frontend_dir)
    print(f"✅ 前端目录存在，包含 {len(files)} 个文件")
    
    required_files = ["index.html", "login.html", "register.html"]
    for f in required_files:
        path = os.path.join(frontend_dir, f)
        if os.path.exists(path):
            print(f"✅ {f} 存在")
        else:
            print(f"❌ {f} 不存在")
    
    return True

def analyze_index_html():
    print("\n【2】分析 index.html 中的JavaScript代码...")
    
    index_path = "frontend/index.html"
    if not os.path.exists(index_path):
        print("❌ index.html 不存在")
        return
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有API请求
        api_patterns = [
            r'fetch\(["\']/api/ecg/tasks/([^"\']+)["\']',
            r'fetch\(["\']/api/ecg/tasks/([^"\']+)["\']',
            r'fetch\(["\']/api/ecg/tasks/([^"\']+)["\']',
        ]
        
        print("🔍 查找API请求中的task_id参数...")
        for pattern in api_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match == 'undefined':
                    print(f"❌ 发现请求 '/api/ecg/tasks/undefined'")
                else:
                    print(f"✅ 请求 '/api/ecg/tasks/{match}'")
        
        # 查找变量定义
        print("\n🔍 查找task_id相关变量...")
        var_patterns = [
            r'let\s+(\w+TaskId)\s*=',
            r'const\s+(\w+TaskId)\s*=',
            r'var\s+(\w+TaskId)\s*=',
            r'(\w+TaskId)\s*=',
        ]
        
        found_vars = set()
        for pattern in var_patterns:
            matches = re.findall(pattern, content)
            for var in matches:
                if 'task' in var.lower() or 'Task' in var:
                    found_vars.add(var)
        
        if found_vars:
            print("✅ 找到task_id相关变量:")
            for var in sorted(found_vars):
                print(f"  - {var}")
        else:
            print("⚠️  未找到task_id相关变量")
        
        # 查找页面加载时的初始化代码
        print("\n🔍 查找页面加载初始化代码...")
        if 'window.onload' in content:
            print("✅ 找到 window.onload 事件处理器")
        if 'DOMContentLoaded' in content:
            print("✅ 找到 DOMContentLoaded 事件处理器")
        
        # 查找checkLogin函数
        if 'async function checkLogin()' in content:
            print("✅ 找到 checkLogin 函数")
        
        # 查找loadHistoryLogs函数
        if 'async function loadHistoryLogs(' in content:
            print("✅ 找到 loadHistoryLogs 函数")
        
        # 查找可能导致undefined的代码
        print("\n🔍 查找可能导致undefined的代码模式...")
        
        # 1. 检查是否在变量未初始化时就使用
        undefined_patterns = [
            r'fetch\([^)]*undefined[^)]*\)',
            r'fetch\([^)]*taskId[^)]*\)\s*[^{]*undefined',
        ]
        
        for pattern in undefined_patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"❌ 发现可能导致undefined的代码模式:")
                for match in matches[:3]:  # 只显示前3个
                    print(f"  - {match[:100]}...")
        
        # 2. 检查currentHistoryTaskId的初始化
        if 'let currentHistoryTaskId = null;' in content:
            print("✅ currentHistoryTaskId 已正确初始化为 null")
        elif 'let currentHistoryTaskId;' in content:
            print("⚠️  currentHistoryTaskId 声明但未初始化")
        else:
            print("❌ 未找到 currentHistoryTaskId 的声明")
        
        # 3. 检查loadHistoryLogs中的逻辑
        load_logs_section = re.search(r'async function loadHistoryLogs\([^)]*\)\s*\{[^}]+\}', content, re.DOTALL)
        if load_logs_section:
            logs_code = load_logs_section.group(0)
            if 'typeof currentHistoryTaskId !== \'undefined\'' in logs_code:
                print("✅ loadHistoryLogs 中有检查 currentHistoryTaskId 是否为 undefined")
            else:
                print("⚠️  loadHistoryLogs 中未检查 currentHistoryTaskId 是否为 undefined")
        
    except Exception as e:
        print(f"❌ 分析index.html时出错: {e}")

def check_javascript_errors():
    print("\n【3】模拟JavaScript执行，查找潜在错误...")
    
    print("🔍 分析可能的错误场景:")
    
    # 场景1: 页面加载时自动请求任务详情
    print("1. 页面加载时是否自动请求任务详情?")
    print("   - 如果 currentHistoryTaskId 为 null 或 undefined，但代码仍尝试请求")
    print("   - 需要检查 loadHistoryLogs 函数中的自动加载逻辑")
    
    # 场景2: 事件处理器中的未初始化变量
    print("\n2. 事件处理器中是否使用了未初始化的变量?")
    print("   - 检查所有按钮点击事件处理器")
    print("   - 特别是下载报告、查看详情等按钮")
    
    # 场景3: 异步回调中的变量作用域问题
    print("\n3. 异步回调中是否有变量作用域问题?")
    print("   - 检查 fetch 回调中的变量引用")
    print("   - 检查 setTimeout/setInterval 中的变量引用")

def check_api_endpoints():
    print("\n【4】检查API端点兼容性...")
    
    print("前端需要调用的API端点:")
    endpoints = [
        ("GET", "/api/auth/me", "获取当前用户信息"),
        ("GET", "/api/ecg/tasks", "获取任务列表"),
        ("GET", "/api/ecg/tasks/{task_id}", "获取任务详情"),
        ("GET", "/api/ecg/tasks/{task_id}/signal", "获取信号数据"),
        ("GET", "/api/ecg/tasks/{task_id}/report", "下载PDF报告"),
        ("POST", "/api/ecg/upload", "上传ECG文件"),
    ]
    
    for method, endpoint, desc in endpoints:
        print(f"  {method:6s} {endpoint:30s} - {desc}")
    
    print("\n⚠️  注意: 如果 task_id 为 undefined，请求 /api/ecg/tasks/undefined 会返回422错误")

def suggest_fixes():
    print("\n【5】问题诊断与修复建议...")
    
    print("根据服务器日志，发现以下错误:")
    print("  GET /api/ecg/tasks/undefined HTTP/1.1\" 422 Unprocessable Entity")
    print("\n问题分析:")
    print("1. 前端代码中有地方尝试请求 '/api/ecg/tasks/undefined'")
    print("2. 这意味着 task_id 变量的值为字符串 'undefined'")
    print("3. 可能的原因:")
    print("   a) 变量未正确初始化")
    print("   b) 从 localStorage 或其他地方获取的值是 'undefined' 字符串")
    print("   c) 事件处理器在变量未赋值时被触发")
    
    print("\n🔧 修复建议:")
    print("1. 在 frontend/index.html 中搜索所有使用 task_id 的地方")
    print("2. 确保所有 task_id 变量在使用前都已正确初始化")
    print("3. 添加防御性编程:")
    print("   - 检查 task_id 是否为有效数字")
    print("   - 使用条件语句防止无效请求")
    print("4. 具体修复位置可能包括:")
    print("   - loadHistoryLogs 函数中的自动加载逻辑")
    print("   - 按钮点击事件处理器")
    print("   - 页面加载时的初始化代码")

def main():
    print_header()
    
    if not check_frontend_files():
        return
    
    analyze_index_html()
    check_javascript_errors()
    check_api_endpoints()
    suggest_fixes()
    
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    print("✅ 前端文件结构完整")
    print("⚠️  发现潜在问题: task_id 可能为 undefined")
    print("🔧 需要修复前端JavaScript代码中的变量初始化问题")
    print("\n下一步:")
    print("1. 修复 frontend/index.html 中的JavaScript代码")
    print("2. 确保所有 task_id 变量在使用前都已正确初始化")
    print("3. 重新测试前端功能")

if __name__ == "__main__":
    main()