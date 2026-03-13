#!/bin/bash
# ECG系统完整检查脚本

echo "=================================="
echo "ECG系统完整检查"
echo "=================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查计数
PASSED=0
FAILED=0

# 运行单个检查
run_check() {
    local layer=$1
    local script=$2
    local name=$3
    
    echo ""
    echo "=========================================="
    echo "第${layer}层：${name}"
    echo "=========================================="
    
    if python $script; then
        echo -e "${GREEN}✅ 第${layer}层检查通过${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ 第${layer}层检查失败${NC}"
        ((FAILED++))
        return 1
    fi
}

# 第1-5层：后端检查（不需要服务器）
echo "开始后端检查（第1-5层）..."
echo ""

run_check 1 "tests/check_layer1_database.py" "数据库层"
run_check 2 "tests/check_layer2_models.py" "数据模型层"
run_check 3 "tests/check_layer3_algorithms.py" "算法层"
run_check 4 "tests/check_layer4_services.py" "服务层"
run_check 5 "tests/check_layer5_api.py" "API层"

echo ""
echo "=========================================="
echo "后端检查完成"
echo "=========================================="
echo -e "通过: ${GREEN}${PASSED}${NC} | 失败: ${RED}${FAILED}${NC}"
echo ""

# 如果后端检查失败，停止
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}后端检查发现问题，请先修复后再继续${NC}"
    exit 1
fi

# 检查服务器是否运行
echo "检查服务器状态..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 服务器正在运行${NC}"
    echo ""
    
    # 第6-7层：前端和集成测试（需要服务器）
    echo "开始前端和集成测试（第6-7层）..."
    echo ""
    
    run_check 6 "tests/check_layer6_frontend.py" "前端层"
    run_check 7 "tests/check_layer7_integration.py" "集成测试"
    
else
    echo -e "${YELLOW}⚠️  服务器未运行${NC}"
    echo ""
    echo "请先启动服务器："
    echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "然后运行前端和集成测试："
    echo "  python tests/check_layer6_frontend.py"
    echo "  python tests/check_layer7_integration.py"
    echo ""
fi

# 最终总结
echo ""
echo "=========================================="
echo "检查总结"
echo "=========================================="
echo -e "通过: ${GREEN}${PASSED}${NC} | 失败: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 所有检查通过！系统运行正常！${NC}"
    exit 0
else
    echo -e "${RED}发现问题，请查看上面的错误信息${NC}"
    exit 1
fi
