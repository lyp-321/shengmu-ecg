#!/bin/bash

echo "=========================================="
echo "ECG多模态模型训练脚本"
echo "=========================================="

# 激活conda环境
echo "激活conda环境: ai"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai

# 检查Python版本
echo ""
echo "Python版本:"
python --version

# 创建必要的目录
echo ""
echo "创建目录..."
mkdir -p app/algorithms/models
mkdir -p experiments/results
mkdir -p logs

# 训练传统机器学习模型
echo ""
echo "=========================================="
echo "步骤 1: 训练传统机器学习模型"
echo "=========================================="
python scripts/train_traditional_ml.py 2>&1 | tee logs/train_ml.log

# 训练深度学习模型
echo ""
echo "=========================================="
echo "步骤 2: 训练深度学习模型"
echo "=========================================="
python scripts/train_multimodal_models.py 2>&1 | tee logs/train_dl.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: app/algorithms/models/"
echo "结果保存在: experiments/results/"
echo "日志保存在: logs/"
