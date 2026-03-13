#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化训练结果
生成混淆矩阵、ROC曲线、PR曲线等图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 类别名称
CLASS_NAMES = ['正常', '室性早搏', '其他异常']

def plot_confusion_matrices(results_file='experiments/results/ml_results_patient_wise.json',
                            output_dir='experiments/results'):
    """绘制所有模型的混淆矩阵"""
    # 加载结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 为每个模型绘制混淆矩阵
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        cm = np.array(model_results['confusion_matrix'])
        
        # 归一化混淆矩阵（按行归一化，显示召回率）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热力图
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   ax=axes[idx], cbar_kws={'label': '比例'})
        
        axes[idx].set_title(f'{model_name}\n准确率: {model_results["accuracy"]:.2%}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('真实类别', fontsize=10)
        axes[idx].set_xlabel('预测类别', fontsize=10)
    
    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {output_path}")
    plt.close()


def plot_model_comparison(results_file='experiments/results/ml_results_patient_wise.json',
                         output_dir='experiments/results'):
    """绘制模型性能对比图"""
    # 加载结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 提取数据
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    precision = [results[m]['precision'] for m in models]
    recall = [results[m]['recall'] for m in models]
    f1_score = [results[m]['f1_score'] for m in models]
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracy, width, label='准确率', color='#3498db')
    ax.bar(x - 0.5*width, precision, width, label='精确率', color='#2ecc71')
    ax.bar(x + 0.5*width, recall, width, label='召回率', color='#f39c12')
    ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('传统机器学习模型性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0.94, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i, m in enumerate(models):
        ax.text(i, accuracy[i] + 0.002, f'{accuracy[i]:.3f}', 
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 模型对比图已保存: {output_path}")
    plt.close()


def plot_f1_per_class(results_file='experiments/results/ml_results_patient_wise.json',
                     output_dir='experiments/results'):
    """绘制每个类别的F1-Score对比"""
    # 加载结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 提取数据
    models = list(results.keys())
    f1_class0 = [results[m]['f1_per_class'][0] for m in models]
    f1_class1 = [results[m]['f1_per_class'][1] for m in models]
    f1_class2 = [results[m]['f1_per_class'][2] for m in models]
    
    # 绘制分组柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, f1_class0, width, label=CLASS_NAMES[0], color='#3498db')
    ax.bar(x, f1_class1, width, label=CLASS_NAMES[1], color='#e74c3c')
    ax.bar(x + width, f1_class2, width, label=CLASS_NAMES[2], color='#f39c12')
    
    ax.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('各类别F1-Score对比（按患者划分）', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for i in range(len(models)):
        if f1_class1[i] > 0.01:
            ax.text(i, f1_class1[i] + 0.02, f'{f1_class1[i]:.3f}', 
                   ha='center', va='bottom', fontsize=8, color='red')
        if f1_class2[i] > 0.01:
            ax.text(i + width, f1_class2[i] + 0.02, f'{f1_class2[i]:.3f}', 
                   ha='center', va='bottom', fontsize=8, color='orange')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'f1_per_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 类别F1-Score对比图已保存: {output_path}")
    plt.close()


def plot_training_time(results_file='experiments/results/ml_results_patient_wise.json',
                      output_dir='experiments/results'):
    """绘制训练时间对比"""
    # 加载结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 提取数据
    models = list(results.keys())
    train_times = [results[m]['train_time'] for m in models]
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    bars = ax.bar(models, train_times, color=colors, alpha=0.8)
    
    ax.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax.set_ylabel('训练时间 (秒)', fontsize=12, fontweight='bold')
    ax.set_title('模型训练时间对比', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, time in zip(bars, train_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练时间对比图已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("="*60)
    print("生成可视化图表")
    print("="*60)
    
    results_file = 'experiments/results/ml_results_patient_wise.json'
    output_dir = 'experiments/results'
    
    # 检查结果文件是否存在
    if not Path(results_file).exists():
        print(f"❌ 结果文件不存在: {results_file}")
        print("请先运行训练脚本: python scripts/train_traditional_ml.py")
        return
    
    print(f"\n读取结果文件: {results_file}")
    
    # 生成各种图表
    print("\n1. 生成混淆矩阵...")
    plot_confusion_matrices(results_file, output_dir)
    
    print("\n2. 生成模型性能对比图...")
    plot_model_comparison(results_file, output_dir)
    
    print("\n3. 生成类别F1-Score对比图...")
    plot_f1_per_class(results_file, output_dir)
    
    print("\n4. 生成训练时间对比图...")
    plot_training_time(results_file, output_dir)
    
    print("\n" + "="*60)
    print("✅ 所有图表生成完成！")
    print("="*60)
    print(f"\n图表保存位置: {output_dir}/")
    print("  - confusion_matrices.png    (混淆矩阵)")
    print("  - model_comparison.png      (模型性能对比)")
    print("  - f1_per_class.png          (类别F1-Score)")
    print("  - training_time.png         (训练时间)")


if __name__ == '__main__':
    main()
