"""
生成消融实验报告的所有图表
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体和样式
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 创建输出目录
os.makedirs('experiments/results/ablation_figures', exist_ok=True)

# 加载实验结果
with open('experiments/results/dl_results_patient_wise.json', 'r') as f:
    results = json.load(f)

# ============================================================
# 图1：验证集 vs 测试集 Macro-F1 对比
# ============================================================
def plot_val_vs_test_f1():
    models = ['ResNet1D', 'SEResNet1D', 'Transformer', 'BiLSTM', 'TCN', 'Inception']
    
    # 验证集 F1（从训练日志手动提取）
    val_f1 = {
        'ResNet1D': 0.5704,
        'SEResNet1D': 0.7412,
        'Transformer': 0.6640,
        'BiLSTM': 0.4667,
        'TCN': 0.5129,
        'Inception': 0.4942
    }
    
    # 测试集 F1
    test_f1 = {name: results[name]['macro_f1'] for name in models}
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, [val_f1[m] for m in models], width, 
                   label='Validation Set', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, [test_f1[m] for m in models], width,
                   label='Test Set', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro-F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Validation vs Test Performance: Distribution Shift Effect', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('experiments/results/ablation_figures/1_val_vs_test_f1.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图1: 验证集 vs 测试集 F1 对比")


# ============================================================
# 图2：各模型在三个类别上的 F1 对比（测试集）
# ============================================================
def plot_per_class_f1():
    models = ['ResNet1D', 'SEResNet1D', 'Transformer', 'BiLSTM', 'TCN', 'Inception']
    classes = ['Normal', 'PVC', 'Other']
    
    data = {
        'Normal': [results[m]['f1_per_class'][0] for m in models],
        'PVC': [results[m]['f1_per_class'][1] for m in models],
        'Other': [results[m]['f1_per_class'][2] for m in models]
    }
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, data['Normal'], width, label='Normal', 
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, data['PVC'], width, label='PVC', 
                   color='#f39c12', alpha=0.8)
    bars3 = ax.bar(x + width, data['Other'], width, label='Other', 
                   color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Comparison on Test Set (Extreme Imbalance: 98.7% Normal)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('experiments/results/ablation_figures/2_per_class_f1.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图2: 各类别 F1 对比")


# ============================================================
# 图3：SEResNet1D 混淆矩阵热力图
# ============================================================
def plot_confusion_matrix():
    cm = np.array(results['SEResNet1D']['confusion_matrix'])
    classes = ['Normal\n(12332)', 'PVC\n(102)', 'Other\n(60)']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 归一化到 [0, 1] 用于颜色映射
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 设置刻度
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Predicted Label')
    
    ax.set_title('SEResNet1D Confusion Matrix (Test Set)\nBest Model: Macro-F1=0.3277', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 在每个格子中显示数值
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})',
                   ha="center", va="center", fontsize=11,
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiments/results/ablation_figures/3_seresnet_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图3: SEResNet1D 混淆矩阵")


# ============================================================
# 图4：模型参数量 vs 性能散点图
# ============================================================
def plot_params_vs_performance():
    models = ['ResNet1D', 'SEResNet1D', 'Transformer', 'BiLSTM', 'TCN', 'Inception']
    
    # 参数量（百万）
    params = {
        'ResNet1D': 2.1,
        'SEResNet1D': 2.2,
        'Transformer': 1.8,
        'BiLSTM': 1.5,
        'TCN': 1.2,
        'Inception': 2.5
    }
    
    # 验证集 F1
    val_f1 = {
        'ResNet1D': 0.5704,
        'SEResNet1D': 0.7412,
        'Transformer': 0.6640,
        'BiLSTM': 0.4667,
        'TCN': 0.5129,
        'Inception': 0.4942
    }
    
    # 训练时间（秒）
    train_time = {
        'ResNet1D': 286,
        'SEResNet1D': 288,
        'Transformer': 345,
        'BiLSTM': 1940,
        'TCN': 1308,
        'Inception': 1775
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：参数量 vs F1
    x1 = [params[m] for m in models]
    y1 = [val_f1[m] for m in models]
    colors = ['#e74c3c' if m == 'SEResNet1D' else '#3498db' for m in models]
    sizes = [200 if m == 'SEResNet1D' else 100 for m in models]
    
    ax1.scatter(x1, y1, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    for i, m in enumerate(models):
        ax1.annotate(m, (x1[i], y1[i]), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold' if m == 'SEResNet1D' else 'normal')
    
    ax1.set_xlabel('Parameters (Million)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Macro-F1', fontsize=12, fontweight='bold')
    ax1.set_title('Model Efficiency: Parameters vs Performance', 
                  fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 子图2：训练时间 vs F1
    x2 = [train_time[m] for m in models]
    y2 = [val_f1[m] for m in models]
    
    ax2.scatter(x2, y2, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    for i, m in enumerate(models):
        ax2.annotate(m, (x2[i], y2[i]), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold' if m == 'SEResNet1D' else 'normal')
    
    ax2.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Macro-F1', fontsize=12, fontweight='bold')
    ax2.set_title('Model Efficiency: Training Time vs Performance', 
                  fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('experiments/results/ablation_figures/4_efficiency_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图4: 模型效率分析")


# ============================================================
# 图5：数据分布对比（训练/验证/测试）
# ============================================================
def plot_data_distribution():
    datasets = ['Training', 'Validation', 'Test']
    normal = [24780, 10757, 12332]
    pvc = [6263, 1506, 102]
    other = [727, 782, 60]
    
    # 计算百分比
    totals = [sum(x) for x in zip(normal, pvc, other)]
    normal_pct = [n/t*100 for n, t in zip(normal, totals)]
    pvc_pct = [p/t*100 for p, t in zip(pvc, totals)]
    other_pct = [o/t*100 for o, t in zip(other, totals)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：绝对数量堆叠柱状图
    x = np.arange(len(datasets))
    width = 0.5
    
    p1 = ax1.bar(x, normal, width, label='Normal', color='#2ecc71', alpha=0.8)
    p2 = ax1.bar(x, pvc, width, bottom=normal, label='PVC', color='#f39c12', alpha=0.8)
    p3 = ax1.bar(x, other, width, bottom=[n+p for n, p in zip(normal, pvc)], 
                label='Other', color='#9b59b6', alpha=0.8)
    
    ax1.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Sample Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加总数标签
    for i, (n, p, o) in enumerate(zip(normal, pvc, other)):
        total = n + p + o
        ax1.text(i, total + 500, f'Total: {total:,}', 
                ha='center', fontsize=10, fontweight='bold')
    
    # 子图2：百分比堆叠柱状图
    p1 = ax2.bar(x, normal_pct, width, label='Normal', color='#2ecc71', alpha=0.8)
    p2 = ax2.bar(x, pvc_pct, width, bottom=normal_pct, label='PVC', color='#f39c12', alpha=0.8)
    p3 = ax2.bar(x, other_pct, width, bottom=[n+p for n, p in zip(normal_pct, pvc_pct)], 
                label='Other', color='#9b59b6', alpha=0.8)
    
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Class Distribution (Distribution Shift!)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # 添加百分比标签
    for i in range(len(datasets)):
        ax2.text(i, normal_pct[i]/2, f'{normal_pct[i]:.1f}%', 
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax2.text(i, normal_pct[i] + pvc_pct[i]/2, f'{pvc_pct[i]:.1f}%', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax2.text(i, normal_pct[i] + pvc_pct[i] + other_pct[i]/2, f'{other_pct[i]:.1f}%', 
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiments/results/ablation_figures/5_data_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图5: 数据分布对比")


# ============================================================
# 图6：Precision vs Recall 雷达图（测试集）
# ============================================================
def plot_precision_recall_radar():
    from math import pi
    
    models = ['ResNet1D', 'SEResNet1D', 'Transformer', 'BiLSTM', 'TCN', 'Inception']
    
    # 从混淆矩阵计算 Precision 和 Recall（Normal 类）
    metrics = {}
    for m in models:
        cm = np.array(results[m]['confusion_matrix'])
        # Normal 类的 Precision 和 Recall
        precision = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
        recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
        f1 = results[m]['f1_per_class'][0]
        macro_f1 = results[m]['macro_f1']
        metrics[m] = [precision, recall, f1, macro_f1]
    
    categories = ['Precision\n(Normal)', 'Recall\n(Normal)', 'F1\n(Normal)', 'Macro-F1']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, (m, color) in enumerate(zip(models, colors)):
        values = metrics[m]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=m, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Model Performance Radar Chart (Test Set)', 
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('experiments/results/ablation_figures/6_performance_radar.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图6: 性能雷达图")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("生成消融实验图表")
    print("=" * 60)
    
    plot_val_vs_test_f1()
    plot_per_class_f1()
    plot_confusion_matrix()
    plot_params_vs_performance()
    plot_data_distribution()
    plot_precision_recall_radar()
    
    print("\n" + "=" * 60)
    print("✅ 所有图表已生成！")
    print("保存位置: experiments/results/ablation_figures/")
    print("=" * 60)
    print("\n生成的图表：")
    print("  1. 1_val_vs_test_f1.png - 验证集 vs 测试集 F1 对比")
    print("  2. 2_per_class_f1.png - 各类别 F1 对比")
    print("  3. 3_seresnet_confusion_matrix.png - SEResNet1D 混淆矩阵")
    print("  4. 4_efficiency_analysis.png - 模型效率分析")
    print("  5. 5_data_distribution.png - 数据分布对比")
    print("  6. 6_performance_radar.png - 性能雷达图")


if __name__ == '__main__':
    main()
