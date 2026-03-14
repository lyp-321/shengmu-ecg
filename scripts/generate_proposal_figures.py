"""
生成大创申报书所需的图片
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 创建输出目录
import os
os.makedirs('docs/proposal_figures', exist_ok=True)

def plot_confusion_matrix_diagram():
    """图10：混淆矩阵示意图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, '混淆矩阵示意图', ha='center', va='top', 
            fontsize=16, fontweight='bold')
    
    # 绘制2x2矩阵
    colors = ['#d4edda', '#f8d7da', '#f8d7da', '#d4edda']
    labels = ['TP\n(真阳性)', 'FP\n(假阳性)', 'FN\n(假阴性)', 'TN\n(真阴性)']
    positions = [(2, 6), (5.5, 6), (2, 3), (5.5, 3)]
    
    for i, (x, y) in enumerate(positions):
        rect = FancyBboxPatch((x, y), 2.5, 2, boxstyle="round,pad=0.1",
                              facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+1.25, y+1, labels[i], ha='center', va='center',
                fontsize=14, fontweight='bold')
    
    # 添加标签
    ax.text(1, 7, '实际\n标签', ha='center', va='center', fontsize=12, rotation=90)
    ax.text(5, 8.5, '预测标签', ha='center', va='center', fontsize=12)
    
    ax.text(3.25, 8.2, '阳性', ha='center', va='center', fontsize=11)
    ax.text(6.75, 8.2, '阴性', ha='center', va='center', fontsize=11)
    ax.text(1.5, 7, '阳性', ha='center', va='center', fontsize=11)
    ax.text(1.5, 4, '阴性', ha='center', va='center', fontsize=11)
    
    # 添加公式
    formulas = [
        r'准确率 = $\frac{TP + TN}{TP + TN + FP + FN}$',
        r'精确率 = $\frac{TP}{TP + FP}$',
        r'召回率 = $\frac{TP}{TP + FN}$',
        r'F1分数 = $\frac{2 \times 精确率 \times 召回率}{精确率 + 召回率}$'
    ]
    
    for i, formula in enumerate(formulas):
        ax.text(5, 1.8 - i*0.4, formula, ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图10_混淆矩阵示意图.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图10：混淆矩阵示意图")


def plot_se_module():
    """图4：SE模块原理图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 标题
    ax.text(6, 5.7, 'SE模块（Squeeze-and-Excitation）原理图', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # 输入特征图
    rect1 = FancyBboxPatch((0.5, 2), 1.5, 2.5, boxstyle="round,pad=0.05",
                           facecolor='#e3f2fd', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.25, 3.25, '输入\n特征图\n(B,C,L)', ha='center', va='center', fontsize=10)
    
    # Squeeze: Global Average Pooling
    arrow1 = FancyArrowPatch((2.2, 3.25), (3.3, 3.25),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    rect2 = FancyBboxPatch((3.5, 2.5), 1.5, 1.5, boxstyle="round,pad=0.05",
                           facecolor='#fff3e0', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(4.25, 3.25, 'Global\nAvgPool\n(B,C,1)', ha='center', va='center', fontsize=9)
    
    # Excitation: FC layers
    arrow2 = FancyArrowPatch((5.2, 3.25), (6.3, 3.25),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    rect3 = FancyBboxPatch((6.5, 2.5), 1.2, 1.5, boxstyle="round,pad=0.05",
                           facecolor='#f3e5f5', edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(7.1, 3.25, 'FC\n(C→C/16)\nReLU', ha='center', va='center', fontsize=9)
    
    arrow3 = FancyArrowPatch((7.9, 3.25), (8.8, 3.25),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    rect4 = FancyBboxPatch((9, 2.5), 1.2, 1.5, boxstyle="round,pad=0.05",
                           facecolor='#e8f5e9', edgecolor='black', linewidth=2)
    ax.add_patch(rect4)
    ax.text(9.6, 3.25, 'FC\n(C/16→C)\nSigmoid', ha='center', va='center', fontsize=9)
    
    # Scale (通道加权)
    arrow4 = FancyArrowPatch((9.6, 2.3), (1.25, 1.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, 
                            color='red', linestyle='dashed')
    ax.add_patch(arrow4)
    ax.text(5.5, 1.8, '通道加权', ha='center', va='center', fontsize=10, color='red')
    
    # 输出特征图
    rect5 = FancyBboxPatch((0.5, 0.2), 1.5, 1, boxstyle="round,pad=0.05",
                           facecolor='#c8e6c9', edgecolor='black', linewidth=2)
    ax.add_patch(rect5)
    ax.text(1.25, 0.7, '输出特征图\n(B,C,L)', ha='center', va='center', fontsize=10)
    
    # 添加说明
    ax.text(6, 0.3, 'Squeeze: 压缩空间维度，提取全局信息', 
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.text(6, 5.2, 'Excitation: 学习通道间依赖关系，生成通道权重', 
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图4_SE模块原理图.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图4：SE模块原理图")


def plot_stacking_flowchart():
    """图6：Stacking流程图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, 'ML Stacking集成策略流程图', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # 输入
    rect1 = FancyBboxPatch((3.5, 8), 3, 0.8, boxstyle="round,pad=0.05",
                           facecolor='#e3f2fd', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 8.4, '输入信号 → 41维特征 → 标准化', ha='center', va='center', fontsize=10)
    
    # XGBoost粗筛
    arrow1 = FancyArrowPatch((5, 7.8), (5, 7.2),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    rect2 = FancyBboxPatch((3, 6.2), 4, 0.8, boxstyle="round,pad=0.05",
                           facecolor='#fff3e0', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, 6.6, 'XGBoost粗筛（高召回率）', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # 决策分支
    arrow2_left = FancyArrowPatch((5, 6), (2.5, 5.2),
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow2_left)
    ax.text(3.5, 5.7, '预测=正常', ha='center', va='center', fontsize=9, color='green')
    
    arrow2_right = FancyArrowPatch((5, 6), (7.5, 5.2),
                                  arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
    ax.add_patch(arrow2_right)
    ax.text(6.5, 5.7, '预测=异常', ha='center', va='center', fontsize=9, color='red')
    
    # 左分支：直接采信
    rect3 = FancyBboxPatch((1, 4.2), 3, 0.8, boxstyle="round,pad=0.05",
                           facecolor='#c8e6c9', edgecolor='green', linewidth=2)
    ax.add_patch(rect3)
    ax.text(2.5, 4.6, '直接采信\nconf ≈ 0.998', ha='center', va='center', fontsize=10)
    
    # 右分支：CatBoost复核
    rect4 = FancyBboxPatch((6, 4.2), 3, 0.8, boxstyle="round,pad=0.05",
                           facecolor='#ffccbc', edgecolor='red', linewidth=2)
    ax.add_patch(rect4)
    ax.text(7.5, 4.6, 'CatBoost复核\n（高精确率）', ha='center', va='center', fontsize=10)
    
    # 汇总
    arrow3_left = FancyArrowPatch((2.5, 4), (4.5, 3.2),
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3_left)
    
    arrow3_right = FancyArrowPatch((7.5, 4), (5.5, 3.2),
                                  arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3_right)
    
    # 最终输出
    rect5 = FancyBboxPatch((3.5, 2.2), 3, 0.8, boxstyle="round,pad=0.05",
                           facecolor='#b39ddb', edgecolor='black', linewidth=2)
    ax.add_patch(rect5)
    ax.text(5, 2.6, '最终诊断结果', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # 添加说明框
    explanation = [
        '策略优势：',
        '1. XGBoost高召回：不漏掉疑似异常',
        '2. CatBoost高精确：过滤假阳性',
        '3. 两阶段Stacking：平衡召回与精确'
    ]
    y_pos = 1.5
    for text in explanation:
        ax.text(5, y_pos, text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        y_pos -= 0.3
    
    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图6_Stacking流程图.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图6：Stacking流程图")


if __name__ == '__main__':
    print("开始生成申报书图片...")
    plot_confusion_matrix_diagram()
    plot_se_module()
    plot_stacking_flowchart()
    print("\n所有图片生成完成！保存在 docs/proposal_figures/ 目录")
