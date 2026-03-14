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


def plot_ecg_comparison():
    """图1：正常心电图与异常心电图对比"""
    np.random.seed(42)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('心电图波形对比：正常心律 / 室性早搏 / 其他异常', fontsize=14, fontweight='bold')

    t = np.linspace(0, 1, 360)

    def make_normal_beat(t):
        """模拟正常PQRST波形"""
        sig = np.zeros_like(t)
        # P波
        sig += 0.15 * np.exp(-((t - 0.15) ** 2) / (2 * 0.012 ** 2))
        # Q波
        sig -= 0.05 * np.exp(-((t - 0.28) ** 2) / (2 * 0.008 ** 2))
        # R波
        sig += 1.0 * np.exp(-((t - 0.32) ** 2) / (2 * 0.010 ** 2))
        # S波
        sig -= 0.15 * np.exp(-((t - 0.36) ** 2) / (2 * 0.008 ** 2))
        # T波
        sig += 0.25 * np.exp(-((t - 0.55) ** 2) / (2 * 0.030 ** 2))
        return sig + np.random.normal(0, 0.015, len(t))

    def make_pvc_beat(t):
        """模拟室性早搏：宽大畸形QRS，无P波"""
        sig = np.zeros_like(t)
        # 宽大R波（无P波）
        sig += 1.4 * np.exp(-((t - 0.30) ** 2) / (2 * 0.025 ** 2))
        # 深S波
        sig -= 0.6 * np.exp(-((t - 0.42) ** 2) / (2 * 0.020 ** 2))
        # 倒置T波
        sig -= 0.3 * np.exp(-((t - 0.65) ** 2) / (2 * 0.040 ** 2))
        return sig + np.random.normal(0, 0.020, len(t))

    def make_other_beat(t):
        """模拟其他异常：P波异常，RR不规则"""
        sig = np.zeros_like(t)
        # 异常P波（双峰）
        sig += 0.12 * np.exp(-((t - 0.10) ** 2) / (2 * 0.008 ** 2))
        sig += 0.10 * np.exp(-((t - 0.18) ** 2) / (2 * 0.008 ** 2))
        # 正常QRS
        sig += 0.85 * np.exp(-((t - 0.35) ** 2) / (2 * 0.010 ** 2))
        sig -= 0.10 * np.exp(-((t - 0.39) ** 2) / (2 * 0.008 ** 2))
        # 低平T波
        sig += 0.10 * np.exp(-((t - 0.58) ** 2) / (2 * 0.035 ** 2))
        return sig + np.random.normal(0, 0.018, len(t))

    beats = [make_normal_beat(t), make_pvc_beat(t), make_other_beat(t)]
    titles = ['正常心律（Normal）', '室性早搏（PVC）', '其他异常（Other）']
    colors = ['#2196F3', '#F44336', '#FF9800']
    annotations = [
        [('P', 0.15, 0.22), ('QRS', 0.32, 1.12), ('T', 0.55, 0.35)],
        [('宽大QRS', 0.30, 1.52), ('深S波', 0.42, -0.72), ('倒置T', 0.65, -0.42)],
        [('双峰P', 0.14, 0.22), ('QRS', 0.35, 0.97), ('低平T', 0.58, 0.18)],
    ]

    for i, (ax, beat, title, color, annots) in enumerate(zip(axes, beats, titles, colors, annotations)):
        ax.plot(t, beat, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('时间 (s)', fontsize=9)
        ax.set_ylabel('幅度 (mV)', fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        for label, tx, ty in annots:
            ax.annotate(label, xy=(tx, ty * 0.85), xytext=(tx + 0.05, ty * 0.85 + 0.15),
                        fontsize=8, color='black',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图1_心电图波形对比.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图1：心电图波形对比")


def plot_seresnet_architecture():
    """图3：SE-ResNet1D网络结构图"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    ax.text(7, 8.7, 'SE-ResNet1D 网络结构图', ha='center', va='top',
            fontsize=15, fontweight='bold')

    # 各层定义：(x中心, y中心, 宽, 高, 颜色, 标签, 子标签)
    layers = [
        (1.0, 4.5, 1.2, 5.0, '#E3F2FD', '输入\n(1×1000)', '1通道\n1000点'),
        (2.8, 4.5, 1.4, 4.0, '#BBDEFB', 'Conv1d\n+BN+ReLU', 'k=15,s=2\n→64ch'),
        (4.2, 4.5, 1.0, 3.5, '#90CAF9', 'MaxPool', 'k=3,s=2'),
        (5.8, 4.5, 1.6, 5.5, '#FFF9C4', 'Layer1\n×2 SE-Block', '64→64ch'),
        (7.6, 4.5, 1.6, 5.5, '#FFE082', 'Layer2\n×2 SE-Block', '64→128ch\ns=2'),
        (9.4, 4.5, 1.6, 5.5, '#FFB74D', 'Layer3\n×2 SE-Block', '128→256ch\ns=2'),
        (11.0, 4.5, 1.2, 3.0, '#C8E6C9', 'AvgPool\n+Dropout', 'p=0.6'),
        (12.6, 4.5, 1.0, 2.5, '#A5D6A7', 'FC\n(256→3)', '输出'),
    ]

    for (cx, cy, w, h, color, label, sublabel) in layers:
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='#555', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(cx, cy + 0.2, label, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(cx, cy - 0.5, sublabel, ha='center', va='center', fontsize=7.5, color='#444')

    # 箭头连接
    xs = [1.6, 3.5, 4.7, 6.6, 8.4, 10.2, 11.6, 12.1]
    xe = [2.1, 3.7, 5.0, 6.8, 8.6, 10.4, 11.8, 12.1]
    for x1, x2 in zip(xs, xe):
        ax.annotate('', xy=(x2, 4.5), xytext=(x1, 4.5),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # SE残差块放大图（右下角）
    bx, by = 5.0, 1.0
    ax.text(bx + 2.5, by + 1.8, 'SE残差块内部结构', ha='center', fontsize=9,
            fontweight='bold', color='#333')
    se_items = [
        (bx + 0.0, by + 1.2, '#E8EAF6', 'Conv1d\n+BN+ReLU'),
        (bx + 1.5, by + 1.2, '#E8EAF6', 'Conv1d\n+BN'),
        (bx + 3.0, by + 1.2, '#FCE4EC', 'SE模块\n(通道注意力)'),
        (bx + 4.5, by + 1.2, '#E8F5E9', '⊕ 残差\n相加+ReLU'),
    ]
    for (sx, sy, sc, sl) in se_items:
        r = FancyBboxPatch((sx - 0.6, sy - 0.5), 1.2, 1.0,
                           boxstyle="round,pad=0.05", facecolor=sc,
                           edgecolor='#888', linewidth=1.2)
        ax.add_patch(r)
        ax.text(sx, sy, sl, ha='center', va='center', fontsize=7.5)

    for i in range(len(se_items) - 1):
        ax.annotate('', xy=(se_items[i+1][0] - 0.6, se_items[i+1][1]),
                    xytext=(se_items[i][0] + 0.6, se_items[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    # shortcut连线
    ax.annotate('', xy=(bx + 4.5 - 0.6, by + 0.5),
                xytext=(bx - 0.6, by + 0.5),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(bx + 2.0, by - 0.1, 'shortcut（跳跃连接）', ha='center', fontsize=8, color='#E53935')

    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图3_SE-ResNet1D网络结构图.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图3：SE-ResNet1D网络结构图")


def plot_multimodal_fusion():
    """图7：多模态融合架构图"""
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.text(6.5, 8.7, '多模态融合与双端互证架构图', ha='center', va='top',
            fontsize=14, fontweight='bold')

    def box(cx, cy, w, h, color, text, fontsize=9):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor='#555', linewidth=1.5)
        ax.add_patch(r)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    def arrow(x1, y1, x2, y2, color='#333', label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.1, my, label, fontsize=7.5, color=color)

    # 输入
    box(6.5, 8.0, 3.0, 0.7, '#E3F2FD', '输入：心电信号（CSV/DAT）', fontsize=10)

    # 两路分支
    arrow(5.0, 7.65, 3.0, 7.0)
    arrow(8.0, 7.65, 10.0, 7.0)

    # ML分支
    box(3.0, 6.6, 2.8, 0.7, '#FFF9C4', '特征工程（41维）')
    arrow(3.0, 6.25, 3.0, 5.6)
    box(3.0, 5.2, 2.8, 0.7, '#FFE082', 'XGBoost 粗筛\n（高召回）', fontsize=8)
    arrow(3.0, 4.85, 3.0, 4.2)
    box(3.0, 3.8, 2.8, 0.7, '#FFB74D', 'CatBoost 复核\n（高精确）', fontsize=8)
    ax.text(3.0, 3.1, 'ML Stacking\n宏F1=0.444', ha='center', fontsize=8,
            color='#E65100',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))

    # DL分支
    box(10.0, 6.6, 2.8, 0.7, '#E8EAF6', '原始信号（1000点）')
    dl_models = [
        (8.8, 5.5, 'SE-ResNet1D\n(核心)'),
        (10.0, 5.5, 'Transformer'),
        (11.2, 5.5, 'ResNet1D'),
        (8.8, 4.3, 'TCN'),
        (10.0, 4.3, 'Inception'),
        (11.2, 4.3, 'BiLSTM'),
    ]
    for (dx, dy, dlabel) in dl_models:
        r = FancyBboxPatch((dx - 0.55, dy - 0.4), 1.1, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor='#C5CAE9', edgecolor='#555', linewidth=1.2)
        ax.add_patch(r)
        ax.text(dx, dy, dlabel, ha='center', va='center', fontsize=7)
        arrow(10.0, 6.25, dx, dy + 0.4)

    box(10.0, 3.4, 2.8, 0.6, '#9FA8DA', '投票集成（6模型）')
    for (dx, dy, _) in dl_models:
        arrow(dx, dy - 0.4, 10.0, 3.7)
    ax.text(10.0, 2.9, 'DL集成\n验证集F1=0.741', ha='center', fontsize=8,
            color='#283593',
            bbox=dict(boxstyle='round', facecolor='#E8EAF6', alpha=0.8))

    # 融合层
    arrow(3.0, 3.45, 5.5, 2.3, '#555')
    arrow(10.0, 2.65, 7.5, 2.3, '#555')
    box(6.5, 2.0, 4.0, 0.7, '#FCE4EC', '多模态融合引擎\n（双端互证）', fontsize=10)

    # 双端互证说明
    ax.text(6.5, 1.35, 'ML=正常 & DL高置信=异常 → 强制预警（降低漏诊）\nML=异常 & DL高置信=正常 → 过滤假阳性',
            ha='center', fontsize=8, color='#880E4F',
            bbox=dict(boxstyle='round', facecolor='#FCE4EC', alpha=0.6))

    # 输出
    arrow(6.5, 1.65, 6.5, 0.85)
    box(6.5, 0.55, 4.0, 0.6, '#C8E6C9', '诊断结果 + 5级风险预警 + Grad-CAM', fontsize=9)

    # 权重标注
    ax.text(4.8, 2.15, 'w=0.45', fontsize=8, color='#555')
    ax.text(8.0, 2.15, 'w=0.45', fontsize=8, color='#555')

    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图7_多模态融合架构图.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图7：多模态融合架构图")


def plot_system_architecture():
    """图9：系统架构图"""
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.text(6.5, 8.7, '系统整体架构图', ha='center', va='top',
            fontsize=14, fontweight='bold')

    layer_defs = [
        # (y_center, height, bg_color, title, items)
        (7.5, 1.4, '#E3F2FD', '前端层（HTML5 + Chart.js）',
         ['实时波形监控', '历史回溯', 'HRV分析', 'PDF下载', '用户管理']),
        (5.5, 1.4, '#E8F5E9', '后端层（FastAPI + Tortoise ORM）',
         ['/api/auth', '/api/ecg/upload', '/api/ecg/tasks/{id}', 'BackgroundTasks', 'JWT认证']),
        (3.5, 1.4, '#FFF9C4', '算法层',
         ['reader/preprocess', 'features(41维)', 'ML Stacking', 'DL集成(×6)', '多模态融合+Grad-CAM']),
        (1.5, 1.0, '#FCE4EC', '数据层（SQLite）',
         ['users表', 'ecg_tasks表', 'JSON结果字段', '历史记录']),
    ]

    for (yc, h, color, title, items) in layer_defs:
        rect = FancyBboxPatch((0.5, yc - h/2), 12.0, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#555', linewidth=2)
        ax.add_patch(rect)
        ax.text(1.2, yc, title, ha='left', va='center', fontsize=10, fontweight='bold')

        n = len(items)
        xs = np.linspace(3.5, 12.0, n)
        for xi, item in zip(xs, items):
            r = FancyBboxPatch((xi - 0.8, yc - 0.35), 1.6, 0.7,
                               boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor='#aaa', linewidth=1)
            ax.add_patch(r)
            ax.text(xi, yc, item, ha='center', va='center', fontsize=7.5)

    # 层间箭头
    for y1, y2 in [(6.8, 6.2), (4.8, 4.2), (2.8, 2.0)]:
        ax.annotate('', xy=(6.5, y2), xytext=(6.5, y1),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=2))

    ax.text(6.8, 6.5, 'HTTP/REST', fontsize=8, color='#555')
    ax.text(6.8, 4.5, '函数调用', fontsize=8, color='#555')
    ax.text(6.8, 2.4, 'ORM读写', fontsize=8, color='#555')

    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图9_系统架构图.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图9：系统架构图")


def plot_technical_roadmap():
    """图12：完整技术路线流程图"""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.text(5, 13.7, '技术路线流程图', ha='center', va='top',
            fontsize=15, fontweight='bold')

    def box(cx, cy, w, h, color, text, fontsize=9):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='#555', linewidth=1.8)
        ax.add_patch(r)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', wrap=True)

    def arr(x1, y1, x2, y2, color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

    # 主流程节点
    nodes = [
        (5.0, 13.0, 5.5, 0.65, '#E3F2FD', 'MIT-BIH 心律失常数据库（27患者）'),
        (5.0, 12.0, 5.5, 0.65, '#BBDEFB', 'Patient-wise Split\n训练15 / 验证6 / 测试6'),
        (5.0, 11.0, 5.5, 0.65, '#90CAF9', '信号预处理\n带通滤波 · R峰检测 · 心拍分割 · 归一化'),
    ]
    for n in nodes:
        box(*n)
    arr(5, 12.67, 5, 12.33)
    arr(5, 11.67, 5, 11.33)

    # 分叉
    arr(5, 10.67, 2.5, 10.1)
    arr(5, 10.67, 7.5, 10.1)

    # 左：ML分支
    ml_nodes = [
        (2.5, 9.7, 3.8, 0.6, '#FFF9C4', '特征工程（41维）\n时域·频域·小波'),
        (2.5, 8.8, 3.8, 0.6, '#FFE082', '代价敏感学习\nSMOTE · class_weight'),
        (2.5, 7.9, 3.8, 0.6, '#FFB74D', 'ML Stacking\nXGBoost粗筛 + CatBoost复核'),
    ]
    for n in ml_nodes:
        box(*n)
    arr(2.5, 9.4, 2.5, 9.1)
    arr(2.5, 8.5, 2.5, 8.2)

    # 右：DL分支
    dl_nodes = [
        (7.5, 9.7, 3.8, 0.6, '#E8EAF6', '原始信号（1000点）\n4种数据增强'),
        (7.5, 8.8, 3.8, 0.6, '#C5CAE9', '代价敏感训练\nFocal Loss · Label Smooth · MixUp'),
        (7.5, 7.9, 3.8, 0.6, '#9FA8DA', 'SE-ResNet1D（核心）\n+ 5个辅助模型投票'),
    ]
    for n in dl_nodes:
        box(*n)
    arr(7.5, 9.4, 7.5, 9.1)
    arr(7.5, 8.5, 7.5, 8.2)

    # 汇合
    arr(2.5, 7.6, 4.2, 7.0)
    arr(7.5, 7.6, 5.8, 7.0)

    # 融合层
    box(5.0, 6.7, 5.5, 0.55, '#FCE4EC', '多模态融合 + 双端互证机制')
    arr(5, 6.42, 5, 5.95)
    box(5.0, 5.7, 5.5, 0.55, '#F8BBD0', 'Grad-CAM 可解释性热力图')
    arr(5, 5.42, 5, 4.95)
    box(5.0, 4.7, 5.5, 0.55, '#EF9A9A', '5级风险预警评估')
    arr(5, 4.42, 5, 3.95)

    # 系统层
    box(5.0, 3.7, 5.5, 0.55, '#C8E6C9', '系统开发与部署（FastAPI + Web前端）')
    arr(5, 3.42, 5, 2.95)
    box(5.0, 2.7, 5.5, 0.55, '#A5D6A7', 'PDF报告生成 + 历史回溯 + HRV分析')
    arr(5, 2.42, 5, 1.95)
    box(5.0, 1.7, 5.5, 0.55, '#81C784', '临床验证与优化')

    # 左侧标注
    for (y, label, color) in [
        (10.5, '数据层', '#1565C0'),
        (9.0, 'ML路线', '#E65100'),
        (7.5, 'DL路线', '#283593'),
        (6.0, '融合层', '#880E4F'),
        (3.5, '系统层', '#1B5E20'),
    ]:
        ax.text(0.3, y, label, ha='center', va='center', fontsize=8,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.savefig('docs/proposal_figures/图12_技术路线流程图.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 图12：技术路线流程图")


if __name__ == '__main__':
    print("开始生成申报书图片...")
    plot_confusion_matrix_diagram()
    plot_se_module()
    plot_stacking_flowchart()
    plot_ecg_comparison()
    plot_seresnet_architecture()
    plot_multimodal_fusion()
    plot_system_architecture()
    plot_technical_roadmap()
    print("\n所有图片生成完成！保存在 docs/proposal_figures/ 目录")
