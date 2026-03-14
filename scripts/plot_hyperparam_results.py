"""
模型优化历程可视化
展示从初始版本到最终版本，每次改进带来的性能提升
运行：python scripts/plot_hyperparam_results.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = 'experiments/results/hyperparam_search.csv'
SAVE_DIR = 'experiments/results'

# ── 阶段定义（关键词 → 颜色/标签）──
# 匹配优先级从上到下，越具体越靠前
STAGES = [
    ('弃用RF',               '#ff6b35', 'S9: Drop RF\n(Stacking as Engine)'),
    ('扩充label_map',        '#e91e63', 'S8: Expand Labels\n(More Arrhythmia Types)'),
    ('XGB-CatBoost-Stacking','#1abc9c', 'S7: XGB+CatBoost\nStacking'),
    ('LinearSVC',            '#9b59b6', 'S6: LinearSVC\n(Not Converged)'),
    ('单心拍频带',           '#2ecc71', 'S5: Fix Freq Band\n(Single-beat Band)'),
    ('旧频带',               '#3498db', 'S4: Cost-Sensitive\n(Wrong Freq Band)'),
    ('SMOTE(1:1:1)',          '#f1c40f', 'S3: Add SMOTE\n(1:1:1 Balance)'),
    ('第一次',               '#e67e22', 'S2: Fix Leakage\n(Patient-wise Split)'),
    ('数据泄露',             '#e74c3c', 'S1: Data Leakage\n(Beat-wise Split)'),
]

# note里没有上述关键词时的fallback映射
FALLBACK_MAP = {
    'max_depth': '弃用RF',      # max_depth实验 = S9阶段（弃用RF那轮）
    'class2权重20': '扩充label_map',  # 300棵树+class2权重20 = S8阶段
}
# 默认 stage（未匹配时）
DEFAULT_STAGE = ('单心拍频带', '#2ecc71', 'S5: Fix Freq Band\n(Single-beat Band)')

STAGE_ORDER = [s[0] for s in STAGES][::-1]   # S1 → S8


def get_stage(note: str):
    note = str(note)
    for key, color, label in STAGES:
        if key in note:
            return key
    # fallback：按note内容推断
    for fb_key, stage_key in FALLBACK_MAP.items():
        if fb_key in note:
            return stage_key
    return DEFAULT_STAGE[0]


def stage_color(key):
    for k, c, _ in STAGES:
        if k == key:
            return c
    return DEFAULT_STAGE[1]


def stage_label(key):
    for k, _, l in STAGES:
        if k == key:
            return l
    return DEFAULT_STAGE[2]


def plot_optimization_journey():
    df = pd.read_csv(CSV_PATH)
    df = df[df['macro_f1'] > 0].copy()
    df['stage']  = df['note'].apply(get_stage)
    df['exp_no'] = range(1, len(df) + 1)

    os.makedirs(SAVE_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ECG Model Optimization Journey', fontsize=16, fontweight='bold', y=0.98)

    # ── 图1：优化历程折线图 ──
    ax = axes[0, 0]
    for col, label, style, color in [
        ('macro_f1', 'Macro-F1', 'o-',  '#2c3e50'),
        ('f1_pvc',   'PVC F1',   's--', '#e74c3c'),
        ('f1_other', 'Other F1', '^:',  '#3498db'),
    ]:
        ax.plot(df['exp_no'], df[col], style, label=label,
                color=color, linewidth=2, markersize=5, alpha=0.85)

    drawn = set()
    for _, row in df.iterrows():
        s = row['stage']
        if s not in drawn:
            ax.axvline(x=row['exp_no'], color=stage_color(s), linestyle='--', alpha=0.4, linewidth=1.5)
            idx = STAGE_ORDER.index(s) + 1 if s in STAGE_ORDER else '?'
            ax.text(row['exp_no'] + 0.1, 0.58, f"S{idx}",
                    color=stage_color(s), fontsize=8, fontweight='bold')
            drawn.add(s)

    # 标注平台期说明（S4-S8 XGBoost macro_f1 均为 0.531，改动方向正确但未突破瓶颈）
    ax.annotate('Plateau: structural\nimprovements, no\nF1 gain on XGB',
                xy=(22, 0.531), xytext=(26, 0.42),
                fontsize=7, color='#7f8c8d',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.8))

    ax.set_title('Performance over Optimization Steps', fontweight='bold')
    ax.set_xlabel('Experiment No.')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 0.70)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # ── 图2：各阶段最佳 Macro-F1 柱状图（含 Stacking PVC F1 叠加标注）──
    ax = axes[0, 1]
    stage_best    = df.groupby('stage')['macro_f1'].max()
    stage_pvc     = df.groupby('stage')['f1_pvc'].max()
    ordered       = [s for s in STAGE_ORDER if s in stage_best.index]
    values        = [stage_best[s] for s in ordered]
    pvc_values    = [stage_pvc[s]  for s in ordered]
    bar_colors    = [stage_color(s) for s in ordered]
    short_labels  = [f"S{STAGE_ORDER.index(s)+1}" for s in ordered]

    bars = ax.bar(short_labels, values, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.85)
    # 叠加 PVC F1 折线，突出少数类改善趋势
    ax.plot(short_labels, pvc_values, 'D--', color='#e74c3c', linewidth=1.8,
            markersize=6, label='PVC F1', zorder=5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Best Macro-F1 per Stage\n(+ PVC F1 trend)', fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
    legend_patches = [mpatches.Patch(color=stage_color(s), label=stage_label(s)) for s in ordered]
    ax.legend(handles=legend_patches + [
        mpatches.Patch(color='#e74c3c', label='PVC F1 trend')
    ], fontsize=7, loc='upper left')

    # ── 图3：三类 F1 对比（Stacking 单独标注）──
    ax = axes[1, 0]
    x, w = np.arange(len(ordered)), 0.25

    def best_f1(stage, col):
        sub = df[df['stage'] == stage]
        return sub[col].max() if len(sub) > 0 else 0

    normal_vals = [best_f1(s, 'f1_normal') for s in ordered]
    pvc_vals    = [best_f1(s, 'f1_pvc')    for s in ordered]
    other_vals  = [best_f1(s, 'f1_other')  for s in ordered]

    ax.bar(x - w, normal_vals, w, label='Normal',  color='#2ecc71', alpha=0.85)
    ax.bar(x,     pvc_vals,    w, label='PVC',     color='#e74c3c', alpha=0.85)
    ax.bar(x + w, other_vals,  w, label='Other',   color='#3498db', alpha=0.85)

    # 标注 Stacking 的 PVC F1（S7 阶段的真实 Stacking 结果）
    stacking_rows = df[df['model'].str.contains('Stacking', na=False)]
    if len(stacking_rows) > 0:
        best_stack = stacking_rows.loc[stacking_rows['f1_pvc'].idxmax()]
        s7_idx = ordered.index('XGB-CatBoost-Stacking') if 'XGB-CatBoost-Stacking' in ordered else -1
        if s7_idx >= 0:
            ax.annotate(f"Stacking\nPVC={best_stack['f1_pvc']:.3f}",
                        xy=(s7_idx, best_stack['f1_pvc']),
                        xytext=(s7_idx + 0.8, best_stack['f1_pvc'] + 0.08),
                        fontsize=7, color='#c0392b', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1))

    ax.set_title('F1 per Class across Stages', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.1)
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # ── 图4：相对 S1 的提升幅度 ──
    ax = axes[1, 1]
    baseline     = stage_best.get('数据泄露', values[0] if values else 0.33)
    improvements = [(v - baseline) / (baseline + 1e-8) * 100 for v in values]
    bar_colors2  = ['#e74c3c' if v < 0 else '#2ecc71' for v in improvements]

    bars = ax.bar(short_labels, improvements, color=bar_colors2, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if val >= 0 else -3),
                f'{val:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_title('Macro-F1 Improvement vs S1 (%)', fontweight='bold')
    ax.set_ylabel('Improvement (%)'); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'optimization_journey.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {save_path}")

    # 控制台汇总
    print("\n── Optimization Summary ──")
    print(f"{'Stage':<6} {'Best MacroF1':>12} {'PVC F1':>8} {'Other F1':>10} {'vs S1':>8}")
    print("-" * 48)
    for s, v, imp in zip(ordered, values, improvements):
        row = df[df['stage'] == s].loc[df[df['stage'] == s]['macro_f1'].idxmax()]
        idx = STAGE_ORDER.index(s) + 1
        print(f"S{idx:<5} {v:>12.4f} {row['f1_pvc']:>8.4f} {row['f1_other']:>10.4f} {imp:>+7.1f}%")


if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        print(f"找不到: {CSV_PATH}\n请先运行: python scripts/train_traditional_ml.py")
    else:
        plot_optimization_journey()

        df = pd.read_csv(CSV_PATH)
        df = df[df['macro_f1'] > 0].copy()
        top5 = df.nlargest(5, 'macro_f1')[
            ['timestamp', 'model', 'class_weight', 'smote_ratio',
             'n_estimators', 'macro_f1', 'f1_pvc', 'f1_other', 'note']
        ]
        print("\nTop 5 参数组合（按 Macro-F1）:")
        print(top5.to_string(index=False))
