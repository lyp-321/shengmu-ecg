"""
深度学习模型评估脚本（不重新训练，直接加载已有 .pth 文件）
在 Patient-wise 测试集上评估所有 DL 模型，并生成 ML vs DL 性能对比雷达图
运行：python scripts/eval_dl_models.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import wfdb
from tqdm import tqdm

from app.algorithms.deep_models import (
    ResNet1D, SEResNet1D, TransformerECG, BiLSTMECG, TCN, InceptionECG
)

SEGMENT_LENGTH = 1000
NUM_CLASSES    = 3
BATCH_SIZE     = 64
MODEL_DIR      = 'app/algorithms/models'
RESULT_PATH    = 'experiments/results/dl_results_patient_wise.json'
RADAR_PATH     = 'experiments/results/ml_vs_dl_radar.png'

RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '209'
]

LABEL_MAP = {
    'N': 0, 'L': 0, 'R': 0,
    'V': 1, '/': 1,
    'A': 2, 'F': 2, 'f': 2, 'j': 2,
    'E': 2, 'e': 2, 'S': 2, 'a': 2,
    'J': 2, 'n': 2, 'Q': 2,
}

DL_MODEL_DEFS = {
    'ResNet1D':    lambda: ResNet1D(num_classes=NUM_CLASSES),
    'SEResNet1D':  lambda: SEResNet1D(num_classes=NUM_CLASSES),
    'BiLSTM':      lambda: BiLSTMECG(num_classes=NUM_CLASSES, hidden_size=128),
    'TCN':         lambda: TCN(num_classes=NUM_CLASSES),
    'Inception':   lambda: InceptionECG(num_classes=NUM_CLASSES),
    'Transformer': lambda: TransformerECG(num_classes=NUM_CLASSES, d_model=128, nhead=8),
}

DL_MODEL_FILES = {
    'ResNet1D':    'resnet1d_best.pth',
    'SEResNet1D':  'seresnet1d_best.pth',
    'BiLSTM':      'bilstm_best.pth',
    'TCN':         'tcn_best.pth',
    'Inception':   'inception_best.pth',
    'Transformer': 'transformer_best.pth',
}

# ML 已知结果（来自 train_traditional_ml.py 最新一轮）
ML_RESULTS = {
    'XGBoost':            {'accuracy': 0.9584, 'macro_f1': 0.5310, 'f1_per_class': [0.9788, 0.4603, 0.1406]},
    'LightGBM':           {'accuracy': 0.9722, 'macro_f1': 0.4298, 'f1_per_class': [0.9862, 0.2568, 0.0465]},
    'CatBoost':           {'accuracy': 0.9790, 'macro_f1': 0.4470, 'f1_per_class': [0.9895, 0.2782, 0.0732]},
    'Stacking(XGB+Cat)':  {'accuracy': 0.9830, 'macro_f1': 0.4440, 'f1_per_class': [0.9915, 0.3128, 0.0278]},
}


class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.X = torch.FloatTensor(signals)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_test_data(data_dir='data', num_samples=60000):
    """加载数据，返回 DL 格式测试集 + ML 格式测试集（同一批患者）"""
    print("加载 MIT-BIH 数据（Patient-wise 划分）...")

    # ── DL 数据（原始信号片段）──
    X_dl, y_dl, rec_dl = [], [], []
    for record in tqdm(RECORDS, desc="读取记录"):
        record_path = os.path.join(data_dir, record)
        if not os.path.exists(f"{record_path}.dat"):
            continue
        try:
            signal, _ = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            for sample, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in LABEL_MAP:
                    continue
                start = max(0, sample - SEGMENT_LENGTH // 2)
                end   = start + SEGMENT_LENGTH
                if end > len(signal):
                    continue
                seg = signal[start:end, 0].astype(np.float32)
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
                X_dl.append(seg)
                y_dl.append(LABEL_MAP[symbol])
                rec_dl.append(record)
        except Exception as e:
            print(f"  读取 {record} 失败: {e}")

    if len(X_dl) > num_samples:
        np.random.seed(42)
        idx = np.random.choice(len(X_dl), num_samples, replace=False)
        X_dl   = [X_dl[i]   for i in idx]
        y_dl   = [y_dl[i]   for i in idx]
        rec_dl = [rec_dl[i] for i in idx]

    X_dl_arr = np.array(X_dl)[:, np.newaxis, :]
    y_arr    = np.array(y_dl)
    rec_arr  = np.array(rec_dl)

    # 与训练脚本完全相同的划分（random_state=42）
    unique_pts = np.unique(rec_arr)
    _, test_pts = train_test_split(unique_pts, test_size=0.2, random_state=42)
    test_mask = np.isin(rec_arr, test_pts)

    X_test_dl = X_dl_arr[test_mask]
    y_test    = y_arr[test_mask]

    print(f"测试集: {X_test_dl.shape}, 类别分布: {np.bincount(y_test)}")
    print(f"测试患者 ({len(test_pts)}个): {sorted(test_pts)}")
    return X_test_dl, y_test


def evaluate_dl_model(model, test_loader, device, name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}\n模型: {name}")
    print(classification_report(y_true, y_pred,
                                 target_names=['Normal', 'PVC', 'Other'],
                                 zero_division=0))
    print("Confusion Matrix:\n", cm)

    return {
        'accuracy':         float((y_true == y_pred).mean()),
        'macro_f1':         float(macro),
        'f1_per_class':     f1_per.tolist(),
        'confusion_matrix': cm.tolist(),
    }


def plot_radar(dl_results: dict, save_path: str):
    """
    ML vs DL 性能对比雷达图
    5个维度：Accuracy / Macro-F1 / Normal-F1 / PVC-F1 / Other-F1
    """
    categories = ['Accuracy', 'Macro-F1', 'Normal-F1', 'PVC-F1', 'Other-F1']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                              subplot_kw=dict(polar=True))
    fig.suptitle('ML vs DL Performance Radar\n(Patient-wise Test Set)',
                 fontsize=15, fontweight='bold', y=1.02)

    # ── 颜色方案 ──
    ml_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#1abc9c']
    dl_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e91e63', '#ff6b35', '#34495e']

    def draw_radar(ax, all_models, colors, title):
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

        patches = []
        for (name, r), color in zip(all_models.items(), colors):
            fp = r['f1_per_class']
            values = [
                r['accuracy'],
                r['macro_f1'],
                fp[0] if len(fp) > 0 else 0,
                fp[1] if len(fp) > 1 else 0,
                fp[2] if len(fp) > 2 else 0,
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=4)
            ax.fill(angles, values, alpha=0.08, color=color)
            patches.append(mpatches.Patch(color=color, label=name))

        ax.legend(handles=patches, loc='upper right',
                  bbox_to_anchor=(1.35, 1.15), fontsize=9)

    draw_radar(axes[0], ML_RESULTS, ml_colors, 'ML Models')
    draw_radar(axes[1], dl_results,  dl_colors,  'DL Models')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 雷达图已保存: {save_path}")


def plot_bar_comparison(dl_results: dict, save_path: str):
    """ML 最优 vs DL 各模型的 PVC-F1 / Other-F1 / Macro-F1 柱状对比图"""
    all_models = {**ML_RESULTS, **dl_results}
    names      = list(all_models.keys())
    macro_f1s  = [r['macro_f1']                                    for r in all_models.values()]
    pvc_f1s    = [r['f1_per_class'][1] if len(r['f1_per_class'])>1 else 0 for r in all_models.values()]
    other_f1s  = [r['f1_per_class'][2] if len(r['f1_per_class'])>2 else 0 for r in all_models.values()]

    n_ml = len(ML_RESULTS)
    colors = ['#e74c3c'] * n_ml + ['#3498db'] * len(dl_results)

    x  = np.arange(len(names))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.4), 6))

    b1 = ax.bar(x - w, macro_f1s, w, label='Macro-F1',  color=[c + 'cc' for c in colors],
                edgecolor='white')
    b2 = ax.bar(x,     pvc_f1s,   w, label='PVC-F1',    color=colors, edgecolor='white')
    b3 = ax.bar(x + w, other_f1s, w, label='Other-F1',  color=[c + '88' for c in colors],
                edgecolor='white')

    # 分隔线
    ax.axvline(x=n_ml - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(n_ml * 0.5 - 0.5, 0.62, 'ML', ha='center', fontsize=11,
            color='#e74c3c', fontweight='bold')
    ax.text(n_ml + len(dl_results) * 0.5 - 0.5, 0.62, 'DL', ha='center',
            fontsize=11, color='#3498db', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 0.75)
    ax.set_title('ML vs DL: Macro-F1 / PVC-F1 / Other-F1 Comparison\n(Patient-wise Test Set)',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    bar_path = save_path.replace('radar', 'bar_comparison')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 柱状对比图已保存: {bar_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    X_test, y_test = load_test_data()
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # ── 评估 DL 模型 ──
    dl_results = {}
    for name, model_fn in DL_MODEL_DEFS.items():
        pth_path = os.path.join(MODEL_DIR, DL_MODEL_FILES[name])
        if not os.path.exists(pth_path):
            print(f"\n⚠️  {name}: 模型文件不存在，跳过")
            continue
        print(f"\n加载 {name} ...")
        model = model_fn()
        try:
            state = torch.load(pth_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue
        model = model.to(device)
        dl_results[name] = evaluate_dl_model(model, test_loader, device, name)

    if not dl_results:
        print("\n没有可评估的 DL 模型，请先运行 train_multimodal_models.py")
        return

    # ── 汇总打印 ──
    print(f"\n{'='*65}")
    print("DL 模型性能汇总（Patient-wise 测试集）")
    print(f"{'='*65}")
    print(f"{'Model':<15} {'Accuracy':>9} {'Macro-F1':>9} {'Normal-F1':>10} {'PVC-F1':>8} {'Other-F1':>9}")
    print("-" * 65)
    for name, r in dl_results.items():
        fp = r['f1_per_class']
        print(f"{name:<15} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} "
              f"{fp[0]:>10.4f} {fp[1] if len(fp)>1 else 0:>8.4f} "
              f"{fp[2] if len(fp)>2 else 0:>9.4f}")

    print(f"\n{'ML 参考（已知结果）':}")
    print(f"{'Model':<20} {'Accuracy':>9} {'Macro-F1':>9} {'PVC-F1':>8} {'Other-F1':>9}")
    print("-" * 55)
    for name, r in ML_RESULTS.items():
        fp = r['f1_per_class']
        print(f"{name:<20} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} "
              f"{fp[1] if len(fp)>1 else 0:>8.4f} {fp[2] if len(fp)>2 else 0:>9.4f}")

    # ── 保存结果 ──
    os.makedirs('experiments/results', exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dl_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ DL 结果已保存: {RESULT_PATH}")

    # ── 生成图表 ──
    print("\n生成对比图表...")
    plot_radar(dl_results, RADAR_PATH)
    plot_bar_comparison(dl_results, RADAR_PATH)

    # ── 关键发现提示 ──
    best_dl_name = max(dl_results, key=lambda k: dl_results[k]['macro_f1'])
    best_dl      = dl_results[best_dl_name]
    best_ml_pvc  = max(r['f1_per_class'][1] for r in ML_RESULTS.values())
    best_dl_pvc  = max(r['f1_per_class'][1] if len(r['f1_per_class'])>1 else 0
                       for r in dl_results.values())
    best_dl_other = max(r['f1_per_class'][2] if len(r['f1_per_class'])>2 else 0
                        for r in dl_results.values())

    print(f"\n{'='*65}")
    print("关键发现")
    print(f"{'='*65}")
    print(f"最佳 DL 模型: {best_dl_name}  Macro-F1={best_dl['macro_f1']:.4f}")
    print(f"ML 最佳 PVC-F1: {best_ml_pvc:.4f}  DL 最佳 PVC-F1: {best_dl_pvc:.4f}")
    print(f"DL 最佳 Other-F1: {best_dl_other:.4f}  (ML 最佳: {max(r[\"f1_per_class\"][2] for r in ML_RESULTS.values()):.4f})")
    if best_dl_other > 0.10:
        print("✅ DL Other-F1 > 0.10：深度卷积成功捕捉到 ML 41维特征无法描述的怪异波形")
    else:
        print("⚠️  DL Other-F1 仍偏低，类别2样本稀缺是主要瓶颈")


if __name__ == '__main__':
    main()
