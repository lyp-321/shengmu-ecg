"""
训练多模态深度学习模型（ResNet1D / SEResNet1D / BiLSTM / TCN / Inception / Transformer）
数据：MIT-BIH，按患者划分（Patient-wise Split），避免数据泄露
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from datetime import datetime
import wfdb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.algorithms.deep_models import (
    ResNet1D, SEResNet1D, TransformerECG, BiLSTMECG, TCN, InceptionECG
)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'num_samples':         None,       # None = 使用全量数据，不随机截断
    'segment_length':      1000,
    'num_epochs':          60,         # 给 Transformer/BiLSTM 足够收敛空间
    'batch_size':          64,
    'num_classes':         3,
    'early_stop_patience': 12,         # 稍宽松，避免在平台期过早停止
    # 正常:室早:其他异常 ≈ 84:15:1，采样后均衡，class_weight 只做轻微辅助
    'class_weight':        [1.0, 1.5, 2.0],
    'label_smoothing':     0.1,        # 防止过拟合，软化标签
    'mixup_alpha':         0.2,        # MixUp 增强强度
    # 各模型独立学习率（降低初始 lr，配合数据增强）
    'lr': {
        'ResNet1D':    0.0005,
        'SEResNet1D':  0.0005,
        'Inception':   0.0005,
        'TCN':         0.0003,
        'BiLSTM':      0.0002,
        'Transformer': 0.0002,
    },
    # 设为 True 跳过已训练的模型（已有 .pth 文件时用）
    'skip': {
        'ResNet1D':    False,
        'SEResNet1D':  False,
        'Transformer': False,
        'BiLSTM':      False,
        'TCN':         False,
        'Inception':   False,
    }
}

CSV_PATH = 'experiments/results/hyperparam_search.csv'
# DataLoader worker 数：GPU 训练时用多进程预加载，CPU 训练时用 0
NUM_WORKERS = 4

# 实际存在的 MIT-BIH 标准记录（27个患者）
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


# ============================================================
# 数据集
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = torch.FloatTensor(signals)
        self.labels  = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx].clone()
        if self.augment:
            # 1. 加高斯噪声（SNR ~20dB）
            if torch.rand(1) < 0.5:
                x += torch.randn_like(x) * 0.05
            # 2. 幅度缩放 [0.8, 1.2]
            if torch.rand(1) < 0.5:
                x *= (0.8 + torch.rand(1) * 0.4)
            # 3. 时间偏移（循环移位，最多 ±50 个采样点）
            if torch.rand(1) < 0.5:
                shift = torch.randint(-50, 51, (1,)).item()
                x = torch.roll(x, shift, dims=-1)
            # 4. 基线漂移（低频正弦）
            if torch.rand(1) < 0.3:
                t = torch.linspace(0, 2 * 3.14159, x.shape[-1])
                freq = torch.rand(1) * 0.5  # 0~0.5 Hz
                x += torch.sin(2 * 3.14159 * freq * t).unsqueeze(0) * 0.1
        return x, self.labels[idx]


# ============================================================
# 数据加载
# ============================================================
def load_mitbih_data(data_dir='data'):
    """加载 MIT-BIH，返回 (X, y, record_ids)"""
    print("加载 MIT-BIH 数据集（Patient-wise）...")
    seg_len = CONFIG['segment_length']

    X_list, y_list, rec_list = [], [], []

    for record in tqdm(RECORDS, desc="读取记录"):
        record_path = os.path.join(data_dir, record)
        if not os.path.exists(f"{record_path}.dat"):
            print(f"  跳过不存在的记录: {record}")
            continue
        try:
            signal, _ = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            for sample, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in LABEL_MAP:
                    continue
                start = max(0, sample - seg_len // 2)
                end   = start + seg_len
                if end > len(signal):
                    continue
                seg = signal[start:end, 0].astype(np.float32)
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)

                X_list.append(seg)
                y_list.append(LABEL_MAP[symbol])
                rec_list.append(record)
        except Exception as e:
            print(f"  读取 {record} 失败: {e}")

    # 随机采样
    n = CONFIG['num_samples']
    if n is not None and len(X_list) > n:
        np.random.seed(42)
        idx = np.random.choice(len(X_list), n, replace=False)
        X_list   = [X_list[i]   for i in idx]
        y_list   = [y_list[i]   for i in idx]
        rec_list = [rec_list[i] for i in idx]

    X = np.array(X_list)[:, np.newaxis, :]   # (N, 1, seg_len)
    y = np.array(y_list)
    record_ids = np.array(rec_list)

    print(f"加载完成: X={X.shape}, 类别分布={np.bincount(y)}, 患者数={len(np.unique(record_ids))}")
    return X, y, record_ids


# ============================================================
# Focal Loss（解决类别不平衡：自动降低简单样本权重，聚焦难分类样本）
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing
    gamma=2：对已学好的正常类梯度衰减，把注意力集中到 PVC/Other
    label_smoothing：软化标签，防止过拟合
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, num_classes=3):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight
        self.smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # logits: (B, C)  targets: (B,)
        log_prob = F.log_softmax(logits, dim=1)
        prob     = torch.exp(log_prob)
        
        # Label smoothing: 真实标签 = (1-ε) * one_hot + ε/C
        if self.smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(log_prob)
                smooth_targets.fill_(self.smoothing / self.num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)
            # 用软标签计算 loss
            p_t = (prob * smooth_targets).sum(dim=1)
            log_p_t = (log_prob * smooth_targets).sum(dim=1)
        else:
            p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
            log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_w = (1 - p_t) ** self.gamma
        loss = -focal_w * log_p_t
        
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        return loss.mean()


# ============================================================
# MixUp 数据增强（batch 层面）
# ============================================================
def mixup_data(x, y, alpha=0.2):
    """MixUp: 混合两个样本，软化决策边界"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss: 加权两个标签的 loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 训练
# ============================================================
def train_model(model, train_loader, val_loader, device, model_name, class_weights, lr=0.001):
    import time as _time
    model = model.to(device)
    criterion = FocalLoss(
        gamma=2.0, 
        weight=class_weights.to(device),
        label_smoothing=CONFIG['label_smoothing'],
        num_classes=CONFIG['num_classes']
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # ReduceLROnPlateau：val macro-F1 连续 5 轮不涨就 lr×0.5，最低降到 1e-6
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_f1    = 0.0
    no_improve = 0
    patience   = CONFIG['early_stop_patience']
    history    = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    t_start    = _time.time()

    for epoch in range(CONFIG['num_epochs']):
        # ── 训练 ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # MixUp 数据增强（50% 概率）
            if np.random.rand() < 0.5:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
                optimizer.zero_grad()
                out = model(inputs)
                loss = mixup_criterion(criterion, out, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels)
            
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += out.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)

        t_loss /= len(train_loader)
        t_acc   = 100. * t_correct / t_total

        # ── 验证 ──
        model.eval()
        v_loss, v_preds, v_true = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out  = model(inputs)
                v_loss += criterion(out, labels).item()
                v_preds.extend(out.argmax(1).cpu().numpy())
                v_true.extend(labels.cpu().numpy())

        v_loss /= len(val_loader)
        v_acc   = 100. * sum(p == t for p, t in zip(v_preds, v_true)) / len(v_true)
        v_f1    = f1_score(v_true, v_preds, average='macro', zero_division=0)

        scheduler.step(v_f1)   # 按 F1 调整 lr，而不是 loss

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{epoch+1:2d}/{CONFIG['num_epochs']}] "
                  f"train loss={t_loss:.4f} acc={t_acc:.1f}% | "
                  f"val loss={v_loss:.4f} acc={v_acc:.1f}% macro-F1={v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1    = v_f1
            no_improve = 0
            torch.save(model.state_dict(), f'app/algorithms/models/{model_name}_best.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}（{patience}轮无提升）")
                break

    cur_lr     = optimizer.param_groups[0]['lr']
    elapsed    = _time.time() - t_start
    history['train_time'] = elapsed
    print(f"  训练完成 → 最佳验证 Macro-F1={best_f1:.4f}  最终lr={cur_lr:.2e}  耗时={elapsed:.1f}s")
    return history


# ============================================================
# 评估
# ============================================================
def evaluate_model(model, test_loader, device, name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{name}")
    print(classification_report(y_true, y_pred,
                                 target_names=['Normal', 'PVC', 'Other'],
                                 zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy':      float((y_true == y_pred).mean()),
        'macro_f1':      float(macro),
        'f1_per_class':  f1_per.tolist(),
        'confusion_matrix': cm.tolist(),
    }


# ============================================================
# 实验记录（与 ML 保持一致，追加到同一个 CSV）
# ============================================================
def save_results_to_csv(results: dict, csv_path: str = CSV_PATH):
    """将 DL 评估结果追加到 hyperparam_search.csv"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cw_str = str(CONFIG['class_weight']).replace(' ', '')

    rows = []
    for name, r in results.items():
        fp = r.get('f1_per_class', [0, 0, 0])
        rows.append({
            'timestamp':    ts,
            'model':        f'DL_{name}',
            'class_weight': cw_str,
            'smote_ratio':  'N/A',
            'n_estimators': 'N/A',
            'accuracy':     round(r.get('accuracy', 0), 4),
            'macro_f1':     round(r.get('macro_f1', 0), 4),
            'f1_normal':    round(fp[0] if len(fp) > 0 else 0, 4),
            'f1_pvc':       round(fp[1] if len(fp) > 1 else 0, 4),
            'f1_other':     round(fp[2] if len(fp) > 2 else 0, 4),
            'train_time':   round(r.get('train_time', 0), 2),
            'note':         f'DL patient-wise lr={CONFIG["lr"].get(name, "?")} early_stop={CONFIG["early_stop_patience"]}',
        })

    new_df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    print(f"\n✅ DL 实验结果已追加到 {csv_path}（共 {len(combined)} 条记录）")


# ============================================================
# 绘图
# ============================================================
def plot_training_curves(histories):
    save_path = 'experiments/results/training_curves.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in histories.items():
        axes[0].plot(h['train_loss'], label=f'{name} train')
        axes[0].plot(h['val_loss'],   label=f'{name} val', linestyle='--')
        axes[1].plot(h['val_acc'],    label=name)

    axes[0].set_title('Loss Curves');  axes[0].set_xlabel('Epoch'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载数据
    X, y, record_ids = load_mitbih_data()

    # 2. 按患者划分（60% 训练 / 20% 验证 / 20% 测试）
    print("\n按患者划分数据集...")
    unique_patients = np.unique(record_ids)
    train_val_pts, test_pts = train_test_split(unique_patients, test_size=0.2, random_state=42)
    train_pts, val_pts      = train_test_split(train_val_pts,   test_size=0.25, random_state=42)

    print(f"  训练: {len(train_pts)}个患者  验证: {len(val_pts)}个患者  测试: {len(test_pts)}个患者")
    assert not (set(train_pts) & set(test_pts)), "数据泄露！"

    def split(mask): return X[mask], y[mask]
    X_train, y_train = split(np.isin(record_ids, train_pts))
    X_val,   y_val   = split(np.isin(record_ids, val_pts))
    X_test,  y_test  = split(np.isin(record_ids, test_pts))

    print(f"  训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"  验证集: {X_val.shape},   类别分布: {np.bincount(y_val)}")
    print(f"  测试集: {X_test.shape},  类别分布: {np.bincount(y_test)}")

    # 3. WeightedRandomSampler：让每个 epoch 里三类样本数量接近均衡
    # 原理：对少数类样本赋予更高的采样概率，每个 epoch 重复采样少数类
    # 比 SMOTE 快（不生成新样本），比纯 class_weight 稳定（梯度不震荡）
    counts   = np.bincount(y_train)                          # [n0, n1, n2]
    # 每个样本的采样权重 = 1 / 该类样本数（类越少权重越大）
    sample_weights = np.array([1.0 / counts[yi] for yi in y_train], dtype=np.float32)
    sampler  = WeightedRandomSampler(
        weights     = torch.FloatTensor(sample_weights),
        num_samples = len(y_train),   # 保持每 epoch 步数不变，只改采样分布
        replacement = True
    )
    print(f"\n类别分布: {counts}  →  WeightedRandomSampler（每epoch {len(y_train)} 个样本，三类均衡）")

    # 采样已均衡分布，class_weight 只做轻微辅助，避免过度惩罚
    class_weights = torch.FloatTensor(CONFIG['class_weight'])
    print(f"CrossEntropyLoss 辅助权重: {class_weights.numpy()}")

    # 4. DataLoader（训练集用 sampler + augment，不能同时用 shuffle=True）
    nw = NUM_WORKERS if device.type == 'cuda' else 0
    pm = device.type == 'cuda'
    train_loader = DataLoader(ECGDataset(X_train, y_train, augment=True), batch_size=CONFIG['batch_size'],
                              sampler=sampler, num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val,   augment=False), batch_size=CONFIG['batch_size'],
                              num_workers=nw, pin_memory=pm)
    test_loader  = DataLoader(ECGDataset(X_test,  y_test,  augment=False), batch_size=CONFIG['batch_size'],
                              num_workers=nw, pin_memory=pm)

    # 5. 定义模型
    nc = CONFIG['num_classes']
    model_defs = {
        'ResNet1D':    ResNet1D(num_classes=nc),
        'SEResNet1D':  SEResNet1D(num_classes=nc),
        'Transformer': TransformerECG(num_classes=nc, d_model=128, nhead=8),
        'BiLSTM':      BiLSTMECG(num_classes=nc, hidden_size=128),
        'TCN':         TCN(num_classes=nc),
        'Inception':   InceptionECG(num_classes=nc),
    }

    # 6. 训练
    os.makedirs('app/algorithms/models', exist_ok=True)
    histories = {}

    for name, model in model_defs.items():
        if CONFIG['skip'].get(name, False):
            model_path = f'app/algorithms/models/{name.lower()}_best.pth'
            status = "✅ 存在" if os.path.exists(model_path) else "⚠️ 不存在"
            print(f"\n⏭️  跳过 {name}（{status}）")
            continue

        # Transformer 在长序列上显存消耗大，自动降 batch_size
        if name == 'Transformer' and device.type == 'cuda':
            # 重新创建独立的 sampler，避免与 train_loader 共享迭代器状态
            t_sampler = WeightedRandomSampler(
                weights     = torch.FloatTensor(sample_weights),
                num_samples = len(y_train),
                replacement = True
            )
            t_loader = DataLoader(ECGDataset(X_train, y_train, augment=True), batch_size=32,
                                  sampler=t_sampler, num_workers=nw, pin_memory=pm)
            v_loader = DataLoader(ECGDataset(X_val,   y_val,   augment=False), batch_size=32,
                                  num_workers=nw, pin_memory=pm)
            print(f"\n{'='*60}\n训练模型: {name}（batch_size=32 防 OOM）\n{'='*60}")
        else:
            t_loader, v_loader = train_loader, val_loader
            print(f"\n{'='*60}\n训练模型: {name}\n{'='*60}")

        model_lr = CONFIG['lr'].get(name, 0.001)
        histories[name] = train_model(model, t_loader, v_loader, device, name.lower(), class_weights, lr=model_lr)

    # 7. 绘制训练曲线
    if histories:
        plot_training_curves(histories)

    # 8. 测试集评估
    print(f"\n{'='*60}\n测试集评估\n{'='*60}")
    results = {}

    for name, model in model_defs.items():
        model_path = f'app/algorithms/models/{name.lower()}_best.pth'
        if not os.path.exists(model_path):
            print(f"\n{name}: 模型文件不存在，跳过")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        results[name] = evaluate_model(model, test_loader, device, name)
        results[name]['train_time'] = histories.get(name, {}).get('train_time', 0.0)

    # 9. 汇总打印
    if results:
        print(f"\n{'='*60}\nModel Performance Summary\n{'='*60}")
        print(f"{'Model':<15} {'Macro-F1':>9} {'Normal-F1':>10} {'PVC-F1':>8} {'Other-F1':>9}")
        print("-" * 55)
        for name, r in results.items():
            fp = r['f1_per_class']
            print(f"{name:<15} {r['macro_f1']:>9.4f} {fp[0]:>10.4f} "
                  f"{fp[1] if len(fp)>1 else 0:>8.4f} {fp[2] if len(fp)>2 else 0:>9.4f}")

        os.makedirs('experiments/results', exist_ok=True)
        with open('experiments/results/dl_results_patient_wise.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n✅ 结果已保存: experiments/results/dl_results_patient_wise.json")

        # 追加到 CSV（与 ML 实验记录统一）
        save_results_to_csv(results)

    print("\n✅ 完成！模型保存在 app/algorithms/models/")


if __name__ == '__main__':
    main()
