"""
训练多模态深度学习模型（ResNet1D / SEResNet1D / BiLSTM / TCN / Inception / Transformer）
数据：MIT-BIH，按患者划分（Patient-wise Split），避免数据泄露
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt

from app.algorithms.deep_models import (
    ResNet1D, SEResNet1D, TransformerECG, BiLSTMECG, TCN, InceptionECG
)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'num_samples':      60000,
    'segment_length':   1000,
    'num_epochs':       30,
    'batch_size':       32,
    'lr':               0.001,
    'num_classes':      3,
    # 设为 True 跳过已训练的模型
    'skip': {
        'ResNet1D':    False,
        'SEResNet1D':  False,
        'Transformer': False,
        'BiLSTM':      False,
        'TCN':         False,
        'Inception':   False,
    }
}

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
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels  = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


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
    if len(X_list) > n:
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
# 训练
# ============================================================
def train_model(model, train_loader, val_loader, device, model_name, class_weights):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_f1   = 0.0
    best_acc  = 0.0
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(CONFIG['num_epochs']):
        # ── 训练 ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(inputs)
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

        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{epoch+1:2d}/{CONFIG['num_epochs']}] "
                  f"train loss={t_loss:.4f} acc={t_acc:.1f}% | "
                  f"val loss={v_loss:.4f} acc={v_acc:.1f}% macro-F1={v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1  = v_f1
            best_acc = v_acc
            torch.save(model.state_dict(), f'app/algorithms/models/{model_name}_best.pth')

    print(f"  训练完成 → 最佳验证 Macro-F1={best_f1:.4f}  Acc={best_acc:.1f}%")
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
                                 target_names=['正常', '室性早搏', '其他异常'],
                                 zero_division=0))
    print("混淆矩阵:")
    print(cm)

    return {
        'macro_f1':      float(macro),
        'f1_per_class':  f1_per.tolist(),
        'confusion_matrix': cm.tolist(),
    }


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

    # 3. SMOTE（训练集）
    print("\n应用 SMOTE 过采样...")
    try:
        from imblearn.over_sampling import SMOTE
        n, c, l = X_train.shape
        X_2d = X_train.reshape(n, -1)
        X_2d, y_train = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_2d, y_train)
        X_train = X_2d.reshape(-1, c, l)
        print(f"  过采样后: {np.bincount(y_train)}")
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0])
    except ImportError:
        print("  未安装 imbalanced-learn，跳过 SMOTE")
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(cw)

    print(f"  类别权重: {class_weights.numpy().round(2)}")

    # 4. DataLoader
    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val),   batch_size=CONFIG['batch_size'])
    test_loader  = DataLoader(ECGDataset(X_test,  y_test),  batch_size=CONFIG['batch_size'])

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

        print(f"\n{'='*60}\n训练模型: {name}\n{'='*60}")
        histories[name] = train_model(model, train_loader, val_loader, device, name.lower(), class_weights)

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
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        results[name] = evaluate_model(model, test_loader, device, name)

    # 9. 汇总打印
    if results:
        print(f"\n{'='*60}\n模型性能汇总\n{'='*60}")
        print(f"{'模型':<15} {'宏F1':>8} {'正常F1':>8} {'室早F1':>8} {'其他F1':>8}")
        print("-" * 50)
        for name, r in results.items():
            fp = r['f1_per_class']
            print(f"{name:<15} {r['macro_f1']:>8.4f} {fp[0]:>8.4f} "
                  f"{fp[1] if len(fp)>1 else 0:>8.4f} {fp[2] if len(fp)>2 else 0:>8.4f}")

        os.makedirs('experiments/results', exist_ok=True)
        with open('experiments/results/dl_results_patient_wise.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n✅ 结果已保存: experiments/results/dl_results_patient_wise.json")

    print("\n✅ 完成！模型保存在 app/algorithms/models/")


if __name__ == '__main__':
    main()
