"""
训练多模态深度学习模型
使用MIT-BIH数据集
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import wfdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

from app.algorithms.deep_models import (
    ResNet1D, SEResNet1D, TransformerECG, BiLSTMECG, TCN, InceptionECG
)
from app.algorithms.graph_models import GCN, GAT, STGCN, build_ecg_adjacency_matrix


class ECGDataset(Dataset):
    """ECG数据集"""
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def load_mitbih_data(data_dir='data', segment_length=1000, num_samples=5000):
    """
    加载MIT-BIH数据集（按患者划分，避免数据泄露）
    
    Args:
        data_dir: 数据目录
        segment_length: 信号片段长度
        num_samples: 采样数量
    
    Returns:
        X: 信号数据 (num_samples, 1, segment_length)
        y: 标签 (num_samples,)
        record_ids: 患者ID列表
    """
    print("加载MIT-BIH数据集...")
    print("⚠️  使用按患者划分（Patient-wise）策略，避免数据泄露")
    
    # MIT-BIH记录列表（部分）
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # 标签映射（简化为3类）
    # 0: 正常 (N)
    # 1: 室性早搏 (V)
    # 2: 其他异常
    label_map = {
        'N': 0, 'L': 0, 'R': 0,  # 正常
        'V': 1, '/': 1,           # 室性早搏
        'A': 2, 'F': 2, 'f': 2, 'j': 2  # 其他异常
    }
    
    X_list = []
    y_list = []
    record_ids_list = []  # 新增：记录每个样本属于哪个患者
    
    for record in tqdm(records[:30], desc="读取记录"):  # 使用30个患者（从10增加到30）
        try:
            # 读取信号和注释
            record_path = os.path.join(data_dir, record)
            
            # 检查文件是否存在
            if not os.path.exists(f"{record_path}.dat"):
                print(f"跳过不存在的记录: {record}")
                continue
            
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # 提取信号片段
            for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
                if symbol not in label_map:
                    continue
                
                # 提取以R波为中心的片段
                start = max(0, sample - segment_length // 2)
                end = start + segment_length
                
                if end > len(signal):
                    continue
                
                segment = signal[start:end, 0]  # 使用第一个导联
                
                # 归一化
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                
                X_list.append(segment)
                y_list.append(label_map[symbol])
                record_ids_list.append(record)  # 记录患者ID
                
                if len(X_list) >= num_samples:
                    break
            
            if len(X_list) >= num_samples:
                break
                
        except Exception as e:
            print(f"读取记录 {record} 失败: {e}")
            continue
    
    if len(X_list) == 0:
        print("警告: 未能加载MIT-BIH数据，使用模拟数据")
        # 生成模拟数据
        X_list = [np.random.randn(segment_length) for _ in range(num_samples)]
        y_list = [np.random.randint(0, 3) for _ in range(num_samples)]
        record_ids_list = [f'sim_{i%10}' for i in range(num_samples)]
    
    X = np.array(X_list)
    y = np.array(y_list)
    record_ids = np.array(record_ids_list)
    
    # 添加通道维度
    X = X[:, np.newaxis, :]  # (num_samples, 1, segment_length)
    
    print(f"数据加载完成: X.shape={X.shape}, y.shape={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    print(f"患者数量: {len(np.unique(record_ids))}")
    print(f"患者列表: {np.unique(record_ids)}")
    
    return X, y, record_ids


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, 
                device='cpu', model_name='model'):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        model_name: 模型名称
    
    Returns:
        训练历史
    """
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 第一个epoch也打印
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      f'app/algorithms/models/{model_name}_best.pth')
    
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    
    return history


def plot_training_history(histories, save_path='experiments/results/training_curves.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for name, history in histories.items():
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label=f'{name} Train')
        axes[0, 1].plot(history['val_loss'], label=f'{name} Val')
        
        # 准确率曲线
        axes[1, 0].plot(history['train_acc'], label=f'{name} Train')
        axes[1, 1].plot(history['val_acc'], label=f'{name} Val')
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")


def main():
    """主函数"""
    # ========== 配置：跳过已训练的模型 ==========
    # 设置为 True 的模型将被跳过
    skip_models = {
        'ResNet1D': True,      # 已训练完成
        'SEResNet1D': True,    # 已训练完成
        'Transformer': True,   # CPU太慢，暂时跳过
        'BiLSTM': False,
        'TCN': False,
        'Inception': False
    }
    # ==========================================
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据（返回record_ids用于按患者划分）
    # 增加样本数量以加载更多患者数据（从5000增加到30000）
    X, y, record_ids = load_mitbih_data(num_samples=30000)
    
    # ========== 按患者划分数据集（Patient-wise Split）==========
    print("\n" + "="*60)
    print("按患者划分数据集（避免数据泄露）")
    print("="*60)
    
    # 获取唯一的患者ID
    unique_patients = np.unique(record_ids)
    print(f"总患者数: {len(unique_patients)}")
    print(f"患者列表: {unique_patients}")
    
    # 按患者划分（60% 训练，20% 验证，20% 测试）
    from sklearn.model_selection import train_test_split
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )
    
    print(f"\n训练集患者 ({len(train_patients)}个): {train_patients}")
    print(f"验证集患者 ({len(val_patients)}个): {val_patients}")
    print(f"测试集患者 ({len(test_patients)}个): {test_patients}")
    
    # 根据患者ID划分数据
    train_mask = np.isin(record_ids, train_patients)
    val_mask = np.isin(record_ids, val_patients)
    test_mask = np.isin(record_ids, test_patients)
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    print(f"\n数据集划分:")
    print(f"训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"验证集: {X_val.shape}, 类别分布: {np.bincount(y_val)}")
    print(f"测试集: {X_test.shape}, 类别分布: {np.bincount(y_test)}")
    
    # 检查是否有数据泄露
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    if len(train_set & val_set) > 0 or len(train_set & test_set) > 0 or len(val_set & test_set) > 0:
        print(f"⚠️  警告: 发现数据泄露！")
    else:
        print(f"✅ 数据划分正确，无数据泄露")
    
    # 创建数据加载器
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 定义模型
    num_classes = 3
    models = {
        'ResNet1D': ResNet1D(num_classes=num_classes),
        'SEResNet1D': SEResNet1D(num_classes=num_classes),
        'Transformer': TransformerECG(num_classes=num_classes, d_model=128, nhead=8),
        'BiLSTM': BiLSTMECG(num_classes=num_classes, hidden_size=128),
        'TCN': TCN(num_classes=num_classes),
        'Inception': InceptionECG(num_classes=num_classes)
    }
    
    # 训练所有模型
    histories = {}
    
    for name, model in models.items():
        # 检查是否跳过该模型
        if skip_models.get(name, False):
            print(f"\n⏭️  跳过已训练的模型: {name}")
            # 检查模型文件是否存在
            model_path = f'app/algorithms/models/{name.lower()}_best.pth'
            if os.path.exists(model_path):
                print(f"   ✅ 模型文件存在: {model_path}")
            else:
                print(f"   ⚠️  警告: 模型文件不存在，但已设置跳过")
            continue
        
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=30,  # 可以根据需要调整
            lr=0.001,
            device=device,
            model_name=name.lower()
        )
        histories[name] = history
    
    # 绘制训练曲线
    if histories:  # 只有在有新训练的模型时才绘制
        plot_training_history(histories)
    else:
        print("\n⏭️  没有新训练的模型，跳过绘制训练曲线")
    
    # 测试所有模型（添加详细评估指标）
    print(f"\n{'='*60}")
    print("测试集评估（详细指标）")
    print(f"{'='*60}")
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    results = {}
    all_preds = []
    all_labels = []
    all_probs = []
    
    for name, model in models.items():
        # 检查模型文件是否存在
        model_path = f'app/algorithms/models/{name.lower()}_best.pth'
        if not os.path.exists(model_path):
            print(f"{name:15s}: ⏭️  模型文件不存在，跳过测试")
            continue
        
        # 加载最佳模型
        model.load_state_dict(torch.load(
            model_path,
            map_location=device
        ))
        model = model.to(device)
        model.eval()
        
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # 计算详细指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 每个类别的F1-Score
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 打印结果
        print(f"\n{name}")
        print(f"{'='*60}")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1-Score (Weighted): {f1:.4f}")
        print(f"\n每个类别的F1-Score:")
        print(f"  类别0 (正常): {f1_per_class[0]:.4f}")
        if len(f1_per_class) > 1:
            print(f"  类别1 (室性早搏): {f1_per_class[1]:.4f}")
        if len(f1_per_class) > 2:
            print(f"  类别2 (其他异常): {f1_per_class[2]:.4f}")
        
        print(f"\n混淆矩阵:")
        print(cm)
        
        print(f"\n详细分类报告:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['正常', '室性早搏', '其他异常'],
                                   zero_division=0))
        
        # 保存结果
        results[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        all_preds.append(y_pred)
        all_labels.append(y_true)
        all_probs.append(y_prob)
    
    # 打印汇总表格
    if results:
        print(f"\n{'='*60}")
        print("模型性能汇总")
        print(f"{'='*60}")
        print(f"{'模型':<15} {'准确率':<10} {'F1-Score':<10}")
        print("-"*60)
        for name, result in results.items():
            print(f"{name:<15} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f}")
        
        # 保存评估结果
        import json
        os.makedirs('experiments/results', exist_ok=True)
        with open('experiments/results/dl_results_patient_wise.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 保存评估结果到 experiments/results/dl_results_patient_wise.json")
    
    print(f"\n✅ 所有模型训练和评估完成！")
    print(f"模型保存在: app/algorithms/models/")
    print(f"结果保存在: experiments/results/")
    print(f"\n⚠️  注意: 使用按患者划分后，准确率可能会下降，但这是更真实的性能评估")
    print(f"📊 详细评估指标已保存，包括F1-Score、混淆矩阵等")
    print(f"\n{'='*60}")
    print("测试集评估")
    print(f"{'='*60}")
    
    results = {}
    for name, model in models.items():
        # 检查模型文件是否存在
        model_path = f'app/algorithms/models/{name.lower()}_best.pth'
        if not os.path.exists(model_path):
            print(f"{name:15s}: ⏭️  模型文件不存在，跳过测试")
            continue
        
        # 加载最佳模型
        model.load_state_dict(torch.load(
            model_path,
            map_location=device
        ))
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        results[name] = accuracy
        print(f"{name:15s}: {accuracy:.2f}%")
    
    # 保存结果
    with open('experiments/results/model_results.txt', 'w') as f:
        f.write("模型测试结果\n")
        f.write("="*60 + "\n")
        for name, acc in results.items():
            f.write(f"{name:15s}: {acc:.2f}%\n")
    
    print(f"\n✅ 所有模型训练完成！")
    print(f"模型保存在: app/algorithms/models/")
    print(f"结果保存在: experiments/results/")


if __name__ == '__main__':
    main()
