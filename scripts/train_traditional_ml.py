"""
训练传统机器学习模型
XGBoost, LightGBM, CatBoost, RandomForest, SVM
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost未安装，将跳过CatBoost训练")

import joblib
import wfdb
from tqdm import tqdm
from scipy import signal as scipy_signal
import pywt


def extract_time_domain_features(ecg_signal, fs=360):
    """
    提取时域特征
    
    Args:
        ecg_signal: ECG信号
        fs: 采样率
    
    Returns:
        特征字典
    """
    features = {}
    
    # 基本统计特征
    features['mean'] = np.mean(ecg_signal)
    features['std'] = np.std(ecg_signal)
    features['max'] = np.max(ecg_signal)
    features['min'] = np.min(ecg_signal)
    features['median'] = np.median(ecg_signal)
    features['range'] = features['max'] - features['min']
    
    # R波检测（简化版）
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ecg_signal, distance=int(0.6*fs), prominence=0.3)
    
    if len(peaks) > 1:
        # RR间期
        rr_intervals = np.diff(peaks) / fs * 1000  # 转换为毫秒
        
        # HRV特征
        features['hr_mean'] = 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        features['hr_std'] = np.std(60000 / rr_intervals) if len(rr_intervals) > 0 else 0
        features['sdnn'] = np.std(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
    else:
        features['hr_mean'] = 0
        features['hr_std'] = 0
        features['sdnn'] = 0
        features['rmssd'] = 0
        features['pnn50'] = 0
    
    return features


def extract_frequency_domain_features(ecg_signal, fs=360):
    """
    提取频域特征
    
    Args:
        ecg_signal: ECG信号
        fs: 采样率
    
    Returns:
        特征字典
    """
    features = {}
    
    # FFT
    fft_vals = np.fft.fft(ecg_signal)
    fft_freq = np.fft.fftfreq(len(ecg_signal), 1/fs)
    
    # 功率谱
    power = np.abs(fft_vals) ** 2
    
    # 频带能量
    vlf_band = (fft_freq >= 0.003) & (fft_freq < 0.04)
    lf_band = (fft_freq >= 0.04) & (fft_freq < 0.15)
    hf_band = (fft_freq >= 0.15) & (fft_freq < 0.4)
    
    features['vlf_power'] = np.sum(power[vlf_band])
    features['lf_power'] = np.sum(power[lf_band])
    features['hf_power'] = np.sum(power[hf_band])
    features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-8)
    
    # 小波变换特征
    coeffs = pywt.wavedec(ecg_signal, 'db4', level=5)
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_energy_level_{i}'] = np.sum(coeff ** 2)
        features[f'wavelet_std_level_{i}'] = np.std(coeff)
    
    return features


def load_and_extract_features(data_dir='data', num_samples=5000):
    """
    加载MIT-BIH数据并提取特征（按患者划分，避免数据泄露）
    
    Args:
        data_dir: 数据目
        num_samples: 样本数量
    
    Returns:
        X: 特征矩阵
        y: 标签
        record_ids: 患者ID列表（用于按患者划分）
        feature_names:
        第3步：误诊风险预警机制 特征名称
    """
    print("加载MIT-BIH数据并提取特征...")
    print("⚠️  使用按患者划分（Patient-wise）策略，避免数据泄露")
    
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208'
    ]
    
    label_map = {
        'N': 0, 'L': 0, 'R': 0,  # 正常
        'V': 1, '/': 1,           # 室性早搏
        'A': 2, 'F': 2, 'f': 2, 'j': 2  # 其他异常
    }
    
    features_list = []
    labels_list = []
    record_ids_list = []  # 新增：记录每个样本属于哪个患者
    
    for record in tqdm(records, desc="提取特征"):  # 使用全部患者
        try:
            record_path = os.path.join(data_dir, record)
            
            if not os.path.exists(f"{record_path}.dat"):
                continue
            
            signal_data, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            fs = fields['fs']
            
            for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
                if symbol not in label_map:
                    continue
                
                # 提取片段
                start = max(0, sample - 500)
                end = min(len(signal_data), sample + 500)
                segment = signal_data[start:end, 0]
                
                if len(segment) < 100:
                    continue
                
                # 提取特征
                time_features = extract_time_domain_features(segment, fs)
                freq_features = extract_frequency_domain_features(segment, fs)
                
                # 合并特征
                all_features = {**time_features, **freq_features}
                features_list.append(all_features)
                labels_list.append(label_map[symbol])
                record_ids_list.append(record)  # 记录患者ID
                
        except Exception as e:
            print(f"处理记录 {record} 失败: {e}")
            continue
    
    # 读完所有患者后再限制总样本数（随机采样，保持分布）
    if len(features_list) > num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(features_list), num_samples, replace=False)
        features_list = [features_list[i] for i in indices]
        labels_list = [labels_list[i] for i in indices]
        record_ids_list = [record_ids_list[i] for i in indices]
    
    if len(features_list) == 0:
        print("警告: 未能加载数据，使用模拟特征")
        # 生成模拟特征
        num_features = 35
        features_list = [
            {f'feature_{i}': np.random.randn() for i in range(num_features)}
            for _ in range(num_samples)
        ]
        labels_list = [np.random.randint(0, 3) for _ in range(num_samples)]
        record_ids_list = [f'sim_{i%10}' for i in range(num_samples)]
    
    # 转换为DataFrame
    df = pd.DataFrame(features_list)
    X = df.values
    y = np.array(labels_list)
    record_ids = np.array(record_ids_list)
    
    # 处理NaN值
    print(f"\n检查NaN值...")
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"⚠️  发现 {nan_count} 个NaN值，使用0填充")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print(f"✓ 无NaN值")
    
    # 检查inf值
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        print(f"⚠️  发现 {inf_count} 个Inf值，使用0填充")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print(f"✓ 无Inf值")
    
    print(f"特征提取完成: X.shape={X.shape}, y.shape={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    print(f"患者数量: {len(np.unique(record_ids))}")
    print(f"患者列表: {np.unique(record_ids)}")
    
    return X, y, record_ids, df.columns.tolist()


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    训练和评估多个模型（添加详细评估指标）
    
    Args:
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集标签
        feature_names: 特征名称列表
    
    Returns:
        models: 训练好的模型字典
        results: 评估结果字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    
    print("\n" + "="*60)
    print("训练和评估模型")
    print("="*60)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced'),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False, auto_class_weights='Balanced'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {name}")
        print(f"{'='*60}")
        
        # 训练（XGBoost需要手动传sample_weight）
        start_time = time.time()
        if name == 'XGBoost':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # 计算详细指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 计算每个类别的F1-Score
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 打印结果
        print(f"训练时间: {train_time:.2f}秒")
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
        print(classification_report(y_test, y_pred, 
                                   target_names=['正常', '室性早搏', '其他异常'],
                                   zero_division=0))
        
        # 保存结果
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'train_time': train_time
        }
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("模型性能汇总")
    print("="*60)
    print(f"{'模型':<15} {'准确率':<10} {'F1-Score':<10} {'训练时间':<10}")
    print("-"*60)
    for name, result in results.items():
        print(f"{name:<15} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['train_time']:<10.2f}s")
    
    return models, results


def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    results = {}
    
    # 1. Random Forest
    print("\n" + "="*60)
    print("训练 Random Forest")
    print("="*60)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    models['RandomForest'] = rf
    results['RandomForest'] = acc
    
    # 2. XGBoost
    print("\n" + "="*60)
    print("训练 XGBoost")
    print("="*60)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    models['XGBoost'] = xgb_model
    results['XGBoost'] = acc
    
    # 3. LightGBM
    print("\n" + "="*60)
    print("训练 LightGBM")
    print("="*60)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    models['LightGBM'] = lgb_model
    results['LightGBM'] = acc
    
    # 4. CatBoost
    if CATBOOST_AVAILABLE:
        print("\n" + "="*60)
        print("训练 CatBoost")
        print("="*60)
        cat_model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"准确率: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        models['CatBoost'] = cat_model
        results['CatBoost'] = acc
    
    # 5. SVM
    print("\n" + "="*60)
    print("训练 SVM")
    print("="*60)
    
    # SVM不支持NaN，需要先填充缺失值
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train_imputed, y_train)
    y_pred = svm_model.predict(X_test_imputed)
    acc = accuracy_score(y_test, y_pred)
    print(f"准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    models['SVM'] = svm_model
    results['SVM'] = acc
    
    return models, results


def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据并提取特征（返回record_ids用于按患者划分）
    X, y, record_ids, feature_names = load_and_extract_features(num_samples=60000)
    
    # ========== 按患者划分数据集（Patient-wise Split）==========
    print("\n" + "="*60)
    print("按患者划分数据集（避免数据泄露）")
    print("="*60)
    
    # 获取唯一的患者ID
    unique_patients = np.unique(record_ids)
    print(f"总患者数: {len(unique_patients)}")
    print(f"患者列表: {unique_patients}")
    
    # 按患者划分（80% 训练，20% 测试）
    from sklearn.model_selection import train_test_split
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集患者 ({len(train_patients)}个): {train_patients}")
    print(f"测试集患者 ({len(test_patients)}个): {test_patients}")
    
    # 根据患者ID划分数据
    train_mask = np.isin(record_ids, train_patients)
    test_mask = np.isin(record_ids, test_patients)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\n数据集划分:")
    print(f"训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {X_test.shape}, 类别分布: {np.bincount(y_test)}")
    
    # 检查是否有数据泄露
    train_patients_set = set(train_patients)
    test_patients_set = set(test_patients)
    overlap = train_patients_set & test_patients_set
    if len(overlap) > 0:
        print(f"⚠️  警告: 发现数据泄露！重叠患者: {overlap}")
    else:
        print(f"✅ 数据划分正确，无数据泄露")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    models, results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # 保存模型
    print("\n" + "="*60)
    print("保存模型")
    print("="*60)
    
    os.makedirs('app/algorithms/models', exist_ok=True)
    
    joblib.dump(scaler, 'app/algorithms/models/scaler.pkl')
    print("✓ 保存 scaler.pkl")
    
    for name, model in models.items():
        filename = f'app/algorithms/models/{name.lower()}_model.pkl'
        joblib.dump(model, filename)
        print(f"✓ 保存 {filename}")
    
    # 保存评估结果
    import json
    with open('experiments/results/ml_results_patient_wise.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存评估结果到 experiments/results/ml_results_patient_wise.json")
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)
    print(f"⚠️  注意: 使用按患者划分后，准确率可能会下降，但这是更真实的性能评估")
    print(f"📊 详细评估指标已保存，包括F1-Score、混淆矩阵等")
    
    # 保存结果
    with open('experiments/results/ml_results.txt', 'w') as f:
        f.write("传统机器学习模型结果\n")
        f.write("="*60 + "\n")
        for name, acc in results.items():
            f.write(f"{name:15s}: {acc:.4f}\n")
    
    print(f"\n✅ 所有模型训练完成！")
    print(f"模型保存在: app/algorithms/models/")
    print(f"结果保存在: experiments/results/")
    
    # 打印最终结果
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:15s}: {acc:.4f}")


if __name__ == '__main__':
    main()
