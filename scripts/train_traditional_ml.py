"""
训练传统机器学习模型
XGBoost, LightGBM, CatBoost, RandomForest + Stacking集成
每次运行自动将结果追加到 experiments/results/hyperparam_search.csv
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import wfdb
from tqdm import tqdm
import pywt
from scipy.signal import find_peaks

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost未安装，将跳过CatBoost训练")

# ============================================================
# 实验配置（修改这里来做不同实验）
# ============================================================
CONFIG = {
    'num_samples': 60000,
    'smote_ratio': 1.0,
    'class_weight': {0: 1, 1: 5, 2: 20},
    'n_estimators': 300,
    'use_scale_pos_weight': False,
    'note': '弃用RF+Stacking(XGB+CatBoost)为主引擎+SMOTE1.0+class2权重20',
}

CSV_PATH = 'experiments/results/hyperparam_search.csv'


def extract_time_domain_features(ecg_signal, fs=360):
    """提取时域特征"""
    features = {}

    features['mean'] = np.mean(ecg_signal)
    features['std'] = np.std(ecg_signal)
    features['max'] = np.max(ecg_signal)
    features['min'] = np.min(ecg_signal)
    features['median'] = np.median(ecg_signal)
    features['range'] = features['max'] - features['min']
    features['skewness'] = float(pd.Series(ecg_signal).skew())
    features['kurtosis'] = float(pd.Series(ecg_signal).kurtosis())

    peaks, _ = find_peaks(ecg_signal, distance=int(0.6 * fs), prominence=0.3)

    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs * 1000
        features['hr_mean'] = 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        features['hr_std'] = np.std(60000 / rr_intervals) if len(rr_intervals) > 0 else 0
        features['sdnn'] = np.std(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else 0
        features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100 if len(rr_intervals) > 1 else 0
        features['rr_irregularity'] = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
    else:
        features['hr_mean'] = 0
        features['hr_std'] = 0
        features['sdnn'] = 0
        features['rmssd'] = 0
        features['pnn50'] = 0
        features['rr_irregularity'] = 0

    features['r_amplitude'] = np.max(ecg_signal) - np.min(ecg_signal)
    features['r_peak_count'] = len(peaks)

    return features


def extract_frequency_domain_features(ecg_signal, fs=360):
    """提取频域特征（针对单心拍信号优化）"""
    features = {}

    fft_vals = np.fft.fft(ecg_signal)
    fft_freq = np.fft.fftfreq(len(ecg_signal), 1 / fs)
    power = np.abs(fft_vals) ** 2

    # 针对单心拍的有效频带（0.5-40Hz）
    p_t_band = (fft_freq >= 0.5) & (fft_freq < 5)
    qrs_band = (fft_freq >= 5) & (fft_freq < 20)
    hf_band = (fft_freq >= 20) & (fft_freq < 40)
    total_power = np.sum(power[(fft_freq >= 0.5) & (fft_freq < 40)]) + 1e-8

    features['pt_power'] = np.sum(power[p_t_band])
    features['qrs_power'] = np.sum(power[qrs_band])
    features['hf_power'] = np.sum(power[hf_band])
    features['qrs_pt_ratio'] = features['qrs_power'] / (features['pt_power'] + 1e-8)
    features['qrs_power_ratio'] = features['qrs_power'] / total_power

    pos_mask = fft_freq > 0
    if pos_mask.sum() > 0:
        dominant_idx = np.argmax(power[pos_mask])
        features['dominant_freq'] = fft_freq[pos_mask][dominant_idx]
        features['dominant_power'] = power[pos_mask][dominant_idx]
    else:
        features['dominant_freq'] = 0
        features['dominant_power'] = 0

    # 小波变换特征（db4，5层分解）
    coeffs = pywt.wavedec(ecg_signal, 'db4', level=5)
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_energy_level_{i}'] = np.sum(coeff ** 2)
        features[f'wavelet_std_level_{i}'] = np.std(coeff)
        features[f'wavelet_max_level_{i}'] = np.max(np.abs(coeff))

    return features


def load_and_extract_features(data_dir='data', num_samples=60000):
    """加载MIT-BIH数据并提取特征（按患者划分，避免数据泄露）"""
    print("加载MIT-BIH数据并提取特征...")
    print("⚠️  使用按患者划分（Patient-wise）策略，避免数据泄露")

    # 实际存在的MIT-BIH标准记录（27个患者）
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '209'
    ]

    label_map = {
        'N': 0, 'L': 0, 'R': 0,           # 正常
        'V': 1, '/': 1,                     # 室性早搏
        'A': 2, 'F': 2, 'f': 2, 'j': 2,   # 其他异常（原有）
        'E': 2, 'e': 2, 'S': 2, 'a': 2,   # 扩充：室性逸搏、房性逸搏、室上性早搏、房性早搏变体
        'J': 2, 'n': 2, 'Q': 2,            # 扩充：交界性早搏、结性逸搏、未分类异常
    }

    features_list = []
    labels_list = []
    record_ids_list = []

    for record in tqdm(records, desc="提取特征"):
        try:
            record_path = os.path.join(data_dir, record)
            if not os.path.exists(f"{record_path}.dat"):
                continue

            signal_data, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            fs = fields['fs']

            for sample, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in label_map:
                    continue

                start = max(0, sample - 500)
                end = min(len(signal_data), sample + 500)
                segment = signal_data[start:end, 0]

                if len(segment) < 100:
                    continue

                time_features = extract_time_domain_features(segment, fs)
                freq_features = extract_frequency_domain_features(segment, fs)
                features_list.append({**time_features, **freq_features})
                labels_list.append(label_map[symbol])
                record_ids_list.append(record)

        except Exception as e:
            print(f"处理记录 {record} 失败: {e}")
            continue

    # 读完所有患者后再随机采样
    if len(features_list) > num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(features_list), num_samples, replace=False)
        features_list = [features_list[i] for i in indices]
        labels_list = [labels_list[i] for i in indices]
        record_ids_list = [record_ids_list[i] for i in indices]

    df = pd.DataFrame(features_list)
    X = df.values
    y = np.array(labels_list)
    record_ids = np.array(record_ids_list)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"特征提取完成: X.shape={X.shape}, y.shape={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    print(f"患者数量: {len(np.unique(record_ids))}")

    return X, y, record_ids, df.columns.tolist()


def save_results_to_csv(results, config, csv_path=CSV_PATH):
    """将本次实验结果追加到CSV（自动化实验记录）"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cw_str = str(config['class_weight']).replace(' ', '')

    for name, r in results.items():
        f1_per = r.get('f1_per_class', [0, 0, 0])
        macro_f1 = np.mean(f1_per) if len(f1_per) == 3 else r.get('f1_score', 0)
        rows.append({
            'timestamp': ts,
            'model': name,
            'class_weight': cw_str,
            'smote_ratio': config.get('smote_ratio', ''),
            'n_estimators': config.get('n_estimators', ''),
            'accuracy': round(r.get('accuracy', 0), 4),
            'macro_f1': round(macro_f1, 4),
            'f1_normal': round(f1_per[0] if len(f1_per) > 0 else 0, 4),
            'f1_pvc': round(f1_per[1] if len(f1_per) > 1 else 0, 4),
            'f1_other': round(f1_per[2] if len(f1_per) > 2 else 0, 4),
            'train_time': round(r.get('train_time', 0), 2),
            'note': config.get('note', ''),
        })

    new_df = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)
    print(f"\n✅ 实验结果已追加到 {csv_path}（共 {len(combined)} 条记录）")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, config):
    """训练和评估所有模型（RF已弃用：SMOTE下过拟合严重，召回虚高但精确率极低）"""
    cw = config['class_weight']
    n_est = config['n_estimators']
    use_spw = config.get('use_scale_pos_weight', False)

    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=n_est, random_state=42, n_jobs=-1,
            eval_metric='mlogloss', learning_rate=0.1,
            scale_pos_weight=15 if use_spw else 1,
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=n_est, random_state=42, n_jobs=-1,
            verbose=-1, class_weight=cw
        ),
    }
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(
            iterations=n_est, random_state=42, verbose=False,
            auto_class_weights='Balanced'
        )

    results = {}

    print("\n" + "=" * 60)
    print("训练和评估模型")
    print("=" * 60)

    trained = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"训练模型: {name}")
        print(f"{'=' * 60}")

        start = time.time()
        if name == 'XGBoost':
            from sklearn.utils.class_weight import compute_sample_weight
            cw_arr = np.array([cw[int(yi)] for yi in y_train])
            model.fit(X_train, y_train, sample_weight=cw_arr)
        else:
            model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)
        # CatBoost的predict可能返回2D或对象数组，统一转int一维
        y_pred = np.array(y_pred, dtype=int).ravel()

        # 置信度分析：预测概率 < 0.6 的"纠结样本"（风险预警核心数据）
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            max_proba = proba.max(axis=1)
            uncertain_mask = max_proba < 0.6
            uncertain_count = int(uncertain_mask.sum())
            if uncertain_count > 0:
                u_pred = y_pred[uncertain_mask]
                u_true = np.array(y_test, dtype=int)[uncertain_mask]
                uncertain_error = int((u_pred != u_true).sum())
            else:
                uncertain_error = 0
        else:
            uncertain_count, uncertain_error = 0, 0
        acc = accuracy_score(y_test, y_pred)
        f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)
        macro_f1 = np.mean(f1_per)

        print(f"训练时间: {elapsed:.2f}秒")
        print(f"准确率: {acc:.4f}  宏F1: {macro_f1:.4f}")
        print(f"  正常F1={f1_per[0]:.4f}  室早F1={f1_per[1]:.4f}  其他异常F1={f1_per[2] if len(f1_per)>2 else 0:.4f}")
        if uncertain_count > 0:
            print(f"  ⚠️  低置信度样本（prob<0.6）: {uncertain_count}个，其中误诊{uncertain_error}个（{uncertain_error/uncertain_count*100:.1f}%）")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred,
                                    target_names=['正常', '室性早搏', '其他异常'],
                                    zero_division=0))

        results[name] = {
            'accuracy': float(acc),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_per_class': f1_per.tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'train_time': float(elapsed),
            'uncertain_count': int(uncertain_count),
            'uncertain_error_rate': round(uncertain_error / uncertain_count, 4) if uncertain_count > 0 else 0,
        }
        trained[name] = model

    # ===== Stacking：XGBoost粗筛 + CatBoost复核 =====
    if 'XGBoost' in trained and 'CatBoost' in trained:
        print(f"\n{'=' * 60}")
        print("Stacking集成：XGBoost粗筛 + CatBoost复核")
        print("=" * 60)

        xgb_pred = trained['XGBoost'].predict(X_test)
        stacking_pred = xgb_pred.copy()

        # XGBoost认为是异常的样本，交给CatBoost复核
        abnormal_mask = xgb_pred != 0
        if abnormal_mask.sum() > 0:
            cat_pred = trained['CatBoost'].predict(X_test[abnormal_mask]).ravel()
            stacking_pred[abnormal_mask] = cat_pred

        acc = accuracy_score(y_test, stacking_pred)
        f1_per = f1_score(y_test, stacking_pred, average=None, zero_division=0)
        macro_f1 = np.mean(f1_per)

        print(f"准确率: {acc:.4f}  宏F1: {macro_f1:.4f}")
        print(f"  正常F1={f1_per[0]:.4f}  室早F1={f1_per[1]:.4f}  其他异常F1={f1_per[2] if len(f1_per)>2 else 0:.4f}")
        print(confusion_matrix(y_test, stacking_pred))
        print(classification_report(y_test, stacking_pred,
                                    target_names=['正常', '室性早搏', '其他异常'],
                                    zero_division=0))

        results['Stacking(XGB+CatBoost)'] = {
            'accuracy': float(acc),
            'f1_score': float(f1_score(y_test, stacking_pred, average='weighted', zero_division=0)),
            'f1_per_class': f1_per.tolist(),
            'confusion_matrix': confusion_matrix(y_test, stacking_pred).tolist(),
            'train_time': 0.0,
        }

    # 汇总
    print("\n" + "=" * 60)
    print("模型性能汇总")
    print("=" * 60)
    print(f"{'模型':<25} {'准确率':<10} {'宏F1':<10} {'室早F1':<10} {'其他异常F1':<12} {'训练时间'}")
    print("-" * 80)
    for name, r in results.items():
        fp = r['f1_per_class']
        print(f"{name:<25} {r['accuracy']:<10.4f} {np.mean(fp):<10.4f} "
              f"{fp[1] if len(fp)>1 else 0:<10.4f} {fp[2] if len(fp)>2 else 0:<12.4f} {r['train_time']:.2f}s")

    return trained, results


def main():
    np.random.seed(42)

    # 1. 加载数据
    X, y, record_ids, feature_names = load_and_extract_features(
        num_samples=CONFIG['num_samples']
    )

    # 2. 按患者划分（Patient-wise Split）
    print("\n" + "=" * 60)
    print("按患者划分数据集（避免数据泄露）")
    print("=" * 60)

    unique_patients = np.unique(record_ids)
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    print(f"训练集患者 ({len(train_patients)}个): {train_patients}")
    print(f"测试集患者 ({len(test_patients)}个): {test_patients}")

    train_mask = np.isin(record_ids, train_patients)
    test_mask = np.isin(record_ids, test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {X_test.shape}, 类别分布: {np.bincount(y_test)}")

    overlap = set(train_patients) & set(test_patients)
    print(f"{'✅ 无数据泄露' if not overlap else f'⚠️ 数据泄露: {overlap}'}")

    # 3. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. SMOTE过采样（只对训练集）
    print(f"\n应用SMOTE过采样（ratio={CONFIG['smote_ratio']}）...")
    print(f"过采样前训练集类别分布: {np.bincount(y_train)}")
    try:
        from imblearn.over_sampling import SMOTE
        counts = np.bincount(y_train)
        majority = counts[0]
        # sampling_strategy: 指定少数类目标数量
        target = int(majority * CONFIG['smote_ratio'])
        sampling_strategy = {
            1: max(counts[1], target),
            2: max(counts[2], target),
        }
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"过采样后训练集类别分布: {np.bincount(y_train)}")
    except ImportError:
        print("⚠️  未安装imbalanced-learn，跳过SMOTE")

    # 5. 训练模型
    models, results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, CONFIG
    )

    # 6. 自动追加实验结果到CSV
    save_results_to_csv(results, CONFIG)

    # 7. 保存模型
    print("\n保存模型...")
    os.makedirs('app/algorithms/models', exist_ok=True)
    joblib.dump(scaler, 'app/algorithms/models/scaler.pkl')
    print("✓ scaler.pkl")

    model_name_map = {
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'CatBoost': 'catboost',
    }
    for name, model in models.items():
        key = model_name_map.get(name, name.lower())
        path = f'app/algorithms/models/{key}_model.pkl'
        joblib.dump(model, path)
        print(f"✓ {path}")

    # 8. 保存JSON结果
    import json
    with open('experiments/results/ml_results_patient_wise.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 9. 导出特征重要性图
    print("\n导出特征重要性图...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        importance_models = ['XGBoost', 'LightGBM', 'CatBoost']
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        for ax, name in zip(axes, importance_models):
            if name not in models:
                ax.set_visible(False)
                continue
            model = models[name]
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[-15:]
                top_names = [feature_names[i] for i in top_idx]
                top_vals = imp[top_idx]
                colors = ['#e74c3c' if 'wavelet' in n else
                          '#3498db' if any(k in n for k in ['pt_power','qrs','hf','dominant','power_ratio']) else
                          '#2ecc71' for n in top_names]
                ax.barh(top_names, top_vals, color=colors)
                ax.set_title(f'{name} Top-15 Feature Importance', fontsize=12)
                ax.set_xlabel('Importance')
                from matplotlib.patches import Patch
                legend = [Patch(color='#2ecc71', label='Time-domain'),
                          Patch(color='#3498db', label='Frequency'),
                          Patch(color='#e74c3c', label='Wavelet')]
                ax.legend(handles=legend, loc='lower right', fontsize=9)

        plt.suptitle('Feature Importance (Green=Time / Blue=Freq / Red=Wavelet)', fontsize=14)
        plt.tight_layout()
        out_path = 'experiments/results/feature_importance.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ {out_path}")
    except Exception as e:
        print(f"⚠️  特征重要性图生成失败: {e}")

    print("\n✅ 训练完成！")
    print(f"模型: app/algorithms/models/")
    print(f"实验记录: {CSV_PATH}")
    print(f"特征重要性: experiments/results/feature_importance.png")


if __name__ == '__main__':
    main()
