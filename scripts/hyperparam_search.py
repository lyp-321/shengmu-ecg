"""
超参数搜索 + 自动实验记录
自动搜索 class_weight 和 SMOTE 参数组合，记录每次实验结果到 CSV
大创用途：展示通过网格搜索找到最优参数的实验过程
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import time
import itertools
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 复用训练脚本的数据加载函数
from scripts.train_traditional_ml import load_and_extract_features

# ========== 搜索空间配置 ==========
SEARCH_SPACE = {
    # class_weight: 正常:室早:其他异常 的代价比
    'class_weight': [
        {0: 1, 1: 3, 2: 10},
        {0: 1, 1: 5, 2: 15},
        {0: 1, 1: 5, 2: 20},
        {0: 1, 1: 8, 2: 25},
    ],
    # SMOTE目标比例（少数类相对多数类的比例）
    'smote_ratio': [0.5, 1.0],
    # XGBoost树数量
    'n_estimators': [200, 300],
}

RESULT_CSV = 'experiments/results/hyperparam_search.csv'
CSV_FIELDS = [
    'timestamp', 'model', 'class_weight', 'smote_ratio', 'n_estimators',
    'accuracy', 'macro_f1', 'f1_normal', 'f1_pvc', 'f1_other',
    'train_time', 'note'
]


def log_result(row: dict):
    """追加一行结果到CSV"""
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    file_exists = os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(X_train, X_test, y_train, y_test,
                   class_weight, smote_ratio, n_estimators):
    """运行单次实验，返回评估指标"""

    # SMOTE过采样
    X_tr, y_tr = X_train.copy(), y_train.copy()
    if smote_ratio is not None:
        try:
            from imblearn.over_sampling import SMOTE
            counts = np.bincount(y_tr)
            majority = counts.max()
            sampling = {
                cls: max(cnt, int(majority * smote_ratio))
                for cls, cnt in enumerate(counts)
            }
            smote = SMOTE(sampling_strategy=sampling, random_state=42, k_neighbors=5)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        except ImportError:
            print("⚠️  未安装imbalanced-learn，跳过SMOTE")

    # 训练XGBoost（梯度提升对频域特征最有效）
    if class_weight == 'balanced':
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight('balanced', y_tr)
    else:
        from sklearn.utils.class_weight import compute_sample_weight
        # 转换为sample_weight
        weight_map = class_weight
        sw = np.array([weight_map[c] for c in y_tr], dtype=float)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )

    t0 = time.time()
    model.fit(X_tr, y_tr, sample_weight=sw)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(macro_f1, 4),
        'f1_normal': round(f1_per[0], 4),
        'f1_pvc':    round(f1_per[1], 4) if len(f1_per) > 1 else 0,
        'f1_other':  round(f1_per[2], 4) if len(f1_per) > 2 else 0,
        'train_time': round(train_time, 2),
    }


def main():
    print("=" * 60)
    print("超参数搜索实验")
    print(f"结果将保存到: {RESULT_CSV}")
    print("=" * 60)

    # 加载数据（只加载一次）
    print("\n加载数据...")
    X, y, record_ids, _ = load_and_extract_features(num_samples=60000)

    # 按患者划分
    unique_patients = np.unique(record_ids)
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    train_mask = np.isin(record_ids, train_patients)
    test_mask  = np.isin(record_ids, test_patients)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"训练集: {X_train.shape}, 类别分布: {np.bincount(y_train)}")
    print(f"测试集: {X_test.shape},  类别分布: {np.bincount(y_test)}")

    # 生成所有参数组合
    keys = ['class_weight', 'smote_ratio', 'n_estimators']
    combos = list(itertools.product(
        SEARCH_SPACE['class_weight'],
        SEARCH_SPACE['smote_ratio'],
        SEARCH_SPACE['n_estimators'],
    ))
    total = len(combos)
    print(f"\n共 {total} 个参数组合，开始搜索...\n")

    best_macro_f1 = 0
    best_params = None

    for i, (cw, sr, ne) in enumerate(combos, 1):
        cw_str = str(cw) if isinstance(cw, dict) else cw
        print(f"[{i:2d}/{total}] class_weight={cw_str}, smote_ratio={sr}, n_estimators={ne}")

        try:
            metrics = run_experiment(X_train, X_test, y_train, y_test, cw, sr, ne)

            row = {
                'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model':        'XGBoost',
                'class_weight': cw_str,
                'smote_ratio':  sr,
                'n_estimators': ne,
                **metrics,
                'note': '★ BEST' if metrics['macro_f1'] > best_macro_f1 else ''
            }
            log_result(row)

            print(f"         → Acc={metrics['accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}"
                  f"  F1[正常={metrics['f1_normal']:.3f}, 室早={metrics['f1_pvc']:.3f}, 其他={metrics['f1_other']:.3f}]")

            if metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = metrics['macro_f1']
                best_params = row
                print(f"         ★ 新最优！MacroF1={best_macro_f1:.4f}")

        except Exception as e:
            print(f"         ✗ 失败: {e}")
            log_result({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'XGBoost', 'class_weight': cw_str,
                'smote_ratio': sr, 'n_estimators': ne,
                'accuracy': 0, 'macro_f1': 0, 'f1_normal': 0,
                'f1_pvc': 0, 'f1_other': 0, 'train_time': 0,
                'note': f'ERROR: {e}'
            })

    # 打印最优结果
    print("\n" + "=" * 60)
    print("搜索完成！最优参数：")
    print("=" * 60)
    if best_params:
        print(f"  class_weight  : {best_params['class_weight']}")
        print(f"  smote_ratio   : {best_params['smote_ratio']}")
        print(f"  n_estimators  : {best_params['n_estimators']}")
        print(f"  MacroF1       : {best_params['macro_f1']}")
        print(f"  室早F1        : {best_params['f1_pvc']}")
        print(f"  其他异常F1    : {best_params['f1_other']}")
    print(f"\n完整结果已保存到: {RESULT_CSV}")


if __name__ == '__main__':
    main()
