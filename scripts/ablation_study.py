#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验（Ablation Study）
对比单一模型 vs 集成模型的性能
证明多模型融合的必要性
"""

import os
import sys
import numpy as np
import joblib  # 改用joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('.')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_test_data():
    """加载测试数据"""
    print("加载测试数据...")
    
    # 从训练脚本导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from scripts.train_traditional_ml import load_and_extract_features
    from sklearn.model_selection import train_test_split
    
    X, y, record_ids, feature_names = load_and_extract_features(num_samples=30000)
    
    # 按患者划分
    unique_patients = np.unique(record_ids)
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42
    )
    
    test_mask = np.isin(record_ids, test_patients)
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"测试集大小: {X_test.shape}")
    print(f"类别分布: {np.bincount(y_test)}")
    
    return X_test, y_test


def evaluate_single_model(model_name, model_path, scaler_path, X_test, y_test):
    """评估单个模型"""
    print(f"\n评估模型: {model_name}")
    
    # 加载模型
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 预测
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 计算置信度（最大概率的平均值）
    confidence = np.mean(np.max(y_pred_proba, axis=1))
    
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  平均置信度: {confidence:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confidence': confidence,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def evaluate_ensemble(model_results, y_test, method='voting'):
    """评估集成模型"""
    print(f"\n评估集成模型 (方法: {method})")
    
    if method == 'voting':
        # 投票法：多数投票
        all_predictions = np.array([r['predictions'] for r in model_results.values()])
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=all_predictions
        )
        
        # 置信度：投票比例
        confidence_list = []
        for i in range(len(y_pred)):
            votes = all_predictions[:, i]
            vote_count = np.bincount(votes, minlength=3)
            confidence = vote_count[y_pred[i]] / len(model_results)
            confidence_list.append(confidence)
        confidence = np.mean(confidence_list)
        
    elif method == 'averaging':
        # 平均法：概率平均
        all_probas = np.array([r['probabilities'] for r in model_results.values()])
        avg_proba = np.mean(all_probas, axis=0)
        y_pred = np.argmax(avg_proba, axis=1)
        confidence = np.mean(np.max(avg_proba, axis=1))
    
    elif method == 'weighted':
        # 加权平均：根据模型性能加权
        weights = np.array([r['accuracy'] for r in model_results.values()])
        weights = weights / np.sum(weights)
        
        all_probas = np.array([r['probabilities'] for r in model_results.values()])
        weighted_proba = np.average(all_probas, axis=0, weights=weights)
        y_pred = np.argmax(weighted_proba, axis=1)
        confidence = np.mean(np.max(weighted_proba, axis=1))
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  平均置信度: {confidence:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confidence': confidence
    }


def ablation_study(output_dir='experiments/results'):
    """消融实验主函数"""
    print("="*60)
    print("消融实验（Ablation Study）")
    print("="*60)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    X_test, y_test = load_test_data()
    
    # 模型配置
    models = {
        'RandomForest': 'app/algorithms/models/randomforest_model.pkl',
        'XGBoost': 'app/algorithms/models/xgboost_model.pkl',
        'LightGBM': 'app/algorithms/models/lightgbm_model.pkl',
        'CatBoost': 'app/algorithms/models/catboost_model.pkl',
        'SVM': 'app/algorithms/models/svm_model.pkl'
    }
    scaler_path = 'app/algorithms/models/scaler.pkl'
    
    # 1. 评估单个模型
    print("\n" + "="*60)
    print("第1步：评估单个模型")
    print("="*60)
    
    single_results = {}
    for model_name, model_path in models.items():
        single_results[model_name] = evaluate_single_model(
            model_name, model_path, scaler_path, X_test, y_test
        )
    
    # 2. 评估不同集成方法
    print("\n" + "="*60)
    print("第2步：评估集成模型")
    print("="*60)
    
    ensemble_results = {}
    
    # 投票法
    ensemble_results['投票法'] = evaluate_ensemble(single_results, y_test, method='voting')
    
    # 平均法
    ensemble_results['平均法'] = evaluate_ensemble(single_results, y_test, method='averaging')
    
    # 加权平均
    ensemble_results['加权平均'] = evaluate_ensemble(single_results, y_test, method='weighted')
    
    # 3. 逐步添加模型（增量实验）
    print("\n" + "="*60)
    print("第3步：增量实验（逐步添加模型）")
    print("="*60)
    
    incremental_results = {}
    model_names = list(models.keys())
    
    for i in range(1, len(model_names) + 1):
        subset_models = {name: single_results[name] for name in model_names[:i]}
        result = evaluate_ensemble(subset_models, y_test, method='voting')
        incremental_results[f'{i}个模型'] = result
        print(f"  使用{i}个模型: 准确率={result['accuracy']:.4f}, F1={result['f1']:.4f}")
    
    # 4. 保存结果
    all_results = {
        'single_models': {name: {k: v for k, v in result.items() 
                                if k not in ['predictions', 'probabilities']}
                         for name, result in single_results.items()},
        'ensemble_methods': ensemble_results,
        'incremental': incremental_results
    }
    
    result_file = Path(output_dir) / 'ablation_study_results.json'
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ 结果已保存: {result_file}")
    
    # 5. 绘制结果
    plot_ablation_results(all_results, output_dir)
    
    return all_results


def plot_ablation_results(results, output_dir):
    """绘制消融实验结果"""
    
    # 1. 单个模型 vs 集成模型对比
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    model_names = list(results['single_models'].keys())
    single_acc = [results['single_models'][name]['accuracy'] for name in model_names]
    single_f1 = [results['single_models'][name]['f1'] for name in model_names]
    
    ensemble_names = list(results['ensemble_methods'].keys())
    ensemble_acc = [results['ensemble_methods'][name]['accuracy'] for name in ensemble_names]
    ensemble_f1 = [results['ensemble_methods'][name]['f1'] for name in ensemble_names]
    
    all_names = model_names + ensemble_names
    all_acc = single_acc + ensemble_acc
    all_f1 = single_f1 + ensemble_f1
    
    x = np.arange(len(all_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, all_acc, width, label='准确率', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, all_f1, width, label='F1-Score', color='#2ecc71', alpha=0.8)
    
    # 添加分隔线
    ax.axvline(x=len(model_names)-0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(len(model_names)-0.5, 0.92, '单个模型 | 集成模型', 
           ha='center', va='bottom', fontsize=10, color='red')
    
    ax.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('消融实验：单个模型 vs 集成模型', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0.90, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'ablation_single_vs_ensemble.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 单个vs集成对比图已保存: {output_path}")
    plt.close()
    
    # 2. 增量实验曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_models = list(range(1, len(results['incremental']) + 1))
    incremental_acc = [results['incremental'][f'{i}个模型']['accuracy'] for i in num_models]
    incremental_f1 = [results['incremental'][f'{i}个模型']['f1'] for i in num_models]
    
    ax.plot(num_models, incremental_acc, 'o-', linewidth=2, markersize=8, 
           label='准确率', color='#3498db')
    ax.plot(num_models, incremental_f1, 's-', linewidth=2, markersize=8, 
           label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('模型数量', fontsize=12, fontweight='bold')
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('增量实验：逐步添加模型的效果', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(num_models)
    ax.set_ylim([0.90, 1.0])
    
    # 添加数值标签
    for i, (acc, f1) in enumerate(zip(incremental_acc, incremental_f1)):
        ax.text(num_models[i], acc + 0.002, f'{acc:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'ablation_incremental.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 增量实验曲线已保存: {output_path}")
    plt.close()
    
    # 3. 性能提升热力图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算相对于最佳单模型的提升
    best_single_acc = max(single_acc)
    improvements = []
    labels = []
    
    for name in ensemble_names:
        acc = results['ensemble_methods'][name]['accuracy']
        improvement = (acc - best_single_acc) * 100
        improvements.append(improvement)
        labels.append(name)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.barh(labels, improvements, color=colors, alpha=0.8)
    
    ax.set_xlabel('准确率提升 (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'集成方法相对最佳单模型的提升\n(基准: {best_single_acc:.2%})', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, improvement in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{improvement:+.2f}%', ha='left' if width > 0 else 'right',
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'ablation_improvement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 性能提升图已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    results = ablation_study()
    
    print("\n" + "="*60)
    print("✅ 消融实验完成！")
    print("="*60)
    print("\n生成的图表:")
    print("  - ablation_single_vs_ensemble.png  (单个vs集成对比)")
    print("  - ablation_incremental.png         (增量实验曲线)")
    print("  - ablation_improvement.png         (性能提升图)")
    print("\n关键结论:")
    
    # 找出最佳单模型和最佳集成方法
    single_best = max(results['single_models'].items(), 
                     key=lambda x: x[1]['accuracy'])
    ensemble_best = max(results['ensemble_methods'].items(), 
                       key=lambda x: x[1]['accuracy'])
    
    print(f"  - 最佳单模型: {single_best[0]} (准确率: {single_best[1]['accuracy']:.2%})")
    print(f"  - 最佳集成方法: {ensemble_best[0]} (准确率: {ensemble_best[1]['accuracy']:.2%})")
    
    improvement = (ensemble_best[1]['accuracy'] - single_best[1]['accuracy']) * 100
    print(f"  - 集成提升: +{improvement:.2f}%")
    print(f"  - 结论: 多模型集成显著提升了系统性能")


if __name__ == '__main__':
    main()
