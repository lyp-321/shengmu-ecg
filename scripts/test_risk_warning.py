#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试误诊风险预警机制
"""

import sys
sys.path.append('.')

from app.core.risk_warning import RiskWarningSystem, WarningLevel, evaluate_risk


def test_case_1():
    """测试案例1：高置信度，无冲突 - 安全"""
    print("\n" + "="*60)
    print("测试案例1：高置信度，无冲突")
    print("="*60)
    
    ml_predictions = {
        'RandomForest': 0.95,
        'XGBoost': 0.93,
        'LightGBM': 0.94,
        'CatBoost': 0.96,
        'SVM': 0.95
    }
    
    warning = evaluate_risk(
        final_confidence=0.95,
        ml_predictions=ml_predictions
    )
    
    print(f"预警级别: {warning.level.value}")
    print(f"置信度: {warning.confidence:.1%}")
    print(f"需要人工复核: {'是' if warning.need_manual_review else '否'}")
    print(f"预警原因: {', '.join(warning.reasons)}")
    print(f"建议:")
    for suggestion in warning.suggestions:
        print(f"  - {suggestion}")


def test_case_2():
    """测试案例2：低置信度 - 中风险"""
    print("\n" + "="*60)
    print("测试案例2：低置信度")
    print("="*60)
    
    ml_predictions = {
        'RandomForest': 0.65,
        'XGBoost': 0.68,
        'LightGBM': 0.62,
        'CatBoost': 0.67,
        'SVM': 0.64
    }
    
    warning = evaluate_risk(
        final_confidence=0.65,
        ml_predictions=ml_predictions
    )
    
    print(f"预警级别: {warning.level.value}")
    print(f"置信度: {warning.confidence:.1%}")
    print(f"需要人工复核: {'是' if warning.need_manual_review else '否'}")
    print(f"预警原因: {', '.join(warning.reasons)}")
    print(f"建议:")
    for suggestion in warning.suggestions:
        print(f"  - {suggestion}")


def test_case_3():
    """测试案例3：模型预测冲突 - 高风险"""
    print("\n" + "="*60)
    print("测试案例3：ML和DL模型预测冲突")
    print("="*60)
    
    ml_predictions = {
        'RandomForest': 0.85,
        'XGBoost': 0.82,
        'LightGBM': 0.88,
        'CatBoost': 0.84,
        'SVM': 0.86
    }
    
    dl_predictions = {
        'ResNet1D': 0.45,
        'SEResNet1D': 0.42,
        'BiLSTM': 0.48,
        'TCN': 0.40,
        'Inception': 0.46
    }
    
    warning = evaluate_risk(
        final_confidence=0.70,
        ml_predictions=ml_predictions,
        dl_predictions=dl_predictions
    )
    
    print(f"预警级别: {warning.level.value}")
    print(f"置信度: {warning.confidence:.1%}")
    print(f"需要人工复核: {'是' if warning.need_manual_review else '否'}")
    print(f"预警原因:")
    for reason in warning.reasons:
        print(f"  - {reason}")
    print(f"建议:")
    for suggestion in warning.suggestions:
        print(f"  - {suggestion}")


def test_case_4():
    """测试案例4：ML模型内部冲突 - 中风险"""
    print("\n" + "="*60)
    print("测试案例4：ML模型内部预测差异大")
    print("="*60)
    
    ml_predictions = {
        'RandomForest': 0.95,
        'XGBoost': 0.55,  # 差异大
        'LightGBM': 0.92,
        'CatBoost': 0.90,
        'SVM': 0.88
    }
    
    warning = evaluate_risk(
        final_confidence=0.84,
        ml_predictions=ml_predictions
    )
    
    print(f"预警级别: {warning.level.value}")
    print(f"置信度: {warning.confidence:.1%}")
    print(f"需要人工复核: {'是' if warning.need_manual_review else '否'}")
    print(f"预警原因:")
    for reason in warning.reasons:
        print(f"  - {reason}")
    print(f"建议:")
    for suggestion in warning.suggestions:
        print(f"  - {suggestion}")


def test_case_5():
    """测试案例5：预测少数类 + 信号质量差 - 高风险"""
    print("\n" + "="*60)
    print("测试案例5：预测少数类 + 信号质量差")
    print("="*60)
    
    ml_predictions = {
        'RandomForest': 0.75,
        'XGBoost': 0.72,
        'LightGBM': 0.78,
        'CatBoost': 0.74,
        'SVM': 0.76
    }
    
    # 类别分布（类别1只占1%）
    class_distribution = {
        0: 23000,  # 正常
        1: 300,    # 室性早搏（少数类）
        2: 700     # 其他异常
    }
    
    warning = evaluate_risk(
        final_confidence=0.75,
        ml_predictions=ml_predictions,
        predicted_class=1,  # 预测为少数类
        class_distribution=class_distribution,
        signal_quality_score=0.45  # 信号质量差
    )
    
    print(f"预警级别: {warning.level.value}")
    print(f"置信度: {warning.confidence:.1%}")
    print(f"需要人工复核: {'是' if warning.need_manual_review else '否'}")
    print(f"预警原因:")
    for reason in warning.reasons:
        print(f"  - {reason}")
    print(f"建议:")
    for suggestion in warning.suggestions:
        print(f"  - {suggestion}")


def test_case_6():
    """测试案例6：严重低置信度 - 严重风险"""
    print("\n" + "="*60)
    print("测试案例6：严重低置信度")
    print("="*60)
    
    ml_predictions = {
        'RandomForest': 0.42,
        'XGBoost': 0.38,
        'LightGBM': 0.45,
        'CatBoost': 0.40,
        'SVM': 0.43
    }
    
    warning = evaluate_risk(
        final_confidence=0.42,
        ml_predictions=ml_predictions
    )
    
    print(f"预警级别: {warning.level.value}")
    print(f"置信度: {warning.confidence:.1%}")
    print(f"需要人工复核: {'是' if warning.need_manual_review else '否'}")
    print(f"预警原因:")
    for reason in warning.reasons:
        print(f"  - {reason}")
    print(f"建议:")
    for suggestion in warning.suggestions:
        print(f"  - {suggestion}")


def main():
    """主函数"""
    print("="*60)
    print("误诊风险预警机制测试")
    print("="*60)
    
    # 运行所有测试案例
    test_case_1()  # 安全
    test_case_2()  # 中风险
    test_case_3()  # 高风险（ML-DL冲突）
    test_case_4()  # 中风险（ML内部冲突）
    test_case_5()  # 高风险（少数类+信号质量差）
    test_case_6()  # 严重风险（严重低置信度）
    
    print("\n" + "="*60)
    print("✅ 所有测试案例完成！")
    print("="*60)
    print("\n预警机制说明：")
    print("  - 安全：无需预警，可直接使用")
    print("  - 低风险：建议关注")
    print("  - 中风险：需要人工复核")
    print("  - 高风险：强烈建议人工复核")
    print("  - 严重风险：必须人工复核")


if __name__ == '__main__':
    main()
