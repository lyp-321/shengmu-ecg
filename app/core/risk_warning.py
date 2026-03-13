#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
误诊风险预警机制
当模型预测结果存在冲突或置信度过低时，触发人工复核预警
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class WarningLevel(Enum):
    """预警级别"""
    SAFE = "安全"           # 无需预警
    LOW = "低风险"          # 轻微异常，建议关注
    MEDIUM = "中风险"       # 需要人工复核
    HIGH = "高风险"         # 强烈建议人工复核
    CRITICAL = "严重风险"   # 必须人工复核


@dataclass
class RiskWarning:
    """风险预警结果"""
    level: WarningLevel
    confidence: float
    reasons: List[str]
    suggestions: List[str]
    need_manual_review: bool


class RiskWarningSystem:
    """误诊风险预警系统"""
    
    def __init__(self,
                 confidence_threshold_low: float = 0.70,
                 confidence_threshold_critical: float = 0.50,
                 model_conflict_threshold: float = 0.30,
                 prediction_std_threshold: float = 0.25):
        """
        初始化预警系统
        
        Args:
            confidence_threshold_low: 低置信度阈值（<70%触发低风险预警）
            confidence_threshold_critical: 严重低置信度阈值（<50%触发严重预警）
            model_conflict_threshold: 模型冲突阈值（差异>30%触发预警）
            prediction_std_threshold: 预测标准差阈值（>0.25触发预警）
        """
        self.confidence_threshold_low = confidence_threshold_low
        self.confidence_threshold_critical = confidence_threshold_critical
        self.model_conflict_threshold = model_conflict_threshold
        self.prediction_std_threshold = prediction_std_threshold
    
    def check_confidence(self, confidence: float) -> Tuple[bool, str, WarningLevel]:
        """
        检查置信度是否过低
        
        Args:
            confidence: 最终置信度（0-1）
        
        Returns:
            (是否触发预警, 预警原因, 预警级别)
        """
        if confidence < self.confidence_threshold_critical:
            return (
                True,
                f"置信度过低 ({confidence:.1%} < {self.confidence_threshold_critical:.0%})",
                WarningLevel.CRITICAL
            )
        elif confidence < self.confidence_threshold_low:
            return (
                True,
                f"置信度较低 ({confidence:.1%} < {self.confidence_threshold_low:.0%})",
                WarningLevel.MEDIUM
            )
        return False, "", WarningLevel.SAFE
    
    def check_model_conflict(self,
                            ml_predictions: Dict[str, float],
                            dl_predictions: Optional[Dict[str, float]] = None) -> Tuple[bool, str, WarningLevel]:
        """
        检查模型间预测是否冲突
        
        Args:
            ml_predictions: 传统ML模型预测结果 {model_name: confidence}
            dl_predictions: 深度学习模型预测结果（可选）
        
        Returns:
            (是否触发预警, 预警原因, 预警级别)
        """
        reasons = []
        max_level = WarningLevel.SAFE
        
        # 检查ML模型间的冲突
        if ml_predictions:
            ml_confidences = list(ml_predictions.values())
            ml_std = np.std(ml_confidences)
            ml_range = max(ml_confidences) - min(ml_confidences)
            
            if ml_range > self.model_conflict_threshold:
                reasons.append(
                    f"传统ML模型预测差异大 (范围: {ml_range:.1%})"
                )
                max_level = WarningLevel.MEDIUM
            
            if ml_std > self.prediction_std_threshold:
                reasons.append(
                    f"传统ML模型预测不一致 (标准差: {ml_std:.3f})"
                )
                max_level = max(max_level, WarningLevel.LOW, key=lambda x: x.value)
        
        # 检查ML和DL模型间的冲突
        if ml_predictions and dl_predictions:
            ml_avg = np.mean(list(ml_predictions.values()))
            dl_avg = np.mean(list(dl_predictions.values()))
            ml_dl_diff = abs(ml_avg - dl_avg)
            
            if ml_dl_diff > self.model_conflict_threshold:
                reasons.append(
                    f"传统ML与深度学习模型预测冲突 (差异: {ml_dl_diff:.1%})"
                )
                max_level = WarningLevel.HIGH
        
        if reasons:
            return True, "; ".join(reasons), max_level
        return False, "", WarningLevel.SAFE
    
    def check_class_imbalance(self,
                             predicted_class: int,
                             class_distribution: Dict[int, int]) -> Tuple[bool, str, WarningLevel]:
        """
        检查预测类别是否为少数类（可能误诊风险更高）
        
        Args:
            predicted_class: 预测的类别
            class_distribution: 训练集类别分布 {class: count}
        
        Returns:
            (是否触发预警, 预警原因, 预警级别)
        """
        if not class_distribution:
            return False, "", WarningLevel.SAFE
        
        total_samples = sum(class_distribution.values())
        class_ratio = class_distribution.get(predicted_class, 0) / total_samples
        
        # 如果预测为少数类（<5%），触发预警
        if class_ratio < 0.05:
            return (
                True,
                f"预测为少数类（训练集占比仅{class_ratio:.1%}），误诊风险较高",
                WarningLevel.MEDIUM
            )
        
        return False, "", WarningLevel.SAFE
    
    def check_signal_quality(self, signal_quality_score: Optional[float] = None) -> Tuple[bool, str, WarningLevel]:
        """
        检查信号质量
        
        Args:
            signal_quality_score: 信号质量评分（0-1，可选）
        
        Returns:
            (是否触发预警, 预警原因, 预警级别)
        """
        if signal_quality_score is None:
            return False, "", WarningLevel.SAFE
        
        if signal_quality_score < 0.5:
            return (
                True,
                f"信号质量差 (评分: {signal_quality_score:.1%})",
                WarningLevel.HIGH
            )
        elif signal_quality_score < 0.7:
            return (
                True,
                f"信号质量一般 (评分: {signal_quality_score:.1%})",
                WarningLevel.LOW
            )
        
        return False, "", WarningLevel.SAFE
    
    def evaluate(self,
                 final_confidence: float,
                 ml_predictions: Dict[str, float],
                 dl_predictions: Optional[Dict[str, float]] = None,
                 predicted_class: Optional[int] = None,
                 class_distribution: Optional[Dict[int, int]] = None,
                 signal_quality_score: Optional[float] = None) -> RiskWarning:
        """
        综合评估误诊风险
        
        Args:
            final_confidence: 最终置信度
            ml_predictions: 传统ML模型预测结果
            dl_predictions: 深度学习模型预测结果（可选）
            predicted_class: 预测的类别（可选）
            class_distribution: 训练集类别分布（可选）
            signal_quality_score: 信号质量评分（可选）
        
        Returns:
            RiskWarning对象
        """
        reasons = []
        max_level = WarningLevel.SAFE
        
        # 1. 检查置信度
        triggered, reason, level = self.check_confidence(final_confidence)
        if triggered:
            reasons.append(reason)
            max_level = max(max_level, level, key=lambda x: list(WarningLevel).index(x))
        
        # 2. 检查模型冲突
        triggered, reason, level = self.check_model_conflict(ml_predictions, dl_predictions)
        if triggered:
            reasons.append(reason)
            max_level = max(max_level, level, key=lambda x: list(WarningLevel).index(x))
        
        # 3. 检查类别不平衡
        if predicted_class is not None and class_distribution:
            triggered, reason, level = self.check_class_imbalance(predicted_class, class_distribution)
            if triggered:
                reasons.append(reason)
                max_level = max(max_level, level, key=lambda x: list(WarningLevel).index(x))
        
        # 4. 检查信号质量
        if signal_quality_score is not None:
            triggered, reason, level = self.check_signal_quality(signal_quality_score)
            if triggered:
                reasons.append(reason)
                max_level = max(max_level, level, key=lambda x: list(WarningLevel).index(x))
        
        # 生成建议
        suggestions = self._generate_suggestions(max_level, reasons)
        
        # 判断是否需要人工复核
        need_manual_review = max_level in [WarningLevel.MEDIUM, WarningLevel.HIGH, WarningLevel.CRITICAL]
        
        return RiskWarning(
            level=max_level,
            confidence=final_confidence,
            reasons=reasons if reasons else ["无异常"],
            suggestions=suggestions,
            need_manual_review=need_manual_review
        )
    
    def _generate_suggestions(self, level: WarningLevel, reasons: List[str]) -> List[str]:
        """生成建议"""
        suggestions = []
        
        if level == WarningLevel.SAFE:
            suggestions.append("诊断结果可信度高，可直接使用")
        
        elif level == WarningLevel.LOW:
            suggestions.append("建议关注患者后续情况")
            suggestions.append("可结合其他检查结果综合判断")
        
        elif level == WarningLevel.MEDIUM:
            suggestions.append("建议人工复核诊断结果")
            suggestions.append("建议进行更详细的检查")
        
        elif level == WarningLevel.HIGH:
            suggestions.append("强烈建议人工复核")
            suggestions.append("建议重新采集ECG信号")
            suggestions.append("建议进行多导联ECG检查")
        
        elif level == WarningLevel.CRITICAL:
            suggestions.append("必须人工复核，不可直接使用AI诊断结果")
            suggestions.append("建议立即重新采集高质量ECG信号")
            suggestions.append("建议由资深心电图医师诊断")
        
        # 根据具体原因添加针对性建议
        for reason in reasons:
            if "信号质量" in reason:
                suggestions.append("建议检查电极接触是否良好")
            if "少数类" in reason:
                suggestions.append("建议结合临床症状综合判断")
            if "模型冲突" in reason:
                suggestions.append("建议使用多种诊断方法交叉验证")
        
        return list(set(suggestions))  # 去重


# 默认预警系统实例
default_warning_system = RiskWarningSystem()


def evaluate_risk(final_confidence: float,
                 ml_predictions: Dict[str, float],
                 dl_predictions: Optional[Dict[str, float]] = None,
                 predicted_class: Optional[int] = None,
                 class_distribution: Optional[Dict[int, int]] = None,
                 signal_quality_score: Optional[float] = None) -> RiskWarning:
    """
    快捷函数：评估误诊风险
    
    使用默认预警系统进行评估
    """
    return default_warning_system.evaluate(
        final_confidence=final_confidence,
        ml_predictions=ml_predictions,
        dl_predictions=dl_predictions,
        predicted_class=predicted_class,
        class_distribution=class_distribution,
        signal_quality_score=signal_quality_score
    )
