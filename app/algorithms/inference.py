"""
ECG 推理/结果计算模块
"""
from typing import Dict, Any


class ECGInference:
    """ECG 推理引擎"""
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于特征进行推理
        
        Args:
            features: 提取的特征
            
        Returns:
            分析结果
        """
        heart_rate = features.get('heart_rate', 0)
        hrv = features.get('hrv', {})
        
        # 简单的规则推理（可替换为机器学习模型）
        diagnosis = self._rule_based_diagnosis(heart_rate, hrv)
        
        return {
            'heart_rate': heart_rate,
            'hrv_sdnn': hrv.get('sdnn', 0),
            'hrv_rmssd': hrv.get('rmssd', 0),
            'diagnosis': diagnosis,
            'risk_level': self._assess_risk(heart_rate, hrv)
        }
    
    def _rule_based_diagnosis(self, heart_rate: float, hrv: Dict[str, float]) -> str:
        """基于规则的诊断"""
        if heart_rate < 60:
            return "心动过缓"
        elif heart_rate > 100:
            return "心动过速"
        elif 60 <= heart_rate <= 100:
            return "正常窦性心律"
        else:
            return "未知"
    
    def _assess_risk(self, heart_rate: float, hrv: Dict[str, float]) -> str:
        """风险评估"""
        sdnn = hrv.get('sdnn', 0)
        
        if heart_rate < 40 or heart_rate > 120:
            return "高风险"
        elif sdnn < 50:
            return "中风险"
        else:
            return "低风险"
