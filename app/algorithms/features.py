"""
ECG 特征提取模块
"""
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Any, List


class ECGFeatureExtractor:
    """ECG 特征提取器"""
    
    def extract(self, ecg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取 ECG 特征
        
        Args:
            ecg_data: 预处理后的 ECG 数据
            
        Returns:
            特征字典
        """
        signal_data = ecg_data['signal']
        sampling_rate = ecg_data['sampling_rate']
        
        # 1. R 波检测
        r_peaks = self._detect_r_peaks(signal_data, sampling_rate)
        
        # 2. 心率计算
        heart_rate = self._calculate_heart_rate(r_peaks, sampling_rate)
        
        # 3. RR 间期特征
        rr_intervals = self._calculate_rr_intervals(r_peaks, sampling_rate)
        
        # 4. HRV 特征（心率变异性）
        hrv_features = self._calculate_hrv(rr_intervals)
        
        return {
            'r_peaks': r_peaks,
            'heart_rate': heart_rate,
            'rr_intervals': rr_intervals,
            'hrv': hrv_features,
            'signal': signal_data
        }
    
    def _detect_r_peaks(self, signal_data: np.ndarray, fs: float) -> np.ndarray:
        """检测 R 波位置"""
        # 使用简单的峰值检测
        distance = int(0.6 * fs)  # 最小间隔 0.6 秒
        peaks, _ = find_peaks(signal_data, distance=distance, prominence=0.3)
        return peaks
    
    def _calculate_heart_rate(self, r_peaks: np.ndarray, fs: float) -> float:
        """计算平均心率"""
        if len(r_peaks) < 2:
            return 0.0
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = np.mean(rr_intervals)
        heart_rate = 60 / mean_rr if mean_rr > 0 else 0
        return float(heart_rate)
    
    def _calculate_rr_intervals(self, r_peaks: np.ndarray, fs: float) -> List[float]:
        """计算 RR 间期（毫秒）"""
        if len(r_peaks) < 2:
            return []
        rr_intervals = np.diff(r_peaks) / fs * 1000  # 转换为毫秒
        return rr_intervals.tolist()
    
    def _calculate_hrv(self, rr_intervals: List[float]) -> Dict[str, float]:
        """计算心率变异性特征"""
        if len(rr_intervals) < 2:
            return {'sdnn': 0.0, 'rmssd': 0.0}
        
        rr_array = np.array(rr_intervals)
        
        # SDNN: RR 间期标准差
        sdnn = float(np.std(rr_array))
        
        # RMSSD: 相邻 RR 间期差值的均方根
        diff_rr = np.diff(rr_array)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
        
        return {
            'sdnn': sdnn,
            'rmssd': rmssd
        }
