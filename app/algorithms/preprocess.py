"""
ECG 数据预处理模块
"""
import numpy as np
from scipy import signal
from typing import Dict, Any


class ECGPreprocessor:
    """ECG 数据预处理器"""
    
    def __init__(self, lowcut=0.5, highcut=40, order=4):
        """
        初始化预处理器
        
        Args:
            lowcut: 低频截止频率
            highcut: 高频截止频率
            order: 滤波器阶数
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
    
    def process(self, ecg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理 ECG 数据
        
        Args:
            ecg_data: 原始 ECG 数据
            
        Returns:
            预处理后的数据
        """
        signal_data = ecg_data['signal']
        sampling_rate = ecg_data['sampling_rate']
        
        # 1. 去除基线漂移
        detrended = self._remove_baseline(signal_data)
        
        # 2. 带通滤波
        filtered = self._bandpass_filter(detrended, sampling_rate)
        
        # 3. 归一化
        normalized = self._normalize(filtered)
        
        return {
            'signal': normalized,
            'sampling_rate': sampling_rate,
            'preprocessed': True
        }
    
    def _remove_baseline(self, signal_data: np.ndarray) -> np.ndarray:
        """去除基线漂移"""
        # 使用高通滤波或多项式拟合
        return signal_data - np.mean(signal_data)
    
    def _bandpass_filter(self, signal_data: np.ndarray, fs: float) -> np.ndarray:
        """带通滤波"""
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = signal.butter(self.order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    
    def _normalize(self, signal_data: np.ndarray) -> np.ndarray:
        """归一化到 [-1, 1]"""
        max_val = np.max(np.abs(signal_data))
        if max_val > 0:
            return signal_data / max_val
        return signal_data
