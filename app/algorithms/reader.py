"""
ECG 数据读取模块
"""
import numpy as np
from typing import Dict, Any


class ECGReader:
    """ECG 数据读取器"""
    
    def read(self, file_path: str) -> Dict[str, Any]:
        """
        读取 ECG 文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含 ECG 数据的字典
        """
        # TODO: 根据文件格式实现具体读取逻辑
        # 支持常见格式：.dat, .csv, .mat, .edf 等
        
        if file_path.endswith('.csv'):
            return self._read_csv(file_path)
        elif file_path.endswith('.dat'):
            return self._read_dat(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """读取 CSV 格式，支持表头和多列"""
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
            # 如果是单列且无表头，pandas 可能会把第一行当表头，这里做个简单处理
            if df.empty:
                data = np.loadtxt(file_path, delimiter=',')
            else:
                # 尝试寻找包含 'ecg', 'signal', 'lead' 等关键字的列
                target_col = None
                for col in df.columns:
                    if any(key in str(col).lower() for key in ['ecg', 'signal', 'lead', 'val']):
                        target_col = col
                        break
                
                if target_col is not None:
                    data = df[target_col].values
                else:
                    # 否则取第一列数值列
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        data = df[numeric_cols[0]].values
                    else:
                        data = df.iloc[:, 0].values
        except:
            # 回退到 loadtxt
            data = np.loadtxt(file_path, delimiter=',')
            
        return {
            'signal': data.flatten(),
            'sampling_rate': 360,
            'format': 'csv'
        }
    
    def _read_dat(self, file_path: str) -> Dict[str, Any]:
        """读取 DAT 格式，支持 WFDB 和原始二进制"""
        import os
        import wfdb
        
        # 1. 尝试作为 WFDB 记录读取
        record_name = os.path.splitext(file_path)[0]
        if os.path.exists(record_name + '.hea'):
            try:
                record = wfdb.rdrecord(record_name)
                return {
                    'signal': record.p_signal[:, 0].flatten(),
                    'sampling_rate': record.fs,
                    'format': 'wfdb'
                }
            except Exception as e:
                print(f"WFDB read error: {e}")
        
        # 2. 尝试作为原始二进制读取 (假设为 16位整数，常见于 ECG 原始数据)
        try:
            # 尝试读取为 int16
            data = np.fromfile(file_path, dtype=np.int16)
            # 简单归一化
            data = data.astype(np.float32) / 1000.0
            return {
                'signal': data.flatten(),
                'sampling_rate': 360,
                'format': 'binary_int16'
            }
        except Exception as e:
            raise ValueError(f"无法解析 DAT 文件: {e}")
