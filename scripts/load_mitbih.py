import os
import wfdb
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt

# 配置路径
DATA_DIR = "data/mitbih"
os.makedirs(DATA_DIR, exist_ok=True)

# MIT-BIH 记录列表 (选取部分代表性记录)
RECORDS = [
    '100', '101', '103', '105', '106', '111', '113', '115', '118', '119',
    '200', '201', '203', '205', '208', '210', '212', '213', '215', '219'
]

# 标签映射 (AAMI 标准)
# N: 正常, S: 室上性, V: 室性, F: 融合, Q: 未知
LABEL_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # N (Normal)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # S (SVEB)
    'V': 2, 'E': 2,                          # V (VEB)
    'F': 3,                                  # F (Fusion)
    '/': 4, 'f': 4, 'Q': 4                   # Q (Unknown)
}

def download_data():
    """下载 MIT-BIH 数据集"""
    print("正在检查/下载 MIT-BIH 数据...")
    for record in RECORDS:
        dat_path = os.path.join(DATA_DIR, f"{record}.dat")
        hea_path = os.path.join(DATA_DIR, f"{record}.hea")
        atr_path = os.path.join(DATA_DIR, f"{record}.atr")
        
        if not (os.path.exists(dat_path) and os.path.exists(hea_path) and os.path.exists(atr_path)):
            print(f"尝试下载记录 {record}...")
            try:
                wfdb.dl_database('mitdb', DATA_DIR, records=[record], overwrite=True)
            except Exception as e:
                print(f"下载记录 {record} 失败: {e}")
    print("数据下载检查完成。")

def load_record(record_path: str) -> Tuple[np.ndarray, wfdb.Annotation]:
    """读取单条记录的信号和标注"""
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    # 通常使用第一个通道 (MLII)
    signal = record.p_signal[:, 0]
    return signal, annotation

def segment_beats(signal: np.ndarray, annotation: wfdb.Annotation, win_size: int = 180) -> List[dict]:
    """
    心拍分割：以 R 峰为中心，左右各取 win_size 个采样点
    """
    beats = []
    samples = annotation.sample
    symbols = annotation.symbol
    
    for i in range(len(samples)):
        idx = samples[i]
        sym = symbols[i]
        
        # 只处理 AAMI 定义的类别
        if sym in LABEL_MAP:
            # 确保不越界
            if idx - win_size >= 0 and idx + win_size < len(signal):
                beat_signal = signal[idx - win_size : idx + win_size]
                beats.append({
                    'signal': beat_signal,
                    'label': LABEL_MAP[sym],
                    'symbol': sym
                })
    return beats

def main():
    download_data()
    
    all_beats = []
    print("正在处理记录并分割心拍...")
    for record in RECORDS:
        path = os.path.join(DATA_DIR, record)
        if os.path.exists(path + ".dat") and os.path.exists(path + ".hea") and os.path.exists(path + ".atr"):
            try:
                signal, ann = load_record(path)
                beats = segment_beats(signal, ann)
                all_beats.extend(beats)
                print(f"记录 {record}: 提取了 {len(beats)} 个心拍")
            except Exception as e:
                print(f"处理记录 {record} 出错: {e}")
        else:
            print(f"记录 {record} 文件不完整，跳过。")
    
    if not all_beats:
        print("没有提取到任何心拍数据，请检查网络或数据文件。")
        return

    # 转换为 DataFrame 方便后续处理
    df = pd.DataFrame(all_beats)
    print(f"\n总计提取心拍数: {len(df)}")
    print("类别分布:")
    print(df['label'].value_counts())
    
    # 保存处理后的数据
    processed_path = os.path.join(DATA_DIR, "processed_beats.pkl")
    df.to_pickle(processed_path)
    print(f"\n数据处理完成！已保存至 {processed_path}")

if __name__ == "__main__":
    main()
