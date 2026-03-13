#!/usr/bin/env python
"""
将MIT-BIH数据集转换为CSV格式
"""
import wfdb
import pandas as pd
import os
from tqdm import tqdm

def convert_mitbih_to_csv(data_dir='data', output_dir='data/mitbih_csv'):
    """
    将MIT-BIH .dat文件转换为CSV
    
    Args:
        data_dir: MIT-BIH数据目录
        output_dir: 输出CSV目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有MIT-BIH记录编号（只包含纯数字的文件名）
    records = []
    for file in os.listdir(data_dir):
        if file.endswith('.dat'):
            record_name = file.replace('.dat', '')
            # 只处理纯数字的文件名（MIT-BIH原始数据）
            if record_name.isdigit():
                records.append(record_name)
    
    records = sorted(set(records))
    
    print(f"找到 {len(records)} 条MIT-BIH记录")
    print(f"输出目录: {output_dir}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for record in tqdm(records, desc="转换进度"):
        try:
            # 读取记录
            record_path = os.path.join(data_dir, record)
            signal, fields = wfdb.rdsamp(record_path)
            
            # 转换为DataFrame
            df = pd.DataFrame(signal, columns=fields['sig_name'])
            
            # 添加时间列
            df.insert(0, 'time', df.index / fields['fs'])
            
            # 保存为CSV
            output_file = os.path.join(output_dir, f'{record}.csv')
            df.to_csv(output_file, index=False)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 转换失败: {record} - {e}")
            fail_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ 转换完成: {success_count} 成功, {fail_count} 失败")
    print(f"CSV文件保存在: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    convert_mitbih_to_csv()
