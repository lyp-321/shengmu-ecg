"""
生成测试 ECG 数据
"""
import numpy as np
import os

def generate_test_ecg(filename, duration=10, sampling_rate=360, heart_rate=75):
    """
    生成测试 ECG 数据
    
    Args:
        filename: 输出文件名
        duration: 持续时间（秒）
        sampling_rate: 采样率（Hz）
        heart_rate: 心率（bpm）
    """
    # 计算总样本数
    num_samples = duration * sampling_rate
    
    # 时间轴
    t = np.linspace(0, duration, num_samples)
    
    # 心跳周期
    beat_period = 60 / heart_rate
    
    # 生成基础 ECG 信号（简化的 PQRST 波形）
    ecg_signal = np.zeros(num_samples)
    
    for i in range(int(duration / beat_period)):
        # 每个心跳的起始时间
        beat_start = i * beat_period
        
        # P 波（0.08s）
        p_center = beat_start + 0.08
        p_width = 0.04
        p_amp = 0.15
        
        # QRS 波群（0.08s）
        qrs_center = beat_start + 0.16
        qrs_width = 0.04
        qrs_amp = 1.0
        
        # T 波（0.16s）
        t_center = beat_start + 0.36
        t_width = 0.08
        t_amp = 0.3
        
        # 添加波形
        for j, time in enumerate(t):
            # P 波
            if abs(time - p_center) < p_width:
                ecg_signal[j] += p_amp * np.exp(-((time - p_center) / (p_width/3))**2)
            
            # QRS 波
            if abs(time - qrs_center) < qrs_width:
                ecg_signal[j] += qrs_amp * np.exp(-((time - qrs_center) / (qrs_width/4))**2)
            
            # T 波
            if abs(time - t_center) < t_width:
                ecg_signal[j] += t_amp * np.exp(-((time - t_center) / (t_width/3))**2)
    
    # 添加少量噪声
    noise = np.random.normal(0, 0.02, num_samples)
    ecg_signal += noise
    
    # 保存为 CSV
    np.savetxt(filename, ecg_signal, delimiter=',', fmt='%.6f')
    print(f"✅ 已生成测试数据: {filename}")
    print(f"   时长: {duration}秒")
    print(f"   采样率: {sampling_rate} Hz")
    print(f"   心率: {heart_rate} bpm")


if __name__ == "__main__":
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "test_data")
    
    # 创建 test_data 目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成不同心率的测试数据
    print("🔧 生成测试 ECG 数据...\n")
    
    # 正常心率
    generate_test_ecg(os.path.join(output_dir, "normal_ecg.csv"), duration=10, heart_rate=75)
    print()
    
    # 心动过缓
    generate_test_ecg(os.path.join(output_dir, "bradycardia_ecg.csv"), duration=10, heart_rate=50)
    print()
    
    # 心动过速
    generate_test_ecg(os.path.join(output_dir, "tachycardia_ecg.csv"), duration=10, heart_rate=110)
    print()
    
    print("✨ 所有测试数据生成完成！")
    print("\n📁 测试文件位置：")
    print(f"   - {os.path.join(output_dir, 'normal_ecg.csv')} (正常心率 75 bpm)")
    print(f"   - {os.path.join(output_dir, 'bradycardia_ecg.csv')} (心动过缓 50 bpm)")
    print(f"   - {os.path.join(output_dir, 'tachycardia_ecg.csv')} (心动过速 110 bpm)")
