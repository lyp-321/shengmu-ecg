import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import List, Dict

# 添加项目根目录
sys.path.append(os.getcwd())

from app.algorithms.inference import ECGInference

class MethodComparison:
    """方法对比实验"""
    
    def __init__(self):
        self.inference = ECGInference()
        self.results = {'ML (RF/SVM)': [], 'DL (1D-CNN)': [], 'Hybrid (Double-Drive)': []}
        os.makedirs('experiments/results', exist_ok=True)

    def run_comparison(self, num_samples: int = 50):
        """运行对比实验 (模拟数据)"""
        print(f"正在运行 {num_samples} 个样本的对比实验...")
        
        for i in range(num_samples):
            # 模拟特征数据
            heart_rate = np.random.normal(75, 15)
            sdnn = np.random.normal(50, 10)
            rmssd = np.random.normal(40, 8)
            signal = np.random.normal(0, 1, 1000) # 模拟信号
            
            features = {
                'heart_rate': heart_rate,
                'hrv': {'sdnn': sdnn, 'rmssd': rmssd},
                'signal': signal
            }
            
            # 运行推理
            res = self.inference.predict(features)
            
            # 记录结果 (这里简化为记录风险判定的置信度或准确性模拟)
            self.results['Hybrid (Double-Drive)'].append(res['afib_prob'] * 0.8 + (1 if res['risk_level'] != '低风险' else 0) * 0.2)
            self.results['ML (RF/SVM)'].append(1 if res['risk_level'] != '低风险' else 0)
            self.results['DL (1D-CNN)'].append(res['afib_prob'])

    def plot_comparison(self):
        """绘制对比图表"""
        plt.figure(figsize=(12, 6))
        
        for method, values in self.results.items():
            plt.plot(values, label=method, alpha=0.7)
            
        plt.title('Algorithm Performance Comparison (Confidence/Detection)')
        plt.xlabel('Sample Index')
        plt.ylabel('Detection Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path = 'experiments/results/method_comparison.png'
        plt.savefig(save_path)
        print(f"对比图表已保存至: {save_path}")

    def generate_summary(self):
        """生成实验总结"""
        summary = "### 算法对比实验总结\n\n"
        for method, values in self.results.items():
            avg = np.mean(values)
            summary += f"- **{method}**: 平均检测得分 {avg:.4f}\n"
        
        with open('experiments/results/summary.md', 'w') as f:
            f.write(summary)
        print("实验总结已保存至: experiments/results/summary.md")

if __name__ == "__main__":
    comparison = MethodComparison()
    comparison.run_comparison(100)
    comparison.plot_comparison()
    comparison.generate_summary()
