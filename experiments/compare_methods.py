"""
ECG 分析方法对比实验
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


class MethodComparison:
    """方法对比实验"""
    
    def __init__(self):
        self.methods = []
        self.results = {}
    
    def add_method(self, name: str, func):
        """添加待对比的方法"""
        self.methods.append({'name': name, 'func': func})
    
    def run_comparison(self, test_data: List[np.ndarray]) -> Dict:
        """运行对比实验"""
        for method in self.methods:
            method_name = method['name']
            method_func = method['func']
            
            results = []
            for data in test_data:
                result = method_func(data)
                results.append(result)
            
            self.results[method_name] = results
        
        return self.results
    
    def plot_results(self, metric: str = 'accuracy'):
        """绘制对比结果"""
        plt.figure(figsize=(10, 6))
        
        for method_name, results in self.results.items():
            values = [r.get(metric, 0) for r in results]
            plt.plot(values, label=method_name, marker='o')
        
        plt.xlabel('测试样本')
        plt.ylabel(metric)
        plt.title(f'方法对比 - {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'experiments/comparison_{metric}.png')
        plt.close()
    
    def generate_report(self) -> str:
        """生成对比报告"""
        report = "# ECG 分析方法对比报告\n\n"
        
        for method_name, results in self.results.items():
            report += f"## {method_name}\n"
            report += f"- 测试样本数: {len(results)}\n"
            # 添加更多统计信息
            report += "\n"
        
        return report


if __name__ == "__main__":
    # 示例使用
    comparison = MethodComparison()
    
    # TODO: 添加具体的对比方法和测试数据
    print("对比实验模块已准备就绪")
