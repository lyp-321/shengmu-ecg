"""
Grad-CAM 可解释性模块
用于可视化深度学习模型关注的ECG信号区域
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    用于可视化CNN模型关注的信号区域
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        初始化Grad-CAM
        
        Args:
            model: 要解释的模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """前向传播钩子，保存激活值"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播钩子，保存梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入信号 (1, 1, length)
            target_class: 目标类别（None表示使用预测类别）
        
        Returns:
            cam: 热力图 (length,)
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)
        
        # 如果未指定目标类别，使用预测类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 计算权重（全局平均池化梯度）
        weights = torch.mean(self.gradients, dim=2, keepdim=True)  # (1, channels, 1)
        
        # 加权求和激活值
        cam = torch.sum(weights * self.activations, dim=1).squeeze()  # (length,)
        
        # ReLU（只保留正值）
        cam = F.relu(cam)
        
        # 归一化到[0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        生成热力图和预测信息
        
        Args:
            input_tensor: 输入信号 (1, 1, length)
            target_class: 目标类别
        
        Returns:
            cam: 热力图 (length,)
            pred_class: 预测类别
            confidence: 置信度
        """
        # 前向传播获取预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # 生成CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        return cam, pred_class, confidence


def visualize_grad_cam(signal: np.ndarray, cam: np.ndarray, 
                       pred_class: int, confidence: float,
                       save_path: Optional[str] = None) -> None:
    """
    可视化Grad-CAM结果
    
    Args:
        signal: 原始信号 (length,)
        cam: 热力图 (length,)
        pred_class: 预测类别
        confidence: 置信度
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    # 类别名称
    class_names = ['正常', '室性早搏', '其他异常']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # 绘制原始信号
    ax1.plot(signal, 'b-', linewidth=1)
    ax1.set_title(f'原始ECG信号 | 预测: {class_names[pred_class]} | 置信度: {confidence:.2%}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('振幅')
    ax1.grid(True, alpha=0.3)
    
    # 绘制信号+热力图叠加
    ax2.plot(signal, 'b-', linewidth=1, alpha=0.7, label='ECG信号')
    
    # 使用热力图着色背景
    # 将CAM插值到信号长度
    if len(cam) != len(signal):
        from scipy.interpolate import interp1d
        x_cam = np.linspace(0, len(signal)-1, len(cam))
        x_signal = np.arange(len(signal))
        f = interp1d(x_cam, cam, kind='linear', fill_value='extrapolate')
        cam_interp = f(x_signal)
        cam_interp = np.clip(cam_interp, 0, 1)
    else:
        cam_interp = cam
    
    # 使用颜色映射
    for i in range(len(signal)-1):
        alpha = cam_interp[i] * 0.7  # 透明度与CAM值成正比
        if alpha > 0.1:  # 只显示重要区域
            ax2.axvspan(i, i+1, alpha=alpha, color='red')
    
    ax2.set_title('Grad-CAM可解释性分析 | 红色区域 = 模型关注区域', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('振幅')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grad-CAM可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def explain_prediction(model: torch.nn.Module, signal: np.ndarray, 
                       target_layer: torch.nn.Module,
                       device: str = 'cpu') -> dict:
    """
    解释模型预测
    
    Args:
        model: 训练好的模型
        signal: ECG信号 (length,)
        target_layer: 目标层
        device: 设备
    
    Returns:
        解释结果字典
    """
    # 准备输入
    input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, length)
    
    # 创建Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 生成热力图
    cam, pred_class, confidence = grad_cam.generate_heatmap(input_tensor)
    
    # 找出最重要的区域（CAM值最高的区域）
    threshold = 0.5
    important_regions = cam > threshold
    
    # 找出连续的重要区域
    regions = []
    start = None
    for i, is_important in enumerate(important_regions):
        if is_important and start is None:
            start = i
        elif not is_important and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(important_regions)))
    
    # 类别名称
    class_names = ['正常窦性心律', '室性早搏', '其他异常']
    
    return {
        'prediction': class_names[pred_class],
        'prediction_class': int(pred_class),
        'confidence': float(confidence),
        'cam': cam.tolist(),
        'important_regions': regions,
        'explanation': f"模型识别为{class_names[pred_class]}（置信度{confidence:.2%}），"
                      f"主要关注{len(regions)}个信号区域"
    }


if __name__ == '__main__':
    """测试Grad-CAM"""
    print("Grad-CAM可解释性模块")
    print("="*60)
    print("功能:")
    print("1. 可视化深度学习模型关注的ECG信号区域")
    print("2. 生成热力图，红色区域表示模型认为重要的部分")
    print("3. 提供可解释的诊断依据")
    print("="*60)
    
    # 示例：如何使用
    print("\n使用示例:")
    print("""
    from app.algorithms.grad_cam import GradCAM, visualize_grad_cam, explain_prediction
    from app.algorithms.deep_models import ResNet1D
    import torch
    
    # 1. 加载模型
    model = ResNet1D(num_classes=3)
    model.load_state_dict(torch.load('app/algorithms/models/resnet1d_best.pth'))
    model.eval()
    
    # 2. 准备信号
    signal = np.random.randn(1000)  # 替换为真实ECG信号
    
    # 3. 选择目标层（ResNet的最后一个卷积层）
    target_layer = model.layer4[-1].conv2
    
    # 4. 生成解释
    result = explain_prediction(model, signal, target_layer)
    print(result['explanation'])
    
    # 5. 可视化
    input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class, confidence = grad_cam.generate_heatmap(input_tensor)
    visualize_grad_cam(signal, cam, pred_class, confidence, 'grad_cam_result.png')
    """)
