"""
测试Grad-CAM可解释性功能
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import wfdb

from app.algorithms.deep_models import ResNet1D
from app.algorithms.grad_cam import GradCAM, visualize_grad_cam, explain_prediction


def test_grad_cam_with_real_data():
    """使用真实MIT-BIH数据测试Grad-CAM"""
    print("="*60)
    print("Grad-CAM可解释性测试")
    print("="*60)
    
    # 1. 加载模型
    print("\n1. 加载ResNet-1D模型...")
    model = ResNet1D(num_classes=3)
    model_path = 'app/algorithms/models/resnet1d_best.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本: python scripts/train_multimodal_models.py")
        return
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✓ 模型加载成功")
    
    # 2. 加载真实ECG数据
    print("\n2. 加载MIT-BIH数据...")
    record_path = 'data/100'
    
    if not os.path.exists(f"{record_path}.dat"):
        print(f"❌ 数据文件不存在: {record_path}.dat")
        print("请确保MIT-BIH数据集在data/目录下")
        return
    
    signal_data, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # 提取一个心拍片段
    sample_idx = annotation.sample[10]  # 第10个心拍
    start = max(0, sample_idx - 500)
    end = start + 1000
    signal = signal_data[start:end, 0]
    
    # 归一化
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    print(f"✓ 数据加载成功，信号长度: {len(signal)}")
    print(f"  心拍类型: {annotation.symbol[10]}")
    
    # 3. 选择目标层（ResNet的最后一个卷积层）
    print("\n3. 初始化Grad-CAM...")
    target_layer = model.layer4[-1].conv2
    print(f"✓ 目标层: {target_layer}")
    
    # 4. 生成解释
    print("\n4. 生成可解释性分析...")
    result = explain_prediction(model, signal, target_layer, device='cpu')
    
    print(f"\n预测结果:")
    print(f"  诊断: {result['prediction']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"  重要区域数量: {len(result['important_regions'])}")
    print(f"  解释: {result['explanation']}")
    
    if result['important_regions']:
        print(f"\n  重要区域详情:")
        for i, (start, end) in enumerate(result['important_regions'], 1):
            print(f"    区域{i}: 采样点 {start}-{end} (长度: {end-start})")
    
    # 5. 可视化
    print("\n5. 生成可视化图像...")
    input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class, confidence = grad_cam.generate_heatmap(input_tensor)
    
    os.makedirs('experiments/results', exist_ok=True)
    save_path = 'experiments/results/grad_cam_example.png'
    visualize_grad_cam(signal, cam, pred_class, confidence, save_path)
    
    print(f"\n✅ Grad-CAM测试完成！")
    print(f"可视化结果已保存到: {save_path}")
    print(f"\n说明:")
    print(f"  - 蓝色曲线: 原始ECG信号")
    print(f"  - 红色区域: 模型认为重要的区域（越红越重要）")
    print(f"  - 这些区域是模型做出诊断决策的主要依据")


def test_grad_cam_with_test_data():
    """使用测试数据测试Grad-CAM"""
    print("\n" + "="*60)
    print("使用测试数据测试Grad-CAM")
    print("="*60)
    
    # 1. 加载模型
    print("\n1. 加载ResNet-1D模型...")
    model = ResNet1D(num_classes=3)
    model_path = 'app/algorithms/models/resnet1d_best.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✓ 模型加载成功")
    
    # 2. 加载测试数据
    test_files = [
        ('test_data/normal_ecg.csv', '正常心率'),
        ('test_data/bradycardia_ecg.csv', '心动过缓'),
        ('test_data/tachycardia_ecg.csv', '心动过速')
    ]
    
    for file_path, description in test_files:
        if not os.path.exists(file_path):
            print(f"\n⏭️  跳过不存在的文件: {file_path}")
            continue
        
        print(f"\n处理: {description} ({file_path})")
        
        # 读取CSV
        import pandas as pd
        df = pd.read_csv(file_path)
        signal = df['ecg'].values[:1000]  # 取前1000个点
        
        # 归一化
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # 生成解释
        target_layer = model.layer4[-1].conv2
        result = explain_prediction(model, signal, target_layer, device='cpu')
        
        print(f"  预测: {result['prediction']} (置信度: {result['confidence']:.2%})")
        print(f"  重要区域: {len(result['important_regions'])}个")
        
        # 可视化
        input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
        grad_cam = GradCAM(model, target_layer)
        cam, pred_class, confidence = grad_cam.generate_heatmap(input_tensor)
        
        save_path = f'experiments/results/grad_cam_{description}.png'
        visualize_grad_cam(signal, cam, pred_class, confidence, save_path)


if __name__ == '__main__':
    # 测试1: 使用MIT-BIH真实数据
    test_grad_cam_with_real_data()
    
    # 测试2: 使用测试数据
    test_grad_cam_with_test_data()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
    print("\n查看生成的图像:")
    print("  - experiments/results/grad_cam_example.png")
    print("  - experiments/results/grad_cam_正常心率.png")
    print("  - experiments/results/grad_cam_心动过缓.png")
    print("  - experiments/results/grad_cam_心动过速.png")
