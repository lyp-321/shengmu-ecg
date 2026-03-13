"""
测试新增的算法模块
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from app.algorithms.multimodal_fusion import MultiModalFusionEngine
from app.algorithms.deep_models import (
    ResNet1D, SEResNet1D, TransformerECG, BiLSTMECG, TCN, InceptionECG
)
from app.algorithms.graph_models import (
    GCN, GAT, STGCN, build_ecg_adjacency_matrix
)
from app.algorithms.federated_learning import (
    FederatedServer, DifferentialPrivacy, GradientCompression
)


def test_multimodal_fusion():
    """测试多模态融合引擎"""
    print("\n" + "="*60)
    print("测试 1: 多模态融合引擎")
    print("="*60)
    
    try:
        # 初始化融合引擎
        engine = MultiModalFusionEngine()
        print("✓ 融合引擎初始化成功")
        
        # 准备测试数据
        features = {
            'heart_rate': 75.0,
            'hrv': {'sdnn': 50.0, 'rmssd': 30.0},
            'signal': np.random.randn(1000),
            'r_peaks': np.array([100, 300, 500, 700, 900])
        }
        
        # 推理
        result = engine.predict(features)
        print(f"✓ 推理成功")
        print(f"  - 诊断: {result['diagnosis']}")
        print(f"  - 置信度: {result['confidence']:.2%}")
        print(f"  - 风险等级: {result['risk_level']}")
        print(f"  - 信号质量: {result['signal_quality']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_models():
    """测试深度学习模型"""
    print("\n" + "="*60)
    print("测试 2: 深度学习模型")
    print("="*60)
    
    batch_size = 2
    length = 1000
    num_classes = 12
    x = torch.randn(batch_size, 1, length)
    
    models = {
        'ResNet-1D': ResNet1D(num_classes=num_classes),
        'SE-ResNet-1D': SEResNet1D(num_classes=num_classes),
        'Transformer': TransformerECG(num_classes=num_classes),
        'BiLSTM': BiLSTMECG(num_classes=num_classes),
        'TCN': TCN(num_classes=num_classes),
        'Inception': InceptionECG(num_classes=num_classes)
    }
    
    all_passed = True
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, num_classes), \
                f"输出形状错误: {output.shape}"
            
            print(f"✓ {name:15s} - 输出形状: {tuple(output.shape)}")
        except Exception as e:
            print(f"✗ {name:15s} - 失败: {e}")
            all_passed = False
    
    return all_passed


def test_graph_models():
    """测试图神经网络模型"""
    print("\n" + "="*60)
    print("测试 3: 图神经网络模型")
    print("="*60)
    
    batch_size = 2
    num_nodes = 12
    length = 1000
    num_classes = 12
    
    # 多导联ECG数据
    x = torch.randn(batch_size, num_nodes, 1, length)
    
    # 构建邻接矩阵
    try:
        adj = build_ecg_adjacency_matrix(num_nodes=num_nodes, device='cpu')
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        print(f"✓ 邻接矩阵构建成功 - 形状: {tuple(adj.shape)}")
    except Exception as e:
        print(f"✗ 邻接矩阵构建失败: {e}")
        return False
    
    models = {
        'GCN': GCN(num_nodes=num_nodes, num_classes=num_classes),
        'GAT': GAT(num_nodes=num_nodes, num_classes=num_classes),
        'ST-GCN': STGCN(num_nodes=num_nodes, num_classes=num_classes)
    }
    
    all_passed = True
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(x, adj)
            
            assert output.shape == (batch_size, num_classes), \
                f"输出形状错误: {output.shape}"
            
            print(f"✓ {name:10s} - 输出形状: {tuple(output.shape)}")
        except Exception as e:
            print(f"✗ {name:10s} - 失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_federated_learning():
    """测试联邦学习框架"""
    print("\n" + "="*60)
    print("测试 4: 联邦学习框架")
    print("="*60)
    
    try:
        # 创建简单模型
        from app.algorithms.cnn_model import ECG1DCNN
        global_model = ECG1DCNN(num_classes=3)
        
        # 测试服务器
        server = FederatedServer(global_model, algorithm='FedAvg')
        print("✓ 联邦学习服务器初始化成功")
        
        # 模拟客户端模型
        num_clients = 3
        client_models = []
        client_num_samples = []
        
        for i in range(num_clients):
            # 复制全局模型参数并添加噪声
            client_params = {
                name: param.clone() + torch.randn_like(param) * 0.01
                for name, param in global_model.named_parameters()
            }
            client_models.append(client_params)
            client_num_samples.append(100 * (i + 1))
        
        print(f"✓ 模拟 {num_clients} 个客户端")
        
        # 测试聚合
        aggregated = server.aggregate(client_models, client_num_samples=client_num_samples)
        print(f"✓ FedAvg聚合成功 - 参数数量: {len(aggregated)}")
        
        # 测试差分隐私
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, mechanism='Gaussian')
        noisy_params = dp.add_noise(aggregated)
        print(f"✓ 差分隐私保护成功 (ε={dp.epsilon})")
        
        # 测试梯度压缩
        compressor = GradientCompression(method='top_k', compression_ratio=0.1)
        compressed = compressor.compress(aggregated)
        decompressed = compressor.decompress(compressed)
        print(f"✓ 梯度压缩成功 (Top-K, 压缩率=10%)")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_integration():
    """测试推理引擎集成"""
    print("\n" + "="*60)
    print("测试 5: 推理引擎集成")
    print("="*60)
    
    try:
        from app.algorithms.inference import ECGInference
        
        # 初始化推理引擎
        engine = ECGInference()
        print("✓ 推理引擎初始化成功")
        
        # 准备测试数据
        features = {
            'heart_rate': 85.0,
            'hrv': {'sdnn': 45.0, 'rmssd': 28.0},
            'signal': np.random.randn(1000),
            'r_peaks': np.array([100, 300, 500, 700, 900])
        }
        
        # 测试融合模式
        result_fusion = engine.predict(features, use_fusion=True)
        print(f"✓ 融合模式推理成功")
        print(f"  - 诊断: {result_fusion['diagnosis']}")
        print(f"  - 置信度: {result_fusion.get('confidence', 0):.2%}")
        
        # 测试传统模式（向后兼容）
        result_legacy = engine.predict(features, use_fusion=False)
        print(f"✓ 传统模式推理成功")
        print(f"  - 诊断: {result_legacy['diagnosis']}")
        print(f"  - 风险: {result_legacy['risk_level']}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("ECG算法模块测试套件")
    print("="*60)
    
    results = {
        '多模态融合引擎': test_multimodal_fusion(),
        '深度学习模型': test_deep_models(),
        '图神经网络模型': test_graph_models(),
        '联邦学习框架': test_federated_learning(),
        '推理引擎集成': test_inference_integration()
    }
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
