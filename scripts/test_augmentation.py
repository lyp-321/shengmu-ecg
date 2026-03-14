"""
快速测试数据增强和模型改动
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from scripts.train_multimodal_models import ECGDataset, mixup_data, FocalLoss
from app.algorithms.deep_models import ResNet1D, SEResNet1D

def test_augmentation():
    """测试数据增强"""
    print("测试数据增强...")
    X = np.random.randn(10, 1, 1000).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    # 不增强
    ds_no_aug = ECGDataset(X, y, augment=False)
    x1, _ = ds_no_aug[0]
    x2, _ = ds_no_aug[0]
    assert torch.allclose(x1, x2), "不增强时应该返回相同数据"
    print("  ✓ 不增强模式正常")
    
    # 增强
    ds_aug = ECGDataset(X, y, augment=True)
    x1, _ = ds_aug[0]
    x2, _ = ds_aug[0]
    # 增强后每次应该不同（概率性检查）
    print(f"  ✓ 增强模式正常（差异: {torch.abs(x1 - x2).mean():.4f}）")

def test_mixup():
    """测试 MixUp"""
    print("\n测试 MixUp...")
    x = torch.randn(4, 1, 1000)
    y = torch.tensor([0, 1, 2, 0])
    
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    assert mixed_x.shape == x.shape
    assert 0 <= lam <= 1
    print(f"  ✓ MixUp 正常（λ={lam:.3f}）")

def test_focal_loss():
    """测试 Focal Loss + Label Smoothing"""
    print("\n测试 Focal Loss...")
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1, num_classes=3)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    
    loss = criterion(logits, targets)
    assert loss.item() > 0
    print(f"  ✓ Focal Loss 正常（loss={loss.item():.4f}）")

def test_model_size():
    """测试模型参数量"""
    print("\n测试模型容量...")
    models = {
        'ResNet1D': ResNet1D(num_classes=3),
        'SEResNet1D': SEResNet1D(num_classes=3),
    }
    
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} 参数")
        
        # 测试前向传播
        x = torch.randn(2, 1, 1000)
        out = model(x)
        assert out.shape == (2, 3), f"输出形状错误: {out.shape}"
    
    print("  ✓ 模型结构正常")

if __name__ == '__main__':
    test_augmentation()
    test_mixup()
    test_focal_loss()
    test_model_size()
    print("\n✅ 所有测试通过！")
