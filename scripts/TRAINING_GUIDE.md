# 模型训练指南

## 📋 前提条件

### 1. 安装依赖

确保已安装所有必要的Python包：

```bash
conda activate ai
pip install torch torchvision
pip install xgboost lightgbm catboost
pip install scikit-learn
pip install wfdb pywavelets
pip install matplotlib seaborn
pip install tqdm
```

### 2. 准备数据

#### 方法1：使用MIT-BIH数据集（推荐）

MIT-BIH数据集是心电图分析的标准数据集。

**下载方式**：

```bash
# 使用wfdb下载（推荐）
python -c "import wfdb; wfdb.dl_database('mitdb', 'data')"
```

或者手动下载：
- 访问：https://physionet.org/content/mitdb/1.0.0/
- 下载所有 `.dat`, `.hea`, `.atr` 文件到 `data/` 目录

**数据集说明**：
- 48条记录，每条约30分钟
- 采样率：360 Hz
- 包含多种心律失常类型
- 专家标注的R波位置和心律类型

#### 方法2：使用模拟数据

如果无法下载MIT-BIH数据集，训练脚本会自动生成模拟数据。

---

## 🚀 训练步骤

### 快速开始（一键训练）

```bash
# 给脚本添加执行权限
chmod +x scripts/train_all.sh

# 运行训练
./scripts/train_all.sh
```

### 分步训练

#### 步骤1：训练传统机器学习模型

```bash
conda activate ai
python scripts/train_traditional_ml.py
```

**训练的模型**：
- ✅ Random Forest
- ✅ XGBoost
- ✅ LightGBM
- ✅ CatBoost
- ✅ SVM (RBF kernel)

**输出**：
- `app/algorithms/models/scaler.pkl` - 特征标准化器
- `app/algorithms/models/randomforest_model.pkl`
- `app/algorithms/models/xgboost_model.pkl`
- `app/algorithms/models/lightgbm_model.pkl`
- `app/algorithms/models/catboost_model.pkl`
- `app/algorithms/models/svm_model.pkl`

#### 步骤2：训练深度学习模型

```bash
conda activate ai
python scripts/train_multimodal_models.py
```

**训练的模型**：
- ✅ ResNet-1D (18层残差网络)
- ✅ SE-ResNet-1D (ResNet + 注意力机制)
- ✅ Transformer (多头自注意力)
- ✅ BiLSTM (双向LSTM + 注意力)
- ✅ TCN (时间卷积网络)
- ✅ Inception-1D (多尺度卷积)

**输出**：
- `app/algorithms/models/resnet1d_best.pth`
- `app/algorithms/models/seresnet1d_best.pth`
- `app/algorithms/models/transformer_best.pth`
- `app/algorithms/models/bilstm_best.pth`
- `app/algorithms/models/tcn_best.pth`
- `app/algorithms/models/inception_best.pth`
- `experiments/results/training_curves.png` - 训练曲线

---

## 📊 训练参数

### 传统机器学习

```python
# Random Forest
n_estimators=100
max_depth=10

# XGBoost
n_estimators=100
max_depth=6
learning_rate=0.1

# LightGBM
n_estimators=100
max_depth=6
learning_rate=0.1

# SVM
kernel='rbf'
C=1.0
```

### 深度学习

```python
# 通用参数
num_epochs=30
batch_size=32
learning_rate=0.001
optimizer=Adam

# 数据划分
train: 64%
val: 16%
test: 20%
```

---

## 🔧 自定义训练

### 修改训练参数

编辑 `scripts/train_multimodal_models.py`:

```python
# 修改训练轮数
history = train_model(
    model, train_loader, val_loader,
    num_epochs=50,  # 改为50轮
    lr=0.001,
    device=device,
    model_name=name.lower()
)
```

### 修改数据量

```python
# 修改样本数量
X, y = load_mitbih_data(num_samples=10000)  # 改为10000个样本
```

### 使用GPU加速

如果有NVIDIA GPU：

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 训练会自动使用GPU
python scripts/train_multimodal_models.py
```

---

## 📈 查看结果

### 训练日志

```bash
# 查看机器学习训练日志
cat logs/train_ml.log

# 查看深度学习训练日志
cat logs/train_dl.log
```

### 模型性能

```bash
# 查看机器学习结果
cat experiments/results/ml_results.txt

# 查看深度学习结果
cat experiments/results/model_results.txt
```

### 训练曲线

打开 `experiments/results/training_curves.png` 查看：
- 训练损失曲线
- 验证损失曲线
- 训练准确率曲线
- 验证准确率曲线

---

## ⚠️ 常见问题

### Q1: 内存不足

**解决方案**：
```python
# 减少batch_size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 减少样本数量
X, y = load_mitbih_data(num_samples=2000)
```

### Q2: 训练时间太长

**解决方案**：
```python
# 减少训练轮数
num_epochs=10

# 使用更小的模型
# 或者只训练部分模型
models = {
    'ResNet1D': ResNet1D(num_classes=num_classes),
    # 注释掉其他模型
}
```

### Q3: MIT-BIH数据下载失败

**解决方案**：
1. 检查网络连接
2. 使用VPN或代理
3. 手动下载：https://physionet.org/content/mitdb/1.0.0/
4. 或者使用模拟数据（自动生成）

### Q4: CatBoost安装失败

**解决方案**：
```bash
# CatBoost是可选的，可以跳过
# 训练脚本会自动检测并跳过CatBoost

# 或者尝试安装
pip install catboost
```

---

## 📝 训练检查清单

训练前检查：
- [ ] conda环境已激活
- [ ] 所有依赖已安装
- [ ] MIT-BIH数据已下载（或使用模拟数据）
- [ ] 有足够的磁盘空间（至少2GB）
- [ ] 有足够的内存（至少8GB）

训练后检查：
- [ ] 所有模型文件已生成
- [ ] 训练日志无错误
- [ ] 测试准确率合理（>80%）
- [ ] 训练曲线正常（无过拟合）

---

## 🎯 预期结果

### 传统机器学习

| 模型 | 预期准确率 |
|------|-----------|
| Random Forest | 85-90% |
| XGBoost | 88-92% |
| LightGBM | 87-91% |
| CatBoost | 88-92% |
| SVM | 83-88% |

### 深度学习

| 模型 | 预期准确率 |
|------|-----------|
| ResNet-1D | 90-94% |
| SE-ResNet-1D | 91-95% |
| Transformer | 89-93% |
| BiLSTM | 88-92% |
| TCN | 87-91% |
| Inception | 90-94% |

**注意**：实际结果取决于数据质量、训练参数和硬件配置。

---

## 📚 参考资料

- MIT-BIH数据集：https://physionet.org/content/mitdb/1.0.0/
- WFDB文档：https://wfdb.readthedocs.io/
- PyTorch文档：https://pytorch.org/docs/
- XGBoost文档：https://xgboost.readthedocs.io/

---

**版本**：v1.0  
**最后更新**：2026年3月13日
