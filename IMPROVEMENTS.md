# 系统改进说明

## 📅 更新日期：2026-03-13

---

## ✅ 已完成的改进

### 1. 🔒 修复数据泄露问题（Data Leakage）

**问题描述**：
- 原始训练脚本按心拍（Beat-wise）随机划分数据集
- 同一患者的心拍可能同时出现在训练集和测试集
- 导致模型"背住"患者特征，测试准确率虚高

**解决方案**：
- ✅ 修改 `scripts/train_traditional_ml.py`
- ✅ 修改 `scripts/train_multimodal_models.py`
- ✅ 实现按患者划分（Patient-wise Split）
- ✅ 确保训练集和测试集无患者重叠

**代码变更**：
```python
# 修改前（错误）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 修改后（正确）
unique_patients = np.unique(record_ids)
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=42
)
train_mask = np.isin(record_ids, train_patients)
test_mask = np.isin(record_ids, test_patients)
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
```

**预期影响**：
- 准确率可能从99%降到85-90%（更真实）
- 置信度可能从80%降到70-75%
- 但这是更可靠的性能评估

**验证方法**：
```bash
# 重新训练传统ML模型
python scripts/train_traditional_ml.py

# 重新训练深度学习模型
python scripts/train_multimodal_models.py
```

---

### 2. 📊 添加详细评估指标

**问题描述**：
- 原始脚本只报告准确率（Accuracy）
- 对于类别不平衡问题，准确率不够全面
- 缺少F1-Score、混淆矩阵等关键指标

**解决方案**：
- ✅ 添加精确率（Precision）
- ✅ 添加召回率（Recall）
- ✅ 添加F1-Score（总体 + 每个类别）
- ✅ 添加混淆矩阵
- ✅ 添加详细分类报告

**新增指标**：
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# 计算所有指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
f1_per_class = f1_score(y_test, y_pred, average=None)
cm = confusion_matrix(y_test, y_pred)

# 打印详细报告
print(classification_report(y_test, y_pred, 
                           target_names=['正常', '室性早搏', '其他异常']))
```

**输出示例**：
```
模型: RandomForest
准确率 (Accuracy): 0.9200
精确率 (Precision): 0.9150
召回率 (Recall): 0.9200
F1-Score (Weighted): 0.9175

每个类别的F1-Score:
  类别0 (正常): 0.9500
  类别1 (室性早搏): 0.8800
  类别2 (其他异常): 0.7500

混淆矩阵:
[[850  30  10]
 [ 40 130   5]
 [ 15  10  10]]

详细分类报告:
              precision    recall  f1-score   support
      正常       0.94      0.96      0.95       890
    室性早搏       0.76      0.74      0.75       175
    其他异常       0.40      0.29      0.33        35
```

**结果保存**：
- `experiments/results/ml_results_patient_wise.json` - 传统ML模型结果
- `experiments/results/dl_results_patient_wise.json` - 深度学习模型结果

---

### 3. 🔍 Grad-CAM 可解释性

**功能描述**：
- 可视化深度学习模型关注的ECG信号区域
- 生成热力图，红色区域表示模型认为重要的部分
- 提供可解释的诊断依据

**新增文件**：
- `app/algorithms/grad_cam.py` - Grad-CAM核心实现
- `scripts/test_grad_cam.py` - 测试和演示脚本

**核心功能**：
```python
from app.algorithms.grad_cam import GradCAM, visualize_grad_cam, explain_prediction

# 1. 加载模型
model = ResNet1D(num_classes=3)
model.load_state_dict(torch.load('app/algorithms/models/resnet1d_best.pth'))

# 2. 选择目标层
target_layer = model.layer4[-1].conv2

# 3. 生成解释
result = explain_prediction(model, signal, target_layer)
print(result['explanation'])
# 输出: "模型识别为正常窦性心律（置信度95.2%），主要关注3个信号区域"

# 4. 可视化
grad_cam = GradCAM(model, target_layer)
cam, pred_class, confidence = grad_cam.generate_heatmap(input_tensor)
visualize_grad_cam(signal, cam, pred_class, confidence, 'result.png')
```

**使用方法**：
```bash
# 测试Grad-CAM功能
python scripts/test_grad_cam.py

# 查看生成的可视化图像
ls experiments/results/grad_cam_*.png
```

**可视化效果**：
- 上图：原始ECG信号
- 下图：信号 + 热力图叠加
  - 蓝色曲线：原始信号
  - 红色区域：模型关注区域（越红越重要）
  - 这些区域是模型做出诊断决策的主要依据

**应用价值**：
- 提升医生对AI的信任度
- 帮助发现模型的错误模式
- 辅助临床教学和研究

---

## 📈 性能对比

### 修复前 vs 修复后

| 指标 | 修复前 | 修复后（预期） | 说明 |
|------|--------|---------------|------|
| 数据划分方式 | Beat-wise | Patient-wise | 避免数据泄露 |
| 准确率 | 99% | 85-90% | 更真实的评估 |
| 评估指标 | 仅Accuracy | Accuracy + Precision + Recall + F1 + CM | 更全面 |
| 可解释性 | 无 | Grad-CAM可视化 | 提升信任度 |

### 类别不平衡分析

**数据分布**（MIT-BIH）：
```
类别0（正常）：    4194个样本（84%）
类别1（室性早搏）：  740个样本（15%）
类别2（其他异常）：   66个样本（1%）
```

**为什么需要F1-Score**：
- 如果模型总是预测"正常"，准确率也能达到84%
- 但对少数类（室性早搏、其他异常）的识别能力为0
- F1-Score能更好地反映模型对少数类的识别能力

---

## 🔧 使用指南

### 1. 重新训练模型（推荐）

```bash
# 1. 备份旧模型
mkdir -p backup/models
cp app/algorithms/models/*.pkl backup/models/
cp app/algorithms/models/*.pth backup/models/

# 2. 重新训练传统ML模型
python scripts/train_traditional_ml.py

# 3. 重新训练深度学习模型
python scripts/train_multimodal_models.py

# 4. 查看评估结果
cat experiments/results/ml_results_patient_wise.json
cat experiments/results/dl_results_patient_wise.json
```

### 2. 测试Grad-CAM

```bash
# 运行测试脚本
python scripts/test_grad_cam.py

# 查看生成的图像
ls -lh experiments/results/grad_cam_*.png
```

### 3. 集成到报告生成

可以将Grad-CAM集成到PDF报告中：

```python
# 在 app/services/report_service.py 中添加
from app.algorithms.grad_cam import explain_prediction

# 生成可解释性分析
result = explain_prediction(model, signal, target_layer)

# 添加到报告
pdf.drawString(100, y, f"诊断依据: {result['explanation']}")
pdf.drawImage('grad_cam.png', 100, y-200, width=400, height=150)
```

---

## ⚠️ 注意事项

### 1. 性能下降是正常的

修复数据泄露后，准确率可能会下降10-15%，这是正常现象：
- 修复前：模型"背住"了患者特征（过拟合）
- 修复后：模型真正学习到了心律失常的通用特征（泛化）

### 2. 需要重新训练

修改了数据划分方式后，必须重新训练所有模型：
- 旧模型是基于错误的数据划分训练的
- 新模型才能反映真实的性能

### 3. Grad-CAM的局限性

- 只适用于CNN类模型（ResNet、Inception等）
- 不适用于全连接网络或传统ML模型
- 对于Transformer，需要使用Attention可视化

---

## 📚 参考资料

### 数据泄露问题

- [Avoiding Data Leakage in Medical AI](https://arxiv.org/abs/2008.05037)
- [Patient-wise vs Beat-wise Split in ECG Classification](https://ieeexplore.ieee.org/document/8952526)

### Grad-CAM

- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Grad-CAM++ : Improved Visual Explanations](https://arxiv.org/abs/1710.11063)

### 评估指标

- [Precision, Recall, F1-Score Explained](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Handling Imbalanced Datasets](https://imbalanced-learn.org/)

---

## 🎯 下一步计划

### 短期（本周）

- [ ] 重新训练所有模型
- [ ] 验证新模型的性能
- [ ] 更新README中的性能指标
- [ ] 将Grad-CAM集成到Web界面

### 中期（下月）

- [ ] 添加ONNX Runtime加速
- [ ] 训练Lightweight Transformer
- [ ] 添加噪声鲁棒性测试
- [ ] 完善API错误码文档

### 长期（下季度）

- [ ] 多中心临床验证
- [ ] 发表学术论文
- [ ] 申请专利和软著

---

## 📞 联系方式

如有问题或建议，请联系：
- 项目负责人：[你的名字]
- Email: [你的邮箱]
- GitHub Issues: [项目地址]/issues

---

**最后更新**：2026年3月13日  
**版本**：v2.1.0  
**状态**：✅ 改进完成，待测试验证
