# 🫀 ECG智能心电图分析系统

<div align="center">

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**基于多模态深度学习的心电图自动分析系统**

</div>

---

## 📋 项目简介

基于MIT-BIH心律失常数据库，结合传统机器学习与深度学习，实现3类心律失常的自动识别。项目重点在于数据处理的严谨性：Patient-wise划分避免数据泄露、SMOTE+代价敏感权重处理类别不平衡、完整的实验记录与可视化。

### 📊 当前性能（Patient-wise评估）

| 模型 | 准确率 | 宏F1 | 室早F1 | 其他异常F1 |
|------|--------|------|--------|-----------|
| XGBoost | 0.9584 | 0.5310 | 0.4603 | 0.1406 |
| LightGBM | 0.9392 | 0.5133 | 0.4720 | 0.0917 |
| CatBoost | 0.9543 | 0.5279 | 0.4453 | 0.1616 |
| RandomForest | 0.9676 | 0.3533 | 0.0546 | 0.0000 |

> 60000样本，27个患者，SMOTE(1:1:1) + class_weight={0:1, 1:5, 2:20}

在严谨的 Patient-wise 评估下，集成树模型（XGBoost/CatBoost）表现出远超随机森林的泛化能力。通过引入代价敏感权重，系统成功将少数类（室早）的召回率提升了近 10 倍，在保持整体准确率的同时显著改善了临床检出率。

### ⚠️ 局限性说明

类别 2（其他异常）包含房颤、逸搏等多种形态迥异的亚类，受限于 MIT-BIH 数据库中该类别样本极稀缺（<1%），目前模型主要起"预警"作用而非精准诊断。未来计划通过迁移学习（Transfer Learning）引入更大规模的 CPSC 或 PTB-XL 数据库进行跨库训练，以提升该类别的识别精度。

---

## ✨ 功能特性

- ✅ **多格式支持**：CSV、DAT（MIT-BIH WFDB格式）
- ✅ **信号预处理**：Butterworth带通滤波、基线漂移去除、Pan-Tompkins R峰检测
- ✅ **41维特征工程**：时域16维 + 频域7维 + 小波18维（db4，5层分解）
- ✅ **3类心律识别**：正常（N/L/R）、室性早搏（V/融合拍）、其他异常（A/F/E/S/J等）
- ✅ **多模型集成**：4个传统ML模型 + 6个深度学习模型
- ✅ **严格评估**：Patient-wise Split，训练/测试集无患者重叠
- ✅ **类别不平衡处理**：SMOTE过采样 + 代价敏感权重
- ✅ **实验记录**：每次训练自动追加结果到CSV，配套可视化脚本
- ✅ **可解释性**：Grad-CAM热力图，直观展示模型对QRS波群形态异常的敏感度，辅助医生理解算法判断依据，从"黑盒"向"白盒"迈进
- ✅ **风险预警**：5级误诊风险预警机制
- ✅ **Web界面**：FastAPI后端 + HTML前端，支持文件上传和结果查看

---

## 🚀 快速开始

### 环境要求

- Python 3.9+，conda环境名 `ai`
- SQLite 3（已内置）
- 8GB+ RAM（推荐）
- GPU（可选，用于DL模型训练）

### 安装

```bash
conda activate ai
pip install -r requirements.txt
```

### 初始化数据库

```bash
python scripts/init_admin.py
# 默认账号：admin / admin123
```

### 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问：http://localhost:8000/index.html | API文档：http://localhost:8000/docs

---

## 🔬 数据处理流程

### 数据集

MIT-BIH心律失常数据库，27个患者记录，采样率360Hz。

```
记录列表（27个）：
100-109, 111-119, 121-124, 200-202, 209
```

### 标签映射

```python
label_map = {
    'N': 0, 'L': 0, 'R': 0,           # 类别0：正常（窦性/左束支/右束支）
    'V': 1, '/': 1,                     # 类别1：室性早搏
    'A': 2, 'F': 2, 'f': 2, 'j': 2,   # 类别2：其他异常
    'E': 2, 'e': 2, 'S': 2, 'a': 2,   # 室性逸搏、房性逸搏、室上性早搏等
    'J': 2, 'n': 2, 'Q': 2,            # 交界性早搏、结性逸搏、未分类
}
```

原始类别分布：正常~84%，室性早搏~15%，其他异常~1%

### 特征工程（41维）

每个心拍截取R峰前后各500个采样点（约2.8秒）：

**时域特征（16维）**：
- 统计量：均值、标准差、最大值、最小值、中位数、极差、偏度、峰度
- 心率特征：HR均值/标准差、SDNN、RMSSD、pNN50、RR不规则度
- 形态特征：R波幅度、R峰数量

**频域特征（7维）**（针对单心拍0.5-40Hz优化）：
- 基于单心拍波形的低频成分分析（0.5-5Hz），侧重捕捉P/T波的能量重心偏移
- QRS波段功率（5-20Hz）、高频段功率（20-40Hz）
- 功率比：QRS/PT功率比、QRS功率占比
- 主导频率与主导功率

**小波特征（18维）**：
- db4小波，5层分解，每层提取：能量、标准差、最大绝对值

### 数据划分策略

```
Patient-wise Split（按患者划分）：
  训练集：~22个患者（80%）
  测试集：~5个患者（20%）
  ✅ 训练/测试集无患者重叠，避免数据泄露
```

### 类别不平衡处理

```
1. SMOTE过采样（smote_ratio=1.0）
   → 训练集三类样本数量均衡（1:1:1）

2. 代价敏感权重（class_weight={0:1, 1:5, 2:20}）
   → 对少数类预测错误施加更大惩罚
```

---

## 🤖 模型说明

### 传统机器学习模型（4个）

| 模型 | 文件 | 说明 |
|------|------|------|
| XGBoost | xgboost_model.pkl | 梯度提升树，综合最优 |
| LightGBM | lightgbm_model.pkl | 轻量级梯度提升 |
| CatBoost | catboost_model.pkl | 类别特征优化 |
| RandomForest | randomforest_model.pkl | 对少数类识别弱 |

> 模型用 joblib 保存，推理时间 <1秒

### 深度学习模型（6个）

| 模型 | 文件 | 架构说明 |
|------|------|---------|
| ResNet-1D | resnet1d_best.pth | 18层残差网络 |
| SE-ResNet-1D | seresnet1d_best.pth | 残差网络 + SE通道注意力 |
| BiLSTM | bilstm_best.pth | 双向LSTM，捕捉时序依赖 |
| TCN | tcn_best.pth | 时序卷积网络，膨胀因果卷积 |
| Inception | inception_best.pth | 多尺度并行卷积 |
| Transformer | transformer_best.pth | 自注意力机制 |

> DL模型在Windows GPU上训练后迁移至本机，CPU推理约30秒

### 🔍 Grad-CAM 可解释性

本系统通过热力图直观展示模型对QRS波群形态异常的敏感度，辅助医生理解算法判断依据，从"黑盒"向"白盒"迈进。

```bash
# 生成Grad-CAM热力图
python scripts/test_grad_cam.py
# 输出：experiments/results/grad_cam_*.png
```

热力图说明：红色区域为模型判断的关键依据区域，蓝色为次要区域。若模型关注点精准落在异常R峰或QRS波群上，说明模型学到了有临床意义的特征。

---

## 📈 优化历程

实验记录保存在 `experiments/results/hyperparam_search.csv`（33条记录）。

| 阶段 | 关键变更 | XGBoost宏F1 |
|------|---------|------------|
| S1：基线 | Beat-wise划分（数据泄露），30000样本 | 0.38（虚高） |
| S2：修复泄露 | Patient-wise划分，60000样本，27患者 | 0.45 |
| S3：SMOTE | 加入SMOTE过采样（balanced权重） | 0.45 |
| S4：代价权重 | class_weight={0:1,1:4,2:10}，200棵树 | 0.53 |
| S5：修正频带 | 频域特征从HRV频带改为单心拍频带（0.5-40Hz） | 0.53 |
| S6：LinearSVC | 尝试LinearSVC（73秒未收敛，已弃用） | — |
| S7：Stacking | XGBoost粗筛+CatBoost复核（其他异常退化） | 0.48 |
| S8：扩充标签 | label_map加入E/e/S/a/J/n/Q，权重{0:1,1:5,2:20} | 0.53 |

```bash
# 生成优化历程可视化图表
python scripts/plot_hyperparam_results.py
# 输出：experiments/results/optimization_journey.png
```

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                  前端层（Frontend）                      │
│  HTML5 + Chart.js 波形渲染 + JWT认证                    │
└─────────────────────────────────────────────────────────┘
                         ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│                  后端层（Backend）                       │
│  FastAPI + Tortoise ORM + SQLite                        │
│  异步任务处理 / 文件验证 / JWT认证                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  算法层（Algorithm）                     │
│  ├─ 特征提取：41维（时域16 + 频域7 + 小波18）           │
│  ├─ 传统ML：RF / XGBoost / LightGBM / CatBoost         │
│  ├─ 深度学习：ResNet1D / SEResNet1D / BiLSTM /          │
│  │            TCN / Inception / Transformer             │
│  └─ 多模态融合：加权集成                                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  数据层（Database）                      │
│  SQLite + Tortoise ORM（异步）                          │
└─────────────────────────────────────────────────────────┘
```

**技术栈**：
- 后端：FastAPI 0.104.1 / SQLite / PyTorch 2.0+ / Scikit-learn / WFDB 4.1.2
- 信号处理：SciPy（Butterworth滤波）/ PyWavelets（db4小波）/ NumPy FFT
- 类别平衡：imbalanced-learn（SMOTE）
- 前端：HTML5 + CSS3 + Chart.js 4.x

---

## 📁 项目结构

```
ecg-system/
├── app/                          # 后端应用
│   ├── main.py                   # FastAPI入口
│   ├── api/                      # API路由（auth/users/ecg）
│   ├── services/                 # 业务逻辑（ecg_service/report_service）
│   ├── algorithms/               # 算法模块
│   │   ├── reader.py             # WFDB/CSV数据读取
│   │   ├── preprocess.py         # 信号预处理
│   │   ├── features.py           # 特征提取
│   │   ├── inference.py          # 推理引擎
│   │   ├── multimodal_fusion.py  # 多模态融合（41维特征对齐）
│   │   ├── deep_models.py        # 深度学习模型定义
│   │   ├── grad_cam.py           # Grad-CAM可解释性
│   │   └── models/               # 预训练模型文件
│   │       ├── *_model.pkl       # ML模型（joblib格式）
│   │       ├── scaler.pkl        # StandardScaler
│   │       └── *_best.pth        # DL模型（PyTorch格式）
│   ├── models/                   # 数据库ORM模型
│   ├── schemas/                  # Pydantic模型
│   └── core/                     # 安全/日志/异常/风险预警
│
├── scripts/                      # 工具脚本
│   ├── train_traditional_ml.py   # 训练ML模型（主训练脚本）
│   ├── train_multimodal_models.py # 训练DL模型
│   ├── hyperparam_search.py      # 超参数搜索
│   ├── plot_hyperparam_results.py # 可视化实验结果
│   ├── visualize_results.py      # 混淆矩阵/模型对比图
│   ├── test_noise_robustness.py  # 抗干扰测试
│   ├── test_risk_warning.py      # 风险预警测试
│   ├── ablation_study.py         # 消融实验
│   └── test_grad_cam.py          # Grad-CAM测试
│
├── data/                         # MIT-BIH数据集（27个患者）
│   └── {100-209}.{dat,hea,atr}
│
├── experiments/results/          # 实验结果
│   ├── hyperparam_search.csv     # 实验记录（33条）
│   ├── ml_results_patient_wise.json
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── f1_per_class.png
│   ├── optimization_journey.png
│   └── noise_robustness_results.json
│
├── tests/                        # 系统检查脚本（7层）
├── frontend/                     # Web前端
├── logs/                         # 运行日志
├── ecg_system.db                 # SQLite数据库
└── requirements.txt
```

---

## 🔧 模型训练

### 训练传统ML模型

```bash
conda activate ai
python scripts/train_traditional_ml.py
```

训练完成后自动生成：
- `app/algorithms/models/{randomforest,xgboost,lightgbm,catboost}_model.pkl`
- `app/algorithms/models/scaler.pkl`
- 结果追加到 `experiments/results/hyperparam_search.csv`

### 训练深度学习模型

```bash
# 建议在GPU机器上运行（CPU约2-3小时）
python scripts/train_multimodal_models.py
```

### 超参数搜索

```bash
python scripts/hyperparam_search.py
python scripts/plot_hyperparam_results.py
```

---

## 🧪 实验与测试

```bash
# 混淆矩阵 / 模型对比 / F1对比图
python scripts/visualize_results.py

# 抗干扰测试（高斯噪声/工频干扰/基线漂移/EMG）
python scripts/test_noise_robustness.py

# 误诊风险预警测试（6个案例）
python scripts/test_risk_warning.py

# 消融实验（单模型 vs 集成）
python scripts/ablation_study.py

# Grad-CAM可解释性测试
python scripts/test_grad_cam.py
```

---

## 🐛 故障排查

- 模型加载失败：检查 `app/algorithms/models/` 下是否有 `.pkl` 和 `.pth` 文件
- 推理速度慢：DL模型CPU推理约30秒，只用ML模型可<1秒
- 内存不足：10个模型同时加载约需2GB RAM
- 数据库不存在：运行 `python scripts/init_admin.py` 初始化

---

## 🎯 开发进度

- [x] 后端框架（FastAPI + SQLite）
- [x] 用户认证（JWT）
- [x] 数据预处理与特征提取（41维）
- [x] Patient-wise Split（修复数据泄露）
- [x] 传统ML模型训练（4个模型）
- [x] 深度学习模型训练（6个模型）
- [x] 类别不平衡处理（SMOTE + 代价敏感权重）
- [x] 超参数搜索与实验记录（33条历史）
- [x] 多模态融合引擎（特征维度对齐）
- [x] Grad-CAM可解释性
- [x] 误诊风险预警（5级）
- [x] 抗干扰测试
- [x] 消融实验
- [ ] 提升类别2（其他异常）识别率（进行中）
- [ ] Grad-CAM集成到Web报告

---

## 📚 参考资料

- [MIT-BIH心律失常数据库](https://physionet.org/content/mitdb/)
- [Patient-wise Split in ECG Classification](https://ieeexplore.ieee.org/document/8952526)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

---

## 📄 许可证

本项目仅供学习研究使用。

---

<div align="center">

**版本**：v2.1.0 | **最后更新**：2026年3月14日 | **状态**：�� 数据处理优化中

</div>
