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

**传统机器学习（ML）**

| 模型 | 准确率 | 宏F1 | 室早F1 | 其他异常F1 |
|------|--------|------|--------|-----------|
| XGBoost | 0.9584 | 0.5310 | 0.4603 | 0.1406 |
| LightGBM | 0.9722 | 0.4298 | 0.2568 | 0.0465 |
| CatBoost | 0.9790 | 0.4470 | 0.2782 | 0.0732 |
| Stacking(XGB+Cat) | 0.9830 | 0.4440 | 0.3128 | 0.0278 |

> 60000样本，27个患者，Patient-wise Split，SMOTE(1:1:1) + class_weight={0:1, 1:5, 2:20}

**深度学习（DL）**

| 模型 | 准确率 | 宏F1 | 室早F1 | 其他异常F1 |
|------|--------|------|--------|-----------|
| ResNet-1D | — | — | — | — |
| SE-ResNet-1D | — | — | — | — |
| BiLSTM | — | — | — | — |
| TCN | — | — | — | — |
| Inception | — | — | — | — |
| Transformer | — | — | — | — |

> DL模型待GPU机器训练完成后补全

在严谨的 Patient-wise 评估下，Stacking(XGB+CatBoost) 作为 ML 核心引擎，XGBoost 负责高召回粗筛，CatBoost 负责精确率复核，实现双端互证。

### ⚠️ 局限性说明

类别 2（其他异常）包含房颤、逸搏等多种形态迥异的亚类，受限于 MIT-BIH 数据库中该类别样本极稀缺（<1%），目前模型主要起"预警"作用而非精准诊断。未来计划通过迁移学习引入更大规模的 CPSC 或 PTB-XL 数据库进行跨库训练。

---

## ✨ 功能特性

- ✅ **多格式支持**：CSV、DAT（MIT-BIH WFDB格式）
- ✅ **信号预处理**：Butterworth带通滤波、基线漂移去除、Pan-Tompkins R峰检测
- ✅ **41维特征工程**：时域16维 + 频域7维 + 小波18维（db4，5层分解）
- ✅ **3类心律识别**：正常（N/L/R）、室性早搏（V/融合拍）、其他异常（A/F/E/S/J等）
- ✅ **多模型集成**：3个传统ML模型 + Stacking集成 + 6个深度学习模型
- ✅ **严格评估**：Patient-wise Split，训练/测试集无患者重叠
- ✅ **类别不平衡处理**：SMOTE过采样 + 代价敏感权重
- ✅ **实验记录**：每次训练自动追加结果到CSV，配套可视化脚本
- ✅ **可解释性**：Grad-CAM热力图，辅助医生理解算法判断依据
- ✅ **风险预警**：5级误诊风险预警机制 + ML/DL双端互证
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

### 下载数据集

MIT-BIH 数据集不包含在仓库中，需手动下载：

**方式一：wfdb 自动下载（推荐）**
```bash
python -c "
import wfdb
records = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','209'
]
for r in records:
    wfdb.dl_database('mitdb', dl_dir='data', records=[r])
    print(f'下载完成: {r}')
"
```

**方式二：PhysioNet 官网手动下载**
```
https://physionet.org/content/mitdb/1.0.0/
```
下载后将所有 `.dat` / `.hea` / `.atr` 文件放入 `data/` 目录。

**验证下载**
```bash
ls data/*.dat | wc -l   # 应输出 27
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
- 低频成分（0.5-5Hz）、QRS波段功率（5-20Hz）、高频段功率（20-40Hz）
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

### 传统机器学习模型

| 模型 | 文件 | 说明 |
|------|------|------|
| XGBoost | xgboost_model.pkl | 梯度提升树，高召回粗筛 |
| LightGBM | lightgbm_model.pkl | 轻量级梯度提升 |
| CatBoost | catboost_model.pkl | 高精确率复核 |
| Stacking(XGB+Cat) | — | XGBoost粗筛+CatBoost复核，ML核心引擎 |

> 模型用 joblib 保存，推理时间 <1秒

### 深度学习模型（6个）

| 模型 | 文件 | 架构说明 |
|------|------|---------|
| ResNet-1D | resnet1d_best.pth | 残差网络 + Dropout(0.5) |
| SE-ResNet-1D | seresnet1d_best.pth | 残差网络 + SE通道注意力 |
| BiLSTM | bilstm_best.pth | 双向LSTM + Bahdanau注意力 |
| TCN | tcn_best.pth | 时序卷积网络，真因果卷积（只pad左边） |
| Inception | inception_best.pth | 多尺度并行卷积（1/3/5/7核） |
| Transformer | transformer_best.pth | 步长卷积下采样(1000→125) + 自注意力 |

> DL模型建议在GPU机器上训练，CPU推理约30秒

### 🔍 Grad-CAM 可解释性

```bash
python scripts/test_grad_cam.py
# 输出：experiments/results/grad_cam_*.png
```

红色区域为模型判断的关键依据区域。若关注点精准落在异常QRS波群上，说明模型学到了有临床意义的特征。

---

## 📈 优化历程

实验记录保存在 `experiments/results/hyperparam_search.csv`。

| 阶段 | 关键变更 | XGBoost宏F1 |
|------|---------|------------|
| S1：基线 | Beat-wise划分（数据泄露），30000样本 | 0.38（虚高） |
| S2：修复泄露 | Patient-wise划分，60000样本，27患者 | 0.45 |
| S3：SMOTE | 加入SMOTE过采样 | 0.45 |
| S4：代价权重 | class_weight={0:1,1:4,2:10}，200棵树 | 0.53 |
| S5：修正频带 | 频域特征改为单心拍频带（0.5-40Hz） | 0.53 |
| S6：LinearSVC | 尝试LinearSVC（73秒未收敛，已弃用） | — |
| S7：Stacking | XGBoost粗筛+CatBoost复核 | 0.48 |
| S8：扩充标签 | label_map加入E/e/S/a/J/n/Q，权重{0:1,1:5,2:20} | 0.53 |

```bash
python scripts/plot_hyperparam_results.py
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
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  算法层（Algorithm）                     │
│  ├─ 特征提取：41维（时域16 + 频域7 + 小波18）           │
│  ├─ Stacking层：XGBoost粗筛 + CatBoost复核             │
│  ├─ 深度学习：ResNet1D / SEResNet1D / BiLSTM /          │
│  │            TCN / Inception / Transformer             │
│  └─ 双端互证层：ML vs DL 一致性检验 + 风险预警          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  数据层（Database）                      │
│  SQLite + Tortoise ORM（异步）                          │
└─────────────────────────────────────────────────────────┘
```

**技术栈**：
- 后端：FastAPI 0.104.1 / SQLite / PyTorch 2.0+ / Scikit-learn / WFDB 4.1.2
- 信号处理：SciPy / PyWavelets（db4小波）/ NumPy FFT
- 类别平衡：imbalanced-learn（SMOTE）
- 前端：HTML5 + CSS3 + Chart.js 4.x

---

## 📁 项目结构

```
ecg-system/
├── app/
│   ├── algorithms/
│   │   ├── deep_models.py        # DL模型定义（6个）
│   │   ├── multimodal_fusion.py  # 多模态融合 + 双端互证
│   │   ├── features.py           # 41维特征提取
│   │   ├── inference.py          # 推理引擎
│   │   ├── grad_cam.py           # Grad-CAM可解释性
│   │   └── models/               # 模型文件
│   │       ├── xgboost_model.pkl / lightgbm_model.pkl / catboost_model.pkl
│   │       ├── scaler.pkl
│   │       └── *_best.pth        # DL模型
│   ├── api/                      # FastAPI路由
│   ├── core/                     # 安全/日志/风险预警
│   └── main.py
├── scripts/
│   ├── train_traditional_ml.py   # 训练ML（XGB/LGB/CatBoost/Stacking）
│   ├── train_multimodal_models.py # 训练DL（6个模型）
│   ├── eval_dl_models.py         # 评估DL + ML vs DL对比图
│   ├── plot_hyperparam_results.py # 优化历程可视化
│   └── ...
├── data/                         # MIT-BIH数据集（需手动下载，见上方说明）
├── experiments/results/          # 实验结果与图表
├── frontend/                     # Web前端
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
- `app/algorithms/models/{xgboost,lightgbm,catboost}_model.pkl`
- `app/algorithms/models/scaler.pkl`
- 结果追加到 `experiments/results/hyperparam_search.csv`

### 训练深度学习模型

```bash
# 建议在GPU机器上运行（CPU约2-3小时）
conda activate ai
python scripts/train_multimodal_models.py
```

### 评估DL模型 + 生成ML vs DL对比图

```bash
python scripts/eval_dl_models.py
```

---

## 🧪 实验与测试

```bash
# 抗干扰测试（高斯噪声/工频干扰/基线漂移/EMG）
python scripts/test_noise_robustness.py

# 误诊风险预警测试
python scripts/test_risk_warning.py

# 消融实验
python scripts/ablation_study.py

# Grad-CAM可解释性
python scripts/test_grad_cam.py
```

---

## 🐛 故障排查

- `data/` 目录为空：按上方"下载数据集"步骤操作
- 模型加载失败：检查 `app/algorithms/models/` 下是否有 `.pkl` 和 `.pth` 文件
- 推理速度慢：DL模型CPU推理约30秒，只用ML模型可<1秒
- 内存不足：所有模型同时加载约需2GB RAM
- 数据库不存在：运行 `python scripts/init_admin.py` 初始化

---

## 🎯 开发进度

- [x] 后端框架（FastAPI + SQLite）
- [x] 用户认证（JWT）
- [x] 数据预处理与特征提取（41维）
- [x] Patient-wise Split（修复数据泄露）
- [x] 传统ML模型训练（XGB/LGB/CatBoost/Stacking）
- [x] 深度学习模型训练（6个模型）
- [x] 类别不平衡处理（SMOTE + 代价敏感权重 + WeightedRandomSampler）
- [x] 多模态融合引擎 + ML/DL双端互证
- [x] Grad-CAM可解释性
- [x] 误诊风险预警（5级）
- [x] 抗干扰测试 / 消融实验
- [ ] DL模型性能补全（GPU训练中）
- [ ] 提升类别2（其他异常）识别率

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

**版本**：v2.1.0 | **最后更新**：2026年3月14日 | **状态**：🔬 DL模型训练中

</div>
