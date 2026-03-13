# 🫀 ECG智能心电图分析系统

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**基于多模态深度学习的智能心电图分析系统**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [技术架构](#-技术架构) • [模型说明](#-模型说明)

</div>

---

## 📋 项目简介

ECG智能心电图分析系统是一套高精度的心电图自动分析系统，采用多模态深度融合架构，结合传统机器学习与深度学习技术，实现心电图自动分析、风险评估和智能报告生成。

### 🎯 核心特点

- **🧠 多模态融合**：传统ML（5模型）+ 深度学习（5模型）双引擎驱动
- **📊 高准确率**：置信度达80%+，支持3类心律失常识别
- **⚡ 实时分析**：端到端分析时间<30秒
- **🔒 数据安全**：本地部署，数据不出域
- **📱 易用界面**：Web界面，支持多种ECG格式

### 📊 当前性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 模型数量 | 10个 | 5个ML + 5个DL |
| 置信度 | 80%+ | 多模态融合后 |
| 推理时间 | ~30秒 | CPU环境 |
| 支持类别 | 3类 | 正常/室性早搏/其他异常 |
| 模型大小 | 89MB | 全部模型总计 |

---

## ✨ 功能特性

### 核心功能

- ✅ **多格式支持**：CSV、DAT（MIT-BIH）、WFDB等格式
- ✅ **智能预处理**：基线漂移去除、滤波、自动归一化
- ✅ **多模态分析**：
  - 时域模态：5个传统ML模型（RF、XGBoost、LightGBM、CatBoost、SVM）
  - 深度学习模态：5个DL模型（ResNet-1D、SE-ResNet-1D、BiLSTM、TCN、Inception）
  - 频域模态：小波变换 + SVM（规则推理）
  - 图神经网络模态：GCN、GAT（规则推理）
  - 规则引擎：医学先验知识
- ✅ **3类心律识别**：
  - 正常窦性心律（类别0）
  - 室性早搏（类别1）
  - 其他异常（类别2）
- ✅ **实时监控**：实时心电波形显示、动态趋势图
- ✅ **历史回溯**：任务历史查询、波形对比
- ✅ **智能报告**：PDF诊断报告生成
- ✅ **用户管理**：JWT认证、角色权限、数据隔离

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- SQLite 3（已内置）
- 8GB+ RAM（推荐）
- GPU（可选，用于模型训练）

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/ecg-system.git
cd ecg-system
```

### 2. 安装依赖

```bash
# 创建conda环境（推荐）
conda create -n ai python=3.9
conda activate ai

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 初始化数据库

```bash
# 创建管理员账号（会自动创建数据库和表）
python scripts/init_admin.py
```

默认管理员账号：
- 用户名：`admin`
- 密码：`admin123`

### 4. 启动服务

```bash
# 开发模式（自动重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 访问系统

- **前端界面**：http://localhost:8000/index.html
- **登录页面**：http://localhost:8000/login.html
- **API文档**：http://localhost:8000/docs

### 6. 测试系统

系统提供了3个测试ECG数据文件：

```bash
test_data/
├── normal_ecg.csv          # 正常心率 (~75 bpm)
├── bradycardia_ecg.csv     # 心动过缓 (~50 bpm)
└── tachycardia_ecg.csv     # 心动过速 (~110 bpm)
```

**使用Web界面测试**：
1. 登录系统（admin/admin123）
2. 点击上传区域，选择测试文件
3. 等待分析完成（约30秒），查看结果

---

## 🏗️ 技术架构

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  前端层（Frontend）                      │
│  • 实时心电监控（Chart.js波形渲染）                     │
│  • 历史任务管理                                          │
│  • 用户认证与权限                                        │
└─────────────────────────────────────────────────────────┘
                         ↓ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│                  后端层（Backend）                       │
│  FastAPI + Tortoise ORM + SQLite                        │
│  • 异步任务处理                                          │
│  • 文件验证与存储                                        │
│  • JWT认证                                               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  算法层（Algorithm）                     │
│  多模态深度融合引擎                                      │
│  ├─ 时域模态：5个ML模型（78%置信度）                   │
│  ├─ 深度学习模态：5个DL模型（96%置信度）               │
│  ├─ 频域模态：小波变换 + SVM（规则）                   │
│  ├─ 图网络模态：GCN/GAT（规则）                        │
│  └─ 规则引擎：医学先验知识                              │
│                                                          │
│  融合策略：45%时域 + 45%深度 + 10%其他                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  数据层（Database）                      │
│  SQLite + Tortoise ORM（异步）                          │
│  • 用户表（users）                                       │
│  • 任务表（ecg_tasks）                                  │
└─────────────────────────────────────────────────────────┘
```

### 技术栈

**后端技术**：
- FastAPI 0.104.1 - 异步Web框架
- SQLite + Tortoise ORM 0.20.0 - 数据库与异步ORM
- PyTorch 2.0+ - 深度学习框架
- Scikit-learn - 机器学习算法
- NumPy + SciPy - 科学计算
- WFDB 4.1.2 - 医疗波形数据处理

**前端技术**：
- HTML5 + CSS3 + JavaScript ES6+
- Chart.js 4.x - 波形可视化
- Glassmorphism设计 - 毛玻璃效果

**算法技术**：
- 深度学习：ResNet-1D、SE-ResNet-1D、BiLSTM、TCN、Inception
- 传统机器学习：RandomForest、XGBoost、LightGBM、CatBoost、SVM
- 信号处理：Butterworth滤波、小波变换、Pan-Tompkins算法

---

## 🤖 模型说明

### 已训练模型（10个）

#### 传统机器学习模型（5个）

| 模型 | 文件大小 | 训练日期 | 准确率 | 说明 |
|------|---------|---------|--------|------|
| RandomForest | 617KB | 2026-03-13 | 99% | 随机森林，100棵树 |
| XGBoost | 390KB | 2026-03-13 | 99% | 梯度提升树 |
| LightGBM | 768KB | 2026-03-13 | 99% | 轻量级梯度提升 |
| CatBoost | 219KB | 2026-03-13 | 99% | 类别特征优化 |
| SVM | 54KB | 2026-03-13 | 99% | 支持向量机 |

**集成效果**：5模型投票，平均置信度78.21%

#### 深度学习模型（5个）

| 模型 | 文件大小 | 训练日期 | 准确率 | 说明 |
|------|---------|---------|--------|------|
| ResNet-1D | 34MB | 2026-03-13 | 99.6% | 18层残差网络 |
| SE-ResNet-1D | 34MB | 2026-03-13 | 99.6% | 带SE注意力机制 |
| BiLSTM | 2.1MB | 2026-03-13 | 99.9% | 双向LSTM |
| TCN | 3.6MB | 2026-03-13 | 98.5% | 时序卷积网络 |
| Inception | 2.7MB | 2026-03-13 | 99.0% | 多尺度卷积 |

**集成效果**：5模型投票，平均置信度95.81%

#### 待训练模型（1个）

| 模型 | 状态 | 说明 |
|------|------|------|
| Transformer | 待训练 | CPU训练较慢，暂时跳过 |

### 多模态融合策略

```python
# 当前融合权重配置
融合权重 = {
    '时域模态（ML）': 45%,      # 5个传统ML模型
    '深度学习模态（DL）': 45%,   # 5个深度学习模型
    '频域模态': 5%,              # 小波变换（规则）
    '图网络模态': 2%,            # GCN/GAT（规则）
    '规则引擎': 3%               # 医学先验知识
}

# 最终诊断 = 加权融合
最终置信度 = 0.45 × ML置信度 + 0.45 × DL置信度 + 0.10 × 其他
```

### 性能对比

| 方法 | 置信度 | 推理时间 | 说明 |
|------|--------|----------|------|
| 仅传统ML | 78.21% | <1秒 | 5模型集成 |
| 仅深度学习 | 95.81% | ~30秒 | 5模型集成 |
| **多模态融合** | **80.02%** | **~30秒** | 10模型融合 |

---

## 📁 项目结构

```
ecg-system/
├── app/                          # 后端应用
│   ├── main.py                   # FastAPI入口
│   ├── config.py                 # 配置文件（SQLite）
│   ├── api/                      # API路由层
│   │   ├── auth.py               # 认证接口
│   │   ├── users.py              # 用户管理
│   │   └── ecg.py                # ECG分析接口
│   ├── services/                 # 业务逻辑层
│   │   ├── ecg_service.py        # ECG分析服务
│   │   └── report_service.py     # 报告生成服务
│   ├── algorithms/               # 算法模块
│   │   ├── reader.py             # 数据读取
│   │   ├── preprocess.py         # 预处理
│   │   ├── features.py           # 特征提取
│   │   ├── inference.py          # 推理引擎
│   │   ├── multimodal_fusion.py  # 多模态融合引擎 ⭐
│   │   ├── deep_models.py        # 深度学习模型定义
│   │   ├── graph_models.py       # 图神经网络模型
│   │   └── models/               # 预训练模型文件（89MB）
│   │       ├── randomforest_model.pkl
│   │       ├── xgboost_model.pkl
│   │       ├── lightgbm_model.pkl
│   │       ├── catboost_model.pkl
│   │       ├── svm_model.pkl
│   │       ├── scaler.pkl
│   │       ├── resnet1d_best.pth
│   │       ├── seresnet1d_best.pth
│   │       ├── bilstm_best.pth
│   │       ├── tcn_best.pth
│   │       └── inception_best.pth
│   ├── models/                   # 数据库模型
│   │   ├── user.py               # 用户模型
│   │   └── task.py               # 任务模型
│   ├── schemas/                  # Pydantic模型
│   ├── core/                     # 核心模块
│   │   ├── security.py           # JWT认证
│   │   ├── logger.py             # 日志系统
│   │   ├── exceptions.py         # 异常处理
│   │   └── validators.py         # 文件验证
│   └── db/                       # 数据库配置
│
├── frontend/                     # 前端界面
│   ├── index.html                # 主页面
│   ├── login.html                # 登录页面
│   └── register.html             # 注册页面
│
├── scripts/                      # 工具脚本
│   ├── init_admin.py             # 初始化管理员
│   ├── train_traditional_ml.py   # 训练ML模型 ⭐
│   ├── train_multimodal_models.py # 训练DL模型 ⭐
│   ├── train_interactive.py      # 交互式训练
│   ├── visualize_results.py      # 可视化结果 ⭐
│   ├── test_risk_warning.py      # 误诊预警测试 ⭐
│   ├── test_noise_robustness.py  # 抗干扰测试 ⭐
│   ├── ablation_study.py         # 消融实验 ⭐
│   ├── test_grad_cam.py          # Grad-CAM测试
│   ├── generate_test_data.py     # 生成测试数据
│   └── TRAINING_GUIDE.md         # 训练指南
│
├── data/                         # 数据目录
│   ├── 100.dat/hea/atr           # MIT-BIH数据集
│   └── *.csv                     # 上传的ECG文件
│
├── test_data/                    # 测试数据
│   ├── normal_ecg.csv
│   ├── bradycardia_ecg.csv
│   └── tachycardia_ecg.csv
│
├── tests/                        # 系统检查脚本 ⭐
│   ├── check_layer1_database.py
│   ├── check_layer2_models.py
│   ├── check_layer3_algorithms.py
│   ├── check_layer4_services.py
│   ├── check_layer5_api.py
│   ├── check_layer6_frontend.py
│   ├── check_layer7_integration.py
│   ├── run_all_checks.sh         # 一键运行所有检查
│   └── quick_test.py
│
├── experiments/                  # 实验结果 ⭐
│   └── results/
│       ├── confusion_matrices.png
│       ├── model_comparison.png
│       ├── f1_per_class.png
│       ├── training_time.png
│       ├── snr_accuracy_curve.png
│       ├── noise_types_comparison.png
│       ├── ablation_*.png
│       └── *.json                # 结果数据
│
├── logs/                         # 日志文件
│   ├── server.log
│   └── error.log
│
├── ecg_system.db                 # SQLite数据库
├── requirements.txt              # Python依赖
├── .env.example                  # 环境变量模板
├── IMPROVEMENTS.md               # 改进说明 ⭐
└── README.md                     # 本文件
```

---

## 📖 使用指南

### API接口

#### 1. 用户登录

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

响应：
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### 2. 上传ECG文件

```bash
curl -X POST http://localhost:8000/api/ecg/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test_data/normal_ecg.csv"
```

响应：
```json
{
  "id": 1,
  "filename": "normal_ecg.csv",
  "status": "pending",
  "created_at": "2026-03-13T10:00:00"
}
```

#### 3. 查询任务结果

```bash
curl http://localhost:8000/api/ecg/tasks/1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

响应：
```json
{
  "id": 1,
  "filename": "normal_ecg.csv",
  "status": "completed",
  "result": {
    "heart_rate": 75.5,
    "hrv_sdnn": 45.2,
    "diagnosis": "正常窦性心律",
    "confidence": 80.02,
    "risk_level": "低风险"
  }
}
```

完整API文档：http://localhost:8000/docs

---

## 🔧 模型训练

### 训练传统ML模型

```bash
# 使用MIT-BIH数据集训练（30000样本，15个患者）
python scripts/train_traditional_ml.py

# 训练完成后会生成5个模型文件：
# - randomforest_model.pkl
# - xgboost_model.pkl
# - lightgbm_model.pkl
# - catboost_model.pkl
# - svm_model.pkl
```

### 训练深度学习模型

```bash
# 训练所有深度学习模型（约2-3小时，CPU）
python scripts/train_multimodal_models.py

# 训练完成后会生成6个模型文件：
# - resnet1d_best.pth
# - seresnet1d_best.pth
# - bilstm_best.pth
# - tcn_best.pth
# - inception_best.pth
# - transformer_best.pth
```

详细训练指南：[scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)

---

## 🧪 实验与测试

### 可视化结果

```bash
# 生成混淆矩阵、模型对比图等
python scripts/visualize_results.py
```

生成的图表：
- `confusion_matrices.png` - 5个ML模型的混淆矩阵
- `model_comparison.png` - 模型性能对比
- `f1_per_class.png` - 各类别F1-Score对比
- `training_time.png` - 训练时间对比

### 误诊风险预警测试

```bash
# 测试误诊风险预警机制（6个测试案例）
python scripts/test_risk_warning.py
```

预警级别：
- 安全：无需预警，可直接使用
- 低风险：建议关注
- 中风险：需要人工复核
- 高风险：强烈建议人工复核
- 严重风险：必须人工复核

### 抗干扰能力测试

```bash
# 测试模型在不同噪声下的性能
python scripts/test_noise_robustness.py
```

测试内容：
- 高斯白噪声（6个不同SNR）
- 工频干扰（50Hz）
- 基线漂移
- 肌电干扰（EMG）
- 混合噪声

生成图表：
- `snr_accuracy_curve.png` - SNR-准确率曲线
- `noise_types_comparison.png` - 噪声类型对比

### 消融实验

```bash
# 对比单一模型 vs 集成模型
python scripts/ablation_study.py
```

实验内容：
- 单个模型性能评估
- 不同集成方法对比（投票法、平均法、加权平均）
- 增量实验（逐步添加模型的效果）

生成图表：
- `ablation_single_vs_ensemble.png` - 单个vs集成对比
- `ablation_incremental.png` - 增量实验曲线
- `ablation_improvement.png` - 性能提升图

---

## 🐛 故障排查

### 常见问题

**Q1: 模型加载失败**
```bash
# 检查模型文件是否存在
ls -lh app/algorithms/models/

# 应该看到12个文件（10个模型 + 1个scaler + 1个旧版rf_model）
```

**Q2: 数据库连接失败**
```bash
# 检查SQLite数据库文件
ls -lh ecg_system.db

# 如果不存在，运行初始化脚本
python scripts/init_admin.py
```

**Q3: 推理速度慢**
- 深度学习模型在CPU上推理较慢（~30秒）
- 建议使用GPU加速（需要安装CUDA）
- 或者只使用传统ML模型（<1秒）

**Q4: 内存不足**
- 10个模型同时加载需要约2GB内存
- 建议至少8GB RAM
- 可以修改融合引擎，只加载部分模型

---

## 📊 系统检查

项目提供了7层系统检查脚本（位于 `tests/` 目录）：

```bash
# 一键运行所有检查
bash tests/run_all_checks.sh

# 或单独运行各层检查
python tests/check_layer1_database.py    # 第1层：数据库层
python tests/check_layer2_models.py      # 第2层：数据模型层
python tests/check_layer3_algorithms.py  # 第3层：算法层
python tests/check_layer4_services.py    # 第4层：服务层
python tests/check_layer5_api.py         # 第5层：API层
python tests/check_layer6_frontend.py    # 第6层：前端层（需要启动服务器）
python tests/check_layer7_integration.py # 第7层：集成测试（需要启动服务器）
```

**注意**：第6-7层检查需要先启动服务器。

---

## 📚 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [改进说明](IMPROVEMENTS.md)
- [算法模块说明](app/algorithms/README.md)
- [训练指南](scripts/TRAINING_GUIDE.md)

---

## 🎯 开发进度

- [x] 后端框架搭建（FastAPI + SQLite）
- [x] 前端界面开发（HTML + CSS + JS）
- [x] 用户认证系统（JWT）
- [x] 文件上传与验证
- [x] 数据预处理模块
- [x] 特征提取模块
- [x] 传统ML模型训练（5个模型，99%准确率）
- [x] 深度学习模型训练（5个模型，99%+准确率）
- [x] 多模态融合引擎（80%+置信度）
- [x] PDF报告生成
- [x] 系统集成测试
- [ ] Transformer模型训练（待完成）
- [ ] 联邦学习框架（计划中）
- [ ] 可解释性模块（计划中）

**当前完成度**：约85%

---

## 🙏 致谢

- MIT-BIH心律失常数据库
- PhysioNet数据平台
- FastAPI框架
- PyTorch深度学习框架

---

## 📄 许可证

本项目仅供学习研究使用。

---

<div align="center">

**版本**：v2.0.0  
**最后更新**：2026年3月13日  
**状态**：🚀 积极开发中

Made with ❤️ by ECG System Team

</div>
