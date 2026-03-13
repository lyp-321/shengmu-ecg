# ECG系统项目结构

## 📁 目录结构

```
ecg-system/
├── app/                          # 后端应用核心
│   ├── main.py                   # FastAPI应用入口
│   ├── config.py                 # 配置管理
│   │
│   ├── api/                      # API路由层
│   │   ├── __init__.py
│   │   ├── auth.py               # 认证接口（登录/注册）
│   │   ├── users.py              # 用户管理接口
│   │   └── ecg.py                # ECG分析接口
│   │
│   ├── services/                 # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── ecg_service.py        # ECG分析服务
│   │   └── report_service.py     # PDF报告生成
│   │
│   ├── algorithms/               # 算法模块
│   │   ├── __init__.py
│   │   ├── README.md             # 算法说明文档
│   │   ├── reader.py             # 数据读取（CSV/DAT/WFDB）
│   │   ├── preprocess.py         # 信号预处理
│   │   ├── features.py           # 特征提取
│   │   ├── inference.py          # 推理引擎
│   │   ├── cnn_model.py          # CNN模型定义
│   │   ├── multimodal_fusion.py  # 多模态融合引擎
│   │   ├── deep_models.py        # 深度学习模型（ResNet/Transformer等）
│   │   ├── graph_models.py       # 图神经网络（GCN/GAT）
│   │   ├── grad_cam.py           # Grad-CAM可解释性
│   │   ├── federated_learning.py # 联邦学习框架
│   │   └── models/               # 预训练模型文件
│   │       ├── rf_model.pkl
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
│   │
│   ├── models/                   # 数据库模型
│   │   ├── __init__.py
│   │   ├── user.py               # 用户模型
│   │   └── task.py               # 任务模型
│   │
│   ├── schemas/                  # Pydantic数据模型
│   │   ├── __init__.py
│   │   ├── user.py               # 用户Schema
│   │   └── ecg.py                # ECG Schema
│   │
│   ├── core/                     # 核心模块
│   │   ├── __init__.py
│   │   ├── security.py           # 安全相关（JWT/密码）
│   │   ├── deps.py               # 依赖注入
│   │   ├── logger.py             # 日志系统
│   │   ├── exceptions.py         # 异常处理
│   │   ├── validators.py         # 文件验证
│   │   └── risk_warning.py       # 误诊风险预警
│   │
│   └── db/                       # 数据库配置
│       ├── __init__.py
│       ├── session.py            # 数据库连接
│       └── init_data.py          # 初始数据
│
├── frontend/                     # 前端界面
│   ├── index.html                # 主页面（监控+历史+配置）
│   ├── login.html                # 登录页面
│   ├── register.html             # 注册页面
│   └── index_enhanced.html       # 前端增强功能
│
├── scripts/                      # 工具脚本
│   ├── init_admin.py             # 初始化管理员账号
│   ├── generate_test_data.py     # 生成测试数据
│   ├── load_mitbih.py            # 加载MIT-BIH数据集
│   ├── convert_mitbih_to_csv.py  # 转换MIT-BIH为CSV
│   ├── migrate_db.py             # 数据库迁移
│   ├── train_traditional_ml.py   # 训练传统ML模型
│   ├── train_multimodal_models.py # 训练深度学习模型
│   ├── train_interactive.py      # 交互式训练界面
│   ├── train_all.sh              # 一键训练脚本
│   ├── visualize_results.py      # 可视化结果
│   ├── test_risk_warning.py      # 误诊风险预警测试
│   ├── test_noise_robustness.py  # 抗干扰能力测试
│   ├── ablation_study.py         # 消融实验
│   ├── test_grad_cam.py          # Grad-CAM测试
│   └── TRAINING_GUIDE.md         # 训练指南
│
├── data/                         # 数据目录
│   ├── reports/                  # PDF报告存储
│   ├── mitbih/                   # MIT-BIH数据集
│   ├── mitbih_csv/               # MIT-BIH CSV格式
│   ├── 100.dat/hea/atr           # MIT-BIH原始文件
│   └── *.csv                     # 上传的ECG文件
│
├── test_data/                    # 测试数据
│   ├── normal_ecg.csv            # 正常心率测试文件
│   ├── bradycardia_ecg.csv       # 心动过缓测试文件
│   └── tachycardia_ecg.csv       # 心动过速测试文件
│
├── tests/                        # 测试代码
│   ├── __init__.py
│   ├── test_algorithms.py        # 算法测试
│   ├── quick_test.py             # 快速测试
│   ├── check_layer1_database.py  # 第1层：数据库层检查
│   ├── check_layer2_models.py    # 第2层：数据模型层检查
│   ├── check_layer3_algorithms.py # 第3层：算法层检查
│   ├── check_layer4_services.py  # 第4层：服务层检查
│   ├── check_layer5_api.py       # 第5层：API层检查
│   ├── check_layer6_frontend.py  # 第6层：前端层检查
│   ├── check_layer7_integration.py # 第7层：集成测试
│   └── run_all_checks.sh         # 一键运行所有检查
│
├── experiments/                  # 实验对比
│   ├── compare_methods.py        # 方法对比脚本
│   └── results/                  # 实验结果
│       ├── method_comparison.png
│       ├── confusion_matrices.png
│       ├── model_comparison.png
│       ├── f1_per_class.png
│       ├── training_time.png
│       ├── snr_accuracy_curve.png
│       ├── noise_types_comparison.png
│       ├── ablation_*.png
│       ├── ml_results.txt
│       ├── ml_results_patient_wise.json
│       ├── noise_robustness_results.json
│       └── summary.md
│
├── logs/                         # 日志文件
│   ├── server.log                # 服务器日志
│   ├── error.log                 # 错误日志
│   ├── train_ml.log              # ML训练日志
│   └── train_dl.log              # DL训练日志
│
├── backup/                       # 备份目录
│   └── ecg_system.db             # 数据库备份
│
├── catboost_info/                # CatBoost训练信息
│
├── .env                          # 环境变量（不提交）
├── .env.example                  # 环境变量模板
├── .env.training                 # 训练环境变量
├── .gitignore                    # Git忽略文件
├── requirements.txt              # Python依赖
├── ecg_system.db                 # SQLite数据库
├── README.md                     # 项目说明
├── IMPROVEMENTS.md               # 改进说明
└── PROJECT_STRUCTURE.md          # 本文件
```

## 📝 核心文件说明

### 后端核心

| 文件 | 说明 | 重要性 |
|------|------|--------|
| `app/main.py` | FastAPI应用入口，路由注册 | ⭐⭐⭐⭐⭐ |
| `app/config.py` | 配置管理，环境变量 | ⭐⭐⭐⭐⭐ |
| `app/api/ecg.py` | ECG分析API接口 | ⭐⭐⭐⭐⭐ |
| `app/services/ecg_service.py` | ECG分析业务逻辑 | ⭐⭐⭐⭐⭐ |
| `app/algorithms/inference.py` | 推理引擎，模型调用 | ⭐⭐⭐⭐⭐ |
| `app/algorithms/multimodal_fusion.py` | 多模态融合核心 | ⭐⭐⭐⭐⭐ |

### 前端核心

| 文件 | 说明 | 重要性 |
|------|------|--------|
| `frontend/index.html` | 主界面（监控+历史+配置） | ⭐⭐⭐⭐⭐ |
| `frontend/login.html` | 登录界面 | ⭐⭐⭐⭐ |
| `frontend/register.html` | 注册界面 | ⭐⭐⭐⭐ |

### 工具脚本

| 文件 | 说明 | 使用频率 |
|------|------|----------|
| `scripts/init_admin.py` | 初始化管理员 | 首次部署 |
| `scripts/train_traditional_ml.py` | 训练ML模型 | 模型训练 |
| `scripts/train_multimodal_models.py` | 训练DL模型 | 模型训练 |
| `scripts/visualize_results.py` | 可视化结果 | 实验分析 |
| `scripts/test_risk_warning.py` | 风险预警测试 | 功能测试 |
| `scripts/test_noise_robustness.py` | 抗干扰测试 | 性能评估 |
| `scripts/ablation_study.py` | 消融实验 | 模型对比 |
| `scripts/test_grad_cam.py` | Grad-CAM测试 | 可解释性 |
| `scripts/generate_test_data.py` | 生成测试数据 | 开发测试 |

### 文档

| 文件 | 说明 |
|------|------|
| `README.md` | 项目说明、快速开始 |
| `IMPROVEMENTS.md` | 改进说明文档 |
| `PROJECT_STRUCTURE.md` | 项目结构说明（本文件） |
| `app/algorithms/README.md` | 算法详细说明 |
| `scripts/TRAINING_GUIDE.md` | 模型训练指南 |

## 🔄 数据流

```
用户上传ECG文件
    ↓
frontend/index.html
    ↓ HTTP POST
app/api/ecg.py (upload接口)
    ↓
app/services/ecg_service.py (create_task)
    ↓ 后台任务
app/algorithms/reader.py (读取数据)
    ↓
app/algorithms/preprocess.py (预处理)
    ↓
app/algorithms/features.py (特征提取)
    ↓
app/algorithms/inference.py (推理)
    ↓
app/algorithms/multimodal_fusion.py (融合)
    ↓
app/services/report_service.py (生成报告)
    ↓
保存结果到数据库
    ↓
前端轮询获取结果
    ↓
显示分析结果
```

## 🗄️ 数据库表结构

### users表
- id (主键)
- username (用户名，唯一)
- email (邮箱，唯一)
- hashed_password (加密密码)
- full_name (全名)
- role (角色：admin/user)
- is_active (是否激活)
- created_at (创建时间)

### ecg_tasks表
- id (主键)
- filename (文件名)
- file_path (文件路径)
- status (状态：pending/completed/failed)
- result (JSON结果)
- report_path (报告路径)
- error_message (错误信息)
- user_id (外键 → users.id)
- created_at (创建时间)
- completed_at (完成时间)

## 🚀 启动流程

1. **初始化数据库**
   ```bash
   python scripts/init_admin.py
   ```

2. **启动服务器**
   ```bash
   uvicorn app.main:app --reload
   ```

3. **访问系统**
   - 前端：http://localhost:8000/index.html
   - API文档：http://localhost:8000/docs

## 📦 依赖关系

```
FastAPI (Web框架)
├── Tortoise ORM (数据库)
├── python-jose (JWT)
├── passlib (密码加密)
└── uvicorn (ASGI服务器)

PyTorch (深度学习)
├── torchvision
└── CUDA (可选)

Scikit-learn (机器学习)
├── XGBoost
├── LightGBM
├── CatBoost
└── joblib

信号处理
├── NumPy
├── SciPy
├── wfdb
└── PyWavelets

其他
├── ReportLab (PDF生成)
├── Pillow (图像处理)
└── python-multipart (文件上传)
```

## 🔧 开发建议

1. **修改算法**：编辑 `app/algorithms/` 目录下的文件
2. **修改API**：编辑 `app/api/` 目录下的文件
3. **修改前端**：编辑 `frontend/` 目录下的HTML文件
4. **添加测试**：在 `tests/` 目录添加测试文件
5. **查看日志**：检查 `logs/` 目录下的日志文件

## 📊 性能优化

- **模型加载**：使用单例模式，避免重复加载
- **数据库**：使用异步ORM，提高并发性能
- **前端**：使用Canvas渲染，减少DOM操作
- **缓存**：缓存常用数据，减少数据库查询

## 🔒 安全措施

- JWT认证
- 密码加密（bcrypt）
- 文件类型验证
- 文件大小限制
- SQL注入防护（ORM）
- XSS防护（前端转义）
