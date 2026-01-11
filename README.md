# ECG 心电图分析系统

一个基于 FastAPI 的心电图分析系统，提供 ECG 数据上传、预处理、特征提取和智能分析功能。

## 项目结构

```
ecg-system/
├── app/                     # 后端应用
│   ├── main.py             # FastAPI 入口
│   ├── api/                # API 路由层
│   ├── services/           # 业务逻辑层
│   ├── algorithms/         # 算法模块
│   │   ├── reader.py       # 数据读取
│   │   ├── preprocess.py   # 预处理
│   │   ├── features.py     # 特征提取
│   │   └── inference.py    # 推理计算
│   ├── models/             # 数据库模型
│   ├── db/                 # 数据库配置
│   └── schemas/            # Pydantic 模型
├── frontend/               # 前端界面
├── scripts/                # 工具脚本
│   ├── init_admin.py       # 初始化管理员
│   ├── generate_test_data.py # 生成测试数据
│   └── check_env_security.py # 环境检查
├── docs/                   # 项目文档
├── logs/                   # 运行日志
├── data/                   # 数据目录
├── experiments/            # 实验对比
└── requirements.txt        # 依赖列表
```

## 功能特性

- ✅ ECG 文件上传（支持 CSV、DAT 等格式）
- ✅ 自动预处理（去噪、滤波、归一化）
- ✅ 特征提取（R 波检测、心率、HRV）
- ✅ 智能分析（诊断、风险评估）
- ✅ 任务管理（异步处理、状态追踪）
- ✅ 可视化界面

## 快速开始

### 1. 安装依赖

```bash
cd ecg-system
pip install -r requirements.txt
```

### 2. 配置 MySQL 数据库

**创建数据库**：
```sql
CREATE DATABASE ecg_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

**配置连接**（推荐使用 `.env` 文件）：

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入您的配置：
```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_actual_password
MYSQL_DATABASE=ecg_system
```

> **安全提示**：`.env` 文件已被 `.gitignore` 保护，不会被提交到版本控制。

### 3. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动时会自动创建数据库表。

### 4. 访问系统

打开浏览器访问：http://localhost:8000

## API 文档

启动服务后，访问以下地址查看 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 主要 API 接口

### 上传 ECG 文件
```
POST /api/ecg/upload
```

### 查询任务结果
```
GET /api/ecg/tasks/{task_id}
```

### 获取任务列表
```
GET /api/ecg/tasks
```

## 算法流程

1. **数据读取** (`reader.py`)
   - 支持多种 ECG 文件格式
   - 自动识别采样率

2. **预处理** (`preprocess.py`)
   - 基线漂移去除
   - 带通滤波（0.5-40 Hz）
   - 信号归一化

3. **特征提取** (`features.py`)
   - R 波检测
   - 心率计算
   - RR 间期分析
   - HRV 特征（SDNN、RMSSD）

4. **推理分析** (`inference.py`)
   - 心律诊断
   - 风险评估

## 开发指南

### 添加新的算法

在 `app/algorithms/` 目录下创建新模块，然后在 `ecg_service.py` 中集成。

### 扩展数据库模型

在 `app/models/` 中定义新的 ORM 模型，并更新 `init_db.py`。

### 实验对比

使用 `experiments/compare_methods.py` 进行不同算法的性能对比。

## 技术栈

- **后端**: FastAPI + Tortoise ORM
- **算法**: NumPy + SciPy
- **数据库**: MySQL
- **前端**: 原生 HTML/CSS/JavaScript

## 许可证

MIT License

## 作者

ECG System Team
