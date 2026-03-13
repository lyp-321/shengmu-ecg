# ECG系统分层检查计划

## 🎯 检查目标

系统化地检查每一层，确保逻辑正确、代码健壮、功能完整。

---

## 📋 检查顺序

```
第1层：数据库层 (Database Layer)
    ↓
第2层：数据模型层 (Models Layer)
    ↓
第3层：算法层 (Algorithms Layer)
    ↓
第4层：服务层 (Services Layer)
    ↓
第5层：API层 (API Layer)
    ↓
第6层：前端层 (Frontend Layer)
    ↓
第7层：集成测试 (Integration Test)
```

---

## 第1层：数据库层检查

### 检查项目
- [ ] 数据库文件是否存在
- [ ] 表结构是否正确
- [ ] 索引是否创建
- [ ] 数据完整性约束
- [ ] 外键关系

### 检查文件
- `ecg_system.db`
- `app/db/session.py`
- `app/db/init_data.py`

### 检查方法
```bash
# 1. 检查数据库文件
ls -lh ecg_system.db

# 2. 查看表结构
sqlite3 ecg_system.db ".schema"

# 3. 查看数据
sqlite3 ecg_system.db "SELECT * FROM users;"
sqlite3 ecg_system.db "SELECT * FROM ecg_tasks LIMIT 5;"
```

### 预期结果
- 数据库文件存在
- users表和ecg_tasks表存在
- 至少有1个管理员用户

---

## 第2层：数据模型层检查

### 检查项目
- [ ] User模型定义正确
- [ ] ECGTask模型定义正确
- [ ] 字段类型正确
- [ ] 关系定义正确
- [ ] JSON字段序列化

### 检查文件
- `app/models/user.py`
- `app/models/task.py`

### 检查方法
```python
# 创建检查脚本
python -c "
from app.models.user import User
from app.models.task import ECGTask
print('✅ 模型导入成功')
print(f'User字段: {User._meta.fields_map.keys()}')
print(f'ECGTask字段: {ECGTask._meta.fields_map.keys()}')
"
```

### 预期结果
- 模型可以正常导入
- 字段定义完整
- 无语法错误

---

## 第3层：算法层检查

### 检查项目
- [ ] 数据读取模块
- [ ] 预处理模块
- [ ] 特征提取模块
- [ ] 推理引擎
- [ ] 多模态融合引擎
- [ ] 模型文件存在

### 检查文件
- `app/algorithms/reader.py`
- `app/algorithms/preprocess.py`
- `app/algorithms/features.py`
- `app/algorithms/inference.py`
- `app/algorithms/multimodal_fusion.py`
- `app/algorithms/models/*.pkl`

### 检查方法
```python
# 测试算法模块
python -c "
import numpy as np
from app.algorithms.reader import ECGReader
from app.algorithms.preprocess import ECGPreprocessor
from app.algorithms.features import ECGFeatureExtractor
from app.algorithms.inference import ECGInference

# 测试数据
test_signal = np.random.randn(1000)
test_data = {'signal': test_signal, 'sampling_rate': 360}

# 测试预处理
preprocessor = ECGPreprocessor()
processed = preprocessor.process(test_data)
print(f'✅ 预处理成功: {processed.keys()}')

# 测试特征提取
extractor = ECGFeatureExtractor()
features = extractor.extract(processed)
print(f'✅ 特征提取成功: {features.keys()}')

# 测试推理
inference = ECGInference()
result = inference.predict(features)
print(f'✅ 推理成功: {result.keys()}')
"
```

### 预期结果
- 所有模块可以导入
- 数据流通畅
- 返回结果格式正确

---

## 第4层：服务层检查

### 检查项目
- [ ] ECG服务逻辑
- [ ] 报告生成服务
- [ ] 异常处理
- [ ] 日志记录
- [ ] 文件验证

### 检查文件
- `app/services/ecg_service.py`
- `app/services/report_service.py`
- `app/core/logger.py`
- `app/core/exceptions.py`
- `app/core/validators.py`

### 检查方法
```python
# 测试服务层
python -c "
from app.services.ecg_service import ECGService
from app.core.logger import logger
from app.core.validators import FileValidator

service = ECGService()
print(f'✅ ECG服务初始化成功')
print(f'✅ 推理引擎: {service.inference}')
print(f'✅ 日志系统: {logger}')
"
```

### 预期结果
- 服务可以初始化
- 推理引擎单例模式工作正常
- 日志系统正常

---

## 第5层：API层检查

### 检查项目
- [ ] 认证接口
- [ ] 用户管理接口
- [ ] ECG分析接口
- [ ] 路由注册
- [ ] 中间件配置
- [ ] CORS配置

### 检查文件
- `app/main.py`
- `app/api/auth.py`
- `app/api/users.py`
- `app/api/ecg.py`
- `app/core/security.py`
- `app/core/deps.py`

### 检查方法
```bash
# 1. 启动服务器
uvicorn app.main:app --reload &

# 2. 等待启动
sleep 3

# 3. 测试健康检查
curl http://localhost:8000/

# 4. 测试API文档
curl http://localhost:8000/docs

# 5. 测试登录
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

### 预期结果
- 服务器正常启动
- API文档可访问
- 登录接口返回token

---

## 第6层：前端层检查

### 检查项目
- [ ] 页面加载
- [ ] 登录功能
- [ ] 文件上传
- [ ] 结果显示
- [ ] 历史记录
- [ ] 波形渲染
- [ ] 算法模式选择

### 检查文件
- `frontend/index.html`
- `frontend/login.html`
- `frontend/register.html`
- `frontend/index_enhanced.html`

### 检查方法
```bash
# 1. 访问登录页面
curl -I http://localhost:8000/login.html

# 2. 访问主页面
curl -I http://localhost:8000/index.html

# 3. 检查JavaScript语法
# 使用浏览器开发者工具检查Console
```

### 手动检查
1. 打开浏览器访问 http://localhost:8000/login.html
2. 登录系统
3. 上传测试文件
4. 查看分析结果
5. 检查历史记录
6. 测试算法模式切换

### 预期结果
- 页面正常加载
- 登录成功
- 文件上传成功
- 结果正确显示
- 无JavaScript错误

---

## 第7层：集成测试

### 完整流程测试
1. 用户注册
2. 用户登录
3. 上传ECG文件
4. 等待分析完成
5. 查看结果
6. 下载PDF报告
7. 查看历史记录

### 测试脚本
```bash
#!/bin/bash

echo "=== ECG系统集成测试 ==="

# 1. 注册新用户
echo "1. 注册新用户..."
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "test123",
    "full_name": "测试用户"
  }'

# 2. 登录获取token
echo "2. 登录..."
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"test123"}' \
  | grep -o '"access_token":"[^"]*"' \
  | cut -d'"' -f4)

echo "Token: $TOKEN"

# 3. 上传文件
echo "3. 上传ECG文件..."
TASK=$(curl -s -X POST http://localhost:8000/api/ecg/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_data/normal_ecg.csv")

TASK_ID=$(echo $TASK | grep -o '"id":[0-9]*' | cut -d':' -f2)
echo "Task ID: $TASK_ID"

# 4. 等待分析完成
echo "4. 等待分析..."
sleep 5

# 5. 获取结果
echo "5. 获取结果..."
curl -X GET "http://localhost:8000/api/ecg/tasks/$TASK_ID" \
  -H "Authorization: Bearer $TOKEN"

# 6. 获取任务列表
echo "6. 获取任务列表..."
curl -X GET http://localhost:8000/api/ecg/tasks \
  -H "Authorization: Bearer $TOKEN"

echo ""
echo "=== 测试完成 ==="
```

---

## 🔍 问题排查清单

### 数据库问题
- [ ] 表不存在 → 运行 `python scripts/init_admin.py`
- [ ] 数据损坏 → 从 `backup/` 恢复
- [ ] 连接失败 → 检查文件权限

### 算法问题
- [ ] 模型文件缺失 → 运行训练脚本
- [ ] 推理失败 → 检查输入数据格式
- [ ] JSON序列化错误 → 检查numpy类型转换

### API问题
- [ ] 404错误 → 检查路由注册
- [ ] 401错误 → 检查token
- [ ] 500错误 → 查看服务器日志

### 前端问题
- [ ] 页面空白 → 检查Console错误
- [ ] 上传失败 → 检查文件格式
- [ ] 结果不显示 → 检查任务状态

---

## 📊 检查报告模板

```
检查日期: ____________________
检查人员: ____________________

【第1层：数据库层】
- 数据库文件: ✅ / ❌
- 表结构: ✅ / ❌
- 数据完整性: ✅ / ❌
问题: ________________________________

【第2层：数据模型层】
- User模型: ✅ / ❌
- ECGTask模型: ✅ / ❌
- 字段定义: ✅ / ❌
问题: ________________________________

【第3层：算法层】
- 数据读取: ✅ / ❌
- 预处理: ✅ / ❌
- 特征提取: ✅ / ❌
- 推理引擎: ✅ / ❌
- 多模态融合: ✅ / ❌
问题: ________________________________

【第4层：服务层】
- ECG服务: ✅ / ❌
- 报告服务: ✅ / ❌
- 异常处理: ✅ / ❌
问题: ________________________________

【第5层：API层】
- 认证接口: ✅ / ❌
- ECG接口: ✅ / ❌
- 路由注册: ✅ / ❌
问题: ________________________________

【第6层：前端层】
- 页面加载: ✅ / ❌
- 登录功能: ✅ / ❌
- 文件上传: ✅ / ❌
- 结果显示: ✅ / ❌
问题: ________________________________

【第7层：集成测试】
- 完整流程: ✅ / ❌
问题: ________________________________

【总体评价】
□ 完全正常
□ 部分问题（需修复）
□ 严重问题（需重构）

【修复计划】
1. ________________________________
2. ________________________________
3. ________________________________
```

---

## 🚀 开始检查

### 方法1：自动运行所有检查

```bash
# 运行所有检查（推荐）
./run_all_checks.sh
```

### 方法2：手动逐层检查

```bash
# 第1-5层：后端检查（不需要服务器）
python check_layer1_database.py
python check_layer2_models.py
python check_layer3_algorithms.py
python check_layer4_services.py
python check_layer5_api.py

# 启动服务器
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 第6-7层：前端和集成测试（需要服务器）
python check_layer6_frontend.py
python check_layer7_integration.py
```

### 检查脚本说明

- `check_layer1_database.py` - 数据库层检查
- `check_layer2_models.py` - 数据模型层检查
- `check_layer3_algorithms.py` - 算法层检查
- `check_layer4_services.py` - 服务层检查
- `check_layer5_api.py` - API层检查
- `check_layer6_frontend.py` - 前端层检查（需要服务器）
- `check_layer7_integration.py` - 集成测试（需要服务器）
- `run_all_checks.sh` - 自动运行所有检查
