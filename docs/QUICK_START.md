# ECG 系统快速使用指南

## 📊 测试数据已准备就绪
3 个测试 ECG 数据文件：
### 测试文件
1. **test_data/normal_ecg.csv** - 正常心率 (75 bpm)
2. **test_data/bradycardia_ecg.csv** - 心动过缓 (50 bpm)  
3. **test_data/tachycardia_ecg.csv** - 心动过速 (110 bpm)
---
## 🚀 快速开始测试

### 方法 1：使用 Web 界面（推荐）

#### 步骤 1：登录系统
1. 打开浏览器访问：http://localhost:8000/login.html
2. 使用默认管理员账号登录：
   - 用户名：`admin`
   - 密码：`admin123`

#### 步骤 2：上传测试文件
1. 登录成功后会跳转到主页：http://localhost:8000
2. 点击"选择文件"或直接拖拽文件到上传区域
3. 选择测试文件（例如：`test_data/normal_ecg.csv`）
4. 系统会自动上传并分析

#### 步骤 3：查看结果
分析完成后会显示：
- ✅ 心率（bpm）
- ✅ HRV 指标（SDNN、RMSSD）
- ✅ 诊断结果
- ✅ 风险等级

---

### 方法 2：使用 API（命令行）

#### 步骤 1：登录获取 Token
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | grep -o '"access_token":"[^"]*"' \
  | cut -d'"' -f4)

echo "Token: $TOKEN"
```

#### 步骤 2：上传 ECG 文件
```bash
curl -X POST http://localhost:8000/api/ecg/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test_data/normal_ecg.csv"
```

#### 步骤 3：查看任务列表
```bash
curl -X GET http://localhost:8000/api/ecg/tasks \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🧪 预期测试结果

### 正常心率 (normal_ecg.csv)
- 心率：约 75 bpm
- 诊断：正常窦性心律
- 风险等级：低风险

### 心动过缓 (bradycardia_ecg.csv)
- 心率：约 50 bpm
- 诊断：心动过缓
- 风险等级：中风险

### 心动过速 (tachycardia_ecg.csv)
- 心率：约 110 bpm
- 诊断：心动过速
- 风险等级：中风险

---

## 👥 测试用户管理功能

### 注册新用户
1. 访问：http://localhost:8000/register.html
2. 填写注册信息
3. 注册成功后登录

### 测试权限隔离
1. 用普通用户登录
2. 上传 ECG 文件
3. 只能看到自己的任务
4. 用管理员登录可以看到所有任务

---

## 📚 API 文档

访问 Swagger UI 查看完整 API 文档：
http://localhost:8000/docs

可以直接在浏览器中测试所有 API 接口！
---
## 🔧 故障排查
### 如果登录失败
```bash
# 重新创建管理员账号
python scripts/init_admin.py
```
### 如果上传失败
检查 `data/` 目录是否有写入权限

### 如果分析失败
查看服务器日志，检查算法是否正常运行

---