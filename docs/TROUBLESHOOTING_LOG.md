# ECG 系统故障排查与修复记录

本文档记录了在开发 ECG 系统认证与管理功能过程中遇到的关键问题及其解决方案，供后续维护参考。

## 1. 认证系统故障 (401 Unauthorized)

### 🔴 症状
- 用户能成功登录并获取 Token。
- 但使用该 Token 访问受保护接口（如 `/api/auth/me` 或上传接口）时，服务器始终返回 `401 Unauthorized`。
- 本地测试脚本能验证通过，但服务器运行失败。

### 🔍 原因分析
1.  **JWT Subject 类型错误**：JWT 标准（RFC 7519）建议 `sub` (Subject) 字段为字符串。`python-jose` 库严格执行了这一点，而代码中直接使用了整数类型的 `user.id`，导致验证失败。
2.  **依赖库冲突**：环境中同时安装了 `PyJWT` 和 `python-jose`。这两个库都使用 `import jwt` 命名空间，导致运行时加载了错误的库实现。
3.  **环境变量格式错误**：`.env` 文件中变量之间缺少换行符，导致 `SECRET_KEY` 读取包含了其他变量的内容，造成签名密钥不匹配。

### ✅ 解决方案

#### 1. 代码修改 (`app/api/auth.py`)
在生成 Token 时，强制将用户 ID 转换为字符串：

```python
# ❌ 修改前
access_token = create_access_token(data={"sub": user.id})

# ✅ 修改后
access_token = create_access_token(data={"sub": str(user.id)})
```

#### 2. 依赖清理
确保只保留 `python-jose`：

```bash
pip uninstall PyJWT
pip install "python-jose[cryptography]"
```

#### 3. 配置文件修复
确保 `.env` 文件格式正确，每个变量占一行：

```env
MYSQL_PASSWORD=lyp
MYSQL_HOST=localhost
SECRET_KEY=your-secret-key...
```

---

## 2. 数据库 Schema 不一致

### 🔴 症状
- 文件上传接口报错：`Internal Server Error`。
- 错误日志显示：`(1054, "Unknown column 'user_id' in 'field list'")`。

### 🔍 原因分析
虽然在 Python 代码 (`app/models/task.py`) 中为 `ECGTask` 模型添加了 `user` 外键，但数据库表结构未同步更新。Tortoise ORM 的 `generate_schemas=True` 通常只在表不存在时创建表，不会自动修改现有表结构。

### ✅ 解决方案
手动执行 SQL 更新表结构：

```sql
-- 添加 user_id 列
ALTER TABLE ecg_tasks ADD COLUMN user_id INT;

-- 添加外键约束
ALTER TABLE ecg_tasks ADD CONSTRAINT fk_ecg_tasks_users 
FOREIGN KEY (user_id) REFERENCES users(id);
```

> **建议**：在生产环境中，推荐使用 `aerich` 等数据库迁移工具来管理 Schema 变更。

---

## 3. 前端状态卡死 ("分析中")

### 🔴 症状
- 上传文件后，页面一直显示"分析中"。
- 结果区域显示文件名为 "undefined"。
- 浏览器控制台显示 401 错误。

### 🔍 原因分析
前端的轮询函数 `pollResult` 在发送请求时，未在 HTTP 头中携带 `Authorization` Token。由于后端 API 已开启认证保护，未携带 Token 的请求会被拒绝，导致前端无法获取任务的最新状态。

### ✅ 解决方案
修改前端代码 (`frontend/index.html`)，在所有 API 请求中添加 Token：

```javascript
async function pollResult(taskId) {
    const token = localStorage.getItem('access_token'); // 获取 Token
    
    const response = await fetch(`/api/ecg/tasks/${taskId}`, {
        headers: {
            'Authorization': `Bearer ${token}` // ✅ 添加认证头
        }
    });
    // ...
}
```

---

## 4. Pydantic 配置警告

### 🔴 症状
启动服务时出现警告：
`UserWarning: Valid config keys have changed in V2: 'orm_mode' has been renamed to 'from_attributes'`

### 🔍 原因分析
项目使用了 Pydantic V2 版本，但代码中仍使用 V1 版本的配置项 `orm_mode`。

### ✅ 解决方案
更新所有 Schema 类 (`app/schemas/*.py`) 的配置：

```python
class Config:
    # orm_mode = True       # ❌ 旧写法 (V1)
    from_attributes = True  # ✅ 新写法 (V2)
```

---

## 5. 调试技巧总结

在排查此类问题时，以下方法非常有效：

1.  **独立诊断脚本**：编写不依赖 Web 框架的独立脚本 (`diagnose_auth.py`) 来验证核心逻辑（如 Token 生成与验证）。
2.  **调试日志**：在关键路径（生成 Token 处、验证 Token 处）打印 `SECRET_KEY` 的前几位字符，确认密钥一致性。
3.  **捕获异常**：不要在 `try...except` 中简单地吞掉异常（如 `return None`），在调试阶段务必打印具体的异常信息 (`print(e)`)。
4.  **检查依赖**：使用 `pip list | grep ...` 检查是否存在冲突的库。
