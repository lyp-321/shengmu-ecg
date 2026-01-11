# MySQL 数据库配置说明

## 推荐方式：使用 .env 文件

1. 复制 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入您的 MySQL 配置：
```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_actual_password
MYSQL_DATABASE=ecg_system
```

3. `.env` 文件已被 `.gitignore` 保护，不会被提交到 Git

## 备选方式：环境变量

您也可以通过设置环境变量来配置 MySQL 连接信息：

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_DATABASE=ecg_system
```

## 创建数据库

在启动应用之前，请先在 MySQL 中创建数据库：

```sql
CREATE DATABASE ecg_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## 默认配置

如果不设置环境变量，系统将使用以下默认配置：
- 主机：localhost
- 端口：3306
- 用户：root
- 密码：（空）
- 数据库：ecg_system

## 修改配置

您可以直接编辑 `app/config.py` 文件来修改默认配置。
