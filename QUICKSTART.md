# 快速开始指南

## 使用外部 MongoDB 的数据库配置

### 🔐 数据库凭据

| 数据库 | 用户名 | 密码 | 端口 | 数据库 |
|--------|--------|------|------|--------|
| **MongoDB** | `zenking` | `chanjing2025` | 27017 | `rag_db` |
| **Neo4j** | `neo4j` | `chanjing2025` | 7687 | - |
| **Milvus** | `root` | `chanjing2025` | 19530 | `rag_db` |
| **MinIO** | `minioadmin` | `chanjing2025` | 9000 | - |

---

## 🚀 部署步骤

### ✅ 第一步：MongoDB 用户已创建

MongoDB 用户 `zenking` 和数据库 `rag_db` 已经创建完成，可以直接使用。

**验证连接：**
```bash
mongosh "mongodb://zenking:chanjing2025@192.168.8.233:27017/rag_db" --eval "db.adminCommand('ping')"
```

---

### 第二步：启动数据库服务

```bash
# 启动所有服务
./start-databases.sh start

# 查看状态
./start-databases.sh status

# 测试连接
./start-databases.sh test
```

---

### 第三步：配置 RAGAnything

```bash
# 1. 复制配置文件
cp .env.databases .env

# 2. 修改 API Key
nano .env
# 修改 LLM_BINDING_API_KEY 和 EMBEDDING_BINDING_API_KEY

# 3. 运行示例
uv run python examples/database_example.py
```

---

## 📊 连接信息

```
数据库服务器：192.168.8.233
MongoDB:  mongodb://zenking:chanjing2025@192.168.8.233:27017/rag_db
Neo4j:   bolt://neo4j:chanjing2025@192.168.8.233:7687
Milvus:   http://192.168.8.233:19530
MinIO:    http://192.168.8.233:9001
```

---

## 🔧 管理命令

```bash
./start-databases.sh start    # 启动
./start-databases.sh stop     # 停止
./start-databases.sh restart  # 重启
./start-databases.sh status   # 状态
./start-databases.sh logs     # 日志
./start-databases.sh test     # 测试连接
./start-databases.sh info     # 连接信息
```

---

## 🌐 Web 管理界面

### Neo4j 浏览器
```
URL: http://192.168.8.233:7474
用户名: neo4j
密码: chanjing2025
```

### MinIO 控制台
```
URL: http://192.168.8.233:9001
Access Key: minioadmin
Secret Key: chanjing2025
```

### MongoDB Shell
```bash
# 连接到 MongoDB
mongosh "mongodb://zenking:chanjing2025@192.168.8.233:27017/rag_db"

# 查看集合
show collections

# 退出
exit
```

---

## ⚠️ 注意事项

1. **数据库服务器**：所有数据库部署在外部服务器 `192.168.8.233`
2. **MongoDB 用户**：用户 `zenking` 和数据库 `rag_db` 已创建完成
3. **密码特殊字符**：`@` 符号在 URL 中已正确处理
4. **数据库隔离**：RAGAnything 使用独立的 `rag_db` 数据库，与 `tts_service` 隔离

---

## 🐛 故障排查

### MongoDB 连接失败

```bash
# 1. 检查外部 MongoDB 运行状态
# 连接到数据库服务器 192.168.8.233 检查
ping 192.168.8.233

# 2. 使用管理员账户测试
mongosh "mongodb://admin:tts_password_2024@192.168.8.233:27017/"

# 3. 检查用户是否存在
docker exec -it tts_mongodb mongosh "mongodb://admin:tts_password_2024@192.168.8.233:27017/admin" --eval "db.getUsers({filter: {user: 'zenking'}})"

# 4. 测试新用户连接
mongosh "mongodb://zenking:chanjing2025@192.168.8.233:27017/rag_db"
```

### Neo4j 密码错误

```bash
# 重置 Neo4j 密码（如果忘记）
docker exec -it rag-neo4j cypher-shell -u neo4j -p chanjing2025 "ALTER CURRENT USER SET PASSWORD FROM 'chanjing2025' TO 'new_password'"
```

---

## 📝 快速配置模板

```bash
# MongoDB 连接字符串（外部服务器 192.168.8.233）
MONGO_URI=mongodb://zenking:chanjing2025@192.168.8.233:27017/rag_db
MONGO_DATABASE=rag_db

# Neo4j 连接字符串
NEO4J_URI=bolt://192.168.8.233:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=chanjing2025

# MinIO 连接
MINIO_ADDRESS=192.168.8.233:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=chanjing2025
```

---

## ✅ 验证清单

部署前请确认：

- [ ] 数据库服务器 192.168.8.233 可访问
- [ ] 外部 MongoDB (`tts_mongodb`) 正在运行
- [ ] MongoDB 用户 `zenking` 已创建
- [ ] 数据库 `rag_db` 已创建
- [ ] 已启动 Neo4j、Milvus 等服务
- [ ] 所有连接测试通过
- [ ] `.env` 文件已配置正确的 API Key

---

详细文档请查看 [DATABASE_SETUP.md](DATABASE_SETUP.md)
