# Milvus Standalone 部署

本项目使用 Docker Compose 部署 Milvus 向量数据库及其依赖服务。

## 服务说明

| 服务 | 容器名 | 端口 | 说明 |
|------|--------|------|------|
| Milvus | milvus-standalone | 19530, 9091 | 向量数据库 |
| etcd | milvus-etcd | - | 元数据存储 |
| MinIO | milvus-minio | 9000, 9001 | 对象存储 |
| Attu | milvus-attu | 3000 | Web 管理界面 |

## 默认凭证

### Milvus
- 用户名: `root`
- 密码: `Milvus`（**大写 M**，这是 Milvus 的固定默认密码**）
- 连接地址: `http://服务器IP:19530`

### MinIO（对象存储，仅供 Milvus 内部使用）
- Access Key: `minioadmin`
- Secret Key: `minioadmin`
- Console: http://服务器IP:9001

### Attu（Web 管理界面）
- 访问地址: http://服务器IP:3000

## 快速开始

### 启动服务
```bash
cd /data/metahuman_work/RAG-Anything/milvus_deployement
docker-compose up -d
```

### 停止服务
```bash
docker-compose down
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看 Milvus 日志
docker logs -f milvus-standalone
```

## 修改 Milvus 密码

⚠️ **重要**：`docker-compose.yml` 中的 `ROOT_PASSWORD` 环境变量**无效**！

Milvus 在首次启动时使用固定的默认密码，无法通过环境变量修改。

### 方法 1：使用 Python 脚本修改（推荐）

```python
from pymilvus import MilvusClient

# 1. 先用默认密码连接
client = MilvusClient(
    uri='http://192.168.8.233:19530',
    user='root',
    password='Milvus'
)

# 2. 修改密码
client.update_password(
    user_name='root',
    old_password='Milvus',
    new_password='your_new_password'
)
```

### 方法 2：配置超级用户（忘记旧密码时）

编辑 `milvus.yaml` 配置文件：
```yaml
common:
  security:
    superUsers: root  # root 用户重置密码时不需要旧密码
```

## 数据持久化

数据存储在 `./volumes/` 目录：
- `./volumes/etcd/` - etcd 数据
- `./volumes/minio/` - MinIO 数据
- `./volumes/milvus/` - Milvus 数据

⚠️ **警告**：删除 `volumes/milvus/` 目录会丢失所有向量数据！

## 常见问题

### Q: 修改 docker-compose.yml 中的密码后不生效？
A: Milvus 不支持通过环境变量设置初始密码。请使用 `update_password()` API 修改。

### Q: 忘记 Milvus 密码怎么办？
A: 删除 `volumes/milvus/` 目录并重启容器，密码会重置为默认的 `Milvus`（但数据会丢失）。

### Q: 如何在 RAGAnything 中使用？
A: 在 `.env` 文件中配置：
```bash
MILVUS_URI=http://192.168.8.233:19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus
MILVUS_DB_NAME=rag_db
```

## 参考资料

- [Milvus 官方认证文档](https://milvus.io/docs/authenticate.md)
- [Milvus Docker Compose 配置](https://milvus.io/docs/configure-docker.md)
