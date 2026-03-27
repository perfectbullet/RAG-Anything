# Neo4j 部署指南

本目录包含 RAGAnything 项目的 Neo4j 图数据库部署配置。

## 快速开始

### 1. 启动 Neo4j

```bash
# 启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f neo4j

# 停止容器
docker-compose down

# 停止并删除数据卷（⚠️ 会删除所有数据）
docker-compose down -v
```

### 2. 访问 Neo4j Browser

打开浏览器访问：http://localhost:7474

- **用户名：** `neo4j`
- **密码：** `chanjing2025`（在 docker-compose.yml 中配置）

### 3. 配置环境变量

在项目根目录的 `.env` 文件中添加：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=chanjing2025
```

## 配置说明

### 端口

| 端口 | 用途 |
|------|------|
| 7474 | HTTP - Neo4j Browser 可视化界面 |
| 7687 | Bolt - 客户端驱动连接 |

### 内存配置

默认配置（可根据服务器资源调整）：

```yaml
NEO4J_dbms_memory_heap_initial__size=512m
NEO4J_dbms_memory_heap_max__size=2g
NEO4J_dbms_memory_pagecache_size=1g
```

**建议配置：**
- 开发环境：Heap 1-2G，PageCache 512M-1G
- 生产环境：Heap 2-4G，PageCache 2-8G

### 持久化存储

数据存储在 Docker 卷中：

- `neo4j-data` - 数据库文件
- `neo4j-logs` - 日志文件
- `neo4j-import` - CSV 导入目录

## Neo4j Community Edition 警告说明

### 警告信息

```
WARNING: [base] This Neo4j instance does not support creating databases.
Try to use Neo4j Desktop/Enterprise version or DozerDB instead.
Fallback to use the default database.
```

### 警告含义

此警告表示你使用的是 **Neo4j Community Edition**（社区版），它只支持单个数据库。LightRAG 尝试创建独立数据库失败后，自动回退到使用默认数据库（`neo4j`）。

### 对功能的影响

✅ **无功能影响** - 所有核心功能完全正常：
- 实体和关系存储正常
- 图查询和检索正常
- 知识图谱构建正常

⚠️ **唯一限制** - 无法创建多个独立数据库来隔离不同项目的数据。

### 版本对比

| 特性 | Community Edition | Enterprise Edition |
|------|------------------|-------------------|
| 数据库数量 | 1 个（默认） | 支持多个独立数据库 |
| 许可证 | 免费（开源） | 付费（商业） |
| 适用场景 | 开发/测试/小型生产 | 生产环境/多租户 |
| 高可用性 | ❌ | ✅ |
| 备份/恢复 | 手动 | 自动化 |
| 驱动支持 | 完全支持 | 完全支持 |

### 数据隔离方案

如果需要在 Community Edition 中实现数据隔离：

#### 方案 1：使用命名空间前缀（推荐）

```python
from raganything import RAGAnything, RAGAnythingConfig

# 为不同项目使用不同的命名空间
rag_project_a = RAGAnything(
    config=RAGAnythingConfig(
        working_dir="./storage/project_a",
    ),
    lightrag_kwargs={
        "graph_storage": "Neo4JStorage",
        "namespace": "project_a",  # 所有节点添加前缀
    }
)

rag_project_b = RAGAnything(
    config=RAGAnythingConfig(
        working_dir="./storage/project_b",
    ),
    lightrag_kwargs={
        "graph_storage": "Neo4JStorage",
        "namespace": "project_b",
    }
)
```

#### 方案 2：使用节点标签区分

在 Neo4j Browser 中查询时使用标签过滤：

```cypher
// 查询特定项目的节点
MATCH (n:ProjectA_Entity)
RETURN n

// 查询特定项目的关系
MATCH (a:ProjectA_Entity)-[r]->(b:ProjectA_Entity)
RETURN a, r, b
```

#### 方案 3：升级到 Enterprise Edition（生产环境推荐）

如需真正的多数据库隔离，可升级到 Neo4j Enterprise Edition 或使用：
- Neo4j Desktop（本地开发）
- Neo4j AuraDB（云服务）
- DozerDB（开源替代）

## 验证部署

### 1. 检查容器状态

```bash
docker-compose ps
```

预期输出：
```
NAME                IMAGE                 STATUS
neo4j               neo4j:5.20-community  Up (healthy)
```

### 2. 测试连接

```bash
# 使用 Neo4j Browser (http://localhost:7474)
# 或使用 Python 测试
```

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "chanjing2025")
)

# 测试查询
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as total_nodes")
    print(f"Total nodes: {result.single()['total_nodes']}")

driver.close()
```

### 3. 查看 LightRAG 数据

```cypher
// 查看所有节点类型
MATCH (n) RETURN DISTINCT labels(n) as node_types, count(*) as count

// 查看实体节点
MATCH (e:__Entity__) RETURN e LIMIT 25

// 查看关系
MATCH ()-[r]->() RETURN type(r) as relationship_type, count(*) as count
```

## 可选插件

### APOC (Awesome Procedures on Cypher)

```yaml
environment:
  - NEO4J_PLUGINS=["apoc"]
```

使用示例：
```cypher
// 调用 APOC 过程
CALL apoc.help('search')
```

### GDS (Graph Data Science)

```yaml
environment:
  - NEO4J_PLUGINS=["graph-data-science"]
```

使用示例：
```cypher
// 创建图投影
CALL gds.graph.project('myGraph', '*', '*')

// 运行 PageRank
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10
```

## 性能优化

### 1. 索引优化

```cypher
// 为实体属性创建索引
CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:__Entity__) ON (e.entity_id);

// 为关系类型创建索引
CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATED]->() ON (r.weight);
```

### 2. 查询优化

```cypher
// 使用 PROFILE 分析查询性能
PROFILE MATCH (e:__Entity__) WHERE e.entity_id = 'example' RETURN e;

// 使用参数化查询
MATCH (e:__Entity__) WHERE e.entity_id = $entity_id RETURN e
```

### 3. 内存调优

根据数据量调整内存配置：

```yaml
# 小型数据集（< 10万节点）
NEO4J_dbms_memory_heap_max__size=1g
NEO4J_dbms_memory_pagecache_size=512m

# 中型数据集（10-100万节点）
NEO4J_dbms_memory_heap_max__size=2g
NEO4J_dbms_memory_pagecache_size=2g

# 大型数据集（> 100万节点）
NEO4J_dbms_memory_heap_max__size=4g
NEO4J_dbms_memory_pagecache_size=4g
```

## 备份与恢复

### 备份

```bash
# 方案 1：使用 Docker 卷备份
docker run --rm \
  -v neo4j-data:/data \
  -v $(pwd)/backups:/backups \
  alpine tar czf /backups/neo4j-backup-$(date +%Y%m%d).tar.gz /data

# 方案 2：使用 Neo4j 导出工具
docker exec neo4j neo4j-admin database dump neo4j --to-path=/var/lib/neo4j/import
```

### 恢复

```bash
# 从备份恢复
docker run --rm \
  -v neo4j-data:/data \
  -v $(pwd)/backups:/backups \
  alpine tar xzf /backups/neo4j-backup-20240326.tar.gz -C /
```

## 常见问题

### Q1: 如何重置 Neo4j 数据？

```bash
# 停止容器
docker-compose down

# 删除数据卷
docker volume rm neo4j-data neo4j-logs

# 重新启动
docker-compose up -d
```

### Q2: 如何修改密码？

1. 在 Neo4j Browser 中执行：
```cypher
ALTER CURRENT USER SET PASSWORD 'new_password';
```

2. 或修改 `docker-compose.yml` 后重启：
```yaml
environment:
  - NEO4J_AUTH=neo4j/new_password
```

### Q3: 远程连接失败？

检查防火墙和 Neo4j 配置：

```yaml
environment:
  # 允许外部访问
  - NEO4J_server_default__advertised__address=0.0.0.0
```

```bash
# 测试端口连通性
telnet <neo4j_host> 7687
```

### Q4: 内存不足错误？

调整内存限制或增加服务器资源：

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # 增加内存限制
```

## 参考资料

- [Neo4j 官方文档](https://neo4j.com/docs/)
- [Docker Hub - Neo4j](https://hub.docker.com/_/neo4j)
- [LightRAG 图存储配置](https://github.com/HKUDS/LightRAG)
- [Neo4j Browser 指南](https://neo4j.com/docs/browser/)
- [Cypher 查询语言](https://neo4j.com/docs/cypher-manual/)

## 许可证

本配置文件遵循项目主许可证。
