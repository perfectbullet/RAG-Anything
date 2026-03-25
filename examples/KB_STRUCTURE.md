# RAGAnything 知识库目录结构说明

本文档说明 RAGAnything 生成的知识库目录结构和各文件的作用。

## 目录结构总览

```
math_kb_v3/
├── images/                                    # 图像文件目录
│   └── *.jpg                                 # 从PDF中提取的图像（SHA256命名）
├── graph_chunk_entity_relation.graphml       # 知识图谱图结构（XML格式）
├── kv_store_doc_status.json                  # 文档处理状态
├── kv_store_full_docs.json                   # 完整文档内容
├── kv_store_full_entities.json               # 实体列表
├── kv_store_full_relations.json              # 关系列表
├── kv_store_llm_response_cache.json          # LLM响应缓存
├── kv_store_text_chunks.json                 # 文本分块内容
├── vdb_chunks.json                           # 向量数据库：chunk索引
├── vdb_entities.json                         # 向量数据库：实体索引
└── vdb_relationships.json                    # 向量数据库：关系索引
```

## 各文件详解

### 1. `images/` 目录

- **作用**: 存储从PDF文档中提取的所有图像
- **命名方式**: SHA256哈希值（确保唯一性）
- **数量**: 约1500个图像文件
- **用途**: 多模态查询时，VLM需要加载这些图像进行分析

### 2. `kv_store_doc_status.json` (~55KB)

记录每个文档的处理状态。

```json
{
  "文档名": {
    "status": "failed|success",
    "error_msg": "错误信息（如果有）",
    "chunks_count": 1104,          # 总chunk数量
    "chunks_list": [...],          # 所有chunk的ID列表
    "multimodal_processed": true   # 是否处理了多模态内容
  }
}
```

**注意**: 如果 `status` 显示 `failed`，说明处理过程中遇到了错误（如速率限制），需要重新处理。

### 3. `kv_store_full_docs.json` (~455KB)

存储每个文档的完整原始内容，用于文档溯源和完整内容检索。

### 4. `kv_store_full_entities.json` (~117KB)

存储从文档中提取的所有实体名称。

```json
{
  "文档名": {
    "entity_names": ["实体1", "实体2", ...]
  }
}
```

### 5. `kv_store_full_relations.json` (~1.6MB)

存储实体间的关系对。

```json
{
  "文档名": {
    "relation_pairs": [
      ["实体A", "实体B"],
      ...
    ]
  }
}
```

### 6. `kv_store_llm_response_cache.json` (~33MB)

LLM响应缓存，避免重复调用API。

**优势**: 重新处理时可大幅节省API调用成本。

### 7. `kv_store_text_chunks.json` (~3.8MB)

存储所有文本分块的内容。

```json
{
  "chunk-id": {
    "tokens": 512,
    "content": "分块内容..."
  }
}
```

### 8. `vdb_chunks.json` (~16MB)

向量数据库 - chunk向量索引，包含chunk的嵌入向量和元数据，用于语义相似度检索。

### 9. `vdb_entities.json` (~41MB)

向量数据库 - 实体向量索引，包含实体的嵌入向量和元数据，用于实体级别的语义检索。

### 10. `vdb_relationships.json` (~145MB)

向量数据库 - 关系向量索引，包含关系的嵌入向量和元数据，用于关系级别的语义检索。

**最大的文件** - 占用存储空间最多，因为关系数量通常远大于实体数量。

### 11. `graph_chunk_entity_relation.graphml` (~12MB)

知识图谱的图结构表示，格式为 GraphML（基于XML）。可用于：

- 可视化知识图谱
- 图遍历算法
- 图分析工具（如 Gephi、Cytoscape）

## 存储空间分析

| 文件 | 大小 | 说明 |
|------|------|------|
| vdb_relationships.json | 145MB | 最大文件，关系向量 |
| vdb_entities.json | 41MB | 实体向量 |
| kv_store_llm_response_cache.json | 33MB | LLM缓存 |
| vdb_chunks.json | 16MB | chunk向量 |
| kv_store_text_chunks.json | 3.8MB | 文本内容 |
| kv_store_full_relations.json | 1.6MB | 关系列表 |
| images/ | ~114MB | 1500张图像 |
| **总计** | **~360MB** | |

## 文件生成流程

```
1. 解析PDF
   ↓
2. 提取图像 → images/*.jpg
   ↓
3. 分块处理 → kv_store_text_chunks.json
   ↓
4. LLM提取实体/关系 → kv_store_full_entities.json
                      → kv_store_full_relations.json
                      → kv_store_llm_response_cache.json (缓存)
   ↓
5. 构建向量索引 → vdb_chunks.json
                  → vdb_entities.json
                  → vdb_relationships.json
   ↓
6. 构建图谱 → graph_chunk_entity_relation.graphml
   ↓
7. 记录状态 → kv_store_doc_status.json
```

## 故障排查

### 处理状态为 failed

如果 `kv_store_doc_status.json` 中 `status` 为 `failed`:

1. 查看 `error_msg` 了解错误原因
2. 常见错误：
   - `RateLimitError`: API调用速率限制，降低并发数重试
   - `TimeoutError`: 请求超时，检查网络或增加超时时间
3. 已处理的chunks会被保留在 `kv_store_llm_response_cache.json` 中
4. 重新运行处理脚本时，会跳过已缓存的内容

### 删除缓存重新处理

如果想完全重新处理：

```bash
rm kv_store_llm_response_cache.json
```

然后重新运行 `build_kb.py`。

## 相关文件

- `examples/build_kb.py` - 知识库构建脚本
- `examples/query_kb.py` - 知识库查询脚本
