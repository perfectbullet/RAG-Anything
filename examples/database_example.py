"""
RAGAnything 数据库配置示例

本示例展示如何使用 MongoDB + Milvus + Neo4j 作为后端存储。
"""

import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def main():
    """使用数据库后端的 RAGAnything 示例"""

    # ============================================
    # 1. 配置模型函数
    # ============================================
    api_key = "your-api-key"  # 替换为你的 API Key
    base_url = "https://api.openai.com/v1"  # 或你的本地服务地址

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(prompt, system_prompt=None, history_messages=[],
                          image_data=None, messages=None, **kwargs):
        """视觉模型函数（用于图像处理）"""
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=system_prompt,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            },
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # ============================================
    # 2. 创建 RAGAnything 实例（使用数据库后端）
    # ============================================

    # 方式1：通过环境变量配置（推荐）
    # 在 .env 文件中设置：
    # LIGHTRAG_KV_STORAGE=MongoKVStorage
    # LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage
    # LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage
    # LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
    #
    # MONGO_URI=mongodb://root:rag_password_123@localhost:27017/
    # MONGO_DATABASE=rag_db
    # MILVUS_URI=http://localhost:19530
    # MILVUS_TOKEN=base64_encoded_username:password  # echo -n "root:password" | base64
    # MILVUS_DB_NAME=rag_db
    # NEO4J_URI=bolt://localhost:7687
    # NEO4J_USERNAME=neo4j
    # NEO4J_PASSWORD=neo4j_password_123

    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
    )

    # 方式2：通过代码参数配置
    # 注意：MongoDB 和 Neo4j 的连接参数必须通过环境变量设置，不能通过 lightrag_kwargs 传递
    # Milvus 可以通过 vector_db_storage_cls_kwargs 显式传递，也可以使用环境变量
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        ),
        lightrag_kwargs={
            # 存储类型配置（注意类名必须与 LightRAG 中定义的完全一致）
            "kv_storage": "MongoKVStorage",
            "vector_storage": "MilvusVectorDBStorage",
            "doc_status_storage": "MongoDocStatusStorage",
            "graph_storage": "Neo4JStorage",

            # Milvus 显式配置（可选，优先级高于环境变量）
            # 如果 Milvus 启用了认证，需要提供 token
            "vector_db_storage_cls_kwargs": {
                "uri": "http://localhost:19530",
                "token": "cm9vdDpwYXNzd29yZA==",  # base64("root:password")
                "db_name": "rag_db",
            },

            # MongoDB 和 Neo4j 配置：
            # 必须通过环境变量设置，或在代码中设置 os.environ before RAGAnything 初始化
            # 例如：
            # import os
            # os.environ["MONGO_URI"] = "mongodb://root:rag_password_123@localhost:27017/"
            # os.environ["MONGO_DATABASE"] = "rag_db"
            # os.environ["NEO4J_URI"] = "bolt://localhost:7687"
            # os.environ["NEO4J_USERNAME"] = "neo4j"
            # os.environ["NEO4J_PASSWORD"] = "neo4j_password_123"

            # 语言配置
            "addon_params": {
                "language": "Chinese",  # 知识图谱构建和查询的语言
                "entity_types": ["organization", "person", "location", "event", "concept", "method"]
            },
        }
    )

    print("=" * 60)
    print("RAGAnything 已初始化（数据库后端）")
    print("=" * 60)
    print()
    print("📦 存储配置:")
    print("   KV 存储:          MongoDB")
    print("   向量存储:         Milvus (带认证)")
    print("   文档状态存储:     MongoDB")
    print("   图存储:           Neo4j")
    print()
    print("📊 数据库连接:")
    print("   MongoDB:          mongodb://localhost:27017")
    print("   Milvus:           http://localhost:19530 (需要 token)")
    print("   Neo4j:            bolt://localhost:7687")
    print()
    print("💡 提示: 如果 Milvus 启用了认证，请确保在 .env 中设置 MILVUS_TOKEN")
    print()

    # ============================================
    # 3. 处理文档
    # ============================================

    # 示例1：处理 PDF 文档
    # await rag.process_document_complete(
    #     file_path="path/to/your/document.pdf",
    #     output_dir="./output",
    #     parse_method="auto"
    # )

    # 示例2：直接插入内容列表
    content_list = [
        {
            "type": "text",
            "text": "RAGAnything 是一个强大的多模态 RAG 系统，支持 MongoDB、Milvus 和 Neo4j 作为后端存储。",
            "page_idx": 0
        },
        {
            "type": "text",
            "text": "MongoDB 用于存储键值对数据和文档状态，Milvus 用于高性能向量检索，Neo4j 用于知识图谱存储。",
            "page_idx": 1
        }
    ]

    print("📝 插入示例内容...")
    await rag.insert_content_list(
        content_list=content_list,
        file_path="example.txt",
        doc_id="example-doc",
        display_stats=True
    )
    print()

    # ============================================
    # 4. 查询知识库
    # ============================================

    print("🔍 查询示例:")
    print("=" * 60)

    queries = [
        "RAGAnything 支持哪些数据库？",
        "MongoDB 在这个系统中有什么作用？",
        "Milvus 是用来做什么的？"
    ]

    for query in queries:
        print(f"\n问题: {query}")
        result = await rag.aquery(query, mode="hybrid")
        print(f"回答: {result[:200]}...")

    print()
    print("=" * 60)
    print("✅ 示例完成！")
    print()
    print("💡 提示:")
    print("   - 数据已保存到数据库中")
    print("   - 重启程序后数据仍然保留")
    print("   - 可以使用 Neo4j 浏览器查看知识图谱: http://localhost:7474")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          RAGAnything 数据库配置示例                      ║
╚══════════════════════════════════════════════════════════╝

请确保已启动所有数据库服务：

    ./start-databases.sh start

或者：

    docker compose -f docker-compose.yml up -d

""")
    asyncio.run(main())
