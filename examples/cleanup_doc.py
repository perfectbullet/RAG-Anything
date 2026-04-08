#!/usr/bin/env python
"""
清理指定 doc_id 的文档数据

使用方法:
    python cleanup_doc.py <doc_id1> <doc_id2> ...
    python cleanup_doc.py --dry-run <doc_id>  # 预览模式
    python cleanup_doc.py --pattern "prefix_*"  # 批量删除（包含通配符）
    python cleanup_doc.py --verbose  # 显示详细日志

功能说明:
- 支持单个或多个 doc_id 清理
- 支持通配符模式批量删除（需要 pymongo）
- 干运行模式预览将要删除的内容
- 检查各数据库集合中的数据情况

注意:
- 确保已安装所有依赖: pip install -e '.[all]'
- 需要正确配置 .env 文件中的数据库连接信息
- 删除操作不可恢复，请先使用 --dry-run 预览
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


async def cleanup_docs(
    doc_ids: List[str],
    dry_run: bool = False,
    pattern: str = None,
):
    """清理指定的文档"""
    deleted_count = 0

    # 处理通配符模式
    if pattern:
        try:
            from pymongo import MongoClient
        except ImportError:
            logger.error("❌ 使用通配符模式需要安装 pymongo: pip install pymongo")
            return

        # 获取MongoDB配置
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        mongo_database = os.getenv("MONGO_DATABASE", "rag_db")

        try:
            # 连接MongoDB
            mongo_client = MongoClient(mongo_uri)
            db = mongo_client[mongo_database]
            doc_status_collection = db["doc_status"]

            # 构建查询条件
            if "*" in pattern:
                # 简单的通配符支持
                prefix = pattern.replace("*", "")
                matching_docs = list(doc_status_collection.find({"doc_id": {"$regex": f"^{prefix}"}}))
            else:
                # 精确匹配
                matching_docs = list(doc_status_collection.find({"doc_id": pattern}))

            if matching_docs:
                logger.info(f"🔍 找到 {len(matching_docs)} 个匹配的文档")
                for doc in matching_docs:
                    doc_id = doc["doc_id"]
                    doc_ids.append(doc_id)
            else:
                logger.warning(f"⚠️  未找到匹配模式 '{pattern}' 的文档")
                return
        except Exception as e:
            logger.error(f"❌ 查询MongoDB失败: {e}")
            return

    # 去重
    unique_doc_ids = list(set(doc_ids))

    if not unique_doc_ids:
        logger.warning("⚠️  没有要清理的文档")
        return

    logger.info(f"📝 将要清理 {len(unique_doc_ids)} 个文档: {unique_doc_ids}")

    # 直接使用数据库连接进行清理
    try:
        # 从环境变量获取数据库配置
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        mongo_database = os.getenv("MONGO_DATABASE", "rag_db")
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        # 获取Milvus配置
        milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        milvus_user = os.getenv("MILVUS_USER", "root")
        milvus_password = os.getenv("MILVUS_PASSWORD", "Milvus")
        milvus_db_name = os.getenv("MILVUS_DB_NAME", "rag_db")

        # 逐个清理文档
        for doc_id in unique_doc_ids:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] 🗑️  将删除文档: {doc_id}")

                    # 检查各集合中是否有该文档的数据
                    # MongoDB检查
                    try:
                        from pymongo import MongoClient
                        mongo_client = MongoClient(mongo_uri)
                        db = mongo_client[mongo_database]

                        # 检查doc_status
                        doc_status = db["doc_status"].find_one({"doc_id": doc_id})
                        if doc_status:
                            logger.info(f"[DRY RUN]   📄 文档状态: 已存在")

                        # 检查text_chunks
                        text_chunks_count = db["text_chunks"].count_documents({"doc_id": doc_id})
                        if text_chunks_count > 0:
                            logger.info(f"[DRY RUN]   📝 文本块数量: {text_chunks_count}")

                        mongo_client.close()
                    except Exception as e:
                        logger.info(f"[DRY RUN]   📊 MongoDB状态: 无法检查 - {e}")

                    # Neo4j检查
                    try:
                        from neo4j import GraphDatabase
                        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
                        with driver.session() as session:
                            result = session.run("MATCH (n) WHERE n.doc_id = $doc_id RETURN count(n) as count",
                                               doc_id=doc_id)
                            count = result.single()["count"]
                            if count > 0:
                                logger.info(f"[DRY RUN]   🔗 Neo4j节点数量: {count}")
                        driver.close()
                    except Exception as e:
                        logger.info(f"[DRY RUN]   🔗 Neo4j状态: 无法检查 - {e}")

                    # Milvus检查（需要pymilvus）
                    try:
                        import pymilvus
                        from pymilvus import connections, Collection
                        connections.connect(uri=milvus_uri, user=milvus_user, password=milvus_password)

                        # 尝试不同的集合名称
                        collection_names = [f"_{milvus_db_name}", f"{milvus_db_name}", "text_chunks"]
                        found_count = 0

                        for coll_name in collection_names:
                            try:
                                collection = Collection(coll_name)
                                expr = f"doc_id == '{doc_id}'"
                                count = collection.count_rows(expr)
                                if count > 0:
                                    found_count += count
                                    logger.info(f"[DRY RUN]   📐 Milvus({coll_name})向量数量: {count}")
                            except:
                                pass

                        if found_count > 0:
                            logger.info(f"[DRY RUN]   📐 Milvus总向量数量: {found_count}")

                        connections.disconnect("default")
                    except Exception as e:
                        logger.info(f"[DRY RUN]   📐 Milvus状态: 无法检查 - {e}")

                else:
                    logger.info(f"🗑️  正在删除文档: {doc_id}")

                    # 初始化LightRAG以执行删除
                    from lightrag import LightRAG
                    from lightrag.utils import EmbeddingFunc

                    # 简单的嵌入函数（实际使用时应该从VLLM获取）
                    async def dummy_embedding(texts):
                        import numpy as np
                        return [[0.0] * 1024 for _ in texts]

                    embedding_func = EmbeddingFunc(
                        embedding_dim=1024,
                        max_token_size=8192,
                        func=dummy_embedding
                    )

                    rag = LightRAG(
                        working_dir="../math_kb_v3",  # 临时目录
                        kv_storage="MongoKVStorage",
                        vector_storage="MilvusVectorDBStorage",
                        doc_status_storage="MongoDocStatusStorage",
                        graph_storage="Neo4JStorage",
                        embedding_func=embedding_func,
                        lightrag_kwargs={
                            "working_dir": os.getenv("KB_DIR", "../math_kb_v3"),
                            "kv_storage_kwargs": {"uri": mongo_uri, "db_name": mongo_database},
                            "vector_storage_kwargs": {"uri": milvus_uri, "db_name": milvus_db_name},
                            "doc_status_storage_kwargs": {"uri": mongo_uri, "db_name": mongo_database},
                            "graph_storage_kwargs": {"uri": neo4j_uri, "username": neo4j_username, "password": neo4j_password}
                        }
                    )

                    # 执行删除
                    result = await rag.adelete_by_doc_id(doc_id)
                    logger.info(f"✅ 已删除 {doc_id}: {result}")
                    deleted_count += 1

            except Exception as e:
                logger.error(f"❌ 删除文档 {doc_id} 失败: {e}")

        if not dry_run and deleted_count > 0:
            logger.info(f"🎉 成功清理 {deleted_count} 个文档")
        elif dry_run:
            logger.info("🔍 预览模式完成，未实际删除任何数据")

    except Exception as e:
        logger.error(f"❌ 清理失败: {e}")
        logger.error("请检查数据库配置是否正确")


def get_data_safely(storage, collection: str, query: dict):
    """安全地从存储中获取数据"""
    try:
        return storage.get_data(collection, query)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="清理指定 doc_id 的文档数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python cleanup_doc.py doc_001 doc_002
    python cleanup_doc.py --dry-run doc_001
    python cleanup_doc.py --pattern "doc_*"
    python cleanup_doc.py --pattern "2024_*" --dry-run
"""
    )

    parser.add_argument(
        "doc_ids",
        nargs="*",
        help="要删除的 doc_id 列表"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际删除，只显示将要删除的内容"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help="使用通配符模式匹配文档ID（如 'doc_*'）"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )

    args = parser.parse_args()

    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 检查参数
    if not args.doc_ids and not args.pattern:
        parser.error("请提供至少一个 doc_id 或使用 --pattern 参数")

    # 加载环境变量
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=str(env_file), override=False)
            logger.info("✅ 已加载 .env 文件")
        except ImportError:
            logger.warning("⚠️  未安装 python-dotenv，请运行: pip install python-dotenv")
            return
    else:
        logger.warning("⚠️  未找到 .env 文件，请确保环境变量已正确设置")

    try:
        # 运行清理
        asyncio.run(cleanup_docs(
            args.doc_ids,
            args.dry_run,
            args.pattern,
        ))

    except Exception as e:
        logger.error(f"❌ 清理脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()