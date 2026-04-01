#!/usr/bin/env python
"""
流式查询示例：展示流式输出 + 召回数据

本示例展示如何：
1. 使用 aquery_stream_with_sources 进行流式查询
2. 实时显示答案流式输出
3. 在查询结束时展示完整的召回来源数据

环境变量配置（必需）：
# OpenAI兼容API
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=Qwen/Qwen2.5-72B-Instruct

# VLLM向量模型
VLLM_EMBED_URL=http://192.168.8.233:8092/v1
VLLM_EMBED_MODEL=BAAI/bge-m3
VLLM_EMBED_DIM=1024

# Reranker
VLLM_RERANK_URL=http://192.168.8.233:8091/v1
VLLM_RERANK_MODEL=bge-reranker-m3

# 数据库配置
MONGO_URI=mongodb://user:password@host:27017/
MONGO_DATABASE=rag_db
NEO4J_URI=bolt://host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
MILVUS_URI=http://host:19530
MILVUS_USER=root
MILVUS_PASSWORD=password
MILVUS_DB_NAME=rag_db
"""

import os
import asyncio
import logging
import logging.config

from pathlib import Path
from typing import Dict, Any

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import aiohttp
import numpy as np
from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnything, RAGAnythingConfig

# Load .env file
load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"), override=False)


# =============================================================================
# 配置验证
# =============================================================================
REQUIRED_ENV_VARS = {
    # OpenAI API
    "OPENAI_API_BASE": "OpenAI API base URL",
    "OPENAI_API_KEY": "OpenAI API key",
    "OPENAI_MODEL": "OpenAI model name",
    # VLLM Embedding
    "VLLM_EMBED_URL": "VLLM embedding service URL",
    "VLLM_EMBED_MODEL": "VLLM embedding model name",
    "VLLM_EMBED_DIM": "VLLM embedding dimension",
    # Reranker
    "VLLM_RERANK_URL": "VLLM reranker service URL",
    "VLLM_RERANK_MODEL": "VLLM reranker model name",
    # Databases
    "MONGO_URI": "MongoDB connection URI",
    "MONGO_DATABASE": "MongoDB database name",
    "NEO4J_URI": "Neo4j connection URI",
    "NEO4J_USERNAME": "Neo4j username",
    "NEO4J_PASSWORD": "Neo4j password",
    "MILVUS_URI": "Milvus service URI",
    "MILVUS_USER": "Milvus username (default: root)",
    "MILVUS_PASSWORD": "Milvus password",
    "MILVUS_DB_NAME": "Milvus database name",
}


def validate_required_env():
    """验证所有必需的环境变量是否已配置"""
    missing_vars = []
    for var_name, description in REQUIRED_ENV_VARS.items():
        value = os.getenv(var_name)
        if value is None or value.strip() == "":
            missing_vars.append(f"  - {var_name}: {description}")

    if missing_vars:
        error_msg = "❌ 缺少必需的环境变量配置:\n\n" + "\n".join(missing_vars)
        error_msg += f"\n\n请在 .env 文件中配置以上变量后再运行。"
        raise ValueError(error_msg)

    logger.info("✅ 环境变量配置验证通过")


def get_required_env(var_name: str) -> str:
    """获取必需的环境变量，如果不存在则报错"""
    value = os.getenv(var_name)
    if value is None or value.strip() == "":
        raise ValueError(f"缺少必需的环境变量: {var_name}")
    return value.strip()


# =============================================================================
# 日志配置
# =============================================================================
class LongDataFilter(logging.Filter):
    """日志过滤器，截断过长的数据（如 image_data）"""

    def __init__(self, max_length: int = 200):
        super().__init__()
        self.max_length = max_length

    def filter(self, record):
        """过滤日志记录，截断过长的值"""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = self._truncate_long_values(record.msg)
        return True

    def _truncate_long_values(self, text: str) -> str:
        """截断文本中过长的值"""
        import re

        def truncate_match(match):
            key = match.group(1)
            value = match.group(2)
            if len(value) > self.max_length:
                return f"'{key}': '{value[:self.max_length]}...<truncated>'"
            return match.group(0)

        pattern = r"['\"](\w*(?:image_data|messages|content)\w*)['\"]:\s*['\"]([^\"]*)['\"]"
        return re.sub(pattern, truncate_match, text)


def configure_logging():
    """配置日志系统"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "streaming_query_example.log")
    )
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    long_data_filter = LongDataFilter(max_length=200)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                },
            },
            "filters": {
                "long_data_filter": {
                    "()": "streaming_query_example.LongDataFilter",
                    "max_length": 200,
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "filters": ["long_data_filter"],
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)


# =============================================================================
# 模型函数
# =============================================================================
async def llm_model_func(
    prompt: str, system_prompt: str = None, history_messages: Dict = None, **kwargs
) -> str:
    """OpenAI兼容API的LLM函数"""
    return await openai_complete_if_cache(
        get_required_env("OPENAI_MODEL"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=get_required_env("OPENAI_API_BASE"),
        api_key=get_required_env("OPENAI_API_KEY"),
        **kwargs,
    )


async def vllm_embedding_func(texts):
    """VLLM向量嵌入函数"""
    embed_url = get_required_env("VLLM_EMBED_URL")
    embed_model = get_required_env("VLLM_EMBED_MODEL")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{embed_url}/embeddings",
            json={"input": texts, "model": embed_model},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            return np.array(
                [item["embedding"] for item in result["data"]], dtype=np.float32
            )


async def vllm_reranker_func(query, documents, top_k=None, **kwargs):
    """VLLM Reranker函数"""
    rerank_url = get_required_env("VLLM_RERANK_URL")
    rerank_model = get_required_env("VLLM_RERANK_MODEL")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{rerank_url}/rerank",
            json={"model": rerank_model, "query": query, "documents": documents},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            reranked = []
            for item in result.get("results", []):
                idx = item["index"]
                reranked.append(
                    {
                        "doc_id": idx,
                        "index": idx,
                        "relevance_score": item["relevance_score"],
                        "text": item.get("document", {}).get("text", documents[idx]),
                    }
                )
            return reranked[:top_k] if top_k else reranked


def get_embedding_func():
    """创建 EmbeddingFunc"""
    embed_dim = int(get_required_env("VLLM_EMBED_DIM"))
    return EmbeddingFunc(
        embedding_dim=embed_dim, max_token_size=8192, func=vllm_embedding_func
    )


# =============================================================================
# 主函数
# =============================================================================
async def demo_streaming_query():
    """演示流式查询 + 召回数据"""
    print("=" * 70)
    print("流式查询示例 - 流式输出 + 召回数据")
    print("=" * 70)

    # 1. 验证环境变量配置
    print("\n🔍 验证环境变量配置...")
    validate_required_env()

    # 2. 配置 RAGAnythingConfig
    config = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage_db_example"),
    )

    # 3. 数据库配置
    milvus_config = {
        "uri": get_required_env("MILVUS_URI"),
        "user": get_required_env("MILVUS_USER"),
        "password": get_required_env("MILVUS_PASSWORD"),
        "db_name": get_required_env("MILVUS_DB_NAME"),
    }

    logger.info("🔧 初始化 RAGAnything（数据库后端）...")

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=get_embedding_func(),
        lightrag_kwargs={
            "kv_storage": "MongoKVStorage",
            "vector_storage": "MilvusVectorDBStorage",
            "doc_status_storage": "MongoDocStatusStorage",
            "graph_storage": "Neo4JStorage",
            "rerank_model_func": vllm_reranker_func,
            "vector_db_storage_cls_kwargs": milvus_config,
            "cosine_better_than_threshold": 0.5,
            "min_rerank_score": 0.3,
            "enable_llm_cache": False,  # 禁用缓存以测试流式输出
        },
    )

    # 4. 测试查询
    queries = [
        # "在集合中有哪些常用的术语？",
        # "数学必修第一册包含哪些内容？",
        # "北京今天天气怎么样",
        # "请简单给我介绍集合的概念",
        "怎么样毒死一只老鼠"
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"🔍 查询: {query}")
        print(f"{'=' * 70}\n")

        full_answer = ""
        sources_data = None

        try:
            async for chunk in rag.aquery_stream_with_sources(query, mode="hybrid"):
                if chunk["type"] == "sources_info":
                    info = chunk["content"]
                    print("📊 召回数据统计:")
                    print(f"  - 实体: {info['entities_count']}")
                    print(f"  - 关系: {info['relationships_count']}")
                    print(f"  - 文本块: {info['chunks_count']}")
                    print("\n💬 答案流式输出:\n")
                elif chunk["type"] == "chunk":
                    content = chunk["content"]
                    print(content, end="", flush=True)
                    full_answer += content
                elif chunk["type"] == "sources":
                    sources_data = chunk["content"]
                elif chunk["type"] == "error":
                    print(f"\n❌ 错误: {chunk['content']}")

            # 显示完整来源数据
            if sources_data:
                print("\n\n" + "=" * 70)
                print("📚 完整召回数据:")
                print("=" * 70)

                if sources_data.get("entities"):
                    print(f"\n🔹 实体 ({len(sources_data['entities'])} 个):")
                    for i, entity in enumerate(sources_data["entities"][:5]):
                        print(
                            f"  {i+1}. {entity['entity_name']} ({entity['entity_type']})"
                        )
                    if len(sources_data["entities"]) > 5:
                        print(f"  ... 还有 {len(sources_data['entities']) - 5} 个")

                if sources_data.get("relationships"):
                    print(f"\n🔗 关系 ({len(sources_data['relationships'])} 个):")
                    for i, rel in enumerate(sources_data["relationships"][:5]):
                        src = rel.get("src_id", "?")[:30]
                        tgt = rel.get("tgt_id", "?")[:30]
                        desc = rel.get("description", "")[:50]
                        print(f"  {i+1}. {src}... -> {tgt}...: {desc}...")
                    if len(sources_data["relationships"]) > 5:
                        print(f"  ... 还有 {len(sources_data['relationships']) - 5} 个")

                if sources_data.get("chunks"):
                    print(f"\n📄 文本块 ({len(sources_data['chunks'])} 个):")
                    for i, chunk in enumerate(sources_data["chunks"][:3]):
                        chunk_id = chunk.get("chunk_id", "unknown")[:20]
                        similarity = chunk.get("similarity", 0)
                        print(f"  {i+1}. {chunk_id}... (score: {similarity:.2f})")
                    if len(sources_data["chunks"]) > 3:
                        print(f"  ... 还有 {len(sources_data['chunks']) - 3} 个")

        except Exception as e:
            logger.error(f"查询失败: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ 示例执行完成!")
    print("=" * 70)


if __name__ == "__main__":
    configure_logging()

    print("\n流式查询示例")
    print("使用 MongoDB + Neo4j + Milvus + VLLM")
    print("=" * 70)

    # 运行主函数
    try:
        asyncio.run(demo_streaming_query())
    except ValueError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        exit(1)
