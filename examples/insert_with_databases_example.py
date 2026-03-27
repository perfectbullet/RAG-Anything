#!/usr/bin/env python
"""
示例脚本：使用数据库后端插入内容列表

本示例展示如何：
1. 从 .env 文件读取所有配置（无默认值，缺少配置则报错）
2. 使用 MongoDB、Neo4j、Milvus 作为数据库后端
3. 使用 VLLM 作为向量模型
4. 使用 Reranker 进行结果重排序
5. 从现有 content_list.json 加载测试数据并处理 img_path 路径

环境变量配置（必需）：
# OpenAI兼容API
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=Qwen/Qwen2.5-72B-Instruct
VISION_MODEL=Qwen/Qwen3-VL-32B-Instruct

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
import json
import asyncio
import logging
import logging.config
import contextlib
import io
from pathlib import Path
from typing import List, Dict

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import aiohttp
import numpy as np
from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
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
    "VISION_MODEL": "Vision model name",
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
def configure_logging():
    """配置日志系统"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "insert_with_databases_example.log"))
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


# =============================================================================
# 模型函数
# =============================================================================
async def llm_model_func(
    prompt: str, system_prompt: str = None, history_messages: List[Dict] = None, **kwargs
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


async def vision_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    image_data: str = None,
    messages: List[Dict] = None,
    **kwargs,
) -> str:
    """
    视觉模型函数，用于图像分析和VLM增强查询

    支持两种调用模式：
    1. messages格式：多模态VLM增强查询（包含文本和图像的混合消息）
    2. image_data格式：图像处理（base64编码的图像数据）
    """
    # 抑制 stdout（避免打印 image_data）
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        if messages:
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                "",
                system_prompt=system_prompt,
                messages=messages,
                base_url=get_required_env("OPENAI_API_BASE"),
                api_key=get_required_env("OPENAI_API_KEY"),
                **kwargs,
            )
        elif image_data:
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                prompt,
                system_prompt=system_prompt,
                image_data=image_data,
                base_url=get_required_env("OPENAI_API_BASE"),
                api_key=get_required_env("OPENAI_API_KEY"),
                **kwargs,
            )
        else:
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=get_required_env("OPENAI_API_BASE"),
                api_key=get_required_env("OPENAI_API_KEY"),
                **kwargs,
            )

    return result


async def vllm_embedding_func(texts: List[str]) -> List[List[float]]:
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


async def vllm_reranker_func(
    query: str, documents: List[str], top_k: int = None, **kwargs
) -> List[Dict]:
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
                        "score": item["relevance_score"],
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
# 测试数据加载
# =============================================================================
def load_test_content_list():
    """
    从现有 content_list.json 加载测试数据并处理 img_path

    Returns:
        List[Dict]: 处理后的 content_list，img_path 已转换为绝对路径
    """
    vlm_base_dir = Path("/home/zj/RAG-Anything/mineru-out/01-math-16pages-part1-page1-16/vlm")
    json_path = vlm_base_dir / "01-math-16pages-part1-page1-16_content_list.json"

    if not json_path.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {json_path}")

    logger.info(f"📂 加载测试数据: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)

    # 修复 img_path 相对路径
    image_count = 0
    for item in content_list:
        if item.get("type") == "image" and "img_path" in item:
            img_path = item["img_path"]
            # 如果是相对路径，拼接为绝对路径
            if not os.path.isabs(img_path):
                item["img_path"] = str(vlm_base_dir / img_path)
                image_count += 1

    # 统计内容类型
    type_counts = {}
    for item in content_list:
        t = item.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info(f"✅ 加载了 {len(content_list)} 条测试内容")
    logger.info(f"📊 内容类型统计: {type_counts}")
    logger.info(f"🖼️  修复了 {image_count} 个图像路径")

    return content_list


# =============================================================================
# 主函数
# =============================================================================
async def main():
    """主函数"""
    print("=" * 70)
    print("RAGAnything 数据库后端示例 - 使用测试数据")
    print("=" * 70)

    # 1. 验证环境变量配置
    print("\n🔍 验证环境变量配置...")
    validate_required_env()

    # 2. 加载测试数据
    print("\n📂 加载测试数据...")
    content_list = load_test_content_list()

    # 3. 配置 RAGAnything
    config = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage_db_example"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=True,
    )

    # 4. 初始化 RAGAnything（带数据库后端）
    logger.info("🔧 初始化 RAGAnything（数据库后端）...")
    logger.info("📦 存储配置:")
    logger.info("   KV 存储:          MongoDB")
    logger.info("   向量存储:         Milvus")
    logger.info("   文档状态存储:     MongoDB")
    logger.info("   图存储:           Neo4j")
    logger.info("   Reranker:         VLLM")

    # Note: Database connection parameters are read from environment variables by LightRAG storage backends
    # MongoDB: MONGO_URI, MONGO_DATABASE
    # Neo4j: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    # Milvus: MILVUS_URI, MILVUS_USER, MILVUS_PASSWORD, MILVUS_DB_NAME
    # These are already loaded from .env file at line 62 and validated by validate_required_env()

    # For Milvus, we pass explicit parameters via vector_db_storage_cls_kwargs
    # Using username and password authentication instead of token
    milvus_config = {
        "uri": get_required_env("MILVUS_URI"),
        "user": get_required_env("MILVUS_USER"),
        "password": get_required_env("MILVUS_PASSWORD"),
        "db_name": get_required_env("MILVUS_DB_NAME"),
    }

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=get_embedding_func(),
        lightrag_kwargs={
            "kv_storage": "MongoKVStorage",
            "vector_storage": "MilvusVectorDBStorage",
            "doc_status_storage": "MongoDocStatusStorage",
            "graph_storage": "Neo4JStorage",
            "rerank_model_func": vllm_reranker_func,
            "vector_db_storage_cls_kwargs": milvus_config,
        },
    )

    # 5. 插入内容列表
    logger.info("\n📝 插入内容列表到数据库...")
    await rag.insert_content_list(
        content_list=content_list,
        file_path="01-math-16pages-part1-page1-16.pdf",
        doc_id="math-test-doc-001",
        display_stats=True,
    )
    logger.info("✅ 内容列表插入完成!")

    # 6. 示例查询
    logger.info("\n🔍 执行示例查询...")
    test_queries = [
        "数学必修第一册包含哪些内容？",
        "文档中有哪些重要的数学概念？",
    ]

    for query in test_queries:
        logger.info(f"\n[查询]: {query}")
        result = await rag.aquery(query, mode="hybrid")
        logger.info(f"[回答]: {result[:300]}...")

    # 7. 数据库信息
    logger.info("\n" + "=" * 70)
    logger.info("✅ 示例执行完成!")
    logger.info("=" * 70)
    logger.info("📌 数据库连接信息:")

    # 隐藏密码显示 MongoDB URI
    mongo_display = get_required_env("MONGO_URI")
    if "@" in mongo_display:
        # 隐藏密码部分
        parts = mongo_display.split("@")
        auth = parts[0].split("://")[-1]
        if ":" in auth:
            username = auth.split(":")[0]
            mongo_display = f"{parts[0].split('://')[0]}//{username}:***@{parts[1]}"

    logger.info(f"   MongoDB:    {mongo_display}")
    logger.info(f"   Neo4j:      {get_required_env('NEO4J_URI')}")
    logger.info(f"   Milvus:     {get_required_env('MILVUS_URI')}")
    logger.info(f"   Neo4j 浏览: http://{get_required_env('NEO4J_URI').replace('bolt://', '').replace(':7687', '')}:7474")


if __name__ == "__main__":
    # 配置日志
    configure_logging()

    print("\nRAGAnything 数据库后端示例")
    print("使用 MongoDB + Neo4j + Milvus + VLLM")
    print("=" * 70)

    # 运行主函数
    try:
        asyncio.run(main())
    except ValueError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback

        logger.error(traceback.format_exc())
        exit(1)
