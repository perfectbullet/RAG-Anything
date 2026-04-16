#!/usr/bin/env python
"""
示例脚本：使用数据库后端进行内容查询

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
VLLM_EMBED_URL=http://192.168.8.234:8092/v1
VLLM_EMBED_MODEL=BAAI/bge-m3
VLLM_EMBED_DIM=1024

# Reranker
VLLM_RERANK_URL=http://192.168.8.234:8091/v1
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

from raganything.utils import (
    validate_required_env_vars,
    get_required_env,
    ProgressMessage
)

from raganything.query_prompts import get_chinese_query_prompt

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


# Validate environment using utility function
validate_required_env_vars(REQUIRED_ENV_VARS)


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
            # 截断过长的值（特别是 image_data 和 messages）
            if len(value) > self.max_length:
                return f"'{key}': '{value[:self.max_length]}...<truncated>'"
            return match.group(0)

        # 匹配 'key': 'value' 或 "key": "value" 格式
        pattern = r"['\"](\w*(?:image_data|messages|content)\w*)['\"]:\s*['\"]([^'\"]*)['\"]"
        return re.sub(pattern, truncate_match, text)


def configure_logging():
    """配置日志系统"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "insert_with_databases_example.log"))
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # 创建日志过滤器
    # long_data_filter = LongDataFilter(max_length=200)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s: %(message)s", 
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
            },
            "filters": {
                "long_data_filter": {
                    "()": "insert_with_databases_example.LongDataFilter",
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
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                    "filters": ["long_data_filter"],
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
    # 从 kwargs 中移除 image_data 和 messages，避免传递给 openai_complete_if_cache
    kwargs.pop("image_data", None)
    kwargs.pop("messages", None)

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
            # 将 image_data 转换为标准的 OpenAI messages 格式
            image_messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                },
            ]
            # 过滤掉 None 值
            image_messages = [m for m in image_messages if m is not None]
            result = await openai_complete_if_cache(
                get_required_env("VISION_MODEL"),
                "",
                system_prompt=None,  # 已在 messages 中
                messages=image_messages,
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
                        "relevance_score": item["relevance_score"],
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
# 主函数
# =============================================================================
async def main():
    """主函数"""
    print("=" * 70)
    print("RAGAnything 数据库后端示例 - 使用测试数据")
    print("=" * 70)

    # 1. 验证环境变量配置
    print("\n🔍 验证环境变量配置...")
    validate_required_env_vars(REQUIRED_ENV_VARS)

    # 2. 配置 RAGAnythingConfig
    config = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage_db_example"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=True,
    )

    # 3. 初始化 RAGAnything（带数据库后端）
    # 隐藏密码显示 MongoDB URI
    mongo_display = get_required_env("MONGO_URI")
    if "@" in mongo_display:
        # 隐藏密码部分
        parts = mongo_display.split("@")
        auth = parts[0].split("://")[-1]
        if ":" in auth:
            username = auth.split(":")[0]
            mongo_display = f"{parts[0].split('://')[0]}//{username}:***@{parts[1]}"
    
    logger.info("🔧 初始化 RAGAnything（数据库后端）...")
    logger.info("📦 存储配置:")
    logger.info(f"   文档状态存储 (MongoDB): {mongo_display}")
    logger.info(f"   向量存储 (Milvus): {get_required_env('MILVUS_URI')}")
    logger.info(f"   图存储 (Neo4j): http://{get_required_env('NEO4J_URI').replace('bolt://', '').replace(':7687', '')}:7474")
    logger.info(f"   Reranker (VLLM): {get_required_env('VLLM_RERANK_URL')}")
    logger.info(f"   OPENAI_API_BASE (VLLM): {get_required_env('OPENAI_API_BASE')}")
    logger.info(f"   OPENAI_MODEL (VLLM): {get_required_env('OPENAI_MODEL')}")

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
            # LightRAG 配置 (注意: 使用 cosine_better_than_threshold 而不是 cosine_threshold)
            "cosine_better_than_threshold": 0.6,  # 向量相似度阈值
            "min_rerank_score": 0.8,  # 过滤 rerank 分数低于 min_rerank_score的 chunks
            # 缓存开关
            "enable_llm_cache": True,
            # 语言配置
            "addon_params": {
                "language": "Chinese",
                "entity_types": ["organization", "person", "location", "event", "concept", "method"]
            },
        },
    )

    # 4. 初始化 lightrag（执行简单查询触发初始化）
    logger.info("\n🔧 初始化 LightRAG...")
    try:
        await rag._ensure_lightrag_initialized()
        await rag.lightrag.initialize_storages()
        
        # await rag.aquery("test", mode="hybrid")
        logger.info(f"{ProgressMessage.COMPLETED} LightRAG 初始化完成")
    except Exception as e:
        logger.warning(f"⚠️  查询初始化失败（可能没有数据）: {e}")

    # 5. 示例查询
    logger.info("\n🔍 执行示例查询...")
    test_queries = [
        # "请帮我讲解集合的概念.",
        "一个庭审证明",
        # "集合中元素是什么？"
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[查询]: {query}")
        logger.info(f"{'=' * 70}")

        # 流式查询，实时显示答案并提供来源信息
        async for chunk in rag.aquery_stream_with_sources(
            query,
            mode="hybrid",
            system_prompt=get_chinese_query_prompt(),  # 中文系统提示词（无 References）
            top_k=2, # 召回实体/关系数量
            chunk_top_k=3,  # (默认10) - 召回文档块数量
            enable_rerank=True # (默认True) - 是否启用重排序
        ):
            chunk_type = chunk.get("type")
            content = chunk.get("content")
            if chunk_type == "sources_info":
                # 检索信息摘要
                content: dict
                logger.info(f"📊 检索到: {content['entities_count']} 个实体, "
                           f"{content['relationships_count']} 个关系, "
                           f"{content['chunks_count']} 个文档块")

            elif chunk_type == "chunk":
                # 实时显示答案内容
                print(content, end="", flush=True)
            elif chunk_type == "sources":
                # 完整的来源数据
                sources = content
                print("\n\n" + "-" * 50)
                print("📚 来源信息:")
                
                # 【调试代码】
                from datetime import datetime
                import json
                # 保存 sources 为 JSON 文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_file_path = f"json格式数据/{timestamp}_sources.json"
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(sources, f, ensure_ascii=False, indent=2)
                print(f"📁 来源数据已保存到: {json_file_path}")
                
                # 显示实体
                if sources.get("entities"):
                    print(f"\n  实体 ({len(sources['entities'])} 个):")
                    for entity in sources["entities"][:5]:  # 只显示前5个
                        print(f"    - {entity.get('entity_name', 'N/A')}")
                    if len(sources["entities"]) > 5:
                        print(f"    ... 还有 {len(sources['entities']) - 5} 个实体\n\n\n")

                # 显示文档块
                if sources.get("chunks"):
                    print(f"\n  文档块 ({len(sources['chunks'])} 个):")
                    for i, chunk in enumerate(sources["chunks"][:3], 1):  # 只显示前3个
                        content_preview = chunk.get("content", "")
                        rerank_score = chunk.get("rerank_score")
                        score_str = f" [重排分数: {rerank_score:.4f}]" if rerank_score is not None else "重排分数: no score"
                        print(f"[{i}] {score_str}")
                        print(f"[{i}] {content_preview}")
                    if len(sources["chunks"]) > 3:
                        print(f"    ... 还有 {len(sources['chunks']) - 3} 个文档块")
            
            elif chunk_type == "error":
                logger.error(f"❌ 查询出错: {content}")
        print()  # 换行

    logger.info("\n" + "=" * 70)
    logger.info(f"{ProgressMessage.COMPLETED} 示例执行完成!")
    logger.info("=" * 70)


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
