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
from typing import List, Dict, Optional

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import aiohttp
import numpy as np
from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import generate_doc_id_from_path
from raganything.content_list_v2_splitter import ContentListV2Splitter
from raganything.utils import (
    validate_required_env_vars,
    get_required_env,
    load_content_list_v2,
    ContentProcessingProgressTracker,
    RetryConfig,
    ProgressMessage,
)

# Load .env file
load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"), override=False)


# =============================================================================
# 错误类型检测和智能延迟
# =============================================================================
def is_rate_limit_error(error: Exception) -> bool:
    """检测是否为速率限制错误（429、TPM/RPM限制等）"""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    rate_limit_indicators = [
        'rate limit', 'ratelimit', 'rate_limit',
        '429', 'too many requests',
        'tpm limit', 'rpm limit',
        'quota exceeded', 'quota_exceeded',
    ]

    return any(indicator in error_str or indicator in error_type
               for indicator in rate_limit_indicators)


def is_timeout_error(error: Exception) -> bool:
    """检测是否为超时错误"""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    timeout_indicators = [
        'timeout', 'timed out', 'time out',
        'readtimeout', 'connecttimeout',
        'async timeout', 'timeouterror',
        'gateway timeout', '504',
    ]

    return any(indicator in error_str or indicator in error_type
               for indicator in timeout_indicators)


def calculate_smart_delay(
    retry: int,
    error: Exception,
    base_delay: int = 5,
    rate_limit_multiplier: float = 3.0,
    timeout_multiplier: float = 2.0,
    max_delay: int = 120
) -> int:
    """
    计算智能重试延迟时间

    Args:
        retry: 当前重试次数（从0开始）
        error: 捕获的异常
        base_delay: 基础延迟时间（秒）
        rate_limit_multiplier: 速率限制错误的延迟倍数
        timeout_multiplier: 超时错误的延迟倍数
        max_delay: 最大延迟时间（秒）

    Returns:
        计算后的延迟时间（秒）
    """
    # 基础延迟：指数增长
    delay = base_delay * (2 ** retry)

    # 根据错误类型调整延迟
    if is_rate_limit_error(error):
        delay *= rate_limit_multiplier
        logger.warning(f"🚦 检测到速率限制错误，延迟增加到 {delay:.0f} 秒")
    elif is_timeout_error(error):
        delay *= timeout_multiplier
        logger.warning(f"⏱️  检测到超时错误，延迟增加到 {delay:.0f} 秒")

    # 限制最大延迟
    return min(int(delay), max_delay)


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
                "default": {"format": "%(asctime)s - %(levelname)s: %(message)s", "datefmt": "%H:%M:%S"},
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
def load_test_content_list(data_dir: str):
    """
    从指定目录加载 content_list_v2.json 测试数据并处理图片路径

    Args:
        data_dir: 数据目录路径
                 - 指定目录: 自动在该目录下查找 *_content_list_v2.json 文件

    Returns:
        tuple: (content_list_v2, json_path) - 处理后的 content_list_v2 和 json 文件路径
    """
    # 使用新的 utility 函数
    return load_content_list_v2(data_dir)


# =============================================================================
# 进度跟踪管理（断点续传）
# =============================================================================
class ProgressTracker:
    """处理进度跟踪器，支持断点续传 - 使用新的 utility class"""

    def __init__(self, progress_file: Optional[str] = None):
        """
        Args:
            progress_file: 进度文件路径，默认为 ./insert_progress.json
        """
        self.tracker = ContentProcessingProgressTracker(progress_file)

    def _load_progress(self) -> dict:
        """加载进度文件"""
        if Path(self.tracker_file).exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"📂 加载进度文件: {self.tracker_file}")
                started_count = len(progress.get('started', []))
                completed_count = len(progress.get('completed', []))
                failed_count = len(progress.get('failed', []))
                logger.info(f"   上次处理: {started_count} 个进行中, {completed_count} 个已完成, {failed_count} 个失败")
                return progress
            except Exception as e:
                logger.warning(f"⚠️  加载进度文件失败: {e}，将创建新文件")
                return {"started": [], "completed": [], "failed": [], "document_info": {}}
        return {"started": [], "completed": [], "failed": [], "document_info": {}}

    def _save_progress(self):
        """保存进度文件"""
        try:
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracker, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️  保存进度文件失败: {e}")

    def set_document_info(self, file_path: str, total_sections: int):
        """设置文档信息"""
        self.tracker["document_info"] = {
            "file_path": file_path,
            "total_sections": total_sections,
            "last_update": str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else None,
        }
        self._save_progress()

    def is_started(self, doc_id: str) -> bool:
        """检查是否已开始处理"""
        return doc_id in self.tracker.get("started", [])

    def is_completed(self, doc_id: str) -> bool:
        """检查是否已完成"""
        return doc_id in self.tracker.get("completed", [])

    def mark_started(self, doc_id: str, title: Optional[str] = None):
        """标记为开始处理"""
        if "started" not in self.tracker:
            self.tracker["started"] = []
        if doc_id not in self.tracker["started"]:
            self.tracker["started"].append(doc_id)
            self._save_progress()
            logger.info(f"  💾 已标记开始: {title or doc_id}")

    def mark_completed(self, doc_id: str, title: Optional[str] = None):
        """标记为已完成"""
        if "completed" not in self.tracker:
            self.tracker["completed"] = []
        if doc_id not in self.tracker["completed"]:
            self.tracker["completed"].append(doc_id)
            # 从进行中列表移除
            self.tracker["started"] = [d for d in self.tracker.get("started", []) if d != doc_id]
            # 从失败列表中移除（如果之前失败过）
            self.tracker["failed"] = [f for f in self.tracker.get("failed", []) if f["doc_id"] != doc_id]
            self._save_progress()
            logger.info(f"  💾 已标记完成: {title or doc_id}")

    def mark_failed(self, doc_id: str, title: str, error: str, retry_count: int = 0, max_retries: int = 2):
        """标记为失败"""
        if "failed" not in self.tracker:
            self.tracker["failed"] = []

        # 更新或添加失败记录
        failed_record = {
            "doc_id": doc_id,
            "title": title,
            "error": error,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "last_failed": str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else None,
        }

        # 从进行中列表移除
        self.tracker["started"] = [d for d in self.tracker.get("started", []) if d != doc_id]
        # 移除旧的失败记录（如果有）
        self.tracker["failed"] = [f for f in self.tracker["failed"] if f["doc_id"] != doc_id]
        self.tracker["failed"].append(failed_record)
        self._save_progress()
        logger.info(f"  💾 已记录失败: {title}")

    def get_failed_sections(self) -> list:
        """获取失败的section列表"""
        return self.tracker.get("failed", [])

    def get_summary(self) -> dict:
        """获取进度摘要"""
        total = self.tracker["document_info"].get("total_sections", 0)
        started = len(self.tracker.get("started", []))
        completed = len(self.tracker.get("completed", []))
        failed = len(self.tracker.get("failed", []))
        return {
            "total": total,
            "started": started,
            "completed": completed,
            "failed": failed,
            "pending": total - completed,
        }


# =============================================================================
# 按章节处理大文件
# =============================================================================
async def insert_large_file_with_splitter(
    rag,
    content_list_v2: list,
    file_path: str,
    doc_id_prefix: str,
    max_retries: int = 2,
    progress_file: Optional[str] = None,
    batch_delay: float = 1.0,
):
    """
    使用 ContentListV2Splitter 按小节处理大文件

    支持断点续传和失败重试，使用JSON文件记录进度

    Args:
        rag: RAGAnything 实例
        content_list_v2: v2 格式的 content_list (二维数组)
        file_path: 文件路径
        doc_id_prefix: doc_id 前缀
        max_retries: 失败重试次数
        progress_file: 进度文件路径，默认为 ./insert_progress.json

    Returns:
        dict: 处理结果 {"success": [], "failed": [], "skipped": []}
    """
    splitter = ContentListV2Splitter()

    # 按小节拆分
    sections = splitter.split_by_chapters(content_list_v2, doc_id_prefix)

    # 初始化进度跟踪器
    tracker = ProgressTracker(progress_file)
    tracker.set_document_info(file_path, len(sections))

    logger.info("\n" + "=" * 60)
    logger.info(f"📚 检测到 {len(sections)} 个小节")
    summary = tracker.get_summary()
    if summary["started"] > 0 or summary["completed"] > 0 or summary["failed"] > 0:
        logger.info(f"📊 进度: 进行中 {summary['started']}, 已完成 {summary['completed']}, 失败 {summary['failed']}, 待处理 {summary['pending']}")
    logger.info("=" * 60)

    results = {"success": [], "failed": [], "skipped": []}

    for idx, section in enumerate(sections, 1):
        title = section['title']
        doc_id = section['doc_id']
        content = section['content']
        page_range = section['page_range']

        # 检查是否已完成（通过进度文件）
        if tracker.is_completed(doc_id):
            logger.info(f"\n小节 {idx}/{len(sections)}: {title}")
            logger.info(f"  ⏭️  已完成（进度文件），跳过")
            results["skipped"].append(doc_id)
            continue

        logger.info(f"\n小节 {idx}/{len(sections)}: {title}")
        logger.info(f"  页码: {page_range[0]}-{page_range[1]}")
        logger.info(f"  内容: {len(content)} items")
        logger.info(f"  doc_id: {doc_id}")

        # 如果是进行中状态（上次可能中断），先删除旧数据再重新处理
        # if tracker.is_started(doc_id):
        #     logger.warning(f"  {ProgressMessage.FAILED} 上次处理未完成，将删除旧数据后重新处理")
        #     try:
        #         await rag.adelete_by_doc_id(doc_id)
        #     except Exception as e:
        #         logger.warning(f"  ⚠️  删除旧数据失败（可能不存在）: {e}")

        # 标记为开始处理
        tracker.mark_started(doc_id, title)

        # 重试逻辑
        for retry in range(max_retries + 1):
            try:

                await rag.insert_content_list(
                    content_list=content,
                    file_path=file_path,
                    doc_id=doc_id,
                    display_stats=False,
                )
                logger.info(f"  ✅ 完成")
                tracker.mark_completed(doc_id, title)
                results["success"].append(doc_id)
                break
            except Exception as e:
                if retry < max_retries:
                    # 使用智能延迟策略
                    wait_time = calculate_smart_delay(
                        retry=retry,
                        error=e,
                        base_delay=5,
                        rate_limit_multiplier=3.0,
                        timeout_multiplier=2.0,
                        max_delay=120
                    )
                    logger.warning(f"  ⚠️  失败: {e}")
                    logger.info(f"     {wait_time}秒后重试 ({retry + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"  ❌ 最终失败: {e}")
                    tracker.mark_failed(doc_id, title, str(e), retry, max_retries)
                    results["failed"].append({"doc_id": doc_id, "title": title, "error": str(e)})

    # 批处理延迟 - 避免API速率限制
        if idx % 10 == 0 and idx < len(sections) - 1:
            logger.info(f"  ⏱️  批处理延迟 {batch_delay}秒...")
            await asyncio.sleep(batch_delay)

    # 最终汇总
    logger.info("\n" + "=" * 60)
    logger.info("📊 处理汇总:")
    logger.info(f"  ✅ 成功: {len(results['success'])} 个小节")
    logger.info(f"  ⏭️  跳过: {len(results['skipped'])} 个小节")
    logger.info(f"  ❌ 失败: {len(results['failed'])} 个小节")

    # 显示进度文件摘要
    final_summary = tracker.get_summary()
    logger.info(f"\n📂 进度文件摘要 ({tracker.progress_file}):")
    logger.info(f"  总数: {final_summary['total']}")
    logger.info(f"  进行中: {final_summary['started']}")
    logger.info(f"  已完成: {final_summary['completed']}")
    logger.info(f"  失败: {final_summary['failed']}")
    logger.info(f"  待处理: {final_summary['pending']}")

    if results["failed"]:
        logger.warning(f"\n本次失败的小节:")
        for f in results["failed"]:
            logger.warning(f"  - {f.get('title', f['doc_id'])}: {f.get('error', 'Unknown error')[:100]}")

    logger.info(f"\n💡 下次运行将自动跳过已完成的小节，并重试失败的小节")
    logger.info(f"   如需重新开始，请删除进度文件: {tracker.progress_file}")

    return results


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
            "cosine_better_than_threshold": 0.5,  # 向量相似度阈值
            "min_rerank_score": 0.3,  # 过滤 rerank 分数低于 0.2 的 chunks
            # 语言配置
            "addon_params": {
                "language": "Chinese",  # 知识图谱构建和查询的语言
                "entity_types": ["organization", "person", "location", "event", "concept", "method"]
            },
        },
    )

    # 3.5. 初始化 lightrag（执行简单查询触发初始化）
    logger.info("\n🔧 初始化 LightRAG...")
    try:
        await rag.aquery("test", mode="hybrid")
        logger.info(f"{ProgressMessage.COMPLETED} LightRAG 初始化完成")
    except Exception as e:
        logger.warning(f"⚠️  查询初始化失败（可能没有数据）: {e}")

    # 5. 插入内容列表 - 使用小节拆分器处理大文件
    logger.info("📝 插入内容列表到数据库（使用小节拆分器）...")
    content_list_v2, json_file_path = load_test_content_list('/data/metahuman_work/ZengKingMorphe/Digital-Human-Disciplinary-Dataset/高中数学相关资料内容/math-data-v1/04高中数学选择性必修第二册/vlm/')

    # 生成 doc_id 前缀
    doc_id_prefix = generate_doc_id_from_path(json_file_path)
    logger.info(f"📋 doc_id 前缀: {doc_id_prefix}")

    # 使用拆分器按小节处理
    results = await insert_large_file_with_splitter(
        rag=rag,
        content_list_v2=content_list_v2,
        file_path=json_file_path.as_posix(),
        doc_id_prefix=doc_id_prefix,
        max_retries=2,
        batch_delay=2.0,  # 批处理间延迟2秒，避免API速率限制
    )

    logger.info(f"{ProgressMessage.COMPLETED} 内容列表插入完成!")

    # 显示处理结果
    if results["success"]:
        logger.info(f"✅ 成功插入 {len(results['success'])} 个小节")
    if results["skipped"]:
        logger.info(f"⏭️  跳过 {len(results['skipped'])} 个已存在小节")
    if results["failed"]:
        logger.warning(f"❌ {len(results['failed'])} 个小节处理失败")

    # 6. 示例查询
    logger.info("\n🔍 执行示例查询...")
    test_queries = [
        "数学必修第一册包含哪些内容？",
        "文档中有哪些重要的数学概念？",
        "在集合中有哪些常用的术语？",
        "今天北京天气怎么样"
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[查询]: {query}")
        logger.info(f"{'=' * 70}")
        # 再获取最终答案
        result = await rag.aquery(query, mode="hybrid", vlm_enhanced=False)

        logger.info(f"\n[回答]:\n{result}")

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
