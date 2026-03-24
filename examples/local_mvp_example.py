#!/usr/bin/env python3
"""
RAG-Anything 本地MVP示例

使用本地模型和API构建最小化RAG系统：
- Ollama (qwen2.5:14b) - LLM
- VLLM (BAAI/bge-m3) - Embedding
- VLLM (bge-reranker-m3) - Reranker
- MinerU HTTP API - PDF解析

Usage:
    cd /home/zj/RAG-Anything
    /home/zj/miniconda3/envs/rag-anything/bin/python examples/local_mvp_example.py

Author: Claude + zj
Date: 2026-03-23
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional

# 配置日志 - 减少LightRAG的详细日志
logging.basicConfig(
    level=logging.WARNING,  # 默认WARNING级别
    format='%(message)s'   # 简化格式
)
logger = logging.getLogger(__name__)

# 为关键模块设置更严格的日志级别
for module in ['lightrag', 'nano_vectordb', 'raganything.custom_parsers']:
    logging.getLogger(module).setLevel(logging.ERROR)

# 但让我们的logger显示INFO
logger.setLevel(logging.INFO)

# =============================================================================
# 配置 - 根据本地环境修改
# =============================================================================

# 数据目录
MATH_DATA_DIR = Path("/home/zj/ZengKingMorphe/Digital-Human-Disciplinary-Dataset/高中数学相关资料内容")
JEWELRY_DATA_DIR = Path("/home/zj/ZengKingMorphe/Digital-Human-Disciplinary-Dataset/Jewelry_Crafts_Dataset")
WORKING_DIR = "./rag_mvp_storage"

# 本地服务配置
OLLAMA_BASE_URL = "http://192.168.8.230:11434"
OLLAMA_MODEL = "qwen2.5:14b"

VLLM_EMBED_URL = "http://192.168.8.233:8092/v1"
VLLM_EMBED_MODEL = "BAAI/bge-m3"
VLLM_EMBED_DIM = 1024

VLLM_RERANK_URL = "http://192.168.8.233:8091/v1"
VLLM_RERANK_MODEL = "bge-reranker-m3"

MINERU_API_URL = "http://192.168.8.233:8000"

# =============================================================================
# 导入 RAGAnything (本地模型兼容版本)
# =============================================================================

try:
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.custom_parsers import (
        MinerUHttpParser,
        OllamaLLMFunc,
        VLLMEmbeddingFunc,
        VLLMRerankerFunc,
    )
    from raganything.parser import register_parser
except ImportError as e:
    logger.error(f"导入RAGAnything失败: {e}")
    logger.error("请确保在正确的环境中运行: pip install -e '.[all]'")
    exit(1)


# =============================================================================
# LLM 函数 - Ollama
# =============================================================================

async def ollama_llm_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """Ollama LLM 函数."""
    ollama = OllamaLLMFunc(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
    )
    return await ollama(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# =============================================================================
# Embedding 函数 - VLLM (直接HTTP调用，避免openai_embed的默认1536维问题)
# =============================================================================

async def vllm_embedding_func(texts: List[str]) -> List[List[float]]:
    """VLLM embedding 函数 - 直接调用HTTP API."""
    import aiohttp
    import numpy as np

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VLLM_EMBED_URL}/embeddings",
            json={"input": texts, "model": VLLM_EMBED_MODEL},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            # VLLM返回格式: {"data": [{"embedding": [...], ...}, ...]}
            embeddings = [item["embedding"] for item in result["data"]]
            # 返回numpy数组而不是列表
            return np.array(embeddings, dtype=np.float32)


def get_embedding_func():
    """创建LightRAG兼容的embedding函数."""
    from lightrag.utils import EmbeddingFunc
    return EmbeddingFunc(
        embedding_dim=VLLM_EMBED_DIM,
        max_token_size=8192,
        func=vllm_embedding_func,
    )


# =============================================================================
# Reranker 函数 - VLLM
# =============================================================================

async def vllm_reranker_func(
    query: str,
    documents: List[str],
    top_k: int = None,
    **kwargs,
) -> List[Dict]:
    """VLLM reranker 函数."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VLLM_RERANK_URL}/rerank",
            json={
                "model": VLLM_RERANK_MODEL,
                "query": query,
                "documents": documents,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()

            reranked = []
            for item in result.get("results", []):
                idx = item["index"]
                # VLLM格式: item["document"]["text"] 包含原始文本
                # 但我们也需要doc_id索引
                reranked.append({
                    "doc_id": idx,
                    "index": idx,  # 保留原始索引
                    "score": item["relevance_score"],
                    "text": item.get("document", {}).get("text", documents[idx]),
                })

            if top_k:
                reranked = reranked[:top_k]
            return reranked


# =============================================================================
# MVP 应用类
# =============================================================================

class LocalRAGMVP:
    """本地RAG MVP应用."""

    def __init__(self, data_dir: Path = None):
        """
        初始化MVP应用.

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir or MATH_DATA_DIR
        self.working_dir = WORKING_DIR
        self.rag = None

        # 注册自定义解析器
        register_parser("mineru-http", MinerUHttpParser)

    async def check_services(self) -> bool:
        """检查所有本地服务是否可用."""
        print("\n" + "=" * 60)
        print("检查本地服务")
        print("=" * 60)

        services = []

        # 检查 Ollama
        print(f"\n📡 Ollama: {OLLAMA_BASE_URL}")
        ollama = OllamaLLMFunc(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        if ollama.check_installation():
            print(f"   ✅ 可用 (模型: {OLLAMA_MODEL})")
            services.append(True)
        else:
            print(f"   ❌ 不可用")
            services.append(False)

        # 检查 VLLM Embedding
        print(f"\n📡 VLLM Embedding: {VLLM_EMBED_URL}")
        vllm_embed = VLLMEmbeddingFunc(
            base_url=VLLM_EMBED_URL,
            model=VLLM_EMBED_MODEL,
            embedding_dim=VLLM_EMBED_DIM,
        )
        if vllm_embed.check_installation():
            print(f"   ✅ 可用 (模型: {VLLM_EMBED_MODEL}, 维度: {VLLM_EMBED_DIM})")
            services.append(True)
        else:
            print(f"   ❌ 不可用")
            services.append(False)

        # 检查 VLLM Reranker
        print(f"\n📡 VLLM Reranker: {VLLM_RERANK_URL}")
        vllm_rerank = VLLMRerankerFunc(
            base_url=VLLM_RERANK_URL,
            model=VLLM_RERANK_MODEL,
        )
        if vllm_rerank.check_installation():
            print(f"   ✅ 可用 (模型: {VLLM_RERANK_MODEL})")
            services.append(True)
        else:
            print(f"   ❌ 不可用")
            services.append(False)

        # 检查 MinerU API
        print(f"\n📡 MinerU API: {MINERU_API_URL}")
        mineru = MinerUHttpParser(base_url=MINERU_API_URL)
        if mineru.check_installation():
            print(f"   ✅ 可用")
            services.append(True)
        else:
            print(f"   ❌ 不可用")
            services.append(False)

        all_ok = all(services)
        print("\n" + "=" * 60)
        if all_ok:
            print("✅ 所有服务正常!")
        else:
            print("❌ 部分服务不可用，请检查配置")
        print("=" * 60)

        return all_ok

    async def initialize(self):
        """初始化RAG系统."""
        print("\n初始化RAG系统...")

        # 抑制初始化日志
        logging.getLogger('lightrag').setLevel(logging.ERROR)
        logging.getLogger('nano_vectordb').setLevel(logging.ERROR)
        logging.getLogger('raganything').setLevel(logging.ERROR)

        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="mineru-http",
            parse_method="auto",
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=ollama_llm_func,
            embedding_func=get_embedding_func(),
            lightrag_kwargs={
                "rerank_model_func": vllm_reranker_func,
                # 增加超时时间 - 处理大文件需要更长时间
                "default_llm_timeout": 600,  # 10分钟 (默认180秒)
                # 减小chunk大小 - 避免单个chunk太大
                "chunk_token_size": 800,  # 默认1200
                "chunk_overlap_token_size": 100,  # 默认100
                # 减少并发处理数量 - 避免Ollama过载
                "llm_model_max_async": 2,  # 默认4
                "max_parallel_insert": 1,  # 默认2
            },
        )

        # 初始化存储
        await self.rag._ensure_lightrag_initialized()
        await self.rag.lightrag.initialize_storages()

        print(f"✅ RAG系统初始化完成!")
        print(f"   工作目录: {self.working_dir}")

    async def process_files(self, max_files: int = 1, file_pattern: str = "*.pdf"):
        """
        处理文件.

        Args:
            max_files: 最大处理文件数
            file_pattern: 文件匹配模式
        """
        if not self.rag:
            raise RuntimeError("RAG未初始化，请先调用initialize()")

        files = sorted(self.data_dir.glob(file_pattern))

        if not files:
            print(f"⚠️  未找到文件: {self.data_dir}/{file_pattern}")
            return

        print(f"\n📚 找到 {len(files)} 个文件")
        print(f"处理前 {max_files} 个文件...\n")

        # 临时抑制详细日志
        logging.getLogger('lightrag').setLevel(logging.ERROR)
        logging.getLogger('nano_vectordb').setLevel(logging.ERROR)

        for i, file_path in enumerate(files[:max_files]):
            print(f"{'=' * 60}")
            print(f"处理 {i + 1}/{min(max_files, len(files))}: {file_path.name}")
            print(f"文件大小: {file_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"正在处理... (这可能需要几分钟，请耐心等待)")
            print(f"{'=' * 60}")

            try:
                await self.rag.process_document_complete(
                    file_path=str(file_path),
                    display_stats=False,  # 不显示详细统计
                    lang="ch",
                )
                print(f"✅ 完成: {file_path.name}\n")
            except Exception as e:
                print(f"❌ 失败: {e}\n")

    async def process_markdown(self, max_files: int = 3):
        """处理Markdown文件（更快速）."""
        if not self.rag:
            raise RuntimeError("RAG未初始化，请先调用initialize()")

        md_files = sorted(self.data_dir.glob("*.md"))

        if not md_files:
            print(f"⚠️  未找到Markdown文件: {self.data_dir}")
            return

        print(f"\n📚 找到 {len(md_files)} 个Markdown文件")
        print(f"处理前 {max_files} 个文件...")
        print("⏳ 正在处理... (LLM提取实体需要时间，请耐心等待)\n")

        # 临时抑制详细日志
        logging.getLogger('lightrag').setLevel(logging.ERROR)
        logging.getLogger('nano_vectordb').setLevel(logging.ERROR)

        for i, md_file in enumerate(md_files[:max_files]):
            file_size = md_file.stat().st_size / 1024
            print(f"[{i + 1}/{min(max_files, len(md_files))}] {md_file.name} ({file_size:.1f} KB)...", end="", flush=True)

            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                content_list = [{"type": "text", "text": content, "page_idx": 0}]
                await self.rag.insert_content_list(
                    content_list=content_list,
                    file_path=str(md_file),
                    doc_id=md_file.stem,
                    display_stats=False,  # 不显示详细统计
                )
                print(" ✅")
            except Exception as e:
                print(f" ❌ ({e})")
        print()  # 换行

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """
        查询知识库.

        Args:
            question: 查询问题
            mode: 查询模式 (local, global, hybrid, naive)

        Returns:
            查询结果
        """
        if not self.rag:
            raise RuntimeError("RAG未初始化，请先调用initialize()")

        print(f"\n{'─' * 40}")
        print(f"问题: {question}")
        print(f"模式: {mode}")
        print(f"{'─' * 40}")

        try:
            result = await self.rag.aquery(question, mode=mode)
            print(f"\n回答:")
            print(result)
            return result
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def interactive_query(self):
        """交互式查询."""
        if not self.rag:
            raise RuntimeError("RAG未初始化，请先调用initialize()")

        print("\n" + "=" * 60)
        print("交互式查询模式 (输入 'quit' 退出)")
        print("=" * 60)

        while True:
            try:
                question = input("\n请输入问题: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("退出查询模式")
                    break

                if question:
                    await self.query(question)
            except KeyboardInterrupt:
                print("\n退出查询模式")
                break


# =============================================================================
# 主函数
# =============================================================================

async def math_mvp_demo():
    """数学资料MVP演示."""
    print("\n" + "=" * 60)
    print("RAG-Anything 本地MVP - 数学资料")
    print("=" * 60)

    app = LocalRAGMVP(data_dir=MATH_DATA_DIR)

    # 检查服务
    if not await app.check_services():
        print("\n❌ 服务检查失败，退出")
        return False

    # 初始化
    await app.initialize()

    # 检查是否有已处理的数据（检查是否有chunks）
    vdb_chunks_path = Path(app.working_dir) / "vdb_chunks.json"
    if vdb_chunks_path.exists() and vdb_chunks_path.stat().st_size > 100:
        print("\n⚠️  工作目录已有数据，跳过文档处理")
        print(f"   已有数据文件: {vdb_chunks_path.stat().st_size} 字节")
    else:
        # 处理文档 (PDF)
        # await app.process_files(max_files=1, file_pattern="*.pdf")

        # 或处理 Markdown (如果有的话)
        # await app.process_markdown(max_files=3)

        # 添加示例内容
        print("\n添加示例数学内容...")
        sample_content = """
        # 高中数学知识点

        ## 函数

        函数是数学中的基本概念。设A和B是两个非空集合，如果按照某种对应法则f，
        对于集合A中的每一个元素x，在集合B中都有唯一确定的元素y与之对应，
        则称f: A→B为从集合A到集合B的函数。

        ## 三角函数

        在直角三角形中：
        - 正弦：sin(θ) = 对边 / 斜边
        - 余弦：cos(θ) = 邻边 / 斜边
        - 正切：tan(θ) = 对边 / 邻边

        ## 导数

        导数表示函数在某点处的变化率。几何意义是曲线切线的斜率。

        f'(x) = lim_{Δx→0} [f(x + Δx) - f(x)] / Δx
        """

        content_list = [{"type": "text", "text": sample_content, "page_idx": 0}]
        await app.rag.insert_content_list(
            content_list=content_list,
            file_path="sample_math.txt",
            doc_id="sample-math",
            display_stats=True,
        )

    # 示例查询
    queries = [
        "什么是函数？",
        "解释一下三角函数",
        "导数的几何意义是什么？",
    ]

    print("\n" + "=" * 60)
    print("运行示例查询")
    print("=" * 60)

    for q in queries:
        await app.query(q)

    return True


async def jewelry_mvp_demo():
    """珠宝工艺MVP演示."""
    print("\n" + "=" * 60)
    print("RAG-Anything 本地MVP - 珠宝工艺")
    print("=" * 60)

    app = LocalRAGMVP(data_dir=JEWELRY_DATA_DIR)

    # 检查服务
    if not await app.check_services():
        return False

    # 初始化
    await app.initialize()

    # 处理Markdown文件（更快速）
    await app.process_markdown(max_files=3)

    # 示例查询
    queries = [
        "珐琅工艺的制作步骤是什么？",
        "珠宝手绘需要哪些工具？",
    ]

    print("\n" + "=" * 60)
    print("运行示例查询")
    print("=" * 60)

    for q in queries:
        await app.query(q)

    return True


async def main():
    """主函数."""
    import sys

    # 选择演示类型
    demo_type = sys.argv[1] if len(sys.argv) > 1 else "math"

    if demo_type == "jewelry":
        success = await jewelry_mvp_demo()
    elif demo_type == "math":
        success = await math_mvp_demo()
    elif demo_type == "interactive":
        # 交互式模式
        app = LocalRAGMVP()
        if await app.check_services():
            await app.initialize()
            await app.interactive_query()
        success = True
    else:
        print(f"未知演示类型: {demo_type}")
        print("可用选项: math, jewelry, interactive")
        success = False

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
