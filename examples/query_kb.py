#!/usr/bin/env python3
"""
RAG-Anything 知识库查询脚本

Usage:
    # 交互式查询
    python query_kb.py --kb-dir ./rag_kb

    # 单次查询
    python query_kb.py --kb-dir ./rag_kb --query "什么是函数？"

    # 指定查询模式
    python query_kb.py --kb-dir ./rag_kb --query "导数的几何意义" --mode hybrid
"""

import asyncio
import argparse
import logging
import os
import contextlib
import io
from pathlib import Path
from typing import List, Dict
import aiohttp
import numpy as np

# 抑制日志
class ImageDataFilter(logging.Filter):
    """过滤掉包含 image_data 的日志"""
    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            if 'image_data' in msg or msg.startswith("/9j") or "'image_data':" in msg:
                return False
        return True

# 配置日志（带时间戳）
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
for m in ['lightrag', 'nano_vectordb', 'raganything']:
    logger = logging.getLogger(m)
    logger.setLevel(logging.ERROR)
    logger.addFilter(ImageDataFilter())

# =============================================================================
# 配置
# =============================================================================
OLLAMA_BASE_URL = "http://192.168.8.230:11434"
OLLAMA_MODEL = "qwen2.5:14b"

VLLM_EMBED_URL = "http://192.168.8.233:8092/v1"
VLLM_EMBED_MODEL = "BAAI/bge-m3"
VLLM_EMBED_DIM = 1024

VLLM_RERANK_URL = "http://192.168.8.233:8091/v1"
VLLM_RERANK_MODEL = "bge-reranker-m3"

# OpenAI兼容API配置（可选，优先级高于Ollama）
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 推荐使用Qwen模型，更严格遵守输出格式要求
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

# 视觉模型配置（用于VLM增强查询）
VISION_MODEL = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-32B-Instruct")

# =============================================================================
# 导入
# =============================================================================
from raganything import RAGAnything, RAGAnythingConfig
from raganything.custom_parsers import OllamaLLMFunc, VLLMEmbeddingFunc, VLLMRerankerFunc

# =============================================================================
# 函数
# =============================================================================
async def ollama_llm_func(prompt: str, system_prompt: str = None,
                          history_messages: List[Dict] = None, **kwargs) -> str:
    ollama = OllamaLLMFunc(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, timeout=600)
    return await ollama(prompt=prompt, system_prompt=system_prompt,
                       history_messages=history_messages, **kwargs)

async def openai_llm_func(prompt: str, system_prompt: str = None,
                          history_messages: List[Dict] = None, **kwargs) -> str:
    """OpenAI兼容API的LLM函数"""
    from lightrag.llm.openai import openai_complete_if_cache

    return await openai_complete_if_cache(
        OPENAI_MODEL,  # 作为第一个位置参数传递，避免参数冲突
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        **kwargs
    )

async def vision_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    image_data: str = None,
    messages: List[Dict] = None,
    **kwargs
) -> str:
    """
    视觉模型函数，用于VLM增强查询

    支持两种调用模式：
    1. messages格式：多模态VLM增强查询（包含文本和图像的混合消息）
    2. image_data格式：图像处理（base64编码的图像数据）
    """
    from lightrag.llm.openai import openai_complete_if_cache

    # 抑制 stdout（避免打印 image_data）
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        if messages:
            result = await openai_complete_if_cache(
                VISION_MODEL,
                "",
                system_prompt=system_prompt,
                messages=messages,
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                **kwargs
            )
        elif image_data:
            result = await openai_complete_if_cache(
                VISION_MODEL,
                prompt,
                system_prompt=system_prompt,
                image_data=image_data,
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                **kwargs
            )
        else:
            result = await openai_complete_if_cache(
                VISION_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                **kwargs
            )

    return result

async def vllm_embedding_func(texts: List[str]) -> List[List[float]]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VLLM_EMBED_URL}/embeddings",
            json={"input": texts, "model": VLLM_EMBED_MODEL},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            return np.array([item["embedding"] for item in result["data"]], dtype=np.float32)

def get_embedding_func():
    from lightrag.utils import EmbeddingFunc
    return EmbeddingFunc(embedding_dim=VLLM_EMBED_DIM, max_token_size=8192, func=vllm_embedding_func)

async def vllm_reranker_func(query: str, documents: List[str],
                             top_k: int = None, **kwargs) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VLLM_RERANK_URL}/rerank",
            json={"model": VLLM_RERANK_MODEL, "query": query, "documents": documents},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            result = await response.json()
            reranked = []
            for item in result.get("results", []):
                idx = item["index"]
                reranked.append({
                    "doc_id": idx, "index": idx,
                    "score": item["relevance_score"],
                    "text": item.get("document", {}).get("text", documents[idx]),
                })
            return reranked[:top_k] if top_k else reranked

# =============================================================================
# 查询类
# =============================================================================
class KnowledgeBaseQuery:
    """知识库查询器"""

    def __init__(self, kb_dir: str, use_openai: bool = False):
        self.kb_dir = Path(kb_dir)
        self.rag = None
        self.use_openai = use_openai

    async def initialize(self):
        """初始化并加载知识库"""
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"知识库目录不存在: {self.kb_dir}")

        print(f"📁 知识库: {self.kb_dir}")

        # 检查数据
        chunks_file = self.kb_dir / "vdb_chunks.json"
        if not chunks_file.exists() or chunks_file.stat().st_size < 100:
            raise ValueError(f"知识库为空，请先运行 build_kb.py 添加文档")

        # 选择LLM函数
        if self.use_openai:
            if not OPENAI_API_KEY:
                raise ValueError("使用OpenAI API需要设置 OPENAI_API_KEY 环境变量")
            llm_func = openai_llm_func
            vision_func = vision_model_func
            print(f"🤖 使用OpenAI模型: {OPENAI_MODEL}")
            print(f"👁️  使用视觉模型: {VISION_MODEL}")
            print(f"✅ VLM增强查询: 已启用")
        else:
            llm_func = ollama_llm_func
            vision_func = None
            print(f"🤖 使用本地模型: {OLLAMA_MODEL}")

        config = RAGAnythingConfig(working_dir=str(self.kb_dir))

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=get_embedding_func(),
            lightrag_kwargs={
                "rerank_model_func": vllm_reranker_func,
            },
        )

        result = await self.rag._ensure_lightrag_initialized()
        if not result.get("success"):
            error_msg = result.get("error", "Unknown initialization error")
            raise RuntimeError(f"Failed to initialize RAG: {error_msg}")

        # 显示统计
        import json
        with open(chunks_file) as f:
            chunks = json.load(f)
            count = len(chunks) if isinstance(chunks, list) else len(chunks.get("data", {}))
            print(f"📊 文档块数量: {count}")

        print("✅ 知识库已加载")

    async def query(self, question: str, mode: str = "hybrid", vlm_enhanced: bool = None,
                   show_context: bool = True) -> str:
        """
        查询知识库

        Args:
            question: 查询问题
            mode: 查询模式 (local, global, hybrid, naive)
            vlm_enhanced: 是否启用VLM增强 (None=自动检测, True=强制启用, False=禁用)
            show_context: 是否显示召回的上下文信息

        Returns:
            回答内容
        """
        import time
        import json

        print(f"\n❓ 问题: {question}")
        print(f"🔍 模式: {mode}")

        # 显示VLM增强状态
        if vlm_enhanced is None:
            vlm_status = "自动" if self.use_openai else "禁用"
        elif vlm_enhanced:
            vlm_status = "启用"
        else:
            vlm_status = "禁用"
        print(f"👁️  VLM增强: {vlm_status}")

        # 记录开始时间
        query_start_time = time.time()
        print(f"⏱️  开始查询: {time.strftime('%H:%M:%S')}")

        # 执行查询
        result = await self.rag.aquery(question, mode=mode, vlm_enhanced=vlm_enhanced)
        query_end_time = time.time()

        # 计算总时间
        total_time = (query_end_time - query_start_time) * 1000
        print(f"⏱️  总查询时间: {total_time:.0f}ms ({total_time/1000:.2f}s)")
        print(f"⏱️  完成时间: {time.strftime('%H:%M:%S')}")

        # 显示召回的上下文
        if show_context:
            await self._show_query_context(question, mode)

        return result

    async def _show_query_context(self, question: str, mode: str):
        """显示查询召回的上下文信息"""
        import json
        from pathlib import Path
        from lightrag import QueryParam

        print("\n" + "=" * 60)
        print("📚 召回上下文详情")
        print("=" * 60)

        # 获取召回的上下文
        try:
            query_param = QueryParam(mode=mode, only_need_prompt=True)
            context_prompt = await self.rag.lightrag.aquery(question, param=query_param)

            # 显示召回上下文
            print("\n📄 召回的上下文内容:")
            print("-" * 60)

            # 截断过长的内容以便显示
            max_chars = 3000
            if len(context_prompt) > max_chars:
                print(context_prompt[:max_chars])
                print(f"\n... (省约 {len(context_prompt) - max_chars} 字符)")
            else:
                print(context_prompt)

            print("-" * 60)
        except Exception as e:
            print(f"⚠️  无法获取召回上下文: {e}")

        # 显示知识库统计
        chunks_file = self.kb_dir / "vdb_chunks.json"
        if chunks_file.exists():
            with open(chunks_file) as f:
                chunks = json.load(f)
                count = len(chunks) if isinstance(chunks, list) else len(chunks.get("data", {}))
                print(f"\n📊 知识库统计: {count} 个文档块")

        entities_file = self.kb_dir / "vdb_entities.json"
        if entities_file.exists():
            with open(entities_file) as f:
                entities_data = json.load(f)
                if isinstance(entities_data, dict) and "data" in entities_data:
                    entity_count = len(entities_data["data"])
                elif isinstance(entities_data, list):
                    entity_count = len(entities_data)
                else:
                    entity_count = len(entities_data)
                print(f"🧠 知识图谱: {entity_count} 个实体")

        print("=" * 60)

    async def interactive(self):
        """交互式查询模式"""
        print("\n" + "=" * 50)
        print("📖 交互式查询模式")
        print("=" * 50)
        print("可用模式: local, global, hybrid, naive")
        print("输入问题，或输入 'quit' 退出\n")

        mode = "hybrid"

        while True:
            try:
                user_input = input("❓ 你的问题 (或 'quit' 退出): ").strip()

                if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                    print("\n👋 再见！")
                    break

                if not user_input:
                    continue

                # 检查是否是模式切换命令
                if user_input.startswith("--mode "):
                    mode = user_input.split()[1]
                    print(f"✓ 模式已切换为: {mode}")
                    continue

                await self.query(user_input, mode=mode,
                               show_context=getattr(self, 'show_context_interactive', True))

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 查询失败: {e}")


# =============================================================================
# 主函数
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(description="RAG-Anything 知识库查询")
    parser.add_argument("--kb-dir", default="./rag_kb", help="知识库目录")
    parser.add_argument("--query", help="查询问题（不指定则进入交互模式）")
    parser.add_argument("--mode", default="hybrid",
                       choices=["local", "global", "hybrid", "naive"],
                       help="查询模式")
    parser.add_argument("--use-openai", action="store_true",
                       help="使用OpenAI兼容API（优先于环境变量）")
    parser.add_argument("--use-ollama", "--no-openai", action="store_true", dest="use_ollama",
                       help="使用本地Ollama模型（禁用OpenAI）")
    parser.add_argument("--vlm-enhanced", action="store_true",
                       help="强制启用VLM增强查询")
    parser.add_argument("--no-vlm", action="store_true",
                       help="禁用VLM增强查询")
    parser.add_argument("--show-context", action="store_true",
                       help="显示召回的文档和知识图谱详情")
    parser.add_argument("--hide-context", action="store_true",
                       help="隐藏召回的文档和知识图谱详情")

    args = parser.parse_args()

    # 确定VLM增强设置
    vlm_enhanced = None
    if args.vlm_enhanced:
        vlm_enhanced = True
    elif args.no_vlm:
        vlm_enhanced = False

    # 确定是否显示上下文（默认显示）
    if args.hide_context:
        show_context = False
    elif args.show_context:
        show_context = True
    else:
        show_context = True  # 默认显示

    # 确定是否使用OpenAI（优先级：命令行标志 > 环境变量）
    if args.use_ollama:
        use_openai = False  # 显式使用Ollama
    elif args.use_openai:
        use_openai = True   # 显式使用OpenAI
    else:
        use_openai = USE_OPENAI  # 使用环境变量默认值

    print("=" * 60)
    print("RAG-Anything 知识库查询")
    print("=" * 60)

    try:
        query_app = KnowledgeBaseQuery(args.kb_dir, use_openai=use_openai)
        await query_app.initialize()

        if args.query:
            # 单次查询
            result = await query_app.query(args.query, args.mode, vlm_enhanced=vlm_enhanced,
                                         show_context=show_context)
            print(f"\n💡 回答:\n{result}")
        else:
            # 交互模式（默认显示上下文）
            query_app.show_context_interactive = show_context
            await query_app.interactive()

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"\n请先运行: python build_kb.py --kb-dir {args.kb_dir} --pdf <your_file.pdf>")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
