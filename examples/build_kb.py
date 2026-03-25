#!/usr/bin/env python3
"""
RAG-Anything 知识库构建脚本

功能：
1. 创建新的知识库目录
2. 通过MinerU解析PDF文档（支持多模态：文本、图像、表格、公式）
3. 增量更新PDF到知识库

Usage:
    # 创建新知识库并添加PDF
    python build_kb.py --kb-dir ./my_kb --pdf /path/to/file.pdf

    # 增量添加更多PDF
    python build_kb.py --kb-dir ./my_kb --pdf /path/to/another.pdf

    # 处理整个目录
    python build_kb.py --kb-dir ./my_kb --dir /path/to/pdfs
"""

import asyncio
import argparse
import os
import logging
import zipfile
import io
import tempfile
import contextlib
from pathlib import Path
from typing import List, Dict
import requests
import aiohttp
import numpy as np

# 抑制日志
class ImageDataFilter(logging.Filter):
    """过滤掉包含 image_data 的日志"""
    def filter(self, record):
        # 过滤掉包含 image_data 的日志以及 base64 开头的内容
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            if 'image_data' in msg or msg.startswith("/9j") or "'image_data':" in msg:
                return False
        return True

# 配置日志过滤器（带时间戳）
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

MINERU_API_URL = "http://192.168.8.234:8000"

# OpenAI兼容API配置（可选，优先级高于Ollama）
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 推荐使用Qwen模型，更严格遵守输出格式要求
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

# 视觉模型配置（用于图像处理）
VISION_MODEL = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-32B-Instruct")

# =============================================================================
# 导入
# =============================================================================
from raganything import RAGAnything, RAGAnythingConfig
from raganything.custom_parsers import OllamaLLMFunc, VLLMEmbeddingFunc, VLLMRerankerFunc
from raganything.parser import register_parser, Parser

# =============================================================================
# MinerU HTTP 解析器
# =============================================================================
class MinerUHTTPParser(Parser):
    """MinerU HTTP API解析器"""

    def __init__(self, base_url: str = MINERU_API_URL, images_output_dir: str = None):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout = 600  # 10分钟超时
        self.images_output_dir = images_output_dir  # 图片永久保存目录

    def parse_pdf(self, pdf_path: str, **kwargs) -> List[Dict]:
        """
        解析PDF文件

        Args:
            pdf_path: PDF文件路径
            start_page: 起始页码（默认0）
            end_page: 结束页码（默认99999，表示全部）

        Returns:
            content_list: 解析后的内容列表
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        url = f"{self.base_url}/file_parse"

        # 构建表单数据
        data = {
            "lang_list": "ch",
            "backend": "pipeline",
            "parse_method": "auto",
            "formula_enable": "true",
            "table_enable": "true",
            "response_format_zip": "true",
            "return_images": "true",
            "return_content_list": "true",
            "return_md": "false",
            "start_page_id": str(kwargs.get("start_page", 0)),
            "end_page_id": str(kwargs.get("end_page", 99999)),
        }

        try:
            with open(pdf_path, "rb") as f:
                files = {"files": (pdf_path.name, f, "application/pdf")}
                print(f"  正在解析: {pdf_path.name}...")
                response = requests.post(
                    url, files=files, data=data, timeout=self.timeout
                )
                response.raise_for_status()

            # 检查是否返回ZIP文件
            if response.content[:4] == b'PK\x03\x04':
                return self._extract_zip(response.content, pdf_path)
            else:
                # 返回JSON格式
                result = response.json()
                content_list_str = result.get("results", {}).get(pdf_path.name, {}).get("content_list", "[]")
                import json
                return json.loads(content_list_str)

        except requests.exceptions.Timeout:
            raise RuntimeError(f"MinerU请求超时（{self.timeout}s）")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"无法连接到MinerU: {self.base_url}")
        except Exception as e:
            raise RuntimeError(f"MinerU解析失败: {e}")

    def _extract_zip(self, zip_data: bytes, pdf_path: Path) -> List[Dict]:
        """
        从ZIP文件中提取content_list

        ZIP结构:
        output/
        └── document/
            ├── content_list.json    # 核心结构化数据
            ├── auto_md.md
            ├── middle_json.json
            └── images/              # 提取的图片
                ├── xxx.jpg
                └── ...
        """
        content_list = []
        images_dir = None

        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            file_list = zip_ref.namelist()

            # 查找content_list.json（在document/目录下）
            content_json_name = None
            for name in file_list:
                if name.endswith("content_list.json"):
                    content_json_name = name
                    break

            if not content_json_name:
                raise ValueError("ZIP文件中未找到content_list.json")

            # 读取content_list.json
            with zip_ref.open(content_json_name) as f:
                import json
                content_list = json.load(f)

            # 统计内容类型
            type_counts = {}
            for item in content_list:
                t = item.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            # 只有包含图像/表格/公式时才提取图像
            has_multimodal = any(t in type_counts for t in ["image", "table", "equation"])

            if has_multimodal:
                # 创建目录存储图像（优先使用永久目录，否则使用临时目录）
                if self.images_output_dir:
                    images_dir = Path(self.images_output_dir)
                    images_dir.mkdir(parents=True, exist_ok=True)
                else:
                    images_dir = Path(tempfile.mkdtemp(prefix="mineru_images_"))

                # 创建文件名映射：原始文件名 -> 本地路径
                image_path_map = {}

                # 提取图像文件并建立映射
                for name in file_list:
                    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        img_filename = Path(name).name
                        local_img_path = Path(images_dir) / img_filename

                        with zip_ref.open(name) as f, open(local_img_path, 'wb') as img_file:
                            img_file.write(f.read())

                        # 记录文件名映射
                        image_path_map[img_filename] = str(local_img_path)

                # 更新content_list中的图像路径
                for item in content_list:
                    item_type = item.get("type", "")
                    original_img_path = item.get("img_path", "")

                    if not original_img_path:
                        continue

                    # 获取原始文件名
                    original_filename = Path(original_img_path).name

                    # 查找对应的本地路径
                    if original_filename in image_path_map:
                        item["img_path"] = image_path_map[original_filename]

            # 保存临时目录信息用于清理（仅在使用临时目录时）
            if images_dir and not self.images_output_dir:
                content_list.append({
                    "type": "_cleanup",
                    "_temp_dir": str(images_dir),
                    "_description": "临时图像目录，处理完后删除"
                })

        return content_list

# =============================================================================
# LLM/Embedding/Reranker 函数
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
    视觉模型函数，用于图像分析和VLM增强查询

    支持两种调用模式：
    1. messages格式：多模态VLM增强查询（包含文本和图像的混合消息）
    2. image_data格式：图像处理（base64编码的图像数据）
    """
    import sys
    import io
    from lightrag.llm.openai import openai_complete_if_cache

    # 抑制 stdout（避免打印 image_data）
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        # 模式1：多模态消息格式（用于VLM增强查询）
        if messages:
            result = await openai_complete_if_cache(
                VISION_MODEL,
                "",  # 空prompt，内容在messages中
                system_prompt=system_prompt,
                messages=messages,
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                **kwargs
            )
        # 模式2：图像数据处理（用于知识库构建时的图像分析）
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
        # 模式3：纯文本处理（fallback）
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
# 知识库构建类
# =============================================================================
class KnowledgeBaseBuilder:
    """知识库构建器"""

    def __init__(self, kb_dir: str, use_openai: bool = False):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.rag = None
        # 创建图片保存目录
        self.images_dir = self.kb_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.parser = MinerUHTTPParser(images_output_dir=str(self.images_dir))
        self.use_openai = use_openai

    async def initialize(self):
        """初始化RAG系统"""
        print(f"\n🔧 初始化知识库: {self.kb_dir}")

        # 选择LLM函数
        if self.use_openai:
            if not OPENAI_API_KEY:
                raise ValueError("使用OpenAI API需要设置 OPENAI_API_KEY 环境变量")
            llm_func = openai_llm_func
            vision_func = vision_model_func
            print(f"  使用OpenAI模型: {OPENAI_MODEL}")
            print(f"  使用视觉模型: {VISION_MODEL}")
        else:
            llm_func = ollama_llm_func
            vision_func = None
            print(f"  使用本地模型: {OLLAMA_MODEL}")
            print(f"  ⚠️  警告: 本地模型不支持VLM增强，图像处理将被跳过")

        config = RAGAnythingConfig(
            working_dir=str(self.kb_dir),
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=get_embedding_func(),
            lightrag_kwargs={
                "rerank_model_func": vllm_reranker_func,
                "default_llm_timeout": 600,
                "chunk_token_size": 512,
                "llm_model_max_async": 4,  # 降低并发避免API速率限制 (付费版推荐4-6)
                "max_parallel_insert": 4,   # 多模态处理并发数量 (付费版推荐4-6)
                "chunk_overlap_token_size": 128,  # 增加重叠以改善上下文
            },
        )

        await self.rag._ensure_lightrag_initialized()
        await self.rag.lightrag.initialize_storages()

        # 检查已有数据
        chunks_file = self.kb_dir / "vdb_chunks.json"
        if chunks_file.exists() and chunks_file.stat().st_size > 100:
            import json
            with open(chunks_file) as f:
                chunks = json.load(f)
                count = len(chunks) if isinstance(chunks, list) else len(chunks.get("data", {}))
                print(f"  📚 已有 {count} 个文档块")

        print("  ✅ 初始化完成")

    async def add_pdf(self, pdf_path: str, start_page: int = 0, end_page: int = None):
        """
        添加PDF到知识库

        Args:
            pdf_path: PDF文件路径
            start_page: 起始页码
            end_page: 结束页码（None表示全部）
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"  ❌ 文件不存在: {pdf_path}")
            return False

        print(f"\n📄 处理PDF: {pdf_path.name}")
        print(f"   大小: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")

        if end_page is None:
            end_page = 99999

        try:
            # 使用MinerU解析PDF
            content_list = self.parser.parse_pdf(
                str(pdf_path),
                start_page=start_page,
                end_page=end_page,
            )

            # 保存临时目录信息，并在处理完后清理
            temp_dir_to_cleanup = None
            filtered_content_list = []

            for item in content_list:
                if item.get("type") == "_cleanup":
                    temp_dir_to_cleanup = item.get("_temp_dir")
                elif item.get("type") not in ["discarded"]:
                    # 过滤掉discarded内容（页眉页脚等）
                    filtered_content_list.append(item)

            # 统计内容类型
            type_counts = {}
            for item in filtered_content_list:
                t = item.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            print(f"  解析结果:")
            for t, count in sorted(type_counts.items()):
                print(f"    - {t}: {count}")

            # 插入到知识库
            print(f"  正在插入知识库... (可能需要较长时间)")

            await self.rag.insert_content_list(
                content_list=filtered_content_list,
                file_path=str(pdf_path),
                doc_id=pdf_path.stem,
                display_stats=False,
            )

            print(f"  ✅ 完成: {pdf_path.name}")

            # 清理临时图像目录（仅在使用临时目录时有此标记）
            if temp_dir_to_cleanup:
                import shutil
                try:
                    shutil.rmtree(temp_dir_to_cleanup, ignore_errors=True)
                    print(f"  🧹 已清理临时文件")
                except:
                    pass
            else:
                # 图片已保存到永久目录
                print(f"  📁 图片已保存到: {self.images_dir}")

            return True

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def add_directory(self, dir_path: str, pattern: str = "*.pdf"):
        """批量添加目录中的PDF文件"""
        dir_path = Path(dir_path)
        pdf_files = sorted(dir_path.glob(pattern))

        if not pdf_files:
            print(f"  ⚠️  未找到PDF文件: {dir_path}")
            return

        print(f"\n📚 找到 {len(pdf_files)} 个PDF文件")

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]", end=" ")
            await self.add_pdf(str(pdf_file))


# =============================================================================
# 主函数
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(description="RAG-Anything 知识库构建")
    parser.add_argument("--kb-dir", default="./rag_kb", help="知识库目录")
    parser.add_argument("--pdf", help="单个PDF文件路径")
    parser.add_argument("--dir", help="PDF文件目录")
    parser.add_argument("--start-page", type=int, default=0, help="起始页码")
    parser.add_argument("--end-page", type=int, default=None, help="结束页码")
    parser.add_argument("--use-openai", action="store_true",
                       help="使用OpenAI兼容API（优先于环境变量）")
    parser.add_argument("--use-ollama", "--no-openai", action="store_true", dest="use_ollama",
                       help="使用本地Ollama模型（禁用OpenAI）")

    args = parser.parse_args()

    # 确定是否使用OpenAI（优先级：命令行标志 > 环境变量）
    if args.use_ollama:
        use_openai = False  # 显式使用Ollama
    elif args.use_openai:
        use_openai = True   # 显式使用OpenAI
    else:
        use_openai = USE_OPENAI  # 使用环境变量默认值

    print("=" * 60)
    print("RAG-Anything 知识库构建")
    print("=" * 60)

    # 检查服务
    print("\n🔍 检查服务...")
    if use_openai:
        print(f"  OpenAI API: ✅ (配置: {OPENAI_API_BASE})")
    else:
        try:
            import requests
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            print(f"  Ollama: ✅")
        except:
            print(f"  Ollama: ❌")
            return False

    try:
        import requests
        requests.get(f"{MINERU_API_URL}/openapi.json", timeout=5)
        print(f"  MinerU: ✅")
    except:
        print(f"  MinerU: ❌")
        return False

    # 创建知识库
    builder = KnowledgeBaseBuilder(args.kb_dir, use_openai=use_openai)
    await builder.initialize()

    # 处理PDF
    if args.pdf:
        await builder.add_pdf(args.pdf, args.start_page, args.end_page)
    elif args.dir:
        await builder.add_directory(args.dir)
    else:
        print("\n⚠️  请指定 --pdf 或 --dir 参数")
        print("\n示例:")
        print("  python build_kb.py --kb-dir ./math_kb --pdf ./test.pdf")
        print("  python build_kb.py --kb-dir ./math_kb --dir ./pdfs")
        return False

    print("\n" + "=" * 60)
    print("✅ 知识库构建完成!")
    print(f"📁 知识库位置: {args.kb_dir}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
