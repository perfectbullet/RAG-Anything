"""
Custom Parsers for RAGAnything

This module contains custom parser implementations that extend RAGAnything's
parsing capabilities, including HTTP API based parsers.
"""

import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from raganything.parser import Parser, register_parser


class MinerUHttpParser(Parser):
    """MinerU HTTP API parser for remote document parsing.

    This parser connects to a remote MinerU HTTP API instead of using
    the CLI directly, useful for distributed processing scenarios.

    API Endpoint: http://host:port/file_parse
    """

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8000",
        lang: str = "ch",
        backend: str = "hybrid-auto-engine",
        formula_enable: bool = True,
        table_enable: bool = True,
        timeout: int = 300,
    ):
        """Initialize MinerU HTTP API parser.

        Args:
            base_url: Base URL of the MinerU HTTP API
            lang: Default language for OCR (ch, en, etc.)
            backend: Default parsing backend
            formula_enable: Enable formula parsing
            table_enable: Enable table parsing
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.default_lang = lang
        self.default_backend = backend
        self.default_formula_enable = formula_enable
        self.default_table_enable = table_enable
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def _call_parse_api(
        self,
        file_path: str,
        parse_method: str = "auto",
        lang: Optional[str] = None,
        backend: Optional[str] = None,
        formula_enable: Optional[bool] = None,
        table_enable: Optional[bool] = None,
        start_page: int = 0,
        end_page: int = 99999,
        return_content_list: bool = True,
    ) -> Dict[str, Any]:
        """Call MinerU HTTP API for parsing.

        Args:
            file_path: Path to the file to parse
            parse_method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            backend: Parsing backend
            formula_enable: Enable formula parsing
            table_enable: Enable table parsing
            start_page: Starting page number (0-based)
            end_page: Ending page number (0-based)
            return_content_list: Return content list in response

        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/file_parse"

        # Prepare form data
        data = {
            "parse_method": parse_method,
            "lang_list": [lang or self.default_lang],
            "backend": backend or self.default_backend,
            "formula_enable": str(
                formula_enable if formula_enable is not None else self.default_formula_enable
            ),
            "table_enable": str(
                table_enable if table_enable is not None else self.default_table_enable
            ),
            "start_page_id": start_page,
            "end_page_id": end_page,
            "return_content_list": str(return_content_list).lower(),
            "return_md": "false",
        }

        file_path = Path(file_path)
        try:
            with open(file_path, "rb") as f:
                files = {"files": (file_path.name, f, "application/pdf")}

                self.logger.info(f"Sending parse request to: {url}")
                response = requests.post(
                    url, files=files, data=data, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"MinerU API request timed out after {self.timeout}s. "
                f"The file may be too large or the server may be overloaded."
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Failed to connect to MinerU API at {url}. "
                f"Please ensure the server is running: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"MinerU API returned HTTP error: {e.response.status_code} - {e.response.text}"
            )

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Parse PDF document using MinerU HTTP API.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path (not used for HTTP API)
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        self.logger.info(f"Parsing PDF via HTTP API: {pdf_path.name}")

        result = self._call_parse_api(
            str(pdf_path),
            parse_method=method,
            lang=lang,
            **kwargs,
        )

        # Extract content_list from response
        if isinstance(result, dict) and "content_list" in result:
            content_list = result["content_list"]
        elif isinstance(result, list):
            content_list = result
        else:
            raise ValueError(f"Unexpected response format: {type(result)}")

        # Fix image paths to absolute paths
        self._normalize_image_paths(content_list, pdf_path)

        return content_list

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Parse image document using MinerU HTTP API.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path (not used for HTTP API)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        # Images always use OCR method
        result = self._call_parse_api(
            str(image_path),
            parse_method="ocr",
            lang=lang,
            **kwargs,
        )

        if isinstance(result, dict) and "content_list" in result:
            content_list = result["content_list"]
        elif isinstance(result, list):
            content_list = result
        else:
            raise ValueError(f"Unexpected response format: {type(result)}")

        self._normalize_image_paths(content_list, image_path)
        return content_list

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Parse document using MinerU HTTP API based on file extension.

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path (not used for HTTP API)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            # Convert to PDF first, then parse
            pdf_path = self.convert_office_to_pdf(file_path, output_dir)
            return self.parse_pdf(pdf_path, output_dir, method, lang, **kwargs)
        elif ext in self.TEXT_FORMATS:
            # Convert to PDF first, then parse
            pdf_path = self.convert_text_to_pdf(file_path, output_dir)
            return self.parse_pdf(pdf_path, output_dir, method, lang, **kwargs)
        else:
            # Try as PDF
            self.logger.warning(
                f"Unsupported file extension '{ext}', attempting to parse as PDF"
            )
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)

    def _normalize_image_paths(
        self, content_list: List[Dict[str, Any]], base_file: Path
    ) -> None:
        """Normalize image paths to absolute paths.

        For HTTP API, image paths may be relative or URLs.
        This method converts them to absolute file paths if they are local.

        Args:
            content_list: List of content dictionaries to modify in-place
            base_file: Base file path for resolving relative paths
        """
        for item in content_list:
            if isinstance(item, dict):
                for field_name in ["img_path", "table_img_path", "equation_img_path"]:
                    if field_name in item and item[field_name]:
                        img_path = item[field_name]

                        # If it's already an absolute path, keep it
                        if Path(img_path).is_absolute():
                            continue

                        # If it's a URL or base64, keep it as is
                        if img_path.startswith(("http://", "https://", "data:")):
                            continue

                        # Otherwise, make it absolute relative to the base file
                        absolute_path = (base_file.parent / img_path).resolve()
                        item[field_name] = str(absolute_path)

    def check_installation(self) -> bool:
        """Check if MinerU HTTP API is available.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/openapi.json", timeout=5
            )
            is_available = response.status_code == 200
            if is_available:
                self.logger.info(f"MinerU HTTP API is available at: {self.base_url}")
            else:
                self.logger.warning(
                    f"MinerU HTTP API returned status {response.status_code}"
                )
            return is_available
        except requests.exceptions.ConnectionError:
            self.logger.warning(
                f"Cannot connect to MinerU HTTP API at: {self.base_url}"
            )
            return False
        except Exception as e:
            self.logger.warning(f"MinerU HTTP API check failed: {e}")
            return False


class OllamaLLMFunc:
    """OLLAMA LLM function wrapper for async LLM calls.

    This class provides an async callable interface for OLLAMA's
    /api/chat endpoint, compatible with LightRAG's llm_model_func.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:11434",
        model: str = "qwen2.5:14b",
        timeout: int = 120,
    ):
        """Initialize OLLAMA LLM function.

        Args:
            base_url: Base URL of the OLLAMA API
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    async def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict] = None,
        **kwargs,
    ) -> str:
        """Call OLLAMA API for chat completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            **kwargs: Additional parameters

        Returns:
            str: Model response
        """
        import aiohttp

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        timeout = kwargs.get("timeout", self.timeout)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    result = await response.json()
                    return result["message"]["content"]
        except aiohttp.ClientError as e:
            self.logger.error(f"OLLAMA API error: {e}")
            raise RuntimeError(f"OLLAMA API request failed: {e}")

    def check_installation(self) -> bool:
        """Check if OLLAMA API is available.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            is_available = response.status_code == 200
            if is_available:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.model in model_names:
                    self.logger.info(
                        f"OLLAMA API available, model '{self.model}' found"
                    )
                else:
                    self.logger.warning(
                        f"OLLAMA API available but model '{self.model}' not found. "
                        f"Available models: {model_names}"
                    )
            return is_available
        except Exception as e:
            self.logger.warning(f"OLLAMA API check failed: {e}")
            return False


class VLLMEmbeddingFunc:
    """VLLM embedding function wrapper for LightRAG.

    This class provides an async embedding interface using VLLM's
    OpenAI-compatible embedding endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8092/v1",
        model: str = "BAAI/bge-m3",
        embedding_dim: int = 1024,
        max_token_size: int = 8192,
        api_key: str = "dummy",
    ):
        """Initialize VLLM embedding function.

        Args:
            base_url: Base URL of the VLLM API
            model: Model name
            embedding_dim: Embedding dimension
            max_token_size: Maximum token size
            api_key: API key (VLLM doesn't require real key)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List[List[float]]: List of embedding vectors
        """
        import aiohttp
        import numpy as np

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    json={"input": texts, "model": self.model},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    # VLLM返回格式: {"data": [{"embedding": [...], ...}, ...]}
                    embeddings = [item["embedding"] for item in result["data"]]
                    # 返回numpy数组以兼容LightRAG
                    return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"VLLM embedding error: {e}")
            raise

    def to_embedding_func(self):
        """Convert to LightRAG EmbeddingFunc format.

        Returns:
            EmbeddingFunc: LightRAG compatible embedding function
        """
        from lightrag.utils import EmbeddingFunc

        return EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=self.max_token_size,
            func=self,
        )

    def check_installation(self) -> bool:
        """Check if VLLM embedding API is available.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"input": "test", "model": self.model},
                timeout=10,
            )
            is_available = response.status_code == 200
            if is_available:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    actual_dim = len(data["data"][0]["embedding"])
                    self.logger.info(
                        f"VLLM embedding API available, model '{self.model}', "
                        f"dimension: {actual_dim}"
                    )
                    if actual_dim != self.embedding_dim:
                        self.logger.warning(
                            f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                            f"got {actual_dim}"
                        )
            return is_available
        except Exception as e:
            self.logger.warning(f"VLLM embedding API check failed: {e}")
            return False


class VLLMRerankerFunc:
    """VLLM reranker function wrapper for LightRAG.

    This class provides an async reranking interface using VLLM's
    reranker endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.8.233:8091/v1",
        model: str = "bge-reranker-m3",
        timeout: int = 30,
    ):
        """Initialize VLLM reranker function.

        Args:
            base_url: Base URL of the VLLM reranker API
            model: Model name
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    async def __call__(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
        **kwargs,
    ) -> List[Dict]:
        """Rerank documents based on query relevance.

        Args:
            query: Query string
            documents: List of document strings
            top_k: Number of top results to return
            **kwargs: Additional parameters

        Returns:
            List[Dict]: Reranked documents with scores
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/rerank",
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    result = await response.json()

                    # Convert VLLM format to LightRAG format
                    reranked = []
                    for item in result.get("results", []):
                        idx = item["index"]
                        # VLLM格式: item["document"]["text"] 包含原始文本
                        reranked.append({
                            "doc_id": idx,
                            "index": idx,  # 保留原始索引
                            "score": item["relevance_score"],
                            "text": item.get("document", {}).get("text", documents[idx]),
                        })

                    if top_k:
                        reranked = reranked[:top_k]

                    return reranked
        except aiohttp.ClientError as e:
            self.logger.error(f"VLLM reranker error: {e}")
            raise RuntimeError(f"VLLM reranker request failed: {e}")

    def check_installation(self) -> bool:
        """Check if VLLM reranker API is available.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.model,
                    "query": "test query",
                    "documents": ["test document"],
                },
                timeout=10,
            )
            is_available = response.status_code == 200
            if is_available:
                self.logger.info(
                    f"VLLM reranker API available, model '{self.model}'"
                )
            return is_available
        except Exception as e:
            self.logger.warning(f"VLLM reranker API check failed: {e}")
            return False


# Register the custom parser
register_parser("mineru-http", MinerUHttpParser)
