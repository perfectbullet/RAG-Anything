"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from lightrag.utils import logger


def generate_doc_id_from_path(file_path: str | Path, prefix: str = "doc_", length: int = 12) -> str:
    """
    从文件路径生成唯一的 doc_id

    相同的文件路径始终生成相同的 doc_id

    Args:
        file_path: 文件路径
        prefix: doc_id 前缀，默认 "doc_"
        length: hash 后缀长度，默认 12 位（约 281 万亿种组合，碰撞概率极低）

    Returns:
        str: 格式为 "{prefix}{hash}" 的 doc_id，例如 "doc_a1b2c3d4e5f6"

    Example:
        >>> generate_doc_id_from_path("data/test.json")
        'doc_a1b2c3d4e5f6'
        >>> generate_doc_id_from_path("/path/to/file.pdf", prefix="file_")
        'file_1a2b3c4d5e6f'
    """
    path = Path(file_path)
    # 使用绝对路径生成稳定的 hash
    hash_input = str(path.resolve())

    # MD5 hash 取前 N 位
    hash_hex = hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{hash_hex}"


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists and is not a symlink (for security)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        if path.is_symlink():
            logger.warning(f"Blocking symlink for security: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])


def get_image_dimensions(image_path: str) -> Tuple[int, int] | None:
    """
    Get the width and height of an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) in pixels, or None if failed to read
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
            logger.debug(f"Image dimensions: {image_path} -> {width}x{height}")
            return width, height
    except Exception as e:
        logger.warning(f"Failed to get image dimensions for {image_path}: {e}")
        return None


# =============================================================================
# Text Extraction Utilities
# =============================================================================

def extract_text_from_item(item: Dict[str, Any], content_format: str = "standard") -> str:
    """
    Unified text extraction supporting both standard and v2 formats

    Args:
        item: Content item with type and content fields
        content_format: "standard" for flat format, "v2" for nested format

    Returns:
        str: Extracted text content
    """
    item_type = item.get('type', '')
    content = item.get('content', {})

    if not content:
        return ''

    try:
        if content_format == "v2":
            # V2 format has nested content structure
            if item_type == 'title':
                title_content = content.get('title_content', [])
                if title_content and len(title_content) > 0:
                    return title_content[0].get('content', '')
            elif item_type == 'paragraph':
                para_content = content.get('paragraph_content', [])
                if para_content and len(para_content) > 0:
                    # Join multiple text items
                    texts = []
                    for text_item in para_content:
                        if isinstance(text_item, dict):
                            texts.append(text_item.get('content', ''))
                        elif isinstance(text_item, str):
                            texts.append(text_item)
                    return "".join(texts)
            elif item_type == 'list':
                list_items = content.get('list_items', [])
                if list_items:
                    first_item = list_items[0]
                    item_content = first_item.get('item_content', [])
                    if item_content and len(item_content) > 0:
                        return item_content[0].get('content', '')
            elif item_type == 'equation_interline':
                return content.get('math_content', '')
            elif item_type == 'image':
                captions = content.get('image_caption', [])
                if captions:
                    return captions[0] if isinstance(captions, str) else captions[0] if captions else ''
            elif item_type == 'table':
                captions = content.get('table_caption', [])
                if captions:
                    return captions[0] if isinstance(captions, str) else captions[0] if captions else ''
                return content.get('table_body', '')
            elif item_type == 'chart':
                captions = content.get('chart_caption', [])
                if captions:
                    return captions[0] if isinstance(captions, str) else captions[0] if captions else ''
            elif item_type == 'code':
                return content.get('code_content', '')
            elif item_type == 'algorithm':
                return content.get('algorithm_content', '')
            elif item_type == 'index':
                list_items = content.get('list_items', [])
                if list_items:
                    first_item = list_items[0]
                    item_content = first_item.get('item_content', [])
                    if item_content and len(item_content) > 0:
                        return item_content[0].get('content', '')
        else:
            # Standard format (flat structure)
            if item_type == 'title':
                return item.get('title', '')
            elif item_type == 'paragraph':
                return item.get('text', '')
            elif item_type == 'list':
                return item.get('text', '')
            elif item_type == 'equation':
                return item.get('latex', '')
            elif item_type == 'image':
                caption = item.get('image_caption', [])
                return caption[0] if caption else ''
            elif item_type == 'table':
                caption = item.get('table_caption', [])
                return caption[0] if caption else item.get('table_body', '')

    except (KeyError, IndexError, TypeError):
        pass

    return ''


# =============================================================================
# Content Normalization Utilities
# =============================================================================

def normalize_content_list_v2(content_list_v2: List[List[Dict]], base_dir: str) -> List[List[Dict]]:
    """
    Convert v2 format to standard RAGAnything format

    Args:
        content_list_v2: V2 format content (2D array of pages)
        base_dir: Base directory for resolving relative paths

    Returns:
        Normalized content list v2
    """
    base_path = Path(base_dir)
    fixed_count = 0

    # Pre-compile regex patterns for better performance
    clean_pattern = re.compile(r'[^\w\u4e00-\u9fff-]')

    for page_items in content_list_v2:
        for item in page_items:
            content = item.get("content", {})
            if not content:
                continue

            item_type = item.get("type")

            # Image type normalization
            if item_type == "image":
                # Handle image_source.path -> img_path
                if "image_source" in content and "path" in content["image_source"]:
                    img_path = content["image_source"]["path"]
                    if not os.path.isabs(img_path):
                        img_path = str(base_path / img_path)
                    item["img_path"] = img_path
                    fixed_count += 1
                # Handle direct img_path in content
                elif "img_path" in content:
                    img_path = content["img_path"]
                    if not os.path.isabs(img_path):
                        img_path = str(base_path / img_path)
                    item["img_path"] = img_path
                    fixed_count += 1

                # Move caption and footnote to top level
                if "image_caption" in content:
                    item["image_caption"] = content["image_caption"]
                if "image_footnote" in content:
                    item["image_footnote"] = content["image_footnote"]

            # Table type normalization
            elif item_type == "table":
                if "table_path" in content:
                    table_path = content["table_path"]
                    if not os.path.isabs(table_path):
                        table_path = str(base_path / table_path)
                    item["img_path"] = table_path
                    fixed_count += 1

                # Move other fields to top level
                for key in ["table_caption", "table_footnote", "table_body"]:
                    if key in content:
                        item[key] = content[key]

            # Equation type normalization
            elif item_type == "equation_interline":
                if "math_content" in content:
                    item["latex"] = content["math_content"]
                    item["text"] = content.get("math_type", "latex")
                elif "text" in content:
                    item["latex"] = content["text"]
                    item["text"] = content.get("text_format", "latex")

            # Paragraph type normalization
            elif item_type == "paragraph":
                if "paragraph_content" in content:
                    texts = []
                    for text_item in content["paragraph_content"]:
                        if isinstance(text_item, dict):
                            texts.append(text_item.get("content", ""))
                        elif isinstance(text_item, str):
                            texts.append(text_item)
                    item["text"] = "".join(texts)

            # Title type normalization
            elif item_type == "title":
                if "title_content" in content:
                    texts = []
                    for text_item in content["title_content"]:
                        if isinstance(text_item, dict):
                            texts.append(text_item.get("content", ""))
                        elif isinstance(text_item, str):
                            texts.append(text_item)
                    item["text"] = "".join(texts)
                    # Keep level info
                    if "level" in content:
                        item["text_level"] = content["level"]

            # Clean titles for doc_id generation
            if item_type == "title" and "text" in item:
                title = item["text"]
                clean_title = clean_pattern.sub('_', title[:30]).strip('_')
                item["clean_title"] = clean_title

    logger.info(f"✅ Normalized {fixed_count} v2 format items to RAGAnything format")
    return content_list_v2


def load_content_list_v2(data_dir: str) -> Tuple[List[List[Dict]], str]:
    """
    Load content_list_v2.json from specified directory

    Args:
        data_dir: Directory path containing *_content_list_v2.json

    Returns:
        Tuple of (content_list_v2, json_file_path)
    """
    vlm_base_dir = Path(data_dir)

    # Find v2 format files
    json_files = list(vlm_base_dir.glob("*_content_list_v2.json"))

    if not json_files:
        raise FileNotFoundError(f"No *_content_list_v2.json files found in {vlm_base_dir}")

    if len(json_files) > 1:
        logger.warning(f"Multiple v2 files found: {[f.name for f in json_files]}")
        logger.info(f"Using first file: {json_files[0].name}")

    json_path = json_files[0]

    if not json_path.exists():
        raise FileNotFoundError(f"Test data file not found: {json_path}")

    logger.info(f"📂 Loading v2 test data: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        content_list_v2 = json.load(f)

    # Normalize the content
    normalized_content = normalize_content_list_v2(content_list_v2, vlm_base_dir)

    # Get type statistics
    type_counts = {}
    for page_items in normalized_content:
        for item in page_items:
            t = item.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

    logger.info(f"✅ Loaded {len(normalized_content)} pages of test data")
    logger.info(f"📊 Content type statistics: {type_counts}")

    return normalized_content, json_path


# =============================================================================
# Progress Tracking Utilities
# =============================================================================

@dataclass
class ProgressStatus:
    """Progress status enumeration"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 2
    base_wait_time: int = 5


class BaseProgressTracker:
    """Base class for progress tracking with common functionality"""

    def __init__(self, progress_file: str = "./insert_progress.json"):
        self.progress_file = progress_file
        self._progress_data = self._load_progress()
        self._last_saved_state = None

    def _load_progress(self) -> dict:
        """Load progress from file"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"📂 Loaded progress file: {self.progress_file}")

                # Log previous state
                started_count = len(progress.get('started', []))
                completed_count = len(progress.get('completed', []))
                failed_count = len(progress.get('failed', []))
                logger.info(f"   Previous state: {started_count} started, {completed_count} completed, {failed_count} failed")

                return progress
            except Exception as e:
                logger.warning(f"⚠️ Failed to load progress file: {e}, creating new")
                return {"started": [], "completed": [], "failed": [], "document_info": {}}
        return {"started": [], "completed": [], "failed": [], "document_info": {}}

    def _save_progress(self):
        """Save progress to file only if state changed"""
        try:
            current_state = json.dumps(self._progress_data, sort_keys=True)
            if self._last_saved_state != current_state:
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(self._progress_data, f, ensure_ascii=False, indent=2)
                self._last_saved_state = current_state
        except Exception as e:
            logger.warning(f"⚠️ Failed to save progress file: {e}")

    @property
    def started(self) -> List[str]:
        """Get started doc_ids (derived property)"""
        all_ids = set(self._progress_data.get("all", []))
        completed = set(self._progress_data.get("completed", []))
        failed = set(f["doc_id"] for f in self._progress_data.get("failed", []))
        return list(all_ids - completed - failed)

    def get_summary(self) -> dict:
        """Get progress summary"""
        total = self._progress_data["document_info"].get("total_sections", 0)
        started = len(self.started)
        completed = len(self._progress_data.get("completed", []))
        failed = len(self._progress_data.get("failed", []))
        return {
            "total": total,
            "started": started,
            "completed": completed,
            "failed": failed,
            "pending": total - completed,
        }


class ContentProcessingProgressTracker(BaseProgressTracker):
    """Progress tracker specifically for content processing"""

    def __init__(self, progress_file: Optional[str] = None):
        super().__init__(progress_file or "./insert_progress.json")

    def set_document_info(self, file_path: str, total_sections: int):
        """Set document information"""
        self._progress_data["document_info"] = {
            "file_path": file_path,
            "total_sections": total_sections,
            "last_update": str(Path(file_path).stat().st_mtime) if Path(file_path).exists() else None,
        }
        self._save_progress()

    def mark_started(self, doc_id: str, title: Optional[str] = None):
        """Mark a section as started"""
        if "all" not in self._progress_data:
            self._progress_data["all"] = []

        if doc_id not in self._progress_data["all"]:
            self._progress_data["all"].append(doc_id)

        if doc_id not in self._progress_data.get("started", []):
            self._progress_data.setdefault("started", []).append(doc_id)
            self._save_progress()
            logger.info(f"  💾 Marked as started: {title or doc_id}")

    def mark_completed(self, doc_id: str, title: Optional[str] = None):
        """Mark a section as completed"""
        if doc_id not in self._progress_data.get("completed", []):
            self._progress_data.setdefault("completed", []).append(doc_id)
            # Remove from started
            self._progress_data["started"] = [d for d in self._progress_data.get("started", []) if d != doc_id]
            # Remove from failed
            self._progress_data["failed"] = [f for f in self._progress_data.get("failed", []) if f["doc_id"] != doc_id]
            self._save_progress()
            logger.info(f"  💾 Marked as completed: {title or doc_id}")

    def mark_failed(self, doc_id: str, title: str, error: str, retry_count: int = None, max_retries: int = None, retry_config: RetryConfig = None):
        """Mark a section as failed

        Supports two calling conventions:
        - mark_failed(doc_id, title, error, retry_count, max_retries)
        - mark_failed(doc_id, title, error, retry_config=RetryConfig())
        """
        # Handle both calling conventions
        if retry_count is not None and max_retries is not None:
            # Old calling convention: retry_count, max_retries as positional args
            actual_retry_count = retry_count
            actual_max_retries = max_retries
        elif retry_config is not None:
            # New calling convention: retry_config as keyword arg
            actual_retry_count = retry_config.max_retries
            actual_max_retries = retry_config.max_retries
        else:
            # Default values
            actual_retry_count = 0
            actual_max_retries = 2

        failed_record = {
            "doc_id": doc_id,
            "title": title,
            "error": error,
            "retry_count": actual_retry_count,
            "max_retries": actual_max_retries,
            "last_failed": str(Path(self.progress_file).stat().st_mtime) if Path(self.progress_file).exists() else None,
        }

        # Remove from started
        self._progress_data["started"] = [d for d in self._progress_data.get("started", []) if d != doc_id]

        # Remove old failed record and add new one
        self._progress_data["failed"] = [f for f in self._progress_data.get("failed", []) if f["doc_id"] != doc_id]
        self._progress_data.setdefault("failed", []).append(failed_record)
        self._save_progress()
        logger.info(f"  💾 Marked as failed: {title}")

    def is_started(self, doc_id: str) -> bool:
        """Check if a section has been started"""
        return doc_id in self._progress_data.get("started", [])

    def is_completed(self, doc_id: str) -> bool:
        """Check if a section has been completed"""
        return doc_id in self._progress_data.get("completed", [])

    def get_failed_sections(self) -> list:
        """Get list of failed sections"""
        return self._progress_data.get("failed", [])


# =============================================================================
# Environment Validation Utilities
# =============================================================================

def validate_required_env_vars(required_vars: Dict[str, str]) -> None:
    """
    Validate required environment variables

    Args:
        required_vars: Dictionary of variable_name -> description

    Raises:
        ValueError: If any required variable is missing
    """
    missing_vars = []
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if value is None or value.strip() == "":
            missing_vars.append(f"  - {var_name}: {description}")

    if missing_vars:
        error_msg = "❌ Missing required environment variables:\n\n" + "\n".join(missing_vars)
        error_msg += f"\n\nPlease configure these variables in your .env file before running."
        raise ValueError(error_msg)

    logger.info("✅ Environment variables validation passed")


def get_required_env(var_name: str) -> str:
    """
    Get required environment variable or raise error

    Args:
        var_name: Environment variable name

    Returns:
        str: Environment variable value

    Raises:
        ValueError: If variable is missing
    """
    value = os.getenv(var_name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value.strip()


# =============================================================================
# Constants
# =============================================================================

class ContentType(str, Enum):
    """Content type enumeration"""
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST = "list"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CHART = "chart"
    CODE = "code"
    ALGORITHM = "algorithm"
    INDEX = "index"


class ProgressMessage:
    """Progress tracking message constants"""
    COMPLETED = "✅ 完成"
    FAILED = "⚠️ 失败"
    SKIPPED = "⏭️ 跳过"
    STARTED = "🔄 开始"
    RETRYING = "🔄 重试中"


class ChapterPattern:
    """Chapter detection patterns"""
    CHINESE = r'^第[一二三四五六七八九十\d]+章'
    ENGLISH = r'^Chapter\s+\d+'
    SECTION_CHINESE = r'^\d+\.\d+\s+'
    SECTION_ENGLISH = r'^\d+\.\d+\.\d+\s+'
