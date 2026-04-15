"""
Content List V2 Splitter

Provides utilities to split MinerU content_list_v2.json into batches for processing.
The v2 format uses a 2D array structure grouped by pages: [[page0_items], [page1_items], ...]

Example:
    >>> with open("doc_content_list_v2.json") as f:
    ...     content_list_v2 = json.load(f)
    >>> splitter = ContentListV2Splitter()
    >>> batches = splitter.split_by_pages(content_list_v2, pages_per_batch=30)
    >>> for batch in batches:
    ...     await rag.insert_content_list(
    ...         content_list=batch['content'],
    ...         doc_id=batch['doc_id'],
    ...     )
"""

import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from functools import lru_cache
from lightrag.utils import logger
from .utils import (
    extract_text_from_item,
    ContentType,
    ChapterPattern,
    get_image_dimensions,
)


class ContentListV2Splitter:
    """
    content_list_v2.json 切分器

    特点：
    - 利用 v2 格式的页面分组特性（二维数组）
    - 保持语义完整性（标题+内容为一组）
    - 支持多种切分策略
    """

    # 需要跳过的页面辅助类型
    PAGE_AUX_TYPES = {
        'page_header',
        'page_footer',
        'page_number',
        'page_aside_text',
        'page_footnote',
    }

    # 默认小节标题模式（如 1.1, 1.2 等）
    SECTION_PATTERNS = [
        r'^\d+\.\d+\s+',  # 1.1 集合的概念
        r'^\d+\.\d+\.\d+\s+',  # 1.1.1 更细的分层
    ]

    # Milvus varchar field max length
    MAX_DOC_ID_LENGTH = 64

    def _generate_safe_doc_id(
        self,
        prefix: str,
        chapter_num: int,
        title: str,
        part_num: int = None
    ) -> str:
        """
        生成安全的 doc_id，确保不超过 Milvus 的 64 字符限制

        Args:
            prefix: doc_id 前缀
            chapter_num: 章节编号
            title: 章节标题
            part_num: 可选的分段编号

        Returns:
            长度不超过 64 字符的 doc_id
        """
        # 计算基础组件长度
        base_part = f"{prefix}_"

        # 使用标题的 MD5 hash (8位) 替代长标题，确保长度可控且保持唯一性
        title_hash = hashlib.md5(title.encode('utf-8')).hexdigest()[:8]

        doc_id = f"{base_part}{title_hash}"
        return doc_id

    def __init__(
        self,
        chapter_patterns: List[str] = None,
        max_pages_per_chapter: int = 30,
    ):
        """
        Args:
            chapter_patterns: 章节标题正则模式列表
            max_pages_per_chapter: 大章节自动拆分的页数阈值
        """
        # Pre-compile regex patterns for better performance
        if chapter_patterns:
            self.chapter_patterns = [re.compile(pattern) for pattern in chapter_patterns]
        else:
            self.chapter_patterns = [
                re.compile(ChapterPattern.CHINESE),
                re.compile(ChapterPattern.ENGLISH),
            ]
        self.max_pages_per_chapter = max_pages_per_chapter

    def split_by_chapters(
        self,
        content_list_v2: List,
        doc_id_prefix: str = None,
        max_pages: int = None,
        use_sections_as_fallback: bool = True
    ) -> List[Dict[str, Any]]:
        """
        按章节切分，大章节自动按页数拆分

        Args:
            content_list_v2: v2 格式的二维数组
            doc_id_prefix: doc_id 前缀
            max_pages: 大章节自动拆分的页数阈值，默认使用初始化时的值
            use_sections_as_fallback: 如果没有找到章节标题，是否使用小节标题作为切分点
            output_json_path: 可选的 JSON 输出文件路径，用于保存切分结果

        Returns:
            章节列表，每项包含:
            - doc_id: 章节/批次ID
            - title: 章节标题
            - content: 展平后的内容项列表
            - page_range: 页码范围 [start, end]
            - item_count: 内容项数量
            - is_split: 是否为拆分后的大章节
        """
        if max_pages is None:
            max_pages = self.max_pages_per_chapter

        # 检测目录结束页
        toc_end_page = self._detect_toc_end_page_v2(content_list_v2)

        # 查找所有章节标记
        chapters = self._find_chapters_v2(content_list_v2, toc_end_page)

        # 如果没有找到章节，尝试使用小节标题
        if not chapters and use_sections_as_fallback:
            logger.info("📚 未检测到章节标题，尝试使用小节标题进行切分")
            chapters = self._find_sections_v2(content_list_v2, toc_end_page)

        # 处理章节内容并拆分大章节
        result = []
        chapter_num = 0

        for i, chapter in enumerate(chapters):
            start_page = chapter['start_page']
            end_page = chapter['end_page']
            title = chapter['title']
            page_count = end_page - start_page + 1

            # 大章节拆分
            if page_count > max_pages:
                logger.info(f"📚 章节 '{title}' 有 {page_count} 页，超过阈值 {max_pages}，进行拆分")
                sub_batches = self._split_large_chapter(
                    content_list_v2, start_page, end_page,
                    title, doc_id_prefix, i + 1, max_pages
                )
                result.extend(sub_batches)
            else:
                # 正常章节
                chapter_num += 1
                flattened_content = self._flatten_page_content(content_list_v2, start_page, end_page)

                doc_id = self._generate_safe_doc_id(doc_id_prefix, chapter_num, title) if doc_id_prefix else None

                result.append({
                    'doc_id': doc_id,
                    'title': title,
                    # 'content_list': flattened_content,
                    'page_range': [start_page, end_page],
                    'item_count': len(flattened_content),
                    'is_split': False,
                })

        logger.info(f"📚 按章节切分: {len(result)} 个批次（包含拆分后的大章节）")

        return result

    def _detect_content_start_page(self, content_list_v2: List) -> int:
        """
        检测正文开始的页码

        跳过封面、版权页等前置页面
        """
        # 简单策略：跳过前2页（通常是封面和版权页）
        # 可以根据实际需求调整
        return min(2, len(content_list_v2))

    def _detect_toc_end_page_v2(self, content_list_v2: List) -> int:
        """
        检测目录结束的页码

        Returns:
            目录结束后的页码（正文开始的页码）
        """
        toc_keywords = ['目录', '目　录', 'CONTENTS', 'Contents']

        for page_idx, page_items in enumerate(content_list_v2):
            for item in page_items:
                text = self._extract_text_from_item_cached(json.dumps(item, sort_keys=True))
                if text and any(kw in text for kw in toc_keywords):
                    # 找到目录页，返回下一页作为正文开始
                    logger.info(f"📑 检测到目录页: page {page_idx}")
                    return page_idx + 1

        # 默认跳过前3页
        default_start = min(3, len(content_list_v2))
        logger.info(f"📑 未检测到目录，使用默认起始页: {default_start}")
        return default_start

    def _find_chapters_v2(
        self,
        content_list_v2: List,
        start_page: int,
    ) -> List[Dict]:
        """
        查找所有章节标记

        策略：
        1. 收集目录中的章节标题
        2. 在正文中查找真正的章节标题（不在目录中的）

        Returns:
            章节列表，每项包含: title, start_page, end_page
        """
        # 首先收集目录中的章节标题
        toc_chapters = set()
        toc_end_page = self._detect_toc_end_page_v2(content_list_v2)

        for page_idx in range(min(toc_end_page, len(content_list_v2))):
            page_items = content_list_v2[page_idx]
            for item in page_items:
                if item.get('type') == 'title' and item.get('content', {}).get('level') == 1:
                    title = self._extract_text_from_item_cached(json.dumps(item, sort_keys=True))
                    for pattern in self.chapter_patterns:
                        if re.match(pattern, title):
                            toc_chapters.add(title)
                            break

        logger.debug(f"📑 目录中的章节: {toc_chapters}")

        # 在正文中查找章节
        chapters = []
        current_chapter = None

        for page_idx in range(start_page, len(content_list_v2)):
            page_items = content_list_v2[page_idx]

            for item in page_items:
                # 查找一级标题
                if item.get('type') == 'title' and item.get('content', {}).get('level') == 1:
                    title = self._extract_text_from_item_cached(json.dumps(item, sort_keys=True))

                    # 检查是否匹配章节模式
                    is_chapter = False
                    for pattern in self.chapter_patterns:
                        if re.match(pattern, title):
                            is_chapter = True
                            break

                    if is_chapter:
                        # 过滤掉目录中的条目（通常带页码）
                        # 检查标题是否以数字结尾（可能是页码）
                        is_toc_entry = False

                        # 方法1: 检查标题本身是否带页码（如 "第四章 xxx 103"）
                        if re.search(r'\s\d+$', title.strip()):
                            is_toc_entry = True

                        # 方法2: 与目录中的章节比较
                        if not is_toc_entry:
                            for toc_title in toc_chapters:
                                # 如果目录标题包含当前标题且更长，认为是目录条目
                                if toc_title.startswith(title) and len(toc_title) > len(title) + 3:
                                    is_toc_entry = True
                                    break

                        if is_toc_entry:
                            logger.debug(f"  跳过目录项: {title} (page {page_idx})")
                            continue

                        # 这是真正的章节标题
                            # 保存上一章
                            if current_chapter:
                                current_chapter['end_page'] = page_idx - 1
                                chapters.append(current_chapter)

                            # 开始新章节
                            current_chapter = {
                                'title': title,
                                'start_page': page_idx,
                                'end_page': None,
                            }
                            logger.debug(f"检测到章节: {title} (page {page_idx})")

        # 保存最后一章
        if current_chapter:
            current_chapter['end_page'] = len(content_list_v2) - 1
            chapters.append(current_chapter)

        logger.info(f"📚 检测到 {len(chapters)} 个章节")
        return chapters

    def _find_sections_v2(
        self,
        content_list_v2: List,
        start_page: int,
    ) -> List[Dict]:
        """
        查找所有小节标记（作为章节切分的备选方案）

        小节通常是 "1.1", "1.2" 这样的格式

        Returns:
            小节列表，每项包含: title, start_page, end_page
        """
        sections = []
        current_section = None

        for page_idx in range(start_page, len(content_list_v2)):
            page_items = content_list_v2[page_idx]

            for item in page_items:
                # 查找标题
                if item.get('type') == 'title':
                    title = self._extract_text_from_item_cached(json.dumps(item, sort_keys=True))

                    # 检查是否匹配小节模式
                    is_section = False
                    for pattern in self.SECTION_PATTERNS:
                        if re.match(pattern, title):
                            is_section = True
                            break

                    if is_section:
                        # 保存上一节
                        if current_section:
                            current_section['end_page'] = page_idx - 1
                            sections.append(current_section)

                        # 开始新小节
                        current_section = {
                            'title': title,
                            'start_page': page_idx,
                            'end_page': None,
                        }
                        logger.debug(f"检测到小节: {title} (page {page_idx})")

        # 保存最后一节
        if current_section:
            current_section['end_page'] = len(content_list_v2) - 1
            sections.append(current_section)

        if sections:
            logger.info(f"📚 检测到 {len(sections)} 个小节")
        else:
            logger.info("📚 未检测到小节，将把全部内容作为一个批次")

        # 如果没有找到任何小节，返回一个包含全部内容的"章节"
        if not sections:
            sections.append({
                'title': '全部内容',
                'start_page': start_page,
                'end_page': len(content_list_v2) - 1,
            })

        return sections

    def _split_large_chapter(
        self,
        content_list_v2: List,
        start_page: int,
        end_page: int,
        title: str,
        doc_id_prefix: str,
        chapter_num: int,
        max_pages: int,
    ) -> List[Dict]:
        """
        拆分大章节为多个批次
        """
        batches = []
        part_num = 0
        current_start = start_page

        while current_start <= end_page:
            current_end = min(current_start + max_pages - 1, end_page)
            part_num += 1

            # 展平内容
            flattened_content = self._flatten_page_content(content_list_v2, current_start, current_end)

            doc_id = self._generate_safe_doc_id(doc_id_prefix, chapter_num, title, part_num) if doc_id_prefix else None

            batches.append({
                'doc_id': doc_id,
                'title': f"{title} (第{part_num}部分)",
                # 'content_list': flattened_content,
                'page_range': [current_start, current_end],
                'item_count': len(flattened_content),
                'is_split': True,
            })

            current_start = current_end + 1

        return batches

    @lru_cache(maxsize=1000)
    def _extract_text_from_item_cached(self, item_json: str) -> str:
        """
        Cached version of text extraction for v2 format items

        Args:
            item_json: JSON string representation of the item for caching

        Returns:
            Extracted text string
        """
        # Convert JSON string back to dict
        item = json.loads(item_json)
        return extract_text_from_item(item, content_format="v2")

    def _flatten_page_content(self, content_list_v2: List, start_page: int, end_page: int) -> List[Dict]:
        """
        Extract and flatten content from a page range

        Args:
            content_list_v2: V2 format content
            start_page: Starting page index
            end_page: Ending page index

        Returns:
            List of flattened content items with page indices
        """
        flattened_content = []
        for page_idx in range(start_page, end_page + 1):
            if page_idx >= len(content_list_v2):
                break
            for item in content_list_v2[page_idx]:
                if item.get('type') in self.PAGE_AUX_TYPES:
                    continue
                item_with_page = {**item, 'page_idx': page_idx}
                flattened_content.append(item_with_page)
        return flattened_content

    def get_content_list_stats(self, content_list_v2: List) -> Dict[str, Any]:
        """
        获取 content_list_v2 的统计信息

        Returns:
            包含以下字段的字典:
            - total_pages: 总页数
            - total_items: 总内容项数
            - type_counts: 各类型项的数量
            - has_toc: 是否包含目录
            - estimated_chapters: 预估章节数
        """
        total_pages = len(content_list_v2)
        total_items = 0
        type_counts = {}
        has_toc = False
        chapter_count = 0

        for page_items in content_list_v2:
            total_items += len(page_items)
            for item in page_items:
                item_type = item.get('type', 'unknown')
                type_counts[item_type] = type_counts.get(item_type, 0) + 1

                # 检查目录
                text = self._extract_text_from_item_cached(json.dumps(item, sort_keys=True))
                if text and ('目录' in text or 'CONTENTS' in str(text).upper()):
                    has_toc = True

                # 统计一级标题
                if item_type == 'title' and item.get('content', {}).get('level') == 1:
                    for pattern in self.chapter_patterns:
                        if re.match(pattern, text):
                            chapter_count += 1
                            break

        return {
            'total_pages': total_pages,
            'total_items': total_items,
            'type_counts': type_counts,
            'has_toc': has_toc,
            'estimated_chapters': chapter_count,
        }
