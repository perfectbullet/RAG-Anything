"""
Prompt templates for multimodal content processing

Contains all prompt templates used in modal processors for analyzing
different types of content (images, tables, equations, etc.)
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# System prompts for different analysis types
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "你是一位专业的图像分析师。请提供详细、准确的描述。"
)
PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "你是一位专业的图像分析师。请根据可用信息提供详细分析。"
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "你是一位专业的数据分析师。请提供详细的表格分析和具体见解。"
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "你是一位专业的数学分析师。请提供详细的数学分析。"
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "你是一位专业的内容分析师，擅长分析 {content_type} 类型的内容。"
)

# Image analysis prompt template
PROMPTS[
    "vision_prompt"
] = """请详细分析此图像并提供 JSON 格式的响应：

{{
    "detailed_description": "图像的综合详细描述，遵循以下准则：
    - 描述整体构图和布局
    - 识别所有对象、人物、文本和视觉元素
    - 解释元素之间的关系
    - 注意颜色、光照和视觉风格
    - 描述显示的任何动作或活动
    - 包含相关技术细节（如图表、图表等）
    - 始终使用具体名称而非代词",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容及其重要性的简洁总结（最多100字）"
    }}
}}

附加信息：
- 图像路径：{image_path}
- 标题：{captions}
- 注释：{footnotes}

专注于提供准确、详细的视觉分析，以便于知识检索。"""

# Image analysis prompt with context support
PROMPTS[
    "vision_prompt_with_context"
] = """请结合周围上下文详细分析此图像，并提供 JSON 格式的响应：

{{
    "detailed_description": "图像的综合详细描述，遵循以下准则：
    - 描述整体构图和布局
    - 识别所有对象、人物、文本和视觉元素
    - 解释元素之间的关系及其与周围上下文的联系
    - 注意颜色、光照和视觉风格
    - 描述显示的任何动作或活动
    - 包含相关技术细节（如图表、图表等）
    - 在相关时引用与周围内容的联系
    - 始终使用具体名称而非代词",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容、其重要性及与周围内容关系的简洁总结（最多100字）"
    }}
}}

来自周围内容的上下文：
{context}

图像详情：
- 图像路径：{image_path}
- 标题：{captions}
- 注释：{footnotes}

专注于提供准确、详细的视觉分析，并融入上下文，以便于知识检索。"""

# Image analysis prompt with text fallback
PROMPTS["text_prompt"] = """根据以下图像信息，提供分析：

图像路径：{image_path}
标题：{captions}
注释：{footnotes}

{vision_prompt}"""

# Table analysis prompt template
PROMPTS[
    "table_prompt"
] = """请分析此表格内容并提供 JSON 格式的响应：

{{
    "detailed_description": "表格的综合分析，包括：
    - 表格结构和组织
    - 列标题及其含义
    - 关键数据点和模式
    - 统计见解和趋势
    - 数据元素之间的关系
    - 所呈现数据的重要性
    始终使用具体名称和数值，而非通用引用。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "表格用途和关键发现的简洁总结（最多100字）"
    }}
}}

表格信息：
图像路径：{table_img_path}
标题：{table_caption}
内容：{table_body}
注释：{table_footnote}

专注于从表格数据中提取有意义的见解和关系。"""

# Table analysis prompt with context support
PROMPTS[
    "table_prompt_with_context"
] = """请结合周围上下文分析此表格内容，并提供 JSON 格式的响应：

{{
    "detailed_description": "表格的综合分析，包括：
    - 表格结构和组织
    - 列标题及其含义
    - 关键数据点和模式
    - 统计见解和趋势
    - 数据元素之间的关系
    - 所呈现数据相对于周围上下文的重要性
    - 表格如何支持或说明周围内容中的概念
    始终使用具体名称和数值，而非通用引用。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "表格用途、关键发现及其与周围内容关系的简洁总结（最多100字）"
    }}
}}

来自周围内容的上下文：
{context}

表格信息：
图像路径：{table_img_path}
标题：{table_caption}
内容：{table_body}
注释：{table_footnote}

专注于在周围内容的背景下，从表格数据中提取有意义的见解和关系。"""

# Equation analysis prompt template
PROMPTS[
    "equation_prompt"
] = """请分析此数学公式并提供 JSON 格式的响应：

{{
    "detailed_description": "公式的综合分析，包括：
    - 数学含义和解释
    - 变量及其定义
    - 使用的数学运算和函数
    - 应用领域和上下文
    - 物理或理论意义
    - 与其他数学概念的关系
    - 实际应用或用例
    始终使用具体的数学术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "公式用途及其重要性的简洁总结（最多100字）"
    }}
}}

公式信息：
公式：{equation_text}
格式：{equation_format}

专注于提供数学见解并解释公式的重要性。"""

# Equation analysis prompt with context support
PROMPTS[
    "equation_prompt_with_context"
] = """请结合周围上下文分析此数学公式，并提供 JSON 格式的响应：

{{
    "detailed_description": "公式的综合分析，包括：
    - 数学含义和解释
    - 上下文中的变量及其定义
    - 使用的数学运算和函数
    - 基于周围材料的应用领域和上下文
    - 物理或理论意义
    - 与上下文中提到的其他数学概念的关系
    - 实际应用或用例
    - 公式与更广泛讨论或框架的关系
    始终使用具体的数学术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "公式用途、其重要性及其在周围内容中作用的简洁总结（最多100字）"
    }}
}}

来自周围内容的上下文：
{context}

公式信息：
公式：{equation_text}
格式：{equation_format}

专注于在更广泛背景下提供数学见解并解释公式的重要性。"""

# Generic content analysis prompt template
PROMPTS[
    "generic_prompt"
] = """请分析此 {content_type} 类型的内容并提供 JSON 格式的响应：

{{
    "detailed_description": "内容的综合分析，包括：
    - 内容结构和组织
    - 关键信息和要素
    - 组件之间的关系
    - 上下文和重要性
    - 与知识检索相关的相关细节
    始终使用适合 {content_type} 内容的具体术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "内容用途和关键点的简洁总结（最多100字）"
    }}
}}

内容：{content}

专注于提取对知识检索有用的有意义信息。"""

# Generic content analysis prompt with context support
PROMPTS[
    "generic_prompt_with_context"
] = """请结合周围上下文分析此 {content_type} 类型的内容，并提供 JSON 格式的响应：

{{
    "detailed_description": "内容的综合分析，包括：
    - 内容结构和组织
    - 关键信息和要素
    - 组件之间的关系
    - 相对于周围内容的上下文和重要性
    - 此内容如何连接或支持更广泛的讨论
    - 与知识检索相关的相关细节
    始终使用适合 {content_type} 内容的具体术语。",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "内容用途、关键点及其与周围内容关系的简洁总结（最多100字）"
    }}
}}

来自周围内容的上下文：
{context}

内容：{content}

专注于提取对知识检索和理解内容在更广泛背景中作用有用的有意义信息。"""

# Modal chunk templates
PROMPTS["image_chunk"] = """
图像内容分析：
图像路径：{image_path}
标题：{captions}
注释：{footnotes}

视觉分析：{enhanced_caption}"""

PROMPTS["table_chunk"] = """表格分析：
图像路径：{table_img_path}
标题：{table_caption}
结构：{table_body}
注释：{table_footnote}

分析：{enhanced_caption}"""

PROMPTS["equation_chunk"] = """数学公式分析：
公式：{equation_text}
格式：{equation_format}

数学分析：{enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} 内容分析：
内容：{content}

分析：{enhanced_caption}"""

# Query-related prompts
PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "请简要描述这张图像的主要内容、关键要素和重要信息。"
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "你是一位专业的图像分析师，能准确描述图像内容。"
)

PROMPTS[
    "QUERY_TABLE_ANALYSIS"
] = """请分析以下表格数据的主要内容、结构和关键信息：

表格数据：
{table_data}

表格标题：{table_caption}

请简要总结表格的主要内容、数据特征和重要发现。"""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "你是一位专业的数据分析师，能准确分析表格数据。"
)

PROMPTS[
    "QUERY_EQUATION_ANALYSIS"
] = """请解释以下数学公式的含义和用途：

LaTeX 公式：{latex}
公式标题：{equation_caption}

请简要说明此公式的数学含义、应用场景和重要性。"""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "你是一位数学专家，能清晰解释数学公式。"
)

PROMPTS[
    "QUERY_GENERIC_ANALYSIS"
] = """请分析以下 {content_type} 类型的内容并提取其主要信息和关键特征：

内容：{content_str}

请简要总结此内容的主要特征和重要信息。"""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "你是一位专业的内容分析师，能准确分析 {content_type} 类型的内容。"
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\n请根据用户查询和提供的多模态内容信息，提供全面的回答。"
)
