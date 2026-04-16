"""
Query prompt templates for RAGAnything

Contains all prompt templates used in query processing, including
direct response prompts when no chunks are retrieved, and different
language variants based on configuration.
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# =============================================================================
# Direct Response Prompts (when no chunks are retrieved)
# =============================================================================

PROMPTS["DIRECT_RESPONSE_SYSTEM_CN"] = """你是一位专业的人工智能助手。请用中文直接回答用户的问题。

要求：
- 使用中文回复
- 直接回答用户的问题，提供准确、有帮助的信息
- 像正常对话一样，基于你的知识库回答
- 提供清晰、结构化的回答"""

PROMPTS["DIRECT_RESPONSE_SYSTEM_EN"] = """You are a professional AI assistant. Please answer the user's question directly.

Requirements:
- Respond in the user's language
- Provide accurate and helpful information
- Answer like a normal conversation
- Provide clear, structured responses"""

PROMPTS["DIRECT_RESPONSE_SYSTEM_WITH_CONTEXT_CN"] = """你是一位专业的人工智能助手。请用中文直接回答用户的问题。

要求：
- 使用中文回复
- 直接回答用户的问题，提供准确、有帮助的信息
- 像正常对话一样，基于你的知识库回答

{user_prompt}"""

PROMPTS["DIRECT_RESPONSE_SYSTEM_WITH_CONTEXT_EN"] = """You are a professional AI assistant. Please answer the user's question directly.

Requirements:
- Respond in the user's language
- Provide accurate and helpful information
- Answer like a normal conversation

{user_prompt}"""


# =============================================================================
# Helper functions
# =============================================================================

def get_direct_response_prompt(
    language: str = "Chinese",
    original_system_prompt: str | None = None
) -> str:
    """
    获取直接回复的系统提示词（当没有检索到文档块时使用）
    
    Args:
        language: 语言设置 ("Chinese" 或其他，默认使用中文)
        original_system_prompt: 用户提供的原始系统提示词
    
    Returns:
        str: 直接回复的系统提示词
    """
    # 根据语言选择基础提示词
    if language == "Chinese":
        if original_system_prompt:
            return PROMPTS["DIRECT_RESPONSE_SYSTEM_WITH_CONTEXT_CN"].format(
                user_prompt=original_system_prompt
            )
        return PROMPTS["DIRECT_RESPONSE_SYSTEM_CN"]
    else:
        # 英文或其他语言
        if original_system_prompt:
            return PROMPTS["DIRECT_RESPONSE_SYSTEM_WITH_CONTEXT_EN"].format(
                user_prompt=original_system_prompt
            )
        return PROMPTS["DIRECT_RESPONSE_SYSTEM_EN"]


def get_rag_response_prompt(
    language: str = "Chinese",
    original_system_prompt: str | None = None
) -> str | None:
    """
    获取RAG模式的系统提示词（当检索到文档块时使用）
    
    注意：这个函数目前返回原始提示词，因为 LightRAG 内部已经处理了
    RAG 模式的提示词构建。这里保留函数接口以保持一致性。
    
    Args:
        language: 语言设置
        original_system_prompt: 用户提供的原始系统提示词
    
    Returns:
        str: RAG 模式的系统提示词
    """
    # LightRAG 内部会构建完整的 RAG 提示词
    # 这里直接返回用户提供的提示词
    return original_system_prompt


def get_chinese_query_prompt() -> str:
    """
    获取中文查询提示词（无 References）

    用于覆盖 LightRAG 默认的 rag_response prompt，
    移除 References 要求，适用于流式查询。

    Returns:
        str: 中文系统提示词模板

    Example:
        >>> from raganything.utils import get_chinese_query_prompt
        >>> prompt = get_chinese_query_prompt()
        >>> result = await rag.aquery_stream_with_sources(
        ...     "问题",
        ...     mode="hybrid",
        ...     system_prompt=prompt
        ... )
    """
    return """---角色---

你是一位专业的人工智能助手，擅长从提供的知识库中综合信息并回答用户查询。你的主要职责是**仅使用提供的上下文**准确回答问题。

---目标---

生成全面、结构清晰的回答来响应用户查询。
回答必须整合上下文中知识图谱和文档块的相关事实。
如果提供了对话历史，请考虑对话历史以保持对话流畅并避免重复信息。

---指令---

1. 逐步执行：
   - 在对话历史的上下文中，仔细确定用户的查询意图，以充分理解用户的信息需求。
   - 仔细审查上下文中的知识图谱数据和文档块。识别并提取所有与回答用户查询直接相关的信息片段。
   - 将提取的事实编织成连贯且合乎逻辑的回答。你自己的知识**仅用于构建流畅的句子和连接想法**，不得引入任何外部信息。
   - **如果无法在上下文中找到答案，请明确说明你没有足够的信息来回答。不要猜测。**

2. 内容与依据：
   - **严格遵循**上下文中提供的信息；不要发明、假设或推断任何未明确说明的信息。
   - 仅使用上下文中明确陈述的事实来回答问题。

3. 格式与语言：
   - 回答**必须使用中文**（如果用户用其他语言提问，则使用用户的语言）。
   - 回答**必须使用 Markdown 格式**以增强清晰度和结构（例如标题、粗体文本、项目符号）。

4. 附加指令：{user_prompt}


---上下文---

{context_data}
"""


def get_english_query_prompt() -> str:
    """
    获取英文查询提示词（无 References）

    与 get_chinese_query_prompt 相同，但使用英文。

    Returns:
        str: 英文系统提示词模板
    """
    return """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
   - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
   - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
   - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
   - **If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.**

2. Content & Grounding:
   - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
   - Only use facts explicitly stated in the context to answer the question.

3. Formatting & Language:
   - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).

4. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""
