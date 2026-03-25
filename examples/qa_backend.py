#!/usr/bin/env python3
"""
RAG知识库问答 Web服务

Usage:
    uv run python qa_backend.py
    # 或
    uvicorn qa_backend:app --reload --port 8000
"""

import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 抑制日志
class ImageDataFilter(logging.Filter):
    """过滤掉包含 image_data 的日志"""
    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            if 'image_data' in msg or msg.startswith("/9j") or "'image_data':" in msg:
                return False
        return True

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
for m in ['lightrag', 'nano_vectordb', 'raganything']:
    logger = logging.getLogger(m)
    logger.setLevel(logging.ERROR)
    logger.addFilter(ImageDataFilter())

# 导入查询类
from query_kb import KnowledgeBaseQuery


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(title="RAG知识库问答")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局查询器
qb: KnowledgeBaseQuery = None


# =============================================================================
# 数据模型
# =============================================================================
class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    vlm_enhanced: bool = False


class QueryResponse(BaseModel):
    answer: str
    mode: str
    success: bool


class HealthResponse(BaseModel):
    status: str
    kb_dir: str
    initialized: bool


class ConfigResponse(BaseModel):
    kb_dir: str
    use_openai: bool
    model: str
    vision_model: str | None = None


# =============================================================================
# 生命周期
# =============================================================================
@app.on_event("startup")
async def startup():
    global qb
    kb_dir = os.getenv("KB_DIR", "./rag_kb")
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    print("=" * 60)
    print("🚀 RAG知识库问答服务启动")
    print("=" * 60)

    try:
        qb = KnowledgeBaseQuery(kb_dir, use_openai=use_openai)
        await qb.initialize()
        print("✅ 服务已就绪")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    print("👋 服务关闭")


# =============================================================================
# 路由
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def frontend():
    """前端页面"""
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        kb_dir=str(qb.kb_dir) if qb else "not_initialized",
        initialized=qb is not None and qb.rag is not None
    )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    if not qb:
        raise HTTPException(status_code=503, detail="服务未初始化")
    return ConfigResponse(
        kb_dir=str(qb.kb_dir),
        use_openai=qb.use_openai,
        model=os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-72B-Instruct") if qb.use_openai else os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        vision_model=os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-32B-Instruct") if qb.use_openai else None
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not qb:
        raise HTTPException(status_code=503, detail="服务未初始化")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        answer = await qb.query(
            req.question,
            mode=req.mode,
            vlm_enhanced=req.vlm_enhanced if req.vlm_enhanced else None,
            show_context=False
        )
        return QueryResponse(answer=answer, mode=req.mode, success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


# =============================================================================
# 前端HTML
# =============================================================================
FRONTEND_HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识库问答 - RAGAnything</title>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        #app { max-width: 900px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; display: flex; flex-direction: column; height: calc(100vh - 40px); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .status { display: flex; align-items: center; gap: 8px; font-size: 0.875rem; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #4ade80; }
        .status-dot.disconnected { background: #f87171; }
        .status-dot.checking { background: #fbbf24; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .settings { padding: 15px 20px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; display: flex; gap: 15px; align-items: center; }
        .settings label { font-size: 0.875rem; color: #64748b; font-weight: 500; }
        .settings select { padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.875rem; background: white; cursor: pointer; }
        .settings select:focus { outline: none; border-color: #667eea; }
        .chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .message { display: flex; gap: 12px; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { flex-direction: row-reverse; }
        .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; flex-shrink: 0; }
        .message.assistant .avatar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .message.user .avatar { background: #e2e8f0; }
        .message.error .avatar { background: #ef4444; }
        .bubble { max-width: 70%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; }
        .message.assistant .bubble { background: #f1f5f9; color: #1e293b; }
        .message.user .bubble { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .message.error .bubble { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .message.error { flex-direction: row; }
        .message.error .avatar { order: -1; }
        .input-area { padding: 20px; border-top: 1px solid #e2e8f0; display: flex; gap: 12px; }
        .input-area input { flex: 1; padding: 12px 16px; border: 2px solid #e2e8f0; border-radius: 24px; font-size: 1rem; outline: none; transition: border-color 0.2s; }
        .input-area input:focus { border-color: #667eea; }
        .input-area input:disabled { background: #f1f5f9; }
        .input-area button { padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 24px; font-size: 1rem; font-weight: 500; cursor: pointer; transition: opacity 0.2s, transform 0.1s; }
        .input-area button:hover:not(:disabled) { opacity: 0.9; }
        .input-area button:active:not(:disabled) { transform: scale(0.98); }
        .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
        .typing { display: inline-flex; gap: 4px; padding: 8px 0; }
        .typing span { width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; animation: typing 1.4s infinite; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-4px); } }
        .welcome { text-align: center; padding: 60px 20px; color: #64748b; }
        .welcome h2 { font-size: 1.5rem; margin-bottom: 8px; color: #1e293b; }
        .chat-box::-webkit-scrollbar { width: 6px; }
        .chat-box::-webkit-scrollbar-track { background: transparent; }
        .chat-box::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        @media (max-width: 600px) { body { padding: 0; } #app { height: 100vh; border-radius: 0; } .bubble { max-width: 85%; } }
    </style>
</head>
<body>
    <div id="app">
        <div class="header">
            <h1>📚 知识库问答</h1>
            <div class="status">
                <span class="status-dot" :class="{'disconnected': status === 'disconnected', 'checking': status === 'checking'}"></span>
                <span>{{ statusText }}</span>
            </div>
        </div>
        <div class="settings">
            <label>查询模式:</label>
            <select v-model="mode" :disabled="loading">
                <option value="local">本地模式 (Local)</option>
                <option value="global">全局模式 (Global)</option>
                <option value="hybrid">混合模式 (Hybrid)</option>
                <option value="naive">朴素模式 (Naive)</option>
            </select>
            <label style="margin-left: auto;"><input type="checkbox" v-model="vlmEnhanced" :disabled="loading || !config.use_openai"> VLM增强</label>
        </div>
        <div class="chat-box" ref="chatBox">
            <div class="welcome" v-if="messages.length === 0">
                <h2>👋 欢迎使用知识库问答</h2>
                <p>输入问题开始查询，支持数学、科学等知识内容</p>
            </div>
            <div v-for="(msg, idx) in messages" :key="idx" :class="['message', msg.role]">
                <div class="avatar"><span v-if="msg.role === 'user'">❓</span><span v-else-if="msg.role === 'assistant'">💡</span><span v-else>⚠️</span></div>
                <div class="bubble"><span v-if="msg.role === 'assistant' && msg.loading" class="typing"><span></span><span></span><span></span></span><span v-else>{{ msg.content }}</span></div>
            </div>
        </div>
        <div class="input-area">
            <input v-model="question" @keyup.enter="send" placeholder="输入问题..." :disabled="loading || status === 'disconnected'">
            <button @click="send" :disabled="loading || !question.trim() || status === 'disconnected'">{{ loading ? '查询中...' : '发送' }}</button>
        </div>
    </div>
    <script>
        const { createApp } = Vue
        createApp({
            data() { return { question: '', messages: [], loading: false, mode: 'hybrid', vlmEnhanced: false, status: 'checking', statusText: '检查中...', config: { use_openai: false } } },
            mounted() { this.checkHealth(); this.loadConfig(); setInterval(this.checkHealth, 30000); },
            methods: {
                async checkHealth() { try { const res = await fetch('./api/health'); if (res.ok) { this.status = 'connected'; this.statusText = '已连接'; } else throw new Error('fail'); } catch { this.status = 'disconnected'; this.statusText = '未连接'; } },
                async loadConfig() { try { const res = await fetch('./api/config'); if (res.ok) this.config = await res.json(); } catch {} },
                async send() {
                    if (!this.question.trim() || this.loading) return;
                    if (this.status === 'disconnected') { this.messages.push({ role: 'error', content: '无法连接到服务器' }); this.scrollToBottom(); return; }
                    const q = this.question; this.question = ''; this.loading = true;
                    this.messages.push({ role: 'user', content: q });
                    const loadingMsg = { role: 'assistant', content: '', loading: true };
                    this.messages.push(loadingMsg); this.scrollToBottom();
                    try { const res = await fetch('./api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q, mode: this.mode, vlm_enhanced: this.vlmEnhanced }) });
                        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'fail'); }
                        const data = await res.json(); loadingMsg.content = data.answer; loadingMsg.loading = false;
                        if (!data.success) { loadingMsg.role = 'error'; loadingMsg.content = data.answer || '查询失败'; }
                    } catch (e) { loadingMsg.role = 'error'; loadingMsg.content = '请求失败: ' + e.message; loadingMsg.loading = false; }
                    this.loading = false; this.scrollToBottom();
                },
                scrollToBottom() { this.$nextTick(() => { const cb = this.$refs.chatBox; if (cb) cb.scrollTop = cb.scrollHeight; }); }
            }
        }).mount('#app');
    </script>
</body>
</html>'''


# =============================================================================
# 启动
# =============================================================================
def main():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"🌐 访问地址: http://localhost:{port}")
    print(f"📖 前端页面: http://localhost:{port}/")
    print(f"📚 API文档: http://localhost:{port}/docs")

    uvicorn.run("qa_backend:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
