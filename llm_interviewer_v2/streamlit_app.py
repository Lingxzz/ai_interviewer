"""
LLM 工程师智能面试系统 — Streamlit 前端
v3: 信息层级重构 · 全局微交互 · 骨架屏 · 报告页重排版 · 响应式适配
"""

import asyncio
import sys
import os
import time
import json

import streamlit as st
import requests

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "llm_interviewer_test2_0")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from core.orchestrator import OrchestratorAgent
from state.interview_state import AgentRole, InterviewPhase, AgentMessage
from core.models import Topic
from tools.pdf_reader import extract_pdf_text
from agents.self_intro_agent import SelfIntroAgent
from agents.hrd_agent import HRDAgent
from agents.final_report_agent import FinalReportAgent

# ══════════════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI 面试官 · LLM Engineer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 全局样式
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #f6f8fa; color: #1f2328; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #d0d7de;
}
[data-testid="stSidebar"] * { transition: color 0.15s ease; }

/* ── Typography ───────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Sora', sans-serif;
    letter-spacing: -0.025em;
    color: #1f2328;
}

/* ── Inputs ───────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: #ffffff !important;
    border: 1.5px solid #d0d7de !important;
    border-radius: 8px !important;
    color: #1f2328 !important;
    font-family: 'Sora', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput input:hover,
.stTextArea textarea:hover {
    border-color: #0969da !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #0969da !important;
    box-shadow: 0 0 0 3px rgba(9,105,218,0.12) !important;
    outline: none !important;
}

/* ── Buttons — 基础绿色 ────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1a7f37, #2da44e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(26,127,55,0.28) !important;
    filter: brightness(1.05) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 6px rgba(26,127,55,0.2) !important;
}

/* ── 危险按钮 ── */
.danger-btn .stButton > button {
    background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
}
.danger-btn .stButton > button:hover {
    box-shadow: 0 6px 20px rgba(185,28,28,0.28) !important;
}

/* ── 蓝色运行按钮 ── */
.run-btn .stButton > button {
    background: linear-gradient(135deg, #0550ae, #0969da) !important;
}
.run-btn .stButton > button:hover {
    box-shadow: 0 6px 20px rgba(9,105,218,0.28) !important;
}

/* ── 次要按钮（灰色） ── */
.secondary-btn .stButton > button {
    background: #ffffff !important;
    color: #57606a !important;
    border: 1.5px solid #d0d7de !important;
    box-shadow: none !important;
}
.secondary-btn .stButton > button:hover {
    background: #f6f8fa !important;
    border-color: #0969da !important;
    color: #0969da !important;
    box-shadow: none !important;
}

/* ── 信息层级 ─── 三个容器权重不同 ────────────────────────── */

/* 一级：题目卡片（最高权重：强边框 + 蓝色左边条 + 明显阴影） */
.question-card {
    background: #ffffff;
    border: 1.5px solid #b6d4f7;
    border-left: 5px solid #0969da;
    border-radius: 14px;
    padding: 1.8rem 1.8rem 1.4rem;
    margin: 1.2rem 0 0.6rem;
    box-shadow: 0 4px 16px rgba(9,105,218,0.10), 0 1px 4px rgba(31,35,40,0.06);
    transition: box-shadow 0.2s ease;
}
.question-card:hover {
    box-shadow: 0 6px 24px rgba(9,105,218,0.14), 0 2px 6px rgba(31,35,40,0.08);
}

/* 二级：评估卡片（中等权重：淡绿背景 + 细边框） */
.eval-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid #2da44e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0 0.8rem;
    box-shadow: 0 1px 4px rgba(31,35,40,0.04);
    transition: box-shadow 0.2s ease;
}

/* 三级：聊天气泡（最低权重：极细边框 + 无阴影） */
.chat-bot {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 0 14px 14px 14px;
    padding: 0.9rem 1.1rem;
    margin: 0.3rem 0;
    max-width: 82%;
    color: #1f2328;
    transition: border-color 0.2s ease;
}
.chat-bot:hover { border-color: #d0d7de; }

.chat-user {
    background: #ddf4ff;
    border: 1px solid #b6e3ff;
    border-radius: 14px 0 14px 14px;
    padding: 0.9rem 1.1rem;
    margin: 0.3rem 0 0.3rem auto;
    max-width: 82%;
    color: #0550ae;
}

/* 流式反馈（介于二三级之间：浅灰背景）*/
.stream-box {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-left: 3px solid #57606a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    line-height: 1.85;
    color: #1f2328;
    font-size: 0.94rem;
    min-height: 56px;
    transition: border-color 0.2s ease;
}

/* ── 代码沙箱 ─────────────────────────────────────────────── */
.sandbox-header {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 1rem;
    font-size: 0.82rem;
    color: #57606a;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.output-panel {
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    padding: 1rem 1.1rem;
    min-height: 56px;
    border-radius: 8px;
    white-space: pre-wrap;
    line-height: 1.65;
    margin-top: 0.5rem;
    transition: opacity 0.3s ease;
}
.output-ok      { border-left: 3px solid #2da44e; }
.output-err     { border-left: 3px solid #cf222e; }
.output-timeout { border-left: 3px solid #e3b341; }

/* ── 骨架屏动画 ───────────────────────────────────────────── */
@keyframes skeleton-shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
.skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 800px 100%;
    animation: skeleton-shimmer 1.4s infinite linear;
    border-radius: 6px;
}
.skeleton-line {
    height: 14px;
    margin: 10px 0;
    border-radius: 6px;
}
.skeleton-card {
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.6rem 0;
}
/* 打点动画 */
@keyframes dot-blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40%            { opacity: 1;   transform: scale(1); }
}
.thinking-dots span {
    display: inline-block;
    width: 7px; height: 7px;
    background: #0969da;
    border-radius: 50%;
    margin: 0 2px;
    animation: dot-blink 1.4s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── 分数徽章 ─────────────────────────────────────────────── */
.score-badge {
    display: inline-block;
    padding: 0.22rem 0.72rem;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.88rem;
    transition: transform 0.15s ease;
}
.score-badge:hover { transform: scale(1.05); }
.score-high { background:#dafbe1; color:#1a7f37; border:1px solid #82cfad; }
.score-mid  { background:#fff8c5; color:#7d4e00; border:1px solid #d4a72c; }
.score-low  { background:#ffebe9; color:#cf222e; border:1px solid #ff8182; }

/* ── Topic 标签 ───────────────────────────────────────────── */
.topic-tag {
    display: inline-block;
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.12rem 0.55rem;
    font-size: 0.76rem;
    color: #57606a;
    margin-right: 0.35rem;
    transition: background 0.15s ease, border-color 0.15s ease;
}
.topic-tag:hover { background: #eef2f6; border-color: #0969da; color: #0969da; }
.topic-tag-code { background:#ddf4ff; border-color:#54aeff; color:#0550ae; }

/* ── 进度条 ───────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1a7f37, #0969da) !important;
    border-radius: 4px !important;
    transition: width 0.4s ease !important;
}

/* ── 信息框 ───────────────────────────────────────────────── */
.info-box {
    background: #ddf4ff;
    border: 1px solid #b6e3ff;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #0550ae;
    font-size: 0.88rem;
    line-height: 1.6;
}
.warn-box {
    background: #fff8c5;
    border: 1px solid #f1c40f;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #7d4e00;
    font-size: 0.88rem;
}

/* ── Metric ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #0969da;
    box-shadow: 0 2px 8px rgba(9,105,218,0.1);
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #0969da !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: #57606a !important; font-size: 0.82rem !important; }

/* ── 报告页专用 ──────────────────────────────────────────── */
.report-hero {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
    border: 1px solid #b6d4f7;
    border-radius: 16px;
    padding: 2.4rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.report-score-big {
    font-size: 3.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    line-height: 1;
}
.report-label {
    font-size: 0.82rem;
    color: #57606a;
    margin-top: 0.3rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.report-section-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    transition: box-shadow 0.2s ease;
}
.report-section-card:hover {
    box-shadow: 0 4px 16px rgba(31,35,40,0.08);
}
.report-section-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #57606a;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.report-item {
    padding: 0.55rem 0;
    border-bottom: 1px solid #f0f2f5;
    font-size: 0.93rem;
    line-height: 1.55;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    transition: background 0.15s ease;
}
.report-item:last-child { border-bottom: none; }
.report-item:hover { background: #fafbfc; border-radius: 6px; padding-left: 6px; }
.report-item-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}

/* ── 答题详情 expander ────────────────────────────────────── */
.record-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: 0.5rem 0;
    transition: box-shadow 0.2s ease;
}
.record-card:hover { box-shadow: 0 3px 12px rgba(31,35,40,0.08); }

/* ── 分隔线 ───────────────────────────────────────────────── */
hr { border-color: #e8ecf0 !important; margin: 1rem 0 !important; }

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #d0d7de !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 8px rgba(31,35,40,0.06) !important;
}

/* ── Selectbox ────────────────────────────────────────────── */
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #d0d7de !important;
    color: #1f2328 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease !important;
}

/* ── 响应式：窄屏适配 ─────────────────────────────────────── */
@media (max-width: 768px) {
    .question-card { padding: 1.2rem; }
    .chat-bot, .chat-user { max-width: 95%; }
    .report-score-big { font-size: 2.6rem; }
    .report-hero { padding: 1.5rem 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "phase": "setup",
        "orchestrator": None,
        "chat_history": [],
        "current_question": None,
        "current_q_num": 0,
        "awaiting_answer": False,
        "awaiting_followup": False,
        "current_followup_text": "",
        "report": None,
        "routing_log": [],
        "error": None,
        "code_content": "",
        "code_output": None,
        "report_summary_cache": "",  # 流式总评缓存，避免重复生成
        "self_intro_text": "",
        "self_intro_chat": [],
        "self_intro_qa_done": False,
        "self_intro_processing": False,
        "tech_processing": False,
        "hrd_chat": [],
        "hrd_qa_done": False,
        "hrd_processing": False,
        "current_hrd_question": "",
        "hrd_q_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════
# 骨架屏组件
# ══════════════════════════════════════════════════════════════════

def _skeleton_lines(n: int = 3, widths: list = None):
    widths = widths or ["100%", "85%", "70%"]
    lines = "".join(
        f'<div class="skeleton skeleton-line" style="width:{widths[i % len(widths)]}"></div>'
        for i in range(n)
    )
    return f'<div class="skeleton-card">{lines}</div>'


def _thinking_indicator(label: str = "AI 正在思考"):
    return (
        f'<div style="display:flex;align-items:center;gap:0.7rem;'
        f'padding:0.9rem 1.2rem;background:#f6f8fa;border:1px solid #d0d7de;'
        f'border-radius:10px;color:#57606a;font-size:0.9rem;">'
        f'{label}'
        f'<div class="thinking-dots">'
        f'<span></span><span></span><span></span>'
        f'</div></div>'
    )


# ══════════════════════════════════════════════════════════════════
# 流式 LLM 输出
# ══════════════════════════════════════════════════════════════════

def _stream_llm(system: str, user: str):
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model, "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line: continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]": break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta: yield delta
                except Exception:
                    continue


def _stream_into(placeholder, system: str, user: str) -> str:
    """流式写入到 placeholder，返回完整文本。"""
    # 先显示打点动画
    placeholder.markdown(_thinking_indicator(), unsafe_allow_html=True)
    full = ""
    try:
        for chunk in _stream_llm(system, user):
            full += chunk
            placeholder.markdown(
                f'<div class="stream-box">{full}'
                f'<span style="color:#0969da;animation:none">▌</span></div>',
                unsafe_allow_html=True
            )
        placeholder.markdown(f'<div class="stream-box">{full}</div>', unsafe_allow_html=True)
    except Exception as e:
        placeholder.warning(f"流式输出不可用：{e}")
    return full


def _stream_eval_feedback(question_text: str, answer: str, rubric: str) -> str:
    system = (
        "你是一位资深 LLM 工程师面试官。请对候选人的回答给出专业、简洁的评估反馈，"
        "包括：亮点、不足、改进方向。用中文，150字以内，语气温和直接，不需要打分。"
    )
    user = f"【题目】{question_text}\n\n【候选人回答】{answer}\n\n【评分标准参考】{rubric}"
    return _stream_into(st.empty(), system, user)


def _stream_report_summary(name: str, years: int, score_detail: str,
                           avg_score: float, topic_scores: dict) -> str:
    system = (
        "你是一位技术招聘负责人。根据面试记录，用3-4句话给出客观专业的候选人综合评价，"
        "涵盖技术深度、知识广度、表达能力。语言简练，有具体依据。"
    )
    user = (
        f"候选人：{name}，{years}年经验\n平均分：{avg_score:.1f}/10\n"
        f"各方向：{json.dumps(topic_scores, ensure_ascii=False)}\n\n"
        f"面试记录：\n{score_detail[:800]}"
    )
    return _stream_into(st.empty(), system, user)


# ══════════════════════════════════════════════════════════════════
# 代码执行沙箱
# ══════════════════════════════════════════════════════════════════

from tools.code_executor import CodeExecutor

_executor = CodeExecutor(timeout=10)


def run_code_sandbox(code: str, language: str = "python") -> dict:
    if language != "python":
        return {"stdout": "", "stderr": "当前沙箱仅支持 Python", "status": "err"}
    return _executor.run(code)


def _render_code_editor(question_text: str) -> tuple[str, dict | None]:
    st.markdown(
        '<div class="sandbox-header">'
        '⚙️ &nbsp;Python 3 &nbsp;·&nbsp;'
        '<span style="color:#2da44e;">沙箱隔离</span>'
        '&nbsp;·&nbsp;<span style="color:#57606a;">最长执行 10s</span>'
        '</div>',
        unsafe_allow_html=True
    )
    code = st.text_area(
        "代码编辑器",
        value=st.session_state.code_content,
        height=280,
        key="code_editor_area",
        placeholder="# 在此编写你的 Python 代码\n# print() 输出会显示在下方\n\n",
        label_visibility="collapsed",
    )
    col_run, col_clear, _ = st.columns([1, 1, 5])
    output = st.session_state.code_output

    with col_run:
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_clicked = st.button("▶ 运行代码", use_container_width=True, key="run_code_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_clear:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("清空输出", use_container_width=True, key="clear_output_btn"):
            st.session_state.code_output = None
            st.session_state.code_content = code
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked:
        st.session_state.code_content = code
        # 显示骨架屏等待
        with st.spinner(""):
            output = run_code_sandbox(code)
        st.session_state.code_output = output
        st.rerun()

    if output is not None:
        status = output.get("status", "ok")
        status_map = {
            "ok": ("✅ 运行成功", "output-ok"),
            "err": ("❌ 运行错误", "output-err"),
            "timeout": ("⏱ 执行超时", "output-timeout"),
        }
        label, cls = status_map.get(status, ("• 输出", "output-ok"))
        out_text = (output.get("stdout") or "").strip()
        err_text = (output.get("stderr") or "").strip()
        display = out_text + ("\n\n# STDERR:\n" + err_text if err_text else "")
        if not display: display = "（无输出）"
        st.markdown(
            f'<div style="font-size:0.8rem;color:#57606a;margin:0.6rem 0 0.25rem">{label}</div>'
            f'<div class="output-panel {cls}">{display}</div>',
            unsafe_allow_html=True
        )

    return code, output


# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════

def _score_class(s): return "score-high" if s >= 7.5 else ("score-mid" if s >= 5 else "score-low")


def _score_emoji(s): return "🌟" if s >= 8 else ("✅" if s >= 7 else ("⚠️" if s >= 5 else "❌"))


def _add_chat(role, content, meta=None):
    st.session_state.chat_history.append({"role": role, "content": content, "meta": meta or {}})


def _score_badge(s):
    return f'<span class="score-badge {_score_class(s)}">{s:.1f} / 10</span>'


def _is_coding(q): return q is not None and q.topic == Topic.CODING


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result(timeout=120)
        return loop.run_until_complete(coro)
    except Exception:
        return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.5rem 0 0.2rem;">'
            '<span style="font-size:1.4rem;font-weight:700;letter-spacing:-0.02em;">🤖 AI 面试官</span><br>'
            '<span style="font-size:0.8rem;color:#57606a;letter-spacing:0.03em;">LLM ENGINEER ASSESSMENT</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.divider()

        phase = st.session_state.phase
        orch: OrchestratorAgent = st.session_state.orchestrator

        phase_labels = {
            "setup": "信息收集",
            "profiling": "简历分析",
            "self_intro": "自我介绍",
            "interview": "技术面试",
            "hrd_interview": "HRD面试",
            "report": "综合报告",
        }
        if phase == "setup":
            st.markdown('<div class="info-box">📋 请填写候选人信息后开始面试</div>',
                        unsafe_allow_html=True)

        elif orch and orch.state:
            s = orch.state

            # ── 候选人信息 ──
            st.markdown(
                f'<div style="background:#f6f8fa;border:1px solid #e8ecf0;border-radius:10px;'
                f'padding:0.8rem 1rem;margin-bottom:0.8rem;">'
                f'<div style="font-weight:600;font-size:0.95rem;">{s.profile.name}</div>'
                f'<div style="font-size:0.8rem;color:#57606a;margin-top:0.2rem;">'
                f'{s.profile.years_of_experience} 年经验 &nbsp;·&nbsp; '
                f'<code style="font-size:0.75rem;">{s.current_difficulty.value}</code>'
                f'</div></div>',
                unsafe_allow_html=True
            )

            # ── 面试流程进度 ──
            all_phases = ["setup", "profiling", "self_intro", "interview", "hrd_interview", "report"]
            phase_names = ["信息收集", "简历分析", "自我介绍", "技术面试", "HRD面试", "综合报告"]
            cur_idx = all_phases.index(phase) if phase in all_phases else 0
            flow_html = "<div style='margin-bottom:0.8rem;'>"
            for i, (ph, pn) in enumerate(zip(all_phases, phase_names)):
                if i == cur_idx:
                    color, dot = "#0969da", "●"
                elif i < cur_idx:
                    color, dot = "#1a7f37", "✓"
                else:
                    color, dot = "#8b949e", "○"
                flow_html += f"<div style='display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;padding:0.2rem 0;color:{color};font-weight:{'600' if i <= cur_idx else '400'};'><span>{dot}</span><span>{pn}</span></div>"
            flow_html += "</div>"
            st.markdown(flow_html, unsafe_allow_html=True)

            # ── 进度 ──
            completed = s.question_count
            max_q = s.config.max_questions
            pct = completed / max_q if max_q > 0 else 0
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.82rem;color:#57606a;margin-bottom:0.3rem;">'
                f'<span>答题进度</span><span>{completed} / {max_q}</span></div>',
                unsafe_allow_html=True
            )
            st.progress(pct)

            # ── 实时统计（横排） ──
            if completed > 0:
                c1, c2 = st.columns(2)
                with c1: st.metric("平均分", f"{s.average_score:.1f}")
                with c2: st.metric("时长", f"{s.duration_minutes:.0f}min")

            # ── 各方向得分 ──
            if s.scores_by_topic:
                st.markdown('<div style="font-size:0.8rem;font-weight:600;color:#57606a;'
                            'text-transform:uppercase;letter-spacing:0.06em;margin:0.8rem 0 0.4rem;">'
                            '各方向得分</div>', unsafe_allow_html=True)
                for topic, score in s.scores_by_topic.items():
                    color = "#1a7f37" if score >= 7 else "#7d4e00" if score >= 5 else "#cf222e"
                    bar_w = int(score * 10)
                    short = topic.replace("系统设计与架构", "系统设计").replace("与", "/")[:7]
                    st.markdown(
                        f'<div style="margin-bottom:0.55rem;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:0.8rem;color:#57606a;margin-bottom:0.2rem;">'
                        f'<span>{short}</span>'
                        f'<span style="color:{color};font-family:monospace;font-weight:600">'
                        f'{score:.1f}</span></div>'
                        f'<div style="background:#f0f2f5;border-radius:4px;height:5px;">'
                        f'<div style="background:{color};width:{bar_w}%;height:5px;'
                        f'border-radius:4px;transition:width 0.4s ease;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

        st.divider()

        if phase != "setup":
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🔄 重新开始", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                _init_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.routing_log:
            with st.expander("🧠 决策日志", expanded=False):
                icon_map = {"continue": "▶", "deepen": "🔽", "force_topic": "🎯", "wrap_up": "⛔", "finish": "✅"}
                for item in reversed(st.session_state.routing_log[-8:]):
                    icon = icon_map.get(item.get("action", ""), "•")
                    arrow = f' → {item["topic"]}' if item.get("topic") else ""
                    st.markdown(
                        f'<div style="font-size:0.8rem;padding:0.3rem 0;border-bottom:1px solid #f0f2f5;">'
                        f'{icon} <b>{item.get("action", "")}</b>{arrow}<br>'
                        f'<span style="color:#8b949e;font-size:0.75rem;">{item.get("reason", "")}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════════════════════════
# Setup 页
# ══════════════════════════════════════════════════════════════════

def page_setup():
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1.5rem;">'
        '<h1 style="font-size:3rem;font-weight:700;letter-spacing:-0.04em;'
        'background:linear-gradient(135deg,#0969da,#1a7f37);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem;">'
        'AI 面试官</h1>'
        '<p style="color:#57606a;font-size:1.05rem;letter-spacing:0.02em;">'
        'LLM 工程师全栈评估 &nbsp;·&nbsp; 多智能体协作</p>'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()

    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("#### 👤 候选人信息")
        name = st.text_input("姓名 *", placeholder="输入候选人姓名", key="setup_name",
                             on_change=lambda: st.session_state.update({"_name_val": st.session_state.setup_name}))
        # 实时缓存：on_change 不触发时也兜底读一次
        if st.session_state.get("setup_name"):
            st.session_state["_name_val"] = st.session_state.setup_name
        years = st.number_input("工作年限（年）*", min_value=0, max_value=30, value=3,
                                step=1, key="setup_years")
        st.markdown("#### 📋 岗位描述（JD）")
        jd = st.text_area("JD", placeholder="粘贴岗位描述，帮助系统调整考察方向...",
                          height=190, key="setup_jd", label_visibility="collapsed")
    with col_right:
        st.markdown("#### 📄 候选人简历")

        tab1, tab2 = st.tabs(["📝 文本输入", "📎 上传 PDF"])

        resume = ""

        with tab1:
            resume = st.text_area(
                "简历文本",
                placeholder="粘贴候选人简历，系统将分析技能声称并针对性出题...",
                height=330,
                key="setup_resume_text",
                label_visibility="collapsed"
            )

        with tab2:
            uploaded_file = st.file_uploader(
                "上传 PDF 简历",
                type=["pdf"],
                key="setup_resume_pdf",
                label_visibility="collapsed",
                help="支持 PDF 格式，文件大小不超过 10MB"
            )

            if uploaded_file is not None:
                with st.spinner("正在解析 PDF 文件..."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        resume = extract_pdf_text(file_bytes)
                        st.success(f"✅ 成功解析 PDF，共 {len(resume)} 字符")

                        with st.expander("📄 预览提取的文本", expanded=False):
                            st.text_area("预览", resume[:1000] + ("..." if len(resume) > 1000 else ""),
                                         height=200, key="resume_preview", disabled=True)
                    except Exception as e:
                        st.error(f"❌ PDF 解析失败：{e}")
                        resume = ""

    st.divider()

    with st.expander("⚙️ 高级配置", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: max_q = st.slider("最大题目数", 6, 15, 10, key="cfg_max_q")
        with c2: min_q = st.slider("最少题目数", 4, 10, 6, key="cfg_min_q")
        with c3: fu_thr = st.slider("追问分数线", 4.0, 8.0, 6.0, 0.5, key="cfg_fu")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🚀 开始面试", use_container_width=True):
            if not (st.session_state.get("_name_val") or name or "").strip():
                st.error("请输入候选人姓名")
                return
            with st.spinner("正在初始化面试系统..."):
                try:
                    orch = OrchestratorAgent()
                    orch.question_agent
                    st.session_state.orchestrator = orch
                    st.session_state.setup_params = {
                        "name": (st.session_state.get("_name_val") or name or "").strip(), "years": int(years),
                        "jd": jd.strip() or None, "resume": resume.strip() or None,
                        "max_q": max_q, "min_q": min_q, "fu_thr": fu_thr,
                    }
                    st.session_state.phase = "profiling"
                    st.rerun()
                except Exception as e:
                    st.error(f"初始化失败：{e}")


# ══════════════════════════════════════════════════════════════════
# Profiling 页（骨架屏版）
# ══════════════════════════════════════════════════════════════════

def page_profiling():
    params = st.session_state.setup_params
    orch: OrchestratorAgent = st.session_state.orchestrator

    st.markdown("#### 🔍 正在分析候选人信息")
    # 显示骨架屏
    ph = st.empty()
    ph.markdown(
        _skeleton_lines(4, ["100%", "70%", "85%", "55%"]) +
        _skeleton_lines(3, ["80%", "65%", "45%"]),
        unsafe_allow_html=True
    )

    try:
        _run_async(orch._run_initialization_streamlit(
            name=params["name"], years=params["years"],
            jd=params.get("jd") or "", resume=params.get("resume") or "",
        ))
        orch.state.config.max_questions = params["max_q"]
        orch.state.config.min_questions = params["min_q"]
        orch.state.config.follow_up_threshold = params["fu_thr"]

        ph.empty()
        p = orch.state.profile

        # 展示画像
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="report-section-card">'
                f'<div class="report-section-title">👤 候选人画像</div>'
                f'<div style="font-size:0.92rem;line-height:2;">'
                f'姓名：<b>{p.name}</b><br>'
                f'工作年限：<b>{p.years_of_experience} 年</b><br>'
                f'建议难度：<code>{orch.state.current_difficulty.value}</code><br>'
                + (f'简历摘要：{p.resume_summary}<br>' if p.resume_summary else "") +
                f'</div></div>',
                unsafe_allow_html=True
            )
        with c2:
            items = ""
            if p.skills_to_verify:
                items += '<div class="report-section-title">⚠️ 重点验证技能</div>'
                for sk in p.skills_to_verify[:5]:
                    items += f'<div class="report-item"><div class="report-item-dot" style="background:#e3b341"></div>{sk}</div>'
            if p.weak_areas:
                items += '<div class="report-section-title" style="margin-top:0.8rem;">📌 补充考察</div>'
                for a in p.weak_areas[:3]:
                    items += f'<div class="report-item"><div class="report-item-dot" style="background:#cf222e"></div>{a}</div>'
            st.markdown(f'<div class="report-section-card">{items}</div>', unsafe_allow_html=True)

        # 准备第一题
        ctx = orch.state.current_context
        if ctx:
            q = ctx.question
            _add_chat("bot",
                      f"您好，{p.name}！欢迎参加 LLM 工程师技术面试。\n\n"
                      f"本次面试共 **{params['max_q']}** 题，涵盖 Prompt 工程、RAG、Agent 设计、"
                      f"系统架构、代码实现等方向。请尽量详细作答，展示您的理解深度。\n\n准备好了吗？",
                      meta={"type": "intro"}
                      )
            _add_chat("bot",
                      q.text + (f"\n\n💡 提示：{q.hint}" if q.hint else ""),
                      meta={"type": "question", "q_id": q.id, "topic": q.topic.value,
                            "difficulty": q.difficulty.value, "q_num": 1, "is_coding": _is_coding(q)}
                      )
            st.session_state.current_question = q
            st.session_state.current_q_num = 1
            st.session_state.awaiting_answer = True

        st.session_state.phase = "self_intro"
        time.sleep(1.2)
        st.rerun()

    except Exception as e:
        ph.empty()
        st.error(f"画像分析失败：{e}")
        st.exception(e)
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("返回重试"):
            st.session_state.phase = "setup"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Self-Intro 页：候选人自我介绍 + AI提问
# ══════════════════════════════════════════════════════════════════


def page_self_intro():
    orch: OrchestratorAgent = st.session_state.orchestrator
    state = orch.state
    name = state.profile.name

    st.markdown(
        '''<div style="background:linear-gradient(135deg,#f0f7ff,#e8f4fd);border-radius:14px;
        padding:1.5rem 2rem;margin-bottom:1.5rem;border-left:5px solid #0969da;">
        <div style="font-size:1.25rem;font-weight:700;color:#0969da;margin-bottom:0.3rem;">
        🎤 第一轮：自我介绍</div>
        <div style="color:#57606a;font-size:0.9rem;">请候选人进行3-5分钟的自我介绍，面试官将根据内容进行有针对性的提问</div>
        </div>''',
        unsafe_allow_html=True
    )

    # ── 步骤1：收集自我介绍 ──────────────────────────────────────
    if not st.session_state.self_intro_text:
        st.markdown(f"**{name}，请进行自我介绍：**")
        intro = st.text_area(
            "自我介绍",
            placeholder="请介绍您的教育背景、工作经历、技术专长、以及为什么对这个岗位感兴趣...",
            height=200, key="intro_input", label_visibility="collapsed"
        )
        if st.button("✅ 完成自我介绍，开始提问", use_container_width=False):
            if not intro.strip():
                st.warning("请输入自我介绍内容")
                return
            intro = intro.strip()
            st.session_state.self_intro_text = intro
            state.self_intro_text = intro

            # 创建 Agent，生成第一个追问
            agent = SelfIntroAgent()
            st.session_state.self_intro_agent = agent
            with st.spinner("面试官正在阅读您的自我介绍..."):
                reply = agent.first_question(name, state.profile.years_of_experience, intro)
            _handle_self_intro_reply(state, reply)
            st.rerun()
        return

    # ── 步骤2：追问对话 ──────────────────────────────────────────
    # 渲染自我介绍原文
    st.markdown("**候选人自我介绍：**")
    st.markdown(
        f'''<div class="chat-user">
        <div style="font-size:0.75rem;font-weight:600;color:#0550ae;text-transform:uppercase;
        letter-spacing:0.06em;margin-bottom:0.35rem;">候选人</div>
        <div>{st.session_state.self_intro_text.replace(chr(10), "<br>")}</div>
        </div>''', unsafe_allow_html=True
    )

    # 渲染问答历史
    for qa in st.session_state.self_intro_chat:
        if qa["role"] == "question":
            st.markdown(
                f'''<div class="chat-bot">
                <div style="font-size:0.76rem;font-weight:600;color:#0969da;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:0.4rem;">🎙️ 面试官提问</div>
                <div>{qa["text"].replace(chr(10), "<br>")}</div>
                </div>''', unsafe_allow_html=True
            )
        elif qa["role"] == "answer":
            st.markdown(
                f'''<div class="chat-user">
                <div style="font-size:0.75rem;font-weight:600;color:#0550ae;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:0.35rem;">候选人</div>
                <div>{qa["text"].replace(chr(10), "<br>")}</div>
                </div>''', unsafe_allow_html=True
            )

    st.divider()

    # ── 判断当前状态 ──────────────────────────────────────────────
    if st.session_state.self_intro_qa_done:
        # 展示得分卡
        score = state.self_intro_score or 0.0
        evaluation = state.self_intro_evaluation or ""
        st.markdown(
            f'''<div class="eval-card">
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
            {_score_emoji(score)} 自我介绍得分：{_score_badge(score)}
            </div>
            <div style="color:#1a7f37;font-size:0.9rem;">{evaluation}</div>
            </div>''', unsafe_allow_html=True
        )
        st.success("✅ 自我介绍环节完成！")
        if st.button("▶️ 进入技术面试", use_container_width=False):
            st.session_state.phase = "interview"
            st.rerun()
    else:
        # 等待候选人回答当前追问
        answer = st.text_area(
            "回答", placeholder="请详细回答面试官的问题...",
            height=120, key=f"intro_answer_{len(state.self_intro_qa)}", label_visibility="collapsed"
        )
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            submit = st.button("✅ 提交回答", use_container_width=True, key=f"intro_submit_{len(state.self_intro_qa)}")
        with c2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            skip = st.button("跳过", use_container_width=True, key=f"intro_skip_{len(state.self_intro_qa)}")
            st.markdown('</div>', unsafe_allow_html=True)

        if (submit or skip) and not st.session_state.get("self_intro_processing"):
            ans_text = answer.strip() if submit else "（跳过）"
            if submit and not ans_text:
                st.warning("请输入回答内容")
                return

            st.session_state.self_intro_processing = True

            # 找到当前未回答的最后一个问题
            pending_q = ""
            for item in reversed(st.session_state.self_intro_chat):
                if item["role"] == "question":
                    pending_q = item["text"]
                    break

            # 记录回答
            st.session_state.self_intro_chat.append({"role": "answer", "text": ans_text})
            state.self_intro_qa.append({"question": pending_q, "answer": ans_text})

            # 调用 Agent 决定下一步
            agent = st.session_state.get("self_intro_agent") or SelfIntroAgent()
            st.session_state.self_intro_agent = agent
            with st.spinner("面试官正在思考..."):
                reply = agent.next_reply(
                    name,
                    state.profile.years_of_experience,
                    st.session_state.self_intro_text,
                    state.self_intro_qa,
                )
            _handle_self_intro_reply(state, reply)
            st.session_state.self_intro_processing = False
            st.rerun()


def _handle_self_intro_reply(state, reply):
    """处理 SelfIntroAgent 的回复，更新 session_state 和 state。"""
    if reply.done:
        state.self_intro_score = reply.score
        state.self_intro_evaluation = reply.evaluation
        st.session_state.self_intro_qa_done = True
    else:
        st.session_state.self_intro_chat.append({"role": "question", "text": reply.text})


# ══════════════════════════════════════════════════════════════════
# HRD Interview 页 - 修复版本
# ══════════════════════════════════════════════════════════════════


def page_hrd_interview():
    orch: OrchestratorAgent = st.session_state.orchestrator
    state = orch.state
    name = state.profile.name
    years = state.profile.years_of_experience

    # 确保必要的状态存在
    if "hrd_processing" not in st.session_state:
        st.session_state.hrd_processing = False
    if "hrd_submitted" not in st.session_state:
        st.session_state.hrd_submitted = False

    st.markdown(
        '''<div style="background:linear-gradient(135deg,#fff8e1,#fef3cd);border-radius:14px;
        padding:1.5rem 2rem;margin-bottom:1.5rem;border-left:5px solid #e3b341;">
        <div style="font-size:1.25rem;font-weight:700;color:#7d4e00;margin-bottom:0.3rem;">
        👔 第三轮：HRD 综合面试</div>
        <div style="color:#57606a;font-size:0.9rem;">最后一轮面试，HRD将从职业规划、团队协作、薪资期望等维度进行综合考察</div>
        </div>''',
        unsafe_allow_html=True
    )

    hrd_chat = st.session_state.hrd_chat  # list of {"role": "question"/"answer", "text": str}
    hrd_done = st.session_state.hrd_qa_done

    # ── 初始化：第一次进入，创建 Agent 并生成开场 ─────────────────
    if not hrd_chat and not hrd_done:
        tech_report = st.session_state.get("report")
        tech_scores = tech_report.dimension_scores if tech_report else state.scores_by_topic
        tech_summary = tech_report.summary if tech_report else ""
        self_intro_eval = state.self_intro_evaluation or ""

        agent = HRDAgent(
            name=name, years=years,
            self_intro_eval=self_intro_eval,
            tech_scores=tech_scores,
            tech_summary=tech_summary,
        )
        st.session_state.hrd_agent = agent
        state.hrd_qa = []

        with st.spinner("HRD 正在准备面试问题..."):
            reply = agent.opening()
        _handle_hrd_reply(state, reply)
        st.rerun()
        return

    # ── 渲染对话历史 ──────────────────────────────────────────────
    for item in hrd_chat:
        if item["role"] == "question":
            st.markdown(
                f'''<div class="chat-bot">
                <div style="font-size:0.76rem;font-weight:600;color:#7d4e00;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:0.4rem;">👔 HRD 面试官</div>
                <div>{item["text"].replace(chr(10), "<br>")}</div>
                </div>''', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''<div class="chat-user">
                <div style="font-size:0.75rem;font-weight:600;color:#0550ae;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:0.35rem;">候选人</div>
                <div>{item["text"].replace(chr(10), "<br>")}</div>
                </div>''', unsafe_allow_html=True
            )

    st.divider()

    # ── 已完成 ────────────────────────────────────────────────────
    if hrd_done:
        score = state.hrd_score or 0.0
        evaluation = state.hrd_evaluation or ""
        hire_sug = st.session_state.get("hrd_hire_suggestion", "存疑")
        st.markdown(
            f'''<div class="eval-card">
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
            {_score_emoji(score)} HRD面试得分：{_score_badge(score)}
            <span style="color:#57606a;font-size:0.82rem;">初步建议：{hire_sug}</span>
            </div>
            <div style="color:#1a7f37;font-size:0.9rem;">{evaluation}</div>
            </div>''', unsafe_allow_html=True
        )
        st.success("✅ HRD面试完成！")
        if st.button("📊 生成最终综合报告", use_container_width=False):
            tech_report = st.session_state.get("report")
            tech_score = tech_report.total_score if tech_report else state.average_score
            tech_dim = tech_report.dimension_scores if tech_report else state.scores_by_topic
            tech_summary = tech_report.summary if tech_report else ""

            with st.spinner("AI 正在综合三轮评估结果..."):
                final_agent = FinalReportAgent()
                final_report = final_agent.generate(
                    name=name, years=years,
                    self_intro_score=state.self_intro_score or 7.0,
                    self_intro_eval=state.self_intro_evaluation or "",
                    tech_score=tech_score,
                    tech_dim_scores=tech_dim,
                    tech_summary=tech_summary,
                    hrd_score=state.hrd_score or 7.0,
                    hrd_eval=state.hrd_evaluation or "",
                    hrd_suggestion=hire_sug,
                )
            st.session_state.final_report = final_report
            st.session_state.phase = "report"
            st.rerun()
        return

    # ── 等待候选人回答 ────────────────────────────────────────
    # 获取最后一个问题作为当前问题
    current_question = ""
    for item in reversed(hrd_chat):
        if item["role"] == "question":
            current_question = item["text"]
            break

    st.markdown(
        f'<div class="warn-box" style="margin-bottom:1rem;">💬 <strong>当前问题：</strong><br>{current_question}</div>',
        unsafe_allow_html=True)

    answer = st.text_area(
        "您的回答",
        placeholder="请详细回答HRD的问题...",
        height=120,
        key=f"hrd_answer_{len(state.hrd_qa)}",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit = st.button("✅ 提交回答", use_container_width=True, key=f"hrd_submit_{len(state.hrd_qa)}")
    with col2:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        skip = st.button("跳过此题", use_container_width=True, key=f"hrd_skip_{len(state.hrd_qa)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 处理提交
    if submit and not st.session_state.hrd_processing and not st.session_state.hrd_submitted:
        ans_text = answer.strip()
        if not ans_text:
            st.warning("请输入回答内容")
            return

        st.session_state.hrd_submitted = True
        st.session_state.hrd_processing = True

        # 记录回答
        hrd_chat.append({"role": "answer", "text": ans_text})
        state.hrd_qa.append({"question": current_question, "answer": ans_text})

        # 调用 Agent 决定下一步
        agent = st.session_state.get("hrd_agent")
        if not agent:
            tech_report = st.session_state.get("report")
            agent = HRDAgent(
                name=name, years=years,
                self_intro_eval=state.self_intro_evaluation or "",
                tech_scores=tech_report.dimension_scores if tech_report else state.scores_by_topic,
                tech_summary=tech_report.summary if tech_report else "",
            )
            st.session_state.hrd_agent = agent

        with st.spinner("HRD 正在思考下一个问题..."):
            reply = agent.next_reply(state.hrd_qa)

        # 处理回复
        if reply.done:
            state.hrd_score = reply.score
            state.hrd_evaluation = reply.evaluation
            st.session_state.hrd_qa_done = True
            st.session_state.hrd_hire_suggestion = reply.hire_suggestion
        else:
            hrd_chat.append({"role": "question", "text": reply.text})

        # 重置处理标志
        st.session_state.hrd_processing = False
        st.session_state.hrd_submitted = False
        st.rerun()

    # 处理跳过
    elif skip and not st.session_state.hrd_processing and not st.session_state.hrd_submitted:
        st.session_state.hrd_submitted = True
        st.session_state.hrd_processing = True

        # 记录跳过
        hrd_chat.append({"role": "answer", "text": "（跳过）"})
        state.hrd_qa.append({"question": current_question, "answer": "（跳过）"})

        # 调用 Agent 决定下一步
        agent = st.session_state.get("hrd_agent")
        if not agent:
            tech_report = st.session_state.get("report")
            agent = HRDAgent(
                name=name, years=years,
                self_intro_eval=state.self_intro_evaluation or "",
                tech_scores=tech_report.dimension_scores if tech_report else state.scores_by_topic,
                tech_summary=tech_report.summary if tech_report else "",
            )
            st.session_state.hrd_agent = agent

        with st.spinner("HRD 正在思考下一个问题..."):
            reply = agent.next_reply(state.hrd_qa)

        # 处理回复
        if reply.done:
            state.hrd_score = reply.score
            state.hrd_evaluation = reply.evaluation
            st.session_state.hrd_qa_done = True
            st.session_state.hrd_hire_suggestion = reply.hire_suggestion
        else:
            hrd_chat.append({"role": "question", "text": reply.text})

        # 重置处理标志
        st.session_state.hrd_processing = False
        st.session_state.hrd_submitted = False
        st.rerun()


def _handle_hrd_reply(state, reply):
    """处理 HRDAgent 的回复。"""
    if reply.done:
        state.hrd_score = reply.score
        state.hrd_evaluation = reply.evaluation
        st.session_state.hrd_qa_done = True
        st.session_state.hrd_hire_suggestion = reply.hire_suggestion
    else:
        st.session_state.hrd_chat.append({"role": "question", "text": reply.text})


# ══════════════════════════════════════════════════════════════════
# Interview 页
# ══════════════════════════════════════════════════════════════════


def page_interview():
    orch: OrchestratorAgent = st.session_state.orchestrator
    state = orch.state

    # 顶部进度条
    c_title, c_meta = st.columns([3, 1])
    with c_title:
        st.markdown("#### 💬 面试进行中")
    with c_meta:
        if state.question_count > 0:
            st.markdown(
                f'<div style="text-align:right;color:#57606a;font-size:0.85rem;padding-top:0.5rem;">'
                f'第 {state.question_count + (1 if st.session_state.awaiting_answer else 0)} 题 / '
                f'{state.config.max_questions} 题</div>',
                unsafe_allow_html=True
            )

    # 聊天历史
    for msg in st.session_state.chat_history:
        _render_chat_msg(msg)

    st.divider()

    if st.session_state.awaiting_answer:
        _render_answer_input(orch, state)
    elif st.session_state.awaiting_followup:
        _render_followup_input(orch, state)
    elif not state.is_finished:
        _advance_interview(orch, state)


def _render_chat_msg(msg: dict):
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})
    mtype = meta.get("type", "")

    if role == "bot":
        if mtype == "question":
            topic = meta.get("topic", "")
            diff = meta.get("difficulty", "")
            q_num = meta.get("q_num", "")
            is_coding = meta.get("is_coding", False)
            code_cls = " topic-tag-code" if is_coding else ""
            coding_badge = '<span class="topic-tag topic-tag-code">💻 代码题</span>' if is_coding else ""
            st.markdown(
                f'<div class="question-card">'
                f'<div style="margin-bottom:0.9rem;">'
                f'<span class="topic-tag">第 {q_num} 题</span>'
                f'<span class="topic-tag{code_cls}">{topic}</span>'
                f'<span class="topic-tag">{diff}</span>'
                f'{coding_badge}</div>'
                f'<div style="color:#1f2328;line-height:1.75;font-size:0.97rem;">'
                f'{content.replace(chr(10), "<br>")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif mtype == "followup":
            st.markdown(
                f'<div class="chat-bot">'
                f'<div style="font-size:0.76rem;font-weight:600;color:#7d4e00;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem;">🔎 追问</div>'
                f'<div>{content}</div></div>',
                unsafe_allow_html=True
            )
        elif mtype == "eval":
            score = meta.get("score", 0)
            depth = meta.get("depth", "")
            feedback = meta.get("feedback", "")
            st.markdown(
                f'<div class="eval-card">'
                f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">'
                f'{_score_emoji(score)} 本题评分：{_score_badge(score)}'
                f'<span style="color:#57606a;font-size:0.82rem;">深度：{depth}</span>'
                f'</div>'
                f'<div style="color:#1a7f37;font-size:0.9rem;">{feedback}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif mtype == "stream_feedback":
            st.markdown(f'<div class="stream-box">{content}</div>', unsafe_allow_html=True)
        elif mtype == "transition":
            st.markdown(
                f'<div style="color:#57606a;font-size:0.88rem;font-style:italic;'
                f'padding:0.4rem 0;">{content}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bot">{content.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
    else:  # user
        st.markdown(
            f'<div class="chat-user">'
            f'<div style="font-size:0.75rem;font-weight:600;color:#0550ae;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.35rem;">候选人</div>'
            f'<div>{content.replace(chr(10), "<br>")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def _render_answer_input(orch, state):
    current_q = st.session_state.current_question
    is_code = _is_coding(current_q)

    if is_code:
        st.markdown("#### 💻 代码实现")
        st.markdown('<div class="info-box">代码题：在下方编辑器实现，可点击 ▶ 运行验证，确认后提交。</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        code, output = _render_code_editor(current_q.text if current_q else "")

        st.markdown("**补充说明**（设计思路、trade-off，可选）")
        extra = st.text_area("补充", placeholder="说明你的设计决策...",
                             height=80, key="code_extra", label_visibility="collapsed")

        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            submit = st.button("✅ 提交回答", use_container_width=True, key="submit_code")
        with c2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            skip = st.button("跳过", use_container_width=True, key="skip_code")
            st.markdown('</div>', unsafe_allow_html=True)

        if submit or skip:
            if submit:
                parts = [f"```python\n{code}\n```"]
                if output:
                    s, o, e = output.get("status", ""), output.get("stdout", ""), output.get("stderr", "")
                    parts.append(f"\n[运行结果 - {s}]\n" + (f"输出：{o}" if o else "") + (f"\n错误：{e}" if e else ""))
                if extra.strip(): parts.append(f"\n设计说明：{extra.strip()}")
                final = "\n".join(parts)
            else:
                final = ""
            _submit_answer(orch, state, final, is_coding=True)

    else:
        st.markdown("#### ✍️ 请输入您的回答")
        answer = st.text_area("回答", placeholder="详细阐述你的理解，包括原理、实践经验和 trade-off...",
                              height=150, key=f"answer_input_{st.session_state.current_q_num}",
                              label_visibility="collapsed")
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            submit = st.button("✅ 提交回答", use_container_width=True)
        with c2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            skip = st.button("跳过此题", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if (submit or skip) and not st.session_state.get("tech_processing"):
            final = answer.strip() if submit else ""
            if submit and not final:
                st.warning("请输入回答内容，或点击「跳过此题」")
                return
            st.session_state.tech_processing = True
            _submit_answer(orch, state, final)


def _submit_answer(orch, state, final_answer: str, is_coding=False):
    _add_chat("user", final_answer if final_answer else "（跳过）",
              meta={"type": "answer", "is_coding": is_coding})

    if state.current_context:
        state.current_context.answer = final_answer

    st.session_state.awaiting_answer = False
    st.session_state.tech_processing = False
    if is_coding:
        st.session_state.code_output = None
        st.session_state.code_content = ""

    # 骨架屏占位
    ph = st.empty()
    ph.markdown(_thinking_indicator("AI 正在评估"), unsafe_allow_html=True)

    try:
        followup_holder = {"text": None}

        async def capture_followup(text):
            followup_holder["text"] = text
            return ""

        _run_async(orch.evaluator_agent.run(state, capture_followup))
    except Exception as ex:
        ph.empty()
        st.error(f"评估出错：{ex}")
        st.rerun()
        return

    ph.empty()

    # 流式反馈
    if state.completed_records and final_answer:
        last = state.completed_records[-1]
        e, q = last.evaluation, last.question
        st.markdown(
            '<div style="font-size:0.78rem;font-weight:600;color:#57606a;'
            'text-transform:uppercase;letter-spacing:0.06em;margin:0.6rem 0 0.25rem;">'
            '💬 面试官反馈</div>',
            unsafe_allow_html=True
        )
        streamed = _stream_eval_feedback(q.text, final_answer, q.scoring_rubric)
        if streamed:
            _add_chat("bot", streamed, meta={"type": "stream_feedback"})
        _add_chat("bot", "", meta={
            "type": "eval", "score": e.effective_score,
            "depth": e.depth_label, "feedback": e.brief_feedback,
        })
    elif state.completed_records and not final_answer:
        last = state.completed_records[-1]
        _add_chat("bot", "", meta={
            "type": "eval", "score": last.evaluation.effective_score,
            "depth": last.evaluation.depth_label, "feedback": "（已跳过，不计分）",
        })

    if followup_holder["text"]:
        st.session_state.current_followup_text = followup_holder["text"]
        st.session_state.awaiting_followup = True
        _add_chat("bot", followup_holder["text"], meta={"type": "followup"})

    st.rerun()


def _render_followup_input(orch, state):
    st.markdown("#### 🔎 追问")
    st.markdown(f'<div class="warn-box">💬 {st.session_state.current_followup_text}</div>',
                unsafe_allow_html=True)
    st.markdown("")
    fu_answer = st.text_area("追问回答", placeholder="针对追问详细作答...",
                             height=120, key=f"fu_input_{st.session_state.current_q_num}", label_visibility="collapsed")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        submit = st.button("✅ 提交回答", use_container_width=True, key=f"submit_fu_{st.session_state.current_q_num}")
    with c2:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        skip = st.button("跳过", use_container_width=True, key=f"skip_fu_{st.session_state.current_q_num}")
        st.markdown('</div>', unsafe_allow_html=True)

    if submit or skip:
        final = fu_answer.strip() if submit else ""
        _add_chat("user", final if final else "（跳过）", meta={"type": "followup_answer"})
        if state.completed_records:
            ev = state.completed_records[-1].evaluation
            ev.follow_up_answer = final
            if not final: ev.follow_up_score = ev.score * 0.5
        st.session_state.awaiting_followup = False
        st.session_state.tech_processing = False
        st.session_state.current_followup_text = ""
        st.rerun()


def _finish_interview(orch, state):
    """技术面试结束：生成技术报告，然后进入HRD面试。"""
    state.transition_phase(InterviewPhase.WRAPPING_UP, AgentRole.ORCHESTRATOR)
    ph = st.empty()
    ph.markdown(_thinking_indicator("正在生成技术面试评估报告"), unsafe_allow_html=True)
    report = orch.reporter_agent.generate(state)
    ph.empty()
    state.transition_phase(InterviewPhase.HRD_INTERVIEW, AgentRole.ORCHESTRATOR)
    st.session_state.report = report
    st.session_state.report_summary_cache = ""
    # 进入HRD面试，而非直接跳到报告页
    st.session_state.hrd_chat = []
    st.session_state.hrd_qa_done = False
    st.session_state.phase = "hrd_interview"
    st.rerun()


def _advance_interview(orch, state):
    ph = st.empty()
    ph.markdown(_thinking_indicator("Orchestrator 正在决策"), unsafe_allow_html=True)

    try:
        pending = state.consume_messages()
        orch.reflector.reflect()
        orch._update_candidate_model()
        decision = _run_async(orch._decide_routing(pending))

        st.session_state.routing_log.append({
            "action": decision.action, "topic": decision.topic, "reason": decision.reason
        })

        if decision.action in ("wrap_up", "finish"):
            ph.empty()
            if decision.action == "wrap_up":
                _add_chat("bot", f"感谢完成面试。（{decision.reason}）", meta={"type": "system"})
            _finish_interview(orch, state)
            return

        if state.question_count >= state.config.max_questions:
            ph.empty()
            _finish_interview(orch, state)
            return

        force_topic = decision.topic if decision.action == "force_topic" else None
        deepen = decision.action == "deepen"
        ok = orch.question_agent.select(
            state, force_topic=force_topic, deepen=deepen,
            deepen_reason=decision.reason if deepen else "",
        )

        ph.empty()

        if not ok:
            _finish_interview(orch, state)
            return

        ctx = state.current_context
        q = ctx.question
        q_num = state.question_count + 1
        st.session_state.current_question = q
        st.session_state.current_q_num = q_num
        st.session_state.awaiting_answer = True
        st.session_state.code_output = None
        st.session_state.code_content = ""

        if decision.action in ("deepen", "force_topic") and decision.reason:
            _add_chat("bot", decision.reason, meta={"type": "transition"})

        _add_chat("bot",
                  q.text + (f"\n\n💡 提示：{q.hint}" if q.hint else ""),
                  meta={"type": "question", "q_id": q.id, "topic": q.topic.value,
                        "difficulty": q.difficulty.value, "q_num": q_num, "is_coding": _is_coding(q)}
                  )

    except Exception as ex:
        ph.empty()
        st.error(f"推进面试出错：{ex}")
        st.exception(ex)

    st.rerun()


# ══════════════════════════════════════════════════════════════════
# Report 页（重排版）
# ══════════════════════════════════════════════════════════════════

def page_report():
    from core.models import InterviewReport
    report: InterviewReport = st.session_state.report
    orch: OrchestratorAgent  = st.session_state.orchestrator
    state = orch.state
    final_report_obj = st.session_state.get("final_report")  # FinalReport dataclass

    rec_cfg = {
        "强烈推荐": ("#dafbe1", "#1a7f37", "🌟"),
        "推荐":     ("#e6ffed", "#2da44e", "✅"),
        "存疑":     ("#fff8c5", "#7d4e00", "⚠️"),
        "不推荐":   ("#ffebe9", "#cf222e", "❌"),
    }

    # 优先使用最终综合建议
    final_rec = final_report_obj.final_recommendation if final_report_obj else report.hire_recommendation
    final_score = final_report_obj.final_score if final_report_obj else report.total_score
    rec = final_rec
    rec_bg, rec_fg, rec_icon = rec_cfg.get(rec, ("#f6f8fa", "#57606a", "•"))

    # ── 三轮面试总览 ──────────────────────────────────────────────
    st.markdown(
        '''<div style="background:linear-gradient(135deg,#f0f7ff,#e8f4fd);border-radius:14px;
        padding:1.2rem 2rem;margin-bottom:1.2rem;">
        <div style="font-size:1.1rem;font-weight:700;color:#1f2328;margin-bottom:1rem;">
        📊 三轮面试综合评估</div>''' +
        f'''<div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:160px;background:#fff;border-radius:10px;padding:1rem;
        border-left:4px solid #0969da;text-align:center;">
        <div style="font-size:0.75rem;color:#57606a;font-weight:600;margin-bottom:0.3rem;">🎤 自我介绍</div>
        <div style="font-size:2rem;font-weight:700;color:#0969da;">{state.self_intro_score or 0:.1f}</div>
        <div style="font-size:0.7rem;color:#8b949e;">/ 10</div></div>
        <div style="flex:1;min-width:160px;background:#fff;border-radius:10px;padding:1rem;
        border-left:4px solid #1a7f37;text-align:center;">
        <div style="font-size:0.75rem;color:#57606a;font-weight:600;margin-bottom:0.3rem;">💻 技术面试</div>
        <div style="font-size:2rem;font-weight:700;color:#1a7f37;">{report.total_score:.1f}</div>
        <div style="font-size:0.7rem;color:#8b949e;">/ 10</div></div>
        <div style="flex:1;min-width:160px;background:#fff;border-radius:10px;padding:1rem;
        border-left:4px solid #e3b341;text-align:center;">
        <div style="font-size:0.75rem;color:#57606a;font-weight:600;margin-bottom:0.3rem;">👔 HRD面试</div>
        <div style="font-size:2rem;font-weight:700;color:#7d4e00;">{state.hrd_score or 0:.1f}</div>
        <div style="font-size:0.7rem;color:#8b949e;">/ 10</div></div>
        <div style="flex:1;min-width:160px;background:{rec_bg};border-radius:10px;padding:1rem;
        border-left:4px solid {rec_fg};text-align:center;">
        <div style="font-size:0.75rem;color:#57606a;font-weight:600;margin-bottom:0.3rem;">🏆 综合得分</div>
        <div style="font-size:2rem;font-weight:700;color:{rec_fg};">{final_score:.1f}</div>
        <div style="font-size:0.8rem;font-weight:700;color:{rec_fg};">{rec_icon} {rec}</div></div>
        </div>''' +
        '</div>',
        unsafe_allow_html=True
    )

    # 综合总评
    if final_report_obj and final_report_obj.final_summary:
        st.markdown(
            f'''<div class="stream-box" style="margin-bottom:1rem;">
            <div style="font-size:0.8rem;font-weight:600;color:#57606a;margin-bottom:0.4rem;">📝 综合评价</div>
            {final_report_obj.final_summary}
            </div>''', unsafe_allow_html=True
        )

    # 核心优势 & 主要顾虑
    if final_report_obj and (final_report_obj.key_strengths or final_report_obj.key_concerns):
        col_s, col_c = st.columns(2)
        with col_s:
            strengths_html = "".join(
                f'<div class="report-item"><div class="report-item-dot" style="background:#2da44e;margin-top:7px;"></div><div>{s}</div></div>'
                for s in (final_report_obj.key_strengths if final_report_obj else [])
            )
            st.markdown(f'<div class="report-section-card"><div class="report-section-title">✅ 核心优势</div>{strengths_html}</div>', unsafe_allow_html=True)
        with col_c:
            concerns_html = "".join(
                f'<div class="report-item"><div class="report-item-dot" style="background:#cf222e;margin-top:7px;"></div><div>{c}</div></div>'
                for c in (final_report_obj.key_concerns if final_report_obj else [])
            )
            st.markdown(f'<div class="report-section-card"><div class="report-section-title">⚠️ 主要顾虑</div>{concerns_html}</div>', unsafe_allow_html=True)

    if final_report_obj and final_report_obj.onboarding_suggestions:
        sug_html = "".join(
            f'<div class="report-item"><div class="report-item-dot" style="background:#8250df;margin-top:7px;"></div><div>{s}</div></div>'
            for s in (final_report_obj.onboarding_suggestions if final_report_obj else [])
        )
        st.markdown(f'<div class="report-section-card"><div class="report-section-title">💡 入职建议</div>{sug_html}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📋 技术面试详细报告")

    # ── Hero 区（分数 + 建议 + 时间） ────────────────────────────
    jd_block = ""
    if report.jd_match_score is not None:
        jd_block = (
            f'<div style="text-align:center;">'
            f'<div class="report-score-big" style="color:#8250df;">{report.jd_match_score:.1f}</div>'
            f'<div class="report-label">JD 匹配度 / 10</div></div>'
        )

    st.markdown(
        f'<div class="report-hero">'
        f'<div style="font-size:0.8rem;font-weight:600;color:#57606a;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">面试评估报告</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#1f2328;margin-bottom:0.25rem;">'
        f'{report.candidate_name}</div>'
        f'<div style="font-size:0.85rem;color:#57606a;margin-bottom:1.6rem;">'
        f'{report.candidate_years} 年工作经验 &nbsp;·&nbsp; {state.duration_minutes:.0f} 分钟 &nbsp;·&nbsp; '
        f'{len(report.records)} 题</div>'
        f'<div style="display:flex;justify-content:center;align-items:flex-end;gap:3rem;flex-wrap:wrap;">'
        f'<div style="text-align:center;">'
        f'<div class="report-score-big" style="color:#0969da;">{report.total_score:.1f}</div>'
        f'<div class="report-label">综合得分 / 10</div></div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:1.6rem;font-weight:700;padding:0.5rem 1.4rem;border-radius:10px;'
        f'background:{rec_bg};color:{rec_fg};">{rec_icon} {rec}</div>'
        f'<div class="report-label" style="margin-top:0.4rem;">录用建议</div></div>'
        f'{jd_block}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # ── 两列主体布局 ──────────────────────────────────────────────
    col_main, col_side = st.columns([3, 2], gap="large")

    with col_main:

        # 总评（流式，带缓存避免重复生成）
        st.markdown(
            '<div class="report-section-title" style="font-size:0.85rem;margin-bottom:0.5rem;">'
            '📝 总体评价</div>',
            unsafe_allow_html=True
        )
        if st.session_state.report_summary_cache:
            st.markdown(
                f'<div class="stream-box">{st.session_state.report_summary_cache}</div>',
                unsafe_allow_html=True
            )
        elif report.summary:
            st.session_state.report_summary_cache = report.summary
            st.markdown(
                f'<div class="stream-box">{report.summary}</div>',
                unsafe_allow_html=True
            )
        else:
            score_detail = "\n".join(
                f"Q{i+1}[{r.question.topic.value}] {r.evaluation.effective_score:.1f}分 "
                f"{r.evaluation.depth_label} {r.evaluation.detailed_notes[:60]}"
                for i, r in enumerate(report.records)
            )
            summary = _stream_report_summary(
                report.candidate_name, report.candidate_years,
                score_detail, report.total_score, report.dimension_scores
            )
            st.session_state.report_summary_cache = summary

        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

        # 雷达图 - 修复版本
        st.markdown(
            '<div class="report-section-title" style="font-size:0.85rem;margin-bottom:0.3rem;">'
            '📊 各方向得分</div>',
            unsafe_allow_html=True
        )
        if report.dimension_scores:
            try:
                import plotly.graph_objects as go
                topics = list(report.dimension_scores.keys())
                scores = list(report.dimension_scores.values())
                fig = go.Figure(go.Scatterpolar(
                    r=scores + [scores[0]], theta=topics + [topics[0]],
                    fill='toself', fillcolor='rgba(9,105,218,0.1)',
                    line=dict(color='#0969da', width=2),
                    marker=dict(size=5, color='#0969da'),
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,10],
                                        gridcolor='#d0d7de', tickfont=dict(color='#57606a', size=9)),
                        angularaxis=dict(gridcolor='#e8ecf0', tickfont=dict(color='#1f2328', size=10)),
                        bgcolor='#ffffff',  # 使用白色而不是 transparent
                    ),
                    showlegend=False,
                    paper_bgcolor='#ffffff',  # 使用白色而不是 transparent
                    plot_bgcolor='#ffffff',   # 使用白色而不是 transparent
                    margin=dict(l=50, r=50, t=30, b=30), height=320,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                cols = st.columns(len(report.dimension_scores))
                for i, (t, s) in enumerate(report.dimension_scores.items()):
                    with cols[i]: st.metric(t, f"{s:.1f}")

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        # 答题详情
        st.markdown(
            '<div class="report-section-title" style="font-size:0.85rem;margin-bottom:0.5rem;">'
            '📋 答题详情</div>',
            unsafe_allow_html=True
        )
        for i, record in enumerate(report.records, 1):
            q  = record.question
            e  = record.evaluation
            sc = e.effective_score
            is_code = q.topic == Topic.CODING
            with st.expander(
                f"{'💻 ' if is_code else ''}Q{i} [{q.topic.value}]"
                f"　{_score_emoji(sc)} {sc:.1f} 分　{e.depth_label}",
                expanded=False
            ):
                st.markdown(f"**题目：** {q.text}")
                st.divider()
                if is_code and "```" in e.answer:
                    st.markdown(e.answer)
                else:
                    st.markdown(f"**候选人回答：**\n\n{e.answer}")
                if e.follow_up:
                    st.markdown(f"**追问：** {e.follow_up}")
                    if e.follow_up_answer:
                        st.markdown(f"**追问回答：** {e.follow_up_answer}")
                st.divider()
                ca, cb = st.columns(2)
                with ca:
                    if e.keyword_hits:
                        st.markdown("**✅ 命中关键词**")
                        st.markdown("、".join(e.keyword_hits))
                with cb:
                    if e.keyword_misses:
                        st.markdown("**❌ 缺失关键词**")
                        st.markdown("、".join(e.keyword_misses))
                if e.detailed_notes:
                    st.markdown(f"**评估：** {e.detailed_notes}")

    with col_side:

        # 优势
        strength_items = "".join(
            f'<div class="report-item">'
            f'<div class="report-item-dot" style="background:#2da44e;margin-top:7px;"></div>'
            f'<div style="color:#1f2328;">{s}</div></div>'
            for s in report.strengths
        ) or '<div style="color:#8b949e;font-size:0.88rem;">暂无</div>'

        st.markdown(
            f'<div class="report-section-card">'
            f'<div class="report-section-title">✅ 优势亮点</div>'
            f'{strength_items}</div>',
            unsafe_allow_html=True
        )

        # 风险
        risk_items = "".join(
            f'<div class="report-item">'
            f'<div class="report-item-dot" style="background:#cf222e;margin-top:7px;"></div>'
            f'<div style="color:#1f2328;">{r}</div></div>'
            for r in report.risks
        ) or '<div style="color:#8b949e;font-size:0.88rem;">暂无</div>'

        st.markdown(
            f'<div class="report-section-card">'
            f'<div class="report-section-title">⚠️ 风险点</div>'
            f'{risk_items}</div>',
            unsafe_allow_html=True
        )

        # 追问建议
        next_items = "".join(
            f'<div class="report-item">'
            f'<div class="report-item-dot" style="background:#e3b341;margin-top:7px;"></div>'
            f'<div style="color:#1f2328;">{q}</div></div>'
            for q in report.next_round_questions
        ) or '<div style="color:#8b949e;font-size:0.88rem;">暂无</div>'

        st.markdown(
            f'<div class="report-section-card">'
            f'<div class="report-section-title">❓ 下轮追问建议</div>'
            f'{next_items}</div>',
            unsafe_allow_html=True
        )

        # 导出操作
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        report_json = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ 导出完整报告 JSON",
            data=report_json,
            file_name=f"interview_{report.candidate_name}_{report.session_id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("🔄 开始新面试", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 主路由
# ══════════════════════════════════════════════════════════════════

render_sidebar()
phase = st.session_state.phase

if phase == "setup":
    page_setup()
elif phase == "profiling":
    page_profiling()
elif phase == "self_intro":
    page_self_intro()
elif phase == "interview":
    page_interview()
elif phase == "hrd_interview":
    page_hrd_interview()
elif phase == "report":
    page_report()
else:
    st.error(f"未知阶段：{phase}")