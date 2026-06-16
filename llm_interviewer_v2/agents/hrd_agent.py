"""
HRDAgent — 负责最终 HRD 综合面试环节的对话与评分。

职责：
- 基于前两轮（自我介绍 + 技术面试）的结果，有针对性地提问
- 考察维度：职业规划、团队协作、学习能力、薪资与入职意向
- 逐轮追问，完成后输出评分 + 初步录用建议

不负责：
- UI 交互（streamlit_app 负责）
- 综合三轮最终结论（FinalReportAgent 负责）
"""

from __future__ import annotations

import json
import os
import re
import streamlit as st
from dataclasses import dataclass
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv('../.env')

# ─────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────

_SYSTEM_TMPL = """\
你是一位资深 HRD（人力资源总监），正在对候选人进行最终综合面试。

【候选人背景】
姓名：{name}，{years} 年工作经验

【前序面试结论】
自我介绍评估：{self_intro_eval}
技术面试各方向得分：{tech_scores}
技术面试总评：{tech_summary}

【考察维度（逐个覆盖，每次只问一个问题）】
1. 职业规划与动机：为何选择此岗位？未来 3-5 年发展方向？
2. 团队协作与沟通：如何处理技术分歧？跨部门协作经验？
3. 学习与适应能力：如何跟上 AI 快速发展？近期学了什么新技术？
4. 薪资与入职意向：期望薪资范围？最快入职时间？

【输出规则（严格遵守）】
- 若还有维度未覆盖：直接用自然语言提问，语气专业但友好，不输出 JSON
- 若四个维度已基本覆盖（收到 4+ 轮回答后）：输出如下 JSON（用 ```json 包裹）：
{{
  "score": <0-10 浮点数，综合软实力表现>,
  "evaluation": "<2-3 句综合评价>",
  "hire_suggestion": <"强烈推荐" | "推荐" | "存疑" | "不推荐">,
  "done": true
}}
"""

_OPENING_PROMPT = """\
请以 HRD 身份开场（1-2 句话），然后提出第一个问题（从职业规划与动机开始）。"""

_CONTINUE_TMPL = """\
【已有问答记录】
{qa_history}

请继续提问下一个维度，或在四个维度基本覆盖后输出评分 JSON。"""


# ─────────────────────────────────────────────────────────
# 返回值结构
# ─────────────────────────────────────────────────────────

@dataclass
class HRDReply:
    text: str                        # 面试官回复（问题文本）
    done: bool = False               # True 表示本环节结束
    score: Optional[float] = None
    evaluation: str = ""
    hire_suggestion: str = ""        # done=True 时有值


class HRDAgent:
    """
    HRD 综合面试 Agent。
    持有当前面试上下文，逐轮推进对话。
    """

    def __init__(
        self,
        name: str,
        years: int,
        self_intro_eval: str,
        tech_scores: dict,
        tech_summary: str,
    ):
        self._system = _SYSTEM_TMPL.format(
            name=name,
            years=years,
            self_intro_eval=self_intro_eval or "（无评估）",
            tech_scores=json.dumps(tech_scores, ensure_ascii=False) if tech_scores else "（无）",
            tech_summary=tech_summary or "（无）",
        )
        self.model = init_chat_model(
            model=st.secrets["DEEPSEEK_MODEL"],
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url=st.secrets["DEEPSEEK_BASE_URL"],
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )

    def opening(self) -> HRDReply:
        """生成开场白 + 第一个问题。"""
        raw = self._call_llm(_OPENING_PROMPT)
        return self._parse_reply(raw)

    def next_reply(self, qa_pairs: list[dict]) -> HRDReply:
        """
        候选人回答后，继续提问或结束打分。
        qa_pairs: [{"question": str, "answer": str}, ...]
        """
        qa_history = "\n\n".join(
            f"问题{i+1}：{p['question']}\n回答{i+1}：{p['answer']}"
            for i, p in enumerate(qa_pairs)
        )
        prompt = _CONTINUE_TMPL.format(qa_history=qa_history)
        raw = self._call_llm(prompt)
        return self._parse_reply(raw)

    # ─────────────────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            agent = create_agent(model=self.model, system_prompt=self._system)
            response = agent.invoke({"messages": [HumanMessage(prompt)]})
            return response['messages'][-1].content.strip()
        except Exception as e:
            return f"（生成失败：{e}）"

    def _parse_reply(self, raw: str) -> HRDReply:
        json_match = re.search(r"```json\s*([\s\S]*?)```", raw)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if data.get("done"):
                    return HRDReply(
                        text="",
                        done=True,
                        score=float(data.get("score", 7.0)),
                        evaluation=data.get("evaluation", ""),
                        hire_suggestion=data.get("hire_suggestion", "存疑"),
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        return HRDReply(text=raw, done=False)
