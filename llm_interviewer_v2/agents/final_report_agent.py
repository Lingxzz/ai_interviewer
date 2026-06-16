"""
FinalReportAgent — 综合三轮面试结果，生成最终录用建议。

职责：
- 输入：自我介绍评分、技术面试报告、HRD 面试评分
- 加权计算综合分（自我介绍 15% + 技术面试 60% + HRD 25%）
- 调用 LLM 生成结构化最终建议
- 返回 FinalReport 数据对象，供 streamlit_app 展示

不负责：
- 任何面试环节的执行
- UI 渲染
"""

from __future__ import annotations

import json
import os
import streamlit as st
import re
from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv('../.env')

# ─────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────

_SYSTEM = """\
你是资深招聘委员会主席，负责综合三轮面试结果给出客观、可执行的最终录用建议。
严格输出 JSON，不含任何其他文字。"""

_PROMPT_TMPL = """\
候选人：{name}，{years} 年工作经验

【第一轮：自我介绍】（权重 15%）
得分：{self_intro_score:.1f} / 10
评估：{self_intro_eval}

【第二轮：技术面试】（权重 60%）
综合得分：{tech_score:.1f} / 10
各方向：{tech_dim_scores}
技术总评：{tech_summary}

【第三轮：HRD 综合面试】（权重 25%）
得分：{hrd_score:.1f} / 10
评估：{hrd_eval}
HRD 初步建议：{hrd_suggestion}

加权综合参考分：{weighted:.1f} / 10

请综合三轮面试数据，生成最终建议：
{{
  "final_score": <综合分，可在加权参考分基础上结合质性评价适当调整，保留 1 位小数>,
  "final_recommendation": <"强烈推荐" | "推荐" | "存疑" | "不推荐">,
  "final_summary": <2-3 句综合评价，需结合三轮表现>,
  "key_strengths": [<2-4 条核心优势，具体而非泛泛>],
  "key_concerns": [<1-3 条主要顾虑或风险点>],
  "onboarding_suggestions": [<1-2 条如果录用后的入职建议>]
}}"""


# ─────────────────────────────────────────────────────────
# 返回值结构
# ─────────────────────────────────────────────────────────

@dataclass
class FinalReport:
    final_score: float
    final_recommendation: str
    final_summary: str
    key_strengths: list[str] = field(default_factory=list)
    key_concerns: list[str] = field(default_factory=list)
    onboarding_suggestions: list[str] = field(default_factory=list)

    # 各轮原始得分，方便展示
    self_intro_score: float = 0.0
    tech_score: float = 0.0
    hrd_score: float = 0.0


class FinalReportAgent:
    """
    综合报告生成 Agent。面试全部完成后调用一次。
    """

    # 加权比例
    WEIGHT_SELF_INTRO = 0.15
    WEIGHT_TECH       = 0.60
    WEIGHT_HRD        = 0.25

    def __init__(self):
        self.model = init_chat_model(
            model=st.secrets["DEEPSEEK_MODEL"],
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url=st.secrets["DEEPSEEK_BASE_URL"],
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )

    def generate(
        self,
        *,
        name: str,
        years: int,
        self_intro_score: float,
        self_intro_eval: str,
        tech_score: float,
        tech_dim_scores: dict,
        tech_summary: str,
        hrd_score: float,
        hrd_eval: str,
        hrd_suggestion: str,
    ) -> FinalReport:
        weighted = (
            self_intro_score * self.WEIGHT_SELF_INTRO
            + tech_score     * self.WEIGHT_TECH
            + hrd_score      * self.WEIGHT_HRD
        )

        prompt = _PROMPT_TMPL.format(
            name=name,
            years=years,
            self_intro_score=self_intro_score,
            self_intro_eval=self_intro_eval or "（无评估）",
            tech_score=tech_score,
            tech_dim_scores=json.dumps(tech_dim_scores, ensure_ascii=False),
            tech_summary=tech_summary or "（无）",
            hrd_score=hrd_score,
            hrd_eval=hrd_eval or "（无评估）",
            hrd_suggestion=hrd_suggestion or "存疑",
            weighted=round(weighted, 1),
        )

        data = self._call_llm(prompt)

        return FinalReport(
            final_score=float(data.get("final_score", round(weighted, 1))),
            final_recommendation=data.get("final_recommendation", hrd_suggestion or "存疑"),
            final_summary=data.get("final_summary", ""),
            key_strengths=data.get("key_strengths", []),
            key_concerns=data.get("key_concerns", []),
            onboarding_suggestions=data.get("onboarding_suggestions", []),
            self_intro_score=self_intro_score,
            tech_score=tech_score,
            hrd_score=hrd_score,
        )

    # ─────────────────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> dict:
        try:
            agent = create_agent(model=self.model, system_prompt=_SYSTEM)
            response = agent.invoke({"messages": [HumanMessage(prompt)]})
            raw = response['messages'][-1].content.strip()
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            return json.loads(cleaned)
        except Exception:
            return {}
