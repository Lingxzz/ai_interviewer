"""
ResumeAgent — 分析 JD + 简历，生成候选人画像。

职责：
- 解析 JD，提取岗位核心技能要求
- 解析简历，识别声称技能、项目经验、潜在吹水点
- 生成 CandidateProfile，包含 topic 权重覆盖和重点验证项
- 面试开始前运行一次，结果注入 InterviewState

不负责：
- 出题（QuestionAgent 根据 profile 决策）
- 评分（EvaluatorAgent）
"""

from __future__ import annotations

import json
import os
import re
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

import hashlib
from core.models import Difficulty
from tools.rag_store import get_rag_store
from state.interview_state import AgentRole, CandidateProfile, InterviewState

load_dotenv('../.env')

_SYSTEM = """\
你是一位资深技术面试官，擅长通过分析 JD 和简历来制定面试策略。
请客观分析候选人信息，识别需要重点验证的声称能力，以及与 JD 的匹配程度。
严格按照 JSON 格式返回，不要输出任何其他内容。"""

_PROMPT_TMPL = """\
候选人基本信息：
- 姓名：{name}
- 工作年限：{years} 年

JD 要求：
{jd}

简历内容：
{resume}

请分析并返回以下 JSON：
{{
  "claimed_skills": [<简历中声称掌握的技术技能列表，最多10项>],
  "skills_to_verify": [<需要重点验证的声称技能，通常是简历写得过于漂亮或与经验不符的>],
  "weak_areas": [<简历中体现不足、应在面试中补充考察的方向>],
  "jd_required_skills": [<JD 明确要求的核心技能，最多8项>],
  "topic_weight_overrides": {{
    <topic名称>: <0.0-0.5之间的权重，仅覆盖需要调整的topic>
  }},
  "suggested_difficulty": <"junior"|"mid"|"senior"|"staff"，基于简历质量判断>,
  "resume_summary": <2-3句话概括候选人背景>,
  "jd_summary": <1-2句话概括岗位核心要求>,
  "confidence": <0.0-1.0，分析结果的置信度，简历信息不足时给低分>
}}

可用的 topic 名称（topic_weight_overrides 的 key 必须从以下选取）：
Prompt工程、RAG与知识库、Agent与工具调用、效果评估、系统设计与架构、代码实现、成本与可观测性

{similar_jd_section}"""

_PROMPT_NO_RESUME = """\
候选人基本信息：
- 姓名：{name}
- 工作年限：{years} 年
- 无简历信息

JD 要求：
{jd}

仅根据 JD 生成面试策略，返回相同 JSON 格式，confidence 设为 0.5。"""


class ResumeAgent:
    """
    简历分析 Agent。面试开始前运行一次。
    若无 JD 和简历，则返回基于年限的默认画像。
    """

    def __init__(self):
        self.model = init_chat_model(
            model=st.secrets["DEEPSEEK_MODEL"],
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url=st.secrets["DEEPSEEK_BASE_URL"],
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )


    def analyze(self, state: InterviewState) -> None:
        """
        分析 state 中的 JD 和简历，将 CandidateProfile 写入 state。
        同步方法（面试开始前调用，阻塞可接受）。
        """
        profile = self._build_profile(
            name=state.profile.name,
            years=state.profile.years_of_experience,
            jd=state.raw_jd,
            resume=state.raw_resume,
        )
        state.set_profile(profile)

    def _build_profile(
        self,
        name: str,
        years: int,
        jd: str | None,
        resume: str | None,
    ) -> CandidateProfile:
        # 无任何文本材料 → 纯默认画像
        if not jd and not resume:
            return CandidateProfile(
                name=name,
                years_of_experience=years,
                suggested_difficulty=_years_to_difficulty(years),
                confidence=0.3,
                resume_summary="无简历信息",
                jd_summary="无JD信息",
            )

        if jd and not resume:
            prompt = _PROMPT_NO_RESUME.format(name=name, years=years, jd=jd)
        else:
            # ── RAG：检索相似 JD 历史画像，注入 prompt 辅助权重判断 ──
            similar_jd_section = ""
            rag = get_rag_store()
            if jd and rag.retrieve_similar_jds(jd, n=1):
                similar_jds = rag.retrieve_similar_jds(jd, n=2)
                if similar_jds:
                    lines = ["【参考：历史相似JD的权重配置（仅供参考，请结合当前JD判断）】"]
                    for i, hit in enumerate(similar_jds, 1):
                        weights_str = hit.get("topic_weights", "")
                        lines.append(f"相似JD{i}（相似度{hit['similarity']:.2f}）权重参考：{weights_str}")
                    similar_jd_section = "\n".join(lines)

            prompt = _PROMPT_TMPL.format(
                name=name, years=years,
                jd=jd or "（未提供）",
                resume=resume or "（未提供）",
                similar_jd_section=similar_jd_section,
            )

        data = self._call_llm(prompt)

        profile = CandidateProfile(
            name=name,
            years_of_experience=years,
            claimed_skills=data.get("claimed_skills", []),
            skills_to_verify=data.get("skills_to_verify", []),
            weak_areas=data.get("weak_areas", []),
            jd_required_skills=data.get("jd_required_skills", []),
            topic_weight_overrides=data.get("topic_weight_overrides", {}),
            suggested_difficulty=_parse_difficulty(
                data.get("suggested_difficulty", "mid"), years
            ),
            resume_summary=data.get("resume_summary", ""),
            jd_summary=data.get("jd_summary", ""),
            confidence=float(data.get("confidence", 0.7)),
        )

        # ── RAG：把本次 JD 分析结果写入索引，积累历史 ──
        if jd and data.get("topic_weight_overrides"):
            rag = get_rag_store()
            jd_id = "jd_" + hashlib.md5(jd.encode()).hexdigest()[:12]
            rag.add_jd_profile(
                jd_id=jd_id,
                jd_text=jd,
                metadata={
                    "topic_weights": str(data.get("topic_weight_overrides", {})),
                    "required_skills": ", ".join(data.get("jd_required_skills", [])),
                    "difficulty": data.get("suggested_difficulty", "mid"),
                    "jd_summary": data.get("jd_summary", ""),
                },
            )

        return profile

    def _call_llm(self, prompt: str) -> dict:
        try:
            agent = create_agent(
                model=self.model,
                system_prompt=_SYSTEM
            )
            # resp = self.client.messages.create(
            #     model=self.model,
            #     max_tokens=800,
            #     system=_SYSTEM,
            #     messages=[{"role": "user", "content": prompt}],
            # )
            response = agent.invoke({
                "messages": [HumanMessage(prompt)]
            })
            raw = response['messages'][-1].content.strip()
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            return json.loads(cleaned)
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────

def _years_to_difficulty(years: int) -> Difficulty:
    if years < 2:
        return Difficulty.JUNIOR
    if years < 4:
        return Difficulty.MID
    if years < 7:
        return Difficulty.SENIOR
    return Difficulty.STAFF


def _parse_difficulty(value: str, years: int) -> Difficulty:
    mapping = {
        "junior": Difficulty.JUNIOR,
        "mid": Difficulty.MID,
        "senior": Difficulty.SENIOR,
        "staff": Difficulty.STAFF,
    }
    return mapping.get(value.lower(), _years_to_difficulty(years))