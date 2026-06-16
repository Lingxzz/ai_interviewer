"""
ReporterAgent — 面试结束后生成结构化报告。

职责：
- 从 InterviewState 读取全部评估记录
- 调用 LLM 生成综合评价、录用建议、追问建议
- 返回 InterviewReport（可序列化为 JSON）

只读 state，不写入任何字段。
"""

from __future__ import annotations

import json
import os
import re
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

from core.models import InterviewReport, SessionRecord
from state.interview_state import AgentRole, InterviewState

load_dotenv('../.env')

_SYSTEM = """\
你是一位经验丰富的技术招聘负责人，擅长评估 LLM 应用开发工程师的综合能力。
请基于面试记录生成客观、专业的候选人评估报告。
严格按照 JSON 格式输出，不要包含任何额外文字。"""

_PROMPT_TMPL = """\
候选人：{name}，{years} 年工作经验
面试时长：{duration:.0f} 分钟
候选人背景：{resume_summary}
岗位要求：{jd_summary}

各题评估详情：
{score_detail}

整体得分统计：
- 平均分：{avg_score:.1f} / 10
- 各方向：{topic_scores}
- 薄弱方向：{weak_topics}

请生成面试报告：
{{
  "total_score": <综合分数 0-10，综合各题表现，不要直接取平均>,
  "hire_recommendation": <"强烈推荐" | "推荐" | "存疑" | "不推荐">,
  "summary": <2-3句总体评价，结合候选人背景和岗位要求>,
  "strengths": [<3-5条具体优势，每条15字以内>],
  "risks": [<2-4条风险点/短板，每条15字以内>],
  "next_round_questions": [<给下一轮面试官的2-3个有针对性的追问建议>],
  "jd_match_score": <与JD的匹配度 0-10，无JD信息时返回 null>
}}"""


class ReporterAgent:
    """报告生成 Agent，只读 InterviewState，面试结束后调用一次。"""

    def __init__(self):
        self.model = init_chat_model(
            model=os.getenv('DEEPSEEK_MODEL'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )

    def generate(self, state: InterviewState) -> InterviewReport:
        """从 state 生成完整面试报告（同步）。"""
        score_detail = self._format_records(state.completed_records)

        prompt = _PROMPT_TMPL.format(
            name=state.profile.name,
            years=state.profile.years_of_experience,
            duration=state.duration_minutes,
            resume_summary=state.profile.resume_summary or "（无简历信息）",
            jd_summary=state.profile.jd_summary or "（无JD信息）",
            score_detail=score_detail,
            avg_score=state.average_score,
            topic_scores=state.scores_by_topic,
            weak_topics=state.weak_topics or "（无明显薄弱项）",
        )

        data = self._call_llm(prompt, fallback_avg=state.average_score)

        return InterviewReport(
            session_id=state.session_id,
            candidate_name=state.profile.name,
            candidate_years=state.profile.years_of_experience,
            total_score=float(data.get("total_score", state.average_score)),
            dimension_scores=state.scores_by_topic,
            hire_recommendation=data.get("hire_recommendation", "存疑"),
            strengths=data.get("strengths", []),
            risks=data.get("risks", []),
            next_round_questions=data.get("next_round_questions", []),
            jd_match_score=data.get("jd_match_score"),
            records=state.completed_records,
            summary=data.get("summary", ""),
        )

    # ─────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────

    def _format_records(self, records: list[SessionRecord]) -> str:
        lines = []
        for i, r in enumerate(records, 1):
            e = r.evaluation
            depth_map = {1: "表面", 2: "基础", 3: "深入", 4: "专家"}
            line = (
                f"Q{i} [{r.question.topic.value}]"
                f"{'(动态)' if r.question.id.startswith('dyn_') else ''}"
                f" {r.question.text[:60]}...\n"
                f"   得分: {e.effective_score:.1f}/10"
                f"  深度: {depth_map.get(e.depth, '未知')}\n"
                f"   命中关键词: {', '.join(e.keyword_hits) or '无'}\n"
                f"   缺失关键词: {', '.join(e.keyword_misses) or '无'}\n"
                f"   评估: {e.detailed_notes[:120]}"
            )
            if e.follow_up_score is not None:
                line += f"\n   追问得分: {e.follow_up_score:.1f}"
            lines.append(line)
        return "\n\n".join(lines)

    def _call_llm(self, prompt: str, fallback_avg: float) -> dict:
        try:
            agent = create_agent(
                model=self.model,
                system_prompt=_SYSTEM
            )
            # resp = self.client.messages.create(
            #     model=self.model,
            #     max_tokens=1000,
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
            return {
                "total_score": fallback_avg,
                "hire_recommendation": "存疑" if fallback_avg < 6 else "推荐",
                "summary": "报告生成异常，请参考各题得分详情。",
                "strengths": ["需人工评估"],
                "risks": ["报告生成失败"],
                "next_round_questions": ["建议人工复核"],
                "jd_match_score": None,
            }