"""
EvaluatorAgent — 负责单题的完整评估生命周期。

职责边界：
- 对候选人的回答打分、判断理解深度
- 自主决定是否追问、追问什么（最多 MAX_FOLLOW_UPS 轮）
- 将最终评估结果写回 InterviewState
- 向 Orchestrator 发送信号（CONTINUE / DEEPEN / WRAP_UP）

不负责：
- 出题（QuestionAgent 负责）
- IO 交互（Orchestrator 负责）
- 报告生成（ReporterAgent 负责）
"""

from __future__ import annotations

import json
import os
import streamlit as st
import re
from typing import AsyncIterator, Callable, Awaitable
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

from core.models import Evaluation, Question
from tools.rag_store import get_rag_store
from state.interview_state import (
    AgentMessage, AgentRole, AgentSignal,
    InterviewPhase, InterviewState, QuestionContext,
)

load_dotenv('../.env')

# ─────────────────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一位资深的 LLM 应用开发工程师面试官，擅长评估候选人对 Prompt 工程、RAG、Agent、\
系统设计等方向的掌握深度。

评估原则：
1. 区分"背过概念"和"真正理解"——能复述定义不等于理解原理
2. 关注候选人是否主动提到 trade-off 和局限性，这是深度的核心标志
3. 优先追问候选人「提到但没展开」的关键点，而不是重复已经回答好的内容
4. 评分严格：8分以上要求对细节有清晰认知；10分要求专家级视角

严格按照 JSON 格式返回，不要输出任何其他内容。"""

_EVAL_TMPL = """\
【面试题目】
{question_text}

【评分标准】
{scoring_rubric}

【期望关键词】{expected_keywords}

【候选人回答】
{answer}

请返回以下 JSON：
{{
  "score": <0-10 浮点数>,
  "depth": <1-4 整数：1=表面 2=基础 3=深入 4=专家>,
  "keyword_hits": [<回答中体现的关键词>],
  "keyword_misses": [<回答中缺失的重要关键词>],
  "needs_follow_up": <bool，score<7 或存在明显盲点时为 true>,
  "follow_up_question": <追问内容字符串，needs_follow_up=false 时为 null>,
  "follow_up_focus": <追问想验证的核心点，如"是否真正理解 chunk 策略的选型逻辑">,
  "brief_feedback": <给候选人的简短反馈，1-2句，≤60字>,
  "detailed_notes": <给面试官的详细评估，包括优缺点分析>
}}"""

_EVAL_WITH_RAG_TMPL = """\【面试题目】
{question_text}

【评分标准】
{scoring_rubric}

【期望关键词】{expected_keywords}

【RAG 参考：相似题目的高分作答要点】
{rag_context}

【候选人回答】
{answer}

请参考上述 RAG 检索到的相似题目高分要点，结合本题评分标准，返回以下 JSON：
{{
  "score": <0-10 浮点数>,
  "depth": <1-4 整数：1=表面 2=基础 3=深入 4=专家>,
  "keyword_hits": [<回答中体现的关键词>],
  "keyword_misses": [<回答中缺失的重要关键词>],
  "needs_follow_up": <bool，score<7 或存在明显盲点时为 true>,
  "follow_up_question": <追问内容字符串，needs_follow_up=false 时为 null>,
  "follow_up_focus": <追问想验证的核心点>,
  "brief_feedback": <给候选人的简短反馈，1-2句，≤60字>,
  "detailed_notes": <给面试官的详细评估，包括优缺点分析>
}}"""

_FOLLOW_UP_TMPL = """\
【原始题目】{question_text}

【对话历史】
{history}

【本轮追问】{follow_up_question}
【候选人追问回答】{follow_up_answer}

【追问目的】{follow_up_focus}

请评估本轮追问回答，返回 JSON：
{{
  "score": <0-10 浮点数，仅评估本轮回答质量>,
  "revealed": <bool，候选人是否展现了追问目的想验证的能力>,
  "needs_another_follow_up": <bool，是否还有必要继续追问>,
  "next_follow_up_question": <下一个追问，needs_another_follow_up=false 时为 null>,
  "next_follow_up_focus": <下一追问的验证目的，无追问时为 null>,
  "notes": <简短评估说明>
}}"""

_WRAP_UP_CHECK_TMPL = """\
候选人在本次面试中的表现摘要：
- 已完成题数：{completed}
- 平均分：{avg_score:.1f}
- 各方向得分：{topic_scores}
- 薄弱方向：{weak_topics}

当前候选人对刚才问题的得分：{current_score:.1f}

请判断是否应该提前结束面试（候选人整体表现明显不达标），返回 JSON：
{{
  "should_wrap_up": <bool>,
  "reason": <如果建议结束，说明原因>
}}"""


# ─────────────────────────────────────────────────────────
# EvaluatorAgent
# ─────────────────────────────────────────────────────────

# IO 回调类型：接收追问文本，返回候选人的追问回答
AskFollowUpFn = Callable[[str], Awaitable[str]]


class EvaluatorAgent:
    """
    有状态的评估 Agent，负责单题的完整评估流程（含多轮追问）。

    使用方式：
        agent = EvaluatorAgent()
        await agent.run(state, ask_fn)

    其中 ask_fn 是 IO 层注入的回调，负责向候选人展示追问并收集回答。
    这样 EvaluatorAgent 本身不依赖任何 IO 框架，便于测试。
    """

    MAX_FOLLOW_UPS = 2          # 单题最多追问轮数
    WRAP_UP_CHECK_AFTER = 4     # 完成 N 题后才考虑提前结束

    def __init__(self):
        self.model = init_chat_model(
            model=st.secrets["DEEPSEEK_MODEL"],
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url=st.secrets["DEEPSEEK_BASE_URL"],
            model_provider='openai',
            extra_body={"thinking":{"type":"disabled"}}
        )

    # ─────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────

    async def run(self, state: InterviewState, ask_fn: AskFollowUpFn) -> None:
        """
        完整处理 state.current_context 中的当前题目。

        流程：
          1. 评估初次回答
          2. 若需要追问，循环追问（最多 MAX_FOLLOW_UPS 轮）
          3. 将最终 Evaluation 写回 state
          4. 向 Orchestrator 发送信号

        Args:
            state:   共享面试状态
            ask_fn:  IO 回调，签名 async (follow_up_text: str) -> str
                     由 OrchestratorAgent / main.py 注入，负责展示追问并收集回答
        """
        ctx = state.current_context
        assert ctx is not None, "EvaluatorAgent.run() 调用时 current_context 不能为 None"

        state.transition_phase(InterviewPhase.FOLLOW_UP, AgentRole.EVALUATOR)

        # ── Step 1: 评估初次回答 ──────────────────────────
        eval_data = self._call_eval(ctx.question, ctx.answer)
        follow_up_question = self._resolve_follow_up(ctx.question, eval_data)
        follow_up_focus = eval_data.get("follow_up_focus", "")

        # ── Step 2: 追问循环 ──────────────────────────────
        follow_up_scores: list[float] = []

        round_idx = 0
        while (
            eval_data.get("needs_follow_up")
            and follow_up_question
            and round_idx < self.MAX_FOLLOW_UPS
        ):
            # 通过注入的 IO 回调获取候选人追问回答（EvaluatorAgent 不关心 IO 细节）
            follow_up_answer = await ask_fn(follow_up_question)

            fu_data = self._call_follow_up_eval(
                question=ctx.question,
                history=ctx.follow_up_rounds,
                follow_up_question=follow_up_question,
                follow_up_answer=follow_up_answer,
                follow_up_focus=follow_up_focus,
            )

            fu_score = float(fu_data.get("score", 5.0))
            follow_up_scores.append(fu_score)

            # 记录本轮追问
            ctx.follow_up_rounds.append({
                "question": follow_up_question,
                "answer":   follow_up_answer,
                "score":    fu_score,
                "focus":    follow_up_focus,
                "revealed": fu_data.get("revealed", False),
                "notes":    fu_data.get("notes", ""),
            })

            # 决定是否继续追问
            if fu_data.get("needs_another_follow_up") and round_idx + 1 < self.MAX_FOLLOW_UPS:
                follow_up_question = fu_data.get("next_follow_up_question")
                follow_up_focus    = fu_data.get("next_follow_up_focus", "")
            else:
                break

            round_idx += 1

        # ── Step 3: 构建最终 Evaluation ──────────────────
        evaluation = self._build_evaluation(
            question=ctx.question,
            answer=ctx.answer,
            eval_data=eval_data,
            follow_up_question=self._resolve_follow_up(ctx.question, eval_data),
            follow_up_scores=follow_up_scores,
            follow_up_rounds=ctx.follow_up_rounds,
        )
        ctx.evaluation = evaluation

        # ── Step 4: 写回 state ────────────────────────────
        state.record_score(evaluation.effective_score)
        state.complete_question()
        state.transition_phase(InterviewPhase.QUESTIONING, AgentRole.EVALUATOR)

        # ── Step 5: 发送信号给 Orchestrator ──────────────
        signal = self._decide_signal(state, evaluation.effective_score, eval_data)
        state.post_message(AgentMessage(
            signal=signal,
            from_agent=AgentRole.EVALUATOR,
            payload=self._signal_payload(signal, state, eval_data),
        ))

    # ─────────────────────────────────────────────────────
    # LLM 调用
    # ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        agent = create_agent(
            model=self.model,
            system_prompt=_SYSTEM_PROMPT
        )
        # resp = self.client.messages.create(
        #     model=self.model,
        #     max_tokens=1000,
        #     system=_SYSTEM_PROMPT,
        #     messages=[{"role": "user", "content": prompt}],
        # )
        response = agent.invoke({
            "messages":[HumanMessage(prompt)]
        })
        return response['messages'][-1].content

    def _parse_json(self, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        return json.loads(cleaned)

    def _safe_call(self, prompt: str, fallback: dict) -> dict:
        """调用 LLM 并解析 JSON，失败时返回 fallback。"""
        try:
            raw = self._call_llm(prompt)
            return self._parse_json(raw)
        except (json.JSONDecodeError, Exception) as e:
            fallback["_parse_error"] = str(e)
            return fallback

    def _call_eval(self, question: Question, answer: str) -> dict:
        # ── RAG 增强：检索相似题目的评分标准作为参考上下文 ──
        rag = get_rag_store()
        rag_context = ""
        if rag.is_built:
            rubric_hits = rag.retrieve_rubrics(
                question_text=question.text,
                answer_text=answer,
                topic=question.topic.value,
                n=2,
            )
            if rubric_hits:
                parts = []
                for i, hit in enumerate(rubric_hits, 1):
                    if hit.get("question_id") == question.id:
                        continue  # 跳过自身
                    parts.append(
                        "参考题" + str(i) + "（相似度" + f"{hit['similarity']:.2f}" + "）：" + hit['question_text'][:60] + "\n"
                        "高分要点：" + hit['rubric']
                    )
                rag_context = "\n".join(parts)
        # 有 RAG 上下文时用增强模板，否则退回原模板
        if rag_context:
            prompt = _EVAL_WITH_RAG_TMPL.format(
                question_text=question.text,
                scoring_rubric=question.scoring_rubric,
                expected_keywords=", ".join(question.expected_keywords),
                rag_context=rag_context,
                answer=answer,
            )
        else:
            prompt = _EVAL_TMPL.format(
                question_text=question.text,
                scoring_rubric=question.scoring_rubric,
                expected_keywords=", ".join(question.expected_keywords),
                answer=answer,
            )

        fallback = {
            "score": 5.0, "depth": 2,
            "keyword_hits": [], "keyword_misses": question.expected_keywords,
            "needs_follow_up": True,
            "follow_up_question": "能否详细说明你的思路？",
            "follow_up_focus": "基础理解验证",
            "brief_feedback": "回答需要更多细节。",
            "detailed_notes": "LLM 评估失败，请人工复核。",
        }
        return self._safe_call(prompt, fallback)

    def _call_follow_up_eval(
        self,
        question: Question,
        history: list[dict],
        follow_up_question: str,
        follow_up_answer: str,
        follow_up_focus: str,
    ) -> dict:
        history_text = "\n".join(
            f"追问{i+1}：{r['question']}\n回答{i+1}：{r['answer']}"
            for i, r in enumerate(history)
        ) or "（无历史追问）"

        prompt = _FOLLOW_UP_TMPL.format(
            question_text=question.text,
            history=history_text,
            follow_up_question=follow_up_question,
            follow_up_answer=follow_up_answer,
            follow_up_focus=follow_up_focus,
        )
        fallback = {
            "score": 5.0, "revealed": False,
            "needs_another_follow_up": False,
            "next_follow_up_question": None,
            "next_follow_up_focus": None,
            "notes": "LLM 评估失败，请人工复核。",
        }
        return self._safe_call(prompt, fallback)

    def _check_wrap_up(self, state: InterviewState, current_score: float) -> bool:
        """让 LLM 判断是否应提前结束面试。"""
        if state.question_count < self.WRAP_UP_CHECK_AFTER:
            return False
        if state.average_score >= 5.0:
            return False  # 均分尚可，不用提前结束

        prompt = _WRAP_UP_CHECK_TMPL.format(
            completed=state.question_count,
            avg_score=state.average_score,
            topic_scores=state.scores_by_topic,
            weak_topics=state.weak_topics,
            current_score=current_score,
        )
        data = self._safe_call(prompt, {"should_wrap_up": False, "reason": ""})
        return bool(data.get("should_wrap_up", False))

    # ─────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────

    def _resolve_follow_up(self, question: Question, eval_data: dict) -> str | None:
        """
        决定追问内容：
        - 优先使用题目预设的 follow_up_template（用关键词填充）
        - 若无模板，使用 LLM 生成的 follow_up_question
        """
        if not eval_data.get("needs_follow_up"):
            return None

        if question.follow_up_template:
            hits = eval_data.get("keyword_hits", [])
            keyword = (
                hits[0] if hits
                else (question.expected_keywords[0] if question.expected_keywords else "")
            )
            return question.follow_up_template.replace("{keyword}", keyword)

        return eval_data.get("follow_up_question")

    def _build_evaluation(
        self,
        question: Question,
        answer: str,
        eval_data: dict,
        follow_up_question: str | None,
        follow_up_scores: list[float],
        follow_up_rounds: list[dict],
    ) -> Evaluation:
        """将 LLM 评估数据和追问得分汇总为 Evaluation 对象。"""
        score = float(eval_data.get("score", 5.0))

        # 追问得分取加权平均：越后面的追问权重越高（递进验证）
        follow_up_score: float | None = None
        if follow_up_scores:
            weights = [1.0 + i * 0.5 for i in range(len(follow_up_scores))]
            follow_up_score = sum(s * w for s, w in zip(follow_up_scores, weights)) / sum(weights)

        # 将追问明细拼入 detailed_notes
        fu_notes = ""
        for i, r in enumerate(follow_up_rounds, 1):
            revealed = "✓ 验证通过" if r.get("revealed") else "✗ 未能验证"
            fu_notes += (
                f"\n[追问{i}] {r['question']}\n"
                f"  回答：{r['answer'][:100]}{'...' if len(r['answer']) > 100 else ''}\n"
                f"  得分：{r['score']:.1f}  {revealed}\n"
                f"  说明：{r.get('notes', '')}"
            )

        detailed_notes = eval_data.get("detailed_notes", "")
        if fu_notes:
            detailed_notes += "\n\n--- 追问记录 ---" + fu_notes

        return Evaluation(
            question_id=question.id,
            answer=answer,
            score=score,
            depth=int(eval_data.get("depth", 2)),
            keyword_hits=eval_data.get("keyword_hits", []),
            keyword_misses=eval_data.get("keyword_misses", []),
            follow_up=follow_up_question,
            brief_feedback=eval_data.get("brief_feedback", ""),
            detailed_notes=detailed_notes,
            follow_up_score=round(follow_up_score, 2) if follow_up_score is not None else None,
            follow_up_answer=follow_up_rounds[-1]["answer"] if follow_up_rounds else None,
        )

    def _decide_signal(
        self, state: InterviewState, score: float, eval_data: dict
    ) -> AgentSignal:
        """根据本题评分和全局状态决定发送什么信号。"""

        # 题目已全部出完
        if state.question_count >= state.config.max_questions:
            return AgentSignal.FINISH

        # 提前结束检查（低分 + 多题后）
        if self._check_wrap_up(state, score):
            return AgentSignal.WRAP_UP

        # 某个方向有盲点，建议加深
        misses = eval_data.get("keyword_misses", [])
        topic = state.completed_records[-1].question.topic.value if state.completed_records else ""
        topic_score = state.scores_by_topic.get(topic, score)
        if topic_score < 5.5 and len(misses) >= 2:
            return AgentSignal.DEEPEN

        return AgentSignal.CONTINUE

    def _signal_payload(
        self, signal: AgentSignal, state: InterviewState, eval_data: dict
    ) -> dict:
        """为不同信号构建 payload，供 Orchestrator 决策使用。"""
        if signal == AgentSignal.DEEPEN:
            topic = (
                state.completed_records[-1].question.topic.value
                if state.completed_records else ""
            )
            return {
                "topic": topic,
                "reason": f"关键词缺失：{eval_data.get('keyword_misses', [])}",
                "topic_score": state.scores_by_topic.get(topic, 0),
            }
        if signal == AgentSignal.WRAP_UP:
            return {
                "avg_score": state.average_score,
                "weak_topics": state.weak_topics,
            }
        if signal == AgentSignal.FINISH:
            return {"completed": state.question_count}
        return {}