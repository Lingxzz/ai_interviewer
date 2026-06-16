"""
QuestionAgent — 负责选题和动态出题。

职责：
- 从题库按权重+难度自适应选题
- 当 Orchestrator 发来 DEEPEN 信号时，在指定 topic 加深选题
- 题库耗尽时调用 LLM 动态生成一道题
- 将选出的题目写入 InterviewState.current_context

不负责：
- 评分（EvaluatorAgent）
- IO 交互（OrchestratorAgent）
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

from core.models import Difficulty, Question, Topic
from state.interview_state import (
    AgentRole, InterviewState, QuestionContext,
)
from questions.bank import ALL_QUESTIONS, QUESTIONS_BY_TOPIC
from tools.rag_store import get_rag_store

load_dotenv('../.env')

# ─────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────

DIFFICULTY_ORDER = [Difficulty.JUNIOR, Difficulty.MID, Difficulty.SENIOR, Difficulty.STAFF]

_DYNAMIC_SYSTEM = """\
你是一位资深 LLM 应用开发工程师面试官。
请根据要求生成一道面试题，严格按照 JSON 格式返回，不要输出任何其他内容。"""

_DYNAMIC_TMPL = """\
请生成一道关于【{topic}】方向的面试题。

要求：
- 难度：{difficulty}（junior=基础概念，mid=工程实践，senior=深度设计，staff=专家视角）
- 背景：{reason}
- 候选人已经回答过以下题目（不要重复考察相同知识点）：
{asked_summary}

返回 JSON：
{{
  "text": <题目正文，清晰具体>,
  "hint": <给候选人的提示，可为 null>,
  "expected_keywords": [<3-6个期望出现的关键词>],
  "scoring_rubric": <评分标准，描述2/4/6/8/10分各需要达到什么水平>,
  "follow_up_template": <预设追问模板，含{{keyword}}占位符，可为 null>
}}"""


class QuestionAgent:
    """
    选题 Agent。每次调用 select() 返回一道题并写入 state。
    支持三种模式：
      1. 普通选题  — 按权重+难度从题库选
      2. 加深选题  — 在指定 topic 内选更难的题（响应 DEEPEN 信号）
      3. 动态出题  — 题库耗尽时 LLM 实时生成
    """

    def __init__(self):
        self.model = init_chat_model(
            model=os.getenv('DEEPSEEK_MODEL'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )

    # ─────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────

    def select(
        self,
        state: InterviewState,
        force_topic: Optional[Topic] = None,
        deepen: bool = False,
        deepen_reason: str = "",
    ) -> bool:
        """
        选出下一道题，写入 state.current_context。
        返回 True 表示成功，False 表示无题可出（应触发结束）。

        Args:
            state:        共享状态
            force_topic:  强制指定 topic（来自 FORCE_TOPIC 信号）
            deepen:       是否在当前 topic 内加深（来自 DEEPEN 信号）
            deepen_reason: 加深原因，用于动态出题的 prompt
        """
        topic = force_topic or self._pick_topic(state)
        difficulty = state.current_difficulty

        # 优先从题库选
        question = self._select_from_bank(state, topic, difficulty, deepen)

        if question is None:
            # 题库耗尽，动态生成
            question = self._generate_dynamic(state, topic, difficulty, deepen_reason)
            if question is None:
                return False
            ctx = QuestionContext(question=question, answer="", is_dynamic=True,
                                  dynamic_reason=deepen_reason or "题库已用尽")
        else:
            ctx = QuestionContext(question=question, answer="")

        state.begin_question(ctx)
        return True

    # ─────────────────────────────────────────────────────
    # Topic 选择
    # ─────────────────────────────────────────────────────

    def _pick_topic(self, state: InterviewState) -> Topic:
        """
        根据配置权重 + 已覆盖次数 + 候选人薄弱方向，选出优先级最高的 topic。
        弱项 topic 额外加权 1.5x，驱动 Agent 多考察短板。
        """
        weights = state.config.topic_weights
        weak = set(state.weak_topics)

        scored: list[tuple[float, Topic]] = []
        for topic, weight in weights.items():
            times = state.topic_counts.get(topic, 0)
            target = max(1, round(state.config.max_questions * weight))
            ratio = times / target
            # 薄弱方向额外加权：ratio 更小 → 优先级更高
            if topic in weak:
                ratio *= 0.67
            scored.append((ratio + random.uniform(0, 0.08), topic))

        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    # ─────────────────────────────────────────────────────
    # 题库选题
    # ─────────────────────────────────────────────────────

    def _select_from_bank(
        self,
        state: InterviewState,
        topic: Topic,
        difficulty: Difficulty,
        deepen: bool,
    ) -> Optional[Question]:
        """
        RAG 语义检索选题（优先），降级到随机选题（兜底）。

        RAG 检索逻辑：
          1. 用候选人简历摘要 + 待验证技能构造查询语句
          2. 在 topic + difficulty 范围内按余弦相似度排序
          3. 取相似度最高、未出过的题目
          4. RAG 无结果时退化为原随机选题逻辑
        """
        used = state.used_question_ids
        idx = DIFFICULTY_ORDER.index(difficulty)

        if deepen:
            preferred_diffs = {DIFFICULTY_ORDER[min(idx + 1, len(DIFFICULTY_ORDER) - 1)],
                               DIFFICULTY_ORDER[idx]}
        else:
            preferred_diffs = {
                DIFFICULTY_ORDER[max(0, idx - 1)],
                DIFFICULTY_ORDER[idx],
                DIFFICULTY_ORDER[min(len(DIFFICULTY_ORDER) - 1, idx + 1)],
            }

        # ── RAG 检索 ──────────────────────────────────────
        rag = get_rag_store()
        if rag.is_built and state.profile:
            profile = state.profile
            # 构造查询：简历摘要 + 待验证技能 + 薄弱方向
            skills_str = " ".join(profile.skills_to_verify[:5])
            weak_str   = " ".join(profile.weak_areas[:3])
            query = (
                f"{profile.resume_summary} "
                f"需要验证的技能：{skills_str} "
                f"薄弱方向：{weak_str}"
            ).strip()

            # 对每个允许的难度分别检索，合并结果
            rag_results: list[dict] = []
            for diff in preferred_diffs:
                hits = rag.retrieve_questions(
                    query=query,
                    topic_filter=topic.value,
                    difficulty_filter=diff.value,
                    exclude_ids=list(used),
                    n=3,
                )
                rag_results.extend(hits)

            # 按相似度降序，取最优
            rag_results.sort(key=lambda x: x["similarity"], reverse=True)

            # 找到对应的 Question 对象
            q_map = {q.id: q for q in ALL_QUESTIONS}
            for hit in rag_results:
                qid = hit.get("id", "")
                if qid and qid not in used and qid in q_map:
                    return q_map[qid]

        # ── 降级：原随机选题逻辑 ──────────────────────────
        candidates = [q for q in QUESTIONS_BY_TOPIC.get(topic, []) if q.id not in used]
        if not candidates:
            candidates = [q for q in ALL_QUESTIONS if q.id not in used]
        if not candidates:
            return None

        preferred = [q for q in candidates if q.difficulty in preferred_diffs]
        pool = preferred if preferred else candidates
        return random.choice(pool)

    # ─────────────────────────────────────────────────────
    # 动态出题
    # ─────────────────────────────────────────────────────

    def _generate_dynamic(
        self,
        state: InterviewState,
        topic: Topic,
        difficulty: Difficulty,
        reason: str,
    ) -> Optional[Question]:
        """题库耗尽时调用 LLM 生成一道新题。"""
        asked_summary = "\n".join(
            f"- [{r.question.topic.value}] {r.question.text[:60]}"
            for r in state.completed_records[-6:]  # 最近6题防 prompt 过长
        ) or "（无）"

        prompt = _DYNAMIC_TMPL.format(
            topic=topic.value,
            difficulty=difficulty.value,
            reason=reason or f"考察候选人在 {topic.value} 方向的掌握深度",
            asked_summary=asked_summary,
        )

        try:
            agent = create_agent(
                model=self.model,
                system_prompt=_DYNAMIC_SYSTEM
            )
            # resp = self.client.messages.create(
            #     model=self.model,
            #     max_tokens=800,
            #     system=_DYNAMIC_SYSTEM,
            #     messages=[{"role": "user", "content": prompt}],
            # )
            response = agent.invoke({
                "messages": [HumanMessage(prompt)]
            })
            raw = response['messages'][-1].content.strip()
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            data = json.loads(cleaned)
        except Exception:
            return None

        # 动态题用特殊 id 前缀，不与题库冲突
        dynamic_id = f"dyn_{topic.value[:4]}_{len(state.used_question_ids):03d}"

        return Question(
            id=dynamic_id,
            topic=topic,
            difficulty=difficulty,
            text=data.get("text", ""),
            hint=data.get("hint"),
            expected_keywords=data.get("expected_keywords", []),
            scoring_rubric=data.get("scoring_rubric", ""),
            follow_up_template=data.get("follow_up_template"),
        )