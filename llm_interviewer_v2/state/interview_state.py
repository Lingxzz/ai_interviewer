"""
InterviewState — 多智能体架构下的共享上下文。

设计原则：
1. 单一数据源：所有 Agent 只读写这一个对象，不互相直接调用
2. 只增不改：历史记录 append-only，保证可回溯
3. 可序列化：支持 JSON 持久化，便于断点恢复和调试
4. 事件日志：每次状态变更都记录操作来源，方便排查 Agent 行为
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from core.models import (
    Difficulty, Evaluation, InterviewConfig,
    Question, SessionRecord, Topic,
)


# ─────────────────────────────────────────────────────────
# 状态枚举
# ─────────────────────────────────────────────────────────

class InterviewPhase(str, Enum):
    """面试的宏观阶段，由 OrchestratorAgent 推进。"""
    INIT           = "init"            # 刚创建，尚未开始
    PROFILING      = "profiling"       # ResumeAgent 正在分析简历/JD
    SELF_INTRO     = "self_intro"      # 候选人自我介绍阶段
    QUESTIONING    = "questioning"     # 正常出题阶段
    FOLLOW_UP      = "follow_up"       # EvaluatorAgent 正在追问
    WRAPPING_UP    = "wrapping_up"     # 题目已出完，生成报告中
    HRD_INTERVIEW  = "hrd_interview"   # HRD 综合面试阶段
    FINISHED       = "finished"        # 全部结束


class AgentRole(str, Enum):
    """写入事件日志时标明操作来源。"""
    ORCHESTRATOR = "orchestrator"
    QUESTION     = "question"
    EVALUATOR    = "evaluator"
    RESUME       = "resume"
    REPORTER     = "reporter"
    SYSTEM       = "system"


# ─────────────────────────────────────────────────────────
# 候选人画像（由 ResumeAgent 填充）
# ─────────────────────────────────────────────────────────

@dataclass
class CandidateProfile:
    """
    ResumeAgent 解析 JD + 简历后产出的结构化画像。
    QuestionAgent 据此调整选题方向和权重。
    """
    name: str
    years_of_experience: int

    # ResumeAgent 分析结果
    claimed_skills: list[str] = field(default_factory=list)
    # 简历中声称熟悉的技术栈，需要重点验证
    skills_to_verify: list[str] = field(default_factory=list)
    # 简历中体现较弱、需要补充考察的方向
    weak_areas: list[str] = field(default_factory=list)
    # JD 要求的核心技能，优先覆盖
    jd_required_skills: list[str] = field(default_factory=list)

    # 动态调整的 Topic 权重覆盖（覆盖 InterviewConfig 默认权重）
    # 例如：{"RAG与知识库": 0.35} 表示该候选人 RAG 方向要多考
    topic_weight_overrides: dict[str, float] = field(default_factory=dict)

    # 初始难度建议（ResumeAgent 根据简历质量给出）
    suggested_difficulty: Difficulty = Difficulty.MID

    # 简历原文摘要（供其他 Agent 参考）
    resume_summary: str = ""
    jd_summary: str = ""

    # ResumeAgent 置信度（0-1），低置信度时 QuestionAgent 会忽略 overrides
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────
# 单题上下文（EvaluatorAgent 的工作单元）
# ─────────────────────────────────────────────────────────

@dataclass
class QuestionContext:
    """
    一道题从出题到评分的完整生命周期数据。
    EvaluatorAgent 在 FOLLOW_UP 阶段读写此对象。
    """
    question: Question
    answer: str
    evaluation: Optional[Evaluation] = None

    # 追问轮次记录（支持多轮追问）
    follow_up_rounds: list[dict] = field(default_factory=list)
    # 每轮格式：{"question": str, "answer": str, "score": float}

    # 是否由 LLM 动态生成的题目（非题库）
    is_dynamic: bool = False
    # 生成原因（便于调试）
    dynamic_reason: str = ""

    @property
    def follow_up_count(self) -> int:
        return len(self.follow_up_rounds)

    @property
    def final_score(self) -> float:
        if not self.evaluation:
            return 0.0
        return self.evaluation.effective_score

    def to_session_record(self) -> SessionRecord:
        assert self.evaluation is not None, "尚未完成评估"
        return SessionRecord(question=self.question, evaluation=self.evaluation)


# ─────────────────────────────────────────────────────────
# Agent 信号（Agent 向 Orchestrator 发送的意图）
# ─────────────────────────────────────────────────────────

class AgentSignal(str, Enum):
    """
    Agent 完成工作后向 Orchestrator 返回的信号。
    Orchestrator 根据信号决定下一步流转。
    """
    CONTINUE          = "continue"           # 正常继续，出下一题
    NEED_FOLLOW_UP    = "need_follow_up"     # 需要追问当前题
    FORCE_TOPIC       = "force_topic"        # 强制切换到某个 Topic
    DEEPEN            = "deepen"             # 当前 Topic 需要加深
    WRAP_UP           = "wrap_up"            # 建议提前结束（候选人明显不行）
    FINISH            = "finish"             # 正常结束


@dataclass
class AgentMessage:
    """Agent 发给 Orchestrator 的消息结构。"""
    signal: AgentSignal
    from_agent: AgentRole
    payload: dict = field(default_factory=dict)
    # 例如 FORCE_TOPIC 时 payload = {"topic": "RAG与知识库"}
    # DEEPEN 时 payload = {"reason": "候选人对 chunk 策略的理解停留在表面"}
    timestamp: datetime = field(default_factory=datetime.now)


# ─────────────────────────────────────────────────────────
# 状态变更事件日志
# ─────────────────────────────────────────────────────────

@dataclass
class StateEvent:
    """
    每次状态变更都记录一条 Event，实现完整的操作审计。
    便于调试多 Agent 系统中的意外行为。
    """
    timestamp: datetime
    agent: AgentRole
    action: str          # 动作描述，如 "phase_transition", "score_recorded"
    detail: str          # 人类可读的详情
    data: dict = field(default_factory=dict)  # 结构化附加数据


# ─────────────────────────────────────────────────────────
# 核心：InterviewState
# ─────────────────────────────────────────────────────────

@dataclass
class InterviewState:
    """
    面试全局共享状态。所有 Agent 通过此对象通信。

    读写约定：
    - OrchestratorAgent：读写 phase、current_context、messages_inbox
    - QuestionAgent：    读 profile/records，写 current_context（出题）
    - EvaluatorAgent：   读写 current_context（评分和追问）
    - ResumeAgent：      写 profile
    - ReporterAgent：    只读，生成报告
    """

    # ── 基本信息 ──────────────────────────────────────────
    session_id: str = field(default_factory=lambda: (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    ))
    config: InterviewConfig = field(default_factory=InterviewConfig)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    # ── 候选人画像（ResumeAgent 填充）────────────────────
    profile: CandidateProfile = field(default_factory=lambda: CandidateProfile(
        name="未知", years_of_experience=0
    ))

    # ── 面试阶段（OrchestratorAgent 推进）────────────────
    phase: InterviewPhase = InterviewPhase.INIT

    # ── 题目记录（append-only）───────────────────────────
    # 已完成的题目（含评分）
    completed_records: list[SessionRecord] = field(default_factory=list)
    # 当前正在进行的题目上下文（None 表示等待出下一题）
    current_context: Optional[QuestionContext] = None

    # ── 难度追踪（QuestionAgent 读，EvaluatorAgent 写）───
    current_difficulty: Difficulty = Difficulty.MID
    # 最近 N 次得分，用于动态调整难度
    recent_scores: list[float] = field(default_factory=list)

    # ── Topic 覆盖计数（QuestionAgent 维护）─────────────
    topic_counts: dict[str, int] = field(default_factory=dict)
    # 已用题目 ID，防止重复（QuestionAgent 维护）
    used_question_ids: set[str] = field(default_factory=set)

    # ── Agent 通信信箱（Orchestrator 消费）──────────────
    # Agent 完成工作后把 AgentMessage 放入 inbox
    # OrchestratorAgent 在每个循环开始时消费
    messages_inbox: list[AgentMessage] = field(default_factory=list)

    # ── 事件日志（只追加）────────────────────────────────
    events: list[StateEvent] = field(default_factory=list)

    # ── 原始输入（供 ReporterAgent 引用）─────────────────
    raw_jd: Optional[str] = None
    raw_resume: Optional[str] = None

    # ── 自我介绍阶段数据 ──────────────────────────────────
    self_intro_text: str = ""
    self_intro_qa: list = field(default_factory=list)
    self_intro_score: Optional[float] = None
    self_intro_evaluation: str = ""

    # ── HRD 面试阶段数据 ──────────────────────────────────
    hrd_qa: list = field(default_factory=list)
    hrd_score: Optional[float] = None
    hrd_evaluation: str = ""

    # ─────────────────────────────────────────────────────
    # 便捷属性
    # ─────────────────────────────────────────────────────

    @property
    def question_count(self) -> int:
        return len(self.completed_records)

    @property
    def is_finished(self) -> bool:
        return self.phase == InterviewPhase.FINISHED

    @property
    def average_score(self) -> float:
        scores = [r.evaluation.effective_score for r in self.completed_records]
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    @property
    def scores_by_topic(self) -> dict[str, float]:
        topic_scores: dict[str, list[float]] = {}
        for r in self.completed_records:
            t = r.question.topic.value
            topic_scores.setdefault(t, []).append(r.evaluation.effective_score)
        return {t: round(sum(s) / len(s), 2) for t, s in topic_scores.items()}

    @property
    def weak_topics(self) -> list[str]:
        """得分低于 6 的方向，供 QuestionAgent 决策加题。"""
        return [t for t, s in self.scores_by_topic.items() if s < 6.0]

    @property
    def duration_minutes(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or datetime.now()
        return round((end - self.started_at).total_seconds() / 60, 1)

    # ─────────────────────────────────────────────────────
    # 状态变更方法（所有写操作都经过这里，自动记录事件）
    # ─────────────────────────────────────────────────────

    def transition_phase(self, new_phase: InterviewPhase, agent: AgentRole) -> None:
        old = self.phase
        self.phase = new_phase
        if new_phase == InterviewPhase.QUESTIONING and not self.started_at:
            self.started_at = datetime.now()
        if new_phase == InterviewPhase.FINISHED:
            self.finished_at = datetime.now()
        self._log(agent, "phase_transition", f"{old} → {new_phase}")

    def set_profile(self, profile: CandidateProfile) -> None:
        self.profile = profile
        self.current_difficulty = profile.suggested_difficulty
        # 将画像中的 topic_weight_overrides 合并进 config
        if profile.topic_weight_overrides and profile.confidence >= 0.7:
            for topic, weight in profile.topic_weight_overrides.items():
                self.config.topic_weights[topic] = weight
        self._log(
            AgentRole.RESUME, "profile_set",
            f"候选人：{profile.name}，{profile.years_of_experience}年经验",
            data={"skills_to_verify": profile.skills_to_verify,
                  "weak_areas": profile.weak_areas}
        )

    def begin_question(self, context: QuestionContext) -> None:
        """QuestionAgent 出完题后调用，开启当前题目上下文。"""
        self.current_context = context
        self.used_question_ids.add(context.question.id)
        topic = context.question.topic.value
        self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
        self._log(
            AgentRole.QUESTION, "question_started",
            f"[{context.question.difficulty.value}] {context.question.topic.value}: "
            f"{context.question.text[:60]}...",
            data={"question_id": context.question.id, "is_dynamic": context.is_dynamic}
        )

    def record_score(self, score: float) -> None:
        """EvaluatorAgent 每次评分后调用，维护难度自适应所需的滑动窗口。"""
        self.recent_scores.append(score)
        if len(self.recent_scores) > 4:
            self.recent_scores = self.recent_scores[-4:]  # 保留最近4次
        self._adjust_difficulty()
        self._log(
            AgentRole.EVALUATOR, "score_recorded",
            f"得分 {score:.1f}，当前难度：{self.current_difficulty.value}",
            data={"score": score, "recent_avg": self._recent_avg()}
        )

    def complete_question(self) -> None:
        """EvaluatorAgent 完成整题（含追问）后调用，将当前题存入 completed_records。"""
        assert self.current_context is not None
        assert self.current_context.evaluation is not None
        self.completed_records.append(self.current_context.to_session_record())
        self._log(
            AgentRole.EVALUATOR, "question_completed",
            f"题目 {self.current_context.question.id} 完成，"
            f"最终得分：{self.current_context.final_score:.1f}",
        )
        self.current_context = None

    def post_message(self, message: AgentMessage) -> None:
        """Agent 向 Orchestrator 发送信号。"""
        self.messages_inbox.append(message)
        self._log(
            message.from_agent, "message_posted",
            f"信号：{message.signal.value}",
            data=message.payload
        )

    def consume_messages(self) -> list[AgentMessage]:
        """OrchestratorAgent 消费所有待处理消息。"""
        msgs = self.messages_inbox[:]
        self.messages_inbox.clear()
        return msgs

    # ─────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────

    def _recent_avg(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def _adjust_difficulty(self) -> None:
        if len(self.recent_scores) < 2:
            return
        avg = self._recent_avg()
        diff_order = [Difficulty.JUNIOR, Difficulty.MID, Difficulty.SENIOR, Difficulty.STAFF]
        idx = diff_order.index(self.current_difficulty)
        if avg >= self.config.difficulty_up_threshold and idx < len(diff_order) - 1:
            self.current_difficulty = diff_order[idx + 1]
        elif avg <= self.config.difficulty_down_threshold and idx > 0:
            self.current_difficulty = diff_order[idx - 1]

    def _log(self, agent: AgentRole, action: str, detail: str, data: dict = None) -> None:
        self.events.append(StateEvent(
            timestamp=datetime.now(),
            agent=agent,
            action=action,
            detail=detail,
            data=data or {},
        ))

    # ─────────────────────────────────────────────────────
    # 序列化（用于持久化 / 断点调试）
    # ─────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """输出当前状态快照，便于调试和持久化。"""
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "candidate": {
                "name": self.profile.name,
                "years": self.profile.years_of_experience,
                "difficulty": self.current_difficulty.value,
            },
            "progress": {
                "completed": self.question_count,
                "max": self.config.max_questions,
                "duration_min": self.duration_minutes,
            },
            "scores": {
                "average": self.average_score,
                "by_topic": self.scores_by_topic,
                "recent": self.recent_scores,
                "weak_topics": self.weak_topics,
            },
            "topic_coverage": self.topic_counts,
            "pending_messages": len(self.messages_inbox),
            "event_count": len(self.events),
        }

    def event_log(self) -> list[dict]:
        """输出完整事件日志，用于追踪 Agent 行为。"""
        return [
            {
                "time": e.timestamp.strftime("%H:%M:%S.%f")[:-3],
                "agent": e.agent.value,
                "action": e.action,
                "detail": e.detail,
                **({"data": e.data} if e.data else {}),
            }
            for e in self.events
        ]