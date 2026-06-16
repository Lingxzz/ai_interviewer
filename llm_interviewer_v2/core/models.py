"""
Core data models for the interview agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Difficulty(str, Enum):
    JUNIOR = "junior"       # 0-2 年
    MID = "mid"             # 2-5 年
    SENIOR = "senior"       # 5+ 年
    STAFF = "staff"         # 技术专家


class Topic(str, Enum):
    PROMPT = "Prompt工程"
    RAG = "RAG与知识库"
    AGENT = "Agent与工具调用"
    EVALUATION = "效果评估"
    SYSTEM = "系统设计与架构"
    CODING = "代码实现"
    COST = "成本与可观测性"


@dataclass
class InterviewConfig:
    max_questions: int = 10
    min_questions: int = 6
    follow_up_threshold: float = 6.0   # score below this triggers follow-up
    difficulty_up_threshold: float = 7.5
    difficulty_down_threshold: float = 4.0
    topic_weights: dict[str, float] = field(default_factory=lambda: {
        Topic.PROMPT: 0.20,
        Topic.RAG: 0.25,
        Topic.AGENT: 0.20,
        Topic.EVALUATION: 0.15,
        Topic.SYSTEM: 0.10,
        Topic.CODING: 0.05,
        Topic.COST: 0.05,
    })


@dataclass
class Question:
    id: str
    topic: Topic
    difficulty: Difficulty
    text: str
    hint: Optional[str]
    expected_keywords: list[str]       # 期望出现的关键词
    scoring_rubric: str                # 给 LLM evaluator 的评分标准
    follow_up_template: Optional[str]  # 追问模板，None 表示由 LLM 动态生成


@dataclass
class Evaluation:
    question_id: str
    answer: str
    score: float           # 0-10
    depth: int             # 1=surface, 2=working, 3=deep, 4=expert
    keyword_hits: list[str]
    keyword_misses: list[str]
    follow_up: Optional[str]
    brief_feedback: str    # 简短反馈，面试现场展示
    detailed_notes: str    # 详细评估，写入报告
    follow_up_score: Optional[float] = None
    follow_up_answer: Optional[str] = None

    @property
    def depth_label(self) -> str:
        return {1: "表面", 2: "基础", 3: "深入", 4: "专家"}[self.depth]

    @property
    def effective_score(self) -> float:
        """Weighted score including follow-up."""
        if self.follow_up_score is not None:
            return self.score * 0.6 + self.follow_up_score * 0.4
        return self.score


@dataclass
class SessionRecord:
    question: Question
    evaluation: Evaluation


@dataclass
class InterviewReport:
    session_id: str
    candidate_name: str
    candidate_years: int
    total_score: float
    dimension_scores: dict[str, float]
    hire_recommendation: str
    strengths: list[str]
    risks: list[str]
    next_round_questions: list[str]
    jd_match_score: Optional[float]
    records: list[SessionRecord]
    summary: str

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "candidate_name": self.candidate_name,
            "candidate_years": self.candidate_years,
            "total_score": round(self.total_score, 2),
            "dimension_scores": {k: round(v, 2) for k, v in self.dimension_scores.items()},
            "hire_recommendation": self.hire_recommendation,
            "strengths": self.strengths,
            "risks": self.risks,
            "next_round_questions": self.next_round_questions,
            "jd_match_score": self.jd_match_score,
            "summary": self.summary,
            "records": [
                {
                    "question": {
                        "id": r.question.id,
                        "topic": r.question.topic,
                        "text": r.question.text,
                    },
                    "score": round(r.evaluation.effective_score, 2),
                    "depth": r.evaluation.depth_label,
                    "follow_up": r.evaluation.follow_up,
                    "follow_up_answer": r.evaluation.follow_up_answer,
                    "notes": r.evaluation.detailed_notes,
                }
                for r in self.records
            ],
        }