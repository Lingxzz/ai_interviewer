# core/orchestrator.py
"""
OrchestratorAgent — 智能体主协调器。

架构职责：
  - 维护面试状态机（InterviewState），是唯一推进 phase 的角色
  - 消费各 Agent 发出的 AgentMessage，通过 LLM 做路由决策
  - 将 IO 回调注入 EvaluatorAgent，自身不直接处理业务逻辑
  - 整合 PlannerAgent / ReflectionAgent 的洞察，辅助 LLM 决策

信号路由表（由 LLM 动态决策，规则兜底）：
  CONTINUE       → 正常出下一题
  DEEPEN         → 在当前 Topic 内出更难的题
  FORCE_TOPIC    → 强制切换到指定 Topic
  WRAP_UP        → 提前结束面试
  FINISH         → 正常结束，生成报告
"""

from __future__ import annotations

import asyncio
import json
import re
import os
from dataclasses import dataclass
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv


from agents.question_agent import QuestionAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.report_agent import ReporterAgent
from agents.resume_agent import ResumeAgent
from tools.rag_store import get_rag_store
from questions.bank import ALL_QUESTIONS
from agents.candidate_model import CandidateModel
from agents.planner_agent import PlannerAgent
from agents.reflection_agent import ReflectionAgent
from state.interview_state import (
    AgentMessage, AgentRole, AgentSignal,
    CandidateProfile, InterviewPhase, InterviewState,
)

load_dotenv('../.env')

# ─────────────────────────────────────────────────────────
# LLM 决策 Prompt
# ─────────────────────────────────────────────────────────

_ROUTING_SYSTEM = """\
你是一位资深技术面试的主持人（Orchestrator）。
你的任务是根据当前面试状态和各 Agent 发出的信号，决定下一步行动。

可用动作：
  continue      — 正常继续，出下一题（默认）
  deepen        — 在当前薄弱方向出更难的题
  force_topic   — 强制切换到某个 Topic（需在 topic 字段指定）
  wrap_up       — 候选人表现明显不足，提前结束
  finish        — 已覆盖足够方向，正常结束

决策原则：
1. 至少完成 min_questions 道题才能 wrap_up 或 finish
2. 某方向平均分 < 5 且还有余量时，优先 deepen 该方向
3. 某方向还未覆盖且权重较高时，可 force_topic
4. 连续 3 题平均分 < 4 时建议 wrap_up
5. 已达 max_questions 时必须 finish

严格返回 JSON，不要输出其他内容：
{
  "action": <"continue"|"deepen"|"force_topic"|"wrap_up"|"finish">,
  "topic": <force_topic 时必填，其他时为 null>,
  "reason": <一句话说明决策依据>
}"""

_ROUTING_TMPL = """\
【面试进度】
已完成题数：{completed} / 最少 {min_q} / 最多 {max_q}
当前难度：{difficulty}
面试时长：{duration:.0f} 分钟

【得分概况】
全局平均分：{avg_score:.1f}
最近 {recent_count} 题得分：{recent_scores}
各方向得分：{topic_scores}
薄弱方向：{weak_topics}
尚未覆盖的方向：{uncovered_topics}

【各 Agent 信号】
{signals}

【PlannerAgent 建议】
{planner_plan}

【ReflectionAgent 洞察】
{reflection}

请做出下一步路由决策。"""


# ─────────────────────────────────────────────────────────
# 决策结果
# ─────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    action: str          # continue / deepen / force_topic / wrap_up / finish
    topic: Optional[str] = None
    reason: str = ""


class OrchestratorAgent:
    """
    智能体主协调器。

    使用 LLM 做信号路由决策，集成 PlannerAgent 和 ReflectionAgent 的洞察，
    以状态机驱动整个面试流程。
    """

    def __init__(self):
        self.model = init_chat_model(
            model=os.getenv('DEEPSEEK_MODEL'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )

        # 子 Agent
        self.question_agent  = QuestionAgent()
        self.evaluator_agent = EvaluatorAgent()
        self.reporter_agent  = ReporterAgent()
        self.resume_agent    = ResumeAgent()

        # 辅助模块
        self.candidate_model = CandidateModel()
        self.planner         = PlannerAgent(self.candidate_model, None)  # state 在 _run 中注入
        self.reflector       = ReflectionAgent(self.candidate_model, None)

        # 共享状态（_run 中初始化）
        self.state: Optional[InterviewState] = None

        # RAG 索引：题库全量向量化（首次创建时执行，~1秒）
        rag = get_rag_store()
        if not rag.is_built:
            rag.build_from_bank(ALL_QUESTIONS)

    # ─────────────────────────────────────────────────────
    # 公开入口
    # ─────────────────────────────────────────────────────

    def run(self):
        """同步入口，内部启动 asyncio 事件循环。"""
        asyncio.run(self._run())

    # ─────────────────────────────────────────────────────
    # 主流程
    # ─────────────────────────────────────────────────────

    async def _run(self):
        print("\n" + "=" * 60)
        print("  LLM 工程师面试系统（智能体版）")
        print("=" * 60 + "\n")

        # ── Step 1: 采集输入 ──────────────────────────────
        name, years, raw_jd, raw_resume = self._collect_inputs()

        # ── Step 2: 初始化状态，注入 state 到辅助模块 ────
        self.state = InterviewState()
        self.state.profile = CandidateProfile(name=name, years_of_experience=years)
        self.state.raw_jd = raw_jd
        self.state.raw_resume = raw_resume
        self.planner.state   = self.state
        self.reflector.state = self.state

        # ── Step 3: ResumeAgent 分析画像 ─────────────────
        print("\n🔍 正在分析候选人信息...", flush=True)
        self.state.transition_phase(InterviewPhase.PROFILING, AgentRole.ORCHESTRATOR)
        self.resume_agent.analyze(self.state)
        p = self.state.profile
        print(f"✅ 画像完成 | 难度建议：{self.state.current_difficulty.value}")
        if p.skills_to_verify:
            print(f"   重点验证：{', '.join(p.skills_to_verify[:4])}")
        if p.weak_areas:
            print(f"   补充考察：{', '.join(p.weak_areas[:3])}")
        print()

        # ── Step 4: 面试主循环 ────────────────────────────
        self.state.transition_phase(InterviewPhase.QUESTIONING, AgentRole.ORCHESTRATOR)
        print(f"面试开始，最多 {self.state.config.max_questions} 道题。\n")
        print("-" * 60)

        while not self.state.is_finished:
            # 4a. 收集上一轮各 Agent 发出的所有信号
            pending_messages = self.state.consume_messages()

            # 4b. 更新候选人画像（用于 Planner/Reflector）
            self._update_candidate_model()

            # 4c. LLM 路由决策
            decision = await self._decide_routing(pending_messages)
            self._log_decision(decision)

            # 4d. 执行决策
            if decision.action in ("wrap_up", "finish"):
                reason_label = "提前结束" if decision.action == "wrap_up" else "正常结束"
                print(f"\n📊 [{reason_label}] {decision.reason}")
                self.state.transition_phase(InterviewPhase.WRAPPING_UP, AgentRole.ORCHESTRATOR)
                await self._generate_report()
                return

            # 强制保护：超出最大题数
            if self.state.question_count >= self.state.config.max_questions:
                self.state.transition_phase(InterviewPhase.WRAPPING_UP, AgentRole.ORCHESTRATOR)
                await self._generate_report()
                return

            # 4e. 选题（根据决策传递参数）
            force_topic   = decision.topic if decision.action == "force_topic" else None
            deepen        = decision.action == "deepen"
            deepen_reason = decision.reason if deepen else ""

            ok = self.question_agent.select(
                self.state,
                force_topic=force_topic,
                deepen=deepen,
                deepen_reason=deepen_reason,
            )
            if not ok:
                print("\n题库已用完，面试结束。")
                break

            # 4f. 展示题目，收集回答
            ctx = self.state.current_context
            q   = ctx.question
            q_num = self.state.question_count + 1
            print(f"\n【第 {q_num} 题】[{q.topic.value}] [{q.difficulty.value}]")
            if decision.action in ("deepen", "force_topic"):
                print(f"   → {decision.reason}")
            print(f"\n{q.text}")
            if q.hint:
                print(f"\n（提示：{q.hint}）")
            print()

            answer = input("候选人回答（直接回车跳过）：\n> ").strip()
            ctx.answer = answer

            # 4g. EvaluatorAgent 评估（含追问）
            print("\n⚙️  正在评估...", end="", flush=True)
            await self.evaluator_agent.run(self.state, self._ask_follow_up)

            # 4h. 展示本题反馈
            if self.state.completed_records:
                last = self.state.completed_records[-1]
                e = last.evaluation
                print(f"\n💬 评分：{e.effective_score:.1f}/10  深度：{e.depth_label}")
                print(f"   {e.brief_feedback}")
            print("-" * 60)

            # 4i. ReflectionAgent 更新洞察
            self.reflector.reflect()

        # ── Step 5: 正常退出循环后生成报告 ────────────────
        self.state.transition_phase(InterviewPhase.WRAPPING_UP, AgentRole.ORCHESTRATOR)
        await self._generate_report()

    # ─────────────────────────────────────────────────────
    # LLM 路由决策
    # ─────────────────────────────────────────────────────

    async def _decide_routing(self, messages: list[AgentMessage]) -> RoutingDecision:
        """
        调用 LLM 根据当前状态和信号消息，决定下一步路由。
        LLM 失败时回退到规则引擎（_fallback_routing）。
        """
        state = self.state

        # 最少题数未达到时，强制继续
        if state.question_count < state.config.min_questions:
            # 但仍可以响应 DEEPEN / FORCE_TOPIC
            for msg in messages:
                if msg.signal == AgentSignal.DEEPEN:
                    return RoutingDecision(
                        action="deepen",
                        reason=msg.payload.get("reason", "该方向需要深入考察"),
                    )
                if msg.signal == AgentSignal.FORCE_TOPIC:
                    return RoutingDecision(
                        action="force_topic",
                        topic=msg.payload.get("topic"),
                        reason="切换到指定方向",
                    )
            return RoutingDecision(action="continue", reason="题目数量未达最少要求，继续出题")

        # 构建 Prompt 上下文
        signals_text = self._format_signals(messages)
        planner_plan = self.planner.plan_next_phase()
        reflection   = self.reflector.reflect() if state.completed_records else {"action": "继续观察"}

        all_topics   = set(t.value for t in state.config.topic_weights.keys())
        covered      = set(state.topic_counts.keys())
        uncovered    = sorted(all_topics - covered)

        prompt = _ROUTING_TMPL.format(
            completed     = state.question_count,
            min_q         = state.config.min_questions,
            max_q         = state.config.max_questions,
            difficulty    = state.current_difficulty.value,
            duration      = state.duration_minutes,
            avg_score     = state.average_score,
            recent_count  = len(state.recent_scores),
            recent_scores = [round(s, 1) for s in state.recent_scores],
            topic_scores  = state.scores_by_topic,
            weak_topics   = state.weak_topics or "（无）",
            uncovered_topics = uncovered or "（已全覆盖）",
            signals       = signals_text,
            planner_plan  = json.dumps(planner_plan, ensure_ascii=False),
            reflection    = json.dumps(reflection,   ensure_ascii=False),
        )

        try:
            agent = create_agent(
                model=self.model,
                system_prompt=_ROUTING_SYSTEM
            )
            # resp = self.client.messages.create(
            #     model=self.model,
            #     max_tokens=300,
            #     system=_ROUTING_SYSTEM,
            #     messages=[{"role": "user", "content": prompt}],
            # )

            response = agent.invoke({
                "messages": [HumanMessage(prompt)]
            })
            raw = response['messages'][-1].content.strip()
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            data    = json.loads(cleaned)
            return RoutingDecision(
                action=data.get("action", "continue"),
                topic =data.get("topic"),
                reason=data.get("reason", ""),
            )
        except Exception as e:
            # LLM 调用失败，回退规则引擎
            return self._fallback_routing(messages)

    def _fallback_routing(self, messages: list[AgentMessage]) -> RoutingDecision:
        """规则兜底：LLM 不可用时的确定性决策。"""
        state = self.state

        if state.question_count >= state.config.max_questions:
            return RoutingDecision(action="finish", reason="已达最大题目数")

        for msg in messages:
            if msg.signal == AgentSignal.FINISH:
                return RoutingDecision(action="finish", reason="EvaluatorAgent 建议结束")
            if msg.signal == AgentSignal.WRAP_UP and state.question_count >= state.config.min_questions:
                return RoutingDecision(action="wrap_up", reason=f"平均分 {state.average_score:.1f}，建议提前结束")
            if msg.signal == AgentSignal.DEEPEN:
                return RoutingDecision(
                    action="deepen",
                    reason=msg.payload.get("reason", "该方向需要深入"),
                )
            if msg.signal == AgentSignal.FORCE_TOPIC:
                return RoutingDecision(
                    action="force_topic",
                    topic=msg.payload.get("topic"),
                    reason="切换 Topic",
                )

        if state.weak_topics and state.question_count >= 3:
            return RoutingDecision(action="deepen", reason=f"薄弱方向：{state.weak_topics[0]}")

        return RoutingDecision(action="continue", reason="正常继续")

    # ─────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────

    def _update_candidate_model(self):
        """将最近一题的评分同步到 CandidateModel（供 Planner/Reflector 使用）。"""
        if not self.state.completed_records:
            return
        last = self.state.completed_records[-1]
        e    = last.evaluation
        topic_key = last.question.topic.name.split("_")[0].capitalize()

        # 构造 CandidateModel 期望的格式
        skill_map = {
            "Prompt": "Prompt",
            "Rag":    "RAG",
            "Agent":  "Agent",
            "System": "SystemDesign",
        }
        skill = skill_map.get(topic_key, "Prompt")
        self.candidate_model.update({
            "score":  e.effective_score,
            "skills": {skill: e.effective_score},
        })

    def _format_signals(self, messages: list[AgentMessage]) -> str:
        if not messages:
            return "（无信号）"
        lines = []
        for m in messages:
            payload_str = json.dumps(m.payload, ensure_ascii=False) if m.payload else ""
            lines.append(f"  [{m.from_agent.value}] {m.signal.value}  {payload_str}")
        return "\n".join(lines)

    def _log_decision(self, decision: RoutingDecision):
        """在终端打印 Orchestrator 的决策（调试可见）。"""
        icons = {
            "continue":    "▶",
            "deepen":      "🔽",
            "force_topic": "🎯",
            "wrap_up":     "⛔",
            "finish":      "✅",
        }
        icon = icons.get(decision.action, "•")
        topic_str = f" → {decision.topic}" if decision.topic else ""
        print(f"\n🤖 Orchestrator 决策：{icon} {decision.action}{topic_str}")
        if decision.reason:
            print(f"   理由：{decision.reason}")

    def _collect_inputs(self) -> tuple[str, int, Optional[str], Optional[str]]:
        """采集候选人信息（纯 IO，无业务逻辑）。"""
        name = input("请输入候选人姓名：").strip() or "候选人"
        years_str = input("请输入工作年限（年）：").strip()
        try:
            years = int(years_str)
        except ValueError:
            years = 3

        print("\n是否提供 JD（岗位描述）？直接回车跳过。")
        raw_jd = self._read_multiline("JD（输入空行结束）：")

        print("\n是否提供简历？直接回车跳过。")
        raw_resume = self._read_multiline("简历（输入空行结束）：")

        return name, years, raw_jd or None, raw_resume or None

    @staticmethod
    def _read_multiline(prompt: str) -> str:
        lines = []
        line = input(prompt)
        while line:
            lines.append(line)
            line = input()
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────
    # IO 回调（注入 EvaluatorAgent）
    # ─────────────────────────────────────────────────────

    async def _ask_follow_up(self, follow_up_text: str) -> str:
        print(f"\n🔎 追问：{follow_up_text}")
        return input("> ").strip()

    # ─────────────────────────────────────────────────────
    # 报告生成
    # ─────────────────────────────────────────────────────

    async def _generate_report(self):
        if not self.state.completed_records:
            print("\n未完成任何题目，无法生成报告。")
            self.state.transition_phase(InterviewPhase.FINISHED, AgentRole.ORCHESTRATOR)
            return

        print("\n\n📝 正在生成面试报告...", flush=True)
        report = self.reporter_agent.generate(self.state)
        self.state.transition_phase(InterviewPhase.FINISHED, AgentRole.ORCHESTRATOR)

        print("\n" + "=" * 60)
        print(f"  面试报告 — {report.candidate_name}")
        print("=" * 60)
        print(f"综合得分：{report.total_score:.1f} / 10")
        print(f"录用建议：{report.hire_recommendation}")
        if report.jd_match_score is not None:
            print(f"JD 匹配度：{report.jd_match_score:.1f} / 10")
        print(f"\n总评：{report.summary}")
        if report.strengths:
            print("\n✅ 优势：")
            for s in report.strengths:
                print(f"   • {s}")
        if report.risks:
            print("\n⚠️  风险点：")
            for r in report.risks:
                print(f"   • {r}")
        if report.next_round_questions:
            print("\n❓ 下一轮追问建议：")
            for q in report.next_round_questions:
                print(f"   • {q}")
        print("\n各方向得分：")
        for topic, score in report.dimension_scores.items():
            print(f"   {topic}: {score:.1f}")
        print(f"\n总题数：{self.state.question_count}  时长：{self.state.duration_minutes:.0f} 分钟")
        print("=" * 60)

    """
    Streamlit 适配的 Orchestrator 方法扩展
    将这些方法添加到原始的 OrchestratorAgent 类中
    """

    async def _run_initialization_streamlit(self, name: str, years: int, jd: str, resume: str):
        """Streamlit 版本的初始化方法"""
        # 初始化状态
        self.state = InterviewState()
        self.state.profile = CandidateProfile(name=name, years_of_experience=years)
        self.state.raw_jd = jd
        self.state.raw_resume = resume
        self.planner.state = self.state
        self.reflector.state = self.state

        # 分析画像
        self.state.transition_phase(InterviewPhase.PROFILING, AgentRole.ORCHESTRATOR)
        self.resume_agent.analyze(self.state)

        # 开始面试
        self.state.transition_phase(InterviewPhase.QUESTIONING, AgentRole.ORCHESTRATOR)

        # 生成第一道题
        self.question_agent.select(self.state)

        return self.state