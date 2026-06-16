"""
题库模块 — 预设面试题目。
"""

from core.models import Difficulty, Question, Topic

# ─────────────────────────────────────────────────────────
# Prompt 工程
# ─────────────────────────────────────────────────────────

PROMPT_QUESTIONS = [
    Question(
        id="prompt_001",
        topic=Topic.PROMPT,
        difficulty=Difficulty.JUNIOR,
        text="什么是 few-shot prompting？它和 zero-shot prompting 有什么区别？请举例说明。",
        hint="可以从示例数量和模型适应性角度来思考",
        expected_keywords=["示例", "上下文学习", "zero-shot", "few-shot", "in-context"],
        scoring_rubric="2分：知道概念区别；4分：能举例；6分：理解背后原理；8分：能讨论何时选择哪种；10分：能分析其局限性和改进方法",
        follow_up_template="你提到了{keyword}，能展开说说在什么场景下 few-shot 效果明显优于 zero-shot 吗？",
    ),
    Question(
        id="prompt_002",
        topic=Topic.PROMPT,
        difficulty=Difficulty.MID,
        text="Chain-of-Thought (CoT) prompting 的核心原理是什么？什么情况下它会失效？",
        hint="思考为什么让模型「展示推理步骤」会提升表现",
        expected_keywords=["推理链", "中间步骤", "复杂推理", "幻觉", "算术", "逻辑"],
        scoring_rubric="2分：知道CoT让模型分步骤回答；4分：理解其提升推理的机制；6分：知道CoT的适用场景；8分：能分析失效情况；10分：了解Self-consistency等改进方法",
        follow_up_template="你提到{keyword}，那 Self-consistency CoT 是如何解决这个问题的？",
    ),
    Question(
        id="prompt_003",
        topic=Topic.PROMPT,
        difficulty=Difficulty.SENIOR,
        text="在生产环境中，你如何管理和版本控制 prompt？如何系统地进行 prompt 优化迭代？",
        hint="考虑工程化视角：测试、监控、回滚",
        expected_keywords=["版本控制", "A/B测试", "评估集", "回归测试", "监控", "迭代"],
        scoring_rubric="4分：有版本控制意识；6分：有测试评估流程；8分：有完整的迭代框架；10分：能讨论自动化 prompt 优化",
        follow_up_template="你提到{keyword}，实际落地时遇到过哪些困难，是如何解决的？",
    ),
    Question(
        id="prompt_004",
        topic=Topic.PROMPT,
        difficulty=Difficulty.JUNIOR,
        text="请解释什么是 prompt injection 攻击，以及有哪些常见的防御手段？",
        hint="考虑攻击者如何通过用户输入篡改模型行为",
        expected_keywords=["注入", "越狱", "输入验证", "系统提示", "隔离", "防御"],
        scoring_rubric="2分：了解基本概念；4分：能举出具体攻击例子；6分：知道几种防御手段；8分：理解防御的局限性；10分：能设计完整的防御体系",
        follow_up_template=None,
    ),
]

# ─────────────────────────────────────────────────────────
# RAG 与知识库
# ─────────────────────────────────────────────────────────

RAG_QUESTIONS = [
    Question(
        id="rag_001",
        topic=Topic.RAG,
        difficulty=Difficulty.JUNIOR,
        text="请描述一个基础 RAG 系统的完整流程，包括离线构建和在线查询两个阶段。",
        hint="从文档处理到最终回答，逐步拆解",
        expected_keywords=["向量化", "分块", "索引", "检索", "生成", "embedding", "召回"],
        scoring_rubric="2分：知道RAG基本概念；4分：能描述完整流程；6分：理解各步骤的作用；8分：能讨论各环节的优化点；10分：能分析不同场景下的架构选型",
        follow_up_template="你提到了{keyword}，chunk 的大小如何选择？有什么 trade-off？",
    ),
    Question(
        id="rag_002",
        topic=Topic.RAG,
        difficulty=Difficulty.MID,
        text="RAG 系统中检索质量差是一个常见问题。你会从哪些维度来诊断和改进检索效果？",
        hint="检索可以从召回率、精准率、排序等多个角度优化",
        expected_keywords=["召回率", "精准率", "重排序", "混合检索", "BM25", "向量检索", "query改写"],
        scoring_rubric="4分：知道基本评估指标；6分：能提出多种优化策略；8分：有系统化的诊断方法；10分：能结合业务场景做针对性优化",
        follow_up_template="你提到{keyword}，能具体说说如何实现 query 改写吗？",
    ),
    Question(
        id="rag_003",
        topic=Topic.RAG,
        difficulty=Difficulty.SENIOR,
        text="如何评估一个 RAG 系统的整体效果？设计一套评估方案。",
        hint="端到端评估 vs 分模块评估，自动化 vs 人工",
        expected_keywords=["RAGAS", "忠实度", "相关性", "上下文精准率", "幻觉检测", "golden set"],
        scoring_rubric="4分：知道需要评估检索和生成两部分；6分：能说出具体指标；8分：有完整评估流程设计；10分：了解 RAGAS 等框架并能讨论其局限性",
        follow_up_template="你提到{keyword}，golden set 应该如何构建和维护？",
    ),
    Question(
        id="rag_004",
        topic=Topic.RAG,
        difficulty=Difficulty.MID,
        text="什么是 HyDE（Hypothetical Document Embeddings）？它解决了什么问题？",
        hint="思考用户 query 和文档之间的语义鸿沟",
        expected_keywords=["假设文档", "语义对齐", "query扩展", "检索增强", "向量空间"],
        scoring_rubric="2分：听说过；4分：能说出基本原理；6分：理解它解决的核心问题；8分：能分析适用场景和局限；10分：能与其他 query 增强方法比较",
        follow_up_template=None,
    ),
    Question(
        id="rag_005",
        topic=Topic.RAG,
        difficulty=Difficulty.STAFF,
        text="在超大规模知识库（亿级文档）场景下，RAG 系统的架构设计需要考虑哪些特殊挑战？",
        hint="检索延迟、存储成本、更新频率、多租户隔离",
        expected_keywords=["HNSW", "分片", "缓存", "近似最近邻", "延迟", "多租户", "增量更新"],
        scoring_rubric="6分：能识别核心挑战；8分：有具体的工程解决方案；10分：能做系统性权衡分析",
        follow_up_template="你提到{keyword}，向量数据库的选型上你会考虑哪些因素？",
    ),
]

# ─────────────────────────────────────────────────────────
# Agent 与工具调用
# ─────────────────────────────────────────────────────────

AGENT_QUESTIONS = [
    Question(
        id="agent_001",
        topic=Topic.AGENT,
        difficulty=Difficulty.JUNIOR,
        text="请解释 LLM Agent 中 ReAct（Reasoning + Acting）框架的工作原理。",
        hint="思考 Thought、Action、Observation 三个循环步骤",
        expected_keywords=["推理", "行动", "观察", "工具调用", "循环", "Thought", "Action"],
        scoring_rubric="2分：知道基本概念；4分：能描述三步循环；6分：理解其优势；8分：知道局限性；10分：能与其他框架（如 Plan-and-Execute）比较",
        follow_up_template="你提到{keyword}，ReAct 在什么情况下会陷入无限循环或错误推理？",
    ),
    Question(
        id="agent_002",
        topic=Topic.AGENT,
        difficulty=Difficulty.MID,
        text="如何设计可靠的工具调用（Function Calling）系统？需要考虑哪些错误处理机制？",
        hint="工具可能失败、返回格式不对、超时等情况",
        expected_keywords=["错误处理", "重试", "超时", "参数验证", "fallback", "幂等", "schema"],
        scoring_rubric="4分：知道基本工具设计；6分：有错误处理意识；8分：有完整的健壮性方案；10分：考虑到边界情况和安全性",
        follow_up_template="你提到{keyword}，如何防止 Agent 调用工具产生不可逆的副作用？",
    ),
    Question(
        id="agent_003",
        topic=Topic.AGENT,
        difficulty=Difficulty.SENIOR,
        text="多 Agent 协作系统中，如何设计 Agent 间的通信和任务分配机制？",
        hint="考虑编排模式、消息传递、状态共享",
        expected_keywords=["编排", "消息队列", "状态机", "任务分解", "并行", "协调者", "共享状态"],
        scoring_rubric="4分：知道多Agent概念；6分：能说出几种协作模式；8分：能分析不同模式的trade-off；10分：有实际设计经验或能给出完整架构",
        follow_up_template="你提到{keyword}，如何处理多 Agent 之间的死锁或任务冲突？",
    ),
    Question(
        id="agent_004",
        topic=Topic.AGENT,
        difficulty=Difficulty.MID,
        text="LLM Agent 在长任务执行中常见哪些失败模式？你会如何提高任务完成率？",
        hint="上下文窗口限制、幻觉、工具调用失败",
        expected_keywords=["上下文压缩", "检查点", "记忆管理", "任务分解", "幻觉", "监控"],
        scoring_rubric="4分：能识别常见失败；6分：有针对性的解决方案；8分：有系统化的可靠性设计；10分：考虑到监控和可观测性",
        follow_up_template=None,
    ),
]

# ─────────────────────────────────────────────────────────
# 效果评估
# ─────────────────────────────────────────────────────────

EVALUATION_QUESTIONS = [
    Question(
        id="eval_001",
        topic=Topic.EVALUATION,
        difficulty=Difficulty.MID,
        text="如何构建一个 LLM 应用的自动化评估流水线？从数据集设计到指标计算。",
        hint="考虑 golden set 构建、自动打分、与人工评估的对齐",
        expected_keywords=["golden set", "LLM-as-judge", "人工评估", "一致性", "指标", "基准"],
        scoring_rubric="4分：有基本评估意识；6分：能描述评估流程；8分：能讨论自动评估的局限；10分：有完整的评估体系设计",
        follow_up_template="你提到{keyword}，LLM-as-judge 有哪些已知的偏见，如何缓解？",
    ),
    Question(
        id="eval_002",
        topic=Topic.EVALUATION,
        difficulty=Difficulty.SENIOR,
        text="你的 LLM 应用上线后，如何持续监控模型性能退化（model drift）？",
        hint="线上监控 vs 离线评估，实时信号 vs 周期性评估",
        expected_keywords=["drift", "监控", "分布变化", "A/B测试", "影子模式", "告警", "反馈"],
        scoring_rubric="4分：知道drift问题；6分：有监控方案；8分：有完整的检测和响应机制；10分：能设计自动化的质量保障体系",
        follow_up_template=None,
    ),
]

# ─────────────────────────────────────────────────────────
# 系统设计与架构
# ─────────────────────────────────────────────────────────

SYSTEM_QUESTIONS = [
    Question(
        id="sys_001",
        topic=Topic.SYSTEM,
        difficulty=Difficulty.SENIOR,
        text="设计一个支持 10 万 QPS 的 LLM 推理服务，你会如何做架构设计？",
        hint="负载均衡、批处理、缓存、降级",
        expected_keywords=["批处理", "负载均衡", "缓存", "流式", "降级", "限流", "副本"],
        scoring_rubric="4分：有基本扩展思路；6分：能识别关键瓶颈；8分：有完整的高可用方案；10分：能做细致的容量规划和成本估算",
        follow_up_template="你提到{keyword}，prompt 缓存（prefix caching）在这个场景中能带来多大收益？",
    ),
    Question(
        id="sys_002",
        topic=Topic.SYSTEM,
        difficulty=Difficulty.MID,
        text="在 LLM 应用中，如何实现有效的流式输出（streaming）？前后端各需要注意什么？",
        hint="SSE、WebSocket、背压处理",
        expected_keywords=["SSE", "WebSocket", "流式", "背压", "token", "CORS", "连接管理"],
        scoring_rubric="2分：知道streaming概念；4分：知道SSE/WebSocket；6分：理解实现细节；8分：考虑到错误处理和重连；10分：有完整的生产级实现经验",
        follow_up_template=None,
    ),
]

# ─────────────────────────────────────────────────────────
# 代码实现
# ─────────────────────────────────────────────────────────

CODING_QUESTIONS = [
    Question(
        id="code_001",
        topic=Topic.CODING,
        difficulty=Difficulty.MID,
        text="请用 Python 实现一个简单的带记忆的对话管理器，支持上下文窗口截断。",
        hint="考虑 token 计数、滑动窗口、系统提示保留",
        expected_keywords=["token计数", "截断", "系统提示", "历史消息", "窗口", "tiktoken"],
        scoring_rubric="4分：能写出基本对话管理；6分：有token截断逻辑；8分：考虑系统提示保留和边界情况；10分：有完整的错误处理和可配置性",
        follow_up_template=None,
    ),
]

# ─────────────────────────────────────────────────────────
# 成本与可观测性
# ─────────────────────────────────────────────────────────

COST_QUESTIONS = [
    Question(
        id="cost_001",
        topic=Topic.COST,
        difficulty=Difficulty.MID,
        text="如何在不明显降低用户体验的前提下，将 LLM API 调用成本降低 50%？",
        hint="缓存、模型路由、压缩、批处理",
        expected_keywords=["缓存", "模型路由", "小模型", "prompt压缩", "批处理", "token优化"],
        scoring_rubric="4分：知道几种降本手段；6分：能分析各自的效果和代价；8分：有完整的降本策略组合；10分：能定量估算各措施的收益",
        follow_up_template="你提到{keyword}，语义缓存（semantic cache）如何判断命中？有什么风险？",
    ),
    Question(
        id="cost_002",
        topic=Topic.COST,
        difficulty=Difficulty.SENIOR,
        text="如何为 LLM 应用建立完整的可观测性（Observability）体系？",
        hint="Trace、Metrics、Logs 三支柱在 LLM 场景中的特殊考量",
        expected_keywords=["trace", "span", "token用量", "延迟", "错误率", "LangSmith", "OpenTelemetry"],
        scoring_rubric="4分：知道基本可观测性概念；6分：能应用到LLM场景；8分：有完整的监控体系设计；10分：有实际落地经验",
        follow_up_template=None,
    ),
]

# ─────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────

QUESTIONS_BY_TOPIC = {
    Topic.PROMPT: PROMPT_QUESTIONS,
    Topic.RAG: RAG_QUESTIONS,
    Topic.AGENT: AGENT_QUESTIONS,
    Topic.EVALUATION: EVALUATION_QUESTIONS,
    Topic.SYSTEM: SYSTEM_QUESTIONS,
    Topic.CODING: CODING_QUESTIONS,
    Topic.COST: COST_QUESTIONS,
}

ALL_QUESTIONS: list[Question] = [
    q for questions in QUESTIONS_BY_TOPIC.values() for q in questions
]
