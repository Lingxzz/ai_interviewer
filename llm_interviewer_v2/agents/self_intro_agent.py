"""
SelfIntroAgent — 负责候选人自我介绍环节的对话与评分。

职责：
- 根据候选人自我介绍内容，生成针对性追问（逐轮）
- 判断追问是否已充分，生成结束信号
- 对整个自我介绍环节打分并输出评估结论

不负责：
- UI 交互（streamlit_app 负责）
- 技术面试（QuestionAgent / EvaluatorAgent 负责）
"""

from __future__ import annotations

import json
import os
import re
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

_SYSTEM = """\
你是一位资深面试官，正在对候选人进行面试前的自我介绍环节评估。

你的任务：
1. 候选人完成自我介绍后，根据内容提出 2-3 个有针对性的追问（每次只问一个）
2. 追问聚焦：工作经历亮点的细节验证、技术选型原因、挑战与收获
3. 所有追问完成后，输出评分 JSON

输出规则（严格遵守）：
- 若还有问题要问：直接用自然语言提问，不要输出 JSON
- 若追问已完成（收到 2-3 轮回答后）：输出如下 JSON（用 ```json 包裹）：
{
  "score": <0-10 浮点数>,
  "evaluation": "<2-3句综合评价，指出亮点和不足>",
  "done": true
}

评分参考：
- 9-10：表达清晰、经历真实有深度、主动提 trade-off
- 7-8：表达流畅、经历较完整、有一定深度
- 5-6：表达一般、经历较浅、缺乏主动思考
- 1-4：表达混乱、内容空洞或明显虚假
"""

_FIRST_TURN_TMPL = """\
候选人姓名：{name}
工作年限：{years} 年
岗位方向：LLM 应用开发工程师

候选人自我介绍如下：
---
{intro}
---

请根据以上自我介绍，提出第一个追问。"""

_CONTINUE_TMPL = """\
候选人姓名：{name}，{years} 年工作经验

【自我介绍原文】
{intro}

【已有问答记录】
{qa_history}

请继续追问，或在已有 2-3 轮回答后输出评分 JSON。"""


# ─────────────────────────────────────────────────────────
# 返回值结构
# ─────────────────────────────────────────────────────────

@dataclass
class SelfIntroReply:
    text: str                      # 面试官回复（追问或空字符串）
    done: bool = False             # True 表示本环节结束
    score: Optional[float] = None  # done=True 时有值
    evaluation: str = ""           # done=True 时有值


class SelfIntroAgent:
    """
    自我介绍环节 Agent。
    每次调用 reply() 输入当前完整对话上下文，返回下一步动作。
    """

    def __init__(self):
        self.model = init_chat_model(
            model=os.getenv('DEEPSEEK_MODEL'),
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL'),
            model_provider='openai',
            extra_body={"thinking": {"type": "disabled"}}
        )

    def first_question(self, name: str, years: int, intro: str) -> SelfIntroReply:
        """
        候选人提交自我介绍后，生成第一个追问。
        """
        prompt = _FIRST_TURN_TMPL.format(name=name, years=years, intro=intro)
        raw = self._call_llm(prompt)
        return self._parse_reply(raw)

    def next_reply(
        self,
        name: str,
        years: int,
        intro: str,
        qa_pairs: list[dict],   # [{"question": str, "answer": str}, ...]
    ) -> SelfIntroReply:
        """
        候选人回答追问后，决定继续追问还是结束并打分。
        qa_pairs: 已完成的追问-回答对列表。
        """
        qa_history = "\n\n".join(
            f"追问{i+1}：{p['question']}\n回答{i+1}：{p['answer']}"
            for i, p in enumerate(qa_pairs)
        )
        prompt = _CONTINUE_TMPL.format(
            name=name, years=years, intro=intro, qa_history=qa_history
        )
        raw = self._call_llm(prompt)
        return self._parse_reply(raw)

    # ─────────────────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            agent = create_agent(model=self.model, system_prompt=_SYSTEM)
            response = agent.invoke({"messages": [HumanMessage(prompt)]})
            return response['messages'][-1].content.strip()
        except Exception as e:
            return f"（生成失败：{e}）"

    def _parse_reply(self, raw: str) -> SelfIntroReply:
        """
        尝试从回复中解析 JSON 结论；若无 JSON，则视为追问文本。
        """
        json_match = re.search(r"```json\s*([\s\S]*?)```", raw)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if data.get("done"):
                    return SelfIntroReply(
                        text="",
                        done=True,
                        score=float(data.get("score", 7.0)),
                        evaluation=data.get("evaluation", ""),
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        # 无 JSON → 继续追问
        return SelfIntroReply(text=raw, done=False)
