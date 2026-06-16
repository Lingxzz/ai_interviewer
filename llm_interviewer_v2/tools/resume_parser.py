# utils/resume_parser.py
import json
import os
import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

logger = logging.getLogger(__name__)


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    调用 DeepSeek API（非流式）
    使用与项目中相同的 Agent 调用方式
    """
    model = init_chat_model(
        model=os.getenv('DEEPSEEK_MODEL'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL'),
        model_provider='openai',
        temperature=temperature,
        extra_body={'thinking': {'type': 'disabled'}}
    )

    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
    )
    resp = agent.invoke({
        'messages': [HumanMessage(content=user_prompt)],
    })
    return resp['messages'][-1].content


def extract_info_from_resume(resume_text: str) -> Dict[str, Any]:
    """
    从简历文本中提取姓名、工作年限、核心技能

    Args:
        resume_text: 简历纯文本内容（建议限制在 3000 字符内）

    Returns:
        {
            "name": str,        # 候选人姓名
            "years": int,       # 工作年限
            "skills": List[str] # 核心技术栈（最多5个）
        }
    """
    system = (
        "你是一个专业的简历解析助手。请从以下简历文本中提取关键信息：\n"
        "1. 候选人姓名（中文或英文，只返回最可能的一个，如果实在找不到返回空字符串）\n"
        "2. 工作年限（整数，仅数字，如果找不到则返回 0）\n"
        "3. 核心技术栈（列表，最多 5 个关键词，如：Python, PyTorch, RAG, 微调, 系统设计）\n"
        "请严格按照以下 JSON 格式返回，不要有任何额外解释或其他字段：\n"
        '{"name": "张三", "years": 5, "skills": ["Python", "PyTorch", "RAG"]}'
    )

    # 限制简历长度，避免超出上下文
    user = f"简历文本：\n{resume_text[:3000]}"

    try:
        result_str = _call_llm(system, user, temperature=0.1)  # 低温度提高准确性
        logger.info(f"LLM 提取结果：{result_str[:200]}")

        # 提取 JSON（可能被 markdown 包裹）
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0]
        elif "```" in result_str:
            result_str = result_str.split("```")[1].split("```")[0]

        data = json.loads(result_str.strip())
        return {
            "name": data.get("name", ""),
            "years": int(data.get("years", 0)),
            "skills": data.get("skills", [])
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败：{e}\n原始返回：{result_str}")
        return {"name": "", "years": 0, "skills": []}
    except Exception as e:
        logger.exception(f"提取信息失败：{e}")
        return {"name": "", "years": 0, "skills": []}


def generate_jd_from_resume(resume_text: str, skills: List[str] = None) -> str:
    """
    根据简历内容和技能生成量身定制的 JD（岗位描述）

    Args:
        resume_text: 简历纯文本内容（建议限制在 2000 字符内）
        skills: 核心技术栈列表（可选，如果提供则会重点突出）

    Returns:
        生成的 JD 文本
    """
    skill_text = ", ".join(skills) if skills else "（从简历中自动识别）"

    system = (
        "你是一位资深招聘专家。请根据候选人简历，为「LLM 工程师」岗位生成一份定制化的 JD。\n"
        "JD 应包括：\n"
        "- 岗位职责（3-4 条）\n"
        "- 任职要求（3-4 条，要结合候选人的技能亮点）\n"
        "- 加分项（1-2 条）\n"
        "语言要求：简洁专业，不要超过 400 字，适合直接用于招聘。"
    )

    user = f"""
    候选人核心技能：{skill_text}

    候选人简历摘要（前 1500 字）：
    {resume_text[:1500]}

    请为这位候选人量身定制 JD：
    """

    try:
        jd_text = _call_llm(system, user, temperature=0.3)  # 稍高温度增加多样性
        return jd_text.strip()
    except Exception as e:
        logger.exception(f"生成 JD 失败：{e}")
        return ""


def extract_resume_summary(resume_text: str) -> str:
    """
    提取简历摘要（用于面试初始化）

    Args:
        resume_text: 简历纯文本内容

    Returns:
        200 字以内的简历摘要
    """
    system = (
        "你是一个专业的简历分析师。请用 150-200 字概括候选人的：\n"
        "1. 核心技能栈\n"
        "2. 主要工作经历（最近 2 段）\n"
        "3. 技术亮点或项目亮点\n"
        "语言简洁，突出重点，不要有废话。"
    )

    user = f"简历文本：\n{resume_text[:2000]}"

    try:
        summary = _call_llm(system, user, temperature=0.2)
        return summary.strip()
    except Exception as e:
        logger.exception(f"生成简历摘要失败：{e}")
        return resume_text[:200]  # 降级：返回前 200 字符