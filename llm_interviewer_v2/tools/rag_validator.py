"""
RAGValidator — 对 RAGStore 的健康检查和功能验证。
"""
from tools.rag_store import get_rag_store
from questions.bank import ALL_QUESTIONS


class RAGValidator:
    def validate(self, description: str = "") -> tuple[bool, str]:
        """
        验证 RAG 索引是否正常。
        Returns: (ok: bool, message: str)
        """
        rag = get_rag_store()
        if not rag.is_built:
            rag.build_from_bank(ALL_QUESTIONS)

        stats = rag.stats()
        if stats["questions"] == 0:
            return False, "题库索引为空，RAG 功能不可用"

        # 做一次探针检索
        hits = rag.retrieve_questions(
            query=description or "RAG 系统设计向量检索",
            n=2,
        )
        if not hits:
            return False, f"索引已建立（{stats}），但检索无结果，请检查 embedding 函数"

        return True, f"RAG 正常：{stats}，探针检索返回 {len(hits)} 条"
