"""
RAGStore — 项目统一向量存储模块。

三个集合（Collection）：
  - questions   : 题库语义索引，供 QuestionAgent 按候选人简历检索相关题目
  - rubrics     : 题目评分标准索引，供 EvaluatorAgent 检索参考答案增强评分
  - jd_profiles : JD 画像索引，供 ResumeAgent 检索相似 JD 辅助权重判断

使用 Chroma 内存模式（ephemeral），进程重启后重建，无需持久化文件。
Embedding 使用 Chroma 内置的 DefaultEmbeddingFunction（all-MiniLM-L6-v2 ONNX），
无需额外安装 sentence-transformers。
"""

from __future__ import annotations

import threading
from typing import Optional

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    _CHROMADB_AVAILABLE = True

except Exception as e:
    print(f"Chroma load failed: {e}")

    chromadb = None
    DefaultEmbeddingFunction = None

    _CHROMADB_AVAILABLE = False

from core.models import Question, Topic


# ─────────────────────────────────────────────────────────
# 单例
# ─────────────────────────────────────────────────────────

_lock = threading.Lock()
_instance: Optional["RAGStore"] = None


def get_rag_store() -> "RAGStore":
    """全局单例，线程安全。"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RAGStore()
    return _instance


# ─────────────────────────────────────────────────────────
# RAGStore
# ─────────────────────────────────────────────────────────

class RAGStore:
    """
    封装三个 Chroma collection 的读写接口。
    应用启动时调用 build_from_bank() 一次性建立索引。
    """

    def __init__(self):
        if not _CHROMADB_AVAILABLE:
            self._client = None
            self._ef = None
            self._questions = None
            self._rubrics = None
            self._jd_profiles = None
            self._built = False
            return

        self._client = chromadb.Client()  # 内存模式，进程内共享
        self._ef = DefaultEmbeddingFunction()

        self._questions = self._client.get_or_create_collection(
            name="questions",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._rubrics = self._client.get_or_create_collection(
            name="rubrics",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._jd_profiles = self._client.get_or_create_collection(
            name="jd_profiles",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._built = False

    # ─────────────────────────────────────────────────────
    # 建立索引
    # ─────────────────────────────────────────────────────

    def build_from_bank(self, questions: list[Question]) -> None:
        """
        将题库全量写入 questions 和 rubrics 两个集合。
        幂等：已建立则跳过。chromadb 未安装时静默跳过。
        """
        if not _CHROMADB_AVAILABLE or self._built:
            return

        q_ids, q_docs, q_metas = [], [], []
        r_ids, r_docs, r_metas = [], [], []

        for q in questions:
            # questions 集合：用题目正文 + 关键词做语义索引文档
            q_doc = f"{q.text} 关键词：{' '.join(q.expected_keywords)}"
            q_ids.append(q.id)
            q_docs.append(q_doc)
            q_metas.append({
                "topic": q.topic.value,
                "difficulty": q.difficulty.value,
                "text": q.text,
                "hint": q.hint or "",
                "keywords": ",".join(q.expected_keywords),
                "rubric": q.scoring_rubric,
                "follow_up": q.follow_up_template or "",
            })

            # rubrics 集合：用评分标准做索引，EvaluatorAgent 检索参考
            r_doc = f"题目：{q.text}\n评分标准：{q.scoring_rubric}\n关键词：{' '.join(q.expected_keywords)}"
            r_ids.append(f"rubric_{q.id}")
            r_docs.append(r_doc)
            r_metas.append({
                "question_id": q.id,
                "topic": q.topic.value,
                "question_text": q.text,
                "rubric": q.scoring_rubric,
                "keywords": ",".join(q.expected_keywords),
            })

        if q_ids:
            self._questions.upsert(ids=q_ids, documents=q_docs, metadatas=q_metas)
        if r_ids:
            self._rubrics.upsert(ids=r_ids, documents=r_docs, metadatas=r_metas)

        self._built = True

    def add_jd_profile(self, jd_id: str, jd_text: str, metadata: dict) -> None:
        if not _CHROMADB_AVAILABLE or not self._jd_profiles:
            return
        """
        写入一条 JD 画像。ResumeAgent 分析完后调用，积累历史。
        metadata 应包含：topic_weights, required_skills, difficulty 等。
        """
        self._jd_profiles.upsert(
            ids=[jd_id],
            documents=[jd_text],
            metadatas=[metadata],
        )

    # ─────────────────────────────────────────────────────
    # 检索接口
    # ─────────────────────────────────────────────────────

    def retrieve_questions(
        self,
        query: str,
        topic_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        exclude_ids: Optional[list[str]] = None,
        n: int = 5,
    ) -> list[dict]:
        """
        按语义检索相关题目。

        Args:
            query:            检索文本（简历摘要 / 技能描述 / 当前上下文）
            topic_filter:     限定 topic（题目的 topic.value）
            difficulty_filter:限定难度
            exclude_ids:      已出过的题目 id，过滤掉
            n:                返回数量

        Returns:
            list of metadata dict，按相关度从高到低。
        """
        where: dict = {}
        if topic_filter and difficulty_filter:
            where = {"$and": [{"topic": topic_filter}, {"difficulty": difficulty_filter}]}
        elif topic_filter:
            where = {"topic": topic_filter}
        elif difficulty_filter:
            where = {"difficulty": difficulty_filter}

        if not _CHROMADB_AVAILABLE or not self._questions:
            return []
        try:
            results = self._questions.query(
                query_texts=[query],
                n_results=min(n * 3, self._questions.count()),  # 多取一些再过滤
                where=where if where else None,
                include=["metadatas", "distances"],
            )
        except Exception:
            return []

        items = []
        seen_ids = set(exclude_ids or [])
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for qid, meta, dist in zip(ids, metadatas, distances):
            if qid in seen_ids:
                continue
            items.append({**meta, "id": qid, "similarity": round(1 - dist, 4)})
            if len(items) >= n:
                break

        return items

    def retrieve_rubrics(
        self,
        question_text: str,
        answer_text: str,
        topic: Optional[str] = None,
        n: int = 3,
    ) -> list[dict]:
        """
        根据题目 + 候选人答案，检索相似题目的评分标准。
        供 EvaluatorAgent 作为评分参考上下文。

        Returns:
            list of {"question_text", "rubric", "keywords", "topic", "similarity"}
        """
        query = f"题目：{question_text}\n候选人答案要点：{answer_text[:200]}"
        where = {"topic": topic} if topic else None

        if not _CHROMADB_AVAILABLE or not self._rubrics:
            return []
        try:
            count = self._rubrics.count()
            if count == 0:
                return []
            results = self._rubrics.query(
                query_texts=[query],
                n_results=min(n, count),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception:
            return []

        items = []
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for meta, dist in zip(metadatas, distances):
            items.append({**meta, "similarity": round(1 - dist, 4)})
        return items

    def retrieve_similar_jds(self, jd_text: str, n: int = 3) -> list[dict]:
        """
        检索历史中相似的 JD 画像，辅助 ResumeAgent 确定 topic 权重。

        Returns:
            list of metadata dict（含 topic_weights 等字段）
        """
        if not _CHROMADB_AVAILABLE or not self._jd_profiles:
            return []
        try:
            count = self._jd_profiles.count()
            if count == 0:
                return []
            results = self._jd_profiles.query(
                query_texts=[jd_text],
                n_results=min(n, count),
                include=["metadatas", "distances"],
            )
        except Exception:
            return []

        items = []
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for meta, dist in zip(metadatas, distances):
            items.append({**meta, "similarity": round(1 - dist, 4)})
        return items

    @property
    def is_built(self) -> bool:
        return self._built

    def stats(self) -> dict:
        if not _CHROMADB_AVAILABLE or not self._questions:
            return {"questions": 0, "rubrics": 0, "jd_profiles": 0, "available": False}
        return {
            "questions": self._questions.count(),
            "rubrics": self._rubrics.count(),
            "jd_profiles": self._jd_profiles.count(),
            "available": True,
        }
