# agents/candidate_model.py

class CandidateModel:
    """
    用于维护候选人在面试过程中的技能画像和能力盲点
    """
    def __init__(self, skills=None):
        # 初始化技能分数
        self.skills = skills or {
            "Prompt": 0,
            "RAG": 0,
            "Agent": 0,
            "SystemDesign": 0
        }
        self.history = []  # 保存每轮评分记录

    def update(self, evaluation):
        """
        更新候选人技能画像
        evaluation: dict, 包含 {"score":float, "skills":{skill_name:score}}
        """
        skill_scores = evaluation.get("skills", {})
        for skill, score in skill_scores.items():
            if skill in self.skills:
                # 简单加权平均更新
                prev = self.skills[skill]
                self.skills[skill] = (prev * len(self.history) + score) / (len(self.history)+1)
        self.history.append(evaluation)

    def get_weak_skills(self, threshold=6.5):
        """
        返回低于阈值的技能列表
        """
        return [skill for skill, score in self.skills.items() if score < threshold]

    def get_profile(self):
        return self.skills.copy()