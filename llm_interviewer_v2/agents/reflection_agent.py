# agents/reflection_agent.py

class ReflectionAgent:
    """
    负责对已完成的题目和评分进行反思，给出下一步建议
    """
    def __init__(self, candidate_model, state):
        self.candidate_model = candidate_model
        self.state = state
        self._reflection_notes = []

    def reflect(self):
        """
        根据历史评估和状态，分析能力盲区，提出行动建议
        """
        weak_skills = self.candidate_model.get_weak_skills()
        reflection = {
            "verified": [skill for skill, score in self.candidate_model.skills.items() if score >= 6.5],
            "weak_skills": weak_skills,
            "action": None
        }

        if weak_skills:
            reflection["action"] = f"Focus on {weak_skills[0]}"
        else:
            reflection["action"] = "Consider wrapping up interview"

        self._reflection_notes.append(reflection)
        return reflection
