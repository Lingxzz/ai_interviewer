# agents/planner_agent.py

class PlannerAgent:
    """
    根据候选人画像、历史题目、信号决定下一步面试策略
    """
    def __init__(self, candidate_model, state):
        self.candidate_model = candidate_model
        self.state = state

    def plan_next_phase(self):
        """
        决定下一阶段目标和难度
        """
        weak_skills = self.candidate_model.get_weak_skills()
        if weak_skills:
            next_skill = weak_skills[0]
            phase = {
                "phase": f"Deep Dive {next_skill}",
                "goal": f"Verify {next_skill} competency",
                "difficulty": "SENIOR" if self.candidate_model.skills[next_skill] >= 6 else "MID"
            }
        else:
            phase = {
                "phase": "System Design Wrap-up",
                "goal": "Verify overall system design understanding",
                "difficulty": "SENIOR"
            }
        return phase
