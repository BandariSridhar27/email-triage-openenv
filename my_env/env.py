import random
from .models import Observation, Action, StepResult
from .data import EMAILS

class EmailEnv:

    def __init__(self):
        self.email = None
        self.step_count = 0
        self.done = False
        self.history = []

    def reset(self):
        self.email = random.choice(EMAILS)
        self.step_count = 0
        self.done = False
        self.history = []

        return Observation(
            email_text=self.email["text"],
            step_count=self.step_count,
            history=[]
        )

    def step(self, action: Action):
        if self.done:
            raise Exception("Episode already finished")

        self.step_count += 1
        reward = 0.0

        # reward shaping
        if action.classification == self.email["label"]:
            reward += 0.3
        else:
            reward -= 0.1   # penalty

        if action.priority == self.email["priority"]:
            reward += 0.3

        if len(action.response) > 10:
            reward += 0.2

        # smart response reward
        if "ignore" in action.response.lower() and self.email["label"] == "spam":
            reward += 0.2

        self.history.append(
            f"{action.classification} | {action.priority} | {action.response}"
            )

        # multi-step logic
        if self.step_count >= 2:
            self.done = True

        return StepResult(
            observation=Observation(
                email_text=self.email["text"],
                step_count=self.step_count,
                history=self.history
            ),
            reward=max(0.0, min(1.0, reward)),  # clamp 0–1
            done=self.done,
            info={}
        )

    def state(self):
        return self.email

    async def close(self):
        pass