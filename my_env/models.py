from pydantic import BaseModel

class Observation(BaseModel):
    email_text: str
    step_count: int
    history: list[str]

class Action(BaseModel):
    classification: str  # spam / important
    priority: str        # low / medium / high
    response: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict