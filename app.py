import gradio as gr
from my_env.env import EmailEnv
from my_env.models import Action

# Simple rule-based model (safe fallback)
def process_email(email_text):
    env = EmailEnv()

    # Properly initialize environment
    obs = env.reset()

    # Inject user email into observation/state
    obs.email_text = email_text

    text = email_text.lower()

    if any(word in text for word in ["free", "win", "offer", "lottery"]):
        action = Action("spam", "low", "Ignore this spam email")
    elif "urgent" in text or "asap" in text:
        action = Action("important", "high", "Reply immediately")
    else:
        action = Action("important", "medium", "Reply to this email")

    #Pass action to environment
    result = env.step(action)

    return (
        action.classification,
        action.priority,
        action.response,
        result.reward
    )

# UI
interface = gr.Interface(
    fn=process_email,
    inputs=gr.Textbox(label="Enter Email Text"),
    outputs=[
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Priority"),
        gr.Textbox(label="Response"),
        gr.Number(label="Reward")
    ],
    title="📧 Email Triage AI (OpenEnv)",
    description="Classify emails and generate responses using AI environment"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
    