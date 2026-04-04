import asyncio
from my_env.env import EmailEnv
from my_env.models import Action

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")

from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

def get_action(email_text):
    prompt = f"""
    You are an email assistant.

    Email:
    {email_text}

    Return:
    classification (spam or important)
    priority (low, medium, high)
    response (short reply)

    Format:
    classification,priority,response
    """

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    text = response.choices[0].message.content.strip()

    try:
        cls, pr, resp = text.split(",", 2)
    except:
        cls, pr, resp = "important", "medium", "Reply to this email"

    return Action(
        classification=cls.strip(),
        priority=pr.strip(),
        response=resp.strip()
    ) 

async def main():
    env = EmailEnv()

    obs = env.reset()
    rewards = []

    log_start("email-task", "email-env", "baseline-model")

    for step in range(1, 4):
        action = get_action(obs.email_text)

        result = env.step(action)

        rewards.append(result.reward)

        log_step(step, str(action), result.reward, result.done)

        if result.done:
            break

    success = sum(rewards) > 0.5
    log_end(success, step, rewards)

if __name__ == "__main__":
    asyncio.run(main())