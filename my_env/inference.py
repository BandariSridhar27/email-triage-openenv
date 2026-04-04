import asyncio
import time
import os
from my_env.env import EmailEnv
from my_env.models import Action

# ---------------- LOGGING ----------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")

# ---------------- LLM SETUP ----------------
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN
    )
else:
    client = None

# ---------------- ACTION LOGIC ----------------
def get_action(email_text):
    text = email_text.lower()

    # 🔹 Rule-based fallback (VERY IMPORTANT)
    if not client:
        if any(word in text for word in ["free", "win", "offer", "lottery"]):
            return Action("spam", "low", "Ignore this spam email")
        elif "urgent" in text or "asap" in text:
            return Action("important", "high", "Reply immediately")
        else:
            return Action("important", "medium", "Reply to this email")

    # 🔹 LLM-based logic
    try:
        prompt = f"""
        You are an email assistant.

        Email:
        {email_text}

        Return:
        classification (spam or important),
        priority (low, medium, high),
        response

        Format:
        classification,priority,response
        """

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        text = response.choices[0].message.content.strip()
        cls, pr, resp = text.split(",", 2)

        return Action(cls.strip(), pr.strip(), resp.strip())

    except Exception as e:
        print("LLM error:", e)

        # 🔹 fallback if API fails
        return Action("important", "medium", "Reply to this email")

# ---------------- MAIN LOOP ----------------
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

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("Runtime error:", e)

    # 🔥 KEEP CONTAINER ALIVE (FINAL FIX)
    print("Container running...")

    while True:
        time.sleep(60)