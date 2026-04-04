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

def get_action(email_text):
    if "free" in email_text.lower() or "win" in email_text.lower():
        return Action(
            classification="spam",
            priority="low",
            response="Ignore this spam email"
        )
    else:
        return Action(
            classification="important",
            priority="high",
            response="Reply and acknowledge the email"
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