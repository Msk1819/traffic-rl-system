from openai import OpenAI
import os
import requests

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

ENV_URL = "https://msk1819-traffic-env.hf.space"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def log_start(**kwargs):
    print("[START]", kwargs)

def log_step(**kwargs):
    print("[STEP]", kwargs)

def log_end(**kwargs):
    print("[END]", kwargs)

def get_action(observation):
    prompt = f"""
You are a traffic control AI.

Observation:
{observation}

Choose best action:
Options = ["A_NS", "A_EW", "B_NS", "B_EW"]

Return only action.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def main():
    log_start(task="traffic", env="traffic-env", model=MODEL_NAME)

    rewards = []

    # RESET ENV
    res = requests.post(f"{ENV_URL}/reset").json()
    observation = res["observation"]

    for step in range(10):
        action = get_action(observation)

        res = requests.post(
            f"{ENV_URL}/step",
            json={"action": action}
        ).json()

        observation = res["observation"]
        reward = res["reward"]
        done = res["done"]

        rewards.append(reward)

        log_step(
            step=step,
            action=action,
            reward=reward,
            done=done,
            error=None
        )

        if done:
            break

    score = sum(rewards) / len(rewards)

    log_end(
        success=True,
        steps=len(rewards),
        score=score,
        rewards=rewards
    )

if __name__ == "__main__":
    main()
