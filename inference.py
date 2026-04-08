from openai import OpenAI
import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://msk1819-traffic-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "default")
API_KEY = os.getenv("HF_TOKEN", "")

def log_start(**kwargs):
    print("[START]", kwargs)

def log_step(**kwargs):
    print("[STEP]", kwargs)

def log_end(**kwargs):
    print("[END]", kwargs)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards = []
    steps_taken = 0

    log_start(task="traffic", env="traffic-env", model=MODEL_NAME)

    # 🔥 RESET ENV
    response = requests.post(f"{API_BASE_URL}/reset")
    data = response.json()

    for step in range(1, 10):
        if data.get("done"):
            break

        # simple action (can be improved)
        action = "A_E"

        # 🔥 STEP ENV
        result = requests.post(
            f"{API_BASE_URL}/step",
            json={"action": action}
        ).json()

        reward = result.get("reward", 0.0)
        done = result.get("done", False)

        rewards.append(reward)
        steps_taken = step

        log_step(
            step=step,
            action=action,
            reward=reward,
            done=done,
            error=None
        )

        data = result

        if done:
            break

    score = sum(rewards) / len(rewards) if rewards else 0.0

    log_end(
        success=True,
        steps=steps_taken,
        score=score,
        rewards=rewards
    )

if __name__ == "__main__":
    main()
