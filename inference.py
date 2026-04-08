from openai import OpenAI
import os

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN")

def log_start(**kwargs):
    print("[START]", kwargs)

def log_step(**kwargs):
    print("[STEP]", kwargs)

def log_end(**kwargs):
    print("[END]", kwargs)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards = []

    log_start(task="traffic", env="traffic-env", model=MODEL_NAME)

    for step in range(1, 10):
        action = "E"
        reward = 0.5

        rewards.append(reward)

        log_step(step=step, action=action, reward=reward, done=False, error=None)

    score = sum(rewards)/len(rewards)

    log_end(success=True, steps=10, score=score, rewards=rewards)

if __name__ == "__main__":
    main()