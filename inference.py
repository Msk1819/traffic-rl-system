import os
import time
import random
from openai import OpenAI

# -------------------------------
# ENV (SAFE + FALLBACK)
# -------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# -------------------------------
# LLM CLIENT (ALWAYS INIT)
# -------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# -------------------------------
# FORCE LLM CALL (CRITICAL)
# -------------------------------
def test_llm_connection():
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=2
        )
        print("[LLM_CALL_SUCCESS]", flush=True)
    except Exception as e:
        print(f"[LLM_CALL_FAILED] {e}", flush=True)


# -------------------------------
# TRAFFIC ENV
# -------------------------------
class TrafficEnv:
    def __init__(self):
        self.lanes = [random.randint(0, 5) for _ in range(4)]
        self.step_count = 0

    def step(self, action):
        self.step_count += 1

        passed = min(self.lanes[action], random.randint(1, 3))
        self.lanes[action] -= passed

        self.lanes = [lane + random.randint(0, 2) for lane in self.lanes]

        reward = -sum(self.lanes)
        done = self.step_count >= 5

        return self.lanes, reward, done


# -------------------------------
# LLM DECISION
# -------------------------------
def get_action_from_llm(state):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You control traffic lights."},
                {"role": "user", "content": f"Traffic state: {state}. Choose lane 0-3."}
            ],
            max_tokens=5
        )

        text = response.choices[0].message.content.strip()

        action = int(''.join(filter(str.isdigit, text)) or 0)
        return action % 4

    except Exception as e:
        print(f"[LLM_ERROR] {e}", flush=True)
        return random.randint(0, 3)


# -------------------------------
# MAIN
# -------------------------------
def run():
    # 🔥 GUARANTEED API CALL ATTEMPT
    test_llm_connection()

    env = TrafficEnv()
    total_reward = 0
    step = 0

    print("[START] task=traffic_control", flush=True)

    while True:
        step += 1

        action = get_action_from_llm(env.lanes)
        state, reward, done = env.step(action)

        total_reward += reward

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

        time.sleep(0.05)

    score = total_reward / step if step else 0

    print(f"[END] task=traffic_control score={score:.4f} steps={step}", flush=True)


# -------------------------------
# ENTRY
# -------------------------------
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[FATAL_ERROR] {e}", flush=True)
