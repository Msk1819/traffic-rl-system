import os
import time
import random

# -------------------------------
# Safe environment handling
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "traffic-rl-default")

# -------------------------------
# Dummy Traffic Environment
# (Replace with your real env later)
# -------------------------------
class TrafficEnv:
    def __init__(self):
        self.lanes = [0, 0, 0, 0]  # N, S, E, W
        self.step_count = 0

    def reset(self):
        self.lanes = [random.randint(0, 5) for _ in range(4)]
        self.step_count = 0
        return self.lanes

    def step(self, action):
        """
        action: 0=N, 1=S, 2=E, 3=W
        """
        self.step_count += 1

        # Cars pass in selected lane
        passed = min(self.lanes[action], random.randint(1, 3))
        self.lanes[action] -= passed

        # New cars arrive randomly
        self.lanes = [lane + random.randint(0, 2) for lane in self.lanes]

        # Reward = negative congestion
        reward = -sum(self.lanes)

        done = self.step_count >= 10
        return self.lanes, reward, done


# -------------------------------
# Dummy Agent (Replace with your RL model)
# -------------------------------
class Agent:
    def select_action(self, state):
        return random.randint(0, 3)


# -------------------------------
# Main Simulation
# -------------------------------
def run_simulation():
    env = TrafficEnv()
    agent = Agent()

    state = env.reset()
    total_reward = 0
    step = 0

    print(f"[START] task=traffic_control model={MODEL_NAME}", flush=True)

    while True:
        step += 1

        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        total_reward += reward

        print(f"[STEP] step={step} reward={reward}", flush=True)

        state = next_state

        if done:
            break

        time.sleep(0.05)  # small delay (safe)

    score = total_reward / step if step > 0 else 0

    print(
        f"[END] task=traffic_control score={score:.4f} steps={step}",
        flush=True
    )


# -------------------------------
# Entry Point (Crash-safe)
# -------------------------------
def main():
    try:
        run_simulation()
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)


if __name__ == "__main__":
    main()
