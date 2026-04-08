import random

class TrafficEnv:
    def __init__(self):
        self.state = {}

    def reset(self):
        self.state = {
            "junction_A": {
                "N": random.randint(0, 10),
                "S": random.randint(0, 10),
                "E": random.randint(0, 10),
                "W": random.randint(0, 10)
            },
            "junction_B": {
                "N": random.randint(0, 10),
                "S": random.randint(0, 10),
                "E": random.randint(0, 10),
                "W": random.randint(0, 10)
            },
            "ambulance": random.choice(["A", "B", None])  # ambulance location
        }

        return self.state

    def step(self, action):
        """
        action format: "A_E", "B_N", etc.
        (junction + direction)
        """

        reward = 0.0

        try:
            junction, direction = action.split("_")
        except:
            junction, direction = "A", "E"  # fallback

        # Get selected junction
        junction_key = "junction_A" if junction == "A" else "junction_B"

        # Get traffic at that direction
        traffic = self.state[junction_key].get(direction, 0)

        # --------------------------
        # REWARD LOGIC
        # --------------------------

        # Reward for clearing traffic
        reward += traffic * 0.1

        # Penalize congestion overall
        total_traffic = sum(self.state["junction_A"].values()) + sum(self.state["junction_B"].values())
        reward -= total_traffic * 0.02

        # 🚑 Emergency priority
        if self.state["ambulance"] == junction:
            reward += 2.0  # strong bonus

        # --------------------------
        # STATE UPDATE (simulate flow)
        # --------------------------

        for j in ["junction_A", "junction_B"]:
            for d in ["N", "S", "E", "W"]:
                # random fluctuation
                self.state[j][d] = max(0, self.state[j][d] + random.randint(-2, 3))

        # Random ambulance movement
        self.state["ambulance"] = random.choice(["A", "B", None])

        done = False

        return self.state, reward, done