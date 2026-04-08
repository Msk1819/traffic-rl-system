import random
import copy

class TrafficEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.junction_A = {d: random.randint(0, 10) for d in ["N", "S", "E", "W"]}
        self.junction_B = {d: random.randint(0, 10) for d in ["N", "S", "E", "W"]}
        self.ambulance = random.choice(["A", "B"])
        self.step_count = 0

        # track previous congestion (for adaptive reward)
        self.prev_total_wait = self._total_wait()

        return self._get_obs()

    def _get_obs(self):
        return {
            "junction_A": self.junction_A,
            "junction_B": self.junction_B,
            "ambulance": self.ambulance,
            "step": self.step_count
        }

    def _total_wait(self):
        return sum(self.junction_A.values()) + sum(self.junction_B.values())

    # 🚀 Traffic prediction (simple but effective)
    def _predict_next(self, junction):
        predicted = {}
        for d in junction:
            predicted[d] = junction[d] + 1  # expected incoming avg
        return predicted

    # 🚀 Apply action
    def _update_junction(self, junction, direction):
        cleared = 0

        if direction == "NS":
            cleared = min(junction["N"] + junction["S"], 6)
            junction["N"] = max(0, junction["N"] - 3)
            junction["S"] = max(0, junction["S"] - 3)
        else:
            cleared = min(junction["E"] + junction["W"], 6)
            junction["E"] = max(0, junction["E"] - 3)
            junction["W"] = max(0, junction["W"] - 3)

        # controlled traffic inflow
        for d in junction:
            junction[d] += random.randint(0, 2)

        return cleared

    def step(self, action):
        self.step_count += 1

        junction = action[0]   # A or B
        direction = action.split("_")[1]

        cleared_A = 0
        cleared_B = 0

        # 🚀 Apply action
        if junction == "A":
            cleared_A = self._update_junction(self.junction_A, direction)
        else:
            cleared_B = self._update_junction(self.junction_B, direction)

        # 🧠 Multi-agent coordination insight
        load_A = sum(self.junction_A.values())
        load_B = sum(self.junction_B.values())

        # 🚀 Prediction
        pred_A = sum(self._predict_next(self.junction_A).values())
        pred_B = sum(self._predict_next(self.junction_B).values())

        # 🔥 ADAPTIVE REWARD
        reward = 0

        # 1. traffic clearing
        reward += cleared_A * 0.7
        reward += cleared_B * 0.7

        # 2. congestion penalty
        total_wait = self._total_wait()
        reward -= total_wait * 0.04

        # 3. improvement bonus (adaptive)
        if total_wait < self.prev_total_wait:
            reward += 2  # improvement reward
        else:
            reward -= 1

        # 4. ambulance priority
        if self.ambulance == junction:
            reward += 4
        else:
            reward -= 2

        # 5. strategic decision (multi-agent coordination)
        if load_A > load_B and junction == "A":
            reward += 1.5
        if load_B > load_A and junction == "B":
            reward += 1.5

        # 6. prediction-based reward
        if pred_A > pred_B and junction == "A":
            reward += 1
        if pred_B > pred_A and junction == "B":
            reward += 1

        # update memory
        self.prev_total_wait = total_wait

        # ambulance moves
        if self.step_count % 5 == 0:
            self.ambulance = random.choice(["A", "B"])

        done = self.step_count >= 60

        return self._get_obs(), round(reward, 2), done
