import os
import json
import time

# -------------------------------
# CONFIG (Safe environment usage)
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "traffic-rl-advanced")

print(f"[INFO] Using model: {MODEL_NAME}")


# -------------------------------
# MODEL
# -------------------------------
class TrafficModel:
    def __init__(self, model_name):
        self.model_name = model_name

        # Track waiting time (for fairness)
        self.wait_time = {"N": 0, "S": 0, "E": 0, "W": 0}

        print(f"[INFO] Model '{model_name}' initialized")

    def update_wait_times(self, chosen_lane, lanes):
        """
        Increase wait time for non-selected lanes
        Reset chosen lane wait
        """
        for lane in lanes:
            if lane == chosen_lane:
                self.wait_time[lane] = 0
            else:
                self.wait_time[lane] += 1

    def predict(self, state):
        try:
            lanes = state.get("lanes", {})
            emergency = state.get("emergency", None)

            if not lanes:
                return "N"

            # 🚑 1. Emergency override (highest priority)
            if emergency and emergency in lanes:
                print(f"[INFO] Emergency detected at {emergency}")
                self.update_wait_times(emergency, lanes)
                return emergency

            # 🧠 2. RL-like scoring
            scores = {}

            for lane, cars in lanes.items():
                traffic_score = cars * 2              # more cars = more priority
                wait_score = self.wait_time[lane] * 3  # fairness boost
                total_score = traffic_score + wait_score

                scores[lane] = total_score

            # 🎯 3. Choose best lane
            best_lane = max(scores, key=scores.get)

            # 🔄 4. Update wait times
            self.update_wait_times(best_lane, lanes)

            print(f"[DEBUG] Scores: {scores}")
            print(f"[DEBUG] Wait Times: {self.wait_time}")

            return best_lane

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "N"


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    try:
        return TrafficModel(MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None


# -------------------------------
# INFERENCE
# -------------------------------
def run_inference(model, input_data):
    try:
        state = input_data.get("state", {})
        action = model.predict(state)

        return {
            "action": action,
            "status": "success"
        }

    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        return {
            "action": "N",
            "status": "failed"
        }


# -------------------------------
# MAIN
# -------------------------------
def main():
    try:
        model = load_model()
        if model is None:
            raise Exception("Model not loaded")

        # Read input (HF validator style)
        try:
            input_str = input()
            input_data = json.loads(input_str)
        except Exception:
            # fallback test input
            input_data = {
                "state": {
                    "lanes": {"N": 2, "S": 3, "E": 1, "W": 4},
                    "emergency": None
                }
            }

        output = run_inference(model, input_data)

        print(json.dumps(output))

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        print(json.dumps({
            "action": "N",
            "status": "error"
        }))


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()
