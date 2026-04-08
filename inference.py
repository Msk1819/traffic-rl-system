import os
import json

# -------------------------------
# CONFIG (Safe environment usage)
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "traffic-rl-default")

print(f"[INFO] Using model: {MODEL_NAME}")


# -------------------------------
# DUMMY MODEL (Replace later)
# -------------------------------
class TrafficModel:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"[INFO] Model '{model_name}' initialized")

    def predict(self, state):
        """
        state: dict with traffic info
        return: action (which signal to give green)
        """
        # Example logic (safe fallback)
        try:
            lanes = state.get("lanes", {})
            if not lanes:
                return "N"  # default

            # Choose lane with max cars
            return max(lanes, key=lanes.get)

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "N"


# -------------------------------
# LOAD MODEL (Safe)
# -------------------------------
def load_model():
    try:
        model = TrafficModel(MODEL_NAME)
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return None


# -------------------------------
# INFERENCE FUNCTION
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
# MAIN ENTRY (IMPORTANT)
# -------------------------------
def main():
    try:
        # Load model
        model = load_model()
        if model is None:
            raise Exception("Model not loaded")

        # Read input from STDIN (HF validator sends input like this)
        try:
            input_str = input()
            input_data = json.loads(input_str)
        except Exception:
            # fallback if no input provided
            input_data = {
                "state": {
                    "lanes": {"N": 2, "S": 3, "E": 1, "W": 4}
                }
            }

        # Run inference
        output = run_inference(model, input_data)

        # Print output as JSON
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
