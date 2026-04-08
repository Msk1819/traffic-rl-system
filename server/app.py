from fastapi import FastAPI
from pydantic import BaseModel
from env import TrafficEnv

# Initialize FastAPI app
app = FastAPI()

# Initialize your environment
env = TrafficEnv()


# -----------------------------
# Input Model (VERY IMPORTANT)
# -----------------------------
class ActionInput(BaseModel):
    action: str


# -----------------------------
# RESET ENDPOINT
# -----------------------------
@app.post("/reset")
def reset():
    state = env.reset()

    return {
        "observation": state,
        "done": False
    }


# -----------------------------
# STEP ENDPOINT
# -----------------------------
@app.post("/step")
def step(action: ActionInput):
    state, reward, done = env.step(action.action)

    return {
        "observation": state,
        "reward": float(reward),
        "done": done
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
