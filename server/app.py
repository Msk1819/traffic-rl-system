from fastapi import FastAPI
from pydantic import BaseModel
from env import TrafficEnv

app = FastAPI()
env = TrafficEnv()

class ActionInput(BaseModel):
    action: str

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs, "done": False}

@app.post("/step")
def step(input: ActionInput):
    obs, reward, done = env.step(input.action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
