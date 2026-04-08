# 🚦 Intelligent Multi-Junction Traffic Control (OpenEnv Compatible)

## 🌟 Overview

This project presents an **AI-ready traffic simulation environment** built for **Reinforcement Learning (RL)** and **Agent-based evaluation systems**.

Unlike traditional static traffic models, this system dynamically simulates **multi-junction traffic flow** and enables intelligent decision-making through reward-based optimization.

The environment is fully **OpenEnv compliant**, **Dockerized**, and designed to be evaluated by **LLM-based agents**.

---

## 🧠 Key Features

### 🚥 Multi-Junction Traffic System

* Simulates **two independent intersections (Junction A & B)**
* Each junction contains 4 directional lanes (N, S, E, W)
* Realistic traffic fluctuations over time

---

### 🚑 Emergency Vehicle Handling

* Ambulance can appear at any junction
* Environment rewards prioritizing emergency routes
* Encourages real-world critical decision-making

---

### 🎯 Reinforcement Learning Ready

* State → traffic distribution across junctions
* Action → select signal direction (e.g., `A_E`, `B_N`)
* Reward → optimized based on congestion + emergency priority

---

### ⚙️ OpenEnv API Design

* `POST /reset` → initializes environment
* `POST /step` → applies action and returns reward
* Fully compatible with automated agent evaluation pipelines

---

### 🐳 Dockerized Deployment

* Runs in isolated environment
* Compatible with Hugging Face Spaces
* Meets evaluation infrastructure requirements

---

## 🏗️ Project Structure

```bash
traffic-rl-system/
│
├── main.py             # FastAPI server (API endpoints)
├── env.py              # Traffic environment logic
├── inference.py        # Agent interaction script
├── openenv.yaml        # OpenEnv specification
├── Dockerfile          # Container configuration
├── requirements.txt    # Dependencies
├── README.md           # Documentation
```

---

## 🚀 Getting Started

### 🔹 Run Locally

```bash
pip install fastapi uvicorn numpy
python -m uvicorn main:app --reload --port 7860
```

Open:
👉 http://localhost:7860/docs

---

### 🔹 Run with Docker

```bash
docker build -t traffic-env .
docker run -p 7860:7860 traffic-env
```

---

## 🔄 API Usage

### 🔹 Reset Environment

```bash
curl -X POST http://localhost:7860/reset
```

---

### 🔹 Take Action

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action":"A_E"}'
```

---

## 📊 State Representation

```json
{
  "junction_A": { "N": 5, "S": 3, "E": 7, "W": 2 },
  "junction_B": { "N": 6, "S": 4, "E": 1, "W": 8 },
  "ambulance": "A"
}
```

---

## 🎯 Reward Logic

* ✅ Reduces congestion across lanes
* 🚑 Prioritizes ambulance movement
* ⚖️ Balances overall traffic efficiency

---

## 🧪 Evaluation Compatibility

This environment is designed to pass:

* ✔ OpenEnv validation
* ✔ Docker build checks
* ✔ Automated agent evaluation
* ✔ Structured inference logging

---

## 🔮 Future Improvements

* Multi-agent coordination across junctions
* Real-time traffic data integration
* Deep Reinforcement Learning (DQN, PPO)
* Visualization dashboard

---

## 👨‍💻 Author

**Shyam Kumar**
GitHub: https://github.com/Msk1819

---

## 📄 License

This project is for research and educational purposes.
