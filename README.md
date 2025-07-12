# DQN Maze Agent 🧠

This project implements a Deep Q-Network (DQN) agent trained to solve a custom 2D continuous maze environment using PyTorch and Gymnasium.

The goal is for the agent to navigate from a fixed starting point to a goal position while avoiding static walls and danger zones.

---

## 🛠️ Installation

1. **Clone this repository:**
```bash
git clone https://github.com/rakshithravi10/DQN-Maze-Agent-.git
cd DQN-Maze-Agent-
```

2. **Create and activate a virtual environment (optional but recommended):**
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

3. **Install the required dependencies:**
```bash
pip install torch numpy matplotlib pygame gymnasium
```

---

## 🚀 How to Run

### 1. **Train and Test the Agent**
```bash
python main.py
```
This will:
- Train the agent using DQN
- Plot and save the training reward curve
- Save the final model to `dqn.pth`
- Evaluate the agent using the trained model

### 2. **Visualize Trained Agent**
```bash
python visualise_agent.py
```
This will:
- Load `dqn.pth`
- Render the agent navigating the maze visually using PyGame

---

## 🧪 Environment Description

- State space: `(x, y)` continuous position ∈ [0, 1]^2
- Action space: Discrete → {up, down, left, right}
- Rewards:
  - Reaching goal: **+50**
  - Danger zone: **−10** and episode ends
  - Wall collision: **−1**, stays in place
  - Step cost: **−0.01**
  - Shaping: distance-based penalty

---

## 📈 Training Results

- Epsilon starts at 1.0 and decays to 0.1
- Agent considered trained when it reaches the goal for **100 consecutive episodes**
- Sample reward curve is saved 

---

## 📌 Notes

- This project was developed as part of a graded assignment for the PADM course.
- Designed for low-resource CPUs 

---

## 📄 License

This project is for academic and educational use.

---

## 👤 Author

Rakshith Ravi  
Master's Student – AI Engineering of Autonomous Systems @ THI

---

For any questions or collaborations, feel free to connect via [LinkedIn](https://www.linkedin.com/in/rakshithravi10).

