# 🛡️ Cyber-RL: Autonomous Cybersecurity Agent (PPO)

An advanced Reinforcement Learning pipeline designed to train an autonomous agent to detect and mitigate cyber attacks (Brute Force, DDoS, Malware) in real-time using **Proximal Policy Optimization (PPO)**.

---

## 📊 Project Architecture

This project uses a **Data-Driven Gymnasium Environment** that simulates network traffic patterns. The agent learns to map system metrics to defensive actions.

### 🧠 The RL Model (PPO)
- **State Space (5 Inputs):** Failed Logins, Alert Count, DDoS Flag, Suspicious Flag, System Health.
- **Action Space (4 Outputs):** 
  - `Block IP` (Defense against Brute Force)
  - `Alert Admin` (Defense against Malware)
  - `Isolate System` (Defense against DDoS)
  - `Ignore` (For Normal Traffic)
- **Framework:** Stable-Baselines3, Gymnasium, PyTorch.

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install gymnasium stable-baselines3[extra] tensorboard matplotlib pandas numpy torch
```

### 2. Prepare the Data
Generate a synthetic dataset of 1,000,000 network traffic samples:
```bash
python generate_dataset.py
```

### 3. Train the Agent
Train the PPO model for 500,000 timesteps. This will normalize observations and rewards for stability.
```bash
python train.py
```
*Generated Files:* `cyber_rl_model.zip` (Model), `vec_normalize.pkl` (Stats), `reward_plot.png` (Graph).

### 4. Automated Evaluation
Run a deterministic evaluation over 10 episodes to see the model's performance on unseen data.
```bash
python evaluate.py
```

### 5. 🛠️ Manual Testing & Accuracy Check
Test the model manually with your own custom data to see how it reacts to specific threats.
```bash
python manual_check.py
```
**Features of Manual Check:**
- **Confidence Scoring:** Shows how "sure" the model is (e.g., 98.5% confidence).
- **Probability Distribution:** Shows the likelihood for all 4 defensive actions.
- **Session Summary:** Tracks your manual test accuracy vs. the dataset baseline.

---

## 📁 File Directory

| File | Description |
| :--- | :--- |
| `cyber_env.py` | Custom Gymnasium environment logic and reward system. |
| `generate_dataset.py` | Script to create realistic mock cyber traffic data. |
| `train.py` | Training pipeline with PPO and reward normalization. |
| `evaluate.py` | Batch testing script for overall accuracy. |
| `manual_check.py` | Interactive CLI tool for manual data input and testing. |
| `cyber_traffic_dataset.csv` | The generated dataset (80% Normal, 20% Attacks). |
| `logs/` | Contains model checkpoints saved during training. |

---

## 📈 Monitoring with TensorBoard
To view detailed learning curves, loss functions, and entropy, run:
```bash
tensorboard --logdir ./ppo_logs/
```
Then open `http://localhost:6006` in your browser.

---

## ⚠️ Mitigation Logic Reference
| Attack Type | Correct Action | Impact if Ignored |
| :--- | :--- | :--- |
| **Normal** | `Ignore` | False Positive Penalty |
| **Brute Force** | `Block IP` | System Health -2.0 |
| **DDoS** | `Isolate System` | System Health -2.0 |
| **Malware** | `Alert Admin` | System Health -2.0 |

---
*Developed for Autonomous Cyber Defense Research.*
