# 🚀 Deep Q-Network for Lunar Lander

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![OpenAI Gym](https://img.shields.io/badge/Gym-0.21%2B-green)](https://gym.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A from-scratch implementation of Deep Q-Learning that teaches a lander to safely touch down on the lunar surface! This project was built as part of my reinforcement learning journey.

## 🎮 Environment

The agent controls a lunar lander in the [OpenAI Gym LunarLander-v2](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) environment:

- **State Space**: 8 continuous variables (position, velocity, angle, etc.)
- **Action Space**: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)
- **Goal**: Land safely between the flags with zero velocity

## 🧠 Algorithm Features

- ✅ **Deep Q-Network** with experience replay
- ✅ **Target network** for stable training
- ✅ **Epsilon-greedy** exploration strategy
- ✅ **Reward shaping** for faster convergence
- ✅ **Double DQN** implementation (optional)
- ✅ **Dueling DQN** architecture (optional)

## 📊 Results

After 2000 episodes of training:
- **Average reward**: 230+ (solves the environment!)
- **Success rate**: 95%
- **Training time**: ~45 minutes on RTX GPU

![Training Rewards](images/rewards.png)
*Learning curve showing reward over episodes*

![Lunar Lander Demo](images/training.gif)
*Trained agent successfully landing*

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dqn-lunar-lander
cd dqn-lunar-lander

# Install dependencies
pip install -r requirements.txt

# Train the agent
python train.py --episodes 2000

# Test a trained model
python test.py --model results/best_model.pth --render
