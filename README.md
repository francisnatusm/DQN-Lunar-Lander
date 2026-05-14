# Deep Q-Network for Lunar Lander

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-green)](https://gymnasium.farama.org/)

A Deep Q-Learning implementation that trains an agent to safely land on the lunar surface using the Gymnasium LunarLander-v3 environment. Built as part of my reinforcement learning journey.

## Environment

The agent controls a lunar lander in the [Gymnasium LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment:

- **State Space**: 8 continuous variables (position, velocity, angle, angular velocity, leg contact)
- **Action Space**: 4 discrete actions (do nothing, fire right engine, fire main engine, fire left engine)
- **Goal**: Land safely between the flags with zero velocity
- **Solved**: Average reward of 200+ over the last 100 episodes

## Algorithm Features

- **Deep Q-Network** with experience replay (memory buffer of 100,000 transitions)
- **Target network** with soft updates (TAU = 0.001) for stable training
- **Epsilon-greedy** exploration with decay (0.995 per episode, min 0.01)
- **Model saving** on success and **model loading** via `--load` for reuse

## Results

- **Average reward**: 200+ (solves the environment)
- **Training time**: ~5 minutes on RTX 4050 Laptop GPU

## Quick Start

```bash
# Clone the repository
git clone https://github.com/francisnatusm/DQN-Lunar-Lander.git
cd DQN-Lunar-Lander

# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the agent from scratch
python3 main.py --no-tests

# Load a pre-trained model (skips training, generates demo video)
python3 main.py --no-tests --load lunar_lander_model.h5
```

## Project Structure

```
├── main.py                  # Training loop, model loading, and video generation
├── utils.py                 # Helper functions (replay buffer, epsilon decay, plotting, video)
├── public_tests.py          # Unit tests for network architecture and loss function
├── lunar_lander_model.h5    # Pre-trained model (saved when environment is solved)
├── requirements.txt         # Python dependencies
└── videos/
    └── lunar_lander.mp4     # Demo video of the trained agent
```

## How It Works

1. The agent interacts with the environment using an epsilon-greedy policy
2. Experiences (state, action, reward, next state) are stored in a replay buffer
3. Mini-batches are sampled to train the Q-network via gradient descent
4. A target network is softly updated to provide stable Q-value targets
5. Training stops when the average reward over 100 episodes reaches 200+
6. The trained model is saved to `lunar_lander_model.h5` for later reuse
