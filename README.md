# Q-Learning Flappy Bird

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Framework](https://img.shields.io/badge/UI-PySide6-blue)
![Status](https://img.shields.io/badge/status-archived-red)

This project is a visual and interactive implementation of a Q-learning agent that learns to play the game Flappy Bird. It features a comprehensive dashboard built with PySide6 and `pyqtgraph` to configure hyperparameters, monitor the training process in real-time, and visualize the agent's performance.

The primary goal of this project is to serve as a clear and straightforward educational tool for understanding the fundamentals of Reinforcement Learning (RL), specifically Q-learning, in a tangible and observable way.

## Demonstration

Below is a demonstration of a fully trained agent playing the game. The agent makes decisions based solely on its learned Q-table, with no random exploration.

![Demonstration of the trained agent playing Flappy Bird](https://github.com/user-attachments/assets/cbf919e4-1b64-4c27-894e-70c8a1be662a)

## Features

- **Interactive Training Dashboard:** A user-friendly interface to start, stop, and monitor the training process.
- **Real-Time Performance Visualization:** Live-plotting of average and maximum scores using `pyqtgraph` to track learning progress.
- **Configurable Hyperparameters:** Easily adjust key Q-learning parameters like learning rate, discount factor, and exploration decay directly from the UI.
- **Non-Blocking Training:** The learning algorithm runs in a separate thread, ensuring the UI remains responsive at all times.
- **Agent Demonstration:** After training, watch the agent play the game using its learned policy.
- **Manual Play Mode:** A separate window to play the game yourself.
- **Self-Contained Script:** The entire application, including the game environment, agent logic, and UI, is contained within a single Python file for simplicity.

## Project Background

This project was originally developed several years ago as a personal exercise to deepen my understanding of reinforcement learning concepts. After sitting in my private repositories for some time, I decided to clean it up and share it publicly in the hope that it might be a useful learning resource for others who are new to the field. Its simplicity is intentional, designed to make the core concepts of Q-learning accessible.

## Getting Started

To run this project, you will need Python 3.7 or newer.

### 1. Prerequisites

Ensure you have the necessary libraries installed. You can install them using pip:
```bash
pip install numpy pyqtgraph PySide6
```

### 2. Running the Application

Clone the repository and run the main Python script:
```bash
git clone https://github.com/dovvnloading/RL-FlappyBird.git
cd RL-Flappy-Bird
python Q_Learning_Flappy.py
```
This will launch the main training dashboard.

## How It Works

The project uses a classic Q-learning algorithm to solve the Flappy Bird environment.

### The Environment
- **State Representation:** The state is a simplified, discretized representation of the game world, defined by a tuple: `(bird_y, bird_velocity, horizontal_distance_to_pipe, vertical_distance_to_pipe)`. This reduces the state space to a manageable size for a Q-table.
- **Actions:** The agent has two possible actions: `0` (do nothing) or `1` (flap).
- **Reward Structure:**
    - `+1` for each frame it stays alive.
    - `+50` for successfully passing through a pipe.
    - `-100` for colliding with a pipe or the ground/ceiling (terminal state).

### The Q-Learning Agent
The agent learns a policy that maps states to actions to maximize its cumulative reward. It uses a dictionary as a Q-table to store the expected rewards for taking an action in a given state (`Q(s, a)`).

The Q-table is updated using the Bellman equation:
`Q(s, a) = Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]`
- `α` (alpha) is the learning rate.
- `γ` (gamma) is the discount factor.
- `R` is the reward received.
- `s'` is the next state.

Exploration is managed using an epsilon-greedy strategy, where the agent initially explores the environment randomly and gradually reduces its exploration rate to exploit its learned knowledge.

## Usage Guide

The application is controlled through the main dashboard window.

### The Training Dashboard

The main window provides all the necessary controls and feedback for the training process.

<img src="https://github.com/user-attachments/assets/45be3f89-ee19-40a2-a86c-ede9ecf12841" width="800" />

### 1. Configure and Start Training

- Use the "Training Parameters" panel to set the number of episodes, learning rates, discount factor, and exploration decay.
- Press **"Start Training"** to begin the learning process. The training will run in the background.
- The **"Stop Training"** button can be used to gracefully interrupt the training process after the current episode finishes.

<img src="https://github.com/user-attachments/assets/aa4206db-3e4f-4d61-9856-0de9f147e91e" width="800" />

### 2. Monitor Performance

- The "Live Statistics" panel shows real-time data, including the current episode, average score, exploration rate, and Q-table size.
- The graph plots the average and maximum scores (calculated over the last 100 episodes) against the number of episodes, providing a clear visual indicator of the agent's learning progress.

### 3. Demonstrate the Trained Agent

- Once training is complete or stopped, the **"Demonstrate"** button becomes active.
- Clicking it will launch a new window where the trained agent plays the game using its learned policy (with exploration turned off).

<img src="https://github.com/user-attachments/assets/8fb7cdb7-5f7b-44d3-9311-b62b33395ae7" width="400" />

## Code Overview

The entire project is contained in `Q_Learning_Flappy.py` and is structured into several key classes:

- **`FlappyBirdEnvironment`**: Manages the game state, physics, and reward logic.
- **`QLearningAgent`**: Implements the Q-learning algorithm, including the Q-table, action selection (epsilon-greedy), and update rule.
- **`TrainingThread`**: A `QThread` that runs the training loop separately from the main UI thread to prevent freezing.
- **`FlappyBirdTrainingApp`**: The main `QMainWindow` that builds the dashboard UI and connects all the components.
- **`FlappyBirdDemoWindow` / `FlappyBirdPlayWindow`**: `QMainWindow` subclasses for demonstrating the agent and for manual gameplay.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
