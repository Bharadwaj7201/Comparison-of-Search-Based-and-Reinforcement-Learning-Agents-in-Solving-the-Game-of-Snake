# Snake_AI_Evaluation.ipynb
# ====================================
# Snake AI: Search vs DQN Agent Evaluation
# ====================================

# Cell 1: Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from RL_Agent.snake_game import SnakeGame
from RL_Agent.search_agent import SearchAgent
from RL_Agent.dqn_agent import DQNAgent

# Settings
EPISODES = 30
DQN_MODEL_PATH = r"C:\Users\Bharadwaj\Downloads\Snake-AI-Retro\RL_Agent\dqn_snake.pth"

# ====================================
# Cell 2: Evaluate Search Agent
# ====================================
def evaluate_search_agent(episodes=EPISODES):
    scores, survival_steps, rewards = [], [], []

    game = SnakeGame(headless=True)
    agent = SearchAgent(game)

    for _ in range(episodes):
        state = game.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action_idx = agent.get_action(state)
            action = [0, 0, 0]
            action[action_idx] = 1
            next_state, reward, done, _ = game.step(action)
            state = next_state
            episode_reward += reward
            steps += 1

        scores.append(game.score)
        survival_steps.append(steps)
        rewards.append(episode_reward)

    return scores, survival_steps, rewards

# ====================================
# Cell 3: Evaluate DQN Agent
# ====================================
def evaluate_dqn_agent(model_path=DQN_MODEL_PATH, episodes=EPISODES):
    scores, survival_steps, rewards = [], [], []

    game = SnakeGame(headless=True)
    state_size = game.get_state_size()
    action_size = game.get_action_size()

    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    agent.epsilon = 0.0  # greedy evaluation

    for _ in range(episodes):
        state = torch.tensor(game.reset(), dtype=torch.float).unsqueeze(0)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action_idx = agent.act(state)
            action = [0, 0, 0]
            action[action_idx] = 1
            next_state, reward, done, _ = game.step(action)
            state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            episode_reward += reward
            steps += 1

        scores.append(game.score)
        survival_steps.append(steps)
        rewards.append(episode_reward)

    return scores, survival_steps, rewards

# ====================================
# Cell 4: Run Evaluation
# ====================================
search_scores, search_steps, search_rewards = evaluate_search_agent()
dqn_scores, dqn_steps, dqn_rewards = evaluate_dqn_agent()

# ====================================
# Cell 5: Print Statistics
# ====================================
def print_stats(name, scores, steps, rewards):
    print(f"\n----------- {name} -----------")
    print("Max Score:", max(scores))
    print("Avg Score:", np.mean(scores))
    print("Avg Steps:", np.mean(steps))
    print("Avg Reward:", np.mean(rewards))

print_stats("Search Agent", search_scores, search_steps, search_rewards)
print_stats("DQN Agent", dqn_scores, dqn_steps, dqn_rewards)


# ====================================
# Cell 7: Plots
# ====================================
plt.figure(figsize=(12,5))
plt.plot(search_scores, label="Search Agent", marker='o')
plt.plot(dqn_scores, label="DQN Agent", marker='x')
plt.title("Score per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(search_steps, label="Search Agent", marker='o')
plt.plot(dqn_steps, label="DQN Agent", marker='x')
plt.title("Survival Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(search_rewards, label="Search Agent", marker='o')
plt.plot(dqn_rewards, label="DQN Agent", marker='x')
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
