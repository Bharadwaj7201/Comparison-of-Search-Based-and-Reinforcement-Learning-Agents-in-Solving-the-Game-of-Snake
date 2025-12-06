import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv("snake_ai_metrics.csv")

# Compute summary statistics
summary = pd.DataFrame({
    "Agent": ["Search", "DQN"],
    "Max Score": [df["search_score"].max(), df["dqn_score"].max()],
    "Avg Score": [df["search_score"].mean(), df["dqn_score"].mean()],
    "Min Score": [df["search_score"].min(), df["dqn_score"].min()],
    "Max Steps": [df["search_steps"].max(), df["dqn_steps"].max()],
    "Avg Steps": [df["search_steps"].mean(), df["dqn_steps"].mean()],
    "Max Reward": [df["search_rewards"].max(), df["dqn_rewards"].max()],
    "Avg Reward": [df["search_rewards"].mean(), df["dqn_rewards"].mean()],
})

print("=== Summary Statistics ===")
print(summary)

# Plot: Score per episode
plt.figure(figsize=(12,5))
plt.plot(df["search_score"], label="Search Agent", marker='o')
plt.plot(df["dqn_score"], label="DQN Agent", marker='x')
plt.title("Score per Episode")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Steps per episode
plt.figure(figsize=(12,5))
plt.plot(df["search_steps"], label="Search Agent", marker='o')
plt.plot(df["dqn_steps"], label="DQN Agent", marker='x')
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Rewards per episode
plt.figure(figsize=(12,5))
plt.plot(df["search_rewards"], label="Search Agent", marker='o')
plt.plot(df["dqn_rewards"], label="DQN Agent", marker='x')
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()
