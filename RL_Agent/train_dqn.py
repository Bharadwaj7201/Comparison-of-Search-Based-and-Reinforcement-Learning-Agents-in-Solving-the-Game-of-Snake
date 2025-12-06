# train_dqn.py

import numpy as np
from dqn_agent import DQNAgent
from snake_game import SnakeGame
import torch

EPISODES = 500
MAX_STEPS = 500
MODEL_PATH = "dqn_snake.pth"

game = SnakeGame()
state_size = game.get_state_size()
action_size = game.get_action_size()
agent = DQNAgent(state_size, action_size)

scores = []

for episode in range(1, EPISODES + 1):
    state = game.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action_idx = agent.act(state)
        action = [0,0,0]
        action[action_idx] = 1

        next_state, reward, done, _ = game.step(action)
        agent.remember(state, action_idx, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward
        if done:
            break

    scores.append(total_reward)
    avg_score = np.mean(scores[-50:])
    print(f"Episode {episode}/{EPISODES} | Score: {total_reward} | Avg(50): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    if episode % 50 == 0:
        torch.save(agent.model.state_dict(), MODEL_PATH)
        print(f"Model saved at episode {episode}")
