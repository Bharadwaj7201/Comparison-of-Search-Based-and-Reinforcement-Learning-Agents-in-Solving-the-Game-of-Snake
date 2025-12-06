# play_dqn.py

import torch
from dqn_agent import DQNAgent
from snake_game import SnakeGame
import time

MODEL_PATH = "dqn_snake.pth"
DELAY = 0.05

game = SnakeGame()
state_size = game.get_state_size()
action_size = game.get_action_size()

agent = DQNAgent(state_size, action_size)
agent.model.load_state_dict(torch.load(MODEL_PATH))
agent.epsilon = 0.0

state = game.reset()
done = False
score = 0

while not done:
    action_idx = agent.act(state)
    action = [0,0,0]
    action[action_idx] = 1

    state, reward, done, _ = game.step(action)
    score += reward
    game.render()
    time.sleep(DELAY)

print(f"Game Over! Final Score: {score}")
