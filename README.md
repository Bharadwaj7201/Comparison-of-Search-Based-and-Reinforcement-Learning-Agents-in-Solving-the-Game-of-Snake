# Comparison-of-Search-Based-and-Reinforcement-Learning-Agents-in-Solving-the-Game-of-Snake
Snake AI showdown: Compare classical search algorithms with reinforcement learning as they navigate, learn, and master the game.
## Introduction
This project implements and compares **Search-Based algorithms** and **Reinforcement Learning techniques** for solving the classic **Snake game**. The search-based approach uses **Breadth-First Search (BFS)** and **A\*** to compute optimal paths toward the food, while the reinforcement learning approach uses a **Deep Q-Network (DQN)** to learn gameplay behavior through interaction with the environment.

The Snake game provides a dynamic, grid-based setting for evaluating decision-making, path planning, adaptability, and long-term survival. Both approaches are implemented within the same environment to ensure a fair and consistent comparison.

---

## Requirements
- Python 3.x  
- NumPy  
- PyGame  
- TensorFlow / PyTorch (for DQN agent)

---

## Workflow

### Game Environment
A unified Snake game environment is designed on a grid where the snake moves in four directions (up, down, left, right). The environment includes walls, snake body constraints, and randomly generated food positions.

---

## Search-Based Agent

### Graph Representation
The grid-based environment is treated as a graph where each free cell represents a node, and edges exist between adjacent reachable cells.

### Breadth-First Search (BFS)
BFS explores the grid level-by-level and guarantees the **shortest path** to the food when a valid path exists.

### A* Search Algorithm
A* improves efficiency by using a heuristic to guide the search toward the food. The **Manhattan distance heuristic** is used due to the grid-based movement of the Snake game.

**Cost Function:**
```
f(n) = g(n) + h(n)
```
Where:
- g(n) is the cost from the start node to the current node  
- h(n) is the heuristic distance to the food  

Safe-path simulation and tail-following strategies are incorporated to reduce the risk of getting trapped after food consumption.

---

## Reinforcement Learning Agent

The Reinforcement Learning agent is implemented using a **Deep Q-Network (DQN)**. The agent learns an optimal policy by interacting with the environment and receiving rewards for its actions.

### State Representation
The state includes:
- Current movement direction
- Danger indicators in all directions
- Relative position of the food

### Reward Scheme
- Positive reward for eating food  
- Negative reward for collisions or death  
- Small positive reward for survival at each step  

The agent uses **ε-greedy exploration**, **experience replay**, and **Q-learning updates** to improve performance over time.

---

## How to Run

1. Clone the repository:
```bash
git clone <repository-url>
cd Snake-AI-Retro
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Snake game:
```bash
python main.py
```

4. Select the agent type (Search-Based or DQN) when prompted.

---

## Comparison
The search-based agent demonstrates deterministic and predictable behavior with optimal path planning but lacks adaptability in complex or crowded environments. In contrast, the reinforcement learning agent requires training time but learns flexible strategies and offers improved long-term performance in dynamic scenarios.

---

## Conclusion
This project highlights the differences between **classical AI search techniques** and **modern reinforcement learning approaches**. Search-based methods perform well in structured settings, while reinforcement learning agents adapt more effectively to dynamic and complex environments such as the Snake game.
