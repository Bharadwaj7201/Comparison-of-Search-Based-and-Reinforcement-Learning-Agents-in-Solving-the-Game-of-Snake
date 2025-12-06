import threading
import subprocess
import os

MODEL_PATH = "RL_Agent/dqn_snake.pth"

def train_rl_agent():
    print("Training RL Agent (no model found)...")
    subprocess.run(["python", "RL_Agent/train_dqn.py"])

def run_rl_agent():
    if not os.path.exists(MODEL_PATH):
        train_rl_agent()
    print("Running RL Agent...")
    subprocess.run(["python", "RL_Agent/play_dqn.py"])

def run_search_agent():
    print("Running Search-Based Agent...")
    subprocess.run(["python", "main_search_ai.py"])

def run_visualisations():
    print("Running Visualisations...")
    subprocess.run(["python", "visualisations.py"])

def run_rl_utilities():
    print("Running RL Agent Utilities...")
    subprocess.run(["python", "RL_Agent/evaluation.py"])
    subprocess.run(["python", "RL_Agent/search_agent.py"])
    subprocess.run(["python", "RL_Agent/snake_game.py"])
    subprocess.run(["python", "RL_Agent/dqn_agent.py"])

def run_evaluation():
    print("Running Final Evaluation...")
    subprocess.run(["python", "Snake_AI_Evaluation.py"])

if __name__ == "__main__":
    # Run all agents and utilities in parallel
    threads = [
        threading.Thread(target=run_rl_agent),
        threading.Thread(target=run_search_agent),
        threading.Thread(target=run_visualisations),
        threading.Thread(target=run_rl_utilities)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Run evaluation after all others complete
    run_evaluation()

    print("✅ All agents, utilities, and final evaluation completed.")