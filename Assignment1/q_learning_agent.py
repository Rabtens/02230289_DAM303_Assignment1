"""
DAM303 – Deep Reinforcement Learning
Programming Assignment 1: Q-Learning Implementation
Student: [Your Name]
College of Science and Technology, Royal University of Bhutan
Department of Software Engineering
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
GRID_SIZE      = 5          # 5x5 grid → 25 states
NUM_STATES     = GRID_SIZE * GRID_SIZE   # 25
NUM_ACTIONS    = 4          # 0=Up, 1=Down, 2=Left, 3=Right

START_STATE    = 0          # top-left
GOAL_STATE     = 24         # bottom-right
OBSTACLE_STATES = [6, 12, 18]

GOAL_REWARD    = 10.0
STEP_REWARD    = -0.1
OBSTACLE_REWARD = -5.0

MAX_STEPS      = 200
NUM_EPISODES   = 1000

ALPHA          = 0.1        # learning rate
GAMMA          = 0.99       # discount factor
EPSILON        = 1.0        # initial exploration rate
EPSILON_DECAY  = 0.995
EPSILON_MIN    = 0.01

ROLLING_WINDOW = 50         # for smoothing plots


# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────

def state_to_rowcol(state: int):
    """Convert flat state index to (row, col)."""
    return divmod(state, GRID_SIZE)


def rowcol_to_state(row: int, col: int) -> int:
    """Convert (row, col) to flat state index."""
    return row * GRID_SIZE + col


def get_next_state(state: int, action: int) -> int:
    """
    Compute the next state given the current state and action.
    Boundary conditions: agent stays in place if it tries to move off the grid.

    Actions:
        0 = Up    (row - 1)
        1 = Down  (row + 1)
        2 = Left  (col - 1)
        3 = Right (col + 1)
    """
    row, col = state_to_rowcol(state)

    if action == 0:   # Up
        row = max(row - 1, 0)
    elif action == 1: # Down
        row = min(row + 1, GRID_SIZE - 1)
    elif action == 2: # Left
        col = max(col - 1, 0)
    elif action == 3: # Right
        col = min(col + 1, GRID_SIZE - 1)

    return rowcol_to_state(row, col)


def get_reward(state: int, next_state: int):
    """
    Return (reward, done, reset_to_start).
    - Goal reached  → +10, episode ends.
    - Obstacle hit  → -5, agent teleports back to start.
    - Any other     → -0.1, episode continues.
    """
    if next_state == GOAL_STATE:
        return GOAL_REWARD, True, False
    elif next_state in OBSTACLE_STATES:
        return OBSTACLE_REWARD, False, True   # penalty + teleport
    else:
        return STEP_REWARD, False, False


def step(state: int, action: int):
    """
    Perform one environment step.
    Returns (next_state, reward, done).
    Applies obstacle teleportation logic.
    """
    next_state = get_next_state(state, action)
    reward, done, teleport = get_reward(state, next_state)
    if teleport:
        next_state = START_STATE   # teleport back to start
    return next_state, reward, done


# ─────────────────────────────────────────────
# Q-Learning Agent
# ─────────────────────────────────────────────

class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy policy."""

    def __init__(self):
        # Q-table: rows = states, cols = actions, initialised to zero
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.epsilon = EPSILON

    def select_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.
        With probability epsilon choose a random action (explore);
        otherwise choose the greedy action (exploit).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)   # explore
        return int(np.argmax(self.q_table[state]))  # exploit

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """
        Bellman equation Q-table update:
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        If the episode is done the future return is 0.
        """
        target = reward + (0.0 if done else GAMMA * np.max(self.q_table[next_state]))
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += ALPHA * td_error

    def decay_epsilon(self):
        """Reduce epsilon after each episode (but never below epsilon_min)."""
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(agent: QLearningAgent):
    """Run Q-Learning for NUM_EPISODES episodes. Returns reward and step logs."""
    reward_log = []
    steps_log  = []

    for episode in range(NUM_EPISODES):
        state       = START_STATE
        total_reward = 0.0
        steps       = 0

        for _ in range(MAX_STEPS):
            action                      = agent.select_action(state)
            next_state, reward, done    = step(state, action)
            agent.update(state, action, reward, next_state, done)

            state        = next_state
            total_reward += reward
            steps        += 1

            if done:
                break

        agent.decay_epsilon()
        reward_log.append(total_reward)
        steps_log.append(steps)

        if (episode + 1) % 100 == 0:
            avg_r = np.mean(reward_log[-100:])
            print(f"Episode {episode + 1:4d}/{NUM_EPISODES} | "
                  f"Avg Reward (last 100): {avg_r:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    return reward_log, steps_log


# ─────────────────────────────────────────────
# Performance plots  (Task 3)
# ─────────────────────────────────────────────

def rolling_average(data, window=ROLLING_WINDOW):
    """Compute a simple rolling (moving) average."""
    result = np.convolve(data, np.ones(window) / window, mode='valid')
    return result


def plot_rewards(reward_log: list, save_dir: str):
    episodes = np.arange(1, NUM_EPISODES + 1)
    smoothed  = rolling_average(reward_log)
    smooth_x  = np.arange(ROLLING_WINDOW, NUM_EPISODES + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, reward_log, color='steelblue', alpha=0.35,
            linewidth=0.8, label='Raw reward')
    ax.plot(smooth_x, smoothed, color='navy', linewidth=2,
            label=f'Rolling avg (window={ROLLING_WINDOW})')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Total Reward', fontsize=13)
    ax.set_title('Q-Learning: Total Reward per Episode', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, 'reward_per_episode.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_steps(steps_log: list, save_dir: str):
    episodes = np.arange(1, NUM_EPISODES + 1)
    smoothed  = rolling_average(steps_log)
    smooth_x  = np.arange(ROLLING_WINDOW, NUM_EPISODES + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, steps_log, color='tomato', alpha=0.35,
            linewidth=0.8, label='Steps per episode')
    ax.plot(smooth_x, smoothed, color='darkred', linewidth=2,
            label=f'Rolling avg (window={ROLLING_WINDOW})')
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Steps to Goal (or Max Steps)', fontsize=13)
    ax.set_title('Q-Learning: Steps per Episode', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, 'steps_per_episode.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Greedy policy evaluation (Task 2)
# ─────────────────────────────────────────────

def run_greedy_path(agent: QLearningAgent):
    """
    Execute the fully greedy policy (epsilon=0) from the start state.
    Print the path taken and whether the goal was reached.
    Also demonstrates obstacle avoidance.
    """
    state  = START_STATE
    path   = [state]
    done   = False
    steps  = 0

    print("\n─── Greedy Policy Path After Training ───")
    while not done and steps < MAX_STEPS:
        action     = int(np.argmax(agent.q_table[state]))
        next_state, reward, done = step(state, action)
        path.append(next_state)
        state  = next_state
        steps += 1

    action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

    # Reconstruct path with state info
    print(f"Path (state indices): {path}")
    print(f"Total steps: {steps}")
    if path[-1] == GOAL_STATE:
        print("✓ Agent reached the GOAL state (24).")
    else:
        print("✗ Agent did NOT reach the goal within the step limit.")

    # Show on grid
    print("\nGrid visualisation (S=Start, G=Goal, O=Obstacle, *=visited):")
    visited = set(path)
    for row in range(GRID_SIZE):
        line = ""
        for col in range(GRID_SIZE):
            s = rowcol_to_state(row, col)
            if s == START_STATE:
                line += " S "
            elif s == GOAL_STATE:
                line += " G "
            elif s in OBSTACLE_STATES:
                line += " O "
            elif s in visited:
                line += " * "
            else:
                line += " . "
        print(line)

    return path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)  # reproducibility

    # Create output directory for plots
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Initialise and train the agent
    print("=== Q-Learning Agent Training ===")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}  |  States: {NUM_STATES}  |  Actions: {NUM_ACTIONS}")
    print(f"Obstacles: {OBSTACLE_STATES}  |  Episodes: {NUM_EPISODES}\n")

    agent = QLearningAgent()
    reward_log, steps_log = train(agent)

    # Plot performance metrics
    print("\n=== Generating Performance Plots ===")
    plot_rewards(reward_log, plots_dir)
    plot_steps(steps_log, plots_dir)

    # Run greedy policy to show obstacle avoidance
    greedy_path = run_greedy_path(agent)

    # Print learned Q-table (optional inspection)
    print("\n=== Learned Q-Table (sample: first 5 states) ===")
    print(f"{'State':>6}  {'Up':>8}  {'Down':>8}  {'Left':>8}  {'Right':>8}")
    for s in range(5):
        q = agent.q_table[s]
        print(f"{s:>6}  {q[0]:>8.4f}  {q[1]:>8.4f}  {q[2]:>8.4f}  {q[3]:>8.4f}")

    print("\nTraining complete. Plots saved in ./plots/")