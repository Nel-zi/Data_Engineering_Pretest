import numpy as np
import matplotlib.pyplot as plt
import random

# Define the Maze Environment
class MazeEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.n_rows, self.n_cols = maze.shape
        self.start = (0, 0)  # Start position
        self.goal = (self.n_rows - 1, self.n_cols - 1)  # Goal position
        self.agent_pos = self.start

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
        # Calculate next position
        next_pos = (self.agent_pos[0] + moves[action][0],
                    self.agent_pos[1] + moves[action][1])

        # Check if the move is valid
        if (0 <= next_pos[0] < self.n_rows and
            0 <= next_pos[1] < self.n_cols and
            self.maze[next_pos[0], next_pos[1]] != 1):
            self.agent_pos = next_pos
            reward = 1 if self.agent_pos == self.goal else 0
        else:
            reward = -1  # Penalty for hitting walls or invalid move

        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

    def render(self):
        display_maze = np.copy(self.maze)
        display_maze[self.agent_pos] = 2  # Represent the agent with '2'
        plt.imshow(display_maze, cmap="viridis")
        plt.show()

# Q-Learning Algorithm
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.n_rows, env.n_cols, 4))  # Rows x Columns x Actions

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # Explore
            return random.randint(0, 3)
        else:  # Exploit
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.max(self.q_table[next_state[0], next_state[1]])
        current_q = self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] = current_q + self.lr * (reward + self.gamma * best_next_action - current_q)

    def train(self, episodes=1000):
        rewards_per_episode = []

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

        return rewards_per_episode

# Define the Maze (0: Walkable, 1: Wall)
maze = np.array([
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0]
])

# Initialize the environment and agent
env = MazeEnvironment(maze)
agent = QLearningAgent(env)

# Train the agent
rewards = agent.train(episodes=1000)

# Plot the training rewards
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Rewards Over Time")
plt.show()

# Test the trained agent
state = env.reset()
env.render()
done = False

while not done:
    action = np.argmax(agent.q_table[state[0], state[1]])
    state, _, done = env.step(action)
    env.render()



### **Summary of the Maze Navigation with Q-Learning**

# This uses the Q-Learning algorithm to train an agent to navigate through a maze while avoiding walls and obstacles. The maze is represented as a 2D grid, with the agent starting at the top-left corner and aiming to reach the bottom-right corner. The agent learns an optimal path through trial and error by interacting with the environment and updating a Q-table based on rewards and penalties.

# **Key Features:**
# - A **custom maze environment** simulates the grid, supporting actions like moving up, down, left, and right.
# - The **Q-Learning algorithm** trains the agent to maximize cumulative rewards by learning the best policy for navigation.
# - Visualization of training progress and the agent's movements using `matplotlib`.

# **How It Works:**
# 1. The agent starts at the maze's initial position and explores possible paths using actions.
# 2. Rewards:
#    - `+1` for reaching the goal.
#    - `-1` for hitting walls or invalid moves.
#    - `0` for valid intermediate steps.
# 3. The agent updates the Q-table using the formula:
#    \[
#    Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a) - Q(s, a) \right]
#    \]
# 4. Over time, the agent shifts from exploration to exploitation as the epsilon value (exploration rate) decays.

# **Results:**
# - A graph shows the agent's improving performance through rewards earned per episode.
# - The maze and the agent's position are visualized step-by-step.

# **Customization Options:**
# - Modify the maze design, training parameters (learning rate, discount factor, epsilon decay), and episode count to suit different challenges.

# **Limitations & Future Enhancements:**
# - Q-Learning struggles with scalability in larger mazes, where Deep Q-Learning (DQN) is recommended.
# - Potential enhancements include dynamic mazes, multiple goals, and time-based challenges.

# This whole thing showcases how reinforcement learning enables agents to autonomously learn and solve navigation problems through iterative improvement.