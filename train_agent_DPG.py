import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import gym_anytrading
import pandas as pd

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size * 2, 32)  # state_size * 2 để làm phẳng từ 30x2 thành 60
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, action_size)

    def forward(self, x):
        x = x.view(-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.output(x), dim=-1)  # Đầu ra xác suất
        return x

# Agent
class PolicyAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.gamma = gamma
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = []  # Stores (state, action, reward)

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy_network(state).cpu().detach().numpy()
        return np.random.choice(self.action_size, p=probs)  # Sample action based on probabilities

    def learn(self):
        R = 0
        returns = []
        for _, _, reward in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns for stability
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute loss and update policy
        policy_loss = []
        for (state, action, reward), R in zip(self.memory, returns):
            state = torch.FloatTensor(state).to(self.device)
            action_prob = self.policy_network(state)[action]
            log_prob = torch.log(action_prob)
            policy_loss.append(-log_prob * R)  # REINFORCE: -logπ(a|s) * R

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []  # Clear memory after updating

# Main program
df = pd.read_csv('gmedata.csv') #Config input data
env = gym.make('stocks-v0', df=df, frame_bound=(30,1000), window_size=30)
state, _ = env.reset()

# Parameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
n_episodes = 300
n_timesteps = 1000

agent = PolicyAgent(state_size, action_size)

for ep in range(n_episodes):
    state, _ = env.reset()
    ep_reward = 0
    for t in range(n_timesteps):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.remember(state, action, reward)
        state = next_state
        ep_reward += reward
        done = terminated or truncated

        if done:
            print(f"Episode {ep+1}, Reward: {ep_reward}")
            agent.learn()  # Update policy after each episode
            break

# Save model weights after training
torch.save(agent.policy_network.state_dict(), "trained_policy_model.pth")
