import random
import numpy as np
import gymnasium as gym
import gym_anytrading
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the Agent class with a PyTorch model
import torch

class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Agent parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.97
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10
        # self.base_update_targetnn_rate = self.update_targetnn_rate

        # Define main and target networks, and move them to the appropriate device
        self.main_network = self.build_nn().to(self.device)
        self.target_network = self.build_nn().to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())

        # Define optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)

    def build_nn(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.state_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = np.array([batch[0] for batch in exp_batch])
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = np.array([batch[2] for batch in exp_batch])
        next_state_batch = np.array([batch[3] for batch in exp_batch])
        terminal_batch = np.array([batch[4] for batch in exp_batch])
        
        # Move data to the correct device
        return (
            torch.FloatTensor(state_batch).to(self.device),
            torch.LongTensor(action_batch).to(self.device),
            torch.FloatTensor(reward_batch).to(self.device),
            torch.FloatTensor(next_state_batch).to(self.device),
            torch.FloatTensor(terminal_batch).to(self.device)
        )

    def train_main_network(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        # Get Q values for current states
        q_values = self.main_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute max Q values for next states using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            max_next_q = next_q_values.max(1)[0]

        # Compute target Q values
        target_q_values = reward_batch + (1 - terminal_batch) * self.gamma * max_next_q #thử xóa terminal batch

        # Compute loss and backpropagate
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # if torch.abs((q_values - target_q_values).mean()) > 0.1:  # Nếu sai lệch Q lớn hơn ngưỡng
        #     self.adaptive_update_rate = max(1, int(self.base_update_targetnn_rate / 2))
        # else:
        #     self.adaptive_update_rate = self.base_update_targetnn_rate  # Giữ nguyên nếu không biến động lớn

    def make_decision(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Convert state to tensor and move it to the correct device
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state)
            # print(q_values)
            return torch.argmax(q_values).item()


# Main program

df = pd.read_csv('gmedata.csv')  #config input data
env = gym.make('stocks-v0', df=df, frame_bound=(30,10000), window_size=30)
state, _ = env.reset()

# Define state_size and action_size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define other parameters
n_episodes = 50
n_timesteps = 10000
batch_size = 512

# Initialize agent
my_agent = MyAgent(state_size, action_size)
total_time_step = 0

for ep in range(n_episodes):
    ep_rewards = 0
    state, _ = env.reset()
    print('episodes ', ep, my_agent.epsilon)
    for t in range(n_timesteps):
        total_time_step += 1

        # Update target network weights
        if total_time_step % my_agent.update_targetnn_rate == 0:
            my_agent.target_network.load_state_dict(my_agent.main_network.state_dict())

        action = my_agent.make_decision(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        my_agent.save_experience(state, action, reward, next_state, terminated)

        state = next_state
        ep_rewards += reward

        done = terminated or truncated
        if done:
            print("Ep ", ep + 1, " reached terminal with reward = ", ep_rewards)
            break

        if len(my_agent.replay_buffer) > batch_size:
            my_agent.train_main_network(batch_size)

    if my_agent.epsilon > my_agent.epsilon_min:
        my_agent.epsilon *= my_agent.epsilon_decay

# Save the model weights
torch.save(my_agent.main_network.state_dict(), "trained_agent_VB.pth")
