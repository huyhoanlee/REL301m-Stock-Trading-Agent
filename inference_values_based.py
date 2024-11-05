import gymnasium as gym
import gym_anytrading
import pandas as pd
import numpy as np
# from stock_agent import MyAgent
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

df = pd.read_csv('Dataset\FPT.csv')
env = gym.make('stocks-v0', df=df, frame_bound=(30,200), window_size=30)
state, _ = env.reset()

# Định nghĩa state_size và action_size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size, action_size)
print(env.action_space, action_size)

class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)

        # Agent parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10

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
    
loaded_model = MyAgent(state_size, action_size).build_nn()
print(loaded_model)
loaded_model.load_state_dict(torch.load("trained_DQN_agent_10_updaterade.pth"))

#------------------
state, info = env.reset()
# print(state)
# while True:
#     # action = env.action_space.sample()
#     state = torch.tensor(state).unsqueeze(0)
#     q_values = loaded_model(state)
#     action = np.argmax(q_values.detach().numpy())
#     # 0 #sell red
#     print(action)
#     observation, reward, terminated, truncated, info = env.step(action)
    
#     print(observation, reward,  terminated, truncated, info)
#     done = terminated or truncated
#     if done: 
#         print("info", info)
#         break
        
#     env.render()
#------------------------


# # Set the model to evaluation mode
# loaded_model.eval()
n_timesteps = 200
total_reward = 0
list_act = []
for t in range(n_timesteps):
    env.render()
    # Lấy state hiện tại đưa vào predict
    state = torch.tensor(state).unsqueeze(0)
    q_values = loaded_model(state)
    max_q_values = np.argmax(q_values.detach().numpy())

    # Action vào env và lấy thông so
    next_state, reward, terminal, truncate, info = env.step(action=max_q_values)
    if terminal or truncate:
        break
    total_reward += reward
    state = next_state
    list_act.append(max_q_values)
    print(t, max_q_values, reward, info)

print("Total list_act:", list_act)
count_sell, count_buy = 0, 0
for i in list_act:
    if i == 0:
        count_sell += 1
    else:
        count_buy += 1
print(count_sell, count_buy)
    