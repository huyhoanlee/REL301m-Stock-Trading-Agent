import gymnasium as gym
import gym_anytrading
import pandas as pd
import numpy as np
# from train_stock_agent_DPG import PolicyNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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

df = pd.read_csv('Dataset\FPT.csv')
env = gym.make('stocks-v0', df=df, frame_bound=(30,200), window_size=30)
state, _ = env.reset()

# Định nghĩa state_size và action_size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size, action_size)
print(env.action_space, action_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = PolicyNetwork(state_size, action_size)#.to(device)
print(loaded_model)
loaded_model.load_state_dict(torch.load("trained_policy_model.pth"))

#------------------
state, info = env.reset()
print(state)
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
has_stock_count = 0 
n_timesteps = 200
total_reward = 0
list_act = []
for t in range(n_timesteps):
    env.render()
    # Lấy state hiện tại đưa vào predict
    state = torch.tensor(state).unsqueeze(0)
    # q_values = loaded_model(state)
    probs = loaded_model(state).cpu().detach().numpy()
    max_q_values = np.random.choice(action_size, p=probs)
    if max_q_values == 0 and has_stock_count == 0:
        max_q_values = 1
    if max_q_values == 0: 
        has_stock_count -= 1
    elif max_q_values == 1:
        has_stock_count += 1

    # Action vào env và lấy thông so
    next_state, reward, terminal, truncate, _ = env.step(action=max_q_values)
    if terminal or truncate:
        break
    total_reward += reward
    state = next_state
    list_act.append(max_q_values)
    print(t, max_q_values, reward)

print("Total list_act:", list_act)
count_sell, count_buy = 0, 0
for i in list_act:
    if i == 0:
        count_sell += 1
    else:
        count_buy += 1
print(count_sell, count_buy)
    