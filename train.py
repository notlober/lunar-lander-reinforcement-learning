import gymnasium as gym
import torch
import numpy as np

# eger egitimi canli izlemek istiyorsaniz: True
DEBUG = True

if DEBUG:
    env = gym.make("LunarLander-v3", render_mode="human")
else:
    env = gym.make("LunarLander-v3")

class NN(torch.nn.Module):
   def __init__(self, dim):
       super().__init__()
       self.lin1 = torch.nn.Linear(8, dim)
       self.lin2 = torch.nn.Linear(dim, dim)
       self.lin3 = torch.nn.Linear(dim, 4)
       
   def forward(self, x):
       x = torch.nn.functional.relu(self.lin1(x))
       x = torch.nn.functional.relu(self.lin2(x))
       return self.lin3(x)

model = NN(64)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
gamma = 0.99

memory = []

observation, info = env.reset(seed=42)
episode_rewards = []
total_episodes = 1000

for episode in range(total_episodes):
   done = False
   total_reward = 0
   
   observation, info = env.reset()
   
   while not done:
       obs_tensor = torch.FloatTensor(observation)
       
       with torch.no_grad():
           action_probs = torch.softmax(model(obs_tensor), dim=0)
           action = torch.multinomial(action_probs, 1).item()
       
       next_observation, reward, terminated, truncated, info = env.step(action)
       done = terminated or truncated
       total_reward += reward
       
       memory.append((observation, action, reward, next_observation, done))
       
       observation = next_observation
       
       if len(memory) >= 32:
           batch = np.random.choice(len(memory), 32, replace=False)
           
           optimizer.zero_grad()
           
           total_loss = 0
           for idx in batch:
               s, a, r, s_next, d = memory[idx]
               
               s = torch.FloatTensor(s)
               s_next = torch.FloatTensor(s_next)
               
               current_q = model(s)[a]
               
               with torch.no_grad():
                   next_q = model(s_next).max()
                   target_q = r + gamma * next_q * (1 - int(d))
               
               loss = torch.nn.functional.mse_loss(current_q, torch.tensor(target_q))
               total_loss += loss
           
           (total_loss / 32).backward()
           optimizer.step()
           
   episode_rewards.append(total_reward)
   
   if (episode + 1) % 10 == 0:
       avg_reward = sum(episode_rewards[-10:]) / 10
       print(f"episode {episode+1}/{total_episodes}, reward: {avg_reward:.2f}")

env.close()
