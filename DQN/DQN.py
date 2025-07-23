import SpaceGame as game
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import numpy as np
import pygame 
from torch.optim import Adam
from loss import weighted_MSE
import random 
from collections import deque

LR = 1e-3
BATCH_SIZE = 64
GAMMA = 0.7
NUM_ACTIONS = 4
HIDDEN_NEURONS = 128
NUM_EPISODES = 10000
INPUT_DIM = 7   
EPSILON = 0.1
EPSILON_DECAY = 0.995
ACTIONS = [0,1,2,3] # 0 -> stay, 1 -> left, 2 -> right, 3 -> shoot
GAME = game.SpaceInvaderGame()

class ReplayBuffer:
  def __init__(self, capacity = 50000):
    self.buffer = deque(maxlen=capacity)
  def __len__(self):
    return len(self.buffer)
  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))
  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)
    
# defining a approximator i.e a neural net
class qnn(nn.Module):
  def __init__(self, input_dim = INPUT_DIM, hidden_neurons = HIDDEN_NEURONS, num_actions = NUM_ACTIONS):
    super(qnn, self).__init__()

    self.input_layer = nn.Linear(input_dim, hidden_neurons)
    self.linear1 = nn.Linear(hidden_neurons, 64)
    self.linear2 = nn.Linear(64, num_actions)
    self.ReLU = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ReLU(self.input_layer(x))
    x = self.ReLU(self.linear1(x))
    x = self.linear2(x)
    return x

q_fn = qnn() #------- Q value function approximator object

class DQN:
  def __init__(self, Q = q_fn, gamma = GAMMA, epsilon_min = EPSILON, actions = ACTIONS, epsilon_decay = EPSILON_DECAY):
    self.replay_buffer = ReplayBuffer()
    self.batch_size = BATCH_SIZE
    self.Q = Q
    self.Q_target = qnn() # clone model
    self.Q_target.load_state_dict(self.Q.state_dict())
    self.Q_target.eval()
    self.target_update_freq = 10 # updating every 10 episodes
    self.epsilon_start = 1.0
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.epsilon = self.epsilon_start
    self.gamma = gamma
  
  def epsilon_greedy_action(self, state ,epsilon = EPSILON):
    if np.random.rand() < epsilon: # Go for Exploration
      return np.random.choice(ACTIONS)
    else: # go for exploitation
      state = torch.tensor(state)
      #print('tensor shape: ', state.shape)
      return torch.argmax(self.Q(state)).item()
  
#  def get_episodes(self):
#    sarsa_list = []
#   initial_state = GAME.reset()
#    is_game_over = False
#    while not is_game_over:
#      in_state = GAME.get_state()
#      initial_action = self.epsilon_greedy_action(state = in_state)
#      nxt_state , reward, is_game_over = GAME.step(initial_action)
#      sarsa_tuple = (initial_state, initial_action, reward, nxt_state)
#      sarsa_list.append(sarsa_tuple)
    
#    return sarsa_list

  #def get_episodes(self):
  #  sarsa_list = []
  #  initial_state = GAME.reset()
  #  is_game_over = False
  #  while not is_game_over:
  #    for event in pygame.event.get():  # <-- this line prevents freezing
  #      if event.type == pygame.QUIT:
  #        pygame.quit()
  #        exit()

  #    in_state = GAME.get_state()
  #    initial_action = self.epsilon_greedy_action(state=in_state, epsilon=self.epsilon)
  #    nxt_state, reward, is_game_over, score = GAME.step(initial_action)
  #    sarsa_tuple = (in_state, initial_action, reward, nxt_state, score)
  #    sarsa_list.append(sarsa_tuple)
  #  return sarsa_list
  
  def get_and_store_experience(self):
    state = GAME.reset()
    done = False
    score = 0

    while not done:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          pygame.quit()
          exit()
        
      action = self.epsilon_greedy_action(state, epsilon=self.epsilon)
      next_state, reward, done, score = GAME.step(action)
      self.replay_buffer.push(state, action, reward, next_state, done)
    
    return score
  
  def load_model(self, model, optimizer, path = 'best_Q_fn.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint.get('step',0)
    print(f"Loaded checkpoint from '{path}'")
    return step
  
  def save_model(self, episode, optimizer):
        torch.save({
      'episode': episode,
        'model_state_dict':self.Q.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },'best_Q_fn.pth')
            
  def train(self, num_episodes=NUM_EPISODES, start_episode = 0):

    optimizer = Adam(self.Q.parameters(), lr=LR)
    loss_fn = weighted_MSE(normalize= True)
    
    try:
      start_episode = self.load_model(self.Q, optimizer)
    except FileNotFoundError:
      print('No saved model found. starting from scratch')
      
  
    for episode in range(start_episode, num_episodes):
      score_val = self.get_and_store_experience()
      if len(self.replay_buffer) >= self.batch_size:
        batch = self.replay_buffer.sample(self.batch_size)

        batch_loss = 0
        best_score = 0
        for state, action, reward, next_state, done in batch:
          state_tensor = torch.tensor(state, dtype = torch.float32)
          next_state_tensor = torch.tensor(next_state, dtype = torch.float32)
          
          q_values = self.Q(state_tensor)
          q_pred = self.Q(state_tensor)[action]

          with torch.no_grad():
            target_q_values = self.Q_target(next_state_tensor)
            q_target = reward if done else reward + self.gamma * torch.max(target_q_values)

          loss = loss_fn(q_pred, q_target,len(batch), score_val)

          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
          optimizer.step()

          batch_loss += loss.item()

          if episode % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
          self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

          if score_val > best_score:
            best_score = score_val
            self.save_model(episode, optimizer)
        print(f"Episode {episode} Score: {score_val} | Loss: {batch_loss:.3f}")
    

if __name__ == '__main__':
  q_net = DQN()
  q_net.train()

