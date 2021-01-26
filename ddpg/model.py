import torch
import torch.nn as nn
import random
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions , hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    
    def forward(self, state, action):
        out = self.fc1(state)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,action],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
























# class Actor(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Actor, self).__init__()
#         self.h1 = nn.Linear(input_size, hidden_size)
#         self.h2 = nn.Linear(hidden_size, output_size)
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.h1(x)
#         x = self.relu(x)
#         x = self.h2(x)
#         x = self.tanh(x)
#         return x




# class Critic(nn.Module):
#     def __init__(self, state_input_size, action_input_size, hidden_size, output_size):
#         super(Critic, self).__init__()
#         self.h1 = nn.Linear(state_input_size, hidden_size)
#         self.h2 = nn.Linear(hidden_size + action_input_size, output_size)
#         self.relu = nn.ReLU()
#     def forward(self, state, action):
#         x = self.h1(state)
#         x = self.relu(x)
#         x = torch.cat([x, action], dim = 1)
        
#         x = self.h2(x)
        
#         return x


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.idx   = 0

        self.state_buffer       = np.zeros((capacity, dims))  
        self.action_buffer      = np.zeros((capacity, 1))
        self.reward_buffer      = np.zeros((capacity, 1))
        self.next_state_buffer  = np.zeros((capacity, dims))  
        

    def __len__(self):
        return len(self.state_buffer)

    def store_transition(self, state, action, reward, next_state):
        
        self.idx %= self.capacity

        state = state.reshape(-1)
        action = action.reshape(-1)
        reward = reward
        next_state = next_state.reshape(-1)
        
    
        self.state_buffer[self.idx]         = state       
        self.action_buffer[self.idx]        = action      
        self.reward_buffer[self.idx]        = reward      
        self.next_state_buffer[self.idx]    = next_state  

        self.idx +=1



        # if(len(self.state_buffer) > self.capacity):
        #     self.state_buffer       = self.state_buffer[1:]
        #     self.action_buffer      = self.action_buffer[1:]
        #     self.reward_buffer      = self.reward_buffer[1:]
        #     self.next_state_buffer  = self.next_state_buffer[1:]
        
        
    def sample(self, n):
        idx = random.sample([i for i in range(len(self.state_buffer))], n)
        return np.array(self.state_buffer)[idx], np.array(self.action_buffer)[idx],\
             np.array(self.reward_buffer)[idx], np.array(self.next_state_buffer)[idx]