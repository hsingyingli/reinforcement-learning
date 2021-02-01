import numpy as np 
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from tqdm  import tqdm
from learner import *
from model import *


class DDPG(object):
    def __init__(self, args, config):
        #----------------- hyper parameters -----------------
        self.batch_size     = args.batch_size
        self.tau            = args.tau
        self.gamma          = args.gamma
        self.max_episode    = args.max_episode
        self.critic_lr      = args.critic_lr
        self.actor_lr       = args.actor_lr
        self.meta_lr        = args.meta_lr
        self.epsilon        = args.epsilon
        self.deepsilon      = args.deepsilon
        self.state_space    = args.feature_size
        self.action_space   = 1
        self.device         = torch.device(args.device)


        # Components
        self.env           =    None
        self.replay_buffer = Memory(args.capacity, self.state_space)
        self.criterion     = nn.MSELoss()
        # self.writer        = SummaryWriter('runs/'+ str(args.exp_id), comment = args.env)
        #------------------ Actor ------------------
        self.online_actor  = Learner(config)
        self.target_actor  = Learner(config)
        self.actor_optim   = torch.optim.Adam(self.online_actor.parameters(), lr = args.meta_lr)

        #------------------ Critic ------------------
        self.online_critic = Critic(self.state_space, self.action_space, args.hidden_size, 1).to(self.device)
        self.target_critic = Critic(self.state_space, self.action_space, args.hidden_size, 1).to(self.device)
        self.critic_optim  = torch.optim.Adam(self.online_critic.parameters(), lr = args.critic_lr)

        #  make target network equal to online network
        self.hard_update(self.target_actor, self.online_actor)
        self.hard_update(self.target_critic, self.online_critic)
        self.update_counter = 0


    def soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(), self.online_critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.online_actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, state, fast_weight):
        
        state  = torch.FloatTensor(state).to(self.device)
        action = self.online_actor(state, fast_weight)
        action = action.cpu().detach().numpy().squeeze(0)
        action = (action + np.random.normal(0.5, self.epsilon, size=self.action_space)).clip(
                    self.env.action_space.low, self.env.action_space.high)
    
        return action
    
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch):
        
        # --------------  update critic -----------------
        with torch.no_grad():
            action       = self.target_actor(next_state_batch)
            
            next_q_value = self.target_critic(next_state_batch, action)
            target_q_batch = reward_batch + self.gamma  * next_q_value

        q_batch = self.online_critic(state_batch, action_batch)

        loss = self.criterion(q_batch, target_q_batch)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        self.writer.add_scalar("Critic loss", loss.cpu().item(), self.update_counter)
                    
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, fast_weight):
        # --------------  update actor -----------------
        action = self.online_actor(state_batch, fast_weight)
        loss = - self.online_critic(state_batch, action).mean()

        # self.actor_optim.zero_grad()
        # loss.backward()
        # self.actor_optim.step()
        # self.writer.add_scalar("Actor loss", loss.cpu().item(), self.update_counter)
        return loss

    def update_representation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def inner_update(self, loss, fast_weight):
        adaptation_weight_counter = 0

        grad = torch.autograd.grad(loss, self.online_actor.get_adaptation_parameters(fast_weights),
                                   create_graph=True)
        new_weights = []

        # --------------  only update prediction layer ------------------

        for p in fast_weights:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                temp_weight = p - self.actor_lr * g
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

    def solve(self):
        total_loss = []
        fast_weight = self.online_actor.parameters()

        for episode in range(self.max_episode):
            state    = self.env.reset()
            episode_reward = 0.   
            finish = False     
            # every steps:
            while not finish:
                #1. get_action + noise
                action = self.choose_action(state, fast_weight)    

                #2. do the action return next state, reward, is_done
                next_state, reward, is_done, info = self.env.step(action)

                if is_done:
                    finish = True
                    continue

                #3. store transition (state, action, reward, next_state)
                self.replay_buffer.store_transition(state, action, reward, next_state)
                
                if self.replay_buffer.pointer > self.replay_buffer.capacity:
                    #4. update            
                    #4.1 random sample from replay buffer
                    
                    state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample(self.batch_size)
                    
                    state_batch         = torch.FloatTensor(state_batch).to(self.device)
                    action_batch        = torch.FloatTensor(action_batch).to(self.device)
                    reward_batch        = torch.FloatTensor(reward_batch).to(self.device)
                    next_state_batch    = torch.FloatTensor(next_state_batch).to(self.device)

                    #4.2 update online critic 
                    self.update_critic(state_batch, action_batch, reward_batch, next_state_batch)
                    
                    #4.3 update online actor using policy gradient
                    loss = self.update_actor(state_batch, action_batch, reward_batch, next_state_batch, fast_weight)
                    total_loss.append(loss)

                    fast_weights = self.inner_update(loss, fast_weight)

                    #4.4 update target critic / actor
                    self.soft_update()
                    self.epsilon *= self.deepsilon
                    self.update_counter += 1
                
                #5. state = next_state, totoal reward += reward
                state = next_state
                episode_reward += reward
             
            print("||Episode: %3d || Episode reward: %.6f"%(episode, episode_reward))
            
        
        return total_loss
       


