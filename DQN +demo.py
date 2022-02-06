# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:02:30 2020

@author: Hang Yu
"""
import gym
import torch
import random
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import est_agent as EAgent
import fb_agent as FAgent

sum_of_feedback = 1000


env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


oracle = Agent(state_size=8, action_size=4, seed=0)
oracle.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# watch an untrained agent


def dqn(n_episodes=200, max_t=1000, eps_start=0.01, eps_end=0.01, wt_start = 1, wt_end = 0.1, 
        eps_decay=30000, wt_decay = 100000, model = 0):
    """
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    epsilon_by_frame = lambda frame_idx: eps_end + (eps_start - eps_end) * math.exp(
            -1. * frame_idx / eps_decay) # decrease epsilon
    weight_by_frame = lambda frame_idx: eps_end + (wt_start - wt_end) * math.exp(
            -1. * frame_idx / eps_decay) # decrease epsilon
    sof = sum_of_feedback * model
    agent = Agent(state_size=8, action_size=4, seed=int(100000*random.random()))
    fb_agent = FAgent.Agent(state_size=8, action_size=4, seed=int(100000*random.random()))
    f_agent = EAgent.EstAgent(state_size=8, action_size=4, seed=int(100000*random.random()))
    est_agent = EAgent.EstAgent(state_size=8, action_size=4, seed=int(100000*random.random()))
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=200)  # last 200 scores
    eps = eps_start                    # initialize epsilon
    cnt = 0
    
    for i in range(model):
        if sof <=0:
            break
        for i in range(max_t):
            state = env.reset()
            sof -= 1
            action = np.random.choice([0,1,2,3], p = oracle.act(state, eps) )
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done or sof <= 0:
                break 
        
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        # if i_episode % (( n_episodes + 1)/model) == 0:
        #     for i in range(max_t):
        #         sof -= 1
        #         action = np.random.choice([0,1,2,3], p = oracle.act(state, eps) )
        #         next_state, reward, done, _ = env.step(action)
        #         agent.step(state, action, reward, next_state, done)
        #         #fb_agent.step(state, action, 1, next_state, done)
        #         state = next_state
        #         if done:
        #             state = env.reset()
        #             break 
        
        score = 0

        for t in range(max_t):
            
            cnt+=1
            prob  =  agent.act(state, eps) #+ fb_agent.act(state,0) #** weight_by_frame(cnt)
            action = np.random.choice([0,1,2,3], p = prob/sum(prob))
            #action = np.random.choice(np.flatnonzero(prob == prob.max()))
            feedback = 0
            
            # if cnt % 20 == 0: # advice 
            #     feedback = 1
            #     action = np.random.choice([0,1,2,3], p = oracle.act(state, eps))
            #advice----------------------------------------
                
            next_state, reward, done, _ = env.step(action)
            
            # if oracle.judge(next_state) != 0:
            #     #print("hhhhhhhhhhhhhhh")
            #     f_agent.step(next_state, 0, oracle.judge(next_state), next_state, done)
            # if f_agent.act(next_state) != 0:
            #     fb_agent.step(state, action, float(f_agent.act(next_state)), next_state, done)
            # flag---------------------------------doesnt work for now
            
            
            
            
            # if cnt % 21 == 0:
            #     if action == np.random.choice([0,1,2,3], p = oracle.act(state, eps)):
            #         feedback += 1
            #     else:
            #         feedback  -= 1
            #evaluation-------------------------------
            
            
            
            score += reward #
            # buffer.auto_add(state, reward, next_state, done)
            
            
            # reward += float(est_agent.act(next_state) - est_agent.act(state))
            # if cnt % 22 == 0  and sof > 0:
            #     sof-= 1
            #     rwd = oracle.estimate(state)
            #     est_agent.step(state, 0, rwd, next_state,done)
            #estimation-----------------------------
            
            
            # if feedback != 0 and sof > 0:
            #     sof -= 1
            #     fb_agent.step(state, action, feedback, next_state, done)
            #feedback learning--------------------------
                
            agent.step(state, action, reward, next_state, done)
            
            # else:
            #     reward = 0 
            state = next_state

            if done:
                break 
        
        
        # for i  in range(buffer.top):
        #     reward =  (i+1)*score/buffer.top
        #     state, action, _, next_state, done = buffer.auto_memory[i]
        #     est_agent.step(next_state,  0, reward, next_state, done)
            
            #est_agent.step(buffer[i].auto_memory[0], 0, reward, buffer[i].auto_memory[3], buffer[i].auto_memory[4])
            #est_agent
        # print(score)
        
        
        
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = epsilon_by_frame(cnt) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} \tCrt Score: {:.2f} \tepsilon: {:.4f}'.format(i_episode, np.mean(scores_window),score,eps), end="")
        if i_episode % 200 == 0:
            print('\rEpisode {}  \tAverage Score: {:.2f} \tCrt Score: {:.2f}'.format(i_episode, np.mean(scores_window),score))
            #torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            
            #break
    return scores


length = 500
times = 10
M=[0 for i in range(length)]
S=[0 for i in range(length)]
N=[0 for i in range(length)]
R=[0 for i in range(length)]

for i in range(times):
    print('\n',i,"-th Trial")
    M = np.sum([dqn(n_episodes=length,model = 1),M], axis=0)
    S = np.sum([dqn(n_episodes=length,model = 2),S], axis=0)
    N = np.sum([dqn(n_episodes=length,model = 5),N], axis=0)
    R = np.sum([dqn(n_episodes=length,model = 1),R], axis=0)


x=[i+1 for i in range(length)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,M/times,color='green',label='M')
plt.plot(x,S/times,color='red',label='S')
plt.plot(x,N/times,color='yellow',label='N')
plt.plot(x,R/times,color='pink',label='R')

plt.legend()


res=M/times
f = open('DQN+DEMO_1.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 


res=S/times
f = open('DQN+DEMO_5.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 

res=N/times
f = open('DQN+DEMO_10.txt', 'w') 
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 

res=R/times
f = open('DQN+DEMO_100.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 


