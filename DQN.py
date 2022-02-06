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


sum_of_feedback = 1000


env = gym.make('LunarLander-v2')



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

    agent = Agent(state_size=7, action_size=12, seed=0)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=200)  # last 200 scores
    eps = eps_start                    # initialize epsilon
    cnt = 0
    

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0

        for t in range(max_t):
            
            cnt+=1
            prob  = agent.act(state, eps)#fb_agent.act(state,0) #*  #* weight_by_frame(cnt)
            action = np.random.choice([i for i in range(12)], p = prob/sum(prob))
            #action = np.random.choice(np.flatnonzero(prob == prob.max()))


                
            next_state, reward, done, _ = env.step(action)

            
            score += reward #


            agent.step(state, action, reward, next_state, done)
            

            state = next_state

            if done:
                break 
        
        
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = epsilon_by_frame(cnt) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} \tCrt Score: {:.2f} \tepsilon: {:.4f}'.format(i_episode, np.mean(scores_window),score,eps), end="")


    return scores


length = 500
times = 10
M=[0 for i in range(length)]


for i in range(times):
    print('\n',i,"-th Trial")
    M = np.sum([dqn(n_episodes=length,model = 1),M], axis=0)
    # S = np.sum([dqn(n_episodes=length,model = 5),S], axis=0)
    # N = np.sum([dqn(n_episodes=length,model = 10),N], axis=0)
    # R = np.sum([dqn(n_episodes=length,model = 100),R], axis=0)


x=[i+1 for i in range(length)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,M/times,color='green',label='M')
# plt.plot(x,S/times,color='red',label='S')
# plt.plot(x,N/times,color='yellow',label='N')
# plt.plot(x,R/times,color='pink',label='R')

plt.legend()


res=M/times
f = open('DQN.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 





