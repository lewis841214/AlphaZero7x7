import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
go_env = gym.make('gym_go:go-v0', size=2, komi=0, reward_method='real')
go_env.step(0)
go_env.step(4)
state,_,_,_=go_env.step(3)
go_env.render('terminal')
print(state[3])
state, reward, done, info=go_env.step(4)
go_env.render('terminal')
print(state[3])
"""
for i in range(45):
    state, reward, done, info = go_env.step(i)
    #print(state[3])
    #print(np.sum(state[3]))
state, reward, done, info = go_env.step(49)
print(reward, done)
state, reward, done, info = go_env.step(49)
print(reward, done)

first_action = (2,5)
state, reward, done, info = go_env.step(first_action)
print(state[3])
inv=np.array(state[3])
print(type(inv))
print(inv.reshape(-1))
go_env.render('terminal')

state, reward, done, info = go_env.step(20)
inv=np.array(state[3])

print(inv.reshape(-1))
go_env.render('terminal')

go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')
go_env.state_=state
#go_env.reward=reward
#go_env.done=done
#go_env.info=info
go_env.render('terminal')
state, reward, done, info = go_env.step(48)
go_env.render('terminal')
state, reward, done, info = go_env.step(47)
go_env.render('terminal')
"""