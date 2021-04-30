import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')

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
"""


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