import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')

first_action = (2,5)
second_action = (5,2)
first_action1 = (3,5)
second_action1 = (4,2)
state, reward, done, info = go_env.step(first_action)
#print(state)
state, reward, done, info = go_env.step(second_action)
#print(state)
state, reward, done, info = go_env.step(first_action1)
#print(state)
state, reward, done, info = go_env.step(second_action1)
#print(state)
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

