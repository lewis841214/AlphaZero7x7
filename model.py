import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import gym
from utils import *

import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0')

class Residual_struct(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        
        self.fitst=nn.Sequential(
          nn.Conv2d(hidden,hidden,3, padding=1), #input ch=3 [balck, white, whos tern]
          nn.BatchNorm2d(hidden),
          nn.ReLU(),
        )
        self.second=nn.Sequential(
          nn.Conv2d(hidden,hidden,3, padding=1),
          nn.BatchNorm2d(hidden),
        )
        self.third=nn.Sequential(
          nn.Conv2d(hidden,hidden,3, padding=1), 
          nn.BatchNorm2d(hidden),
          nn.ReLU(),
        )

    def forward(self, x):
        y=self.fitst(x)
        y=self.second(y)
        y=x+y
        y=self.third(y)
        return y

class Net(nn.Module):
    def __init__(self, Res_num, hidden, width):
        super().__init__()
        self.fitst=nn.Sequential(
            nn.Conv2d(3,hidden,3, padding=1), #input ch=3 [balck, white, whos tern]
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.Res_num=Res_num
        self.Rew_list=[Residual_struct(hidden).to(device) for i in range(Res_num) ]
        self.Policy=nn.Sequential(
            nn.Conv2d(hidden,2,1), #input ch=3 [balck, white, whos tern]
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width*width*2, 722),
            nn.ReLU(),
            nn.Linear(722, width**2+1),# output logits
            #nn.Softmax(),
        )
        self.Value=nn.Sequential(
            nn.Conv2d(hidden,1,1), #input ch=3 [balck, white, whos tern]
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width*width*1, 361),
            nn.ReLU(),
            nn.Linear(361, 256),
            nn.ReLU(),
            nn.Linear(256, 1),# output logits
            nn.Tanh()
        )

    def forward(self, x):
        x=self.fitst(x)
        for i in range(self.Res_num):
            x=self.Rew_list[i](x)
        p=self.Policy(x)
        v=self.Value(x)
        return p,v
def f_(state,model):
    model=model.to(device)
    state=torch.tensor(state[:3], dtype=torch.float).to(device)
    state=state.to(device)
    #state=torch.unsqueeze(state, 0)
    #print(state.shape)
    #print(model)
    #print(state)
    out_p,out_v=model(state.to(device))
    out_p=out_p.clone().detach().cpu()[0].numpy()
    out_v=out_v.clone().detach().cpu()[0].numpy()
    ##print(out_p)
    #print(out_v)
if __name__ == '__main__':
    go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='heuristic')
    state,reward,_,_=go_env.step(0)
    state=torch.tensor(state[:3], dtype=torch.float).to(device)
    state=torch.unsqueeze(state, 0)

    Res_num=10
    hidden=100
    width=7
    model = Net(Res_num,hidden, width).to(device)
    model=model.to(device)
    #print(net(state))
    
    f_(state,model)













"""

    true_p=torch.tensor([[0 for i in range(width**2+1)]], dtype=torch.float,requires_grad=False)
    #true_p=true_p/torch.sum(true_p)
    true_p[0][0]=1
    true_v=torch.tensor([[-0.5]], dtype=torch.float,requires_grad=False)
    output,output_v=model(state)
    #print(output_v)
    #print(output.shape)
    #print(true_p.shape)


    #print(cross_entropy(output,true_p))
    print(output_v,true_v)
    print('torch.sum((output_v-true_v)**2)',torch.sum((output_v-true_v)**2))
    loss=cross_entropy(output,true_p)+torch.sum((output_v-true_v)**2)
    
    print('loss',loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-4)

    for i in range(10):
        output,output_v=model(state)
        loss=cross_entropy(output,true_p)+torch.sum((output_v-true_v)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        print(output, output_v)
        print('true_v',true_v)
        print('output_v',output_v)
        print('right term',torch.sum((output_v-true_v)**2))
    PATH='test_output'
    torch.save(model.state_dict(), PATH)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    """