from MCTS import *
from utils import *
from model import *
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import gym
from utils import *
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def f_random(State_):
    #pp=[1/(size**2+1)]*(size**2+1)
    p=np.random.multinomial(1090, [1/(size**2+1)]*(size**2+1))
    #rint('p=',p)
    p[-1]=0
    p=p/np.sum(p)
    #print('p',p)
    v=(np.random.rand()-0.5)*2
    return p,v
class f_():
    def __init__(self,model):
        self.model=model
        self.model=self.model.to(device)
    def forward(self,state):
        state=torch.tensor(state[:3], dtype=torch.float).to(device)
        state=torch.unsqueeze(state, 0)
        state=state.to(device)
        out_p,out_v=self.model(state.to(device))
        out_p=out_p.clone().detach().cpu()[0].numpy()
        out_v=out_v.clone().detach().cpu()[0].numpy()
        return out_p, out_v
"""

def f_(state):
    model=model.to(device)
    state=torch.tensor(state[:3], dtype=torch.float).to(device)
    state=state.to(device)
    out_p,out_v=model(state.to(device))
    out_p=out_p.clone().detach().cpu()[0].numpy()
    out_v=out_v.clone().detach().cpu()[0].numpy()
    return out_p, out_v
"""
if __name__ == '__main__':
    start=time.time()        
    Generation='random' # 'random', 'by_network'
    if Generation=='by_network':
        Res_num=10
        hidden=100
        width=7
        num_of_select=400
        PATH='test_output'
        model = Net(Res_num,hidden, width)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        model.to(device)
        f=f_(model)
        for i in range(1000):
            mcts=MCTS(f.forward)
            print('time',time.time()-start)
    elif Generation=='random':
        for i in range(1000):
            mcts=MCTS(f_random)
            print('time',time.time()-start)
    
    
    print('time',time.time()-start)

    