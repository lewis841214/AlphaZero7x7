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
    mcts=MCTS(f.forward)

    