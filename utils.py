import numpy as np
import torch
import torch.nn as nn
def prob_select(p):
    summ=0
    r=np.random.rand()
    for i in range(p.shape[0]):
        summ=summ+p[i]
        if r<summ:
            return i

#The reason to define cross entropy is that the nn.CrossEntropyLoss only eat class=? as input but cannot put a distribution in it
def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    #print(- soft_targets )
    #print('pred',pred)
    #print('logsoftmax(pred)',logsoftmax(pred))
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))