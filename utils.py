import numpy as np
def prob_select(p):
    summ=0
    r=np.random.rand()
    for i in range(p.shape[0]):
        summ=summ+p[i]
        if r<summ:
            return i