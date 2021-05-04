import numpy as np
a=np.array([i+6 for i in range(5)])
a=a/np.sum(a)

print(a)
def prob_select(p):
    summ=0
    r=np.random.rand()
    for i in range(p.shape[0]):
        summ=summ+p[i]
        if r<summ:
            return i


all=[0 for i in range(5)]
for i in range(20000):
    all[prob_select(a)]+=1
all=np.array(all)/sum(all)
print(all)