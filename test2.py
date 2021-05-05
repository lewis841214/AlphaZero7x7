import numpy as np
import torch

"""
Test for kronecker gradient
"""
print(torch.__version__)
m=7
p=8
q=9
p0=6
q0=5
A= torch.rand(m, p*q, requires_grad=False)
x=torch.rand(p,p0, requires_grad=True)
y=torch.rand(q,q0, requires_grad=False)
B=torch.rand(p0*q0, p0*q0, requires_grad=False)
print(torch.kron(A, B))
check1=torch.matmul(torch.transpose(torch.kron(x,y), 0, 1),torch.transpose(A,0,1))
check2=torch.matmul(B,check1)
check3=torch.matmul(torch.kron(x,y),check2)
check4=torch.matmul(A,check3)
loss=torch.trace(check4)
loss.backward()
print('x grad',x.grad)

print(loss.grad_fn)
#print(loss.grad_fn.next_functions)
#print(loss.grad_fn.next_functions[0][0])
#print(loss.grad_fn.next_functions[0][0].next_functions)
#print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions)
#print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions)
#print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions)

#print(x.grad_fn)
#print('grad',x.grad)

#x=x-0.0001*x.grad
    
def p_grad_fn(node,layer=0):
    print('_'+'_'*layer,node)
    if node !=None:
            
        for XX in node.next_functions:
            p_grad_fn(XX[0],layer+1)
p_grad_fn(loss.grad_fn)




"""
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

"""


