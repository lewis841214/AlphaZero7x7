import gym
import numpy as np
from pptree import *

size=7
class Tree:
    def __init__(self,parent ,added_position ,env , size , F, Done=False, lambda_=0.5, gamma=1):
        self.size=size
        self.gamma=gamma
        self.lambda_=lambda_
        self.env=env
        self.data = None
        self.State=env.state_
        self.F=F
        self.name=str(added_position)
        #print()
        #print('here',F(self.State))
        self.p,self.v=F(self.State) #This function will load P from Network
        
        self.child=np.array([None for i in range(size**2)])
        self.parent=None
        self.W=0
        #Q is a action value function
        self.act_Q=np.array([0 for i in range(size**2)])
        self.N=np.array([0 for i in range(size**2)])
        
        self.add=added_position
        self.Done=Done
        self.z=None
        # This show the invalid index, then when selection wants to go to here, it skip the invalid step
        self.invalid=self.State[3].reshape(-1)
        
    def back_up(self):
        #現在back up 因為變成action value，action value 會attach 在parent身上。所以往回
        #送的時候，可能還要給出 node number之類的

        
        cur=self.parent
        position=self.add
        while cur!=None:
            cur.W[position]+=self.v
            cur.N[position]+=1
            cur.Q[position]=cur.W[position]/cur.N[position]
            position=cur.add
    def selection(self):
        #This function goes to some node and select a edge with no attached node.
        #But we should consider the stopping criteria
        #We should write a recurssive function to deal with this
        #
        
        #1. 如果自己已經有找到空的node 直接expand
        #2. 如果自己是找到下一個node recursive 到下一個node
        #3. 如果都找不到空的node 下一個也都回傳Done=False 回傳False
        #但其實還要考慮到 不能走的step要怎麼表示的問題。
        if self.Done==False:
            self.S_select=self.act_Q+self.lambda_*(self.p/(1+self.N))
            soted_index=np.argsort(self.S_select)
            soted_index=np.flip(soted_index)
            for i in range(self.size**2):
                if self.invalid[i]==True:
                    #Then jump to next i
                    pass
                elif self.child[soted_index[i]]==None:
                    #Then Do expand
                    self.expand(soted_index[i])
                    return True
                elif self.child[soted_index[i]]!=None:
                    Done=self.child[soted_index[i]].selection()
                    if Done==True:
                        return True
                
    def expand(self, added_position):
        #first put 
        self.env.state_=self.State
        self.env.step(added_position)
        self.child[added_position]=Tree(self, added_position ,self.env , self.size, F=self.F)

    def play(self):
        self.pi=self.N**self.gamma/np.sum(self.N**self.gamma)
        next_action=np.max(self.pi)
    def clear_None(self):
        self.child_none_out=self.child[self.child != np.array(None)]
        for i in range(self.child_none_out.shape[0]):
            self.child_none_out[i].clear_None()
            
#Here create a function output just like Neural network
def f(State_):
        p=np.random.multinomial(107, [1/size**2]*size**2)
        #print(p)
        p=p/np.sum(p)
        v=(np.random.rand()-0.5)*2
        return p,v

class MCTS():
    def __init__(self):
        go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')
        root=Tree(parent=None ,added_position=None ,env=go_env , size=7 , F=f)
        for i in range(4):
            root.selection()
        root.clear_None()
        print_tree(root, childattr='child_none_out', nameattr='name', horizontal=False)
    

if __name__ == '__main__':
    #First create a neural network which will output prabability distribution and action|state value
    mcts=MCTS()
    

