import gym
import numpy as np
from pptree import *

size=2
class Tree:
    def __init__(self,parent ,added_position ,playing_env,env , size , F,layer=0, Done=False, lambda_=1, gamma=1):
        self.action_num=size**2+1 # board size size^2 + pass
        self.size=size
        self.layer=layer
        self.gamma=gamma
        self.lambda_=lambda_
        self.playing_env=playing_env
        self.env=env
        self.State=env.state_
        self.F=F
        self.name=str(added_position)
        #print()
        #print('here',F(self.State))
        self.p,self.v=F(self.State) #This function will load P from Network
        self.child=np.array([None for i in range(self.action_num)])
        self.parent=None
        self.W=np.array([0 for i in range(self.action_num)], dtype=np.float32)
        #Q is a action value function
        self.act_Q=np.array([0 for i in range(self.action_num)], dtype=np.float32)
        self.N=np.array([0 for i in range(self.action_num)])
        
        self.add=added_position
        self.Done=Done
        self.z=None
        # This show the invalid index, then when selection wants to go to here, it skip the invalid step
        self.invalid=self.State[3].reshape(-1)
        self.Seq_recorder=[]
        #self.visual=[self.v,self.W, self.N]
        self.visual=" ".join(str(x) for x in self.W)
        
    def back_up(self):
        #print('in back_up')
        #現在back up 因為變成action value，action value 會attach 在parent身上。所以往回
        #送的時候，可能還要給出 node number之類的

        
        cur=self.parent
        position=self.add
        value=self.v
        print('position',position,'value',value)
        value=-value
        while cur!=None:
            #print('in back_up loop')
            
            #print('cur.W[position]',cur.W[position])
            cur.W[position]=cur.W[position]+value
            #print('cur.W[position]',cur.W[position])
            cur.N[position]+=1
            cur.act_Q[position]=cur.W[position]/cur.N[position]
            position=cur.add
            print('position',position,)
            #print('\n hi',cur.act_Q[position],cur.W[position],cur.N[position],'\n')
            """
            Here add some recorded
            """
            cur.visual=" ".join(str(x) for x in cur.W)
            cur=cur.parent
            value=-value

    def selection(self):
        #print(self.parent)
        #This function goes to some node and select a edge with no attached node.
        #But we should consider the stopping criteria
        #We should write a recurssive function to deal with this
        #
        
        #1. 如果自己已經有找到空的node 直接expand
        #2. 如果自己是找到下一個node recursive 到下一個node
        if self.Done==False:
        
            self.S_select=self.act_Q+self.lambda_*(self.p/(1+self.N))
            #print(self.S_select)
            #print(np.argmax(self.S_select))
            soted_index=np.argsort(self.S_select)
            soted_index=np.flip(soted_index)

            #print('S_select',self.S_select)
            #print('sorted index',soted_index)
            for i in range(self.action_num):
                if soted_index[i]==self.size**2:
                    self.expand(soted_index[i])
                    return True
                elif self.invalid[soted_index[i]]==True:
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
        self.env.render('terminal')
        self.child[added_position]=Tree(self, added_position ,self.playing_env,self.env , self.size, F=self.F, layer=self.layer+1)
        self.child[added_position].parent=self
        self.child[added_position].back_up()
    def play(self):
        if np.sum(self.invalid)==self.size**2:
            next_action=self.action_num
            self.pi=np.array([0 for i in range(self.size**2+1)])
            self.pi[-1]=1
        else:
            self.pi=self.N**self.gamma/np.sum(self.N**self.gamma)
            next_action=np.argmax(self.pi)
        #有了pi之後 我們就可以把pi record下來。每個state 會有一個對應到的pi 跟z 
        self.Seq_recorder.append([self.State, self.pi])



        #接下來 就是用把self.playing_env.step(next_action)
        #並偵測是否已經結束。 已經結束的話 就可以開始製作sequence
        #若還沒結束的話，將now_root移到選擇的那個action
        print(next_action)
        state, reward, done, info=self.playing_env.step(next_action)
    
        if done==1:
            #這邊開始將結果"reward"=[-1 or 1] 1 是黑棋贏的時候塞回去sequence中。
            #(S0, pi0) 為function第一個衡量的情況，也就是黑棋贏的機率
            for i in range(len(self.Seq_recorder)):
                self.Seq_recorder[i].append(reward)
                reward=-reward
            return None
        else:
            return self.child[next_action]
        
    def clear_None(self):
        
        self.child_none_out=self.child[self.child != np.array(None)]
        for i in range(self.child_none_out.shape[0]):
            self.child_none_out[i].clear_None()
            
#Here create a function output just like Neural network
def f(State_):
        p=np.random.multinomial(109, [1/(size**2+1)]*(size**2+1))
        
        p=p/np.sum(p)
        #print('p',p)
        v=(np.random.rand()-0.5)*2
        return p,v

class MCTS():
    def __init__(self):
        go_env = gym.make('gym_go:go-v0', size=2, komi=0, reward_method='real')
        playing_env = gym.make('gym_go:go-v0', size=2, komi=0, reward_method='real')
        root=Tree(parent=None ,added_position=None ,playing_env=playing_env,env=go_env , size=2 , F=f)
        #root.clear_None()
        #print_tree(root, childattr='child_none_out', nameattr='visual', horizontal=False)
        for i in range(50):
            root.selection()
            #root.clear_None()
            #print_tree(root, childattr='child_none_out', nameattr='visual', horizontal=False)
        root.play()
        #root.clear_None()
        #print_tree(root, childattr='child_none_out', nameattr='visual', horizontal=False)
        """
        print(root.child)
        root=root.child_none_out[0]
        print(root.child)
        """
    

if __name__ == '__main__':
    #First create a neural network which will output prabability distribution and action|state value
    mcts=MCTS()
    

