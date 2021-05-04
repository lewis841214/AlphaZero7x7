import gym
import numpy as np
from pptree import *
import os
import time
from utils import *
size=7
class Tree:
    def __init__(self,parent ,added_position ,playing_env,env ,seq_reocord, size , F,layer=0, Done=False, lambda_=1, gamma=1):
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
        self.seq_reocord=seq_reocord
        #self.visual=[self.v,self.W, self.N]
        self.visual=" ".join(str(x) for x in self.W)
        
    def back_up(self,now_node):
        #print('進入back up')
        #print('in back_up')
        #現在back up 因為變成action value，action value 會attach 在parent身上。所以往回
        #送的時候，可能還要給出 node number之類的

        
        cur=self.parent
        position=self.add
        value=self.v
        #print('\nposition',position,'value',value)
        value=-value
        while cur!=now_node.parent:
            #print('in back_up loop')
            
            #print('cur.W[position]',cur.W[position])
            cur.W[position]=cur.W[position]+value
            #print('cur.W[position]',cur.W[position])
            cur.N[position]+=1
            cur.act_Q[position]=cur.W[position]/cur.N[position]
            position=cur.add
            #print('position',position,)
            #print('\n hi',cur.act_Q[position],cur.W[position],cur.N[position],'\n')
            """
            Here add some recorded
            """
            cur.visual="N="+",".join(str(x) for x in cur.N)+"W="+",".join(str(x) for x in cur.W)+"p="+",".join(str(x) for x in cur.p)
            cur=cur.parent
            value=-value
        #print('end back up')

    def selection(self, now_node):
        #print(self.parent)
        #This function goes to some node and select a edge with no attached node.
        #But we should consider the stopping criteria
        #We should write a recurssive function to deal with this
        #
        
        #1. 如果自己已經有找到空的node 直接expand
        #2. 如果自己是找到下一個node recursive 到下一個node
        self.env.state_=self.State
        #print('進入selection')
        #self.env.render('terminal')
        if self.Done!=1:
            #如果說 上一步來的是pass且我們這邊的局勢比較好，則我們也pass
            if self.add==self.size**2:
                #print('進入領域')
                #這邊有一點怪怪的  因為在接近尾聲的時候只要有invalid 步 就會導致
                #接下來偵測換黑或是白子
                #np.sum(self.state[2])>0換白子 np.sum(self.state[2])==0 換黑子=>np.sum(self.state[2])-1<0
                #reward>0 代表黑子贏 <0代表白子贏
                #所以說(np.sum(self.state[2])-1)*reward<0 => 換黑子(-) 黑贏(+) or 換白子 (+) 白贏 (-) 
                if (np.sum(self.State[2])-1)*self.reward<0:
                    
                    if self.child[self.size**2]==None:
                        self.expand(self.size**2,now_node)
                        return True
                    elif self.child[self.size**2]!=None:
                        #這邊就不要再進去selection了，因為下面一個node不會有child。但是我們也要增加這種情況的計數，所以在這邊直接back up即可
                        self.child[self.size**2].back_up(now_node)
            #print('self.act_Q',self.act_Q)
            #print('self.p',self.p)
            self.S_select=self.act_Q+self.lambda_*(self.p/(1+self.N))
            #print('self.act_Q',self.act_Q)
            #self.S_select=self.lambda_*(self.p/(1+self.N))
            #print(self.S_select)
            #print(np.argmax(self.S_select))
            soted_index=np.argsort(self.S_select)
            soted_index=np.flip(soted_index)
            #self.env.render('terminal')
            #print('soted_index',soted_index)
            #print('self.invalid',self.invalid)
            #print('S_select',self.S_select)
            #print('sorted index',soted_index)
            for i in range(self.action_num):
                #print(soted_index[i])
                if soted_index[i]==self.size**2:
                    #print('sel 進入 pass')
                    if self.child[soted_index[i]]==None:
                        self.expand(soted_index[i],now_node)
                        return True
                    elif self.child[soted_index[i]]!=None:
                        Done=self.child[soted_index[i]].selection(now_node)
                        return True

                
                elif self.invalid[soted_index[i]]==True:
                    #Then jump to next i
                    #print(soted_index[i],'inside invalid',self.invalid )
                    pass
                elif self.child[soted_index[i]]==None:
                    #Then Do expand
                    #print(self.invalid)
                    #print('soted_index[i]',soted_index[i])
                    self.expand(soted_index[i],now_node)
               
                    return True
                elif self.child[soted_index[i]]!=None:
                    #print(self.invalid)
                    #print('soted_index[i]',soted_index[i])
                    self.child[soted_index[i]].selection(now_node)
                    return True
                
    def expand(self, added_position,now_node):
        #print('進入expand')
        #first put 
        self.env.state_=self.State
        state, reward, done, info = self.env.step(added_position)
        self.child[added_position]=Tree(self, added_position ,self.playing_env,self.env ,self.seq_reocord, self.size,Done=done, F=self.F, layer=self.layer+1)
        self.child[added_position].parent=self
        self.child[added_position].back_up(now_node)
        self.child[added_position].reward=reward
        # 如果done=1的時候，env就會掛掉 就不能再重新帶入state_了。所以我們要把env.done改成0
        if done==1:
            self.env.done=0
    def play(self):
        try:
            detect=(np.sum(self.State[2])-1)*self.reward<0
        except:
            detect=False

        if np.sum(self.invalid)==self.size**2:
            #print('換誰',self.State[2])
            #print('因為前面')
            next_action=self.action_num-1
            self.pi=np.array([0 for i in range(self.size**2+1)])
            self.pi[-1]=1
        elif detect and np.sum(self.State[4])>0: #代表 上一個人已經pass了
            #print('因為這邊')
            
            next_action=self.action_num-1
            self.pi=np.array([0 for i in range(self.size**2+1)])
            self.pi[-1]=1
            
        else:
            self.pi=self.N**self.gamma/np.sum(self.N**self.gamma)
            
            next_action=prob_select(self.pi)
            #print(self.pi)
            #print(next_action)
            #next_action=np.argmax(self.pi)
        #print(self.pi)
        #有了pi之後 我們就可以把pi record下來。每個state 會有一個對應到的pi 跟z 
        self.seq_reocord.append([self.State, self.pi])

        #接下來 就是用把self.playing_env.step(next_action)
        #並偵測是否已經結束。 已經結束的話 就可以開始製作sequence
        #若還沒結束的話，將now_root移到選擇的那個action
        #print('play',next_action)
        state, reward, done, info=self.playing_env.step(next_action)
        #print('pi',self.pi)
        #self.playing_env.render('terminal')
        #print('經過這邊')
        if done==1:
            #print('進來邊 前')
            #這邊開始將結果"reward"=[-1 or 1] 1 是黑棋贏的時候塞回去sequence中。
            #(S0, pi0) 為function第一個衡量的情況，也就是黑棋贏的機率
            for i in range(len(self.seq_reocord)):
                self.seq_reocord[i].append(reward)
                reward=-reward
            return None
        else:
            #print('進來這邊 換')
            return self.child[next_action]
        
    def clear_None(self):
        
        self.child_none_out=self.child[self.child != np.array(None)]
        for i in range(self.child_none_out.shape[0]):
            self.child_none_out[i].clear_None()
            
#Here create a function output just like Neural network
def f(State_):
    #pp=[1/(size**2+1)]*(size**2+1)
    p=np.random.multinomial(1090, [1/(size**2+1)]*(size**2+1))
    #rint('p=',p)
    p[-1]=0
    p=p/np.sum(p)
    #print('p',p)
    v=(np.random.rand()-0.5)*2
    return p,v

class MCTS():
    def __init__(self):
        #為了在過程中 偵測誰的地盤比較大，所以在go_env(衡量每個node的狀態) 我們把reward設成Heuristic，以便偵測當一個人pass的時候，另一個人如果已經贏了(地盤比較大)那就要pass
        
        go_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='heuristic')
        playing_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')
        seq_reocord=[]
        root=Tree(parent=None ,added_position=None ,playing_env=playing_env,env=go_env , size=size , F=f,seq_reocord=seq_reocord)
        now_node=root
        #root.clear_None()
        #print_tree(root, childattr='child_none_out', nameattr='visual', horizontal=True)
        while now_node!=None:
            for i in range(5):
                now_node.selection(now_node)
                #root.clear_None()
                #print_tree(root, childattr='child_none_out', nameattr='visual', horizontal=True)
            now_node=now_node.play()
            #input('')

        #root.clear_None()
        #print_tree(root, childattr='child_none_out', nameattr='visual', horizontal=True)
        """
        print(root.child)
        root=root.child_none_out[0]
        print(root.child)
        """
        #for i in range(5):
            #print(root.seq_reocord[i])
        print('結束畫面')
        root.playing_env.render('terminal')

if __name__ == '__main__':
    #First create a neural network which will output prabability distribution and action|state value
    start=time.time()
    mcts=MCTS()
    print('time',time.time()-start)
    

