import numpy as np
class Tree:
    def __init__(self,parent ,added_position ,board ,invalid, size , F, Done=False, lambda_=0.5, gamma=1):
        self.size=size
        self.gamma=gamma
        self.data = None
        self.p=np.array(F(board)[0]) #This function will load P from Network
        self.v=F(board)[0]
        self.child=np.array([None for i in range(size**2)])
        self.parent=None
        self.W=0
        #Q is a action value function
        self.act_Q=np.array([0 for i in range(size**2)])
        self.N=np.array([0 for i in range(size**2)])
        self.State=board
        self.add=added_position
        self.Done=Done
        self.z
        # This show the invalid index, then when selection wants to go to here, it skip the invalid step
        self.invalid=invalid
        
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
        if seld.Done==False:
            self.S_select=self.Q+lambda_*(self.p/(1+self.N))
            soted_index=np.argsort(self.S_select)
            soted_index=np.flip(soted_index)
            for i in range(self.size**2):
                if self.invalid[i]==True:
                    #Then jump to next i
                    pass
                elif self.child[soted_index[i]]==None:
                    #Then Do expand
                    self.expand(soted_index[i])
                elif self.child[soted_index[i]]!=None:
                    self.child[soted_index[i]].selection()
                
    def expand(self, added_position):
        #first put 
        self.child[added_position]=Tree(self, added_position ,board ,invalid, size)

    def play(self):
        self.pi=self.N**self.gamma/np.sum(self.N**self.gamma)
        next_action=np.max(self.pi)
class MCTS():
    def __init__(self):
        go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')

if __name__ == '__main__':
    #First create a neural network which will output prabability distribution and action|state value
    size=7
    def f(State_):
        p=np.random.multinomial(107, [1/size**2]*size**2)
        #print(p)
        p=p/np.sum(p)
        v=(np.random.rand()-0.5)*2
        return p,v
    s=2
    print(f(2))
    

