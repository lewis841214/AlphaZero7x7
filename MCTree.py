import gym
import numpy as np
from pptree import *
import random
import time

size = 4

class Tree:
    def __init__(self, parent, p, size, state, env, F, lambda_=1, gamma=1):

        self.parent = parent
        self.state = state
        #state of this node
        self.size = size
        self.env = env
        #go env
        self.f = F
        # NN simulator
        self.w = self.f.valuestate(self.state)
        self.p = p
        # init w with self.state 
        # w won't change
        self.n = 1
        # base
        self.q = self.w
        # value = win rate 
        # include child winrate
        self.childlist = [None]* (self.size ** 2 + 1)
        self.childp = self.f.childp(self.size ** 2 + 1)
        self.childlistP = []
        # size ^2 + pass


        #check if the game is over
        if self.state[5][0][0] == 1:
            self.over = True
        else:
            self.over = False

        self.lambda_ = lambda_
        
        self.gamma = gamma
        #exploration rate

        #if the game isn't over
        if self.over != True:
            #find next move
            #select return a state
            pass
        else:
            #print("Game is over")
            pass
            # game is over

        return 


    def select(self):       

        # if game is over, return w
        if self.over == True:
            self.n += 1
            return self.w

        self.env.state_ = self.state
        #load current node state
        validmoves = self.env.valid_moves()
        childstates = self.env.children()
        #get valid moves and child states

        best = 0
        bestvalue = -1
        #find best value node
        for i in range(len(self.childlist)):
            if self.childlist[i] != None:
                # if this childnode has been created
                # use winnrate to value the node
                # let p = 1
                if bestvalue < (self.childlist[i].winrate() + self.lambda_ * self.p / ( 1 + self.childlist[i].n)):
                    best = i
                    bestvalue = (self.childlist[i].winrate() + self.lambda_ * self.p / ( 1 + self.childlist[i].n))

            elif validmoves[i] == 1:
                # if this node hasn't been created
                # use state value to value the node
                if bestvalue < self.f.valuestate(childstates[i] + self.lambda_):
                    best = i
                    bestvalue = self.f.valuestate(childstates[i] + self.lambda_)
            else:
                # not a valid move
                #print("not a valid move")
                pass
        
        if bestvalue == -1:
            #there is no valid move
            print("Select: no valid moves")
            return -1

        # got a best value, use the value to continue searching the tree
        # return the value of the added new node
        # backup update w & n until root
        if self.childlist[best] == None:
            # creat a new node
            self.childlist[best] = Tree(self, self.childp[best], self.size, childstates[best], self.env, self.f, self.lambda_, self.gamma)
            self.childlistP.append(self.childlist[best])
            updatevalue = self.childlist[best].w
        else:
            # allready a node here, keep going
            updatevalue = self.childlist[best].select()


        self.q += updatevalue
        self.n += 1
        # update myself
        return -updatevalue
        # return update value to parent

    
    def findchild(self, state):
        #findout a state is in childlist or not
        for i in range(len(self.childlist)):
            if self.childlist[i] == state:
                return i
        
        return -1

    def play(self):
        sum = 0
        if len(self.childlist) == 1:
            return self.childlist[0].state

        for i in range(len(self.childlist)-1):
            if self.childlist[i] != None:
                sum += self.childlist[i].n ** (1 / self.gamma)
        # sum up each child's N
        print("Play:")
        print(sum)
        sum = random.uniform(0,sum)
        print(sum)

        for i in range(len(self.childlist)):
            if self.childlist[i] != None:
                sum = sum - self.childlist[i].n ** (1 / self.gamma)
            if sum < 0:
                return self.childlist[i].state

        print("Tree.Play: random choose child error")




    def winrate(self):
        return float(self.q / self.n)
# nn simulator
class F:
    def __init__(self):
        pass

    def valuestate(self, state):
        return hash(state.tobytes) % 10000

    def childp(self,size):
        #generate p for each child
        t = []
        for i in range(size):
            t.append(random.uniform(0,1))
        
        return t


class MCTS():
    def __init__(self):
        self.size = 4
        self.simu_round = 10
        go_env = gym.make('gym_go:go-v0', size=self.size, komi=0, reward_method='real')
        f = F()
        r = 1
        while r:
            root = Tree(None, 1, self.size, go_env.state_, go_env, f)
            for i in range(self.simu_round):
                root.select()

            for i in range(len(root.childlist)):
                if root.childlist[i] != None:
                    print(root.childlist[i].n, end = ' ')
            go_env.state_ = root.play()
            go_env.render('terminal')


            r = input()


    

if __name__ == '__main__':
    #First create a neural network which will output prabability distribution and action|state value
    mcts=MCTS()

    