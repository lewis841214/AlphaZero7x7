import random

class node:
    def __init__(self, parent):
        self.parent = parent
        self.child = []
        self.childactionvalue = []
        self.state = 0
        self.value = 0
        self.count = 0
        self.win = 0


    def bestchild(self):
        n = -1
        v = -1
        if len(self.child) <= 0:
            return -1
        for i in range(len(self.child)):
            if self.childactionvalue[i] > v:
                n = i
                v = self.childactionvalue[i]

        return self.child[n]
    
    def pself(self):
        print(f"State: {self.state}")
        print(f"Parent Node: {self.parent}")
        print(f"child lengh: {len(self.child)}")
        print(f"Value: {self.value}")
        return 0

    def addchild(self,newchild,state):
        self.child.append(newchild)
        newchild.state = state
        self.childactionvalue.append(random.randrange(1,100))