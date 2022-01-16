import numpy as np
import random

class Field:
    def __init__(self,stat,st=[]):
        self.states = stat
        self.state = st
        

    def done(self):
        if self.states[self.state[0]][self.state[1]] == 1:
            return True
        return False

    def get_possible_actions(self):
        actions = [0,1,2,3]
        if self.state[1] == 0:
            actions.remove(0)
        elif self.state[1] == len(self.states[0]) - 1:
            actions.remove(1)

        if self.state[0] == 0:
            actions.remove(2)
        elif self.state[0] == 2:
            actions.remove(3)
        return actions
    def update_next_state(self,action):
        x,y = self.state
        if action == 0:
            if self.state[1] == 0:
                return self.state,-10
            self.state= x,y-1
        if action == 1:
            if self.state[1] == len(self.states[0]) - 1:
                return self.state,-10
            self.state =x, y+ 1
        if action == 2:
            if self.state[0] == 0:
                return self.state,-10
            self.state =x- 1,y
        if action == 3:
            if self.state[0] == 2:
                return self.state,-10
            self.state =x+ 1,y
        if self.states[self.state[0]][self.state[1]] == -1:
            return self.state,-10
        reward = self.states[self.state[0]][self.state[1]]
        return self.state, reward  

    def train(this):
        q_table = np.zeros((len(this.states),len(this.states[0]), 4))
        alpha= .5
        gamma = 0.5
        epsilon = 0.5
        for _ in range(10000):
            field = Field(this.states,(random.randrange(0,3),random.randrange(0, len(this.states[0]))))
            while not field.done():
                actions = field.get_possible_actions()
                if random.uniform(0,1)<epsilon:
                    action = random.choice(actions)
                else:
                    action = np.argmax(q_table[field.state])
                cur_x = field.state[0]
                cur_y = field.state[1]
                nxt_state,reward = field.update_next_state(action)
                q_table[cur_x, cur_y, action] = (1 - alpha)*q_table[cur_x, cur_y, action] + alpha*(reward + gamma*np.max(q_table[nxt_state[0], nxt_state[1]]))
        
        return q_table
            
    
states =  [[-1,0,0,0,-1,0,0,-1,0,-1,1],[-1,0,-1,0,0,-1,0,0,0,0,0],[0,0,0,-1,0,0,0,-1,0,-1,-1]]
path = np.zeros((3,11))
field = Field(states,(2,0))
steps = 1
path[field.state[0]][field.state[1]] = np.nan
q_table = field.train()
field.state
while not field.done():
    action = np.argmax(q_table[field.state])
    next_state,reward= field.update_next_state(action)
    path[next_state[0]][next_state[1]] = steps
    steps +=1
print(path)
