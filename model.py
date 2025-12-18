import torch
import numpy as np
from helpers_HW.HW3.ion_trap import IonTrapEnv

class model():
    def __init__(self):
        srv = [3,3,3]
        KWARGS = {'phases': {'pulse_angles': [np.pi/2], 'pulse_phases': [np.pi/2], 'ms_phases': [-np.pi/2]}, 
                'num_ions': 3, 
                'goal': [srv],
                'max_steps': 10
                } 
        self.env = IonTrapEnv(**KWARGS)
        self.agent = PS_agent()
    
    def pred(self, samples): 
        preds = []
        for s in samples:
            #Need to set initial state of env somehow
            state = s
            done = False
            actions = []

            while not done:
                action = self.agent.act(state, greedy=True) 
                state, reward, done = self.env.step(action)
                actions.append(action)

            preds.append(torch.tensor(actions, dtype=torch.long))               
            
        return preds
    
class PS_agent:
    gamma : float
    eta : float
    def __init__(self, gamma, eta, num_states, num_actions):
        
        self.gamma = gamma
        self.eta = eta

        self.hmatrix = np.ones((num_states, num_actions))

        self.reset_gmatrix()

    def reset_gmatrix(self):
        self.gmatrix = np.zeros_like(self.hmatrix)
        
    def update_gmatrix(self):
        self.gmatrix *= (1 - self.eta)
        return self.gmatrix
    
    def learn(self, reward):    
        self.hmatrix = self.hmatrix - self.gamma * (self.hmatrix - 1) + self.gmatrix * reward

    def deliberate(self, state):
        # Compute probabilities for current action and sample        
        # probs_a = softmax(self.hmatrix[state, :])
        probs_a = self.hmatrix[state, :] / np.sum(self.hmatrix[state, :])
        action = np.random.choice(np.arange(len(probs_a)), p=probs_a)
        
        # Update glow matrix
        self.update_gmatrix()
        self.gmatrix[state, action] = 1

        return action
    
    def act(self, state, greedy=False):
        probs = self.hmatrix[state] / np.sum(self.hmatrix[state])

        if greedy:
            return np.argmax(probs)
        else:
            return np.random.choice(len(probs), p=probs)
        
'''Mail:
class model:
    ....
    def pred(self, samples):               

        for srv in samples:
               
            KWARGS = {..., 'goal': [srv]}
            env = IonTrapEnv(**KWARGS)
                       
            You favourite loop:

                a = self.your_agent_act(state, srv)
                               
                state, r, done = env.step(a)
                               
                # Save the action sequence for current srv
                sequence_for_this_srv.append(a)

            # collect all sequences to return it                               
            sequences.append(sequence_for_this_srv)
'''