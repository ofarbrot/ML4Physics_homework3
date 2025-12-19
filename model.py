import torch
import numpy as np
from helpers_HW.HW3.ion_trap import IonTrapEnv
#Import agent

class model():
    def __init__(self):
        None
 
    def pred(self, samples):            
        sequences = []

        for srv in samples:
            sequence = []
               
            KWARGS = {'phases': {'pulse_angles': [np.pi/2], 'pulse_phases': [np.pi/2], 'ms_phases': [-np.pi/2]}, # Gates available
                'num_ions': 3, 
                'goal': [srv],
                'max_steps': 10
                } 

            env = IonTrapEnv(**KWARGS)
            
            agent = None
                       
            while not done:
                a = agent.act(state, greedy=True) 
                state, reward, done = env.step(a)
                sequence.append(a)

            sequences.append(torch.tensor(sequence, dtype=torch.long)) 

        return sequences
