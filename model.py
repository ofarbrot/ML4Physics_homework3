import torch

class model():
    def __init__(self):
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def pred(self, samples):

        # This agent will just create random gate sequences        
        preds = []
        for s in samples:
            num_gates = torch.randint(3,10,(1,))
            gate_sequence = torch.randint(0, 7, (num_gates,))
            preds.append(gate_sequence)
            
        return preds