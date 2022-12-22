#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class Classifier(nn.Module):
    def __init__(self, node, device):
        super(Classifier, self).__init__()
        
        self.device = device
        
        self.classify = nn.Sequential(
            nn.Linear(node, 2),
            nn.ELU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        ).to(device)
        
    def forward(self, input):
        out = self.classify(input)
        return out
#%%
def main():
    """Baseline Classifier"""
    model = Classifier(4, 'cpu')
    for x in model.parameters():
        print(x.shape)
    batch = torch.rand(10, 4)
    
    pred = model(batch)
    
    assert pred.shape == (10, 1)
    
    print("Downstream: Baseline Classifier pass test!")
#%%
if __name__ == '__main__':
    main()
#%%