import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class net(nn.Module):
    
    
    def __init__(self,input_size,output_size):
        super(net,self).__init__()
        self.fc1 = nn.Linear(in_features=input_size,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=64)
        self.fc3 = nn.Linear(in_features=64,out_features=output_size)
    
    
    def forward(self, x):
        if isinstance(x,np.ndarray):
            x = torch.tensor(x,dtype=torch.float)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

