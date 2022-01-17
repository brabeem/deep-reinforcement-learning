import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self,state_size,action_size):
            super(Actor,self).__init__()
            self.fc1 = nn.Linear(state_size,64)
            self.fc2 = nn.Linear(64,64)
            self.fc3 = nn.Linear(64,action_size)
    def forward(self,state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))


class Critic(nn.Module):
    def __init__(self,state_size,action_size):
            super(Critic,self).__init__()
            self.fc1 = nn.Linear(state_size + action_size,64)
            self.fc2 = nn.Linear(64,64)
            self.fc3 = nn.Linear(64,1)
    
    def forward(self,state,action):
        x  = torch.cat((state,action),dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


