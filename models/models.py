import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

def initialize_weights(m):
  if isinstance(m, nn.Linear): #or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
    init.xavier_uniform(m.weight.data)

class PolicyNetwork(nn.Module):
  def __init__(self):
    super(PolicyNetwork, self).__init__()
    self.affine1 = nn.Linear(17, 128)
    self.affine2 = nn.Linear(128, 6)

    self.saved_actions = []
    self.rewards = []

  def forward(self, x):
    x = F.relu(self.affine1(x))
    action_scores = self.affine2(x)
    return action_scores