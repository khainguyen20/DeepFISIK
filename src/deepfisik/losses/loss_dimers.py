import torch
import numpy as np
import sys

class MultiMSE(torch.nn.Module):
    def __init__(self):
        super(MultiMSE,self).__init__()

        self.mse = torch.nn.MSELoss()

    def forward(self, true, pred):

        l1 = self.mse(true[0],pred[0])
        l2 = self.mse(true[1],pred[1])
        l3 = self.mse(true[2],pred[2])
        l4 = self.mse(true[3],pred[3])
        l5 = self.mse(true[4],pred[4])

        loss = l1 + l2 + l3 + l4 + l5

        return l1,l2,l3,l4,l5, loss
