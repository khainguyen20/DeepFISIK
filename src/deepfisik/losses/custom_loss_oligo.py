import torch
import numpy as np
import sys

class BCE(torch.nn.Module):
    def __init__(self):
        super(BCE,self).__init__()

        self.bce = torch.nn.BCELoss()

    def forward(self, true, pred):

        loss = self.bce(pred,true)

        return  loss
