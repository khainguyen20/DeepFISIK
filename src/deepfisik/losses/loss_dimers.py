# Copyright (C) 2026, Jaqaman Lab - UTSouthwestern
#
# This file is part of DeepFISIK.
#
# DeepFISIK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepFISIK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepFISIK.  If not, see <http://www.gnu.org/licenses/>.

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
