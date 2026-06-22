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

class BCE(torch.nn.Module):
    def __init__(self):
        super(BCE,self).__init__()

        self.bce = torch.nn.BCELoss()

    def forward(self, true, pred):

        loss = self.bce(pred,true)

        return  loss
