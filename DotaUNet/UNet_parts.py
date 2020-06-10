import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

import pytest

class Input(nn.Module):
    def __init__(self, state_length, action_length, out_length):
        super().__init__()
        self.layer = nn.Linear(state_length + action_length, out_length)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=2)
        x = self.layer(x)
        return x

class Down(nn.Module):
    def __init__(self, in_length, out_length):
        super().__init__()
        self.layer = nn.Linear(in_length, out_length)

    def forward(self, x):
        return self.layer(x)

class ConcatConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_length, out_length):
        super().__init__()
        self.layer = nn.Linear(in_length, out_length)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer(x)
        x = self.dropout(x)
        return x

class Output(nn.Module):
    def __init__(self, in_length, out_length):
        super().__init__()
        self.layer = nn.Linear(in_length, out_length)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer(x)
        x = self.dropout(x)
        return x