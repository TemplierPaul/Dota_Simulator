from DotaUNet.UNet_parts import *
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

class FFNet(nn.Module):
    def __init__(self, state_length, action_length, n_hidden=256, n_hidden_layer=1):
        super(FFNet, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        self.input_layer = Input(state_length, action_length, n_hidden)

        # hidden layers
        self.hidden = nn.ModuleList()

        for i in range(n_hidden_layer):
            self.hidden.append(torch.nn.Linear(n_hidden, n_hidden))

        self.output_layer = Output(n_hidden, state_length)

        if self.use_cuda:
            print("Using CUDA")
            self.cuda()

    def forward(self, state, action):
        if self.use_cuda:
            state = state.cuda()
            action = action.cuda()

        x = self.input_layer(state, action)
        for h in self.hidden:
            x = F.relu(h(x))  # activation function for hidden layers
        x = self.output_layer(x)  # linear output
        return x.cpu()

class DotaUNet(nn.Module):
    def __init__(self, state_length=208, action_length=29):
        super(DotaUNet, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        self.input_layer = Input(state_length, action_length, 256)
        self.down1 = Down(256, 128)
        self.down2 = Down(128, 64)

        self.bottom = Down(64, 64)

        self.up1 = Up(64, 128)
        self.concat1 = ConcatConv()
        self.up2 = Up(128, 256)
        self.concat2 = ConcatConv()

        self.output_layer = Output(256, state_length)

        if self.use_cuda:
            print("Using CUDA")
            self.cuda()

    def forward(self, state, action):
        if self.use_cuda:
            state = state.cuda()
            action = action.cuda()
        x1 = self.input_layer(state, action)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.bottom(x3)
        x = self.up1(x)
        x = self.concat1(x2, x)
        x = self.up2(x)
        x = self.concat2(x1, x)
        x = self.output_layer(x)
        return x.cpu()