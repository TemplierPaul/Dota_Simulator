import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

# from Surrogate.DotaUNet.UNet_parts import *
from DotaUNet.UNet import *

import pytest

def get_random_tensors(records_nb=100, channels=1, record_len=208):
    vect = np.random.rand(records_nb, channels, record_len)
    return Variable(torch.tensor(vect)).float()

def test_random_data():
    t = get_random_tensors()
    assert type(t) == torch.Tensor
    assert t.shape == (100, 1, 208)

    t = get_random_tensors(channels=2)
    assert type(t) == torch.Tensor
    assert t.shape == (100, 2, 208)

def test_input():
    net = Input(208, 29, 256)
    state = get_random_tensors(record_len=208)
    action = get_random_tensors(record_len=29)
    y = net(state, action)
    assert y.shape == (100, 1, 256)

def test_down():
    down = Down(15, 10)
    x = get_random_tensors(record_len=15)
    y = down(x)
    assert y.shape == (100, 1, 10)
    train_for_test(down, 15, 10)

def test_concat_conv():
    concat = ConcatConv()
    x1, x2 = get_random_tensors(record_len=10), get_random_tensors(record_len=10)
    x_out = concat(x1, x2)
    assert x_out.shape == (100, 1, 10)

def test_up():
    up = Up(10, 15)
    x = get_random_tensors(record_len=10)
    y = up(x)
    assert y.shape == (100, 1, 15)
    train_for_test(up, 10, 15)

def test_output():
    out = Output(256, 208)
    x = get_random_tensors(record_len=256)
    y = out(x)
    assert y.shape == (100, 1, 208)

def test_Unet():
    net = DotaUNet(208, 29)
    state = get_random_tensors(record_len=208)
    action = get_random_tensors(record_len=29)
    y = net(state, action)
    assert y.shape == (100, 1, 208)

    y = get_random_tensors(record_len=208)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()

    for i in range(10):
        next_state = net(state, action)  # input x and predict based on x
        loss = loss_func(next_state, state)

        print(i, loss)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    return net


def train_for_test(net, in_length=208, out_legth=1):
    x = get_random_tensors(record_len=in_length)
    y = get_random_tensors(record_len=out_legth)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()

    for i in range(10):
        y_pred = net(x)  # input x and predict based on x
        loss = loss_func(y_pred, y)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    return net

