import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from preprocessing import *

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_hidden_layer=1):
        super(Net, self).__init__()

        # hidden layers
        self.hidden = nn.ModuleList([torch.nn.Linear(n_feature, n_hidden)])  

        if n_hidden_layer >1:
            for i in range(n_hidden_layer-1):
                self.hidden.append(torch.nn.Linear(n_hidden, n_hidden))

        self.output = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        for h in self.hidden:
            x = F.relu(h(x))  # activation function for hidden layers
        x = self.output(x)  # linear output
        return x


class DotaNet():
    def __init__(self, n_feature=232, n_hidden=256, n_output=209, n_hidden_layer=1, train_steps=200):
        self.net = Net(n_feature, n_hidden, n_output, n_hidden_layer)  # define the network

        self.train_steps=train_steps

        self.cuda = torch.cuda.is_available()
        if self.cuda:  # CUDA
            self.net.cuda()
            print('Using CUDA')

        print(self.net)  # net architecture

        learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()

    def fit(self, X, y, steps=None, plot_loss=True):
        if steps is None:
            steps = self.train_steps

        X_train, X_val, y_train, y_val = train_test_split(X, y) 
        
        X_tens = self.to_tensor(X_train)
        y_tens = self.to_tensor(y_train)
        X_tens_val = self.to_tensor(X_val)
        y_tens_val = self.to_tensor(y_val)
        print("Training for %d steps" % steps)

        print_step = min(1000, steps/20)

        # train the network
        train_losses, val_losses = [], []
        t0 = time.time()
        for i in range(steps):
            y_pred = self.net(X_tens)  # input x and predict based on x
            loss = self.loss_func(y_pred, y_tens)
            train_losses.append(loss.item())

            y_val_pred = self.net(X_tens_val)
            val_loss = self.loss_func(y_val_pred, y_tens_val)
            val_losses.append(val_loss.item())

            if i%(print_step)==0:
                dt = time.time()-t0
                print("%ds - Step %d   \tTrain Loss %.4f  \tVal Loss %.4f" % (dt, i, loss.item(), val_loss.item()))
                print(val_losses[-6:-1])

            if len(val_losses)>1 and val_loss > max(val_losses[-6:-1]):
                print("Overfitting:\n", val_losses[-6:-1])
                break


            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients


        # Last loss evaluation
        y_pred = self.net(X_tens)  # input x and predict based on x
        loss = self.loss_func(y_pred, y_tens)
        train_losses.append(loss.item())

        y_val_pred = self.net(X_tens_val)
        val_loss = self.loss_func(y_val_pred, y_tens_val)
        val_losses.append(val_loss.item())
        print("%ds - Step %d   \tTrain Loss %.4f  \tVal Loss %.4f" % (dt, i, loss.item(), val_loss.item()))

        if plot_loss:
            plt.figure(figsize=(9, 9))

            plt.plot(train_losses, label="Train loss")
            plt.plot(val_losses, label="Validation loss")

            plt.ylim(0, max(max(train_losses), max(val_losses))*1.1)
            plt.xlabel("Steps")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.show()
        return self

    def predict(self, X):
        X_tens = self.to_tensor(X)
        with torch.no_grad():
            return self.net(X_tens).cpu()

    def to_tensor(self, data):
        if type(data) == torch.Tensor:
            v =  data
        elif type(data) == pd.DataFrame:
            v = Variable(torch.tensor(data.values)).float()
        else:
            v = Variable(torch.tensor(data)).float()
        if self.cuda:
            return v.cuda()
        else:
            return v
