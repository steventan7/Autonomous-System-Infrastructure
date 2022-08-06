import os
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#dropout rate for 2d dropout in convolutional layers and dropout rate for fully connected layers
conv_dropout_rate, fc_dropout_rate = 0.125, 0.125

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding = 1)
        self.dropout1 = nn.Dropout2d(conv_dropout_rate)
        self.conv2 = nn.Conv2d(32, 64, 5, padding = 1)
        self.dropout2 = nn.Dropout2d(conv_dropout_rate)
        self.conv3 = nn.Conv2d(64, 64, 5, padding = 1)
        self.dropout3 = nn.Dropout2d(conv_dropout_rate)

        # layer weight, depending on the input image size
        self.fc1 = nn.Linear(7*12*36, 100)
        self.out = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.conv1(x.float())))
        x = self.dropout2(F.relu(self.conv2(x.float())))
        x = self.dropout3(F.relu(self.conv3(x.float())))
        x = F.relu(self.conv3(x))  # 160x120x32
        x = nn.Flatten(x, 1)

        x = F.dropout(F.relu(self.fc1(x)),
            p = fc_dropout_rate, training = self.training)
        x = self.out(x)
        return x

net = NeuralNetwork()
print(net)
