import torch
import torch.nn as nn
import numpy as np


def create_model(
        input_size=4,
        hidden1_size=128,
        hidden2_size=128,
        hidden3_size=50,
        num_classes=4,
):
    return SimpleNet(input_size,hidden1_size,hidden2_size,hidden3_size,num_classes)


class SimpleNet(nn.Module):

    def __init__(self, input_size, hidden1_size,hidden2_size,hidden3_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden1_size)
        #self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size,hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size,hidden3_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden3_size,num_classes)



    def forward(self, x):
        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out
