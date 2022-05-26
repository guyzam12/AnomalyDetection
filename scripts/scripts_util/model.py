import torch
import torch.nn as nn
import numpy as np


def create_model(
        hidden1_size=100,
        input_size=4,
        hidden2_size=50,
        num_classes=3,
):
    return SimpleNet(input_size,hidden1_size,hidden2_size,num_classes)


class SimpleNet(nn.Module):

    def __init__(self, input_size, hidden1_size,hidden2_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size,hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size,num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
