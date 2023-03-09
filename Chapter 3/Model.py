import numpy as np
from torch import nn, flatten
import torch.nn.functional as F
import torch
from settings import input_shape

class Network(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(1000, 15)


    def feature_extraction(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = flatten(x, 1)
        return x

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.fc1(x)
        return x