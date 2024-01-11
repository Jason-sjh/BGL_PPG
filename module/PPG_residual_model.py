# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import os
import pickle
import argparse

from scipy.stats import iqr, skew
from scipy.fftpack import fft

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class PPG_Res_Model(nn.Module):
    def __init__(self):
        super(PPG_Res_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.fc = nn.Linear(64, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = torch.mean(x, 2)  # Global Average Pooling
        x = self.fc(x)

        return x

