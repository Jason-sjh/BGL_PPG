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


class PPG_model(nn.Module):
    def __init__(self):
        super(PPG_model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 510, 64)  # 입력 데이터 크기에 따라 조정
        self.fc2 = nn.Linear(64, 1)  # 혈당 예측을 위한 출력 뉴런 1개

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x
