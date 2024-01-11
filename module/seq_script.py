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

def train(model, device, train_loader, optimizer, criterion):
    model.train()  
    total_train_loss = 0.0

    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.float().to(device)
        X_train_batch = X_train_batch.unsqueeze(1).float() 
        optimizer.zero_grad()
        train_outputs = model(X_train_batch).squeeze()         
        
        train_loss = criterion(train_outputs, y_train_batch)
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss
    
def validate(model, device, val_loader, criterion):
    model.eval() 
    total_valid_loss = 0.0
    with torch.no_grad():
        for X_valid_batch, y_valid_batch in val_loader:
            X_valid_batch, y_valid_batch = X_valid_batch.to(device), y_valid_batch.to(device)
            X_valid_batch = X_valid_batch.unsqueeze(1).float()  
            valid_outputs = model(X_valid_batch).squeeze()         
            valid_loss = criterion(valid_outputs, y_valid_batch)
            total_valid_loss += valid_loss.item()

    avg_valid_loss = total_valid_loss / len(val_loader)
    return avg_valid_loss
    
    
def validate_result(model, device, val_loader, criterion):
    pred = torch.FloatTensor().to(device)
    y_ref = torch.FloatTensor().to(device)
    model.eval() 
    
    with torch.no_grad():
        for X_valid_batch, y_valid_batch in val_loader:
            X_valid_batch, y_valid_batch = X_valid_batch.to(device), y_valid_batch.to(device)
            X_valid_batch = X_valid_batch.unsqueeze(1).float()  
            valid_outputs = model(X_valid_batch).squeeze()         
            val_outputs = model(X_valid_batch)
            pred = torch.cat((pred, val_outputs), dim = 0)
            y_ref = torch.cat((y_ref, y_valid_batch), dim = 0)

    y_ref = y_ref.cpu().numpy()
    pred = pred.cpu().numpy()
    
    return y_ref, pred


