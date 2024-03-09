#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: robertc

Defines the functions preparing the data to enter a PyTorch model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List

class PolynomialDataset(Dataset):
    """
    Defines a Dataset class to handle the non-linear regression data.
    """
    def __init__(self, features, labels):
        self.labels = torch.Tensor(labels)
        self.features = torch.Tensor(features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def generate_dataloaders(X: List[float],
                         y: List[float],
                         batch_size: int = 128,
                         test_size: float = 0.2):
    """
    Generates train_dataloader and test_dataloader from the features X and 
    labels y.

    Parameters
    ----------
    X : List[float]
        Features
    y : List[float]
        Labels
    batch_size : int, optional
        Number of samples in the batch. The default is 128.
    test_size : float, optional
        Fraction of data to enter the test DataLoader. Must be betwee 0 and 1
        The default is 0.2.

    Returns a tuple 
    (train_dataloader, test_dataloader, test_features, test_labels)
    """
    # Check of `test_size` is in correct range of values
    if test_size < 0 or test_size > 1:
        raise ValueError("`test_size` must be a float between 0 and 1")
        
    # Convert data into torch.Tensors
    X, y = torch.tensor(X, dtype = torch.float), torch.tensor(y, dtype = torch.float)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size)
    
    # Create datasets
    train_dataset = PolynomialDataset(X_train, y_train)
    test_dataset = PolynomialDataset(X_test, y_test)
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader, X_test, y_test
    