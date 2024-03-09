#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:47:13 2024

@author: robertc

Helper functions and classes for the non-linear regression.
"""
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def training_step(model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  dataloader: torch.utils.data.DataLoader,
                  device):
    """
    Performs the training step.
    Returns training loss.
    """
    # Setup the training loss per batch
    train_loss = 0.
    
    # Put model in training mode
    model.train()
    
    # Loop through training batches
    for batch, (X,y) in enumerate(dataloader):
        # Setup the prediction tensor
        y_preds = torch.tensor([])
    
        # Put data on target device
        X, y = X.to(device), y.to(device)
        
        # Loop over features
        for idx in range(len(X)):
            # Forward pass
            y_preds = torch.cat((y_preds, model(X[[idx]])))
        
        # Calculate the loss
        loss = loss_fn(y_preds, y)
        train_loss += loss
        
        # Optimizer - zero gradients
        optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()
        
        # Optimizer step
        optimizer.step()

    # Return the loss per batch
    return train_loss / len(dataloader)

def testing_step(model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 device):
    """
    Performs a testing step.
    Returns test loss.
    """
    # Setup the test loss per batch
    test_loss = 0.
    
    # Put the model in evaluation mode
    model.eval()
    
    with torch.inference_mode():
        # Loop over testing batches
        for batch, (X,y) in enumerate(dataloader):
            # Setup the prediction tensor
            y_preds = torch.tensor([])
            
            # Put data on target device
            X, y = X.to(device), y.to(device)
            
            # Loop over features
            for idx in range(len(X)):
                # Forward pass
                y_preds = torch.cat((y_preds, model(X[[idx]])))
              
            # Calculate the loss
            loss = loss_fn(y_preds, y)
            test_loss += loss

    # Return the loss per batch
    return test_loss / len(dataloader)
              
def train_model(num_epochs: int,
                model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                device,
                make_plots: bool = True):
    """
    Trains the model for num_epochs epochs.
    Plots the loss curves. 
    Returns the lists with loss functions after each epoch.
    """
    # Setup the lists for loss curves
    train_loss_list, test_loss_list = [], []
    
    #Loop over epochs
    for epoch in tqdm(range(num_epochs)):
        # Training step
        train_loss = training_step(model = model,
                                   loss_fn = loss_fn,
                                   optimizer = optimizer,
                                   dataloader = train_dataloader,
                                   device = device)
        train_loss_list.append(train_loss.item())
        
        # Testing step
        test_loss = testing_step(model = model,
                                 loss_fn = loss_fn,
                                 dataloader = test_dataloader, 
                                 device = device)
        test_loss_list.append(test_loss.item())
    
    # Plot the loss curves
    if make_plots == True:
        plt.plot(train_loss_list, label = "Train loss")
        plt.plot(test_loss_list, label = "Test loss")
        plt.grid()
        plt.legend()
        plt.title("Loss curves")
        
    return train_loss_list, test_loss_list
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    