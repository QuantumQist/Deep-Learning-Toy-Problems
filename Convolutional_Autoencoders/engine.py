# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:32:26 2024

@author: rczup
"""

"""
Contains functions for training and testing a PyTorch model.

The train_step(), test_step() and train() are originally from 
https://youtu.be/V_xro1bcAuA?si=e5_5khnsgzyBFYSR
with modifications allowing to train different heads.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from pathlib import Path

import utils

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               model_mode: str = "autoencoder") -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    model_mode: Defines the mode of operation of the model. 

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X, model_mode)

        # 2. Calculate  and accumulate loss
        if model_mode == "autoencoder":
            loss = loss_fn(y_pred, X)
        else:
            loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              model_mode: str = "autoencoder") -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    model_mode: Defines the mode of operation of the model

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X, model_mode)

            # 2. Calculate and accumulate loss
            if model_mode == "autoencoder":
                loss = loss_fn(test_pred_logits, X)
            else:
                loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          model_mode: str = "autoencoder") -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    model_mode: Defines the mode of operation of the model

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              test_loss: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              test_loss: [1.2641, 1.5706]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "test_loss": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            model_mode=model_mode)
        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            model_mode=model_mode)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # Return the filled results at the end of the epochs
    return results

def train_autoencoder(model: torch.nn.Module, 
                      train_dataloader: torch.utils.data.DataLoader, 
                      test_dataloader: torch.utils.data.DataLoader, 
                      epochs: int,
                      device: torch.device,
                      save_path: str,
                      save_file_name: str):
    """
    Trains model`s encoder and decoder parts. Saves trained model's state
    dictionary to save_path path. 

    Parameters
    ----------
    model : torch.nn.Module
        Module containing the autoencoder elements
    train_dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be trained on.
    test_dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be tested on.
    epochs : int
        An integer indicating how many epochs to train for.
    device : torch.device
        A target device to compute on (e.g. "cuda" or "cpu").
    save_path : str
        Path where the trained model parameters will be saved.
    save_file_name: str
        Name of the file where the model will be saved.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              test_loss: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              test_loss: [1.2641, 1.5706]} 
    """
    # Make sure only the `encoder` and `decoder` parameters are trainable
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    for param in model.decoder.parameters():
        param.requires_grad = True
        
    # Setup loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)
    
    # Train model
    model_results = train(
        model = model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        loss_fn = loss_fn,
        epochs = epochs,
        device = device,
        model_mode = "autoencoder"
    )
    
    # Save model
    utils.save_model(model, save_path, save_file_name)
    
    return model_results

def train_classifier(model: torch.nn.Module, 
                      train_dataloader: torch.utils.data.DataLoader, 
                      test_dataloader: torch.utils.data.DataLoader, 
                      epochs: int,
                      device: torch.device,
                      save_path: str,
                      save_file_name: str):
    """
    Trains model`s classifier part. Saves trained model's state
    dictionary to save_path path. 

    Parameters
    ----------
    model : torch.nn.Module
        Module containing the autoencoder elements
    train_dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be trained on.
    test_dataloader : torch.utils.data.DataLoader
        A DataLoader instance for the model to be tested on.
    epochs : int
        An integer indicating how many epochs to train for.
    device : torch.device
        A target device to compute on (e.g. "cuda" or "cpu").
    save_path : str
        Path where the trained model parameters will be saved.
    save_file_name: str
        Name of the file where the model will be saved.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              test_loss: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              test_loss: [1.2641, 1.5706]} 
    """
    # Make sure only the `encoder` and `decoder` parameters are trainable
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Setup loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)
    
    # Train model
    model_results = train(
        model = model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        loss_fn = loss_fn,
        epochs = epochs,
        device = device,
        model_mode = "classifier"
    )
    
    # Save model
    utils.save_model(model, save_path, save_file_name)
    
    return model_results