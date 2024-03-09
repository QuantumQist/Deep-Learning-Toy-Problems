#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:07:36 2024

@author: robertc
"""
import torch
from torch import nn

class NonLinearFit(nn.Module):
  """Defines a neural network model with non-linear fit"""
  def __init__(self, hidden_layers: int, hidden_layer_nodes: int):
    super().__init__()

    # Setup the number of hidden layers
    if hidden_layers > 0 and type(hidden_layers)==int:
      self.hidden_layers = hidden_layers
    else:
      raise Exception("Number of hidden layers must be an integer greater than 0.")
    # Setup the number of nodes in the hidden layer
    if hidden_layer_nodes > 0 and type(hidden_layers)==int:
      self.hidden_layer_nodes = hidden_layer_nodes
    else:
      raise Exception("Number of hidden layer nodes must be an integer greater than 0.")

    # Setup the empty layer stack
    # We will construct a list of nn.Modules and then unpack it
    self.layers = []

    # Initial NN layer
    self.layers.append( nn.Sequential(
        nn.Linear(in_features = 1, out_features = self.hidden_layer_nodes),
        nn.ReLU()
    ))

    # Middle NN layers
    for __ in range(self.hidden_layers - 1):
      self.layers.append( nn.Sequential(
          nn.Linear(in_features = self.hidden_layer_nodes, out_features = self.hidden_layer_nodes),
          nn.ReLU()
      ))

    # Final NN layer
    self.layers.append( nn.Sequential(
        nn.Linear(in_features = self.hidden_layer_nodes, out_features = 1)
        #nn.Tanh()
    ))

    # Convert the list of layers into a sequential layer stack
    self.layers = nn.Sequential(*self.layers)

  # Overwrite forward pass
  def forward(self, x):
    return self.layers(x)