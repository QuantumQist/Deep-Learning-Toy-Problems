# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:03:30 2024

@author: rczup
"""

from torch import nn

class CNN_Autoencoder(nn.Module):
    """
    CNN autoencoder with 16 perceptrons in latent representation
    """
    def __init__(self, hidden_layer_size, dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=2, padding =1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = 3, 
                               stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )
        
        # Classifier 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 16, out_features = self.hidden_layer_size),
            nn.Dropout(p = self.dropout),
            nn.ReLU(),
            nn.Linear(in_features = self.hidden_layer_size, out_features = 5)
        )
        
    def forward(self, x, mode = "autoencoder"):
        """
        Defines the forward pass.
        The `mode` parameter defines the operation of the model. It must
        have one of the following values
        > "autoencoder" - returns the output of encoder -> decoder
        > "encoder" - returns the output of the encoder
        > "classifier" - returns the output of encoder -> classifier 
        """
        if mode == "autoencoder":
            return self.decoder(self.encoder(x))
        elif mode == "encoder":
            return self.encoder(x)
        elif mode == "classifier":
            return self.classifier(self.encoder(x))