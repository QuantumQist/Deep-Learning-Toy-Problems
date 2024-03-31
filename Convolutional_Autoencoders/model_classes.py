# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:24:09 2024

@author: rczup
"""
from torch import nn

class CNN_Autoencoder(nn.Module):
    """
    CNN autoencoder with 64 perceptrons in latent representation.
    Network from https://youtu.be/zp8clK9yCro?si=gLD7SKyjN6pMwdT7
    with added classifier head.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=2, padding =1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride=2, padding =1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 7)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 7),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3,
                              stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3,
                              stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )
        
        # Classifier 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 64, out_features = 10)
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
        
class CNN_Tiny_Autoencoder(nn.Module):
    """
    CNN autoencoder with 8 perceptrons in latent representation
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride=2, padding =1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 8, out_channels = 8, kernel_size = 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 8, kernel_size = 3, 
                               stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 8, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )
        
        # Classifier 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 8, out_features = 10)
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

class CNN_Medium_Autoencoder(nn.Module):
    """
    CNN autoencoder with 16 perceptrons in latent representation
    """
    def __init__(self):
        super().__init__()
        
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
            nn.Linear(in_features = 16, out_features = 10)
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