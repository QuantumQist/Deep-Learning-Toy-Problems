# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:12:26 2024

@author: rczup
"""

import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path
import pickle
from typing import Dict, List
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def plot_reconstructions(model: torch.nn.Module,
                         dataset: torch.utils.data.Dataset):
    """
    Plots 3 random samples from `dataset` dataset and their 
    model reconstructions

    Parameters
    ----------
    model : torch.nn.Module
        Model used to generate the reconstructions
    dataset : torch.utils.data.Dataset
        Dataset with images to reconstruct

    Returns nothing
    """
    # Get random sample of 3 images 
    random_idxs = random.sample(range(len(dataset)), k=3)
    
    # Make predictions and plot them
    fig, axs = plt.subplots(3, 2, figsize = (6,8))
    model.eval()
    for i in range(3):
        image = dataset.__getitem__(random_idxs[i])[0]
        label = dataset.__getitem__(random_idxs[i])[1]
        # Forward pass
        with torch.inference_mode():
            reconstruction = model(image, "autoencoder")
            pred_label = model(image.unsqueeze(dim=0), "classifier").argmax().item()
        axs[i,0].imshow(image.squeeze(), cmap="gray")
        axs[i,1].imshow(reconstruction.squeeze().numpy(), cmap="gray")
        axs[i,0].set_axis_off()
        axs[i,1].set_axis_off()
        axs[i,0].set_title(f"True label: {label}")
        axs[i,1].set_title(f"Prediction: {pred_label}")
    plt.suptitle("Left: original | Right: reconstruction")
    
def save_model(model: torch.nn.Module,
               path: str,
               file_name: str):
    """
    Saves PyTorch model to a specified path

    Parameters
    ----------
    model : torch.nn.Module
        Model to be saved
    path : str
        Path where the model will be saved
    file_name : str
        Name of the file to save the model 

    Returns nothing.
    """
    assert file_name.endswith(".pth"), "file_name should end with '.pth'"
    
    # Create model directory path
    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)
    
    # Create model save path
    MODEL_NAME = file_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    # Save the model
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)
    
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
    
def save_dict(dictionary: dict, save_path: str, save_name: str):
    """
    Saves a dictionarty to a terget dirctory.
    Based on: https://pynative.com/python-save-dictionary-to-file/

    Parameters
    ----------
    dictionary : dict
        Dictionary to be saved.
    save_path : str
        Save path.
    save_name : str
        File name where the dictionary will be stored
    """
    assert save_name.endswith(".pkl"), "save_name should end with '.pt' or '.pth'"
    
    # Create model directory path
    MODEL_PATH = Path(save_path)
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)
    
    SAVE_PATH = Path(save_path)
    TOTAL_SAVE_PATH = SAVE_PATH / save_name    
    
    with open(TOTAL_SAVE_PATH, 'wb') as fp:
        pickle.dump(dictionary, fp)
        print('dictionary saved successfully to file')    
    
    
def load_dict(save_path: str) -> dict:
    """
    Loads a dictionary from save_path

    Parameters
    ----------
    save_path : str
        path to the dictionary to be loaded.


    Returns the loaded dictionary

    """
    with open(save_path, 'rb') as fp:
        return pickle.load(fp)
    
def plot_loss_curves(model_results: List[Dict],
                     model_names: List[str]):
    """
    Plots train and test loss curves for the data contained in model_results

    Parameters
    ----------
    model_results: List[Dict]
        List of dictionaries with train loss and test loss values.
    model_names: List[str]
        List of the names of the models.

    Returns nothing
    """    
    # Check if the number of model results agrees with the number of model names
    if len(model_results) != len(model_names):
        raise Exception("Number of model_results does not agree with the number of model names.")
    
    # Generate a list of x values (epochs)
    epochs = range(1,len(model_results[0]["train_loss"])+1)
    
    # Make plot
    fig, axs = plt.subplots(1, 2, figsize = (8,4), sharey=True)
    
    for i in range(len(model_results)):
        axs[0].plot(epochs,model_results[i]["train_loss"], "o-", label = model_names[i])
    
    for i in range(len(model_results)):
        axs[1].plot(epochs,model_results[i]["test_loss"], "o-", label = model_names[i])
        
    # Set titles
    axs[0].set_title("Train loss")
    axs[1].set_title("Test loss")
    
    # Show legend
    axs[1].legend()
    
    # Adjust space between plots
    plt.subplots_adjust(wspace=0.05)
    
    # Set general plot properties
    for i in range(2):
        axs[i].set_ylim(bottom=0)
        axs[i].grid()
        axs[i].set_xlabel("Epochs")  
    
def plot_many_reconstructions(model_list: List[torch.nn.Module],
                             dataset: torch.utils.data.Dataset,
                             model_names: List[str]):
    """
    Plots 3 random samples from `dataset` dataset and their 
    model reconstructions for all models in model_list.

    Parameters
    ----------
    model_list: List[torch.nn.Module]
        Models used to generate the reconstructions
    dataset : torch.utils.data.Dataset
        Dataset with images to reconstruct
    model_names: List[str]
        List of the names of the models.

    Returns nothing
    """
    NUM_SAMPLES = 5
    # Get random sample of 3 images 
    random_idxs = random.sample(range(len(dataset)), k=NUM_SAMPLES)
    
    # Get number of columns in plots = number of models + 1
    ncols = len(model_list) + 1
    
    # Put all models in evaluation mode
    for model in model_list:
        model.eval()
    
    # Make predictions and plot them
    fig, axs = plt.subplots(NUM_SAMPLES, ncols, figsize = (6,9))
    axs[0,0].set_title("Original")
    for i in range(NUM_SAMPLES):
        # Get original image and label, plot them in column 0
        image = dataset.__getitem__(random_idxs[i])[0]
        label = dataset.__getitem__(random_idxs[i])[1]
        axs[i,0].imshow(image.squeeze(), cmap="gray")
        axs[i,0].set_xlabel(f"True label: {label}")
        axs[i,0].set_xticks([])
        axs[i,0].set_yticks([])
        axs[i,0].xaxis.set_label_position('top') 
        
        for j in range(len(model_list)):
            # Get model predictions
            with torch.inference_mode():
                reconstruction = model_list[j](image, "autoencoder")
                pred_label = model_list[j](image.unsqueeze(dim=0), "classifier").argmax().item()
            # Plot prediction
            axs[i,j+1].imshow(reconstruction.squeeze().numpy(), cmap="gray")
            if pred_label == label:
                axs[i,j+1].set_xlabel(f"Prediction: {pred_label}", c = "green")
            else:
                axs[i,j+1].set_xlabel(f"Prediction: {pred_label}", c = "red")
            # Remove ticks and put the label on top
            axs[i,j+1].set_xticks([])
            axs[i,j+1].set_yticks([])
            axs[i,j+1].xaxis.set_label_position('top') 
            
    for j in range(len(model_list)):
        axs[0,j+1].set_title(model_names[j]) 
    
def compute_confusion_matrix(model: torch.nn.Module,
                            dataset: torch.utils.data.Dataset,
                            device: str,
                            make_plot: bool = True):
    """
    Computes confusion matrix of the model.
    
    Parameters
    ----------
    model: torch.nn.Module
        Model to be evaluated
    dataset: torch.utils.data.Dataset
        Data used to compute the confusion matrix
    device: str
        Device where the model is stored, e.g. "cpu"
    make_plot: bool
        Plots confusion matrix if True
        
    Returns the accuracy of the model
    """
    # Setup prediction list and true label list
    y_preds, true_labels = [], []
    # Setup counter to calculate accuracy
    correct_predictions = 0
    # Put model in evaluation
    model.eval()
    # Loop over all images in dataset
    for i in range(len(dataset)):
        # Get image and label from dataset
        image = dataset.__getitem__(i)[0]
        label = dataset.__getitem__(i)[1]
        true_labels.append(label)
        # Make prediction with the model
        pred_label = model(image.unsqueeze(dim=0), "classifier").argmax().item()
        y_preds.append(pred_label)
        if label == pred_label:
            correct_predictions += 1
        
    # Setup an instance of confusion matrix
    confmat = ConfusionMatrix(task="multiclass", 
                              num_classes=len(set(true_labels)))
    confmat_tensor = confmat(
        preds=torch.tensor(y_preds),
        target=torch.tensor(true_labels))
    
    if make_plot == True:
        fig, ax = plot_confusion_matrix(
            conf_mat = confmat_tensor.numpy(), # matplotlib likes working with numpy
            figsize = (10,7),
            class_names = set(true_labels),
            show_normed = True
        )
        
    return correct_predictions / len(dataset)

    
    
    
    
    
    