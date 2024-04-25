# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:57:31 2024

@author: rczup

We include here the utility functions.
"""
import random
import matplotlib.pyplot as plt
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def plot_random_images(test_dataset, class_names):
    """
    Plots random 9 images from test_dataset.
    
    Based on:
    https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb
    """
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_dataset), k=9):
        test_samples.append(sample)
        test_labels.append(label)
    
    # Plot predictions
    plt.figure(figsize=(9,9))
    nrows , ncols = 3, 3
    for i, sample in enumerate(test_samples):
        # Create a subplot for each sample
        plt.subplot(nrows, ncols, i+1)
    
        # Plot target image
        plt.imshow(sample.squeeze(), cmap="gray")
    
        # Get the truth label
        truth_label = class_names[test_labels[i]]
        
        # Create a title of the plot
        plt.title(f"Label: {truth_label}")
    
        plt.axis(False)
        
def plot_reconstructions(model: torch.nn.Module,
                         dataset: torch.utils.data.Dataset,
                         device,
                         class_names):
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
        image = dataset.__getitem__(random_idxs[i])[0].to(device)
        label = class_names[ dataset.__getitem__(random_idxs[i])[1] ]
        # Forward pass
        with torch.inference_mode():
            reconstruction = model(image, "autoencoder")
            pred_label = class_names[ model(image.unsqueeze(dim=0), "classifier").argmax().item() ]
        axs[i,0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axs[i,1].imshow(reconstruction.squeeze().cpu().numpy(), cmap="gray")
        axs[i,0].set_axis_off()
        axs[i,1].set_axis_off()
        axs[i,0].set_title(f"True label: {label}")
        axs[i,1].set_title(f"Prediction: {pred_label}")
    plt.suptitle("Left: original | Right: reconstruction")

def plot_loss_curve(model_results: dict):
    """
    Plots train and test loss curves for the data contained in model_results

    Parameters
    ----------
    model_results: dict
        Dictionary with model resutls

    Returns nothing
    """    
    
    # Generate a list of x values (epochs)
    epochs = range(1,len(model_results["train_loss"]))
    
    # Make plot
    plt.figure(figsize = (6,4))
    plt.plot(model_results["train_loss"], label = "train loss")
    plt.plot(model_results["test_loss"], label = "test loss")
    plt.legend()
    plt.xlabel("Epoch")

def compute_confusion_matrix(model: torch.nn.Module,
                            dataset: torch.utils.data.Dataset,
                            device: str,
                            class_names: list,
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
    class_names: list
        List with class names
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
            class_names = class_names,
            show_normed = True
        )
        
    return correct_predictions / len(dataset)