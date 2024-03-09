#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File containing various utility functions for PyTorch model training.
"""
import torch
from pathlib import Path
import os
import pickle

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """
  Saves a PyTorch model to a target directory.
  Downloaded from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular/utils.py

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
def delete_files_in_directory(directory_path):
    """
    Deletes all files from directory_path.
    Source: https://www.tutorialspoint.com/How-to-delete-all-files-in-a-directory-with-Python

    """
    try:
      with os.scandir(directory_path) as entries:
        for entry in entries:
          if entry.is_file():
             os.unlink(entry.path)
      print("All files deleted successfully.")
    except OSError:
      print("Error occurred while deleting files.")
      
def load_model(model: torch.nn.Module, model_path: str, model_name: str):
    """
    Loads the model paramaters from model_path/model_name into model.

    Parameters
    ----------
    model: torch.nn.Module
        The neural network model to be updated
    model_path : str
        Path containing the model
    model_name : str
        Name of the file with the model.
    """
    MODEL_PATH = Path(model_path)
    MODEL_SAVE_PATH = MODEL_PATH / model_name
    
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    
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

