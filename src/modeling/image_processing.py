"""Utility functions for processing images"""
import numpy as np
from PIL import Image
import natsort
import glob
import os, sys, csv
import matplotlib.pyplot as plt
from typing import Tuple


def _load_image(path: str, target_resize_dims: set = (64,64)) -> np.ndarray:
    """ Loads a singular image given an image path

    Parameters
    ----------
    path : str
        The path of an image (jpg, png, etc)
    
    Returns
    -------
    img : np.ndarray
        The image, loaded in, and foramtted as a numpy array.

    """
    print(f"Loading image at path: {path}")
    i = Image.open(path)
    i = resize_image(i, target_resize_dims)
    return np.array(i)

def load_dataset(directory_path: str, labels_path: str, target_resize_dims: set = (64,64), max_size: int = None) -> Tuple[np.ndarray, list]:
    """ Loads our entire dataset. 

    Parameters
    ----------
    directory_path : str    
        The location of the dataset. In this repo it will be resources/data
    labels_path : str
        The location of the labels csv. In this repo it will be resources/labels.txt

    """
    print(f"Grabbing images from {directory_path} and labels from {labels_path}")
    
    # Loads in the images as numpy arrays and stores them in an array called `cards`
    cards= [] # 

    # Do this if we only want to load in a fixed number
    if (max_size):
        for f in natsort.natsorted(glob.glob(directory_path + "*" ))[:max_size]: # Iterate through a sorted list of file names
            cards.append(_load_image(f)) # Load the image at the file path as a numpy array
    else:
        for f in natsort.natsorted(glob.glob(directory_path + "*" ))[:max_size]: # Iterate through a sorted list of file names
            cards.append(_load_image(f)) # Load the image at the file path as a numpy array
    # Loads in our labels and stores them in an array called `labels`
    labels = [] 
    with open(labels_path, "r") as fr: # Open the labels path and read in the bytes
        reader = csv.reader(fr) # Create a CSV reader to read row-by-row
        for row in reader: # Read row by row
            labels.append(row[1]) # Add the 1st column from the row, which is our color.

    # Convert both to numpy arrays and return them as a tuple
    return np.asarray(cards), np.asarray(labels)

def reshape_image(image: np.ndarray, target_dims: set = (64, 64, 3)):
    """Attempts to reshape the image to the provided dims.
    
    Parameters
    ----------
    image : np.ndarray
        The image in question
    target_dims : set
        The dimensions to resize the image to.

    Returns
    -------
    image : np.ndarray
        The image, potentially resized
    """
    try:
        image = np.reshape(image, target_dims) # Reshape the image to the target dimension
    except Exception as e:
        print(f"Could not reshape image. Returning base image. -- {e}")
    return image

def resize_image(image: Image, target_dims: set) -> Image:
    """ Attempts a lossy rescale of the image.

    Parameters
    ----------
    image : Image
        A PIL Image to resize
    target_dims : set
        A list of dimes to resize to.
    """
    return image.resize(target_dims)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the images. Note: this is not RGB safe and we probably wont use it.

    Parameters
    ----------
    image : np.ndarray
        The image to resize
    
    Returns
    -------
    norm : np.ndarray
        The normalized image
    """
    norm = (image - np.min(image)) / (np.max(image) - np.min(image)) # Normalize the image between the min and max
    return norm