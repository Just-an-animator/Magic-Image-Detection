"""Utility functions for processing images"""
import numpy as np
from PIL import Image

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
        image = np.reshape(image, target_dims)
    except Exception as e:
        print(f"Could not reshape image. Returning base image. -- {e}")
    return image

def normalize_image(image: np.ndarray):
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
    norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return norm