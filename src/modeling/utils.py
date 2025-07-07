"""Generalized utils for modeling"""
import numpy as np
from src.globals import LABEL_DICT

def one_hot_encode(labels: np.ndarray):
    """Performs one hot encoding on the labels.

    Say we have a list of labels: [1, 2, 3, 4] with four images. Instead of using the actual value "4" to 
    represent an image of class 4, we can represent it with [0,0,0,1]. We can repeat this for all
    available labels.
    
    Parameters
    ----------
    labels : np.ndarray
        The labels
    
    Returns
    -------
    ohe : np.ndarray
        The array of labels, one hot encoded
    """
    int_encoded = []
    for label in labels:
        int_encoded.append(LABEL_DICT.get(label, 0)) # Gets the integer value from LABEL_DICT, defaulting to 0 if not found.

    int_encoded = np.asarray(int_encoded) # Convert from a list back to an ndarray

    ohe = np.zeros((int_encoded.size, int_encoded.max()+1)) # Create a numpy array to store our one hot encoded values
    ohe[np.arange(int_encoded.size), int_encoded] = 1 # Do our one hot

    return np.asarray(ohe)
