"""Generalized utils for modeling"""
import numpy as np

label_dict = {
    "white": 1,
    "red": 2,
    "green": 3,
    "blue": 4,
    "black": 5,
    "gold": 6,
    "colorless": 7
}

def one_hot_encode(labels: np.ndarray):
    """Performs one hot encoding on the labels.
    
    Parameters
    ----------
    labels : np.ndarray
        The labels
    
    Returns
    -------
    ohe : np.ndarray
        The array of labels, one hot encoded
    """
    pass
