"""Image processing utils."""
import numpy as np
import PIL
import natsort
import glob
import os,sys
from typing import Tuple


def _load_image(path: str) -> np.ndarray:
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

def load_dataset(directory_path: str, labels_path: str) -> Tuple[np.ndarray, list]:
    """ Loads our entire dataset. 

    Parameters
    ----------
    directory_path : str    
        The location of the dataset. In this repo it will be resources/data
    labels_path : str
        The location of the labels csv. In this repo it will be resources/labels.txt

    """
    print(f"Grabbing images from {directory_path} and labels from {labels_path}")

if __name__ == "__main__":
    print("Testing image loading...")
    _load_image("../resources/data/0.jpg")
    load_dataset("../resources/data", "../resources/labels.txt")