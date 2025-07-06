"""Image processing utils."""
import numpy as np
from PIL import Image
import natsort
import glob
import os, sys, csv
import matplotlib.pyplot as plt
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
    i = Image.open(path)
    return np.array(i)

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
    
    # Loads in the images as numpy arrays and stores them in an array called `cards`
    cards= []
    for f in natsort.natsorted(glob.glob(directory_path + "*" )):
        cards.append(_load_image(f))
        print(f)
    
    # Loads in our labels and stores them in an array called `labels`
    labels = []
    with open(labels_path, "r") as fr:
        reader = csv.reader(fr)
        for row in reader:
            labels.append(row[1])

    # Convert both to numpy arrays and return them as a tuple
    return np.asarray(cards), np.asarray(labels)
    
if __name__ == "__main__":
    print("Testing image loading...")

    # Get our cards and labels as numpy arrays
    cards, labels = load_dataset("..\\resources\\data\\", "..\\resources\\labels.txt")
    
    # Uncomment if we want to see the images
    # plt.imshow(cards[0])
    # plt.show()

    print(f"labels: {labels}")