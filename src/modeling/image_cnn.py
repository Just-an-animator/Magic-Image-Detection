"""Basic CNN inference"""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class TypeInferencer:
    """Define a model that determines card type given an image.

    Creates a keras model that runs a simple CNN over a provided image, and classifies the card by its
    mana color.

    Attributes
    ----------
    model : models.Model
        The keras model
    
    Methods
    -------
    create_model
        Creates the model
    train(images, labels)
        Trains the model with the given images and labels
    inference(image)
        Runs inference on a single image
    get_summary -> str
        Gets the summary of the model
    """
    def __init__(self):
        self.model: models.Model = None
    
    def create_model(self,):
        """Create the model"""
        self.model = models.Sequential()

        # Add the layers
        self.model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(128, (9, 9), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        
        # We have 7 classes
        self.model.add(layers.Dense(7))

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    def train(self, images: np.ndarray, labels: np.ndarray):
        """Train the model.

        Parameter
        ---------
        images : np.ndarray
            A list of loaded images
        labels : np.ndarray
            A list of labels    
        """
        res = self.model.fit(images, labels, epochs=10, batch_size=1, shuffle=True)
        return res

    def inference(self, image: np.ndarray) -> np.ndarray:
        """Run inference on an image.
        
        Parameters
        ----------
        image : np.ndarray
            The image to run inference on
        
        Returns
        -------
        res : np.ndarray
            The inference results as a one hot encoded array
        """
        return self.model.predict(image)
    
    def get_summary(self,):
        """Get the summary of the model.

        Returns
        -------
        _ : str
            The summary of the model
        """
        return self.model.summary()


if __name__ == "__main__":
    print("Can use keras")
    ti = TypeInferencer()
    ti.create_model()
    ti.get_summary()
