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

        # First, add an Input layer to the model so it can take in arrays of size 64x64x3 (our card image)
        self.model.add(layers.Input(shape=(64,64,3)))

        # Run a 3x3 convolutional kernel over the image 32 "times", producing a new "image" of shape 64x64x32. This is known as the number of convolutional filters.
        # Each one of these 32 times should, ideally, learn a unique filter over the image.
        self.model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))

        # Downsample the result from the above convolutional layer to a new shape of 32x32x64.
        self.model.add(layers.MaxPooling2D((2,2)))

        # Add another convolutional layer, this time scanning in a 3x3 filter 64 times.
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

        # Downsample again
        self.model.add(layers.MaxPooling2D((2, 2)))
    
        # Add a final convolutional layer, this time even more.
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

        # Final downsample
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Squish the entire resulting image into a 1d array.
        self.model.add(layers.Flatten())

        # Add a fully connected layer
        # ReLU drops out any results below 0. Wouldnt worry too much, but relu is standard inner-model dense layer practice.
        self.model.add(layers.Dense(128, activation='relu'))

        # Randomly drop 50% of the neurons, to stimulate training
        self.model.add(layers.Dropout(0.5))

        # Add a final dense layer, this time "converging" to 7, which will be our number of labels. 
        # Softmax results in "probabilities" of each label type. We then find the maximum of these probabilities and that is our best label.
        self.model.add(layers.Dense(7, activation='softmax'))
    

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    def train(self, images: np.ndarray, labels: np.ndarray, epochs: int = 10, batch_size: int = 1, shuffle: bool = True):
        """Train the model.

        Parameter
        ---------
        images : np.ndarray
            A list of loaded images
        labels : np.ndarray
            A list of labels    
        """
        res = self.model.fit(images, labels, epochs=epochs, batch_size=batch_size, shuffle=shuffle)
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
            The inference results as a softmax array
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
