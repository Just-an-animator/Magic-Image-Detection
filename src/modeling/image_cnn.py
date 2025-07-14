"""Basic CNN inference."""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
    def __init__(self, epochs: int = 100):
        self.model: models.Model = None
        self.history: any = None
        self.epochs = epochs
    
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
        self.model.add(layers.Dense(27, activation='softmax'))

        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    def save_weights(self, model_checkpoint_path: str = "./"):
        """Save the model weights so we dont have to train the model every
        run.
        """
        self.model.save_weights(model_checkpoint_path + "/cnn_checkpoints.h5")

    def plot_metrics(self):
        """
        """
        epochs = range(0, self.epochs)
        loss = self.history.history["loss"]
        acc = self.history.history["accuracy"]

        d = {"epochs": epochs, "loss": loss}
        d2 = {"epochs": epochs, "acc": acc}

        p1 = pd.DataFrame(data=d)
        p2 = pd.DataFrame(data=d2)

        sns.color_palette("rocket")
        ax1 = sns.lineplot(x="epochs", y="acc", data=p2)
        ax1.set_title("Accuracy during 100 training steps")
        ax1.xaxis.grid(True)
        ax1.yaxis.grid(True)

        ax1.set_xlabel("# of Epochs")
        ax1.set_ylabel("Classification Accuracy")
        # sns.move_legend(ax1, "upper left")

        plt.show()



    def load_weights(self, model_checkpoints_path: str = "./cnn_checkpoints.h5"):
        """ Load the weights of the model so we can run inference and fine tune.

        Parameters
        ----------
        model_checkpoints_path : str
            The checkpoints of the model.
        """
        try:
            self.model.load_weights(model_checkpoints_path)
        except Exception as e:
            print(f"Could not load model weights: -- {e}")
    
    def train(self, images: np.ndarray, labels: np.ndarray, epochs: int = 10, batch_size: int = 1, shuffle: bool = True):
        """Train the model.

        Parameter
        ---------
        images : np.ndarray
            A list of loaded images
        labels : np.ndarray
            A list of labels
        """
        self.history = self.model.fit(images, labels, epochs=epochs, batch_size=batch_size, shuffle=shuffle)
        return self.history

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
