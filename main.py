from src.image_processing import load_dataset
from src.modeling.image_cnn import TypeInferencer
import numpy as np

if __name__ == "__main__":
    print("Using main as our entrypoint")

    # Get the data
    cards, labels = load_dataset(directory_path="./resources/data/",
                                 labels_path="./resources/labels.csv")
    print(f"Cards vs labels: {np.shape(cards)}, {np.shape(labels)}",)

    # Get the model
    inferencer = TypeInferencer()
    inferencer.create_model()
    inferencer.get_summary()