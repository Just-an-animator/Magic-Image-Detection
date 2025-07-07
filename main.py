from src.modeling.image_processing import load_dataset, reshape_image
from src.modeling.image_cnn import TypeInferencer
from src.modeling.utils import one_hot_encode
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

    # Perform one hot on our labels
    ohe_labels = one_hot_encode(labels)

    print(f"OHE: {ohe_labels}")
    print(np.shape(ohe_labels))
    print(np.shape(cards))

    inferencer.train(cards, ohe_labels)