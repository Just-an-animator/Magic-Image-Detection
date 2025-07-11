from src.modeling.image_processing import load_dataset, reshape_image, _load_image
from src.modeling.image_cnn import TypeInferencer
from src.modeling.utils import one_hot_encode
import numpy as np

if __name__ == "__main__":
    print("Using main as our entrypoint")

    # Get the data
    cards, labels = load_dataset(directory_path="./resources/data/",
                                 labels_path="./resources/labels.csv", max_size=8)
    print(f"Cards vs labels: {np.shape(cards)}, {np.shape(labels)}",)

    # Get the model
    inferencer = TypeInferencer()
    inferencer.create_model()
    inferencer.get_summary()

    # Perform one hot on our labels
    ohe_labels = one_hot_encode(labels)

    inferencer.train(cards, ohe_labels, epochs=25)

    # Test on kathril
    katty = np.array([_load_image("./resources/data/0.jpg", target_resize_dims=(64,64))])
    results = np.argmax(inferencer.inference(katty))
    print(f"We have determined that Kathril, Aspect Warper, which should have a label of 5 is: {results}")