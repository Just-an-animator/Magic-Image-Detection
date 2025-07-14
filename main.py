"""Main entrypoint for the model."""
from src.modeling.image_processing import load_dataset, reshape_image, _load_image
from src.modeling.image_cnn import TypeInferencer
from src.modeling.utils import one_hot_encode
from src.globals import LABEL_DICT
import numpy as np
import argparse


def _get_label_name(label: int):
    """Get the label name given the integer ecoded label.

    Parameters
    ----------
    label : int
        The argmax'd softmax'd result of the model.
    
    Return
    ------
    k : str
        The label str name (abzan, esper, etc).
    """
    for k, v in LABEL_DICT.items():
        if (v == label):
            return k
    return "NONE"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to run inference, train, and evaluate componnets from MagicCNN")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Flag to train the model.")
    mode_group.add_argument("--inference", action="store_true", help="Flag to run inference.")
    parser.add_argument("--epochs", type=int, required=False, help="The number of training epochs", default=100)
    parser.add_argument("--card", type=str, required=False, help="Single card file path to run inference on")
    parser.add_argument("--data_dir", type=str, required=False, help="Directory containing the data.", default="./resources/data/")
    parser.add_argument("--labels_file", type=str, required=False, help="Labels CSV file location", default="./resources/labels.csv")
    parser.add_argument("--save", action="store_true", required=False, help="Whether we want to save weights or not")
    parser.add_argument("--checkpoint_dir", type=str, required=False, help="Directory to save checkpoints.", default="./resources/checkpoints")

    args = parser.parse_args()

    if args.train:
        print("Training the model!")
        cards, labels = load_dataset(directory_path=args.data_dir,
                                     labels_path=args.labels_file)
        inferencer = TypeInferencer()
        inferencer.create_model()
        inferencer.get_summary()
        ohe_labels = one_hot_encode(labels)
        inferencer.train(cards, ohe_labels, epochs=args.epochs)

        if args.save:
            inferencer.save_weights(args.chekpoint_dir)

    if args.inference:
        inferencer = TypeInferencer()
        inferencer.create_model()
        inferencer.get_summary()
        inferencer.load_weights(args.checkpoint_dir + "cnn_checkpoints.h5")
        card = np.array([_load_image(args.card, target_resize_dims=(64, 64))])
        res = np.argmax(inferencer.inference(card))
        print(f"Ran inference on {args.card}: got label {res}, {_get_label_name(res)}")


    # # Get the data
    # cards, labels = load_dataset(directory_path="./resources/data/",
    #                              labels_path="./resources/labels.csv")
    # print(f"Cards vs labels: {np.shape(cards)}, {np.shape(labels)}",)

    # # Get the model
    # inferencer = TypeInferencer()
    # inferencer.create_model()
    # inferencer.get_summary()

    # # Perform one hot on our labels
    # ohe_labels = one_hot_encode(labels)

    # inferencer.train(cards, ohe_labels, epochs=100)

    # inferencer.save_weights("/Users/e361818/Projects/pet/Magic-Image-Detection/resources/checkpoints")

    # # Test on kathril
    # katty = np.array([_load_image("./resources/data/0.jpg", target_resize_dims=(64,64))])
    # results = np.argmax(inferencer.inference(katty))
    # print(f"We have determined that Kathril, Aspect Warper, which should have a label of 5 is: {results}")