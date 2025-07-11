# Magic-Image-Detection

![MagicCNN](resources/docs/text-1752255948540.png)

Library and tools to extract information from Magic: The Gathering cards
using traditional AI methods.

## Table of Contents  
[Functionality](#functionality)  
[Setup](#setup)  
[Running](#running)     
[Data](#data)   
[Modelling](#modelling)

## Functionality
* Classify card-type based on color / clan (White, Esper, Abzan, etc) using a multiclass CNN.

### Future improvements
* Detect MTG cards live during a webcam feed using OpenCV, and forward the detected card to the CNN.
* Extract information about the detected card and store it in
a local database (Mana cost, card type, abilities, etc)

## Setup
Ideally use `conda` or `venv` to manage environments, as we use `python3.10` for
all our models. 

### Setup: Conda
Run the following command to create a conda environment:
```
conda env create -n tf-magic python=3.10
```
Then activate the environment:
```
conda activate tf-magic
```
Finally install the `requirements.txt` to get all of the packages:
```
pip install -r requirements.txt
```

### Setup: venv
Run the following command to create a virtual environment. The path is within
the `.gitignore` so it won't get pushed up. (Requires `python3.10` installed beforehand)
```
python3.10 -m venv ./venv
```
Activate the environment:
```
chmod +X ./venv/bin/activate
./venv/bin/activate
```
Finally pip install the requirements:
```
python3.10 -m pip install -r requirements.txt
```

## Running
To run the system, either use the scripts provided in the `scripts/` directory,
or use the `main.py` entrypoint. The args go as follows:

| Arg    | Type | Desc | 
| -------- | ------- | ------- |
| -h, --help  | flag    | Shows the arguments |
| --train | flag     | Toggles training of the model |
| --inference    | flag    | Toggles inference of the model |
| --epochs `EPOCHS` | int | The number of epochs to train for, if --train is flagged | 
| --card `CARD` | str | The filepath to the card for inference, if --inference is flagged |
| --data_dir `DIR` | str | The filepath for the dataset directory |
| --labels_file `FILE` | str | The filepath for the labels file |
| --save | flag | Flag to save the model weights |
| --checkpoint_dir `DIR` | str | The directory to save model weights |


## Data
All our current training data is pushed up with the repo, but as the project
expands, we will probably pull from a database. Cards are stored in `resources/data`,
with all card art being pulled directly from `scryfall.com`. Cards should be stored
with an incremental integer as the file name (i.e. `0.jpg` for the 0th card). 

Labels are stored in a simple CSV file in `resources/labels.csv`. This CSV only has
two columns: one for the integer section of the filename (the `0` in `0.jpg`) and the color. The color should be a string name, and is mapped to an integer value in `globals.py`.

The color / clan guide goes as follows:
| Name    | Color pairings | Index |
| -------- | ------- | ------- |
| white  | white    | 0 |
| red | red    | 1| 
| green | green | 2 |
| blue | blue | 3 |
| black | black | 4 |
| gold | gold | 5 | 
| colorless | colorless | 6 |
| gruul | Red, Green | 7 |
| simic | Green, Blue | 8 |
| golgari | Black, Green | 9 |
| selesyna | White, Green | 10 |
| dimir | Blue, black | 11 |
| azorius | White, Blue | 12 |
| izzet | Blue, Red | 13 | 
| boros | White, Red | 14 |
| rakdos | Black, Red | 15 |
| orzhov | White, Black | 16 |
| mardu | Red, White, Black | 17 |
| jeskai | Blue, Red, White | 18 | 
| abzan | White, Black, Green | 19 |
| sultai | Black, Green, Blue | 20 |
| temur | Green, Blue, Red | 21 |
| esper | White, Blue, Black | 22 |
| grixis| Blue, Black, Red | 23 |
| bant | Green, White, Blue | 24 |
| jund | Black, Red, Green | 25 |
| naya | Red, Green, White | 26 |

## Modelling
Currently, all modelling work is done in `tensorflow<2.11`, primarily for python
3.10 compatability. We use `keras` for the current modelling, with `numpy` and 
`pillow` use for preprocessing and matrix manipulation. 

Models should be wrapped in a base `TypeInferencer` class to keep method
signatures consistent across different model types. This will most definitely
be base-classed in the future.

The core model, at the moment, is a simple CNN with the following definition:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 64, 64, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 32, 32, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 32, 32, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 16, 16, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 16, 16, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 8, 8, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 8192)              0         
                                                                 
 dense (Dense)               (None, 128)               1048704   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 27)                3483      
                                                                 
=================================================================
Total params: 1,145,435
Trainable params: 1,145,435
Non-trainable params: 0
```

Future models (segmentation, extraction, etc) will most definitely be complicated
but will continue to be documented here.