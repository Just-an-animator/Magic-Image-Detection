# Magic-Image-Detection

![MagicCNN](resources/docs/text-1752255948540.png)

Library and tools to extract information from Magic: The Gathering cards
using traditional AI methods. The goal of this project 
## Current functionality:
* Classify card-type based on color / clan (White, Esper, Abzan, etc) using a multiclass CNN.

## Future improvements
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

## Data
All our current training data is pushed up with the repo, but as the project
expands, we will probably pull from a database. Cards are stored in `resources/data`,
with all card art being pulled directly from `scryfall.com`. Cards should be stored
with an incremental integer as the file name (i.e. `0.jpg` for the 0th card). 

Labels are stored in a simple CSV file in `resources/labels.csv`. This CSV only has
two columns: one for the integer section of the filename (the `0` in `0.jpg`) and the color. The color should be a string name, and is mapped to an integer value in `globals.py`.

## Modelling
tbd