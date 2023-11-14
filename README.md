# Ensembles of Quantum Classifiers

This repository contains the code for the paper Ensembles of Quantum Classifiers.

Datasets and code are based on: https://github.com/ZarHenry96/quantum-ml-pipeline.

## Project structure

- classification/
    - classifiers/ - classical and quantum classifiers
    - ensembles/ - ensemble methods implementation
- data/
    - dataset/ - datasets
    - folds/ - data folding used for tests
- test_ex.py - example testing script

## Run

### Setup environment

#### Get code
    git clone https://github.com/emiliantolo/ensembles-quantum-classifiers.git
    cd ensembles-quantum-classifiers

#### Install dependencies
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Run experiments

    python text_ex.py --data 0 --fold 0 

Optional parameters are ```--data (-d)```, ```--fold (-f)```, and ```--classifier (-c)```, with the slicing or indexing of the related lists defined in the script, with format: ```(-)?[0-9]\*(:)?(-)?[0-9]\*```.

    python test_ex.py          # run all experiments 
    python test_ex.py -d 1:-1  # skip the first and last datasets
    python test_ex.py -f :3    # run the first 3 folds
    python test_ex.py -c 0     # run only the first classifier
