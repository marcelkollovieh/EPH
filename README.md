## About
This is the code for the submission "Expected Probabilistic Hierarchies" (submission number: 5350).
## Installation
To install the requirements using pip:

    pip install -r requirements.txt 
    pip install -e .

## Usage
A simple demo how to load data and train EPH is given in the notebook `demo.ipynb`. 
To reproduce the results from the paper, we recommend using the script `eph_experiment.py`. 
The corresponding configurations and datasets are provided in `./configs` and `./datasets`, respectively.

Example usage for the `citeseer` dataset:

    python eph_experiment.py -c configs/exp_das_citeseer.yaml