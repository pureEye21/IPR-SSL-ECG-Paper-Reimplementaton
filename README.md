# SSL-ECG-Paper-Reimplementaton
This is the reimplementation of the paper [Sarkar and Etemad 2020 "Self-Supervised ECG Representation Learning for Emotion Recognition"](https://ieeexplore.ieee.org/document/9161416) for my course [Intelligent Pattern Recognition](https://home.iitk.ac.in/~sandhan/ipr.html) project

## Setup

I used Python's virtual environment. Follow below steps to setup the environment

1.**Create a virtual environment**:
   ```bash
   python -m venv myenv  # Replace 'myenv' with your desired environment name
```

2.**Activate the environment**:
```bash
source myenv/bin/activate
```
3.**Install required libraries**:

Install all the libraries mentioned in the requirements.txt. One example is mentioned below - 
```
pip install torch
```

**Adding folders:**

After installing the environment, follow below steps

create a folder `cache` on the top level, and then one for each dataset you want to use inside create -  `cache/dreamer`, `cache/amigos`

create a folder `checkpoints` on the top level

also edit the file `src/constants.py` and change the variables towards the correct base paths

## Mode of usage
`src/main.py` is the start point for running the model where you can specify different types of parameters. 

The dataset will be preprocessed and the `npy` files will be stored in `cache/dataset`

At certain checkpoints model will be saved in `checkpoints/`

I uploaded the pretrained models in `src/data_model`

For the pretraining purpose you must download each of the datasets:
- [DREAMER](https://zenodo.org/record/546113#.YKLY0WYzaHs)
- [AMIGOS](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html)

You must configure the basepath for the datasets in `src/constants.py`
