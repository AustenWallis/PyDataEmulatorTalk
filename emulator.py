""" Author: Austen Wallis
    Date: 2024-10-12
    Description: This script is used to train a neural network to emulate the 
    output of a radiative transfer model. This is an example used in combination
    with a presentation 'Faster Models, Faster Answers'.
    Copyright (c) 2024 Austen Wallis
"""

# %%
###########################################
print('STEP: Importing Libraries')
###########################################

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import fnmatch
import itertools
import pandas as pd

# %%
###########################################
print('STEP: Loading the data')
###########################################

path_to_data = 'small_spectrum_dataset' # Location of Files
number_of_files = len(os.listdir(path_to_data)) # Number of Files

# Example file name: run01_WMdot4e-11_d2_vinf1p5.spec
# The names states the parameters and values associated

spectra = [] # List for full paths to each file
for iteration in range(1, number_of_files): # Iterated to have a sorted list
    filename = fnmatch.filter(os.listdir(path_to_data), f'run{iteration:02}_*')[0]
    spectra.append(os.path.join(path_to_data, filename)) # Full path to each file

# wavelengths consistent across files, reversing array for increasing wavelength
wavelengths = np.loadtxt(spectra[0], usecols=1, skiprows=81)[::-1] 

spectral_data = [] # List for the spectral data
for spectrum in tqdm(spectra):
    # Load data and add to a list
    data = np.loadtxt(spectrum, usecols=16, skiprows=81) # Skip header, column 16, 60° inclination
    spectral_data.append(data)
    
spectral_data = np.array(spectral_data) # Convert to numpy array

# Example file name: run01_WMdot4e-11_d2_vinf1p5.spec
# Harcoded values for the parameters in the spectrum dataset (for demonstration)
unique_values = [[4e-11, 1e-10, 4e-10, 1e-9, 3e-9],
                       [2, 5, 8, 12, 16],
                       [1, 1.5, 2, 2.5, 3]]

# Wind mass loss rate, geometry of wind cone and wind speed, See Sirocco for more.
parameter_names = ['WMdot', 'd', 'vinf'] 

# using itertools to get all combinations of parameters
unique_combinations = list(itertools.product(*unique_values))

# convert to a pandas dataframe
df = pd.DataFrame(unique_combinations, columns=parameter_names)



# %%
###########################################
print('STEP: Train Test Split')
###########################################

# When train test splitting, be sure that any standardisation on the test data
# is done using the mean and standard deviation from the training data. This is
# to prevent data leakage.

X_train, X_test, y_train, y_test = train_test_split(df, spectral_data, 
                                                    test_size=0.2, 
                                                    random_state=8765)


# %%
###########################################
print('STEP: Standardisation')
###########################################
# A sensible next step, be sure test data uses the mean and standard deviation
# from the training data, to avoid data leakage.


# %%
###########################################
print('STEP: Training the Neural Network')
###########################################
# Generate a emulator using a neural network


# %%
###########################################
print('STEP: Compare emulations to data')
###########################################
# Use your emulator to predict the spectra for the test data
# Compare the predicted spectra to the actual spectra
# How well did you do ?!

