 #!/usr/bin/env python3
"""
File: test.py
Description: Mini project 1 main file
"""
import torch
from torch.utils import data
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn') 
matplotlib.rcParams['font.family'] = 'serif'  
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Avoid iOS bug with pyplot. 

# Data loading

# Import utils
from utils.plot_utils import *
from utils.ml_utils import *

BASIC_MODELS = ['LinearNet', 'LeNet', 'ResNet']



if __name__ == "__main__":

    for name in BASIC_MODELS:
        print("\n\nEvaluation of model: {}".format(name))
        
        # Basic models (without Auxiliary loss nor weight sharing)
        print("\nBasic:")
        full_train_test(model_name=name, auxiliary_loss=False, is_siamese=False, weight_sharing=False)
        
        # Basic models with Auxiliary loss
        print("\nBasic with Auxiliary loss:")
        full_train_test(model_name=name, auxiliary_loss=True, is_siamese=False, aux_rate=1, weight_sharing=False)
        
        # Pseudo-Siemese models (without Auxiliary loss or weight sharing)
        print("\nPseudo-Siemese:")
        full_train_test(model_name=name, auxiliary_loss=False, is_siamese=True, weight_sharing=False)
        
        # Pseudo-Siemese models with Auxiliary loss
        print("\nPseudo-Siemese with Auxiliary loss:")
        full_train_test(model_name=name,  auxiliary_loss=True, is_siamese=True, aux_rate=1, weight_sharing=False)
        
        # Siamese models (with weight sharing)
        print("\nSiemese:")
        full_train_test(model_name=name,  auxiliary_loss=False, is_siamese=True, weight_sharing=True)
        
        # Siamese models with Auxliary loss
        print("\nSiemese with Auxiliary loss:")
        full_train_test(model_name=name,  auxiliary_loss=True, is_siamese=True, aux_rate=1, weight_sharing=True)
        
