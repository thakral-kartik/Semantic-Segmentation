import numpy as np
import os

def get_new_dataset():
    x_test = np.load(os.path.join('npys', 'x_test_covid.npy'))
    y_test = np.load(os.path.join('npys', 'y_test_mask.npy'))
    name_list = list(np.load(os.path.join('npys', 'x_test_name_list_covid.npy')))
    
    return x_test, y_test, name_list