###############################################################################
### IMPORTS
###############################################################################
# I/O
import os
import h5py
import sys
sys.path.append('..')
import matplotlib.pyplot as plt

# DeepL
from utils.data_standardization import unstandardize_rd, standardize_rd, map_intervals

# Math
import numpy as np
from scipy.ndimage import sobel, generic_gradient_magnitude

###############################################################################
### Purpose: add additional data
###############################################################################

# Load current dataset
h5_file = h5py.File(os.path.join('..',
                               '..',
                               'data', 
                               'dataset_resized_rd_summed.h5'),
                  'r+')

# For each patient
for key in list(h5_file.keys()):

    # Grab the dose plan
    dose = unstandardize_rd(h5_file[key]['dose'][()])
    
    # Manage isodose
    # - convert to nearest 10
    # - standardize according to dose min/max across patients (CAN't if we want a crossentropy-like loss)
    #isodose = standardize_rd(np.round(dose/10)*10)
    # - OR don't normalize
    isodose = np.round(dose/10)
    
    # Test it visually
    '''
    image = None
    for h in range(isodose.shape[2]):
        if image is None:
            image = plt.imshow(isodose[:, :, h], cmap='jet', vmin=0, vmax=7) # resp vmax=1
        else:
            image.set_data(isodose[:, :, h])
        plt.pause(0.0001)
        plt.draw()
    '''
    
    # Add to the dataset
    '''
    # Delete the entry if it already exists
    if key in h5_file.keys():
        if 'isodose' in h5_file[key].keys():
            del h5_file[key + '/isodose']
    # Add
    h5_file.create_dataset(key + '/isodose',
                           data=isodose,
                           compression='gzip')
    '''
    
    # Manage edges
    edges = generic_gradient_magnitude(dose, sobel)
    
    # Debug
    # > trying to figure what each function does, how it behaves with
    # 3D inputs
    # > answer is: it does what we want, let's replicate it with tensors
    # during loss computation
    '''
    edges_x = sobel(dose, 0)
    edges_y = sobel(dose, 1)
    edges_z = sobel(dose, 2)

    mag = np.sqrt(edges_x**2 + edges_y**2 + edges_z**2)
    
    print(np.array_equal(edges, mag))
    '''    
    
    vmin = np.min(edges)
    vmax = np.max(edges)
    edges = map_intervals(edges, vmin, vmax, 0, 1)
    
    # Test it visually
    '''
    image = None
    for h in range(edges.shape[2]):
        if image is None:
            image = plt.imshow(edges[:, :, h], cmap='jet', vmin=0, vmax=1)
        else:
            image.set_data(edges[:, :, h])
        plt.pause(0.0001)
        plt.draw()
    '''
    
    #sys.exit()
    
    # Add to the dataset
    #'''
    # Delete the entry if it already exists
    if key in h5_file.keys():
        if 'edges' in h5_file[key].keys():
            del h5_file[key + '/edges']
    # Add
    h5_file.create_dataset(key + '/edges',
                           data=edges,
                           compression='gzip')
    #'''