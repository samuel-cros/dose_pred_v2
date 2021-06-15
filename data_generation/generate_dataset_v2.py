###############################################################################
### IMPORTS
###############################################################################
# I/O
import os
import h5py
import sys
sys.path.append('..')
import pickle

# Math
import numpy as np
from utils.data_standardization import standardize_ct, standardize_rd
from scipy.ndimage import zoom

###############################################################################
### Subfunctions
###############################################################################
# mask_to_channel
# - goal: get the corresponding channel (int between 1 and 20) from a mask name
# - input: str mask name
# - output: int channel number (between 1 and 20)
def mask_to_channel(mask_name):
    if (mask_name == "ptv 1"):
        return 1
    elif (mask_name == "ctv 1"):
        return 2
    elif (mask_name == "gtv 1"):
        return 3
    elif (mask_name == "canal medullaire"):
        return 4
    elif (mask_name == "canal medul pv"):
        return 5
    elif (mask_name == "oesophage"):
        return 6
    elif (mask_name == "cavite orale"):
        return 7
    elif (mask_name == "mandibule"):
        return 8
    elif (mask_name == "trachee"):
        return 9
    elif (mask_name == "tronc"):
        return 10
    elif (mask_name == "tronc pv"):
        return 11
    elif (mask_name == "parotide g"):
        return 12
    elif (mask_name == "parotide d"):
        return 13
    elif (mask_name == "oreille int g"):
        return 14
    elif (mask_name == "oreille int d"):
        return 15
    elif (mask_name == "oeil g"):
        return 16
    elif (mask_name == "oeil d"):
        return 17
    elif (mask_name == "sous-max g"):
        return 18
    elif (mask_name == "sous-max d"):
        return 19
    elif (mask_name == "nerf optique g"):
        return 20
    else:
        raise NameError("Unknown channel : %s" % mask_name)
    
# pad_and_resize
# - goal: pad a given array's height to the closest multiple of 16
# - output: padded array
def pad_to_closest_16(array):
    missing_height_to_16 = 16 - (array.shape[2] % 16)
    lower_padding = missing_height_to_16 // 2
    higher_padding = lower_padding + missing_height_to_16 % 2
    padded_array = np.zeros((array.shape[0],
                             array.shape[1],
                             array.shape[2]+lower_padding+higher_padding))
    padded_array[:, :, lower_padding:array.shape[2]+lower_padding] = array
    return padded_array   

###############################################################################
### Purpose: reformat input -> output
###############################################################################

## Input
# - 1 file
#   - rd
#   - ct
#   - body
#   - oar1, oar2, ...
#   - ptv 1 and/or ctv 1 and/or gtv 1

## DONE
## DATASET 2
# idea: cleaner version of current + ready to load
# - resize by zoom with value 0.25
resize_factor = (0.25, 0.25, 0.25)
# - padd to a multiple of 16 (2*2*2*2) for the 4 max poolings
# - mask ct using body channel
# - mask rd using body channel
# - input = 0:CT | 1:PTV | 2:CTV | 3:GTV  | 4:OAR1 | ... | 20:OAR17
# - dose = Summed RD
# (- isodose = Isodose lines)
# (- edges = ...)
# - body (for sampling)

# 38 average empty slices
# 256 average height

####

## DATASET 3?
# idea: all OARs on the same channel, all TVs on the same channel
# + ready to load
# - mask ct using body channel
# - mask rd using body channel
# - input = 0:CT | 1:TVs | 2:OARs
# - ouput = summed RD
# - body (for sampling)

###############################################################################
### MAIN
###############################################################################

# Paths
path_to_current_data = os.path.join('..', '..', 'data', 'CHUM', 'npy')
path_to_new_data = os.path.join('..', '..', 'data')

# Init h5 dataset
h5_file = h5py.File(os.path.join(path_to_new_data, 
                            'dataset_resized_rd_summed.h5'), 
                    'w')

# Prescribed dose
dict_file = open("dict_ID_Gy.pkl", "rb")
dict_ID_Gy = pickle.load(dict_file)

# Init list ids
list_IDs = np.load('selected_ids.npy')

# For file in path_to_current_data
for ID in list_IDs:
    print(ID)
    
    # Load npz file
    data = np.load(os.path.join(path_to_current_data, ID + '.npz'))
    
    # Grab input shape
    input_shape = data['ct'].shape
    
    ## Crop the volume along empty heights
    # Use body for it since it's later used for masking
    zoomed_body = zoom(data['body'], resize_factor)
    # Compute non_zero height values
    non_zero_heights = zoom(data['body'], resize_factor).nonzero()[2]
    non_zero_heights.sort()
    # Grab lower and higher bounds
    lower_b, higher_b = non_zero_heights[0], non_zero_heights[-1]
    
    # Manage body
    body = pad_to_closest_16(zoomed_body[:, :, lower_b:higher_b])
    
    h5_file.create_dataset(ID + '/body', 
                           data=body, 
                           compression='gzip')
    
    # Init input
    new_input = np.zeros((body.shape[0],
                          body.shape[1],
                          body.shape[2], 21))
    
    # Init output
    dose = np.zeros(body.shape)

    # Manage CT
    # - min/max normalization
    # - resized, cropped and padded
    # - masking with body
    print('CT')
    new_input[:, :, :, 0] = \
        pad_to_closest_16(zoom(standardize_ct(data['ct']),
                                resize_factor)[:, :, lower_b:higher_b]) * body
    
    # Manage MASKS
    # - resized, cropped and padded
    # - masking with body
    # could be not masked, structures are always relevant and we admit they don't
    # present much defects
    omit = ['ct', 'rd', 'body']
    for field in data.keys():
        if field not in omit:
            print(field)
            new_input[:, :, :, mask_to_channel(field)] = \
                pad_to_closest_16(zoom(data[field],
                                       resize_factor)[:, :, lower_b:higher_b]) * body
                
    h5_file.create_dataset(ID + '/input',
                           data=new_input,
                           compression='gzip')
    
    # Manage RD
    # - min/max normalization
    # - resized, cropped and padded
    # - masking with body
    print('RD')
    dose = \
        pad_to_closest_16(zoom(standardize_rd(np.sum(data['rd'], axis=0)),
                               resize_factor)[:, :, lower_b:higher_b]) * body
    
    h5_file.create_dataset(ID + '/dose', 
                           data=dose, 
                           compression='gzip')
    
    # Manage prescribed dose
    h5_file.create_dataset(ID + '/prescribed_dose',
                           data=dict_ID_Gy[ID])
    