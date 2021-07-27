###############################################################################
## Imports
###############################################################################
# Math
import numpy as np
import sys
sys.path.append('..')
from utils.data_standardization import unstandardize_rd
from metrics import *

#

# IO
import h5py
from pathlib import Path
import os
import argparse
import csv

###############################################################################
## Subroutines
###############################################################################
    
# get_biggest_tv
# - goal: get ptv, ctv or gtv (in that order) depending on availability
# - input: input for a given patient where 
#       - input[:, :, :, 1] is the ptv 1
#       - input[:, :, :, 2] is the ctv 1
#       - input[:, :, :, 3] is the gtv 1
# - output: the available tv mask
def get_biggest_tv(input_data):
    if input_data[:, :, :, 1].any():
        return input_data[:, :, :, 1]
    else:
        if input_data[:, :, :, 2].any():
            return input_data[:, :, :, 2]
        else:
            if input_data[:, :, :, 3].any():
                return input_data[:, :, :, 3]
            else:
                raise ValueError("Input should include at least one of " + \
                    "the following: ptv 1, ctv 1 or gtv 1.")

###############################################################################
## Args
###############################################################################

parser = argparse.ArgumentParser(description='Generate metrics results on groundtruth')

# Arguments
parser.add_argument('-set', '--kind_of_set', type=str, required=True)

# Additional defaults
#parser.set_defaults()
args = parser.parse_args()

# Load dataset
dataset = h5py.File(os.path.join('..', '..', 'data', 
                                 'dataset_resized_rd_summed.h5'), 'r')

# Setup paths
path_to_results = 'metrics_on_gt'
Path(path_to_results).mkdir(parents=True, exist_ok=True)

# Setup CSV
metrics_gt_csv = open(os.path.join(path_to_results, 
                                   'metrics_gt_' + args.kind_of_set + '.csv'), 
                            'w',
                            newline='')
smax_fields = ['Smax ' + str(i) for i in range(1, 21)]
smean_fields = ['Smean ' + str(i) for i in range(1, 21)]
fields = ['ID', 'D99', 'D98', 'D95', 'Dmax'] + smax_fields + \
    smean_fields + ['H1', 'H2']
metrics_gt_writer = csv.DictWriter(metrics_gt_csv, fieldnames=fields)
metrics_gt_writer.writeheader()

# For each seed
for seed_id in range(1, 4):
    
    # Load list_IDs
    list_IDs = list(np.load(os.path.join('list_IDs', 'seed' + str(seed_id), 
                                         args.kind_of_set + '_IDs.npy')))
    
    
    # Debug
    #list_IDs = list_IDs[:2]
    
    # Remove troubling cases (VALIDATION SET)
    if '7017044' in list_IDs: list_IDs.remove('7017044')
    if '668957' in list_IDs: list_IDs.remove('668957')
    if '7021217' in list_IDs: list_IDs.remove('7021217')
    if '5010908' in list_IDs: list_IDs.remove('5010908')
    
    # Init average row
    average_row = {}
    for field_name in fields:
        average_row[field_name] = 0
        
    # Count the number of structures so we can compute the average
    # on the right number of patients (some patients lack segmentations)
    count_struct_m_dose = {}
    for s_field_name in smax_fields + smean_fields:
        count_struct_m_dose[s_field_name] = 0
    
    # For each patient
    for id in list_IDs:
        
        # Setup
        print(id)
        row = {}
        row['ID'] = id
        
        # Fetch the dose plan, prescribed dose and other masks
        plan = unstandardize_rd(dataset[id]['dose'][()])
        tumor_segmentation_bin = get_biggest_tv(dataset[id]['input'])
        prescribed_dose = dataset[id]['prescribed_dose'][()]/100
        tumor_segmentation_gy = tumor_segmentation_bin * prescribed_dose
        
        #######################################################################
        # PTV coverage (DXX)
        #######################################################################        
        # D99
        coverage_value = 99
        row['D99'] = ptv_coverage(plan, tumor_segmentation_gy, coverage_value)
        average_row['D99'] += row['D99']
        
        # D98
        coverage_value = 98
        row['D98'] = ptv_coverage(plan, tumor_segmentation_gy, coverage_value)
        average_row['D98'] += row['D98']

        # D95
        coverage_value = 95
        row['D95'] = ptv_coverage(plan, tumor_segmentation_gy, coverage_value)
        average_row['D95'] += row['D95']    
        
        #######################################################################
        # Dmax (Maximum dose of the plan)
        #######################################################################
        row['Dmax'] = np.max(plan) 
        average_row['Dmax'] += row['Dmax']    
        
        #######################################################################
        # Structure max, mean dose (Dmax, Dmean)
        #######################################################################
        # Go along the masks stored between input[1] and input[20]
        # as 0 is for the CT
        for mask in range(1, dataset[id]['input'].shape[-1]):
            
            # if the mask is not empty
            if dataset[id]['input'][:, :, :, mask].any():
                row['Smax ' + str(mask)], row['Smean ' + str(mask)] = \
                    structure_m_dose(plan, dataset[id]['input'][:, :, :, mask][()])
                average_row['Smax ' + str(mask)] += row['Smax ' + str(mask)]
                count_struct_m_dose['Smax ' + str(mask)] += 1 
                average_row['Smean ' + str(mask)] += row['Smean ' + str(mask)]
                count_struct_m_dose['Smean ' + str(mask)] += 1 
        
        #######################################################################
        # Homogeneity 1 (Homogeneity)
        #######################################################################
        row['H1'] = homogeneity_1(plan, tumor_segmentation_gy)
        average_row['H1'] += row['H1']

        #######################################################################
        # Homogeneity 2 (Dose homogeneity index DHI)
        #######################################################################
        row['H2'] = homogeneity_2(plan, tumor_segmentation_gy)
        average_row['H2'] += row['H2']         
        
        # Write row
        metrics_gt_writer.writerow(row)
        
    ###########################################################################
    # Compute average across patients
    ###########################################################################
    for field in average_row:
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the segmentation
        if field in smax_fields + smean_fields:
            average_row[field] /= max(count_struct_m_dose[field], 1)
        else:
            average_row[field] /= len(list_IDs)
        
    average_row['ID'] = 'Average for seed ' + str(seed_id)
    metrics_gt_writer.writerow(average_row)
    
    average_across_seeds_writer.writerow({})
    average_across_seeds_writer.writerow({})