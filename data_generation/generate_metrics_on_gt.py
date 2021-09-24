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
#       = CHUM
#       - input[:, :, :, 1] is the ptv 1
#       - input[:, :, :, 2] is the ctv 1
#       - input[:, :, :, 3] is the gtv 1
#       = OpenKBP
#       - input[:, :, :, 10] is the ptv70
#       - input[:, :, :, 9] is the ptv63
#       - input[:, :, :, 8] is the ptv56
# - output: the available tv mask
def get_biggest_tv(input_data, dataset):
    
    if dataset == 'CHUM':
        if input_data[:, :, :, 1].any():
            return input_data[:, :, :, 1], 1
        else:
            if input_data[:, :, :, 2].any():
                return input_data[:, :, :, 2], 1
            else:
                if input_data[:, :, :, 3].any():
                    return input_data[:, :, :, 3], 1
                else:
                    raise ValueError("Input should include at least one of " + \
                        "the following: ptv 1, ctv 1 or gtv 1.")
    elif dataset == 'OpenKBP':
        if input_data[:, :, :, 10].any():
            return input_data[:, :, :, 10], 70
        else:
            if input_data[:, :, :, 9].any():
                return input_data[:, :, :, 9], 63
            else:
                if input_data[:, :, :, 8].any():
                    return input_data[:, :, :, 8], 56
                else:
                    raise ValueError("Input should include at least one of " + \
                        "the following: ptv70, ptv63 or ptv56.")
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")

###############################################################################
## Args
###############################################################################

parser = argparse.ArgumentParser(description='Generate metrics results on groundtruth')

# Arguments
parser.add_argument('-set', '--kind_of_set', type=str, required=True)
parser.add_argument('-dset', '--dataset', type=str, required=True)

# Additional defaults
#parser.set_defaults()
args = parser.parse_args()

# Load dataset and other setups
if args.dataset == 'CHUM':
    dataset = h5py.File(os.path.join('..', '..', 'data', 
                                    'dataset_resized_rd_summed.h5'), 'r')
    seed_range = 4
elif args.dataset == 'OpenKBP':
    if args.kind_of_set == 'validation':
        dataset = h5py.File(os.path.join('..', 
                                        '..', 
                                        '..', 
                                        'shared',
                                        'dataset_validation2'), 'r')
    elif args.kind_of_set == 'test':
        dataset = h5py.File(os.path.join('..', 
                                        '..',
                                        '..', 
                                        'shared',
                                        'dataset_test2'), 'r')
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
        
    seed_range = 2

# Setup paths
path_to_results = 'metrics_on_gt'
Path(path_to_results).mkdir(parents=True, exist_ok=True)

# Setup CSV
metrics_gt_csv = open(os.path.join(path_to_results, 
                                   'metrics_gt_' + args.kind_of_set + '_' + \
                                       args.dataset + '.csv'), 
                            'w',
                            newline='')
fields = ['ID', 'HI', 'H2', 'D99', 'D98', 'D95', 'Dmax', 'CI', "van't Riet", 'R50']
metrics_gt_writer = csv.DictWriter(metrics_gt_csv, fieldnames=fields)
metrics_gt_writer.writeheader()

# For each seed
for seed_id in range(1, seed_range):
    
    # Load list_IDs
    if args.dataset == 'CHUM':
        list_IDs = list(np.load(os.path.join('list_IDs', 'seed' + str(seed_id), 
                                         args.kind_of_set + '_IDs.npy')))
    elif args.dataset == 'OpenKBP':
        list_IDs = list(dataset.keys())
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
    
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
    
    # For each patient
    for id in list_IDs:
        
        # Setup
        print(id)
        row = {}
        row['ID'] = id
        
        # Fetch the dose plan, prescribed dose and other masks
        plan = unstandardize_rd(dataset[id]['dose'][()], args.dataset) 
        tumor_segmentation_bin, prescribed_dose = get_biggest_tv(dataset[id]['input'], args.dataset)
        
        if args.dataset == 'CHUM':
            prescribed_dose = dataset[id]['prescribed_dose'][()]/100
            plan *= dataset[id]['body']
        elif args.dataset == 'OpenKBP':
            dose_score_mask = dataset[id]['pdm'][()]
            plan *= dose_score_mask[0, :, :, :, 0]
        else:
            raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
            
        tumor_segmentation_gy = tumor_segmentation_bin * prescribed_dose
        
        #######################################################################
        # Homogeneity 1 (Homogeneity)
        #######################################################################
        row['HI'] = homogeneity_1(plan, tumor_segmentation_gy)
        average_row['HI'] += row['HI']

        #######################################################################
        # Homogeneity 2 (Dose homogeneity index DHI)
        #######################################################################
        row['H2'] = homogeneity_2(plan, tumor_segmentation_gy)
        average_row['H2'] += row['H2']  
        
        #######################################################################
        # PTV coverage (DXX)
        #######################################################################        
        # D99
        coverage_value = 99
        row['D99'] = ptv_coverage(plan, tumor_segmentation_gy, coverage_value, prescribed_dose)
        average_row['D99'] += row['D99']
        
        # D98
        coverage_value = 98
        row['D98'] = ptv_coverage(plan, tumor_segmentation_gy, coverage_value, prescribed_dose)
        average_row['D98'] += row['D98']

        # D95
        coverage_value = 95
        row['D95'] = ptv_coverage(plan, tumor_segmentation_gy, coverage_value, prescribed_dose)
        average_row['D95'] += row['D95']  
        
        #######################################################################
        # Dmax (whole plan)
        #######################################################################
        row['Dmax'] = max_dose_error_vs_prescribed(prescribed_dose, plan)
        average_row['Dmax'] += row['Dmax'] 
        
        #######################################################################
        # Conformity index CI
        #######################################################################
        row['CI'] = conformity_index(plan, tumor_segmentation_gy, prescribed_dose)
        average_row['CI'] += row['CI']
        
        ###############################################################################
        # van't Riet conformation number
        ###############################################################################
        row["van't Riet"] = vant_riet(plan, tumor_segmentation_gy, prescribed_dose)
        average_row["van't Riet"] += row["van't Riet"]
        
        ###############################################################################
        # Dose spillage (R50)
        ###############################################################################
        row['R50'] = dose_spillage(plan, tumor_segmentation_bin, prescribed_dose)
        average_row['R50'] += row['R50']
        
        # Write row
        metrics_gt_writer.writerow(row)
        
    ###########################################################################
    # Compute average across patients
    ###########################################################################
    for field in average_row:
        average_row[field] /= len(list_IDs)
        
    average_row['ID'] = 'Average for seed ' + str(seed_id)
    metrics_gt_writer.writerow(average_row)
    
    metrics_gt_writer.writerow({})
    metrics_gt_writer.writerow({})