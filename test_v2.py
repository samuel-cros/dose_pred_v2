###############################################################################
## Imports
###############################################################################
# Math
import numpy as np
from utils.data_standardization import unstandardize_rd, map_intervals 
from utils.data_standardization import rd_min_value, rd_max_value

# DeepL
import keras
from metrics import *
from utils.data_standardization import unstandardize_rd
from unet_model_v2 import mse_closs

# IO
import argparse
import h5py
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import csv

###############################################################################
## Limit memory allocation to minimum needed
###############################################################################
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

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

parser = argparse.ArgumentParser(description='Generate or evaluate a prediction')

# Arguments            
parser.add_argument('-mode', '--test_mode', type=str, required=True,
                    help='Mode for testing the model: center_patch') # TODO
parser.add_argument('-path', '--path_to_model_folder', type=str, 
                    required=True, help='Path to the model folder')
parser.add_argument('-mname', '--model_name', type=str, required=True,
                    help='Name of the model')
parser.add_argument('-set', '--kind_of_set', type=str, required=True)
parser.add_argument('-ids', '--list_of_ids', nargs='+', type=str, 
                    required=False, help='List of ids to test')
parser.add_argument('-use_closs', '--use_consistency_losses', action='store_true', 
                    help='Use additional consistency losses')

# Additional defaults
parser.set_defaults(use_ct=False, use_gy=False, use_smaller_intervals=False,
                    use_consistency_losses=False)
args = parser.parse_args()

###############################################################################
## Main
###############################################################################

# Create results folder per model
path_to_results = os.path.join(args.path_to_model_folder, 'results_' + \
    args.kind_of_set)
Path(path_to_results).mkdir(parents=True, exist_ok=True)
    
# Load ids
list_IDs = args.list_of_ids \
            if (args.kind_of_set == "manual") \
            else (np.load(os.path.join(args.path_to_model_folder, 
                args.kind_of_set + "_IDs.npy")))
list_IDs = list(list_IDs) 

## Debug
#list_IDs = ['668957'], ['5447536']
#list_IDs = list_IDs[:2]

# Load dataset
dataset = h5py.File(os.path.join('..', 
                                 'data', 
                                 'dataset_resized_rd_summed.h5'), 'r')
        
###############################################################################      
###############################################################################
# GENERATE PREDICTIONS
###############################################################################
###############################################################################
# - predict doses
# - save them
if args.test_mode == 'generate_predictions':
    
    ###########################################################################
    # Setup
    ########################################################################### 
    # Load model
    
    if args.use_consistency_losses:
        model = keras.models.load_model(os.path.join(args.path_to_model_folder, 
                                                    args.model_name),
                                        custom_objects={'mse_closs': mse_closs})
    else:
        model = keras.models.load_model(os.path.join(args.path_to_model_folder, 
                                                    args.model_name))
     
    # Patch, prediction and channel dimension
    prediction_dim = (128, 128, None)
    n_input_channels = 21
    
    # Specific medical volumes
    tumor_volumes = ['ptv 1', 'ctv 1', 'gtv 1']
    
    # Setup path for results
    path_to_predicted_volumes = os.path.join(path_to_results, 
                                             'predicted_volumes_' + \
                                                 args.test_mode,)
    Path(path_to_predicted_volumes).mkdir(parents=True, exist_ok=True)
    
    ###########################################################################
    # Main
    ###########################################################################
    # For each id
    for id in list_IDs:
        
        # Grab input shape
        input_shape = dataset[id]['body'].shape
        
        # Init        
        body = dataset[id]['body']
        
        # Prediction
        t0 = time.time()
        prediction = \
            model.predict(np.expand_dims(dataset[id]['input'], axis=0))[0, :, :, :, :]
                    
        # Masking using the body channel
        prediction *= np.expand_dims(body, axis=-1)
        
        print("Time spent predicting:", time.time() - t0)
        
        # Save compressed
        np.savez_compressed(os.path.join(path_to_predicted_volumes, id), 
                            prediction)
        
        # Debug
        #sys.exit()
        
###############################################################################
###############################################################################
# EVALUATE PREDICTIONS
###############################################################################
###############################################################################
# - predictions need to be generated prior to running this code
# - generates a csv file with the evaluation results associated with the
# given predictions (path to a folder)
elif args.test_mode == 'evaluate_predictions':
    
    ###########################################################################
    # Setup
    ###########################################################################
    # Setup paths
    path_to_predicted_volumes = \
        os.path.join(path_to_results, 
                     'predicted_volumes_generate_predictions',)
    Path(path_to_predicted_volumes).mkdir(parents=True, exist_ok=True)
    
    # Setup CSV
    metrics_pred_csv = open(os.path.join(path_to_predicted_volumes, 
                                         'metrics_pred.csv'), 
                            'w',
                            newline='')

    fields = ['ID', 'PTV coverage', 'Structure max dose', 'Structure mean dose', 
            'Homogeneity', "van't Riet conformation number", 'Dose spillage', 
            'Conformity index', "Paddick's conformity index", 'Gradient index']
    metrics_pred_writer = csv.DictWriter(metrics_pred_csv, fieldnames=fields)
    metrics_pred_writer.writeheader()
    average_row = {}
    average_row['PTV coverage'] = {}
    average_row['Structure max dose'] = {}
    average_row['Structure mean dose'] = {}
    average_row['Homogeneity'] = {}
    
    # Count the number of structures so we can compute the average
    # on the right number of patients
    count_struct_m_dose = {}
    
    # Go through the predictions
    list_of_predictions = os.listdir(path_to_predicted_volumes)
    if 'metrics_pred.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred.csv')
    if 'metrics_pred_rf.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_rf.csv')

    # Remove troubling cases
    if '7017044.npz' in list_of_predictions: list_of_predictions.remove('7017044.npz')
    if '668957.npz' in list_of_predictions: list_of_predictions.remove('668957.npz')
    if '7021217.npz' in list_of_predictions: list_of_predictions.remove('7021217.npz')
    if '5010908.npz' in list_of_predictions: list_of_predictions.remove('5010908.npz')
    #list_IDs.remove('5447536')
    #list_of_predictions = ['978722.npz']
    
    ###########################################################################
    # Main
    ###########################################################################
    for file in list_of_predictions:
    
        # Compute the metrics
        
        # Setup
        id = file.split('.npz')[0]
        print(id)
        row = {}
        row['ID'] = id
    
        plan = \
            unstandardize_rd(np.load(os.path.join(path_to_predicted_volumes, 
                                                file))['arr_0'][:, :, :, 0])

        '''
        plt.imshow(plan[:, :, 30], cmap='jet', vmin=0, vmax=80)
        plt.show()

        sys.exit()
        '''

        tumor_segmentation_bin = get_biggest_tv(dataset[id]['input'])
        prescribed_dose = dataset[id]['prescribed_dose'][()]/100
        tumor_segmentation_gy = tumor_segmentation_bin * prescribed_dose
            
        #######################################################################
        # PTV coverage (DXX)
        #######################################################################
        row['PTV coverage'] = {}
        
        # D99
        coverage_value = 99
        row['PTV coverage']['D99'] = ptv_coverage(plan, 
                                                tumor_segmentation_gy, 
                                                coverage_value)
        #print('D99', row['PTV coverage']['D99'])
        try:
            average_row['PTV coverage']['D99'] += row['PTV coverage']['D99']
        # Manage the first time adding the key in the dict
        except KeyError:
            average_row['PTV coverage']['D99'] = row['PTV coverage']['D99']
        # D98
        coverage_value = 98
        row['PTV coverage']['D98'] = ptv_coverage(plan, 
                                                tumor_segmentation_gy, 
                                                coverage_value)
        #print('D98', row['PTV coverage']['D98'])
        try:
            average_row['PTV coverage']['D98'] += row['PTV coverage']['D98']
        except KeyError:
            average_row['PTV coverage']['D98'] = row['PTV coverage']['D98']

        # D95
        coverage_value = 95
        row['PTV coverage']['D95'] = ptv_coverage(plan, 
                                                tumor_segmentation_gy, 
                                                coverage_value)
        #print('D95', row['PTV coverage']['D95'])
        try:
            average_row['PTV coverage']['D95'] += row['PTV coverage']['D95']
        except KeyError:
            average_row['PTV coverage']['D95'] = row['PTV coverage']['D95']

        # D50
        coverage_value = 50
        row['PTV coverage']['D50'] = ptv_coverage(plan, 
                                                tumor_segmentation_gy, 
                                                coverage_value)
        #print('D50', row['PTV coverage']['D50'])
        try:
            average_row['PTV coverage']['D50'] += row['PTV coverage']['D50']
        except KeyError:
            average_row['PTV coverage']['D50'] = row['PTV coverage']['D50']
        
        # D2
        coverage_value = 2
        row['PTV coverage']['D2'] = ptv_coverage(plan, 
                                                tumor_segmentation_gy, 
                                                coverage_value)
        #print('D2', row['PTV coverage']['D2'])
        try:
            average_row['PTV coverage']['D2'] += row['PTV coverage']['D2']
        except KeyError:
            average_row['PTV coverage']['D2'] = row['PTV coverage']['D2']
        
        
        #######################################################################
        # Structure max, mean dose (Dmax, Dmean)
        #######################################################################
        
        row['Structure max dose'], row['Structure mean dose'] = {}, {}
        
        # Dmax
        # Go along the masks stored between input[1] and input[20]
        # as 0 is for the CT
        for mask in range(1, dataset[id]['input'].shape[-1]):
            
            # if the mask is not empty
            if dataset[id]['input'][:, :, :, mask].any():
                row['Structure max dose'][mask], row['Structure mean dose'][mask] = \
                    structure_m_dose(plan, dataset[id]['input'][:, :, :, mask][()])
                
                try:
                    average_row['Structure max dose'][mask] += \
                        row['Structure max dose'][mask] 
                    average_row['Structure mean dose'][mask] += \
                        row['Structure mean dose'][mask]
                    count_struct_m_dose[mask] += 1
                except KeyError:
                    average_row['Structure max dose'][mask] = \
                        row['Structure max dose'][mask] 
                    average_row['Structure mean dose'][mask] = \
                        row['Structure mean dose'][mask]
                    count_struct_m_dose[mask] = 1
                
                    
        #######################################################################
        # Homogeneity 1 (Homogeneity)
        #######################################################################
        row['Homogeneity'] = {}
        
        row['Homogeneity']['H1'] = homogeneity_1(plan, tumor_segmentation_gy)
        #print('H1', row['Homogeneity']['H1'])
        try:
            average_row['Homogeneity']['H1'] += row['Homogeneity']['H1']
        except KeyError:
            average_row['Homogeneity']['H1'] = row['Homogeneity']['H1']

        #######################################################################
        # Homogeneity 2 (Dose homogeneity index DHI)
        #######################################################################
        
        row['Homogeneity']['H2'] = homogeneity_2(plan, tumor_segmentation_gy)
        #print('H2', row['Homogeneity']['H2'])
        try:
            average_row['Homogeneity']['H2'] += row['Homogeneity']['H2']
        except KeyError:
            average_row['Homogeneity']['H2'] = row['Homogeneity']['H2']

        #######################################################################
        # Homogeneity 3 (Homogeneity index HI)
        #######################################################################
        
        row['Homogeneity']['H3'] = homogeneity_3(plan, 
                        tumor_segmentation_gy, 
                        prescribed_dose) 
        # It's the max in our case only since we are using the same prescribed dose
        # over the whole tumor
        
        #print('H3', row['Homogeneity']['H3'])
        try:
            average_row['Homogeneity']['H3'] += row['Homogeneity']['H3']
        except KeyError:
            average_row['Homogeneity']['H3'] = row['Homogeneity']['H3']
            
        #######################################################################
        # van't Riet conformation number
        #######################################################################
        
        row["van't Riet conformation number"] = vant_riet(plan, 
                            tumor_segmentation_gy, 
                            prescribed_dose) 
        # It's the max in our case only since we are using the same prescribed dose
        # over the whole tumor
        
        #print("van't Riet", row["van't Riet conformation number"])
        try:
            average_row["van't Riet conformation number"] += \
                row["van't Riet conformation number"]
        except KeyError:
            average_row["van't Riet conformation number"] = \
                row["van't Riet conformation number"]
                
                
        #######################################################################
        # Dose spillage
        #######################################################################
        
        row['Dose spillage'] = dose_spillage(plan, 
                            tumor_segmentation_gy, 
                            prescribed_dose) 
        # It's the max in our case only since we are using the same prescribed dose
        # over the whole tumor
        
        #print("Dose spillage", row['Dose spillage'])
        try:
            average_row['Dose spillage'] += row['Dose spillage']  
        except KeyError:
            average_row['Dose spillage'] = row['Dose spillage'] 
            
        #######################################################################
        # Conformity index CI
        #######################################################################
        
        row['Conformity index'] = conformity_index(plan, 
                            tumor_segmentation_gy, 
                            prescribed_dose) 
        # It's the max in our case only since we are using the same prescribed dose
        # over the whole tumor
        
        #print("CI", row['Conformity index'])
        try:
            average_row['Conformity index'] += row['Conformity index']  
        except KeyError:
            average_row['Conformity index'] = row['Conformity index'] 
            
        #######################################################################
        # Paddick's conformity index PCI
        #######################################################################
        
        row["Paddick's conformity index"] = conformity_index(plan, 
                            tumor_segmentation_gy, 
                            prescribed_dose) 
        # It's the max in our case only since we are using the same prescribed dose
        # over the whole tumor
        
        #print("PCI", row["Paddick's conformity index"])
        try:
            average_row["Paddick's conformity index"] \
                += row["Paddick's conformity index"]
        except KeyError:
            average_row["Paddick's conformity index"] \
                = row["Paddick's conformity index"]
                
        #######################################################################
        # Gradient index GI
        #######################################################################
        
        row['Gradient index'] = gradient_index(plan, prescribed_dose)
        
        # It's the max in our case only since we are using the same prescribed dose
        # over the whole tumor

        #print("GI", row['Gradient index'])
        try:
            average_row['Gradient index'] += row['Gradient index']
        except KeyError:
            average_row['Gradient index'] = row['Gradient index']
            
        #######################################################################
        # Dose volume histogram
        #######################################################################
            
        #
        metrics_pred_writer.writerow(row)
            
        
    ###########################################################################
    # Compute average across patients
    ###########################################################################
    for field in average_row:
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the structure
        if field == "Structure max dose" or field == "Structure mean dose":
            for subfield in average_row[field]:
                average_row[field][subfield] /= count_struct_m_dose[subfield]
        elif isinstance(average_row[field], dict):
            for subfield in average_row[field]:
                average_row[field][subfield] /= len(list_of_predictions)
        else:
            average_row[field] /= len(list_of_predictions)
        
    average_row['ID'] = 'Average'
    metrics_pred_writer.writerow(average_row)

    ###########################################################################
    # Cleanup
    ###########################################################################  
    metrics_pred_csv.close()

###############################################################################
###############################################################################
# EVALUATE PREDICTIONS RF (Refactored)
###############################################################################
###############################################################################
# - predictions need to be generated prior to running this code
# - generates a csv file with the evaluation results associated with the
# given predictions (path to a folder)
elif args.test_mode == 'evaluate_predictions_rf':
    
    ###########################################################################
    # Setup
    ###########################################################################
    # Setup paths
    path_to_predicted_volumes = \
        os.path.join(path_to_results, 
                     'predicted_volumes_generate_predictions',)
    Path(path_to_predicted_volumes).mkdir(parents=True, exist_ok=True)
    
    # Setup CSV
    metrics_pred_csv = open(os.path.join(path_to_predicted_volumes, 
                                         'metrics_pred_rf.csv'), 
                            'w',
                            newline='')

    smax_fields = ['Smax ' + str(i) for i in range(1, 21)]
    smean_fields = ['Smean ' + str(i) for i in range(1, 21)]
    fields = ['ID', 'D99', 'D98', 'D95', 'Dmax'] + smax_fields + \
        smean_fields + ['H1', 'H2']
    metrics_pred_writer = csv.DictWriter(metrics_pred_csv, fieldnames=fields)
    metrics_pred_writer.writeheader()
    
    # Init average row
    average_row = {}
    for field_name in fields:
        average_row[field_name] = 0
    
    # Count the number of structures so we can compute the average
    # on the right number of patients (some patients lack segmentations)
    count_struct_m_dose = {}
    for s_field_name in smax_fields + smean_fields:
        count_struct_m_dose[s_field_name] = 0
    
    # Go through the predictions
    list_of_predictions = os.listdir(path_to_predicted_volumes)
    if 'metrics_pred.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred.csv')
    if 'metrics_pred_rf.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_rf.csv')

    # Remove troubling cases
    if '7017044.npz' in list_of_predictions: list_of_predictions.remove('7017044.npz')
    if '668957.npz' in list_of_predictions: list_of_predictions.remove('668957.npz')
    if '7021217.npz' in list_of_predictions: list_of_predictions.remove('7021217.npz')
    if '5010908.npz' in list_of_predictions: list_of_predictions.remove('5010908.npz')
    
    #list_IDs.remove('5447536')
    #list_of_predictions = ['2170598.npz', '171272.npz']
    
    ###########################################################################
    # Main
    ###########################################################################
    for file in list_of_predictions:
    
        # Compute the metrics
        
        # Setup
        id = file.split('.npz')[0]
        print(id)
        row = {}
        row['ID'] = id
        
        plan = \
                unstandardize_rd(np.load(os.path.join(path_to_predicted_volumes, 
                                                    file))['arr_0'][:, :, :, 0])

        '''
        plt.imshow(plan[:, :, 30], cmap='jet', vmin=0, vmax=80)
        plt.show()

        sys.exit()
        '''

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
        metrics_pred_writer.writerow(row)   
        
    ###########################################################################
    # Compute average across patients
    ###########################################################################
    for field in average_row:
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the segmentation
        if field in smax_fields + smean_fields:
            average_row[field] /= max(count_struct_m_dose[field], 1)
        else:
            average_row[field] /= len(list_of_predictions)
        
    average_row['ID'] = 'Average'
    metrics_pred_writer.writerow(average_row)

    ###########################################################################
    # Cleanup
    ###########################################################################  
    metrics_pred_csv.close()