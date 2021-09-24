###############################################################################
## Imports
###############################################################################
# Math
import numpy as np
from utils.data_standardization import unstandardize_rd, map_intervals 

# DeepL
import keras
from metrics import *
from utils.data_standardization import unstandardize_rd
from unet_model_v2 import mse_closs_encapsulated, mse_dvh_loss_encapsulated, mse_dvh_closs_encapsulated
import tensorflow as tf

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
    
# channel_to_mask
# - goal: get the corresponding mask from a int between 1 and 20 or 1 and 10
# - input: int channel number (between 1 and 20 or 1 and 10)
# - output: str mask name
def channel_to_mask(channel_number, dataset):
    
    if dataset == 'CHUM':
        if (channel_number == 1):
            return "PTV"
        elif (channel_number == 2):
            return "CTV"
        elif (channel_number == 3):
            return "GTV"
        elif (channel_number == 4):
            return "Med. canal"
        elif (channel_number == 5):
            return "Outer med. canal"
        elif (channel_number == 6):
            return "Esophagus"
        elif (channel_number == 7):
            return "Oral cavity"
        elif (channel_number == 8):
            return "Mandible"
        elif (channel_number == 9):
            return "Trachea"
        elif (channel_number == 10):
            return "Trunk"
        elif (channel_number == 11):
            return "Outer trunk"
        elif (channel_number == 12):
            return "Left parotid"
        elif (channel_number == 13):
            return "Right parotid"
        elif (channel_number == 14):
            return "Left inner ear"
        elif (channel_number == 15):
            return "Right inner ear"
        elif (channel_number == 16):
            return "Left eye"
        elif (channel_number == 17):
            return "Right eye"
        elif (channel_number == 18):
            return "Left sub. max."
        elif (channel_number == 19):
            return "Right sub. max."
        elif (channel_number == 20):
            return "Left optic nerve"
        else:
            raise NameError("Unknown channel : %s" % channel_number)
        
    elif dataset == 'OpenKBP':
        if (channel_number == 1):
            return "Brainstem"
        elif (channel_number == 2):
            return "Spinal cord"
        elif (channel_number == 3):
            return "Right parotid"
        elif (channel_number == 4):
            return "Left parotid"
        elif (channel_number == 5):
            return "Esophagus"
        elif (channel_number == 6):
            return "Larynx"
        elif (channel_number == 7):
            return "Mandible"
        elif (channel_number == 8):
            return "PTV 56"
        elif (channel_number == 9):
            return "PTV 63"
        elif (channel_number == 10):
            return "PTV 70"
        else:
            raise NameError("Unknown channel : %s" % channel_number)
    
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP.")
        

###############################################################################
## Args
###############################################################################

parser = argparse.ArgumentParser(description='Generate or evaluate predictions')

# Arguments            
parser.add_argument('-mode', '--test_mode', type=str, required=True,
                    help='Mode for testing the model: center_patch') # TODO
parser.add_argument('-path', '--path_to_model_folder', type=str, 
                    required=True, help='Path to the model folder')
parser.add_argument('-mname', '--model_name', type=str, required=True,
                    help='Name of the model')
parser.add_argument('-dset', '--dataset', type=str, required=True,
                    help='Two kinds of supported dataset: CHUM or OpenKBP')
parser.add_argument('-set', '--kind_of_set', type=str, required=True)
parser.add_argument('-ids', '--list_of_ids', nargs='+', type=str, 
                    required=False, help='List of ids to test')
parser.add_argument('-use_closs', '--use_consistency_losses', action='store_true', 
                    help='Use additional consistency losses')
parser.add_argument('-use_dvh_loss', action='store_true', 
                    help='Use additional DVH losses')
parser.add_argument('-use_dvh_closs', action='store_true', 
                    help='Use additional DVH-CLoss loss')

# Additional defaults
parser.set_defaults(use_ct=False, use_gy=False, use_smaller_intervals=False,
                    use_consistency_losses=False, use_dvh_loss=False, 
                    use_dvh_closs=False)
args = parser.parse_args()

###############################################################################
## Main
###############################################################################

# Create results folder per model
path_to_results = os.path.join(args.path_to_model_folder, 'results_' + \
    args.kind_of_set)
Path(path_to_results).mkdir(parents=True, exist_ok=True)

# CHUM init
if args.dataset == 'CHUM':
    
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
    
# OpenKBP init
elif args.dataset == 'OpenKBP':

    # Load dataset
    if args.kind_of_set == 'train':
        dataset = h5py.File(os.path.join('..', 
                                        '..', 
                                        'shared',
                                        'dataset_training2'), 'r')
    elif args.kind_of_set == 'validation':
        dataset = h5py.File(os.path.join('..', 
                                        '..', 
                                        'shared',
                                        'dataset_validation2'), 'r')
        
    elif args.kind_of_set == 'test':
        dataset = h5py.File(os.path.join('..', 
                                        '..', 
                                        'shared',
                                        'dataset_test2'), 'r')
        
    else:
        raise ValueError("Unknown kind of set. Handled sets are train, validation or test")
    
    # Load ids
    list_IDs = list(dataset.keys())
        
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
                                        custom_objects={'mse_closs': mse_closs_encapsulated(args.dataset)})
    elif args.use_dvh_loss:
        model = keras.models.load_model(os.path.join(args.path_to_model_folder, 
                                                    args.model_name),
                                        custom_objects={'mse_dvh_loss': mse_dvh_loss_encapsulated(tf.zeros((1, 128, 128, 128, 21)), args.dataset)})
    elif args.use_dvh_closs:
        model = keras.models.load_model(os.path.join(args.path_to_model_folder, 
                                                    args.model_name),
                                        custom_objects={'mse_dvh_closs': mse_dvh_closs_encapsulated(tf.zeros((1, 128, 128, 128, 21)), args.dataset)})
    else:
        model = keras.models.load_model(os.path.join(args.path_to_model_folder, 
                                                    args.model_name))
     
    # Patch, prediction and channel dimension
    prediction_dim = (128, 128, None)

    if args.dataset == 'CHUM':
        n_input_channels = 21

    elif args.dataset == 'OpenKBP':
        n_input_channels = 11

    else:
        raise ValueError("Unknown dataset. Handled dataset are CHUM and OpenKBP")
    
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
        input_shape = dataset[id]['dose'].shape
        
        # Prediction
        t0 = time.time()
        prediction = \
            model.predict(np.expand_dims(dataset[id]['input'], axis=0))[0, :, :, :, :]
                    
        # Masking using the body channel - CHUM dataset only
        if args.dataset == 'CHUM':    
            prediction *= np.expand_dims(dataset[id]['body'], axis=-1)

        # Masking using the pdm channel - KBP dataset only
        elif args.dataset == 'OpenKBP':
            prediction *= dataset[id]['pdm'][0, :, :, :, :]
        
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
elif args.test_mode == 'evaluate_predictions_old':
    
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
    
    if args.dataset == 'CHUM':
        n_input_channels = 21
    elif args.dataset == 'OpenKBP':
        n_input_channels = 11
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")

    smax_fields = ['Smax ' + str(i) for i in range(1, n_input_channels)]
    smean_fields = ['Smean ' + str(i) for i in range(1, n_input_channels)]
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
                                                    file))['arr_0'][:, :, :, 0],
                                 args.dataset)

        '''
        plt.imshow(plan[:, :, 30], cmap='jet', vmin=0, vmax=80)
        plt.show()

        sys.exit()
        '''

        tumor_segmentation_bin, prescribed_dose = get_biggest_tv(dataset[id]['input'], args.dataset)
        
        if args.dataset == 'CHUM':
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
    
###############################################################################
###############################################################################
# EVALUATE PREDICTIONS FINAL
###############################################################################
###############################################################################
# - predictions need to be generated prior to running this code
# - generates a csv file with the evaluation results associated with the
# given predictions (path to a folder)
elif args.test_mode == 'evaluate_predictions_final':
    
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
                                         'metrics_pred_final.csv'), 
                            'w',
                            newline='')
    
    if args.dataset == 'CHUM':
        n_input_channels = 21
    elif args.dataset == 'OpenKBP':
        n_input_channels = 11
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")

    dmax_structure_fields = ['Dmax_pe ' + channel_to_mask(i, args.dataset) for i in range(1, n_input_channels)]
    dmean_structure_fields = ['Dmean_pe ' + channel_to_mask(i, args.dataset) for i in range(1, n_input_channels)]
    fields = ['ID'] + \
             ['D99_pe', 'D98_pe', 'D95_pe', 'Dmax_pe', 'HI', 'H2'] + \
             ['D99', 'D98', 'D95', 'Dmax', 'CI', "van't Riet", 'R50'] + \
                 dmax_structure_fields + dmean_structure_fields
    metrics_pred_writer = csv.DictWriter(metrics_pred_csv, fieldnames=fields)
    metrics_pred_writer.writeheader()
    
    # Init average row
    average_row = {}
    for field_name in fields:
        average_row[field_name] = 0
    
    # Count the number of structures so we can compute the average
    # on the right number of patients (some patients lack segmentations)
    count_structures = {}
    for s_field_name in dmax_structure_fields + dmean_structure_fields:
        count_structures[s_field_name] = 0
    
    # Go through the predictions
    list_of_predictions = os.listdir(path_to_predicted_volumes)
    if 'metrics_pred.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred.csv')
    if 'metrics_pred_rf.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_rf.csv')
    if 'metrics_pred_final.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_final.csv')

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
        
        try:
            plan = \
                    unstandardize_rd(np.load(os.path.join(path_to_predicted_volumes, 
                                                        file))['arr_0'][:, :, :, 0],
                                    args.dataset)
        except IndexError:
            plan = \
                    unstandardize_rd(np.load(os.path.join(path_to_predicted_volumes, 
                                                        file))['arr_0'][:, :, :],
                                    args.dataset)
                
        ref_plan = unstandardize_rd(dataset[id]['dose'][()], args.dataset)

        '''
        plt.imshow(plan[:, :, 30], cmap='jet', vmin=0, vmax=80)
        plt.show()

        sys.exit()
        '''

        tumor_segmentation_bin, prescribed_dose = get_biggest_tv(dataset[id]['input'], args.dataset)
        
        if args.dataset == 'CHUM':
            prescribed_dose = dataset[id]['prescribed_dose'][()]/100
        tumor_segmentation_gy = tumor_segmentation_bin * prescribed_dose
        
        #######################################################################
        # PTV coverage (DXX) - Percent error
        #######################################################################        
        # D99
        coverage_value = 99
        row['D99_pe'] = ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose)
        average_row['D99_pe'] += row['D99_pe']
        
        # D98
        coverage_value = 98
        row['D98_pe'] = ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose)
        average_row['D98_pe'] += row['D98_pe']

        # D95
        coverage_value = 95
        row['D95_pe'] = ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose)
        average_row['D95_pe'] += row['D95_pe'] 
        
        #######################################################################
        # Dmax - Percent error
        #######################################################################
        # Dmax_pe
        row['Dmax_pe'] = dmax_absolute_error(prescribed_dose, plan, ref_plan)
        average_row['Dmax_pe'] += row['Dmax_pe']
        
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
        
        # Go along the masks stored between input[1] and input[20 or 10]
        # as 0 is for the CT
        for channel in range(1, dataset[id]['input'].shape[-1]):
            
            # if the mask is not empty
            if dataset[id]['input'][:, :, :, channel].any():
                
                structure_segmentation = dataset[id]['input'][:, :, :, channel][()]
                
                ###############################################################
                # Dmax_pe per structure
                ###############################################################
                row['Dmax_pe ' + channel_to_mask(channel, args.dataset)] = dmax_structure_absolute_error(prescribed_dose, plan, ref_plan, structure_segmentation)
                average_row['Dmax_pe ' + channel_to_mask(channel, args.dataset)] += row['Dmax_pe ' + channel_to_mask(channel, args.dataset)]
                count_structures['Dmax_pe ' + channel_to_mask(channel, args.dataset)] += 1 
                
                ###############################################################
                # Dmean_pe per structure
                ###############################################################
                row['Dmean_pe ' + channel_to_mask(channel, args.dataset)] = dmean_structure_absolute_error(prescribed_dose, plan, ref_plan, structure_segmentation)
                average_row['Dmean_pe ' + channel_to_mask(channel, args.dataset)] += row['Dmean_pe ' + channel_to_mask(channel, args.dataset)]
                count_structures['Dmean_pe ' + channel_to_mask(channel, args.dataset)] += 1 
                       
        
        # Write row
        metrics_pred_writer.writerow(row)   
        
    ###########################################################################
    # Compute average across patients
    ###########################################################################
    for field in average_row:
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the segmentation
        if field in dmax_structure_fields + dmean_structure_fields:
            average_row[field] /= max(count_structures[field], 1)
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
                                         'metrics_pred_last.csv'), 
                            'w',
                            newline='')
    
    if args.dataset == 'CHUM':
        n_input_channels = 21
    elif args.dataset == 'OpenKBP':
        n_input_channels = 11
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")

    dmax_structure_fields = ['Dmax pe ' + channel_to_mask(i, args.dataset) for i in range(1, n_input_channels)]
    dmean_structure_fields = ['Dmean pe ' + channel_to_mask(i, args.dataset) for i in range(1, n_input_channels)]
    fields = ['ID'] + \
             ['D99 pe', 'D98 pe', 'D95 pe', 'Dmax pe', 'HI', 'H2'] + \
             ['D99', 'D98', 'D95', 'Dmax', 'CI', "van't Riet", 'R50'] + \
             ['Dose score', 'DVH score'] + \
             dmax_structure_fields + dmean_structure_fields
    metrics_pred_writer = csv.DictWriter(metrics_pred_csv, fieldnames=fields)
    metrics_pred_writer.writeheader()
    
    # Init average row
    average_row = {}
    for field_name in fields:
        average_row[field_name] = 0
    
    # Count the number of structures so we can compute the average
    # on the right number of patients (some patients lack segmentations)
    count_structures = {}
    for s_field_name in dmax_structure_fields + dmean_structure_fields:
        count_structures[s_field_name] = 0
    
    # Go through the predictions
    list_of_predictions = os.listdir(path_to_predicted_volumes)
    if 'metrics_pred.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred.csv')
    if 'metrics_pred_rf.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_rf.csv')
    if 'metrics_pred_final.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_final.csv')
    if 'metrics_pred_last.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_last.csv')
    if 'metrics_pred_last-save.csv' in list_of_predictions: list_of_predictions.remove('metrics_pred_last-save.csv')

    # Remove troubling cases
    if '7017044.npz' in list_of_predictions: list_of_predictions.remove('7017044.npz')
    if '668957.npz' in list_of_predictions: list_of_predictions.remove('668957.npz')
    if '7021217.npz' in list_of_predictions: list_of_predictions.remove('7021217.npz')
    if '5010908.npz' in list_of_predictions: list_of_predictions.remove('5010908.npz')
    if 'pt_220.npz' in list_of_predictions: list_of_predictions.remove('pt_220.npz')
    
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
        
        try:
            plan = \
                    unstandardize_rd(np.load(os.path.join(path_to_predicted_volumes, 
                                                        file))['arr_0'][:, :, :, 0],
                                    args.dataset)
        # Old formatting
        except IndexError:
            plan = \
                    unstandardize_rd(np.load(os.path.join(path_to_predicted_volumes, 
                                                        file))['arr_0'][:, :, :],
                                    args.dataset)
                
        ref_plan = unstandardize_rd(dataset[id]['dose'][()], args.dataset)

        '''
        plt.imshow(plan[:, :, 30], cmap='jet', vmin=0, vmax=80)
        plt.show()

        sys.exit()
        '''

        tumor_segmentation_bin, prescribed_dose = get_biggest_tv(dataset[id]['input'], args.dataset)
        
        if args.dataset == 'CHUM':
            prescribed_dose = dataset[id]['prescribed_dose'][()]/100
            dose_score_mask = dataset[id]['body'][()]
            tv_channels = dataset[id]['input'][:, :, :, 1:4]
            oar_channels = dataset[id]['input'][:, :, :, 4:20] # excluding the optic nerve, too small for this metric
            voxel_size = (1.,1.,1.)
        elif args.dataset == 'OpenKBP':
            dose_score_mask = dataset[id]['pdm'][()]
            tv_channels = dataset[id]['input'][:, :, :, 8:11]
            oar_channels = dataset[id]['input'][:, :, :, 1:8]
            voxel_size = dataset[id]['vox_dim'][()]
            plan = np.multiply(plan, dose_score_mask[0, :, :, :, 0])
        tumor_segmentation_gy = tumor_segmentation_bin * prescribed_dose
        
        #######################################################################
        # PTV coverage (DXX) - Percent error
        #######################################################################        
        # D99
        coverage_value = 99
        row['D99 pe'] = ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose)
        average_row['D99 pe'] += row['D99 pe']
        
        # D98
        coverage_value = 98
        row['D98 pe'] = ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose)
        average_row['D98 pe'] += row['D98 pe']

        # D95
        coverage_value = 95
        row['D95 pe'] = ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose)
        average_row['D95 pe'] += row['D95 pe'] 
        
        #######################################################################
        # Dmax - Percent error
        #######################################################################
        # Dmax pe
        row['Dmax pe'] = dmax_absolute_error(prescribed_dose, plan, ref_plan)
        average_row['Dmax pe'] += row['Dmax pe']
        
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
        
        ###############################################################################
        # Dose score
        ###############################################################################
        row['Dose score'] = dose_score(plan, ref_plan, dose_score_mask)
        average_row['Dose score'] += row['Dose score']
        
        ###############################################################################
        # DVH score
        ###############################################################################
        row['DVH score'] = dvh_score(plan, ref_plan, oar_channels, tv_channels, voxel_size)
        average_row['DVH score'] += row['DVH score']
            
        
        # Go along the masks stored between input[1] and input[20 or 10]
        # as 0 is for the CT
        for channel in range(1, dataset[id]['input'].shape[-1]):
            
            # if the mask is not empty
            if dataset[id]['input'][:, :, :, channel].any():
                
                structure_segmentation = dataset[id]['input'][:, :, :, channel][()]
                
                ###############################################################
                # Dmax pe per structure
                ###############################################################
                row['Dmax pe ' + channel_to_mask(channel, args.dataset)] = dmax_structure_absolute_error(prescribed_dose, plan, ref_plan, structure_segmentation)
                average_row['Dmax pe ' + channel_to_mask(channel, args.dataset)] += row['Dmax pe ' + channel_to_mask(channel, args.dataset)]
                count_structures['Dmax pe ' + channel_to_mask(channel, args.dataset)] += 1 
                
                ###############################################################
                # Dmean pe per structure
                ###############################################################
                row['Dmean pe ' + channel_to_mask(channel, args.dataset)] = dmean_structure_absolute_error(prescribed_dose, plan, ref_plan, structure_segmentation)
                average_row['Dmean pe ' + channel_to_mask(channel, args.dataset)] += row['Dmean pe ' + channel_to_mask(channel, args.dataset)]
                count_structures['Dmean pe ' + channel_to_mask(channel, args.dataset)] += 1 
        
        # Write row
        metrics_pred_writer.writerow(row)   
        
    ###########################################################################
    # Compute average across patients
    ###########################################################################
    for field in average_row:
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the segmentation
        if field in dmax_structure_fields + dmean_structure_fields:
            average_row[field] /= max(count_structures[field], 1)
        else:
            average_row[field] /= len(list_of_predictions)
        
    average_row['ID'] = 'Average'
    metrics_pred_writer.writerow(average_row)

    ###########################################################################
    # Cleanup
    ###########################################################################  
    metrics_pred_csv.close()