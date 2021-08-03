###############################################################################
## Imports
###############################################################################
# Math
import numpy as np
import math

# DeepL
from sklearn.model_selection import train_test_split
#from data_generator_multi import DataGenerator
from data_gen_v2 import DataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model
import tensorflow as tf
from keras.layers import *
from unet_model_v2 import unet_3D, ablation_unet_3D, ablation_hdunet_3D, mono_branch_unet_3D #, load_pretrained_weights

# IO
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import h5py

###############################################################################
## Limit memory allocation to minimum needed
###############################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

###############################################################################
## Sub-functions
###############################################################################

class CustomSaver(Callback):
    def __init__(self, path):
        self.path = path
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch in [10, 50, 100, 200, 400]:
            self.model.save(os.path.join(self.path, 
                                         "model_{}.h5".format(epoch)))
            

###############################################################################
## Args
###############################################################################

parser = argparse.ArgumentParser(description='Train a given model')

# Arguments
parser.add_argument('-path', '--path_to_main_folder', type=str, required=True,
                    help='Path to the output folder')
parser.add_argument('-ids', '--path_to_ids', type=str, required=True,
                    help='Path to the ids list (data_generation/ids_lists/XXX')
parser.add_argument('-o', '--optim', type=str, required=True,
                    help='Optimizer')
parser.add_argument('-lr', type=float, required=True, help='Learning rate')
parser.add_argument('-drop', '--dropout_value', type=float, required=True,
                    help='Dropout')
parser.add_argument('-e', '--n_epochs', type=int, required=True,
                    help='Number of epochs')
#parser.add_argument('-aug', '--augmentation', dest='augmentation',
#                    action='store_true', help='Use data augmentation')
#parser.add_argument('-no-aug', '--no-augmentation', dest='augmentation',
#                    action='store_false', help="Don't use data augmentation")
parser.add_argument('-seed', type=int, required=True, help='Random seeding')
parser.add_argument('-w', '--initial_weights', type=str,
                    help='Path to the initial weights')
parser.add_argument('-loss', type=str, required=True, 
                    help='Loss (e.g mse, mae, rmse)')
parser.add_argument('-final_activation', type=str, required=True,
                    help='Final activation (e.g sigmoid, linear, tanh')
parser.add_argument('-use_hdunet', action='store_true', 
                    help='Use the HD version of the U-Net')
parser.add_argument('-use_attention', action='store_true', 
                    help='Use the attention version of the U-Net')
parser.add_argument('-use_shared_encoder', action='store_true', 
                    help='Use the shared-encoder version of the U-Net')
parser.add_argument('-use_closs', '--use_consistency_losses', action='store_true', 
                    help='Use additional consistency losses')


# TO REDO, make sure it's differentiable
parser.add_argument('-use_dose_score', action='store_true', 
                    help='Use the dose score as a metric for U-Net')

# Additional defaults
parser.set_defaults(augmentation=False, use_hdunet=False, use_attention=False,
use_shared_encoder=False)
args = parser.parse_args()

## Seeding
from numpy.random import seed
seed(args.seed)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(args.seed)

# Manage folder for generated files
Path(args.path_to_main_folder).mkdir(parents=True, exist_ok=True)

path_to_generated_files = os.path.join(args.path_to_main_folder, 'dr_' + \
        str(args.dropout_value) + '_o_' + args.optim + '_lr_' + str(args.lr) +\
            '_e_' + str(args.n_epochs)) + '_loss_' + args.loss

if args.initial_weights is not None:
    path_to_generated_files += '_transfer'

if args.use_hdunet:
    path_to_generated_files += '_hd'
    
if args.use_attention:
    path_to_generated_files += '_att'

if args.use_shared_encoder:
    path_to_generated_files += '_shared_encoder'

if args.use_dose_score:
    path_to_generated_files += '_dscore'
    
if args.use_consistency_losses:
    path_to_generated_files += '_closs'

Path(path_to_generated_files).mkdir(parents=True, exist_ok=True)

###############################################################################
## Splitting
###############################################################################
# Load IDs
IDs = np.load(args.path_to_ids) # Down to 149 patients

# Debug
#IDs = ['1230200' for i in range(30)]
#print(IDs)
#IDs = IDs[:6]

# Split in train 70%, validation 15%, test 15%
train_IDs, other_IDs = train_test_split(IDs, test_size=0.3)
validation_IDs, test_IDs = train_test_split(other_IDs, test_size=0.5)

# Save for testing
np.save(os.path.join(path_to_generated_files, 'train_IDs'), train_IDs)
np.save(os.path.join(path_to_generated_files, 'validation_IDs'), 
                        validation_IDs)
np.save(os.path.join(path_to_generated_files, "test_IDs"), test_IDs)

###############################################################################
## Parameters
###############################################################################
h5_dataset = h5py.File(os.path.join('..', 
                                    'data',
                                    'dataset_resized_rd_summed.h5'), 'r')

n_input_channels= 21

if args.use_shared_encoder:
    n_output_channels = 3
else:
    n_output_channels = 1
    
n_convolutions = 2 # per block

training_params = {'patch_dim': (128, 128, None),
          'batch_size': 1,
          'dataset': h5_dataset,
          'n_output_channels': n_output_channels,
          'use_shared_encoder': args.use_shared_encoder,
          'shuffle': True,
          'augmentation': args.augmentation}

validation_params = {'patch_dim': (128, 128, None),
          'batch_size': 1,
          'dataset': h5_dataset,
          'n_output_channels': n_output_channels,
          'use_shared_encoder': args.use_shared_encoder,
          'shuffle': False,
          'augmentation': False}

# Generators
training_generator = DataGenerator("train", train_IDs, **training_params)
validation_generator = DataGenerator("validation", validation_IDs,
                                     **validation_params)

# Define model
input_shape = (training_params['patch_dim'][0], 
               training_params['patch_dim'][1],
               training_params['patch_dim'][2], 
               n_input_channels)
'''
model = unet_3D(input_shape=input_shape, 
                number_of_pooling=args.number_of_pooling, 
                dropout=args.dropout_value, 
                optim=args.optim, 
                lr=args.lr)
'''

# Added support for additional architectures
if args.use_hdunet:
    model = ablation_hdunet_3D(input_shape, n_output_channels, args.dropout_value, 
                      n_convolutions, args.optim, args.lr, args.loss,
                      args.final_activation, args.use_attention,
                      use_consistency_losses)
elif args.use_shared_encoder:
    model = branch_unet_3D(input_shape, n_output_channels, args.dropout_value, 
                             n_convolutions, args.optim, args.lr, args.loss,
                             args.final_activation, args.use_attention)
else:
    model = ablation_unet_3D(input_shape, n_output_channels, args.dropout_value, 
                             n_convolutions, args.optim, args.lr, args.loss,
                             args.final_activation, args.use_attention, 
                             args.use_dose_score, args.use_consistency_losses)


# Load pretrained model
if args.initial_weights is not None:
    model = load_model(args.initial_weights)

# Callbacks
mc_validation_loss = ModelCheckpoint(os.path.join(path_to_generated_files, 
                                                  'best_model.h5'), 
                        monitor='val_' + args.loss, mode='min', 
                        save_best_only=True, verbose=1)

mc_epoch = CustomSaver(path_to_generated_files)

callbacks = [mc_epoch, mc_validation_loss]

###############################################
## Training
###############################################
history = model.fit_generator(generator=training_generator, 
                              validation_data=validation_generator,
                              epochs=args.n_epochs,
                              callbacks=callbacks,
                              max_queue_size=16,
                              workers=8)
model.save(os.path.join(path_to_generated_files, "model.h5"))

###############################################
## Results
###############################################

# Get training and validation accuracy histories
training_accuracy = history.history[args.loss]
np.save(os.path.join(path_to_generated_files, 'training_' + args.loss), 
                        training_accuracy)
validation_accuracy = history.history['val_' + args.loss]
np.save(os.path.join(path_to_generated_files, 'validation_' + args.loss), 
                        validation_accuracy)

# Create count of the number of epochs
epoch_count = range(1, len(training_accuracy) + 1)

# Compute min validation value
min_value = min(validation_accuracy)
min_index = list(validation_accuracy).index(min_value)

# Plot
plt.plot(epoch_count, training_accuracy, c='tab:blue', ls='-')
plt.plot(epoch_count, validation_accuracy, c='tab:blue', ls='--')
plt.title('Average Training Loss (solid) and Validation Loss (dotted)')
plt.scatter([min_index], [min_value], c='tab:red', ls=':')

ax = plt.gca()
ax.set_ylim(0, 0.2)

formatted_min_value = ("{:.4f}".format(min_value))
plt.legend(['Training ' + args.loss.upper(),
            'Validation ' + args.loss.upper(),
            'Min ' + args.loss.upper() + ' value: ' + formatted_min_value])
plt.xlabel('Epoch')
plt.ylabel(args.loss.upper())

# Save
plt.savefig(os.path.join(path_to_generated_files, args.loss + '.png'))
