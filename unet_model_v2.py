## Imports
# DeepL
from keras.layers import Conv3D, BatchNormalization, Activation, Input, \
    MaxPooling3D, UpSampling3D, Conv3DTranspose
from keras.layers import add, multiply, concatenate
from keras import Model
from keras.optimizers import Adam, RMSprop
from metrics import *
from keras import backend as K
from scipy.ndimage import sobel, generic_gradient_magnitude
from tensorflow.losses import mean_squared_error
import tensorflow as tf

from utils.data_standardization import standardize_rd_tensor, unstandardize_rd_tensor

###############################################################################
## Parameters
###############################################################################

kernel_value = 'he_normal'
batch_norm = True
inner_activation = 'relu' # leakyrelu
#final_activation = 'linear' # sigmoid, tanh, linear

###############################################################################
## Subfunctions 
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
    
## Consistency losses
# > normalization / denormalization issue
#   - isolines: done, to test
#   - edges: not mandatory with mono branch
# > 3D Sobel
#   - done
# > link with output, y_true or y_true[0] / y_true[1] / y_true[2] ?
#   - computed on both pred and true at run time

# Isodose consistency loss
@tf.function
def isodose_consistency_loss(dataset, y_true, y_pred):
    
    # Define isolines values
    d_i_values = standardize_rd_tensor(tf.round(unstandardize_rd_tensor(y_true, dataset)), dataset)
    
    # # if dose_i > d_ip1, return || dose_i - d_ip1 ||_2
    # if dose_i < d_i, return || dose_i - d_i ||_2
    res_1 = tf.where(y_pred < d_i_values, tf.abs(y_pred - d_i_values), tf.abs(y_pred - (d_i_values + standardize_rd_tensor(1.0, dataset))))
    
    # if d_i < dose_i < d_ip1, return 0
    res_2 = tf.where(tf.math.logical_and(y_pred >= d_i_values, y_pred <= (d_i_values + standardize_rd_tensor(1.0, dataset))), tf.zeros(tf.shape(res_1)), res_1)
    
    # MSE on res_2
    return tf.math.reduce_sum(res_2 ** 2) / float(tf.math.reduce_prod(tf.shape(res_2)))
    

# Texture consistency loss
@tf.function
def texture_consistency_loss(y_true, y_pred):
    
    # Define sobel filters    
    sobel_x = tf.constant([[[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],
                           [[-2,  0,  2], [-4,  0,  4], [-2,  0,  2]],
                           [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]]], tf.float32)
    sobel_x_reshaped = tf.reshape(sobel_x, [3, 3, 3, 1, 1])
    
    sobel_y = tf.constant([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                           [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
                           [[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]]], tf.float32)
    sobel_y_reshaped = tf.reshape(sobel_y, [3, 3, 3, 1, 1])
    
    sobel_z = tf.constant([[[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
                           [[-2, -4, -2], [ 0,  0,  0], [ 2,  4,  2]],
                           [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]], tf.float32)
    sobel_z_reshaped = tf.reshape(sobel_z, [3, 3, 3, 1, 1])
    
    # Compute on ref
    y_true_filters_x = tf.nn.conv3d(input = y_true, 
                             filters = sobel_x_reshaped,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME')
    
    y_true_filters_y = tf.nn.conv3d(input = y_true, 
                             filters = sobel_y_reshaped,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME')
    
    y_true_filters_z = tf.nn.conv3d(input = y_true, 
                             filters = sobel_z_reshaped,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME')
    
    y_true_filters = tf.stack([y_true_filters_x, y_true_filters_y, y_true_filters_z])
    
    # Compute on pred
    y_pred_filters_x = tf.nn.conv3d(input = y_pred, 
                             filters = sobel_x_reshaped,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME')
    
    y_pred_filters_y = tf.nn.conv3d(input = y_pred, 
                             filters = sobel_y_reshaped,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME')
    
    y_pred_filters_z = tf.nn.conv3d(input = y_pred, 
                             filters = sobel_z_reshaped,
                             strides = [1, 1, 1, 1, 1],
                             padding = 'SAME')
    
    y_pred_filters = tf.stack([y_pred_filters_x, y_pred_filters_y, y_pred_filters_z])
    
    # MSE on ref, pred
    return mean_squared_error(y_pred_filters, y_true_filters)
    
# MSE + Consistency losses encapsulator
def mse_closs_encapsulated(dataset):
    
    '''
    # MSE + Consistency losses (Both2)
    def mse_closs(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + \
                    0.1* isodose_consistency_loss(dataset, y_true, y_pred) + \
                        0.01* texture_consistency_loss(y_true, y_pred)
    '''
                        
    # MSE + Edges only (Edges3)
    def mse_closs(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + \
                        0.001* texture_consistency_loss(y_true, y_pred)
        
    return mse_closs


# --------------------------------------------------------------------------- #

## DVH loss
@tf.function
def dvh_loss(input_data, dataset, y_true, y_pred):
    
    # Init
    # Simplified shape
    #x_y_z_shape = [y_pred.get_shape()[1].value, y_pred.get_shape()[2].value, y_pred.get_shape()[3].value]
    
    # For open-kbp and CHUM, the '0' channel is always a CT-scan
    structures = input_data[0, :, :, :, 1:]
    
    # Max dose in Gy for the interval
    max_dose = 80
    
    # Bin width, number of elements in the interval
    # lower = more accurate, higher = faster
    step = 1 # resp 10 for CHUM, inf for OpenKBP
            
    # Steepness parameter
    # m = 1, local minima might be less defined, but gradients behave better
    # for our optimization problem
    m = 1
    
    #
    result = 0.
    
    # Reshape and unstandardize
    y_pred_unstandardized = unstandardize_rd_tensor(y_pred[0, :, :, :, 0], dataset)
    y_true_unstandardized = unstandardize_rd_tensor(y_true[0, :, :, :, 0], dataset)
    
    # TODO, make it less heavy computation-wise
    # result 128*128*h*s
    # result_s = tf.expand_dims(sigmoid((y - dt)), -1) * structures_s
    # column_y = tf.multiply(structures, tf.expand_dims(tf.math.sigmoid((y_pred_unstandardized - dt)), -1))
    # dvh_loss_final += tf.reduce_sum(column_y - column_y') / 
    
    sum_structures = tf.math.reduce_sum(structures, axis = [0, 1, 2])
        
    # For each d_value dt in [0, step, ..., max_dose]
    for dt in range(0, max_dose+1, step):
        
        # Compute
        # result_s = tf.expand_dims(sigmoid((y - dt)), -1) * structures_s
        vs_dt_matrix_pred = tf.math.reduce_sum(tf.expand_dims(tf.math.sigmoid((m/step) * (y_pred_unstandardized - dt)), -1) * structures, axis = [0, 1, 2]) / tf.where(tf.equal(sum_structures, 0), tf.ones(tf.shape(sum_structures)), sum_structures)
        vs_dt_matrix_true = tf.math.reduce_sum(tf.expand_dims(tf.math.sigmoid((m/step) * (y_true_unstandardized - dt)), -1) * structures, axis = [0, 1, 2]) / tf.where(tf.equal(sum_structures, 0), tf.ones(tf.shape(sum_structures)), sum_structures)
        
        # Compute
        # Sum_s(Sum_t((vs_dt_true - vs_dt_pred)**2))
        # or Sum_s(||DVH_s_true - DVH_s_pred||**2)
        result += tf.math.reduce_sum((vs_dt_matrix_true - vs_dt_matrix_pred)**2)
            

    # Return
    # [result / (n_s * n_t)]
    return (result / (structures.get_shape()[-1].value * ((max_dose + 1) / step)))


# DVH Loss encapsulator
def mse_dvh_loss_encapsulated(input_data, dataset):
    
    def mse_dvh_loss(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + \
                0.1 * dvh_loss(input_data, dataset, y_true, y_pred)
    
    return mse_dvh_loss

# --------------------------------------------------------------------------- #

## DVH + C-Loss encapsulator
def mse_dvh_closs_encapsulated(input_data, dataset):
    
    def mse_dvh_closs(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + \
                0.1 * dvh_loss(input_data, dataset, y_true, y_pred) + \
                    0.001 * texture_consistency_loss(y_true, y_pred)
    
    return mse_dvh_closs

# --------------------------------------------------------------------------- #

###############################################################################
## Blocks 
###############################################################################

## Remarks
# - inner_activation, batch_norm, dropout can be added as a parameter

###############################################################################
# Conv Block
# - CONV-BN-ACTI * n_convolutions
# - MaxPooling
def conv_block(output_size, previous_layer, n_convolutions, inner_activation,
               kernel_initializer, batch_norm, dropout):
    # Convolve X times
    block = previous_layer
    for i in range(n_convolutions):
        block = Conv3D(output_size, 
                       kernel_size = 3, 
                       padding = 'same', 
                       kernel_initializer = kernel_initializer)(block)
        block = BatchNormalization()(block)
        block = Activation(inner_activation)(block)
        
    # Pool
    pool = MaxPooling3D(pool_size=(2,2,2))(block)
    
    # Return 'block' for future skip connection and 'pool' for next layer
    return block, pool

###############################################################################
# Up-Conv Block
# - Upsample + CONV-BN-ACTI
# - Concat + CONV-BN-ACTI * n_convolutions
def up_conv_block(output_size, previous_layer, skip_connections_layer, 
                  n_convolutions, activation, kernel_initializer, batch_norm, 
                  dropout):
    block = previous_layer
    # Deconvolve
    block = UpSampling3D(size = (2,2,2))(block)
    block = Conv3DTranspose(output_size, 
                            kernel_size = 3, 
                            padding = 'same', 
                            kernel_initializer = kernel_initializer)(block)
    block = BatchNormalization()(block)
    block = Activation(inner_activation)(block)
    # Merge using concatenation
    block = concatenate([skip_connections_layer, block], axis = 4)
    # Convolve X times
    for i in range(n_convolutions):
        block = Conv3D(output_size, 
                       kernel_size = 3, 
                       padding = 'same', 
                       kernel_initializer = kernel_initializer)(block)
        block = BatchNormalization()(block)
        block = Activation(inner_activation)(block)
    # Return 'block' for next layer
    return block

###############################################################################
# Up-Conv Block with attention - reviewed
# source: https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/blob/master/network.py
# - Upsample + CONV on previous layer = phi_previous
# - CONV on skip connection layer = theta_skip
# - SUM(theta_skip, phi_previous) + RELU = f
# - CONV on f = psi_f
# - SIGMOID on psi_f = rate
# - MUL(skip_connections_layer, rate) = attention
# i.e skip_connection_layer modified by att
# - resume with concatenation and convolutions
def up_conv_block_att(output_size, previous_layer, skip_connections_layer, 
                  n_convolutions, activation, kernel_initializer, batch_norm, 
                  dropout):
    block = previous_layer
    # Deconvolve
    block = UpSampling3D(size = (2,2,2))(block)
    
    # Fully deconvolve?
    block = Conv3DTranspose(output_size, 
                            kernel_size = 3, 
                            padding = 'same', 
                            kernel_initializer = kernel_initializer)(block)
    block = BatchNormalization()(block)
    block = Activation(inner_activation)(block)
    
    ####
    # Shape should be [0:None, 1:Length, 2:Width, 3:Height, 4:Channels]
    inter_channel = previous_layer.get_shape().as_list()[4] // 4 # resp 8
    
    theta_skip = Conv3D(inter_channel, 
                        kernel_size=1, 
                        strides=1)(skip_connections_layer)
    
    phi_previous = Conv3D(inter_channel, 
                          kernel_size=1, 
                          strides=1)(block)
    
    f = Activation('relu')(add([theta_skip, phi_previous]))
    
    psi_f = Conv3D(1, 
                   kernel_size=1, 
                   strides=1)(f)

    rate = Activation('sigmoid')(psi_f)

    attention = multiply([skip_connections_layer, rate])
    #####
    
    # Merge using concatenation
    block = concatenate([attention, block], axis = 4)
    # Convolve X times
    for i in range(n_convolutions):
        block = Conv3D(output_size, 
                       kernel_size = 3, 
                       padding = 'same', 
                       kernel_initializer = kernel_initializer)(block)
        block = BatchNormalization()(block)
        block = Activation(inner_activation)(block)
    # Return 'block' for next layer
    return block

###############################################################################
# Conv Block with dense elements
# - (CONV-BN-ACTI + concat) * n_convolutions
# - MaxPooling
def dense_conv_block(output_size, previous_layer, n_convolutions, activation, 
                     kernel_initializer, batch_norm, dropout):
    # Dense convolve X times
    block = previous_layer
    for i in range(n_convolutions):
        previous_data = block
        block = Conv3D(output_size, 
                       kernel_size = 3, 
                       padding = 'same', 
                       kernel_initializer = kernel_initializer)(block)
        
        block = BatchNormalization()(block)
        block = Activation(inner_activation)(block)
        
        block = concatenate([previous_data, block])
    # Dense pool
    pool = MaxPooling3D(pool_size=(2,2,2))(block)
    additional_conv = Conv3D(output_size, 
                             kernel_size = 3,
                             padding = 'same', 
                             strides = 2,
                             kernel_initializer = kernel_initializer)(block)
    additional_conv = BatchNormalization()(additional_conv)
    additional_conv = Activation(inner_activation)(additional_conv)
    pool = concatenate([pool, additional_conv])
    # Return 'block' for future skip connection and 'pool' for next layer
    return block, pool
###############################################################################

###############################################################################
## Net
###############################################################################

###############################################################################
# UNET
def unet_3D(input_size, n_output_channels, dropout_value,
            n_convolutions_per_block, optim, lr, loss, final_activation):
    inputs = Input(input_size)

    ###########################################################################
    ## Architecture
    ###########################################################################
    
    # x32 layers going down
    conv32, pool32 = conv_block(32, inputs, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm,
                                dropout_value)
    # x64 layers going down
    conv64, pool64 = conv_block(64, pool32, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm, 
                                dropout_value)
    # x128 layers going down
    conv128, pool128 = conv_block(128, pool64, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x256 layers going down
    conv256, pool256 = conv_block(256, pool128, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x512 layers (twice as many convolutions)
    conv512, pool512 = conv_block(512, pool256, n_convolutions_per_block*2, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x256 layers going up
    deconv256 = up_conv_block(256, conv512, conv256, n_convolutions_per_block, 
                              inner_activation, kernel_value, batch_norm, 
                              dropout_value)
    # x128 layers going up
    deconv128 = up_conv_block(128, deconv256, conv128, 
                              n_convolutions_per_block, inner_activation, 
                              kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    deconv64 = up_conv_block(64, deconv128, conv64, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)
    # x32 layers going up
    deconv32 = up_conv_block(32, deconv64, conv32, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)

    # Output
    convFIN = Conv3D(n_output_channels, 
                     kernel_size = 1, 
                     activation = final_activation)(deconv32)

    ###########################################################################
    ## Model
    ###########################################################################

    model = Model(inputs = inputs, outputs = convFIN)

    # Manage optimizer, add loss and metric(s)
    if optim == 'adam':
        model.compile(optimizer = Adam(lr = lr), loss = loss, 
                      metrics = [loss])
    elif optim == 'rmsprop':
        model.compile(optimizer = RMSprop(lr = lr), loss = loss, 
                      metrics = [loss])
    else:
        raise NameError('Unknown optimizer.')
    
    model.summary()

    return model

###############################################################################
# UNET
def ablation_unet_3D(input_size, n_output_channels, dropout_value,
                     n_convolutions_per_block, optim, lr, loss, 
                     final_activation, dataset, use_attention,
                     use_consistency_losses, use_dvh_loss, use_dvh_closs):
    inputs = Input(input_size)

    ###########################################################################
    ## Architecture
    ###########################################################################
    
    # x32 layers going down
    conv32, pool32 = conv_block(32, inputs, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm,
                                dropout_value)
    # x64 layers going down
    conv64, pool64 = conv_block(64, pool32, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm, 
                                dropout_value)
    # x128 layers going down
    conv128, pool128 = conv_block(128, pool64, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x256 layers going down
    conv256, pool256 = conv_block(256, pool128, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x512 layers (twice as many convolutions)
    conv512, pool512 = conv_block(512, pool256, n_convolutions_per_block*2, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    
    # Manage attention
    if use_attention:
        current_up_conv_block = up_conv_block_att
    else:  
        current_up_conv_block = up_conv_block
    
    # x256 layers going up
    deconv256 = current_up_conv_block(256, conv512, conv256, n_convolutions_per_block, 
                                      inner_activation, kernel_value, batch_norm, 
                                      dropout_value)
    # x128 layers going up
    deconv128 = current_up_conv_block(128, deconv256, conv128, 
                                      n_convolutions_per_block, inner_activation, 
                                      kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    deconv64 = current_up_conv_block(64, deconv128, conv64, n_convolutions_per_block, 
                                     inner_activation, kernel_value, batch_norm, 
                                     dropout_value)
    # x32 layers going up
    deconv32 = current_up_conv_block(32, deconv64, conv32, n_convolutions_per_block, 
                                     inner_activation, kernel_value, batch_norm, 
                                     dropout_value)

    # Output
    convFIN = Conv3D(n_output_channels, 
                     kernel_size = 1, 
                     activation = final_activation)(deconv32)

    ###########################################################################
    ## Model
    ###########################################################################

    model = Model(inputs = inputs, outputs = convFIN)
    
    # Manage optimizer
    if optim == 'adam':
        optimizer = Adam(lr = lr)
    elif optim == 'rmsprop':
        optimizer = RMSprop(lr = lr)
    else:
        raise NameError('Unknown optimizer.')
        
    # Consistency losses
    if use_consistency_losses:
        model.compile(optimizer = optimizer, 
                      loss = mse_closs_encapsulated(dataset), 
                      metrics = [mse_closs_encapsulated(dataset)])
        
    # DVH loss
    elif use_dvh_loss:
        model.compile(optimizer = optimizer, 
                      loss = mse_dvh_loss_encapsulated(inputs, dataset), 
                      metrics = [mse_dvh_loss_encapsulated(inputs, dataset)])
        
    # DVH C-Loss
    elif use_dvh_closs:
        mode.compile(optimizer = optimizer,
                     loss = mse_dvh_closs_encapsulated(inputs, dataset),
                     metrics = [mse_dvh_closs_encapsulated(inputs, dataset)])
    
    # Normal case
    else:
        model.compile(optimizer = optimizer, loss = loss, 
                      metrics = [loss])
    
    model.summary()

    return model

###############################################################################
# HD-UNET
def ablation_hdunet_3D(input_size, n_output_channels, dropout_value, 
            n_convolutions_per_block, optim, lr, loss, final_activation, dataset, 
            use_attention, use_consistency_losses, use_dvh_loss,
            use_dvh_closs):
    
    inputs = Input(input_size)
    
    # Idea: fixed CONV 16 + Concat previous data
    
    # x32 layers going down
    conv32, pool32 = dense_conv_block(16, inputs, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm, 
                                dropout_value)
    # x64 layers going down
    conv64, pool64 = dense_conv_block(16, pool32, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm, 
                                dropout_value)
    # x128 layers going down
    conv128, pool128 = dense_conv_block(16, pool64, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x256 layers going down
    conv256, pool256 = dense_conv_block(16, pool128, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x512 layers (twice as many convolutions)
    conv512, pool512 = dense_conv_block(16, pool256, n_convolutions_per_block*2, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    
    # Manage attention
    if use_attention:
        current_up_conv_block = up_conv_block_att
    else:  
        current_up_conv_block = up_conv_block
    
    # x256 layers going up
    deconv256 = current_up_conv_block(16, conv512, conv256, n_convolutions_per_block, 
                                      inner_activation, kernel_value, batch_norm, 
                                      dropout_value)
    # x128 layers going up
    deconv128 = current_up_conv_block(16, deconv256, conv128, n_convolutions_per_block, 
                                      inner_activation, kernel_value, batch_norm, 
                                      dropout_value)
    # x64 layers going up
    deconv64 = current_up_conv_block(16, deconv128, conv64, n_convolutions_per_block, 
                                     inner_activation, kernel_value, batch_norm, 
                                     dropout_value)
    # x32 layers going up
    deconv32 = current_up_conv_block(16, deconv64, conv32, n_convolutions_per_block, 
                                     inner_activation, kernel_value, batch_norm, 
                                     dropout_value)

    # Output
    convFIN = Conv3D(n_output_channels, 
                     kernel_size = 1, 
                     activation = final_activation)(deconv32)

    ###########################################################################
    ## Model
    ###########################################################################

    model = Model(inputs = inputs, outputs = convFIN)

   # Manage optimizer
    if optim == 'adam':
        optimizer = Adam(lr = lr)
    elif optim == 'rmsprop':
        optimizer = RMSprop(lr = lr)
    else:
        raise NameError('Unknown optimizer.')
        
    # Consistency losses
    if use_consistency_losses:
        model.compile(optimizer = optimizer, 
                      loss = mse_closs_encapsulated(dataset), 
                      metrics = [mse_closs_encapsulated(dataset)])
        
    # DVH loss
    elif use_dvh_loss:
        model.compile(optimizer = optimizer, 
                      loss = mse_dvh_loss_encapsulated(inputs, dataset), 
                      metrics = [mse_dvh_loss_encapsulated(inputs, dataset)])
        
    # DVH C-Loss
    elif use_dvh_closs:
        mode.compile(optimizer = optimizer,
                     loss = mse_dvh_closs_encapsulated(inputs, dataset),
                     metrics = [mse_dvh_closs_encapsulated(inputs, dataset)])
    
    # Normal case
    else:
        model.compile(optimizer = optimizer, loss = loss, 
                      metrics = [loss])
    
    model.summary()

    return model

###############################################################################
# BRANCH UNET
def branch_unet_3D(input_size, n_output_channels, dropout_value,
                   n_convolutions_per_block, optim, lr, loss, final_activation,
                   use_attention):
    inputs = Input(input_size)

    ###########################################################################
    ## Architecture
    ###########################################################################
    
    # x32 layers going down
    conv32, pool32 = conv_block(32, inputs, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm,
                                dropout_value)
    # x64 layers going down
    conv64, pool64 = conv_block(64, pool32, n_convolutions_per_block, 
                                inner_activation, kernel_value, batch_norm, 
                                dropout_value)
    # x128 layers going down
    conv128, pool128 = conv_block(128, pool64, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x256 layers going down
    conv256, pool256 = conv_block(256, pool128, n_convolutions_per_block, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    # x512 layers (twice as many convolutions)
    conv512, pool512 = conv_block(512, pool256, n_convolutions_per_block*2, 
                                  inner_activation, kernel_value, batch_norm, 
                                  dropout_value)
    
    # BRANCH A
    # - dose prediction
    
    # x256 layers going up
    deconv256_dose = up_conv_block_att(256, conv512, conv256, n_convolutions_per_block, 
                              inner_activation, kernel_value, batch_norm, 
                              dropout_value)
    # x128 layers going up
    deconv128_dose = up_conv_block_att(128, deconv256_dose, conv128, 
                              n_convolutions_per_block, inner_activation, 
                              kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    deconv64_dose = up_conv_block_att(64, deconv128_dose, conv64, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)
    # x32 layers going up
    deconv32_dose = up_conv_block_att(32, deconv64_dose, conv32, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)

    # Output
    convFIN_dose = Conv3D(n_output_channels, 
                          kernel_size = 1, 
                          activation = final_activation)(deconv32_dose)
    
    # BRANCH B
    # - isodose prediction
    
    # x256 layers going up
    deconv256_isodose = up_conv_block_att(256, conv512, conv256, n_convolutions_per_block, 
                              inner_activation, kernel_value, batch_norm, 
                              dropout_value)
    # x128 layers going up
    deconv128_isodose = up_conv_block_att(128, deconv256_isodose, conv128, 
                              n_convolutions_per_block, inner_activation, 
                              kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    deconv64_isodose = up_conv_block_att(64, deconv128_isodose, conv64, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)
    # x32 layers going up
    deconv32_isodose = up_conv_block_att(32, deconv64_isodose, conv32, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)

    # Output
    convFIN_isodose = Conv3D(n_output_channels, 
                             kernel_size = 1, 
                             activation = final_activation)(deconv32_isodose)
    
    # BRANCH C
    # - edges prediction
    
    # x256 layers going up
    deconv256_edges = up_conv_block_att(256, conv512, conv256, n_convolutions_per_block, 
                              inner_activation, kernel_value, batch_norm, 
                              dropout_value)
    # x128 layers going up
    deconv128_edges = up_conv_block_att(128, deconv256_edges, conv128, 
                              n_convolutions_per_block, inner_activation, 
                              kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    deconv64_edges = up_conv_block_att(64, deconv128_edges, conv64, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)
    # x32 layers going up
    deconv32_edges = up_conv_block_att(32, deconv64_edges, conv32, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)

    # Output
    convFIN_edges = Conv3D(n_output_channels, 
                             kernel_size = 1, 
                             activation = final_activation)(deconv32_edges)

    ###########################################################################
    ## Model
    ###########################################################################
    model = Model(inputs = inputs, outputs = [convFIN_dose, 
                                              convFIN_isodose, 
                                              convFIN_edges])
    
    # Losses and loss weights
    custom_loss = {'convFIN_dose': loss, 
                   'convFIN_isodose': 'SparseCategoricalCrossentropy', 
                   'convFIN_edges': loss}
    
    custom_loss_weights = {'convFIN_dose': 1, 
                           'convFIN_isodose': 1, 
                           'convFIN_edges': 1} 

    # edges might be harder to predict, so its weight could start high and then decrease

    # Manage optimizer, add loss and metric(s)
    if optim == 'adam':
        model.compile(optimizer = Adam(lr = lr), 
                      loss = custom_loss, 
                      loss_weights = custom_loss_weights,
                      metrics = custom_loss)
    elif optim == 'rmsprop':
        model.compile(optimizer = RMSprop(lr = lr), 
                      loss = custom_loss, 
                      loss_weights = custom_loss_weights,
                      metrics = custom_loss)
    else:
        raise NameError('Unknown optimizer.')
    
    model.summary()

    return model