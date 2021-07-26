## Imports
# DeepL
from keras.layers import Conv3D, BatchNormalization, Activation, Input, \
    MaxPooling3D, UpSampling3D, Conv3DTranspose
from keras.layers import add, multiply, concatenate
from keras import Model
from keras.optimizers import Adam, RMSprop
from metrics import *

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
                
# Dose score
# Computes different metrics and gathers them into a score
def dose_score(input_data, reference, prediction):
    
    # Design choice
    # Here we are using the reference dose whereas at test time we use the
    # ideal prescribed dose to the tumor.
    # Our goal is to match the reference dose more than it is to match the
    # ideal dose, since it can't physically be administered. It's a tradeoff,
    # but only the reference plan gives us the deliverable aspect.
    reference_tumor_gy = reference[get_biggest_tv(input_data[0]).nonzero()]
    
    # PTV Coverage D98, D95, D50, D5, D2
    D98 = abs(ptv_coverage(prediction[0, :, : , :, 0], 
                           reference_tumor_gy, 
                           98))
            
    D95 = abs(ptv_coverage(prediction[0, :, : , :, 0], 
                           reference_tumor_gy, 
                           95))
              
    D50 = abs(ptv_coverage(prediction[0, :, : , :, 0], 
                           reference_tumor_gy, 
                           50))
    
    D5 = abs(ptv_coverage(prediction[0, :, : , :, 0], 
                           reference_tumor_gy, 
                           5))
    
    D2 = abs(ptv_coverage(prediction[0, :, : , :, 0], 
                           reference_tumor_gy, 
                           2))
    
    # Homogeneity 1, Homogeneity 2, Homogeneity 3
    H1 = (D2 - D98) / D50
    H2 = D95 / D5
    H3 = max(prediction) / max(reference)
    
    # Score
    return abs(1 - (D98 + D95 + D2 + (1 + H1) + H2 + H3) / 6)
    
# Custom loss
def custom_loss(input_data, factors, reference, prediction):
    
    return (factors[0] * keras.losses.mean_squared_error(reference, prediction) + \
            factors[1] * dose_score(input_data, reference, prediction))
    

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
# TODO TOTEST - reviewed
# source: https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/blob/master/network.py
# Up-Conv Block with attention
# - Upsample + CONV on previous layer = phi_previous
# - CONV on skip connection layer = theta_skip
# - SUM(theta_skip, phi_previous) + RELU = f
# - CONV on f = psi_f
# - SIGMOID on psi_f = rate
# - MUL(skip_connections_layer, rate) = attention
# i.e skip_connection_layer modified by att
# - resume with concatenation and convolutions
# Rq: Conv3DTranspose? BN?
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
                     final_activation, use_attention, use_dose_score):
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
    
    # Manage dose score
    if use_dose_score:
        model.compile(optimizer = optimizer, loss = custom_loss(inputs, [100, 1]), 
                      metrics = [custom_loss(inputs, [100, 1])])
    else:
        model.compile(optimizer = optimizer, loss = loss, 
                      metrics = [loss])
    
    model.summary()

    return model

###############################################################################
# HD-UNET
def hdunet_3D(input_size, n_output_channels, dropout_value, 
            n_convolutions_per_block, optim, lr, loss, final_activation):
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
    # x256 layers going up
    deconv256 = up_conv_block(16, conv512, conv256, n_convolutions_per_block, 
                              inner_activation, kernel_value, batch_norm, 
                              dropout_value)
    # x128 layers going up
    deconv128 = up_conv_block(16, deconv256, conv128, 
                              n_convolutions_per_block, inner_activation, 
                              kernel_value, batch_norm, dropout_value)
    # x64 layers going up
    deconv64 = up_conv_block(16, deconv128, conv64, n_convolutions_per_block, 
                             inner_activation, kernel_value, batch_norm, 
                             dropout_value)
    # x32 layers going up
    deconv32 = up_conv_block(16, deconv64, conv32, n_convolutions_per_block, 
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
