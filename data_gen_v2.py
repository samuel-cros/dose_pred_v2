###############################################################################
### IMPORTS
###############################################################################
# Math
import numpy as np

# DeepL
import keras

# IO

###############################################################################
### Subfunctions
###############################################################################

###############################################################################
### MAIN
###############################################################################
# DataGenerator
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self, train_or_validation, list_IDs, patch_dim, batch_size, dataset, 
        n_input_channels, n_output_channels, shuffle=True, 
        augmentation=False):
        'Initialization'
        self.patch_dim = patch_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.dataset = dataset
        self.on_epoch_end()
        if train_or_validation == 'validation' and not shuffle:
            self.shuffle_once()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def shuffle_once(self):
        'Updates indices once'
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Grab maximum height
        max_height = 0
        for ID in list_IDs_temp:
            max_height = max(max_height, self.dataset[ID]['dose'].shape[2])
        
        # Initialization
        X = np.empty((self.batch_size, 
                      self.patch_dim[0],
                      self.patch_dim[1],
                      max_height, 
                      self.n_input_channels))
        y = np.empty((self.batch_size, 
                      self.patch_dim[0],
                      self.patch_dim[1],
                      max_height, 
                      self.n_output_channels))

        # Generate data
        # in the future, for batchsize of 2+, pad inputs to max_height
        for i, ID in enumerate(list_IDs_temp):
        
            # Store sample
            X[i,] = self.dataset[ID]['input'] #[()]

            y[i,] = np.expand_dims(self.dataset[ID]['dose'], axis=-1) #[()]

        return X, y

    ###########################################################################
    ### Generate input and output
    ###########################################################################
    def generate_data(self, ID):

        #######################################################################
        ### OUTPUT
        #######################################################################

        #######################################################################
        ### INPUT
        #######################################################################
          
        #######################################################################
        ### Return
        #######################################################################
        pass

##################################################################
