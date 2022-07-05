# Dose plan prediction project #

This repository contains a significant part of the final set of codes that allowed me to produce 3D dose plans for Head and Neck (H&N) cancer patients on two datasets during my Master's: a dataset provided by the Centre Hospitalier de l'Université de Montréal (CHUM) and a public dataset available under the OpenKBP challenge.

See published paper here: https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.13655

### Train model V2 (train_model_v2.py) ###

Launches training and generates model weights, Dice values (.npy) and curves (.png), train/validation/test IDs.

#### Arguments ####

* __path__, __path_to_main_folder__ : A string specifiying the name of the folder that will contain the outputs.
* __ids__, __path_to_ids__ : A string specifiying the name of the folder that contains the different seeded ids.
* __o__, __optim__ : A string specifiying the name optimizer to be used (supported optimizers: 'adam' or 'rmsprop').
* __lr__ : A float specifying the learning rate.
* __drop__, __dropout_value__ : A float specifying the dropout value.
* __e__, __n_epochs__ : An integer specifying the number of epochs.
* __seed__ : An integer specifying the seed for numpy and tensorflow operations to ensure reproductibility.
* __loss__ : A string specifying the loss to use for training (e.g mse, mae, rmse).
* __final_activation__ : An string specifying the kind of final activation to use in the fully connected layer (e.g sigmoid, linear, tanh).
* __use_hdunet__ : A flag ensuring we use the HD U-Net model variant.
* __use_attention__ : A flag ensuring we use the Attention U-Net model variant.
* __use_closs__ : A flag ensuring we use the consistency loss variant.
* __use_dvh_loss__ : A flag ensuring we use the DVH loss variant.
* __use_dvh_closs__ : A flag ensuring we use the DVH Consistency loss variant.
* __dset__ : A string specifying the dataset we are using (supported datasets: 'CHUM' and 'OpenKBP').

### Test model (test_model_v2.py) ###

Launches testing and generates metrics scores per patient and averages across a given set of the data.

#### Arguments ####

* __mode__, __test_mode__ : An string specifying the test mode (supported test modes: 'generate_predictions' and 'evaluate_predictions').
* __path__, __path_to_model_folder__ : A string specifiying the path to the model folder where the model file is located and where outpus will be generated.
* __mname__, __model_name__ : An string specifying the name of the model file.
* __dset__ : A string specifying the dataset we are using (supported datasets: 'CHUM' and 'OpenKBP').
* __set__, __kind_of_set__ : A string specifying the set to generated for (either 'train', 'validation', 'test' or 'manual').
* __ids__, __list_of_ids__ : Multiple strings specifying the IDs of patients to be tested (required when set is 'manual').
* __use_closs__ : A flag ensuring we use the consistency loss variant.
* __use_dvh_loss__ : A flag ensuring we use the DVH loss variant.
* __use_dvh_closs__ : A flag ensuring we use the DVH Consistency loss variant.

### Data generator (data_gen_v2.py) ###

Redefinition of keras.utils.Sequence DataGenerator.

### U-net models (unet_model_v2.py) ###

Model definitions of a 3D U-Net and a 3D HD U-Net with a thorough definition of blocks with classic CNN modules as well as attention variants. Also contains the definition of different losses: texture and isodose consistency losses, DVH loss and a combination of both.

### Metrics (metrics.py) ###

Definition of all the medical metrics used during the project notably PTV coverage, mean and max doses, homogeneities, dose errors, several conformity indexes, Van't Riet conformation number, spillage, dose score from OpenKBP, DVH score. Also contains the mathematical metrics used notably MAE and MSE per structure.

### Utils (utils/) ###

Folder containing a data standardization script to (de)normalize inputs and outputs.

### Jobs (jobs_v2/) ###

Folder containing an example of job launched on a proxy calculation server.

### Data preparation and generation (data_generation/) ###

A folder containing ids folders, groundtruth metrics and scripts used during data preparation. 




