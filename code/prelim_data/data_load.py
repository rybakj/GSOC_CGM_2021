import pandas as pd
import numpy as np

import os
from os import listdir
import h5py
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import pickle

def get_data_from_directory(directory):
    '''
    Extract all data from a given directory, split them into physical and spectral properties,
    and return concatenated numpy arrays for both

    Inputs:
    - directory path (string)

    Outputs:
    - numpy array of physical properties
    - numpy array of spectral properties
    '''


    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    files_list = listdir(directory)
    for data_file in files_list:
        data = h5py.File(directory + '/' + str(data_file), 'r')

        physical_prop, spectral_prop, _ = get_properties_from_file(data)

        if phys_prop_joint.size == 0:
            phys_prop_joint = physical_prop
        else:
            phys_prop_joint = np.concatenate((phys_prop_joint, physical_prop), axis=0)

        if spectral_prop_joint.size == 0:
            spectral_prop_joint = spectral_prop
        else:
            spectral_prop_joint = np.concatenate((spectral_prop_joint, spectral_prop), axis=0)

    return (phys_prop_joint, spectral_prop_joint)


def get_properties_from_file(data):
    '''
    For an (opened) h5py file, extract physical proeprties, spectral properties and targets

    Inputs:
    - Open h5py file (obttained by calling 'h5py.File(...)')

    Outputs:
    - numpy array of physical properties
    - numpy array of spectral properties
    - numpy array of targets
    '''

    inputs = data['Data']['x_values']
    spectral_prop, physical_prop = inputs[:, :-5], inputs[:, -5:]
    targets = data['Data']['y_values']

    return (physical_prop, spectral_prop, targets)


def get_data_from_directory_list(directory_list):
    '''Get all spectral and physical data contained in a list of directories

    Output:
    - directory with phys and spectral properties under different keys. Data returned as numpy arrays (dict. values)'''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    for directory in directory_list:
        physical_prop, spectral_prop = get_data_from_directory(directory)
        print("Directory:", directory,
              "Physical properties shape:", physical_prop.shape,
              "Spectral prop shape:", spectral_prop.shape)

        if phys_prop_joint.size == 0:
            phys_prop_joint = physical_prop
        else:
            phys_prop_joint = np.concatenate((phys_prop_joint, physical_prop), axis=0)

        if spectral_prop_joint.size == 0:
            spectral_prop_joint = spectral_prop
        else:
            spectral_prop_joint = np.concatenate((spectral_prop_joint, spectral_prop), axis=0)

    data = {"physical_properties": phys_prop_joint,
            "spectral_properties": spectral_prop_joint}
    # data = np.concatenate( (phys_prop_joint, spectral_prop_joint), axis = 1 )

    return (data)


def pooling_for_x(x, bin_size, stats):
    '''For a given dataframe of data, apply pooling based on the statistics supplied.
    If the length of x isn't divisible by bin_size, 0s are appended at the end of x.

    Inputs:
    - x (dataframe/np arry)
    - bin_size: integer
    - stats: object calculating the summary statistics which should be used for pooling.
    For example, "np.mean". '''
    padding_size = np.ceil(x.shape[1] / bin_size) * bin_size - x.shape[1]

    if bin_size > 0:
        zeros_array = np.zeros((x.shape[0], int(padding_size)))

        x = np.concatenate((x, zeros_array + x[:, -1:]), axis=1)

    x_pooled = x.reshape(x.shape[0], -1, bin_size)
    x_pooled = stats(x_pooled, axis=2)

    return (x_pooled)

def dump_object(object_name, object_tosave):
    '''Saves a given object under a given name in directory ./outputs (if such directory
     does not exist the function creates it.'''
    if not os.path.isdir('outputs'):
            os.mkdir('outputs')
    # file_tosave.to_csv('outputs/' + filename + '.csv', index=False)

    with open('outputs/' + object_name, 'wb') as dump_location:
        pickle.dump(object_tosave, dump_location)

    return()

def get_wavelength(start, stop, step, pooling_width = 1):
    '''
    Caluclate sequence of wavelength

    :param start: first wavelength observed
    :param stop: last wavelength
    :param step: resolution
    :param pooling_width: If pooling applied, size of pooling window. Otherwise set to 1
    :return: wavelengths of the pooled spectra (np.array
    '''
    original_wavelengths = np.arange(start, stop, step)

    pooled_wavelength_stepsize = step * pooling_width # 10 is the pooling width
    pooled_wavelengths = np.arange(start, stop, pooled_wavelength_stepsize)

    return( pooled_wavelengths )


def load_split_pool(train_size, val_size, pooling_width, scale = True, take_log = False,
                    normalize_by_wflux = False, normalize_by_sflux = False, directory_list = None):
    '''Load the spectral and physical data, apply the train/val/test split and pooling of a given width.
    For explanatory data (x), these are loaded, split, pooled, standardised to have zero mean and unit variance in this
    order.

    Returns dictionary with y, x and x pooled data.

    Note that the code load data from a specific directory - you may need to change this inside the function if using a different
    directory structure.

    Also note that the dataset is shuffled randomly with a fixe random state to allow reporducibility.'''

    if directory_list == None:
        directory_list = ['Complete_Spectral_Data\Training_Data', 'Complete_Spectral_Data\Test_Data']

    else:
        directory_list = directory_list

    data_all = get_data_from_directory_list(directory_list)

    spectral_data = data_all["spectral_properties"]
    physical_data = data_all["physical_properties"]

    print("Spectral data shape", spectral_data.shape)
    print("Physical data shape", physical_data.shape)

    spectral_data, physical_data = shuffle(spectral_data, physical_data, random_state=5684)

    x_train = spectral_data[: train_size, :]
    x_val = spectral_data[train_size: train_size + val_size, :]
    x_test = spectral_data[train_size + val_size:]



    y_train = physical_data[: train_size, :]
    y_val = physical_data[train_size: train_size + val_size, :]
    y_test = physical_data[train_size + val_size:, :]

    x_val_df = pd.DataFrame(x_val)

    train_df_pooled = pooling_for_x(x_train, pooling_width, np.mean)
    val_df_pooled = pooling_for_x(x_val, pooling_width, np.mean)
    test_df_pooled = pooling_for_x(x_test, pooling_width, np.mean)

    if take_log == True:
        train_df_pooled = -np.log( train_df_pooled + 1e-5 )
        val_df_pooled = -np.log(  val_df_pooled  + 1e-5 )
        test_df_pooled = -np.log( test_df_pooled + 1e-5 )

    if normalize_by_sflux == True:

        train_df_pooled_total_flux = np.abs(train_df_pooled-1).std(axis = 1).reshape(-1, 1)  #/ np.abs(train_df_pooled-1).sum().sum()
        val_df_pooled_total_flux = np.abs(val_df_pooled-1).std(axis=1).reshape(-1, 1) #/ np.abs(val_df_pooled-1).sum().sum()
        test_df_pooled_total_flux = np.abs(test_df_pooled-1).std(axis=1).reshape(-1, 1) #/ np.abs(test_df_pooled-1).sum().sum().sum()

        # val_df_pooled_total_flux = train_df_pooled_total_flux
        # test_df_pooled_total_flux = train_df_pooled_total_flux

        train_df_pooled = train_df_pooled / train_df_pooled_total_flux
        val_df_pooled = val_df_pooled / val_df_pooled_total_flux
        test_df_pooled = test_df_pooled / test_df_pooled_total_flux

    if normalize_by_wflux == True:

        train_df_pooled_total_flux = np.abs(train_df_pooled-1).sum(axis = 0).reshape(1,-1)
        # val_df_pooled_total_flux = val_df_pooled.sum(axis=0).reshape(1,-1)
        # test_df_pooled_total_flux = test_df_pooled.sum(axis=0).reshape(1,-1)

        val_df_pooled_total_flux = train_df_pooled_total_flux
        test_df_pooled_total_flux = train_df_pooled_total_flux

        train_df_pooled = train_df_pooled / train_df_pooled_total_flux
        val_df_pooled = val_df_pooled / val_df_pooled_total_flux
        test_df_pooled = test_df_pooled / test_df_pooled_total_flux


    if scale == True:

        scaler = StandardScaler()

        scaler.fit(train_df_pooled)

        train_df_pooled = pd.DataFrame(scaler.transform(train_df_pooled))
        val_df_pooled = pd.DataFrame(scaler.transform(val_df_pooled))
        test_df_pooled = pd.DataFrame(scaler.transform(test_df_pooled))

        mean = scaler.mean_
        std = scaler.scale_

    else:
        mean = np.nan
        std = np.nan

    data_dict = dict()


    data_dict["x"] = dict()

    data_dict["x"]["train"] = x_train
    data_dict["x"]["test"] = x_val
    data_dict["x"]["val"] = x_test


    data_dict["x_pooled"] = dict()
    data_dict["x_pooled"]["train"] = train_df_pooled
    data_dict["x_pooled"]["val"] = val_df_pooled
    data_dict["x_pooled"]["test"] = test_df_pooled

    try:
        data_dict["x_pooled"]["train_df_pooled_total_flux"] = train_df_pooled_total_flux
        data_dict["x_pooled"]["val_df_pooled_total_flux"] = val_df_pooled_total_flux
        data_dict["x_pooled"]["test_df_pooled_total_flux"] = test_df_pooled_total_flux
    except:
        pass

    data_dict["y"] = dict()
    data_dict["y"]["train"] = y_train
    data_dict["y"]["val"] = y_val
    data_dict["y"]["test"] = y_test

    data_dict["parameters"] = dict()
    data_dict["parameters"]["mean"] = mean
    data_dict["parameters"]["std"] = std

    wavelengths = get_wavelength(start=900, stop=2000.01, step=0.01, pooling_width=pooling_width)
    data_dict["wavelengths"] = wavelengths

    return (data_dict)

