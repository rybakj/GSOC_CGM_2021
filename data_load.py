import pandas as pd
import numpy as np

from os import listdir
import h5py


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
        data = h5py.File(directory + '\\' + str(data_file), 'r')

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

