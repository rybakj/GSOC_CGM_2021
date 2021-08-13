<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import h5py
import os
from os import listdir
from sklearn.utils import shuffle

# phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
#              "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
#              "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
#              "halo_orientation_angles1", "halo_orientation_angles2",
#              "ray_orientation_angles1", "ray_orientation_angles2",
#              "ray_number"]





def pooling_for_x(x, bin_size, stats):
    '''For a given dataframe of data, apply pooling based on the statistics supplied.
    If the length of x isn't divisible by bin_size, 0s are appended at the end of x.

    Inputs:
    - x (dataframe/np arry)
    - bin_size: integer
    - stats: object calculating the summary statistics which should be used for pooling.
    For example, "np.mean". '''

    if x.size > 0:
        padding_size = np.ceil(x.shape[1] / bin_size) * bin_size - x.shape[1]

        if bin_size > 0:
            zeros_array = np.zeros((x.shape[0], int(padding_size)))

            x = np.concatenate((x, zeros_array + x[:, -1:]), axis=1)

        x_pooled = x.reshape(x.shape[0], -1, bin_size)
        x_pooled = stats(x_pooled, axis=2)

    # Take care of val set potentially empty
    else:
        x_pooled = np.nan

    return(x_pooled)


def get_data_from_directory(directory):
    '''
    Extract all data from a given directory, split them into physical and spectral properties,
    and return concatenated numpy arrays for both

    Inputs:
    - directory path (string)

    Outputs:
    - dictionary containing the data and column names of physical properties
    '''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    files_list = listdir(directory)

    for data_file in files_list:
        file = h5py.File(directory + '/' + str(data_file), 'r')

        data_from_dile_dict = get_data_from_file(file)

        physical_prop = data_from_dile_dict["physical_properties"]
        spectral_prop = data_from_dile_dict["spectral_properties"]
        phys_col_names = data_from_dile_dict["physical_properties_cols"]

        if phys_prop_joint.size == 0:
            phys_prop_joint = physical_prop
        else:
            phys_prop_joint = np.concatenate((phys_prop_joint, physical_prop), axis=0)

        if spectral_prop_joint.size == 0:
            spectral_prop_joint = spectral_prop
        else:
            spectral_prop_joint = np.concatenate((spectral_prop_joint, spectral_prop), axis=0)

    data = {"physical_properties": phys_prop_joint,
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_col_names}

    return (data)



def get_data_from_file(file):


    # f=h5py.File('halos_0_100.hdf5', 'r')


    phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
                 "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
                 "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
                 "halo_orientation_angles1", "halo_orientation_angles2",
                 "ray_orientation_angles1", "ray_orientation_angles2",
                 "ray_number"]

    spectral_data_all = np.array([])
    y_data_all = np.array([])


    for halo in file.keys():

        spectral_data_halo = file[halo]["spectral_data"]
        spectral_data_halo = spectral_data_halo[:, 40 : -40]
        # print("Trimming spectrum by 40 wavelengths on each side")

        bad_trajectories = file[halo]["bad_trajectories"]
        zero_length = file[halo]["zero_length"]
        black_holes = file[halo]["PartType5"]
        physical_data = file[halo]["physical_data"]

        if bad_trajectories.size == 0:
            bad_trajectories = np.zeros((physical_data.shape[0],1))
        else:
            bad_trajectories = np.ones((physical_data.shape[0],1))* bad_trajectories.size


        if zero_length.size == 0: zero_length = np.zeros((physical_data.shape[0],1))


        y_data_halo = np.concatenate( [bad_trajectories, zero_length, physical_data], axis = 1 )
        y_data_df_halo = pd.DataFrame(y_data_halo, columns = phys_col_names )


        if spectral_data_all.size == 0:
            spectral_data_all = spectral_data_halo
        else:
            spectral_data_all = np.concatenate((spectral_data_all, spectral_data_halo), axis=0)

        if y_data_all.size == 0:
            y_data_all = y_data_halo
        else:
            y_data_all = np.concatenate((y_data_all, y_data_halo), axis=0)


    data = {"physical_properties": y_data_all,
            "spectral_properties": spectral_data_all,
           "physical_properties_cols": phys_col_names}

    return( data )


def get_data_from_directory_list(directory_list):
    '''Get all spectral and physical data contained in a list of directories

    Output:
    - directory with phys and spectral properties under different keys. Data returned as numpy arrays (dict. values)'''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    for directory in directory_list:
        data_from_directory = get_data_from_directory(directory)

        physical_prop = data_from_directory["physical_properties"]
        spectral_prop = data_from_directory["spectral_properties"]
        phys_prop_columns = data_from_directory["physical_properties_cols"]

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
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_prop_columns}

    return (data)


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


def load_split_pool(train_size, val_size, pooling_width=1, scale=False,
                    directory_list=None):
    '''Load the spectral and physical data, apply the train/val/test split and pooling of a given width.
    For explanatory data (x), these are loaded, split, pooled, standardised to have zero mean and unit variance in this
    order.

    Returns dictionary with y, x and x pooled data.

    Note that the code load data from a specific directory - you may need to change this inside the function if using a different
    directory structure.

    Also note that the dataset is shuffled randomly with a fixed random state to allow reporducibility of results.'''

    data_all = get_data_from_directory_list(directory_list)

    spectral_data = data_all["spectral_properties"]
    physical_data = data_all["physical_properties"]
    physical_col_names = data_all["physical_properties_cols"]

    print("Spectral data shape", spectral_data.shape)
    print("Physical data shape", physical_data.shape)

    spectral_data, physical_data = shuffle(spectral_data, physical_data, random_state=687)

    if val_size == "max":
        test_size = 0
        val_size = spectral_data.shape[1] - train_size
    else:
        pass

    x_train = spectral_data[: train_size, :]
    x_val = spectral_data[train_size: train_size + val_size, :]
    x_test = spectral_data[train_size + val_size:]

    y_train = physical_data[: train_size, :]
    y_val = physical_data[train_size: train_size + val_size, :]
    y_test = physical_data[train_size + val_size:, :]

    # save to dictionary
    data_dict = dict()

    data_dict["physical_col_names"] = physical_col_names

    data_dict["x"] = dict()

    data_dict["x"]["train"] = x_train
    data_dict["x"]["val"] = x_val
    data_dict["x"]["test"] = x_test

    data_dict["y"] = dict()
    data_dict["y"]["train"] = y_train
    data_dict["y"]["val"] = y_val
    data_dict["y"]["test"] = y_test

    if pooling_width is not None:
        x_train = pooling_for_x(x_train, pooling_width, np.mean)
        x_val = pooling_for_x(x_val, pooling_width, np.mean)
        x_test = pooling_for_x(x_test, pooling_width, np.mean)

    if scale == True:

        scaler = StandardScaler()

        scaler.fit(x_train)

        x_train = pd.DataFrame(scaler.transform(train_df_pooled))
        x_val = pd.DataFrame(scaler.transform(val_df_pooled))
        x_test = pd.DataFrame(scaler.transform(test_df_pooled))

        mean = scaler.mean_
        std = scaler.scale_

    else:
        mean = np.nan
        std = np.nan

    if pooling_width is not None or scale == True:
        data_dict["x_transformed"] = dict()
        data_dict["x_transformed"]["train"] = x_train
        data_dict["x_transformed"]["val"] = x_val
        data_dict["x_transformed"]["test"] = x_test

    # transformation / scaling parameters
    data_dict["parameters"] = dict()
    data_dict["parameters"]["mean"] = mean
    data_dict["parameters"]["std"] = std

    # get wavelengths
    if pooling_width is not None:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01, pooling_width=pooling_width)
        # wavelengths = wavelengths[np.floor(40/pooling_width).astype(int):-np.floor(40/pooling_width).astype(int)]
    else:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01)
        # wavelengths = wavelengths[40:-40]

    data_dict["wavelengths"] = wavelengths

=======
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import h5py
import os
from os import listdir
from sklearn.utils import shuffle

# phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
#              "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
#              "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
#              "halo_orientation_angles1", "halo_orientation_angles2",
#              "ray_orientation_angles1", "ray_orientation_angles2",
#              "ray_number"]





def pooling_for_x(x, bin_size, stats):
    '''For a given dataframe of data, apply pooling based on the statistics supplied.
    If the length of x isn't divisible by bin_size, 0s are appended at the end of x.

    Inputs:
    - x (dataframe/np arry)
    - bin_size: integer
    - stats: object calculating the summary statistics which should be used for pooling.
    For example, "np.mean". '''

    if x.size > 0:
        padding_size = np.ceil(x.shape[1] / bin_size) * bin_size - x.shape[1]

        if bin_size > 0:
            zeros_array = np.zeros((x.shape[0], int(padding_size)))

            x = np.concatenate((x, zeros_array + x[:, -1:]), axis=1)

        x_pooled = x.reshape(x.shape[0], -1, bin_size)
        x_pooled = stats(x_pooled, axis=2)

    # Take care of val set potentially empty
    else:
        x_pooled = np.nan

    return(x_pooled)


def get_data_from_directory(directory):
    '''
    Extract all data from a given directory, split them into physical and spectral properties,
    and return concatenated numpy arrays for both

    Inputs:
    - directory path (string)

    Outputs:
    - dictionary containing the data and column names of physical properties
    '''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    files_list = listdir(directory)

    for data_file in files_list:
        file = h5py.File(directory + '/' + str(data_file), 'r')

        data_from_dile_dict = get_data_from_file(file)

        physical_prop = data_from_dile_dict["physical_properties"]
        spectral_prop = data_from_dile_dict["spectral_properties"]
        phys_col_names = data_from_dile_dict["physical_properties_cols"]

        if phys_prop_joint.size == 0:
            phys_prop_joint = physical_prop
        else:
            phys_prop_joint = np.concatenate((phys_prop_joint, physical_prop), axis=0)

        if spectral_prop_joint.size == 0:
            spectral_prop_joint = spectral_prop
        else:
            spectral_prop_joint = np.concatenate((spectral_prop_joint, spectral_prop), axis=0)

    data = {"physical_properties": phys_prop_joint,
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_col_names}

    return (data)



def get_data_from_file(file):


    # f=h5py.File('halos_0_100.hdf5', 'r')


    phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
                 "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
                 "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
                 "halo_orientation_angles1", "halo_orientation_angles2",
                 "ray_orientation_angles1", "ray_orientation_angles2",
                 "ray_number"]

    spectral_data_all = np.array([])
    y_data_all = np.array([])


    for halo in file.keys():

        spectral_data_halo = file[halo]["spectral_data"]
        spectral_data_halo = spectral_data_halo[:, 40 : -40]
        # print("Trimming spectrum by 40 wavelengths on each side")

        bad_trajectories = file[halo]["bad_trajectories"]
        zero_length = file[halo]["zero_length"]
        black_holes = file[halo]["PartType5"]
        physical_data = file[halo]["physical_data"]

        if bad_trajectories.size == 0:
            bad_trajectories = np.zeros((physical_data.shape[0],1))
        else:
            bad_trajectories = np.ones((physical_data.shape[0],1))* bad_trajectories.size


        if zero_length.size == 0: zero_length = np.zeros((physical_data.shape[0],1))


        y_data_halo = np.concatenate( [bad_trajectories, zero_length, physical_data], axis = 1 )
        y_data_df_halo = pd.DataFrame(y_data_halo, columns = phys_col_names )


        if spectral_data_all.size == 0:
            spectral_data_all = spectral_data_halo
        else:
            spectral_data_all = np.concatenate((spectral_data_all, spectral_data_halo), axis=0)

        if y_data_all.size == 0:
            y_data_all = y_data_halo
        else:
            y_data_all = np.concatenate((y_data_all, y_data_halo), axis=0)


    data = {"physical_properties": y_data_all,
            "spectral_properties": spectral_data_all,
           "physical_properties_cols": phys_col_names}

    return( data )


def get_data_from_directory_list(directory_list):
    '''Get all spectral and physical data contained in a list of directories

    Output:
    - directory with phys and spectral properties under different keys. Data returned as numpy arrays (dict. values)'''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    for directory in directory_list:
        data_from_directory = get_data_from_directory(directory)

        physical_prop = data_from_directory["physical_properties"]
        spectral_prop = data_from_directory["spectral_properties"]
        phys_prop_columns = data_from_directory["physical_properties_cols"]

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
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_prop_columns}

    return (data)


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


def load_split_pool(train_size, val_size, pooling_width=1, scale=False,
                    directory_list=None):
    '''Load the spectral and physical data, apply the train/val/test split and pooling of a given width.
    For explanatory data (x), these are loaded, split, pooled, standardised to have zero mean and unit variance in this
    order.

    Returns dictionary with y, x and x pooled data.

    Note that the code load data from a specific directory - you may need to change this inside the function if using a different
    directory structure.

    Also note that the dataset is shuffled randomly with a fixed random state to allow reporducibility of results.'''

    data_all = get_data_from_directory_list(directory_list)

    spectral_data = data_all["spectral_properties"]
    physical_data = data_all["physical_properties"]
    physical_col_names = data_all["physical_properties_cols"]

    print("Spectral data shape", spectral_data.shape)
    print("Physical data shape", physical_data.shape)

    spectral_data, physical_data = shuffle(spectral_data, physical_data, random_state=687)

    if val_size == "max":
        test_size = 0
        val_size = spectral_data.shape[1] - train_size
    else:
        pass

    x_train = spectral_data[: train_size, :]
    x_val = spectral_data[train_size: train_size + val_size, :]
    x_test = spectral_data[train_size + val_size:]

    y_train = physical_data[: train_size, :]
    y_val = physical_data[train_size: train_size + val_size, :]
    y_test = physical_data[train_size + val_size:, :]

    # save to dictionary
    data_dict = dict()

    data_dict["physical_col_names"] = physical_col_names

    data_dict["x"] = dict()

    data_dict["x"]["train"] = x_train
    data_dict["x"]["val"] = x_val
    data_dict["x"]["test"] = x_test

    data_dict["y"] = dict()
    data_dict["y"]["train"] = y_train
    data_dict["y"]["val"] = y_val
    data_dict["y"]["test"] = y_test

    if pooling_width is not None:
        x_train = pooling_for_x(x_train, pooling_width, np.mean)
        x_val = pooling_for_x(x_val, pooling_width, np.mean)
        x_test = pooling_for_x(x_test, pooling_width, np.mean)

    if scale == True:

        scaler = StandardScaler()

        scaler.fit(x_train)

        x_train = pd.DataFrame(scaler.transform(train_df_pooled))
        x_val = pd.DataFrame(scaler.transform(val_df_pooled))
        x_test = pd.DataFrame(scaler.transform(test_df_pooled))

        mean = scaler.mean_
        std = scaler.scale_

    else:
        mean = np.nan
        std = np.nan

    if pooling_width is not None or scale == True:
        data_dict["x_transformed"] = dict()
        data_dict["x_transformed"]["train"] = x_train
        data_dict["x_transformed"]["val"] = x_val
        data_dict["x_transformed"]["test"] = x_test

    # transformation / scaling parameters
    data_dict["parameters"] = dict()
    data_dict["parameters"]["mean"] = mean
    data_dict["parameters"]["std"] = std

    # get wavelengths
    if pooling_width is not None:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01, pooling_width=pooling_width)
        # wavelengths = wavelengths[np.floor(40/pooling_width).astype(int):-np.floor(40/pooling_width).astype(int)]
    else:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01)
        # wavelengths = wavelengths[40:-40]

    data_dict["wavelengths"] = wavelengths

>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
=======
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import h5py
import os
from os import listdir
from sklearn.utils import shuffle

# phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
#              "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
#              "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
#              "halo_orientation_angles1", "halo_orientation_angles2",
#              "ray_orientation_angles1", "ray_orientation_angles2",
#              "ray_number"]





def pooling_for_x(x, bin_size, stats):
    '''For a given dataframe of data, apply pooling based on the statistics supplied.
    If the length of x isn't divisible by bin_size, 0s are appended at the end of x.

    Inputs:
    - x (dataframe/np arry)
    - bin_size: integer
    - stats: object calculating the summary statistics which should be used for pooling.
    For example, "np.mean". '''

    if x.size > 0:
        padding_size = np.ceil(x.shape[1] / bin_size) * bin_size - x.shape[1]

        if bin_size > 0:
            zeros_array = np.zeros((x.shape[0], int(padding_size)))

            x = np.concatenate((x, zeros_array + x[:, -1:]), axis=1)

        x_pooled = x.reshape(x.shape[0], -1, bin_size)
        x_pooled = stats(x_pooled, axis=2)

    # Take care of val set potentially empty
    else:
        x_pooled = np.nan

    return(x_pooled)


def get_data_from_directory(directory):
    '''
    Extract all data from a given directory, split them into physical and spectral properties,
    and return concatenated numpy arrays for both

    Inputs:
    - directory path (string)

    Outputs:
    - dictionary containing the data and column names of physical properties
    '''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    files_list = listdir(directory)

    for data_file in files_list:
        file = h5py.File(directory + '/' + str(data_file), 'r')

        data_from_dile_dict = get_data_from_file(file)

        physical_prop = data_from_dile_dict["physical_properties"]
        spectral_prop = data_from_dile_dict["spectral_properties"]
        phys_col_names = data_from_dile_dict["physical_properties_cols"]

        if phys_prop_joint.size == 0:
            phys_prop_joint = physical_prop
        else:
            phys_prop_joint = np.concatenate((phys_prop_joint, physical_prop), axis=0)

        if spectral_prop_joint.size == 0:
            spectral_prop_joint = spectral_prop
        else:
            spectral_prop_joint = np.concatenate((spectral_prop_joint, spectral_prop), axis=0)

    data = {"physical_properties": phys_prop_joint,
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_col_names}

    return (data)



def get_data_from_file(file):


    # f=h5py.File('halos_0_100.hdf5', 'r')


    phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
                 "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
                 "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
                 "halo_orientation_angles1", "halo_orientation_angles2",
                 "ray_orientation_angles1", "ray_orientation_angles2",
                 "ray_number"]

    spectral_data_all = np.array([])
    y_data_all = np.array([])


    for halo in file.keys():

        spectral_data_halo = file[halo]["spectral_data"]
        spectral_data_halo = spectral_data_halo[:, 40 : -40]
        # print("Trimming spectrum by 40 wavelengths on each side")

        bad_trajectories = file[halo]["bad_trajectories"]
        zero_length = file[halo]["zero_length"]
        black_holes = file[halo]["PartType5"]
        physical_data = file[halo]["physical_data"]

        if bad_trajectories.size == 0:
            bad_trajectories = np.zeros((physical_data.shape[0],1))
        else:
            bad_trajectories = np.ones((physical_data.shape[0],1))* bad_trajectories.size


        if zero_length.size == 0: zero_length = np.zeros((physical_data.shape[0],1))


        y_data_halo = np.concatenate( [bad_trajectories, zero_length, physical_data], axis = 1 )
        y_data_df_halo = pd.DataFrame(y_data_halo, columns = phys_col_names )


        if spectral_data_all.size == 0:
            spectral_data_all = spectral_data_halo
        else:
            spectral_data_all = np.concatenate((spectral_data_all, spectral_data_halo), axis=0)

        if y_data_all.size == 0:
            y_data_all = y_data_halo
        else:
            y_data_all = np.concatenate((y_data_all, y_data_halo), axis=0)


    data = {"physical_properties": y_data_all,
            "spectral_properties": spectral_data_all,
           "physical_properties_cols": phys_col_names}

    return( data )


def get_data_from_directory_list(directory_list):
    '''Get all spectral and physical data contained in a list of directories

    Output:
    - directory with phys and spectral properties under different keys. Data returned as numpy arrays (dict. values)'''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    for directory in directory_list:
        data_from_directory = get_data_from_directory(directory)

        physical_prop = data_from_directory["physical_properties"]
        spectral_prop = data_from_directory["spectral_properties"]
        phys_prop_columns = data_from_directory["physical_properties_cols"]

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
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_prop_columns}

    return (data)


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


def load_split_pool(train_size, val_size, pooling_width=1, scale=False,
                    directory_list=None):
    '''Load the spectral and physical data, apply the train/val/test split and pooling of a given width.
    For explanatory data (x), these are loaded, split, pooled, standardised to have zero mean and unit variance in this
    order.

    Returns dictionary with y, x and x pooled data.

    Note that the code load data from a specific directory - you may need to change this inside the function if using a different
    directory structure.

    Also note that the dataset is shuffled randomly with a fixed random state to allow reporducibility of results.'''

    data_all = get_data_from_directory_list(directory_list)

    spectral_data = data_all["spectral_properties"]
    physical_data = data_all["physical_properties"]
    physical_col_names = data_all["physical_properties_cols"]

    print("Spectral data shape", spectral_data.shape)
    print("Physical data shape", physical_data.shape)

    spectral_data, physical_data = shuffle(spectral_data, physical_data, random_state=687)

    if val_size == "max":
        test_size = 0
        val_size = spectral_data.shape[1] - train_size
    else:
        pass

    x_train = spectral_data[: train_size, :]
    x_val = spectral_data[train_size: train_size + val_size, :]
    x_test = spectral_data[train_size + val_size:]

    y_train = physical_data[: train_size, :]
    y_val = physical_data[train_size: train_size + val_size, :]
    y_test = physical_data[train_size + val_size:, :]

    # save to dictionary
    data_dict = dict()

    data_dict["physical_col_names"] = physical_col_names

    data_dict["x"] = dict()

    data_dict["x"]["train"] = x_train
    data_dict["x"]["val"] = x_val
    data_dict["x"]["test"] = x_test

    data_dict["y"] = dict()
    data_dict["y"]["train"] = y_train
    data_dict["y"]["val"] = y_val
    data_dict["y"]["test"] = y_test

    if pooling_width is not None:
        x_train = pooling_for_x(x_train, pooling_width, np.mean)
        x_val = pooling_for_x(x_val, pooling_width, np.mean)
        x_test = pooling_for_x(x_test, pooling_width, np.mean)

    if scale == True:

        scaler = StandardScaler()

        scaler.fit(x_train)

        x_train = pd.DataFrame(scaler.transform(train_df_pooled))
        x_val = pd.DataFrame(scaler.transform(val_df_pooled))
        x_test = pd.DataFrame(scaler.transform(test_df_pooled))

        mean = scaler.mean_
        std = scaler.scale_

    else:
        mean = np.nan
        std = np.nan

    if pooling_width is not None or scale == True:
        data_dict["x_transformed"] = dict()
        data_dict["x_transformed"]["train"] = x_train
        data_dict["x_transformed"]["val"] = x_val
        data_dict["x_transformed"]["test"] = x_test

    # transformation / scaling parameters
    data_dict["parameters"] = dict()
    data_dict["parameters"]["mean"] = mean
    data_dict["parameters"]["std"] = std

    # get wavelengths
    if pooling_width is not None:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01, pooling_width=pooling_width)
        # wavelengths = wavelengths[np.floor(40/pooling_width).astype(int):-np.floor(40/pooling_width).astype(int)]
    else:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01)
        # wavelengths = wavelengths[40:-40]

    data_dict["wavelengths"] = wavelengths

>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
=======
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import h5py
import os
from os import listdir
from sklearn.utils import shuffle

# phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
#              "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
#              "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
#              "halo_orientation_angles1", "halo_orientation_angles2",
#              "ray_orientation_angles1", "ray_orientation_angles2",
#              "ray_number"]





def pooling_for_x(x, bin_size, stats):
    '''For a given dataframe of data, apply pooling based on the statistics supplied.
    If the length of x isn't divisible by bin_size, 0s are appended at the end of x.

    Inputs:
    - x (dataframe/np arry)
    - bin_size: integer
    - stats: object calculating the summary statistics which should be used for pooling.
    For example, "np.mean". '''

    if x.size > 0:
        padding_size = np.ceil(x.shape[1] / bin_size) * bin_size - x.shape[1]

        if bin_size > 0:
            zeros_array = np.zeros((x.shape[0], int(padding_size)))

            x = np.concatenate((x, zeros_array + x[:, -1:]), axis=1)

        x_pooled = x.reshape(x.shape[0], -1, bin_size)
        x_pooled = stats(x_pooled, axis=2)

    # Take care of val set potentially empty
    else:
        x_pooled = np.nan

    return(x_pooled)


def get_data_from_directory(directory):
    '''
    Extract all data from a given directory, split them into physical and spectral properties,
    and return concatenated numpy arrays for both

    Inputs:
    - directory path (string)

    Outputs:
    - dictionary containing the data and column names of physical properties
    '''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    files_list = listdir(directory)

    for data_file in files_list:
        file = h5py.File(directory + '/' + str(data_file), 'r')

        data_from_dile_dict = get_data_from_file(file)

        physical_prop = data_from_dile_dict["physical_properties"]
        spectral_prop = data_from_dile_dict["spectral_properties"]
        phys_col_names = data_from_dile_dict["physical_properties_cols"]

        if phys_prop_joint.size == 0:
            phys_prop_joint = physical_prop
        else:
            phys_prop_joint = np.concatenate((phys_prop_joint, physical_prop), axis=0)

        if spectral_prop_joint.size == 0:
            spectral_prop_joint = spectral_prop
        else:
            spectral_prop_joint = np.concatenate((spectral_prop_joint, spectral_prop), axis=0)

    data = {"physical_properties": phys_prop_joint,
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_col_names}

    return (data)



def get_data_from_file(file):


    # f=h5py.File('halos_0_100.hdf5', 'r')


    phys_col_names = ["bad_trajectories", "zero_length", "impact_param", "r_v", "stellar_radius", "circularity",
                 "halo_unit_vector1", "halo_unit_vector2","halo_unit_vector3",
                 "ray_unit_vector1", "ray_unit_vector2", "ray_unit_vector3",
                 "halo_orientation_angles1", "halo_orientation_angles2",
                 "ray_orientation_angles1", "ray_orientation_angles2",
                 "ray_number"]

    spectral_data_all = np.array([])
    y_data_all = np.array([])


    for halo in file.keys():

        spectral_data_halo = file[halo]["spectral_data"]
        spectral_data_halo = spectral_data_halo[:, 40 : -40]
        # print("Trimming spectrum by 40 wavelengths on each side")

        bad_trajectories = file[halo]["bad_trajectories"]
        zero_length = file[halo]["zero_length"]
        black_holes = file[halo]["PartType5"]
        physical_data = file[halo]["physical_data"]

        if bad_trajectories.size == 0:
            bad_trajectories = np.zeros((physical_data.shape[0],1))
        else:
            bad_trajectories = np.ones((physical_data.shape[0],1))* bad_trajectories.size


        if zero_length.size == 0: zero_length = np.zeros((physical_data.shape[0],1))


        y_data_halo = np.concatenate( [bad_trajectories, zero_length, physical_data], axis = 1 )
        y_data_df_halo = pd.DataFrame(y_data_halo, columns = phys_col_names )


        if spectral_data_all.size == 0:
            spectral_data_all = spectral_data_halo
        else:
            spectral_data_all = np.concatenate((spectral_data_all, spectral_data_halo), axis=0)

        if y_data_all.size == 0:
            y_data_all = y_data_halo
        else:
            y_data_all = np.concatenate((y_data_all, y_data_halo), axis=0)


    data = {"physical_properties": y_data_all,
            "spectral_properties": spectral_data_all,
           "physical_properties_cols": phys_col_names}

    return( data )


def get_data_from_directory_list(directory_list):
    '''Get all spectral and physical data contained in a list of directories

    Output:
    - directory with phys and spectral properties under different keys. Data returned as numpy arrays (dict. values)'''

    phys_prop_joint = np.array([])
    spectral_prop_joint = np.array([])

    for directory in directory_list:
        data_from_directory = get_data_from_directory(directory)

        physical_prop = data_from_directory["physical_properties"]
        spectral_prop = data_from_directory["spectral_properties"]
        phys_prop_columns = data_from_directory["physical_properties_cols"]

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
            "spectral_properties": spectral_prop_joint,
            "physical_properties_cols": phys_prop_columns}

    return (data)


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


def load_split_pool(train_size, val_size, pooling_width=1, scale=False,
                    directory_list=None):
    '''Load the spectral and physical data, apply the train/val/test split and pooling of a given width.
    For explanatory data (x), these are loaded, split, pooled, standardised to have zero mean and unit variance in this
    order.

    Returns dictionary with y, x and x pooled data.

    Note that the code load data from a specific directory - you may need to change this inside the function if using a different
    directory structure.

    Also note that the dataset is shuffled randomly with a fixed random state to allow reporducibility of results.'''

    data_all = get_data_from_directory_list(directory_list)

    spectral_data = data_all["spectral_properties"]
    physical_data = data_all["physical_properties"]
    physical_col_names = data_all["physical_properties_cols"]

    print("Spectral data shape", spectral_data.shape)
    print("Physical data shape", physical_data.shape)

    spectral_data, physical_data = shuffle(spectral_data, physical_data, random_state=687)

    if val_size == "max":
        test_size = 0
        val_size = spectral_data.shape[1] - train_size
    else:
        pass

    x_train = spectral_data[: train_size, :]
    x_val = spectral_data[train_size: train_size + val_size, :]
    x_test = spectral_data[train_size + val_size:]

    y_train = physical_data[: train_size, :]
    y_val = physical_data[train_size: train_size + val_size, :]
    y_test = physical_data[train_size + val_size:, :]

    # save to dictionary
    data_dict = dict()

    data_dict["physical_col_names"] = physical_col_names

    data_dict["x"] = dict()

    data_dict["x"]["train"] = x_train
    data_dict["x"]["val"] = x_val
    data_dict["x"]["test"] = x_test

    data_dict["y"] = dict()
    data_dict["y"]["train"] = y_train
    data_dict["y"]["val"] = y_val
    data_dict["y"]["test"] = y_test

    if pooling_width is not None:
        x_train = pooling_for_x(x_train, pooling_width, np.mean)
        x_val = pooling_for_x(x_val, pooling_width, np.mean)
        x_test = pooling_for_x(x_test, pooling_width, np.mean)

    if scale == True:

        scaler = StandardScaler()

        scaler.fit(x_train)

        x_train = pd.DataFrame(scaler.transform(train_df_pooled))
        x_val = pd.DataFrame(scaler.transform(val_df_pooled))
        x_test = pd.DataFrame(scaler.transform(test_df_pooled))

        mean = scaler.mean_
        std = scaler.scale_

    else:
        mean = np.nan
        std = np.nan

    if pooling_width is not None or scale == True:
        data_dict["x_transformed"] = dict()
        data_dict["x_transformed"]["train"] = x_train
        data_dict["x_transformed"]["val"] = x_val
        data_dict["x_transformed"]["test"] = x_test

    # transformation / scaling parameters
    data_dict["parameters"] = dict()
    data_dict["parameters"]["mean"] = mean
    data_dict["parameters"]["std"] = std

    # get wavelengths
    if pooling_width is not None:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01, pooling_width=pooling_width)
        # wavelengths = wavelengths[np.floor(40/pooling_width).astype(int):-np.floor(40/pooling_width).astype(int)]
    else:
        wavelengths = get_wavelength(start=900.4, stop=2100.001-0.4, step=0.01)
        # wavelengths = wavelengths[40:-40]

    data_dict["wavelengths"] = wavelengths

>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
    return (data_dict)