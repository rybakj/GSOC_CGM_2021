<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os
import pickle

import itertools


def plot_pairplot(df, vars, hue_var, cmap_scheme, scatter_colour="dodgerblue", log=False):
    '''
    Plot 3D plots: two explanatory vairables are on x and y axis and the response is colored. Along diagonal (i.e. x and y vars are equal), plot scatter with response on y axis and explanatory variable
    on x axis. This is very similar to "sns.pairplot" with "hue" option selected, but allows for continuous responses (= hue variables). Also differs in diagonal plots.

    Inputs:
    - df: datafram with all data.
    - vars: explanatory vars - column names of "df".
    - hue_var: response variable, i.e. variable basedon which points will be colored.
    - cmap_scheme: colour scheme.
    - scatter_colour: what color to use on 2D plots that appear on the diagonal.
    - log: boolean, if True, take logarithm of base 10 of the "hue_var" (i.e. vaiable based on which the colouring is done).

    Outputs:
    - figure
    '''

    # keep track of variables already plotted
    included_list = list()

    plot_df = df.loc[:, vars].copy()
    # form square grid for plots
    fig, ax = plt.subplots(len(vars), len(vars), figsize=(16, 16))
    # initialise row and column indices
    r = 0
    c = 0
    # take log or not
    if log == True:
        hue_series = np.log10(df[hue_var])
    else:
        hue_series = df[hue_var]
    # iterate through combinations of vars
    for i in itertools.product(vars, vars):
        # if variable against itslef, plot scatter (2D plot of response var)
        if i[0] == i[1]:
            ax[r, c].scatter(x=plot_df[i[0]], y=hue_series,
                             cmap=cmap_scheme, s=12, alpha=0.8, c=scatter_colour)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(hue_var)

        # if vars differ, plot 3D plot of response var
        if i[0] != i[1]:
            included_list.append(
                i[0])  # add first var to the list as all vars will be plotted aginst it after outer loop is over

            sc = ax[r, c].scatter(x=plot_df[i[0]], y=plot_df[i[1]], c=hue_series,
                                  cmap=cmap_scheme, s=12)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(i[1])

        # if all columns in a given row are already plotted, move to the next row
        if c == len(vars) - 1:
            r += 1
            c = 0
        # otherwise plot to the next column in the same row
        else:
            c += 1
    # make it prettier
    fig.subplots_adjust(bottom=0, right=1.3, top=0.9)
    cax = plt.axes([1.35, 0.54, 0.01, 0.36])
    fig.colorbar(sc, cax=cax);

    return (fig)


def ols_backward_elimination(Y, X, X_columns):
    '''
    Backward search using R-squared as a comparison metric.

    Specifically:
    1. For a given response and explanatory variables, fit the full linear regression model (with intercept).
    2. Drop the variable which produces the lowest decrease in R-squared. (Constant is never dropped).
    3. Repeat until only constant is left (i.e. all variables have been dropped).

    Inputs:
    - Y: pd.Series or np.array of target variable
    - X: expanatory variables (daaframe or np.array)
    - X_columns: names of variables (in order in which they appear in X). This is used to produce the sequence at whih variables have been eliminated.

    Returns:
    - r2_list: list of R-squared at successive eliminations (order corresponds to "features_eliminated").
    - features_eliminated: list of eliminated features (the first feature in the list is the one that was eliminated the first, so the last important one).
    - r2_adj_list: list of adjusted R-squared at successive eliminations (order corresponds to "features_eliminated").
    '''

    # initialise outputs
    features_eliminated = list()
    r2_list = list()
    r2_adj_list = list()

    # add intercept/constant
    X = sm.add_constant(X)
    # form dataframe of expl variables (this will be redouced at each iteration by dropping the least important variable, as measured by R-squared).
    X_reduced = pd.DataFrame(X, columns=X_columns.insert(item="c", loc=0)).copy()

    # Iterate until only constant is left in X (i.e. until all explanatory variables have been dropped).
    while len(X_reduced.columns) > 1:
        # initialise max r-sqaured at -inf
        max_r2 = - np.inf

        # iterate through features (except the constant)
        for feature in X_reduced.columns[1:]:
            # drop feature
            X = X_reduced.drop(feature, axis=1)
            # estimate
            model = sm.OLS(Y, X)
            results = model.fit()
            r2 = results.rsquared  # .mse_resid # .rsquared
            # if eliminating this feautre gives the highest r2 so far, record it
            if r2 > max_r2:
                max_r2 = r2
                min_feature = feature
                # print(min_feature)
            else:
                pass

        # keep track of eliminated features
        features_eliminated.append(min_feature)
        # drop the least important feature
        X_reduced.drop(min_feature, axis=1, inplace=True)

        # ovewrite X, re-fit etc
        X = X_reduced
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()

        # keep track of both R-squared and adjusted R-squared
        r2_list.append(results.rsquared)
        r2_adj_list.append(results.rsquared_adj)

    return (r2_list, features_eliminated, r2_adj_list)


def dump_object(object_to_save, location, filename):
    '''Saves a given object under a given name in a given location / direcotry (if such directory
     does not exist the function creates it.

     Inputs:
     - object_to_save: object to save.
     - location: directory/folder where to save the object
     - filename: how to name the saved file

     Returns:
     - "": empty. The function performs action, returns nothing.
     '''
    if not os.path.isdir(location):
        try:
            os.mkdir(location)
        except:
            print("Failed")

    file_path = location + "/" + filename

    with open(file_path, 'wb') as dump_location:
        pickle.dump(object_to_save, dump_location)

    return ()


# def runs_of_ones(bits):
#     for bit, group in itertools.groupby(bits):
#         if bit:
#             yield sum(group)

def runs_of_ones_list(bits):
    '''
    For a list of boolean values (or 0/1), count the number of consecutive True (or 1). The count is reset once False is encountered.
    '''
    return [sum(g) for b, g in itertools.groupby(bits) if b]


def plot_reconstruction(wavelengths, reconstructions, original, spectrum_number,
                        col_original="Orange", col_reconstr="darkred",
                        model_name=None, title=None):
    '''
    Plots original spectrum and it reconstruction.

    Inputs:
    - wavelengths: numpy array of wavelengths
    - reconstructions: numpy array of reconstructions (2D array)
    - original: numpy array (2D) of original spectra
    - spectrum_number (integer): which spectrum to plots - defines a row index applied to "reconstructions" and "original"
    - col_oringal (string): color of the original spectrum
    - col_reconstr (string): color of the reconstructed spectrum
    - model_name (string): name of the model used for reconstructions (used for plot legend)
    - titel (string): plot title

    Returns:
    - empty (the function just pproduces a plot, returns nothing / empty object)
    '''

    if model_name != None:
        model_label = "(" + str(model_name) + ")"
    else:
        model_label = ""

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
    ax[0].plot(wavelengths, original[spectrum_number, :], label="Original spectrum", linewidth=0.75,
               color=col_original)
    ax[0].legend(fontsize=14)
    ax[0].set_xlabel("Wavelength", fontsize=14)
    ax[0].set_ylabel("Intensity", fontsize=14)
    if title is not None: ax[0].set_title(title, fontsize=16)

    ax[1].plot(wavelengths, reconstructions[spectrum_number, :], label="Reconstructed spectrum " + str(model_label),
               linewidth=0.75,
               color="dodgerblue")
    ax[1].legend(fontsize=14)
    ax[1].set_xlabel("Wavelength", fontsize=14)
    ax[1].set_ylabel("Intensity", fontsize=14)

    ax[0].set_ylim([0, 1.25])
    ax[1].set_ylim([0, 1.25])

    return ()



def scale_down(x, threshold, x_std=None):
    '''
    Based on a given threshold, identify spectral lines as lines more than threshold*std from the mean.
    Return total volume of the signal lines and the position of noise and signal lines.

    Inputs:
    - x: numpy array of data
    - threshold (float): threshold for how many standard deviations away from the mean constitutes a signal
    - x_std: stadanrd deviation to use. If None, x_std is set to x.std()

    Returns
    - x_std
    - total_flux: total volume of signal spectral lines
    - noise: location of noise lines
    - signals: location of signal lines

    '''
    if (x_std == None):
        x_std = x.std()
    else:
        pass

    signals = (np.abs(x - 1) > (threshold * x_std)) * (x < 1)
    noise = np.abs(signals - 1)

    # total flux: integrate "signal" wavelengths between line 1 and spectral line
    total_flux = np.multiply(np.abs(x - 1), signals).sum(axis=1)


    return (x_std, total_flux, noise, signals)
=======
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os
import pickle

import itertools


def plot_pairplot(df, vars, hue_var, cmap_scheme, scatter_colour="dodgerblue", log=False):
    '''
    Plot 3D plots: two explanatory vairables are on x and y axis and the response is colored. Along diagonal (i.e. x and y vars are equal), plot scatter with response on y axis and explanatory variable
    on x axis. This is very similar to "sns.pairplot" with "hue" option selected, but allows for continuous responses (= hue variables). Also differs in diagonal plots.

    Inputs:
    - df: datafram with all data.
    - vars: explanatory vars - column names of "df".
    - hue_var: response variable, i.e. variable basedon which points will be colored.
    - cmap_scheme: colour scheme.
    - scatter_colour: what color to use on 2D plots that appear on the diagonal.
    - log: boolean, if True, take logarithm of base 10 of the "hue_var" (i.e. vaiable based on which the colouring is done).

    Outputs:
    - figure
    '''

    # keep track of variables already plotted
    included_list = list()

    plot_df = df.loc[:, vars].copy()
    # form square grid for plots
    fig, ax = plt.subplots(len(vars), len(vars), figsize=(16, 16))
    # initialise row and column indices
    r = 0
    c = 0
    # take log or not
    if log == True:
        hue_series = np.log10(df[hue_var])
    else:
        hue_series = df[hue_var]
    # iterate through combinations of vars
    for i in itertools.product(vars, vars):
        # if variable against itslef, plot scatter (2D plot of response var)
        if i[0] == i[1]:
            ax[r, c].scatter(x=plot_df[i[0]], y=hue_series,
                             cmap=cmap_scheme, s=12, alpha=0.8, c=scatter_colour)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(hue_var)

        # if vars differ, plot 3D plot of response var
        if i[0] != i[1]:
            included_list.append(
                i[0])  # add first var to the list as all vars will be plotted aginst it after outer loop is over

            sc = ax[r, c].scatter(x=plot_df[i[0]], y=plot_df[i[1]], c=hue_series,
                                  cmap=cmap_scheme, s=12)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(i[1])

        # if all columns in a given row are already plotted, move to the next row
        if c == len(vars) - 1:
            r += 1
            c = 0
        # otherwise plot to the next column in the same row
        else:
            c += 1
    # make it prettier
    fig.subplots_adjust(bottom=0, right=1.3, top=0.9)
    cax = plt.axes([1.35, 0.54, 0.01, 0.36])
    fig.colorbar(sc, cax=cax);

    return (fig)


def ols_backward_elimination(Y, X, X_columns):
    '''
    Backward search using R-squared as a comparison metric.

    Specifically:
    1. For a given response and explanatory variables, fit the full linear regression model (with intercept).
    2. Drop the variable which produces the lowest decrease in R-squared. (Constant is never dropped).
    3. Repeat until only constant is left (i.e. all variables have been dropped).

    Inputs:
    - Y: pd.Series or np.array of target variable
    - X: expanatory variables (daaframe or np.array)
    - X_columns: names of variables (in order in which they appear in X). This is used to produce the sequence at whih variables have been eliminated.

    Returns:
    - r2_list: list of R-squared at successive eliminations (order corresponds to "features_eliminated").
    - features_eliminated: list of eliminated features (the first feature in the list is the one that was eliminated the first, so the last important one).
    - r2_adj_list: list of adjusted R-squared at successive eliminations (order corresponds to "features_eliminated").
    '''

    # initialise outputs
    features_eliminated = list()
    r2_list = list()
    r2_adj_list = list()

    # add intercept/constant
    X = sm.add_constant(X)
    # form dataframe of expl variables (this will be redouced at each iteration by dropping the least important variable, as measured by R-squared).
    X_reduced = pd.DataFrame(X, columns=X_columns.insert(item="c", loc=0)).copy()

    # Iterate until only constant is left in X (i.e. until all explanatory variables have been dropped).
    while len(X_reduced.columns) > 1:
        # initialise max r-sqaured at -inf
        max_r2 = - np.inf

        # iterate through features (except the constant)
        for feature in X_reduced.columns[1:]:
            # drop feature
            X = X_reduced.drop(feature, axis=1)
            # estimate
            model = sm.OLS(Y, X)
            results = model.fit()
            r2 = results.rsquared  # .mse_resid # .rsquared
            # if eliminating this feautre gives the highest r2 so far, record it
            if r2 > max_r2:
                max_r2 = r2
                min_feature = feature
                # print(min_feature)
            else:
                pass

        # keep track of eliminated features
        features_eliminated.append(min_feature)
        # drop the least important feature
        X_reduced.drop(min_feature, axis=1, inplace=True)

        # ovewrite X, re-fit etc
        X = X_reduced
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()

        # keep track of both R-squared and adjusted R-squared
        r2_list.append(results.rsquared)
        r2_adj_list.append(results.rsquared_adj)

    return (r2_list, features_eliminated, r2_adj_list)


def dump_object(object_to_save, location, filename):
    '''Saves a given object under a given name in a given location / direcotry (if such directory
     does not exist the function creates it.

     Inputs:
     - object_to_save: object to save.
     - location: directory/folder where to save the object
     - filename: how to name the saved file

     Returns:
     - "": empty. The function performs action, returns nothing.
     '''
    if not os.path.isdir(location):
        try:
            os.mkdir(location)
        except:
            print("Failed")

    file_path = location + "/" + filename

    with open(file_path, 'wb') as dump_location:
        pickle.dump(object_to_save, dump_location)

    return ()


# def runs_of_ones(bits):
#     for bit, group in itertools.groupby(bits):
#         if bit:
#             yield sum(group)

def runs_of_ones_list(bits):
    '''
    For a list of boolean values (or 0/1), count the number of consecutive True (or 1). The count is reset once False is encountered.
    '''
    return [sum(g) for b, g in itertools.groupby(bits) if b]


def plot_reconstruction(wavelengths, reconstructions, original, spectrum_number,
                        col_original="Orange", col_reconstr="darkred",
                        model_name=None, title=None):
    '''
    Plots original spectrum and it reconstruction.

    Inputs:
    - wavelengths: numpy array of wavelengths
    - reconstructions: numpy array of reconstructions (2D array)
    - original: numpy array (2D) of original spectra
    - spectrum_number (integer): which spectrum to plots - defines a row index applied to "reconstructions" and "original"
    - col_oringal (string): color of the original spectrum
    - col_reconstr (string): color of the reconstructed spectrum
    - model_name (string): name of the model used for reconstructions (used for plot legend)
    - titel (string): plot title

    Returns:
    - empty (the function just pproduces a plot, returns nothing / empty object)
    '''

    if model_name != None:
        model_label = "(" + str(model_name) + ")"
    else:
        model_label = ""

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
    ax[0].plot(wavelengths, original[spectrum_number, :], label="Original spectrum", linewidth=0.75,
               color=col_original)
    ax[0].legend(fontsize=14)
    ax[0].set_xlabel("Wavelength", fontsize=14)
    ax[0].set_ylabel("Intensity", fontsize=14)
    if title is not None: ax[0].set_title(title, fontsize=16)

    ax[1].plot(wavelengths, reconstructions[spectrum_number, :], label="Reconstructed spectrum " + str(model_label),
               linewidth=0.75,
               color="dodgerblue")
    ax[1].legend(fontsize=14)
    ax[1].set_xlabel("Wavelength", fontsize=14)
    ax[1].set_ylabel("Intensity", fontsize=14)

    ax[0].set_ylim([0, 1.25])
    ax[1].set_ylim([0, 1.25])

    return ()



def scale_down(x, threshold, x_std=None):
    '''
    Based on a given threshold, identify spectral lines as lines more than threshold*std from the mean.
    Return total volume of the signal lines and the position of noise and signal lines.

    Inputs:
    - x: numpy array of data
    - threshold (float): threshold for how many standard deviations away from the mean constitutes a signal
    - x_std: stadanrd deviation to use. If None, x_std is set to x.std()

    Returns
    - x_std
    - total_flux: total volume of signal spectral lines
    - noise: location of noise lines
    - signals: location of signal lines

    '''
    if (x_std == None):
        x_std = x.std()
    else:
        pass

    signals = (np.abs(x - 1) > (threshold * x_std)) * (x < 1)
    noise = np.abs(signals - 1)

    # total flux: integrate "signal" wavelengths between line 1 and spectral line
    total_flux = np.multiply(np.abs(x - 1), signals).sum(axis=1)


    return (x_std, total_flux, noise, signals)
>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
=======
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os
import pickle

import itertools


def plot_pairplot(df, vars, hue_var, cmap_scheme, scatter_colour="dodgerblue", log=False):
    '''
    Plot 3D plots: two explanatory vairables are on x and y axis and the response is colored. Along diagonal (i.e. x and y vars are equal), plot scatter with response on y axis and explanatory variable
    on x axis. This is very similar to "sns.pairplot" with "hue" option selected, but allows for continuous responses (= hue variables). Also differs in diagonal plots.

    Inputs:
    - df: datafram with all data.
    - vars: explanatory vars - column names of "df".
    - hue_var: response variable, i.e. variable basedon which points will be colored.
    - cmap_scheme: colour scheme.
    - scatter_colour: what color to use on 2D plots that appear on the diagonal.
    - log: boolean, if True, take logarithm of base 10 of the "hue_var" (i.e. vaiable based on which the colouring is done).

    Outputs:
    - figure
    '''

    # keep track of variables already plotted
    included_list = list()

    plot_df = df.loc[:, vars].copy()
    # form square grid for plots
    fig, ax = plt.subplots(len(vars), len(vars), figsize=(16, 16))
    # initialise row and column indices
    r = 0
    c = 0
    # take log or not
    if log == True:
        hue_series = np.log10(df[hue_var])
    else:
        hue_series = df[hue_var]
    # iterate through combinations of vars
    for i in itertools.product(vars, vars):
        # if variable against itslef, plot scatter (2D plot of response var)
        if i[0] == i[1]:
            ax[r, c].scatter(x=plot_df[i[0]], y=hue_series,
                             cmap=cmap_scheme, s=12, alpha=0.8, c=scatter_colour)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(hue_var)

        # if vars differ, plot 3D plot of response var
        if i[0] != i[1]:
            included_list.append(
                i[0])  # add first var to the list as all vars will be plotted aginst it after outer loop is over

            sc = ax[r, c].scatter(x=plot_df[i[0]], y=plot_df[i[1]], c=hue_series,
                                  cmap=cmap_scheme, s=12)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(i[1])

        # if all columns in a given row are already plotted, move to the next row
        if c == len(vars) - 1:
            r += 1
            c = 0
        # otherwise plot to the next column in the same row
        else:
            c += 1
    # make it prettier
    fig.subplots_adjust(bottom=0, right=1.3, top=0.9)
    cax = plt.axes([1.35, 0.54, 0.01, 0.36])
    fig.colorbar(sc, cax=cax);

    return (fig)


def ols_backward_elimination(Y, X, X_columns):
    '''
    Backward search using R-squared as a comparison metric.

    Specifically:
    1. For a given response and explanatory variables, fit the full linear regression model (with intercept).
    2. Drop the variable which produces the lowest decrease in R-squared. (Constant is never dropped).
    3. Repeat until only constant is left (i.e. all variables have been dropped).

    Inputs:
    - Y: pd.Series or np.array of target variable
    - X: expanatory variables (daaframe or np.array)
    - X_columns: names of variables (in order in which they appear in X). This is used to produce the sequence at whih variables have been eliminated.

    Returns:
    - r2_list: list of R-squared at successive eliminations (order corresponds to "features_eliminated").
    - features_eliminated: list of eliminated features (the first feature in the list is the one that was eliminated the first, so the last important one).
    - r2_adj_list: list of adjusted R-squared at successive eliminations (order corresponds to "features_eliminated").
    '''

    # initialise outputs
    features_eliminated = list()
    r2_list = list()
    r2_adj_list = list()

    # add intercept/constant
    X = sm.add_constant(X)
    # form dataframe of expl variables (this will be redouced at each iteration by dropping the least important variable, as measured by R-squared).
    X_reduced = pd.DataFrame(X, columns=X_columns.insert(item="c", loc=0)).copy()

    # Iterate until only constant is left in X (i.e. until all explanatory variables have been dropped).
    while len(X_reduced.columns) > 1:
        # initialise max r-sqaured at -inf
        max_r2 = - np.inf

        # iterate through features (except the constant)
        for feature in X_reduced.columns[1:]:
            # drop feature
            X = X_reduced.drop(feature, axis=1)
            # estimate
            model = sm.OLS(Y, X)
            results = model.fit()
            r2 = results.rsquared  # .mse_resid # .rsquared
            # if eliminating this feautre gives the highest r2 so far, record it
            if r2 > max_r2:
                max_r2 = r2
                min_feature = feature
                # print(min_feature)
            else:
                pass

        # keep track of eliminated features
        features_eliminated.append(min_feature)
        # drop the least important feature
        X_reduced.drop(min_feature, axis=1, inplace=True)

        # ovewrite X, re-fit etc
        X = X_reduced
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()

        # keep track of both R-squared and adjusted R-squared
        r2_list.append(results.rsquared)
        r2_adj_list.append(results.rsquared_adj)

    return (r2_list, features_eliminated, r2_adj_list)


def dump_object(object_to_save, location, filename):
    '''Saves a given object under a given name in a given location / direcotry (if such directory
     does not exist the function creates it.

     Inputs:
     - object_to_save: object to save.
     - location: directory/folder where to save the object
     - filename: how to name the saved file

     Returns:
     - "": empty. The function performs action, returns nothing.
     '''
    if not os.path.isdir(location):
        try:
            os.mkdir(location)
        except:
            print("Failed")

    file_path = location + "/" + filename

    with open(file_path, 'wb') as dump_location:
        pickle.dump(object_to_save, dump_location)

    return ()


# def runs_of_ones(bits):
#     for bit, group in itertools.groupby(bits):
#         if bit:
#             yield sum(group)

def runs_of_ones_list(bits):
    '''
    For a list of boolean values (or 0/1), count the number of consecutive True (or 1). The count is reset once False is encountered.
    '''
    return [sum(g) for b, g in itertools.groupby(bits) if b]


def plot_reconstruction(wavelengths, reconstructions, original, spectrum_number,
                        col_original="Orange", col_reconstr="darkred",
                        model_name=None, title=None):
    '''
    Plots original spectrum and it reconstruction.

    Inputs:
    - wavelengths: numpy array of wavelengths
    - reconstructions: numpy array of reconstructions (2D array)
    - original: numpy array (2D) of original spectra
    - spectrum_number (integer): which spectrum to plots - defines a row index applied to "reconstructions" and "original"
    - col_oringal (string): color of the original spectrum
    - col_reconstr (string): color of the reconstructed spectrum
    - model_name (string): name of the model used for reconstructions (used for plot legend)
    - titel (string): plot title

    Returns:
    - empty (the function just pproduces a plot, returns nothing / empty object)
    '''

    if model_name != None:
        model_label = "(" + str(model_name) + ")"
    else:
        model_label = ""

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
    ax[0].plot(wavelengths, original[spectrum_number, :], label="Original spectrum", linewidth=0.75,
               color=col_original)
    ax[0].legend(fontsize=14)
    ax[0].set_xlabel("Wavelength", fontsize=14)
    ax[0].set_ylabel("Intensity", fontsize=14)
    if title is not None: ax[0].set_title(title, fontsize=16)

    ax[1].plot(wavelengths, reconstructions[spectrum_number, :], label="Reconstructed spectrum " + str(model_label),
               linewidth=0.75,
               color="dodgerblue")
    ax[1].legend(fontsize=14)
    ax[1].set_xlabel("Wavelength", fontsize=14)
    ax[1].set_ylabel("Intensity", fontsize=14)

    ax[0].set_ylim([0, 1.25])
    ax[1].set_ylim([0, 1.25])

    return ()



def scale_down(x, threshold, x_std=None):
    '''
    Based on a given threshold, identify spectral lines as lines more than threshold*std from the mean.
    Return total volume of the signal lines and the position of noise and signal lines.

    Inputs:
    - x: numpy array of data
    - threshold (float): threshold for how many standard deviations away from the mean constitutes a signal
    - x_std: stadanrd deviation to use. If None, x_std is set to x.std()

    Returns
    - x_std
    - total_flux: total volume of signal spectral lines
    - noise: location of noise lines
    - signals: location of signal lines

    '''
    if (x_std == None):
        x_std = x.std()
    else:
        pass

    signals = (np.abs(x - 1) > (threshold * x_std)) * (x < 1)
    noise = np.abs(signals - 1)

    # total flux: integrate "signal" wavelengths between line 1 and spectral line
    total_flux = np.multiply(np.abs(x - 1), signals).sum(axis=1)


    return (x_std, total_flux, noise, signals)
>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
=======
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os
import pickle

import itertools


def plot_pairplot(df, vars, hue_var, cmap_scheme, scatter_colour="dodgerblue", log=False):
    '''
    Plot 3D plots: two explanatory vairables are on x and y axis and the response is colored. Along diagonal (i.e. x and y vars are equal), plot scatter with response on y axis and explanatory variable
    on x axis. This is very similar to "sns.pairplot" with "hue" option selected, but allows for continuous responses (= hue variables). Also differs in diagonal plots.

    Inputs:
    - df: datafram with all data.
    - vars: explanatory vars - column names of "df".
    - hue_var: response variable, i.e. variable basedon which points will be colored.
    - cmap_scheme: colour scheme.
    - scatter_colour: what color to use on 2D plots that appear on the diagonal.
    - log: boolean, if True, take logarithm of base 10 of the "hue_var" (i.e. vaiable based on which the colouring is done).

    Outputs:
    - figure
    '''

    # keep track of variables already plotted
    included_list = list()

    plot_df = df.loc[:, vars].copy()
    # form square grid for plots
    fig, ax = plt.subplots(len(vars), len(vars), figsize=(16, 16))
    # initialise row and column indices
    r = 0
    c = 0
    # take log or not
    if log == True:
        hue_series = np.log10(df[hue_var])
    else:
        hue_series = df[hue_var]
    # iterate through combinations of vars
    for i in itertools.product(vars, vars):
        # if variable against itslef, plot scatter (2D plot of response var)
        if i[0] == i[1]:
            ax[r, c].scatter(x=plot_df[i[0]], y=hue_series,
                             cmap=cmap_scheme, s=12, alpha=0.8, c=scatter_colour)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(hue_var)

        # if vars differ, plot 3D plot of response var
        if i[0] != i[1]:
            included_list.append(
                i[0])  # add first var to the list as all vars will be plotted aginst it after outer loop is over

            sc = ax[r, c].scatter(x=plot_df[i[0]], y=plot_df[i[1]], c=hue_series,
                                  cmap=cmap_scheme, s=12)
            ax[r, c].set_xlabel(i[0])
            ax[r, c].set_ylabel(i[1])

        # if all columns in a given row are already plotted, move to the next row
        if c == len(vars) - 1:
            r += 1
            c = 0
        # otherwise plot to the next column in the same row
        else:
            c += 1
    # make it prettier
    fig.subplots_adjust(bottom=0, right=1.3, top=0.9)
    cax = plt.axes([1.35, 0.54, 0.01, 0.36])
    fig.colorbar(sc, cax=cax);

    return (fig)


def ols_backward_elimination(Y, X, X_columns):
    '''
    Backward search using R-squared as a comparison metric.

    Specifically:
    1. For a given response and explanatory variables, fit the full linear regression model (with intercept).
    2. Drop the variable which produces the lowest decrease in R-squared. (Constant is never dropped).
    3. Repeat until only constant is left (i.e. all variables have been dropped).

    Inputs:
    - Y: pd.Series or np.array of target variable
    - X: expanatory variables (daaframe or np.array)
    - X_columns: names of variables (in order in which they appear in X). This is used to produce the sequence at whih variables have been eliminated.

    Returns:
    - r2_list: list of R-squared at successive eliminations (order corresponds to "features_eliminated").
    - features_eliminated: list of eliminated features (the first feature in the list is the one that was eliminated the first, so the last important one).
    - r2_adj_list: list of adjusted R-squared at successive eliminations (order corresponds to "features_eliminated").
    '''

    # initialise outputs
    features_eliminated = list()
    r2_list = list()
    r2_adj_list = list()

    # add intercept/constant
    X = sm.add_constant(X)
    # form dataframe of expl variables (this will be redouced at each iteration by dropping the least important variable, as measured by R-squared).
    X_reduced = pd.DataFrame(X, columns=X_columns.insert(item="c", loc=0)).copy()

    # Iterate until only constant is left in X (i.e. until all explanatory variables have been dropped).
    while len(X_reduced.columns) > 1:
        # initialise max r-sqaured at -inf
        max_r2 = - np.inf

        # iterate through features (except the constant)
        for feature in X_reduced.columns[1:]:
            # drop feature
            X = X_reduced.drop(feature, axis=1)
            # estimate
            model = sm.OLS(Y, X)
            results = model.fit()
            r2 = results.rsquared  # .mse_resid # .rsquared
            # if eliminating this feautre gives the highest r2 so far, record it
            if r2 > max_r2:
                max_r2 = r2
                min_feature = feature
                # print(min_feature)
            else:
                pass

        # keep track of eliminated features
        features_eliminated.append(min_feature)
        # drop the least important feature
        X_reduced.drop(min_feature, axis=1, inplace=True)

        # ovewrite X, re-fit etc
        X = X_reduced
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()

        # keep track of both R-squared and adjusted R-squared
        r2_list.append(results.rsquared)
        r2_adj_list.append(results.rsquared_adj)

    return (r2_list, features_eliminated, r2_adj_list)


def dump_object(object_to_save, location, filename):
    '''Saves a given object under a given name in a given location / direcotry (if such directory
     does not exist the function creates it.

     Inputs:
     - object_to_save: object to save.
     - location: directory/folder where to save the object
     - filename: how to name the saved file

     Returns:
     - "": empty. The function performs action, returns nothing.
     '''
    if not os.path.isdir(location):
        try:
            os.mkdir(location)
        except:
            print("Failed")

    file_path = location + "/" + filename

    with open(file_path, 'wb') as dump_location:
        pickle.dump(object_to_save, dump_location)

    return ()


# def runs_of_ones(bits):
#     for bit, group in itertools.groupby(bits):
#         if bit:
#             yield sum(group)

def runs_of_ones_list(bits):
    '''
    For a list of boolean values (or 0/1), count the number of consecutive True (or 1). The count is reset once False is encountered.
    '''
    return [sum(g) for b, g in itertools.groupby(bits) if b]


def plot_reconstruction(wavelengths, reconstructions, original, spectrum_number,
                        col_original="Orange", col_reconstr="darkred",
                        model_name=None, title=None):
    '''
    Plots original spectrum and it reconstruction.

    Inputs:
    - wavelengths: numpy array of wavelengths
    - reconstructions: numpy array of reconstructions (2D array)
    - original: numpy array (2D) of original spectra
    - spectrum_number (integer): which spectrum to plots - defines a row index applied to "reconstructions" and "original"
    - col_oringal (string): color of the original spectrum
    - col_reconstr (string): color of the reconstructed spectrum
    - model_name (string): name of the model used for reconstructions (used for plot legend)
    - titel (string): plot title

    Returns:
    - empty (the function just pproduces a plot, returns nothing / empty object)
    '''

    if model_name != None:
        model_label = "(" + str(model_name) + ")"
    else:
        model_label = ""

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 12))
    ax[0].plot(wavelengths, original[spectrum_number, :], label="Original spectrum", linewidth=0.75,
               color=col_original)
    ax[0].legend(fontsize=14)
    ax[0].set_xlabel("Wavelength", fontsize=14)
    ax[0].set_ylabel("Intensity", fontsize=14)
    if title is not None: ax[0].set_title(title, fontsize=16)

    ax[1].plot(wavelengths, reconstructions[spectrum_number, :], label="Reconstructed spectrum " + str(model_label),
               linewidth=0.75,
               color="dodgerblue")
    ax[1].legend(fontsize=14)
    ax[1].set_xlabel("Wavelength", fontsize=14)
    ax[1].set_ylabel("Intensity", fontsize=14)

    ax[0].set_ylim([0, 1.25])
    ax[1].set_ylim([0, 1.25])

    return ()



def scale_down(x, threshold, x_std=None):
    '''
    Based on a given threshold, identify spectral lines as lines more than threshold*std from the mean.
    Return total volume of the signal lines and the position of noise and signal lines.

    Inputs:
    - x: numpy array of data
    - threshold (float): threshold for how many standard deviations away from the mean constitutes a signal
    - x_std: stadanrd deviation to use. If None, x_std is set to x.std()

    Returns
    - x_std
    - total_flux: total volume of signal spectral lines
    - noise: location of noise lines
    - signals: location of signal lines

    '''
    if (x_std == None):
        x_std = x.std()
    else:
        pass

    signals = (np.abs(x - 1) > (threshold * x_std)) * (x < 1)
    noise = np.abs(signals - 1)

    # total flux: integrate "signal" wavelengths between line 1 and spectral line
    total_flux = np.multiply(np.abs(x - 1), signals).sum(axis=1)


    return (x_std, total_flux, noise, signals)
>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
