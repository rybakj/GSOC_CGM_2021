<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras import backend as BK

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Concatenate, Conv2D,
                                     UpSampling2D, BatchNormalization)

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


# region Deep Learning models

# region General dl functions

def custom_tanh(x, target_min=0, target_max=None):
    '''
    For given max and min values, return tanh activation with values in this range

    Inputs:
    - x: values to be transformed (np.array)
    - target_min: minimum value of tanh function (float)
    - target_max: max value of tanh function (float)

    Returns:
    - x transformed by tanh activation (np.array)
    '''

    x_02range = BK.tanh(x) + 1  # x mapped into range(0,2)
    scale = (target_max - target_min) / 2.  # calculate target range of transformed x

    return (x_02range * scale + target_min)


def add_layer_to_list(layer_list, filename, hidden_units, activation="relu"):
    '''
    Append layers with valid unit number to a list. Layers with units set to "None" or zero are ignored.

    Inputs:
    - layer_list: list of dense layers to which to append
    - hidden_units (integer or float): numbe rof hidden units for this layer
    - activation: what activation to use

    Return:
    - list with appended layer.
    '''

    if (hidden_units is not None) & (hidden_units != 0):
        layer_list.append(dict(hidden_units=hidden_units, activation=activation))
        filename = filename + "_dense_" + str(hidden_units)
    else:
        pass

    return (layer_list, filename)



def get_checkpoint_callback(checkpoint_filepath_dir, file_name):
    '''
    Create ModelCheckpoint callback object, which saves weights of the best model so far (the model with minimal loss on a validation set).

    Inputs:
    -  checkpoint_filepath: path where to save the model weights

    Returns
    - checkpoint callback (object)
    - filepath used for checkpointing (string)
    '''
    checkpoint_filepath = os.path.join(checkpoint_filepath_dir, file_name)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    return (model_checkpoint_callback, checkpoint_filepath)

# endregion

# region VAE Model

def get_indep_normal_prior(latent_dim):
    '''
    Create mutlivariate normal prior with identity covariance matrix.

    Inputs:
    - latent_dim (integer): dimensionality of the latent space

    Outputs:
    - prior (distribution object)
    '''
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim))

    return (prior)


def get_prior_gauss_mixture(num_modes, latent_dim):
    """
    This function creates an instance of a MixtureSameFamily distribution, specifically a mixture of Gaussian distribution.
    The mixing distribution is uniform with equal probabilities over component distributions.

    Inputs:
    - num_modes (integer): number of modes of mixture distribution.
    - latent_dim (integer): dimension of the latent space.

    Returns:
    - distribution instance with the specified characteristics.
    """

    gm = tfd.MixtureSameFamily(
        # Mixing distribution:
        # Gaussians are equally likely
        mixture_distribution=tfd.Categorical(probs=tf.convert_to_tensor([1 / num_modes] * num_modes, np.float32)),

        # Component distributions:
        # Enforce PD covariance matrix by softplus activation
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(initial_value=tf.random.normal([num_modes, latent_dim]), trainable=True),
            # We need to have scale: NUM_MODES x LATENT_DIM as Gaussian correpsonding to each mode will have
            # a different covariance matrix
            scale_diag=tfp.util.TransformedVariable(initial_value=tf.ones([num_modes, latent_dim]),
                                                    bijector=tfb.Softplus())
        )
    )

    return (gm)


def get_kl_regularizer(prior_distribution, weight=1, use_exact_kl=False):
    """
    This function should create an instance of the KLDivergenceRegularizer for a given prior distribution and KL penalty weight.

    Inputs:
    - prior_distribution (tfd object)
    - weight (float): weight given to KL divergence penalty

    Returns:
    - KLDivergenceRegularizer instance.
    """
    divergence_regularizer = tfpl.KLDivergenceRegularizer(prior_distribution,
                                                          use_exact_kl=use_exact_kl,
                                                          test_points_fn=lambda q: q.sample(15),
                                                          test_points_reduce_axis=None,
                                                          weight=weight)

    return (divergence_regularizer)


def get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim):
    '''
    Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
    - dense_layers: List of dense layers, where layers appear in the list in the desired order.
        Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
        The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

    - kl_regularizer: an instance of KLDivergenceRegularizer
    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

    Outputs:
    - Model instance
    '''

    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(x_train_shape[1],))])

    # add dense unit in reverse order
    for dense_layer in dense_layers:
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    # add dense layer with linear activation to generate parameters of the encoding distribution
    model.add(Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)))

    # add kl divergence regularizer
    model.add(tfpl.MultivariateNormalTriL(event_size=latent_dim,
                                          activity_regularizer=kl_regularizer))

    # print model summary
    # print( model.summary() )

    return (model)


def get_vae_decoder(x_train_shape, dense_layers, latent_dim, output_type, decoding_activation):
    '''
    Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
    Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
    - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
      the decoder and encoder are symmetrical.

        Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).

        The final decoder layer models latent space as a multivariate Normal distribution as is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
    - output_type: (string): Specifies the type of output from decoder. Two options:
        1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
        2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Outputs:
    - Model instance
    '''
    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,))])

    # add dense unit in reverse order
    for dense_layer in reversed(dense_layers):
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    if output_type == "reconstructions":
        # add a layer with custom tanh activation to map back to the input space.
        model.add(Dense(units=x_train_shape[1], name="decoding_layer", activation=decoding_activation))

    elif output_type == "distributions":
        model.add(Dense(x_train_shape[1] * 2)),

        model.add(
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :x_train_shape[1]],
                               scale=tf.keras.activations.softmax(t[..., x_train_shape[1]:]))
                )
            )
        )

    return (model)



def get_vae(x_train_shape, dense_layers, latent_dim, prior, kl_weight, num_modes=None,
            decoder_output_type="reconstructions", decoding_activation=None):
    '''
    Return a compiled variational autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - prior (string): prior distribution on latent space. Allows two options:
          1. "indep_normal" for multivariate nromal distribution with identity covariance matrix.
          2. "gauss_mixture" for Guassian mixture prior.
    - kl_weight: weighted of KL divergence penalty (higher means more penalization of deviations of latent space from prior).
    - num_modes (integer): if prior == "gauss_mixture", specifies the number of modes.
    - decoder_output_type (string): Specifies the type of output from decoder. Two options:
      1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
      2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Returns:
    - vae model
    - encoder object
    - decoder object
    '''
    if prior == "indep_normal":
        prior = get_indep_normal_prior(latent_dim)
        use_exact_kl = True
    elif prior == "gauss_mixture":
        prior = get_prior_gauss_mixture(num_modes, latent_dim)
        use_exact_kl = False

    kl_regularizer = get_kl_regularizer(prior, weight=kl_weight, use_exact_kl=use_exact_kl)

    encoder = get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim)
    print(encoder.summary())
    decoder = get_vae_decoder(x_train_shape, dense_layers, latent_dim, decoder_output_type, decoding_activation)
    print(decoder.summary())

    vae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate = 1e-3

    vae_model.compile(optimizer=optimizer, loss='mean_squared_error')

    return (vae_model, encoder, decoder)

# endregion

#region AE Model

def get_ae_encoder(x_train_shape, dense_layers, latent_dim):
  '''
  Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
  - dense_layers: List of dense layers, where layers appear in the list in the desired order.
      Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
      The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

  Outputs:
  - Model instance
  '''

  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (x_train_shape[1], ) ) ])

  # add dense unit in reverse order
  for dense_layer in dense_layers:
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"] ) )

  # add a linear layer mapping to the latent space
  model.add( Dense( units = latent_dim, name = "encoding_layer" ) )


  return(model)



def get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
  '''
  Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
  Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
  - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
    the decoder and encoder are symmetrical.

      Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
  - decoding_activation: activation function for the last decoder layer.

  Outputs:
  - Model instance
  '''


  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (latent_dim, ) ) ])

  # add dense unit in reverse order
  for dense_layer in reversed(dense_layers):
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"]) )

  # add a layer with custom tanh activation to map back to the input space.
  model.add( Dense( units = x_train_shape[1], name = "decoding_layer", activation = decoding_activation ) )


  return(model)


def get_autoencoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
    '''
    Return a compiled autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - decoding_activation: activation function for the last decoder layer.
    '''

    encoder = get_ae_encoder(x_train_shape, dense_layers, latent_dim)
    print(  encoder.summary() )
    decoder = get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation)
    print(  decoder.summary() )
    # Initialise and compile VAE
    ae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate  = 1e-3

    ae_model.compile(optimizer=optimizer, loss = 'mean_squared_error')

    return(ae_model, encoder, decoder)

#region


=======
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras import backend as BK

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Concatenate, Conv2D,
                                     UpSampling2D, BatchNormalization)

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


# region Deep Learning models

# region General dl functions

def custom_tanh(x, target_min=0, target_max=None):
    '''
    For given max and min values, return tanh activation with values in this range

    Inputs:
    - x: values to be transformed (np.array)
    - target_min: minimum value of tanh function (float)
    - target_max: max value of tanh function (float)

    Returns:
    - x transformed by tanh activation (np.array)
    '''

    x_02range = BK.tanh(x) + 1  # x mapped into range(0,2)
    scale = (target_max - target_min) / 2.  # calculate target range of transformed x

    return (x_02range * scale + target_min)


def add_layer_to_list(layer_list, filename, hidden_units, activation="relu"):
    '''
    Append layers with valid unit number to a list. Layers with units set to "None" or zero are ignored.

    Inputs:
    - layer_list: list of dense layers to which to append
    - hidden_units (integer or float): numbe rof hidden units for this layer
    - activation: what activation to use

    Return:
    - list with appended layer.
    '''

    if (hidden_units is not None) & (hidden_units != 0):
        layer_list.append(dict(hidden_units=hidden_units, activation=activation))
        filename = filename + "_dense_" + str(hidden_units)
    else:
        pass

    return (layer_list, filename)



def get_checkpoint_callback(checkpoint_filepath_dir, file_name):
    '''
    Create ModelCheckpoint callback object, which saves weights of the best model so far (the model with minimal loss on a validation set).

    Inputs:
    -  checkpoint_filepath: path where to save the model weights

    Returns
    - checkpoint callback (object)
    - filepath used for checkpointing (string)
    '''
    checkpoint_filepath = os.path.join(checkpoint_filepath_dir, file_name)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    return (model_checkpoint_callback, checkpoint_filepath)

# endregion

# region VAE Model

def get_indep_normal_prior(latent_dim):
    '''
    Create mutlivariate normal prior with identity covariance matrix.

    Inputs:
    - latent_dim (integer): dimensionality of the latent space

    Outputs:
    - prior (distribution object)
    '''
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim))

    return (prior)


def get_prior_gauss_mixture(num_modes, latent_dim):
    """
    This function creates an instance of a MixtureSameFamily distribution, specifically a mixture of Gaussian distribution.
    The mixing distribution is uniform with equal probabilities over component distributions.

    Inputs:
    - num_modes (integer): number of modes of mixture distribution.
    - latent_dim (integer): dimension of the latent space.

    Returns:
    - distribution instance with the specified characteristics.
    """

    gm = tfd.MixtureSameFamily(
        # Mixing distribution:
        # Gaussians are equally likely
        mixture_distribution=tfd.Categorical(probs=tf.convert_to_tensor([1 / num_modes] * num_modes, np.float32)),

        # Component distributions:
        # Enforce PD covariance matrix by softplus activation
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(initial_value=tf.random.normal([num_modes, latent_dim]), trainable=True),
            # We need to have scale: NUM_MODES x LATENT_DIM as Gaussian correpsonding to each mode will have
            # a different covariance matrix
            scale_diag=tfp.util.TransformedVariable(initial_value=tf.ones([num_modes, latent_dim]),
                                                    bijector=tfb.Softplus())
        )
    )

    return (gm)


def get_kl_regularizer(prior_distribution, weight=1, use_exact_kl=False):
    """
    This function should create an instance of the KLDivergenceRegularizer for a given prior distribution and KL penalty weight.

    Inputs:
    - prior_distribution (tfd object)
    - weight (float): weight given to KL divergence penalty

    Returns:
    - KLDivergenceRegularizer instance.
    """
    divergence_regularizer = tfpl.KLDivergenceRegularizer(prior_distribution,
                                                          use_exact_kl=use_exact_kl,
                                                          test_points_fn=lambda q: q.sample(15),
                                                          test_points_reduce_axis=None,
                                                          weight=weight)

    return (divergence_regularizer)


def get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim):
    '''
    Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
    - dense_layers: List of dense layers, where layers appear in the list in the desired order.
        Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
        The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

    - kl_regularizer: an instance of KLDivergenceRegularizer
    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

    Outputs:
    - Model instance
    '''

    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(x_train_shape[1],))])

    # add dense unit in reverse order
    for dense_layer in dense_layers:
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    # add dense layer with linear activation to generate parameters of the encoding distribution
    model.add(Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)))

    # add kl divergence regularizer
    model.add(tfpl.MultivariateNormalTriL(event_size=latent_dim,
                                          activity_regularizer=kl_regularizer))

    # print model summary
    # print( model.summary() )

    return (model)


def get_vae_decoder(x_train_shape, dense_layers, latent_dim, output_type, decoding_activation):
    '''
    Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
    Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
    - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
      the decoder and encoder are symmetrical.

        Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).

        The final decoder layer models latent space as a multivariate Normal distribution as is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
    - output_type: (string): Specifies the type of output from decoder. Two options:
        1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
        2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Outputs:
    - Model instance
    '''
    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,))])

    # add dense unit in reverse order
    for dense_layer in reversed(dense_layers):
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    if output_type == "reconstructions":
        # add a layer with custom tanh activation to map back to the input space.
        model.add(Dense(units=x_train_shape[1], name="decoding_layer", activation=decoding_activation))

    elif output_type == "distributions":
        model.add(Dense(x_train_shape[1] * 2)),

        model.add(
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :x_train_shape[1]],
                               scale=tf.keras.activations.softmax(t[..., x_train_shape[1]:]))
                )
            )
        )

    return (model)



def get_vae(x_train_shape, dense_layers, latent_dim, prior, kl_weight, num_modes=None,
            decoder_output_type="reconstructions", decoding_activation=None):
    '''
    Return a compiled variational autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - prior (string): prior distribution on latent space. Allows two options:
          1. "indep_normal" for multivariate nromal distribution with identity covariance matrix.
          2. "gauss_mixture" for Guassian mixture prior.
    - kl_weight: weighted of KL divergence penalty (higher means more penalization of deviations of latent space from prior).
    - num_modes (integer): if prior == "gauss_mixture", specifies the number of modes.
    - decoder_output_type (string): Specifies the type of output from decoder. Two options:
      1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
      2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Returns:
    - vae model
    - encoder object
    - decoder object
    '''
    if prior == "indep_normal":
        prior = get_indep_normal_prior(latent_dim)
        use_exact_kl = True
    elif prior == "gauss_mixture":
        prior = get_prior_gauss_mixture(num_modes, latent_dim)
        use_exact_kl = False

    kl_regularizer = get_kl_regularizer(prior, weight=kl_weight, use_exact_kl=use_exact_kl)

    encoder = get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim)
    print(encoder.summary())
    decoder = get_vae_decoder(x_train_shape, dense_layers, latent_dim, decoder_output_type, decoding_activation)
    print(decoder.summary())

    vae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate = 1e-3

    vae_model.compile(optimizer=optimizer, loss='mean_squared_error')

    return (vae_model, encoder, decoder)

# endregion

#region AE Model

def get_ae_encoder(x_train_shape, dense_layers, latent_dim):
  '''
  Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
  - dense_layers: List of dense layers, where layers appear in the list in the desired order.
      Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
      The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

  Outputs:
  - Model instance
  '''

  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (x_train_shape[1], ) ) ])

  # add dense unit in reverse order
  for dense_layer in dense_layers:
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"] ) )

  # add a linear layer mapping to the latent space
  model.add( Dense( units = latent_dim, name = "encoding_layer" ) )


  return(model)



def get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
  '''
  Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
  Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
  - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
    the decoder and encoder are symmetrical.

      Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
  - decoding_activation: activation function for the last decoder layer.

  Outputs:
  - Model instance
  '''


  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (latent_dim, ) ) ])

  # add dense unit in reverse order
  for dense_layer in reversed(dense_layers):
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"]) )

  # add a layer with custom tanh activation to map back to the input space.
  model.add( Dense( units = x_train_shape[1], name = "decoding_layer", activation = decoding_activation ) )


  return(model)


def get_autoencoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
    '''
    Return a compiled autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - decoding_activation: activation function for the last decoder layer.
    '''

    encoder = get_ae_encoder(x_train_shape, dense_layers, latent_dim)
    print(  encoder.summary() )
    decoder = get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation)
    print(  decoder.summary() )
    # Initialise and compile VAE
    ae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate  = 1e-3

    ae_model.compile(optimizer=optimizer, loss = 'mean_squared_error')

    return(ae_model, encoder, decoder)

#region


>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
=======
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras import backend as BK

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Concatenate, Conv2D,
                                     UpSampling2D, BatchNormalization)

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


# region Deep Learning models

# region General dl functions

def custom_tanh(x, target_min=0, target_max=None):
    '''
    For given max and min values, return tanh activation with values in this range

    Inputs:
    - x: values to be transformed (np.array)
    - target_min: minimum value of tanh function (float)
    - target_max: max value of tanh function (float)

    Returns:
    - x transformed by tanh activation (np.array)
    '''

    x_02range = BK.tanh(x) + 1  # x mapped into range(0,2)
    scale = (target_max - target_min) / 2.  # calculate target range of transformed x

    return (x_02range * scale + target_min)


def add_layer_to_list(layer_list, filename, hidden_units, activation="relu"):
    '''
    Append layers with valid unit number to a list. Layers with units set to "None" or zero are ignored.

    Inputs:
    - layer_list: list of dense layers to which to append
    - hidden_units (integer or float): numbe rof hidden units for this layer
    - activation: what activation to use

    Return:
    - list with appended layer.
    '''

    if (hidden_units is not None) & (hidden_units != 0):
        layer_list.append(dict(hidden_units=hidden_units, activation=activation))
        filename = filename + "_dense_" + str(hidden_units)
    else:
        pass

    return (layer_list, filename)



def get_checkpoint_callback(checkpoint_filepath_dir, file_name):
    '''
    Create ModelCheckpoint callback object, which saves weights of the best model so far (the model with minimal loss on a validation set).

    Inputs:
    -  checkpoint_filepath: path where to save the model weights

    Returns
    - checkpoint callback (object)
    - filepath used for checkpointing (string)
    '''
    checkpoint_filepath = os.path.join(checkpoint_filepath_dir, file_name)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    return (model_checkpoint_callback, checkpoint_filepath)

# endregion

# region VAE Model

def get_indep_normal_prior(latent_dim):
    '''
    Create mutlivariate normal prior with identity covariance matrix.

    Inputs:
    - latent_dim (integer): dimensionality of the latent space

    Outputs:
    - prior (distribution object)
    '''
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim))

    return (prior)


def get_prior_gauss_mixture(num_modes, latent_dim):
    """
    This function creates an instance of a MixtureSameFamily distribution, specifically a mixture of Gaussian distribution.
    The mixing distribution is uniform with equal probabilities over component distributions.

    Inputs:
    - num_modes (integer): number of modes of mixture distribution.
    - latent_dim (integer): dimension of the latent space.

    Returns:
    - distribution instance with the specified characteristics.
    """

    gm = tfd.MixtureSameFamily(
        # Mixing distribution:
        # Gaussians are equally likely
        mixture_distribution=tfd.Categorical(probs=tf.convert_to_tensor([1 / num_modes] * num_modes, np.float32)),

        # Component distributions:
        # Enforce PD covariance matrix by softplus activation
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(initial_value=tf.random.normal([num_modes, latent_dim]), trainable=True),
            # We need to have scale: NUM_MODES x LATENT_DIM as Gaussian correpsonding to each mode will have
            # a different covariance matrix
            scale_diag=tfp.util.TransformedVariable(initial_value=tf.ones([num_modes, latent_dim]),
                                                    bijector=tfb.Softplus())
        )
    )

    return (gm)


def get_kl_regularizer(prior_distribution, weight=1, use_exact_kl=False):
    """
    This function should create an instance of the KLDivergenceRegularizer for a given prior distribution and KL penalty weight.

    Inputs:
    - prior_distribution (tfd object)
    - weight (float): weight given to KL divergence penalty

    Returns:
    - KLDivergenceRegularizer instance.
    """
    divergence_regularizer = tfpl.KLDivergenceRegularizer(prior_distribution,
                                                          use_exact_kl=use_exact_kl,
                                                          test_points_fn=lambda q: q.sample(15),
                                                          test_points_reduce_axis=None,
                                                          weight=weight)

    return (divergence_regularizer)


def get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim):
    '''
    Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
    - dense_layers: List of dense layers, where layers appear in the list in the desired order.
        Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
        The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

    - kl_regularizer: an instance of KLDivergenceRegularizer
    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

    Outputs:
    - Model instance
    '''

    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(x_train_shape[1],))])

    # add dense unit in reverse order
    for dense_layer in dense_layers:
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    # add dense layer with linear activation to generate parameters of the encoding distribution
    model.add(Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)))

    # add kl divergence regularizer
    model.add(tfpl.MultivariateNormalTriL(event_size=latent_dim,
                                          activity_regularizer=kl_regularizer))

    # print model summary
    # print( model.summary() )

    return (model)


def get_vae_decoder(x_train_shape, dense_layers, latent_dim, output_type, decoding_activation):
    '''
    Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
    Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
    - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
      the decoder and encoder are symmetrical.

        Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).

        The final decoder layer models latent space as a multivariate Normal distribution as is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
    - output_type: (string): Specifies the type of output from decoder. Two options:
        1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
        2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Outputs:
    - Model instance
    '''
    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,))])

    # add dense unit in reverse order
    for dense_layer in reversed(dense_layers):
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    if output_type == "reconstructions":
        # add a layer with custom tanh activation to map back to the input space.
        model.add(Dense(units=x_train_shape[1], name="decoding_layer", activation=decoding_activation))

    elif output_type == "distributions":
        model.add(Dense(x_train_shape[1] * 2)),

        model.add(
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :x_train_shape[1]],
                               scale=tf.keras.activations.softmax(t[..., x_train_shape[1]:]))
                )
            )
        )

    return (model)



def get_vae(x_train_shape, dense_layers, latent_dim, prior, kl_weight, num_modes=None,
            decoder_output_type="reconstructions", decoding_activation=None):
    '''
    Return a compiled variational autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - prior (string): prior distribution on latent space. Allows two options:
          1. "indep_normal" for multivariate nromal distribution with identity covariance matrix.
          2. "gauss_mixture" for Guassian mixture prior.
    - kl_weight: weighted of KL divergence penalty (higher means more penalization of deviations of latent space from prior).
    - num_modes (integer): if prior == "gauss_mixture", specifies the number of modes.
    - decoder_output_type (string): Specifies the type of output from decoder. Two options:
      1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
      2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Returns:
    - vae model
    - encoder object
    - decoder object
    '''
    if prior == "indep_normal":
        prior = get_indep_normal_prior(latent_dim)
        use_exact_kl = True
    elif prior == "gauss_mixture":
        prior = get_prior_gauss_mixture(num_modes, latent_dim)
        use_exact_kl = False

    kl_regularizer = get_kl_regularizer(prior, weight=kl_weight, use_exact_kl=use_exact_kl)

    encoder = get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim)
    print(encoder.summary())
    decoder = get_vae_decoder(x_train_shape, dense_layers, latent_dim, decoder_output_type, decoding_activation)
    print(decoder.summary())

    vae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate = 1e-3

    vae_model.compile(optimizer=optimizer, loss='mean_squared_error')

    return (vae_model, encoder, decoder)

# endregion

#region AE Model

def get_ae_encoder(x_train_shape, dense_layers, latent_dim):
  '''
  Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
  - dense_layers: List of dense layers, where layers appear in the list in the desired order.
      Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
      The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

  Outputs:
  - Model instance
  '''

  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (x_train_shape[1], ) ) ])

  # add dense unit in reverse order
  for dense_layer in dense_layers:
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"] ) )

  # add a linear layer mapping to the latent space
  model.add( Dense( units = latent_dim, name = "encoding_layer" ) )


  return(model)



def get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
  '''
  Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
  Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
  - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
    the decoder and encoder are symmetrical.

      Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
  - decoding_activation: activation function for the last decoder layer.

  Outputs:
  - Model instance
  '''


  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (latent_dim, ) ) ])

  # add dense unit in reverse order
  for dense_layer in reversed(dense_layers):
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"]) )

  # add a layer with custom tanh activation to map back to the input space.
  model.add( Dense( units = x_train_shape[1], name = "decoding_layer", activation = decoding_activation ) )


  return(model)


def get_autoencoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
    '''
    Return a compiled autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - decoding_activation: activation function for the last decoder layer.
    '''

    encoder = get_ae_encoder(x_train_shape, dense_layers, latent_dim)
    print(  encoder.summary() )
    decoder = get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation)
    print(  decoder.summary() )
    # Initialise and compile VAE
    ae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate  = 1e-3

    ae_model.compile(optimizer=optimizer, loss = 'mean_squared_error')

    return(ae_model, encoder, decoder)

#region


>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
=======
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras import backend as BK

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Concatenate, Conv2D,
                                     UpSampling2D, BatchNormalization)

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


# region Deep Learning models

# region General dl functions

def custom_tanh(x, target_min=0, target_max=None):
    '''
    For given max and min values, return tanh activation with values in this range

    Inputs:
    - x: values to be transformed (np.array)
    - target_min: minimum value of tanh function (float)
    - target_max: max value of tanh function (float)

    Returns:
    - x transformed by tanh activation (np.array)
    '''

    x_02range = BK.tanh(x) + 1  # x mapped into range(0,2)
    scale = (target_max - target_min) / 2.  # calculate target range of transformed x

    return (x_02range * scale + target_min)


def add_layer_to_list(layer_list, filename, hidden_units, activation="relu"):
    '''
    Append layers with valid unit number to a list. Layers with units set to "None" or zero are ignored.

    Inputs:
    - layer_list: list of dense layers to which to append
    - hidden_units (integer or float): numbe rof hidden units for this layer
    - activation: what activation to use

    Return:
    - list with appended layer.
    '''

    if (hidden_units is not None) & (hidden_units != 0):
        layer_list.append(dict(hidden_units=hidden_units, activation=activation))
        filename = filename + "_dense_" + str(hidden_units)
    else:
        pass

    return (layer_list, filename)



def get_checkpoint_callback(checkpoint_filepath_dir, file_name):
    '''
    Create ModelCheckpoint callback object, which saves weights of the best model so far (the model with minimal loss on a validation set).

    Inputs:
    -  checkpoint_filepath: path where to save the model weights

    Returns
    - checkpoint callback (object)
    - filepath used for checkpointing (string)
    '''
    checkpoint_filepath = os.path.join(checkpoint_filepath_dir, file_name)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    return (model_checkpoint_callback, checkpoint_filepath)

# endregion

# region VAE Model

def get_indep_normal_prior(latent_dim):
    '''
    Create mutlivariate normal prior with identity covariance matrix.

    Inputs:
    - latent_dim (integer): dimensionality of the latent space

    Outputs:
    - prior (distribution object)
    '''
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim))

    return (prior)


def get_prior_gauss_mixture(num_modes, latent_dim):
    """
    This function creates an instance of a MixtureSameFamily distribution, specifically a mixture of Gaussian distribution.
    The mixing distribution is uniform with equal probabilities over component distributions.

    Inputs:
    - num_modes (integer): number of modes of mixture distribution.
    - latent_dim (integer): dimension of the latent space.

    Returns:
    - distribution instance with the specified characteristics.
    """

    gm = tfd.MixtureSameFamily(
        # Mixing distribution:
        # Gaussians are equally likely
        mixture_distribution=tfd.Categorical(probs=tf.convert_to_tensor([1 / num_modes] * num_modes, np.float32)),

        # Component distributions:
        # Enforce PD covariance matrix by softplus activation
        components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.Variable(initial_value=tf.random.normal([num_modes, latent_dim]), trainable=True),
            # We need to have scale: NUM_MODES x LATENT_DIM as Gaussian correpsonding to each mode will have
            # a different covariance matrix
            scale_diag=tfp.util.TransformedVariable(initial_value=tf.ones([num_modes, latent_dim]),
                                                    bijector=tfb.Softplus())
        )
    )

    return (gm)


def get_kl_regularizer(prior_distribution, weight=1, use_exact_kl=False):
    """
    This function should create an instance of the KLDivergenceRegularizer for a given prior distribution and KL penalty weight.

    Inputs:
    - prior_distribution (tfd object)
    - weight (float): weight given to KL divergence penalty

    Returns:
    - KLDivergenceRegularizer instance.
    """
    divergence_regularizer = tfpl.KLDivergenceRegularizer(prior_distribution,
                                                          use_exact_kl=use_exact_kl,
                                                          test_points_fn=lambda q: q.sample(15),
                                                          test_points_reduce_axis=None,
                                                          weight=weight)

    return (divergence_regularizer)


def get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim):
    '''
    Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
    - dense_layers: List of dense layers, where layers appear in the list in the desired order.
        Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
        The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

    - kl_regularizer: an instance of KLDivergenceRegularizer
    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

    Outputs:
    - Model instance
    '''

    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(x_train_shape[1],))])

    # add dense unit in reverse order
    for dense_layer in dense_layers:
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    # add dense layer with linear activation to generate parameters of the encoding distribution
    model.add(Dense(tfpl.MultivariateNormalTriL.params_size(latent_dim)))

    # add kl divergence regularizer
    model.add(tfpl.MultivariateNormalTriL(event_size=latent_dim,
                                          activity_regularizer=kl_regularizer))

    # print model summary
    # print( model.summary() )

    return (model)


def get_vae_decoder(x_train_shape, dense_layers, latent_dim, output_type, decoding_activation):
    '''
    Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
    Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

    Inputs:
    - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
    - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
      the decoder and encoder are symmetrical.

        Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
        For example,

        dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                          {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

        will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).

        The final decoder layer models latent space as a multivariate Normal distribution as is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
    - output_type: (string): Specifies the type of output from decoder. Two options:
        1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
        2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Outputs:
    - Model instance
    '''
    # initialise model with input layer
    model = Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,))])

    # add dense unit in reverse order
    for dense_layer in reversed(dense_layers):
        model.add(Dense(units=dense_layer["hidden_units"], activation=dense_layer["activation"]))

    if output_type == "reconstructions":
        # add a layer with custom tanh activation to map back to the input space.
        model.add(Dense(units=x_train_shape[1], name="decoding_layer", activation=decoding_activation))

    elif output_type == "distributions":
        model.add(Dense(x_train_shape[1] * 2)),

        model.add(
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :x_train_shape[1]],
                               scale=tf.keras.activations.softmax(t[..., x_train_shape[1]:]))
                )
            )
        )

    return (model)



def get_vae(x_train_shape, dense_layers, latent_dim, prior, kl_weight, num_modes=None,
            decoder_output_type="reconstructions", decoding_activation=None):
    '''
    Return a compiled variational autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - prior (string): prior distribution on latent space. Allows two options:
          1. "indep_normal" for multivariate nromal distribution with identity covariance matrix.
          2. "gauss_mixture" for Guassian mixture prior.
    - kl_weight: weighted of KL divergence penalty (higher means more penalization of deviations of latent space from prior).
    - num_modes (integer): if prior == "gauss_mixture", specifies the number of modes.
    - decoder_output_type (string): Specifies the type of output from decoder. Two options:
      1. "reconstructions": Decoder outputs (point) reconstructions passed through custom tanh activation.
      2. "distributions": Instead of poitn distribution model each reconstruction by (independent) normal distribution. Decoder then outputs this distriubtion function.
    - decoding_activation: activation function for the last decoder layer, used only if "decoder_output_type" == "reconstructions".

    Returns:
    - vae model
    - encoder object
    - decoder object
    '''
    if prior == "indep_normal":
        prior = get_indep_normal_prior(latent_dim)
        use_exact_kl = True
    elif prior == "gauss_mixture":
        prior = get_prior_gauss_mixture(num_modes, latent_dim)
        use_exact_kl = False

    kl_regularizer = get_kl_regularizer(prior, weight=kl_weight, use_exact_kl=use_exact_kl)

    encoder = get_vae_encoder(x_train_shape, dense_layers, kl_regularizer, latent_dim)
    print(encoder.summary())
    decoder = get_vae_decoder(x_train_shape, dense_layers, latent_dim, decoder_output_type, decoding_activation)
    print(decoder.summary())

    vae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate = 1e-3

    vae_model.compile(optimizer=optimizer, loss='mean_squared_error')

    return (vae_model, encoder, decoder)

# endregion

#region AE Model

def get_ae_encoder(x_train_shape, dense_layers, latent_dim):
  '''
  Constructs an encoder network for a given shape of input, dense layer specification and latent dimensions.

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining InputLayer
  - dense_layers: List of dense layers, where layers appear in the list in the desired order.
      Each layer is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to an encoder with first hidden layer with 256 units, and second hidden layer with 128 units (both with ReLu activations).
      The final encoder layer with linear activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)

  Outputs:
  - Model instance
  '''

  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (x_train_shape[1], ) ) ])

  # add dense unit in reverse order
  for dense_layer in dense_layers:
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"] ) )

  # add a linear layer mapping to the latent space
  model.add( Dense( units = latent_dim, name = "encoding_layer" ) )


  return(model)



def get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
  '''
  Constructs an decoder network for a given shape of input, dense layer specification and latent dimensions.
  Note that dense layers are added in reverse order (so that we end up with decoder symmetrical to the encoder based on the same "dense_layers").

  Inputs:
  - x_train_shape: Shape of training dataset (tuple). Only the second dimension (number of columns / features) matters. Used for defining a final output layer.
  - dense_layers: List of dense layers, where layers appear in the list in the order in which we want them to appear in the ENCODER. They are added to decoder in reverse order, so that
    the decoder and encoder are symmetrical.

      Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256, 'name': 'dense_1'},
                        {'activation': 'relu', 'hidden_units': 128, 'name': 'dense_2'}]

      will lead to a decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

  - latent_dim (integer): dimensionality of the latent space (i.e. a bottleneck size)
  - decoding_activation: activation function for the last decoder layer.

  Outputs:
  - Model instance
  '''


  # initialise model with input layer
  model = Sequential([ tf.keras.layers.InputLayer(input_shape = (latent_dim, ) ) ])

  # add dense unit in reverse order
  for dense_layer in reversed(dense_layers):
    model.add( Dense( units = dense_layer["hidden_units"], activation = dense_layer["activation"]) )

  # add a layer with custom tanh activation to map back to the input space.
  model.add( Dense( units = x_train_shape[1], name = "decoding_layer", activation = decoding_activation ) )


  return(model)


def get_autoencoder(x_train_shape, dense_layers, latent_dim, decoding_activation):
    '''
    Return a compiled autoencoder model with given characteristics.

    Inputs:
    - x_train_shape (tuple): shape of input data.
    - dense_layers (list): list of dense layers. Each layer within the list is a dictionary and needs to have keywords "hidden_units" (float), "activation" (string) and "name" (string).
      For example,

      dense_layers = [{'activation': 'relu', 'hidden_units': 256},
                        {'activation': 'relu', 'hidden_units': 128}]

      will lead to an encoder with first hidden layer with 256 units and second layer with 128 units (both with ReLu activations).
      The decoder will be constructed symmetrically, that is, first layer will have decoder with first hidden layer with 128 units, and second hidden layer with 256 units (both with ReLu activations).
      The finalencoding layer and the final decoder layer with custom tanh activation is added by default (so it shouldn't appear in "dense_layers" list)

    - latent_dim: bottleneck size / dimensionality of the latent space
    - decoding_activation: activation function for the last decoder layer.
    '''

    encoder = get_ae_encoder(x_train_shape, dense_layers, latent_dim)
    print(  encoder.summary() )
    decoder = get_ae_decoder(x_train_shape, dense_layers, latent_dim, decoding_activation)
    print(  decoder.summary() )
    # Initialise and compile VAE
    ae_model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    optimizer = tf.keras.optimizers.Adam()
    optimizer.learning_rate  = 1e-3

    ae_model.compile(optimizer=optimizer, loss = 'mean_squared_error')

    return(ae_model, encoder, decoder)

#region


>>>>>>> 4db1f245e9ea3598f9016d0e652d7f0a0b739c77
# endregion