import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import keras.backend as K
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Dense, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import L1L2
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from Chapter4.image_utils import dim_ordering_unfix, dim_ordering_shape

mpl.use("Agg")

def model_generator():
    model = Sequential()
    nch = 256
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Conv2D(int(nch / 2), (h, h), padding="same", kernel_regularizer=reg()))

