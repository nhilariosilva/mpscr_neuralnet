import os, shutil
import subprocess
import json

from net_model import *

# --------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------ This class for the Model is designed for the OASIS-3 dataset ------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------

class MPScrModel(MPScrModelStructure):
    def __init__(self, log_a, log_phi, C, C_inv, sup, theta_min = None, theta_max = None, link = "logit", verbose = 0):
        super().__init__(log_a, log_phi, C, C_inv, sup, theta_min, theta_max, link, verbose)
            
    def define_structure(self, shape_input):
        self.shape_input = shape_input
        
        # Gera uma imagem inteira de zeros com as dimensões do modelo
        dummy_input = keras.layers.Input(shape = self.shape_input)
        self.dummy_input = dummy_input

        # initializer = initializers.GlorotUniform(seed = seed)    
        # self.convolution1 = keras.layers.Conv2D(filters = 12, kernel_size = [7,7], padding = "same", activation = tf.nn.leaky_relu,
        #                                         kernel_initializer = initializer, dtype = tf.float32)
        # self.pooling1 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        # self.convolution2 = keras.layers.Conv2D(filters = 24, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
        #                                         kernel_initializer = initializer, dtype = tf.float32)
        # self.pooling2 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        # self.convolution3 = keras.layers.Conv2D(filters = 48, kernel_size = [3,3], padding = "same", activation = tf.nn.leaky_relu,
        #                                         kernel_initializer = initializer, dtype = tf.float32)
        # self.pooling3 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        # self.convolution4 = keras.layers.Conv2D(filters = 96, kernel_size = [3,3], padding = "same", activation = tf.nn.leaky_relu,
        #                                         kernel_initializer = initializer, dtype = tf.float32)
        # self.pooling4 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)

        initializer = initializers.GlorotUniform()
        self.convolution1 = keras.layers.Conv2D(filters = 8, kernel_size = [7,7], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling1 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        self.convolution2 = keras.layers.Conv2D(filters = 16, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling2 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        self.convolution3 = keras.layers.Conv2D(filters = 32, kernel_size = [3,3], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling3 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        self.convolution4 = keras.layers.Conv2D(filters = 64, kernel_size = [3,3], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling4 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
    
        self.flatten = keras.layers.Reshape(target_shape=(-1,))
        self.dense1 = keras.layers.Dense(units = 128, activation = tf.nn.tanh, dtype = tf.float32)
        self.dense2 = keras.layers.Dense(units = 1, dtype = tf.float32, activation = None, use_bias = False)

        # Initialize the model weights (if not called beforehand, the method .get_weights() returns an empty list)
        self(dummy_input)
        
    def call(self, x_input):
        x = self.convolution1(x_input)
        x = self.pooling1(x)
        x = self.convolution2(x)
        x = self.pooling2(x)
        x = self.convolution3(x)
        x = self.pooling3(x)
        x = self.convolution4(x)
        x = self.pooling4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.cast(x, dtype = tf.float64)
        return x
    
    def copy(self):
        new_model = MPScrModel(self.log_a, self.log_phi, self.C, self.C_inv, self.sup, self.theta_min, self.theta_max, self.link)
        new_model.define_structure(shape_input = self.shape_input)
        new_model.set_weights(self.get_weights())
        return new_model
        
    def build_graph(self):
        return tf.keras.Model(inputs = self.dummy_input, outputs = self.call(self.dummy_input))
    