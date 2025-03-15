import os, shutil
import subprocess
import json

from net_model import *

class MPScrModel(MPScrModelStructure):
    def __init__(self, log_a, log_phi, C, C_inv, sup, theta_min = None, theta_max = None, link = "logit", verbose = 0):
        super().__init__(log_a, log_phi, C, C_inv, sup, theta_min, theta_max, link, verbose)
            
    def define_structure(self, shape_input, seed = 1):
        self.shape_input = shape_input
        
        # Gera uma imagem inteira de zeros com as dimensões do modelo
        dummy_input = keras.layers.Input(shape = self.shape_input)
        
        # initializer = initializers.HeNormal(seed = seed)
        initializer = tf.random_normal_initializer(stddev = 0.005)
        
        self.convolution1 = keras.layers.Conv2D(filters = 4, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling1 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        self.convolution2 = keras.layers.Conv2D(filters = 12, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling2 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        self.convolution3 = keras.layers.Conv2D(filters = 32, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
                                                kernel_initializer = initializer, dtype = tf.float32)
        self.pooling3 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
    
        self.flatten = keras.layers.Reshape(target_shape=(-1,))
        self.dense1 = keras.layers.Dense(units = 128, activation = tf.nn.tanh, dtype = tf.float32)
        self.dense2 = keras.layers.Dense(dtype = tf.float32, units = 1, activation = None, use_bias = False)

        # Initialize the model weights (if not called beforehand, the method .get_weights() returns an empty list)
        self(dummy_input)
        
    def call(self, x_input):
        x = self.convolution1(x_input)
        x = self.pooling1(x)
        x = self.convolution2(x)
        x = self.pooling2(x)
        x = self.convolution3(x)
        x = self.pooling3(x)
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
    
    
# Funções traduzidas para o treinamento da rede neural - Poisson
# a_poisson = lambda x : tf.math.exp(-tf.math.lgamma(x+1))
# phi_poisson = lambda theta : theta
# C_poisson = lambda theta : tf.math.exp(theta)
# C_inv_poisson = lambda x : tf.math.log(x)
# B_poisson = 101
# sup_poisson = np.arange(0, B_poisson, 1).astype(np.float64)