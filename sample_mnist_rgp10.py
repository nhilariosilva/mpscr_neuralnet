import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from time import time

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

import lifelines

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

import os, shutil
from pathlib import Path
import json
import subprocess

import mps
import pwexp
from net_model import *
from mps_models import *

# This file aims to generate only the simulations for the RGP(-1/10) distribution to keep it reproducible. This experiment would originally be in Scenario 2, but because when q < 0 in the RGP
# setting, we can achieve cure probability intervals very close to the unit interval (0,1) we can safely use this distribution for the cure probabilities specified in Scenario 1

def get_theta(log_a, log_phi, C, C_inv, sup, p_0, theta_min = None, theta_max = None):
    '''
        Given the specifications for the latent causes distribution and a vector with cure probabilities,
        inverts the cure probability function and returns the theta parameters for each individual
    '''
    theta = C_inv( np.exp(log_a(0.0) - np.log(p_0)) )
    
    # Se theta é limitado inferiormente por um valor theta_min > 0, valores de theta obtidos abaixo do limite são levados para o limite inferior do parâmetro
    if(theta_min is not None):
        theta[theta <= theta_min] = theta_min + 1.0e-5
    # Se theta é limitado superiormente por um valor theta_max > 0, valores de theta obtidos acima do limite são levados para o limite superior do parâmetro
    if(theta_min is not None):
        theta[theta >= theta_max] = theta_max - 1.0e-5
        
    return theta


def generate_data(log_a, log_phi, theta, sup, low_c, high_c):
    '''
        Dada a especificação do modelo e um vetor com os parâmetros individuais, gera os tempos de vida e censuras de cada indivíduo.
        low_c e high_c definem o intervalo para a geração dos tempos de censura, seguindo uma distribuição U[low_c, high_c]
    '''
    n = len(theta)
    m = mps.rvs(log_a, log_phi, theta, sup, size = 10)
    
    cured = np.zeros(n)
    delta = cured.copy()
    t = cured.copy()
    
    # Censorship times
    c = np.random.uniform(low = low_c, high = high_c, size = n)
    
    for i in range(n):
        if(m[i] == 0):
            t[i] = c[i]
            cured[i] = 1
        else:
            # Risco base segue uma distribuição Exp(1)
            z = expon.rvs(loc = 0.0, scale = 1.0, size = int(m[i]))
            t[i] = np.min(z)
    
    # Atualiza as posições não censuradas para delta = 1
    delta[t < c] = 1
    # Os tempos censurados passam a assumir o valor do tempo de censura
    t[t >= c] = c[t >= c]
    
    # Retorna os tempos, deltas e o vetor de causas latentes (que na prática é desconhecido)
    return m, t, delta, cured

def join_datasets(n_train, n_val, n_test, theta_train, theta_val, theta_test, m_train, m_val, m_test, t_train, t_val, t_test, delta_train, delta_val, delta_test):
    sets = np.concatenate([np.repeat("train", n_train), np.repeat("val", n_val), np.repeat("test", n_test)])
    theta = np.concatenate([theta_train, theta_val, theta_test])
    m = np.concatenate([m_train, m_val, m_test])
    t = np.concatenate([t_train, t_val, t_test])
    delta = np.concatenate([delta_train, delta_val, delta_test])
    return pd.DataFrame({"theta": theta, "m": m, "t": t, "delta": delta, "set": sets})

def sample_single_bootstrap_rgp10(cure_probs_dict_vec, directory, file_index):
    '''
        Get a single bootstrap sample from the Fashion-MNIST dataset considering each distribution from scenario 1.
    '''
    filename = "data_{}.csv".format(file_index)

    # ---------------------------- Sample the indices from the original dataset ----------------------------
    
    df_indices = pd.read_csv("{}/indices_{}.csv".format(directory, file_index))
    indices = df_indices["index"].to_numpy()
    sets = df_indices["set"].to_numpy()

    # Indices for train and validation
    i_train_val = indices[ (sets == "train") | (sets == "val") ]
    i_test = indices[ sets == "test" ]
    
    n_train = int(np.sum(sets == "train"))
    n_val = int(np.sum(sets == "val"))
    n_test = int(np.sum(sets == "test"))
    n = n_train + n_val + n_test
    
    # The labels for the train set are the first n_train sampled indices in i_train_val
    label_train = train_labels[i_train_val[:n_train]]
    # The labels for the validation set are the last n_train sampled indices in i_train_val
    label_val = train_labels[i_train_val[n_train:]]
    # Takes the labels for the test set
    label_test = test_labels[i_test]
    
    p_train = cure_probs_dict_vec(label_train)
    p_val = cure_probs_dict_vec(label_val)
    p_test = cure_probs_dict_vec(label_test)

    # The censored times follow a U(low_c, high_c) distribution - To control the censored and cured observations properly, we should have a different distribution 
    # for each of the chosen distributions for M
    low_c = 0
    high_c = 6
    
    # ---------------------------- RPG(-1/10) ----------------------------
    q = -1.0/10.0
    # RPG(-1/10) - Training data
    theta_train_rgp10 = get_theta(log_a_rgp(q), log_phi_rgp(q), C_rgp(q), C_inv_rgp(q), sup_rgp(q), p_train, theta_min = theta_min_rgp, theta_max = theta_max_rgp(q))
    m_train_rgp10, t_train_rgp10, delta_train_rgp10, cured_train_rgp10 = \
        generate_data(log_a_rgp(q), log_phi_rgp(q), theta_train_rgp10, sup_rgp(q), low_c, high_c)
    # RPG(-1/10) - Validation data
    theta_val_rgp10 = get_theta(log_a_rgp(q), log_phi_rgp(q), C_rgp(q), C_inv_rgp(q), sup_rgp(q), p_val, theta_min = theta_min_rgp, theta_max = theta_max_rgp(q))
    m_val_rgp10, t_val_rgp10, delta_val_rgp10, cured_val_rgp10 = \
        generate_data(log_a_rgp(q), log_phi_rgp(q), theta_val_rgp10, sup_rgp(q), low_c, high_c)
    # RPG(-1/10) - Test data
    theta_test_rgp10 = get_theta(log_a_rgp(q), log_phi_rgp(q), C_rgp(q), C_inv_rgp(q), sup_rgp(q), p_test, theta_min = theta_min_rgp, theta_max = theta_max_rgp(q))
    m_test_rgp10, t_test_rgp10, delta_test_rgp10, cured_test_rgp10 = \
        generate_data(log_a_rgp(q), log_phi_rgp(q), theta_test_rgp10, sup_rgp(q), low_c, high_c)
    # Save the DataFrame with the simulated values for the RGP(-1/10)
    rgp10_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_rgp10, theta_val_rgp10, theta_test_rgp10,
        m_train_rgp10, m_val_rgp10, m_test_rgp10,
        t_train_rgp10, t_val_rgp10, t_test_rgp10,
        delta_train_rgp10, delta_val_rgp10, delta_test_rgp10
    )
    rgp10_data.to_csv("{}/rgp10/{}".format(directory, filename), index = False)


if(__name__ == "__main__"):

    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    i_valid_train = pd.Series(train_labels).isin([0,1,2,3,4]).to_numpy()
    i_valid_test = pd.Series(test_labels).isin([0,1,2,3,4]).to_numpy()

    # Filters to take only the images with labels in [0, 1, 2, 3, 4]
    train_labels = train_labels[i_valid_train]
    test_labels = test_labels[i_valid_test]
    
    print("Creating directories structure")
    dists_scenario1 = ["rgp10"]
    for dist in dists_scenario1:
        Path("SimulationDataset/Scenario1/n500/{}".format(dist)).mkdir(parents=True, exist_ok=True)
        Path("SimulationDataset/Scenario1/n1000/{}".format(dist)).mkdir(parents=True, exist_ok=True)
        Path("SimulationDataset/Scenario1/n3000/{}".format(dist)).mkdir(parents=True, exist_ok=True)

    print("---------------------------- Scenario 1 - RPG(-1/10) ----------------------------")
    cure_probs_dict1 = {0: 0.9, 1:0.45, 2:0.22, 3:0.14, 4: 0.08}
    cure_probs_dict1 = np.vectorize(cure_probs_dict1.get)

    train_sizes = [500, 1000, 3000] # 3 sample sizes
    val_sizes = [108, 214, 643]
    test_sizes = [108, 214, 643]
    
    n_replicates = 100

    np.random.seed(333)
    
    for j in range(len(train_sizes)):
        n_train = train_sizes[j]
        print("n = {}".format(n_train))
        for i in tqdm(range(n_replicates)):
            sample_single_bootstrap_rgp10(
                cure_probs_dict_vec = cure_probs_dict1,
                directory = "SimulationDataset/Scenario1/n{}".format(n_train), file_index = i+1
            )
    