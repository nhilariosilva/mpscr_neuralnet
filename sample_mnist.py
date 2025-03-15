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
    m = mps.rvs(log_a, log_phi, theta, sup)
    
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

def sample_single_bootstrap_scenario1(i_train, i_test, n_train, n_val, n_test, cure_probs_dict_vec, directory, file_index):
    '''
        Get a single bootstrap sample from the Fashion-MNIST dataset considering each distribution from scenario 1.
    '''
    n = n_train + n_val + n_test
    filename = "data_{}.csv".format(file_index)

    # ---------------------------- Sample the indices from the original dataset ----------------------------
    
    # Indices for train and validation
    i_train_val = np.random.choice(i_train, size = n_train + n_val, replace = True)
    i_test = np.random.choice(i_test, size = n_test, replace = True)

    sets = np.concatenate([np.repeat("train", n_train), np.repeat("val", n_val), np.repeat("test", n_test)])
    indices = np.concatenate([i_train_val, i_test])
    pd.DataFrame({"index": indices, "set": sets}).to_csv("{}/indices_{}.csv".format(directory, file_index), index = False)

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
    
    # ---------------------------- Poisson ----------------------------
    # Poisson - Training data
    theta_train_poisson = get_theta(log_a_poisson, log_phi_poisson, C_poisson, C_inv_poisson, sup_poisson, p_train, theta_min = theta_min_poisson, theta_max = theta_max_poisson)
    m_train_poisson, t_train_poisson, delta_train_poisson, cured_train_poisson = \
        generate_data(log_a_poisson, log_phi_poisson, theta_train_poisson, sup_poisson, low_c, high_c)
    # Poisson - Validation data
    theta_val_poisson = get_theta(log_a_poisson, log_phi_poisson, C_poisson, C_inv_poisson, sup_poisson, p_val, theta_min = theta_min_poisson, theta_max = theta_max_poisson)
    m_val_poisson, t_val_poisson, delta_val_poisson, cured_val_poisson = \
        generate_data(log_a_poisson, log_phi_poisson, theta_val_poisson, sup_poisson, low_c, high_c)
    # Poisson - Test data
    theta_test_poisson = get_theta(log_a_poisson, log_phi_poisson, C_poisson, C_inv_poisson, sup_poisson, p_test, theta_min = theta_min_poisson, theta_max = theta_max_poisson)
    m_test_poisson, t_test_poisson, delta_test_poisson, cured_test_poisson = \
        generate_data(log_a_poisson, log_phi_poisson, theta_test_poisson, sup_poisson, low_c, high_c)
    # Save the DataFrame with the simulated values for the Poisson
    poisson_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_poisson, theta_val_poisson, theta_test_poisson,
        m_train_poisson, m_val_poisson, m_test_poisson,
        t_train_poisson, t_val_poisson, t_test_poisson,
        delta_train_poisson, delta_val_poisson, delta_test_poisson
    )
    poisson_data.to_csv("{}/poisson/{}".format(directory, filename), index = False)
    
    # ---------------------------- Logarithmic ----------------------------
    # Logarithmic - Training data
    theta_train_log = get_theta(log_a_log, log_phi_log, C_log, C_inv_log, sup_log, p_train, theta_min = theta_min_log, theta_max = theta_max_log)
    m_train_log, t_train_log, delta_train_log, cured_train_log = \
        generate_data(log_a_log, log_phi_log, theta_train_log, sup_log, low_c, high_c)
    # Logarithmic - Validation data
    theta_val_log = get_theta(log_a_log, log_phi_log, C_log, C_inv_log, sup_log, p_val, theta_min = theta_min_log, theta_max = theta_max_log)
    m_val_log, t_val_log, delta_val_log, cured_val_log = \
        generate_data(log_a_log, log_phi_log, theta_val_log, sup_log, low_c, high_c)
    # Logarithmic - Test data
    theta_test_log = get_theta(log_a_log, log_phi_log, C_log, C_inv_log, sup_log, p_test, theta_min = theta_min_log, theta_max = theta_max_log)
    m_test_log, t_test_log, delta_test_log, cured_test_log = \
        generate_data(log_a_log, log_phi_log, theta_test_log, sup_log, low_c, high_c)
    # Save the DataFrame with the simulated values for the Logarithmic
    log_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_log, theta_val_log, theta_test_log,
        m_train_log, m_val_log, m_test_log,
        t_train_log, t_val_log, t_test_log,
        delta_train_log, delta_val_log, delta_test_log
    )
    log_data.to_csv("{}/logarithmic/{}".format(directory, filename), index = False)

    # ---------------------------- Geometric ----------------------------
    # Geometric - Training data
    theta_train_geo = get_theta(log_a_mvnb(1), log_phi_mvnb(1), C_mvnb(1), C_inv_mvnb(1), sup_mvnb, p_train, theta_min = theta_min_mvnb, theta_max = theta_max_mvnb)
    m_train_geo, t_train_geo, delta_train_geo, cured_train_geo = \
        generate_data(log_a_mvnb(1), log_phi_mvnb(1), theta_train_geo, sup_mvnb, low_c, high_c)
    # Geometric - Validation data
    theta_val_geo = get_theta(log_a_mvnb(1), log_phi_mvnb(1), C_mvnb(1), C_inv_mvnb(1), sup_mvnb, p_val, theta_min = theta_min_mvnb, theta_max = theta_max_mvnb)
    m_val_geo, t_val_geo, delta_val_geo, cured_val_geo = \
        generate_data(log_a_mvnb(1), log_phi_mvnb(1), theta_val_geo, sup_mvnb, low_c, high_c)
    # Geometric - Test data
    theta_test_geo = get_theta(log_a_mvnb(1), log_phi_mvnb(1), C_mvnb(1), C_inv_mvnb(1), sup_mvnb, p_test, theta_min = theta_min_mvnb, theta_max = theta_max_mvnb)
    m_test_geo, t_test_geo, delta_test_geo, cured_test_geo = \
        generate_data(log_a_mvnb(1), log_phi_mvnb(1), theta_test_geo, sup_mvnb, low_c, high_c)
    # Save the DataFrame with the simulated values for the Geometric
    geo_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_geo, theta_val_geo, theta_test_geo,
        m_train_geo, m_val_geo, m_test_geo,
        t_train_geo, t_val_geo, t_test_geo,
        delta_train_geo, delta_val_geo, delta_test_geo
    )
    geo_data.to_csv("{}/geometric/{}".format(directory, filename), index = False)
    
    # ---------------------------- NB(q = 2) ----------------------------
    # NB(2) - Training data
    # theta_train_2nb = get_theta(log_a_nb(2), log_phi_nb(2), C_nb(2), C_inv_nb(2), sup_nb, p_train, theta_min = theta_min_nb, theta_max = theta_max_nb)
    # m_train_2nb, t_train_2nb, delta_train_2nb, cured_train_2nb = \
    #     generate_data(log_a_nb(1), log_phi_nb(1), theta_train_2nb, sup_nb, low_c, high_c)
    # # NB(2) - Validation data
    # theta_val_2nb = get_theta(log_a_nb(2), log_phi_nb(2), C_nb(2), C_inv_nb(2), sup_nb, p_val, theta_min = theta_min_nb, theta_max = theta_max_nb)
    # m_val_2nb, t_val_2nb, delta_val_2nb, cured_val_2nb = \
    #     generate_data(log_a_nb(2), log_phi_nb(2), theta_val_2nb, sup_nb, low_c, high_c)
    # # NB(2) - Test data
    # theta_test_2nb = get_theta(log_a_nb(2), log_phi_nb(2), C_nb(2), C_inv_nb(2), sup_nb, p_test, theta_min = theta_min_nb, theta_max = theta_max_nb)
    # m_test_2nb, t_test_2nb, delta_test_2nb, cured_test_2nb = \
    #     generate_data(log_a_nb(2), log_phi_nb(2), theta_test_2nb, sup_nb, low_c, high_c)
    # # Save the DataFrame with the simulated values for the NB(2)
    # nb2_data = join_datasets(
    #     n_train, n_val, n_test,
    #     theta_train_2nb, theta_val_2nb, theta_test_2nb,
    #     m_train_2nb, m_val_2nb, m_test_2nb,
    #     t_train_2nb, t_val_2nb, t_test_2nb,
    #     delta_train_2nb, delta_val_2nb, delta_test_2nb
    # )
    # nb2_data.to_csv("{}/nb2/{}".format(directory, filename), index = False)

    # ---------------------------- MVNB(q = 1/2) ----------------------------
    # MVNB(2) - Training data
    theta_train_2mvnb = get_theta(log_a_mvnb(1/2), log_phi_mvnb(1/2), C_nb(1/2), C_inv_mvnb(1/2), sup_mvnb, p_train, theta_min = theta_min_mvnb, theta_max = theta_max_mvnb)
    m_train_2mvnb, t_train_2mvnb, delta_train_2mvnb, cured_train_2mvnb = \
        generate_data(log_a_mvnb(1/2), log_phi_mvnb(1/2), theta_train_2mvnb, sup_mvnb, low_c, high_c)
    # MVNB(2) - Validation data
    theta_val_2mvnb = get_theta(log_a_mvnb(1/2), log_phi_mvnb(1/2), C_mvnb(1/2), C_inv_mvnb(1/2), sup_mvnb, p_val, theta_min = theta_min_mvnb, theta_max = theta_max_mvnb)
    m_val_2mvnb, t_val_2mvnb, delta_val_2mvnb, cured_val_2mvnb = \
        generate_data(log_a_mvnb(1/2), log_phi_mvnb(1/2), theta_val_2mvnb, sup_mvnb, low_c, high_c)
    # MVNB(2) - Test data
    theta_test_2mvnb = get_theta(log_a_mvnb(1/2), log_phi_mvnb(1/2), C_mvnb(1/2), C_inv_mvnb(1/2), sup_mvnb, p_test, theta_min = theta_min_mvnb, theta_max = theta_max_mvnb)
    m_test_2mvnb, t_test_2mvnb, delta_test_2mvnb, cured_test_2mvnb = \
        generate_data(log_a_mvnb(1/2), log_phi_mvnb(1/2), theta_test_2mvnb, sup_mvnb, low_c, high_c)
    # Save the DataFrame with the simulated values for the MVNB(2)
    nb2_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_2mvnb, theta_val_2mvnb, theta_test_2mvnb,
        m_train_2mvnb, m_val_2mvnb, m_test_2mvnb,
        t_train_2mvnb, t_val_2mvnb, t_test_2mvnb,
        delta_train_2mvnb, delta_val_2mvnb, delta_test_2mvnb
    )
    nb2_data.to_csv("{}/mvnb2/{}".format(directory, filename), index = False)
    
    
    # ---------------------------- Bernoulli ----------------------------
    # Bernoulli - Training data
    theta_train_bern = get_theta(log_a_bin(1), log_phi_bin(1), C_bin(1), C_inv_bin(1), sup_bin(1), p_train, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_train_bern, t_train_bern, delta_train_bern, cured_train_bern = \
        generate_data(log_a_bin(1), log_phi_bin(1), theta_train_bern, sup_bin(1), low_c, high_c)
    # Bernoulli - Validation data
    theta_val_bern = get_theta(log_a_bin(1), log_phi_bin(1), C_bin(1), C_inv_bin(1), sup_bin(1), p_val, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_val_bern, t_val_bern, delta_val_bern, cured_val_bern = \
        generate_data(log_a_bin(1), log_phi_bin(1), theta_val_bern, sup_bin(1), low_c, high_c)
    # Bernoulli - Test data
    theta_test_bern = get_theta(log_a_bin(1), log_phi_bin(1), C_bin(1), C_inv_bin(1), sup_bin(1), p_test, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_test_bern, t_test_bern, delta_test_bern, cured_test_bern = \
        generate_data(log_a_bin(1), log_phi_bin(1), theta_test_bern, sup_bin(1), low_c, high_c)
    # Save the DataFrame with the simulated values for the Bernoulli
    bern_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_bern, theta_val_bern, theta_test_bern,
        m_train_bern, m_val_bern, m_test_bern,
        t_train_bern, t_val_bern, t_test_bern,
        delta_train_bern, delta_val_bern, delta_test_bern
    )
    bern_data.to_csv("{}/bernoulli/{}".format(directory, filename), index = False)

    # ---------------------------- Binomial (q = 5) ----------------------------
    # Binomial(5) - Training data
    theta_train_5bin = get_theta(log_a_bin(5), log_phi_bin(5), C_bin(5), C_inv_bin(5), sup_bin(5), p_train, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_train_5bin, t_train_5bin, delta_train_5bin, cured_train_5bin = \
        generate_data(log_a_bin(5), log_phi_bin(5), theta_train_5bin, sup_bin(5), low_c, high_c)
    # Binomial(5) - Validation data
    theta_val_5bin = get_theta(log_a_bin(5), log_phi_bin(5), C_bin(5), C_inv_bin(5), sup_bin(5), p_val, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_val_5bin, t_val_5bin, delta_val_5bin, cured_val_5bin = \
        generate_data(log_a_bin(5), log_phi_bin(5), theta_val_5bin, sup_bin(5), low_c, high_c)
    # Binomial(5) - Test data
    theta_test_5bin = get_theta(log_a_bin(5), log_phi_bin(5), C_bin(5), C_inv_bin(5), sup_bin(5), p_test, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_test_5bin, t_test_5bin, delta_test_5bin, cured_test_5bin = \
        generate_data(log_a_bin(5), log_phi_bin(5), theta_test_5bin, sup_bin(5), low_c, high_c)
    # Save the DataFrame with the simulated values for the Bin(5)
    bin5_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_5bin, theta_val_5bin, theta_test_5bin,
        m_train_5bin, m_val_5bin, m_test_5bin,
        t_train_5bin, t_val_5bin, t_test_5bin,
        delta_train_5bin, delta_val_5bin, delta_test_5bin
    )
    bin5_data.to_csv("{}/bin5/{}".format(directory, filename), index = False)


def sample_single_bootstrap_scenario2(i_train, i_test, n_train, n_val, n_test, cure_probs_dict_vec, directory, file_index):
    '''
        Get a single bootstrap sample from the Fashion-MNIST dataset considering each distribution from scenario 1.
    '''
    n = n_train + n_val + n_test
    filename = "data_{}.csv".format(file_index)

    # ---------------------------- Sample the indices from the original dataset ----------------------------
    
    # Indices for train and validation
    i_train_val = np.random.choice(i_train, size = n_train + n_val, replace = True)
    i_test = np.random.choice(i_test, size = n_test, replace = True)

    sets = np.concatenate([np.repeat("train", n_train), np.repeat("val", n_val), np.repeat("test", n_test)])
    indices = np.concatenate([i_train_val, i_test])
    pd.DataFrame({"index": indices, "set": sets}).to_csv("{}/indices_{}.csv".format(directory, file_index))

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

    # ---------------------------- Borel ----------------------------
    # Borel - Training data
    theta_train_borel = get_theta(log_a_borel, log_phi_borel, C_borel, C_inv_borel, sup_borel, p_train, theta_min = theta_min_borel, theta_max = theta_max_borel)
    m_train_borel, t_train_borel, delta_train_borel, cured_train_borel = \
        generate_data(log_a_borel, log_phi_borel, theta_train_borel, sup_borel, low_c, high_c)
    # Borel - Validation data
    theta_val_borel = get_theta(log_a_borel, log_phi_borel, C_borel, C_inv_borel, sup_borel, p_val, theta_min = theta_min_borel, theta_max = theta_max_borel)
    m_val_borel, t_val_borel, delta_val_borel, cured_val_borel = \
        generate_data(log_a_borel, log_phi_borel, theta_val_borel, sup_borel, low_c, high_c)
    # Borel - Test data
    theta_test_borel = get_theta(log_a_borel, log_phi_borel, C_borel, C_inv_borel, sup_borel, p_test, theta_min = theta_min_borel, theta_max = theta_max_borel)
    m_test_borel, t_test_borel, delta_test_borel, cured_test_borel = \
        generate_data(log_a_borel, log_phi_borel, theta_test_borel, sup_borel, low_c, high_c)
    # Save the DataFrame with the simulated values for the Borel
    borel_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_borel, theta_val_borel, theta_test_borel,
        m_train_borel, m_val_borel, m_test_borel,
        t_train_borel, t_val_borel, t_test_borel,
        delta_train_borel, delta_val_borel, delta_test_borel
    )
    borel_data.to_csv("{}/borel/{}".format(directory, filename), index = False)

    # ---------------------------- RGP(q = 2) ----------------------------
    # RGP(2) - Training data
    theta_train_rgp2 = get_theta(log_a_rgp(2), log_phi_rgp(2), C_rgp(2), C_inv_rgp(2), sup_rgp(2), p_train, theta_min = theta_min_rgp, theta_max = theta_max_rgp(2))
    m_train_rgp2, t_train_rgp2, delta_train_rgp2, cured_train_rgp2 = \
        generate_data(log_a_rgp(2), log_phi_rgp(2), theta_train_rgp2, sup_rgp(2), low_c, high_c)
    # RGP(2) - Validation data
    theta_val_rgp2 = get_theta(log_a_rgp(2), log_phi_rgp(2), C_rgp(2), C_inv_rgp(2), sup_rgp(2), p_val, theta_min = theta_min_rgp, theta_max = theta_max_rgp(2))
    m_val_rgp2, t_val_rgp2, delta_val_rgp2, cured_val_rgp2 = \
        generate_data(log_a_rgp(2), log_phi_rgp(2), theta_val_rgp2, sup_rgp(2), low_c, high_c)
    # RGP(2) - Test data
    theta_test_rgp2 = get_theta(log_a_rgp(2), log_phi_rgp(2), C_rgp(2), C_inv_rgp(2), sup_rgp(2), p_test, theta_min = theta_min_rgp, theta_max = theta_max_rgp(2))
    m_test_rgp2, t_test_rgp2, delta_test_rgp2, cured_test_rgp2 = \
        generate_data(log_a_rgp(2), log_phi_rgp(2), theta_test_rgp2, sup_rgp(2), low_c, high_c)
    # Save the DataFrame with the simulated values for the RGP(2)
    rgp2_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_rgp2, theta_val_rgp2, theta_test_rgp2,
        m_train_rgp2, m_val_rgp2, m_test_rgp2,
        t_train_rgp2, t_val_rgp2, t_test_rgp2,
        delta_train_rgp2, delta_val_rgp2, delta_test_rgp2
    )
    rgp2_data.to_csv("{}/rgp2/{}".format(directory, filename), index = False)

    # ---------------------------- Haight ----------------------------
    # Haight - Training data
    theta_train_haight = get_theta(log_a_haight, log_phi_haight, C_haight, C_inv_haight, sup_haight, p_train, theta_min = theta_min_haight, theta_max = theta_max_haight)
    m_train_haight, t_train_haight, delta_train_haight, cured_train_haight = \
        generate_data(log_a_haight, log_phi_haight, theta_train_haight, sup_haight, low_c, high_c)
    # Haight - Validation data
    theta_val_haight = get_theta(log_a_haight, log_phi_haight, C_haight, C_inv_haight, sup_haight, p_val, theta_min = theta_min_haight, theta_max = theta_max_haight)
    m_val_haight, t_val_haight, delta_val_haight, cured_val_haight = \
        generate_data(log_a_haight, log_phi_haight, theta_val_haight, sup_haight, low_c, high_c)
    # Haight - Test data
    theta_test_haight = get_theta(log_a_haight, log_phi_haight, C_haight, C_inv_haight, sup_haight, p_test, theta_min = theta_min_haight, theta_max = theta_max_haight)
    m_test_haight, t_test_haight, delta_test_haight, cured_test_haight = \
        generate_data(log_a_haight, log_phi_haight, theta_test_haight, sup_haight, low_c, high_c)
    # Save the DataFrame with the simulated values for the Haight
    haight_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_haight, theta_val_haight, theta_test_haight,
        m_train_haight, m_val_haight, m_test_haight,
        t_train_haight, t_val_haight, t_test_haight,
        delta_train_haight, delta_val_haight, delta_test_haight
    )
    haight_data.to_csv("{}/haight/{}".format(directory, filename), index = False)

    # ---------------------------- Geeta(q = 3) ----------------------------
    # Geeta(3) - Training data
    theta_train_geeta3 = get_theta(log_a_geeta(3), log_phi_geeta(3), C_geeta(3), C_inv_geeta(3), sup_geeta, p_train, theta_min = theta_min_geeta, theta_max = theta_max_geeta(3))
    m_train_geeta3, t_train_geeta3, delta_train_geeta3, cured_train_geeta3 = \
        generate_data(log_a_geeta(3), log_phi_geeta(3), theta_train_geeta3, sup_geeta, low_c, high_c)
    # Geeta(3) - Validation data
    theta_val_geeta3 = get_theta(log_a_geeta(3), log_phi_geeta(3), C_geeta(3), C_inv_geeta(3), sup_geeta, p_val, theta_min = theta_min_geeta, theta_max = theta_max_geeta(3))
    m_val_geeta3, t_val_geeta3, delta_val_geeta3, cured_val_geeta3 = \
        generate_data(log_a_geeta(3), log_phi_geeta(3), theta_val_geeta3, sup_geeta, low_c, high_c)
    # Geeta(3) - Test data
    theta_test_geeta3 = get_theta(log_a_geeta(3), log_phi_geeta(3), C_geeta(3), C_inv_geeta(3), sup_geeta, p_test, theta_min = theta_min_geeta, theta_max = theta_max_geeta(3))
    m_test_geeta3, t_test_geeta3, delta_test_geeta3, cured_test_geeta3 = \
        generate_data(log_a_geeta(3), log_phi_geeta(3), theta_test_geeta3, sup_geeta, low_c, high_c)
    # Save the DataFrame with the simulated values for the Geeta(3)
    geeta3_data = join_datasets(
        n_train, n_val, n_test,
        theta_train_geeta3, theta_val_geeta3, theta_test_geeta3,
        m_train_geeta3, m_val_geeta3, m_test_geeta3,
        t_train_geeta3, t_val_geeta3, t_test_geeta3,
        delta_train_geeta3, delta_val_geeta3, delta_test_geeta3
    )
    geeta3_data.to_csv("{}/geeta3/{}".format(directory, filename), index = False)

    
if(__name__ == "__main__"):

    deleted_directory = False
    
    print("Do you want to delete all the directories before starting the generation?")
    r = input()
    if(r[0] == "y"):
        shutil.rmtree("SimulationDataset")
        deleted_directory = True
    
    np.random.seed(111)
    
    print("------------------------------------------------------------------------------------")
    
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    i_valid_train = pd.Series(train_labels).isin([0,1,2,3,4]).to_numpy()
    i_valid_test = pd.Series(test_labels).isin([0,1,2,3,4]).to_numpy()

    # Filters to take only the images with labels in [0, 1, 2, 3, 4]
    train_labels = train_labels[i_valid_train]
    test_labels = test_labels[i_valid_test]

    # Indices for each set of filtered data
    i_train = np.arange(train_labels.shape[0])
    i_test = np.arange(test_labels.shape[0])

    print("Dimension of the train set: {}".format(train_images.shape))
    print("Dimension of the test set: {}".format(test_images.shape))
    print("------------------------------------------------------------------------------------")
    
    print("Creating directories structures")
    dists_scenario1 = ["poisson", "logarithmic", "geometric", "mvnb2", "bernoulli", "bin5"]
    dists_scenario2 = ["borel", "rgp2", "rgp10", "haight", "geeta3"]
    for dist in dists_scenario1:
        Path("SimulationDataset/Scenario1/n500/{}".format(dist)).mkdir(parents=True, exist_ok=True)
        Path("SimulationDataset/Scenario1/n1000/{}".format(dist)).mkdir(parents=True, exist_ok=True)
        Path("SimulationDataset/Scenario1/n3000/{}".format(dist)).mkdir(parents=True, exist_ok=True)
    for dist in dists_scenario2:
        Path("SimulationDataset/Scenario2/n500/{}".format(dist)).mkdir(parents=True, exist_ok=True)
        Path("SimulationDataset/Scenario2/n1000/{}".format(dist)).mkdir(parents=True, exist_ok=True)
        Path("SimulationDataset/Scenario2/n3000/{}".format(dist)).mkdir(parents=True, exist_ok=True)

    if(not deleted_directory):
        print("Do you want to run scenario 1? It may delete files that already exist!")
        r = input()
    # If the directory was deleted, it's better to just run everything without having to ask the user
    if(deleted_directory or r[0].lower() == "y"):
        print("---------------------------- Scenario 1 ----------------------------")
        cure_probs_dict1 = {0: 0.9, 1:0.45, 2:0.22, 3:0.14, 4: 0.08}
        cure_probs_dict1 = np.vectorize(cure_probs_dict1.get)
    
        train_sizes = [500, 1000, 3000] # 3 sample sizes
        val_sizes = [108, 214, 643]
        test_sizes = [108, 214, 643]
        
        n_replicates = 100
        
        for j in range(len(train_sizes)):
            n_train = train_sizes[j]
            n_val = val_sizes[j]
            n_test = test_sizes[j]
            print("n = {}".format(n_train))
            for i in tqdm(range(n_replicates)):
                sample_single_bootstrap_scenario1(
                    i_train,
                    i_test,
                    n_train, n_val, n_test,
                    cure_probs_dict_vec = cure_probs_dict1,
                    directory = "SimulationDataset/Scenario1/n{}".format(n_train), file_index = i+1
                )

    np.random.seed(222)

    if(not deleted_directory):
        print("Do you want to run scenario 2? It may delete files that already exist!")
        r = input()
    if(deleted_directory or r[0].lower() == "y"):
        print("---------------------------- Scenario 2 ----------------------------")
        cure_probs_dict2 = {0: 0.95, 1:0.85, 2:0.75, 3:0.65, 4: 0.55}
        cure_probs_dict2 = np.vectorize(cure_probs_dict2.get)
    
        train_sizes = [500, 1000, 3000] # 3 sample sizes
        val_sizes = [108, 214, 643]
        test_sizes = [108, 214, 643]
    
        n_replicates = 100
        
        for j in range(len(train_sizes)):
            n_train = train_sizes[j]
            n_val = val_sizes[j]
            n_test = test_sizes[j]
            print("n = {}".format(n_train))
            for i in tqdm(range(n_replicates)):
                sample_single_bootstrap_scenario2(
                    i_train,
                    i_test,
                    n_train, n_val, n_test,
                    cure_probs_dict_vec = cure_probs_dict2,
                    directory = "SimulationDataset/Scenario2/n{}".format(n_train), file_index = i+1
                )
