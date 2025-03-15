import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

from time import time

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

import os, shutil
from pathlib import Path
import json
import subprocess

from net_model import *
from custom_model import *
from mps_models import *

import mps
import pwexp

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

i_valid_train = pd.Series(train_labels).isin([0,1,2,3,4]).to_numpy()
i_valid_test = pd.Series(test_labels).isin([0,1,2,3,4]).to_numpy()

# Filters to take only the images with labels in [0, 1, 2, 3, 4]
train_images = train_images[i_valid_train]
train_images = train_images / np.max(train_images)
train_shape = train_images.shape
# Adds one more dimension for keras to identify the "colors" dimension
train_images = np.reshape(train_images, (train_shape[0], train_shape[1], train_shape[2], 1))

test_images = test_images[i_valid_test]
test_images = test_images / np.max(test_images)
test_shape = test_images.shape
# Adds one more dimension for keras to identify the "colors" dimension
test_images = np.reshape(test_images, (test_shape[0], test_shape[1], test_shape[2], 1))

train_labels = train_labels[i_valid_train]
test_labels = test_labels[i_valid_test]

def load_file(data_dir, file_index, distribution, train_images, test_images):
    '''
        Example:
            data_dir = "SimulationDataset/Scenario1/n500"
            file_index = 20
            distribution = "poisson"
    '''
    index_path = "{}/indices_{}.csv".format(data_dir, file_index, distribution)
    data_path = "{}/{}/data_{}.csv".format(data_dir, distribution, file_index)
    df_index = pd.read_csv(index_path)
    df_data = pd.read_csv(data_path)

    index_train = df_index.loc[df_index.set == "train","index"].to_numpy()
    index_val = df_index.loc[df_index.set == "val","index"].to_numpy()
    index_test = df_index.loc[df_index.set == "test","index"].to_numpy()

    # Values for the thetas
    theta_train = df_data.loc[df_data.set == "train", "theta"]
    theta_val = df_data.loc[df_data.set == "val", "theta"]
    theta_test = df_data.loc[df_data.set == "test", "theta"]
    # Values for the latent variable
    m_train = df_data.loc[df_data.set == "train", "m"]
    m_val = df_data.loc[df_data.set == "val", "m"]
    m_test = df_data.loc[df_data.set == "test", "m"]
    # Values for the time variable
    t_train = df_data.loc[df_data.set == "train", "t"]
    t_val = df_data.loc[df_data.set == "val", "t"]
    t_test = df_data.loc[df_data.set == "test", "t"]
    # Values for the censorship indicators
    delta_train = df_data.loc[df_data.set == "train", "delta"]
    delta_val = df_data.loc[df_data.set == "val", "delta"]
    delta_test = df_data.loc[df_data.set == "test", "delta"]

    img_train = train_images[index_train,:,:]
    img_val = train_images[index_val,:,:]
    img_test = test_images[index_test,:,:]

    result = {
        "theta_train": theta_train, "theta_val": theta_val, "theta_test": theta_test,
        "m_train": m_train, "m_val": m_val, "m_test": m_test,
        "t_train": t_train, "t_val": t_val, "t_test": t_test,
        "delta_train": delta_train, "delta_val": delta_val, "delta_test": delta_test,
        "img_train": img_train, "img_val": img_val, "img_test": img_test,
        "index_train": index_train, "index_val": index_val, "index_test": index_test
    }
    
    return result

def select_model(distribution, q):
    if(distribution == "poisson"):      
        log_a_str = log_a_poisson_str
        log_phi_str = log_phi_poisson_str
        C_str = C_poisson_str
        C_inv_str = C_inv_poisson_str
        sup_str = sup_poisson_str
        theta_min = None
        theta_max = None
    elif(distribution == "logarithmic"):
        log_a_str = log_a_log_str
        log_phi_str = log_phi_log_str
        C_str = C_log_str
        C_inv_str = C_inv_log_str
        sup_str = sup_log_str
        theta_min = 0
        theta_max = 1
    elif(distribution == "nb" or distribution == "mvnb"):
        if(q is None):
            raise Exception("Please, specify the fixed parameter (q) for the distribution.")
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant({}, dtype = tf.float64)".format(q)
        log_a_str = log_a_mvnb_str.format(q_argument)
        log_phi_str = log_phi_mvnb_str.format(q_argument)
        C_str = C_mvnb_str.format(q_argument)
        C_inv_str = C_inv_mvnb_str.format(q_argument)
        sup_str = sup_mvnb_str.format(q_argument)
        theta_min = None
        theta_max = None
    elif(distribution == "geometric"):
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant(1, dtype = tf.float64)"
        log_a_str = log_a_mvnb_str.format(q_argument)
        log_phi_str = log_phi_mvnb_str.format(q_argument)
        C_str = C_mvnb_str.format(q_argument)
        C_inv_str = C_inv_mvnb_str.format(q_argument)
        sup_str = sup_mvnb_str.format(q_argument)
        theta_min = None
        theta_max = None
    elif(distribution == "bin" or distribution == "binomial"): 
        if(q is None):
            raise Exception("Please, specify the fixed parameter (q) for the distribution.")
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant({}, dtype = tf.float64)".format(q)
        log_a_str = log_a_bin_str.format(q_argument)
        log_phi_str = log_phi_bin_str.format(q_argument)
        C_str = C_bin_str.format(q_argument)
        C_inv_str = C_inv_bin_str.format(q_argument)
        sup_str = sup_bin_str.format(q_argument)
        theta_min = 0
        theta_max = 1
    elif(distribution == "bernoulli"):
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant(1, dtype = tf.float64)"
        log_a_str = log_a_bin_str.format(q_argument)
        log_phi_str = log_phi_bin_str.format(q_argument)
        C_str = C_bin_str.format(q_argument)
        C_inv_str = C_inv_bin_str.format(q_argument)
        sup_str = sup_bin_str.format(q_argument)
        theta_min = 0
        theta_max = 1
    elif(distribution == "rgp"):
        if(q is None):
            raise Exception("Please, specify the fixed parameter (q) for the distribution.")
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant({}, dtype = tf.float64)".format(q)
        log_a_str = log_a_rgp_str.format(q_argument)
        log_phi_str = log_phi_rgp_str.format(q_argument)
        C_str = C_rgp_str.format(q_argument)
        C_inv_str = C_inv_rgp_str.format(q_argument)
        sup_str = sup_rgp_str.format(q_argument)
        theta_min = 0
        theta_max = np.abs(1/q)
    elif(distribution == "borel"):
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant(1, dtype = tf.float64)"
        log_a_str = log_a_rgp_str.format(q_argument)
        log_phi_str = log_phi_rgp_str.format(q_argument)
        C_str = C_rgp_str.format(q_argument)
        C_inv_str = C_inv_rgp_str.format(q_argument)
        sup_str = sup_rgp_str.format(q_argument)
        theta_min = 0
        theta_max = 1
    elif(distribution == "geeta"):
        if(q is None):
            raise Exception("Please, specify the fixed parameter (q) for the distribution.")
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant({}, dtype = tf.float64)".format(q)
        log_a_str = log_a_geeta_str.format(q_argument)
        log_phi_str = log_phi_geeta_str.format(q_argument)
        C_str = C_geeta_str.format(q_argument)
        C_inv_str = C_inv_geeta_str.format(q_argument)
        sup_str = sup_geeta_str.format(q_argument)
        theta_min = 0
        theta_max = np.abs(1/q)
    elif(distribution == "haight"):
        # In the EM.py file, we must ensure that q is of type tf.float64 for it to work properly
        q_argument = "tf.constant(2, dtype = tf.float64)"
        log_a_str = log_a_geeta_str.format(q_argument)
        log_phi_str = log_phi_geeta_str.format(q_argument)
        C_str = C_geeta_str.format(q_argument)
        C_inv_str = C_inv_geeta_str.format(q_argument)
        sup_str = sup_geeta_str.format(q_argument)
        theta_min = 0
        theta_max = 1/2

    return log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max

def fit_cure_model(distribution, q,
                   t_train, t_val,
                   delta_train, delta_val,
                   img_train, img_val,
                   max_iterations = 100,
                   early_stopping_em = True, early_stopping_em_warmup = 5, early_stopping_em_eps = 1.0e-6,
                   epochs = 100, batch_size = None, shuffle = True,
                   learning_rate = 0.001, run_eagerly = False,
                   early_stopping_nn = True, early_stopping_min_delta_nn = 0.0, early_stopping_patience_nn = 5,
                   reduce_lr = True, reduce_lr_steps = 10, reduce_lr_factor = 0.1,
                   verbose = 1, seed = 1):
    alpha0, s_t = initialize_alpha_s(t_train, n_cuts = 5)

    # Select the MPS functions based on the chosen distribution
    log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max = select_model(distribution, q)

    set_all_seeds(seed)
    # Because it only serves to initialize the model weights, the distribution does not matter in this case (that's why we use the Poisson here)
    dummy_mps_model = MPScrModel(log_a_poisson_tf, log_phi_poisson_tf, C_poisson_tf, C_inv_poisson_tf, sup_poisson)
    dummy_mps_model.define_structure(shape_input = img_train[0].shape, seed = seed)

    # If batch_size is null, use just one big batch
    if(batch_size is None):
        batch_size = len(t_train)
    
    results = call_EM("EM.py",
                      log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max,
                      dummy_mps_model, alpha0, s_t,
                      img_train, t_train, delta_train, delta_train,
                      max_iterations = max_iterations,
                      early_stopping_em = early_stopping_em, early_stopping_em_warmup = early_stopping_em_warmup, early_stopping_em_eps = early_stopping_em_eps,
                      epochs = epochs, batch_size = batch_size, shuffle = shuffle,
                      learning_rate = learning_rate, run_eagerly = run_eagerly,
                      early_stopping_nn = early_stopping_nn, early_stopping_min_delta_nn = early_stopping_min_delta_nn, early_stopping_patience_nn = early_stopping_patience_nn,
                      reduce_lr = reduce_lr, reduce_lr_steps = reduce_lr_steps, reduce_lr_factor = reduce_lr_factor,
                      validation = True,
                      x_val = img_val, t_val = t_val, delta_val = delta_val, m_val = delta_val,
                      verbose = verbose, seed = seed, alpha_known = False)
    return results



print("Creating directories structures")
dists_scenario1 = ["poisson", "logarithmic", "geometric", "mvnb2", "bernoulli", "bin5", "rgp10"]
dists_scenario2 = ["borel", "rgp2", "haight", "geeta3"]
for dist in dists_scenario1:
    for j in range(1,101):
        Path("SimulationResults/Scenario1/n500/{}/{}".format(dist,j)).mkdir(parents=True, exist_ok=True)
        Path("SimulationResults/Scenario1/n1000/{}/{}".format(dist,j)).mkdir(parents=True, exist_ok=True)
        Path("SimulationResults/Scenario1/n3000/{}/{}".format(dist,j)).mkdir(parents=True, exist_ok=True)
for dist in dists_scenario2:
    for j in range(1,101):
        Path("SimulationResults/Scenario2/n500/{}/{}".format(dist,j)).mkdir(parents=True, exist_ok=True)
        Path("SimulationResults/Scenario2/n1000/{}/{}".format(dist,j)).mkdir(parents=True, exist_ok=True)
        Path("SimulationResults/Scenario2/n3000/{}/{}".format(dist,j)).mkdir(parents=True, exist_ok=True)

def run_scenario(data_dir, distribution, q, train_images, test_images, start_index = 1, seed = 1):
    '''
        Example:
            data_dir = "SimulationDataset/Scenario1/n500"
            distribution = "poisson"
    '''

    # The name of the distribution does not have any numbers. Simulations like the RPG which are considered for two different values of q have different numbers associated to their directory, which must be conserved for the function to know where to save the files. But for selection of the model in select_model, we must use only "rgp"
    distribution_name = ''.join([i for i in distribution if not i.isdigit()])
    
    # Select the functions associated to the chosen distribution model
    log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max = select_model(distribution_name, q)
    log_a_tf = eval(log_a_str)
    log_phi_tf = eval(log_phi_str)
    C_tf = eval(C_str)
    C_inv_tf = eval(C_inv_str)
    sup_tf = eval(sup_str)
    
    execution_times = []
    loss_values = []
    loss_val_values = []
    converged = []
    steps = []
    for i in tqdm(range(start_index, 101)):
        # Load the simulated dataset
        sim_dataset = load_file(data_dir, i, distribution, train_images, test_images)
        
        _, s_t = initialize_alpha_s(sim_dataset["t_train"], n_cuts = 5)
        start_time = time()
        result = fit_cure_model(distribution_name, q,
                                sim_dataset["t_train"], sim_dataset["t_val"],
                                sim_dataset["delta_train"], sim_dataset["delta_val"],
                                sim_dataset["img_train"], sim_dataset["img_val"],
                                batch_size = None, max_iterations = 100,
                                seed = seed, verbose = 0)
        elapsed_time = time() - start_time
        execution_times.append(elapsed_time)
        
        # Recover information on the inference of parameters
        eta_train_pred = result["new_model"].predict(sim_dataset["img_train"], verbose = 0)
        eta_val_pred = result["new_model"].predict(sim_dataset["img_val"], verbose = 0)
        eta_test_pred = result["new_model"].predict(sim_dataset["img_test"], verbose = 0)
        
        p_train_pred = result["new_model"].link_func(eta_train_pred).numpy().flatten()
        p_val_pred = result["new_model"].link_func(eta_val_pred).numpy().flatten()
        p_test_pred = result["new_model"].link_func(eta_test_pred).numpy().flatten()
        p_pred = np.concatenate([p_train_pred, p_val_pred, p_test_pred])
        
        log_p_train_pred = np.log(p_train_pred)
        log_p_val_pred = np.log(p_val_pred)
        log_p_test_pred = np.log(p_test_pred)
        
        log_a0 = log_a_tf( tf.constant(0.0, dtype = tf.float64) )
        
        theta_train_pred = C_inv_tf( np.exp(log_a0 - log_p_train_pred) ).numpy()
        theta_val_pred = C_inv_tf( np.exp(log_a0 - log_p_val_pred) ).numpy()
        theta_test_pred = C_inv_tf( np.exp(log_a0 - log_p_test_pred) ).numpy()
        theta_pred = np.concatenate([theta_train_pred, theta_val_pred, theta_test_pred])
        
        m_train_pred = result["m_history"][-1]
        m_val_pred = result["m_val_history"][-1]
        m_test_pred = update_m_mps(result["new_model"], result["new_alpha"], s_t, sim_dataset["img_test"], sim_dataset["t_test"], sim_dataset["delta_test"])
        m_pred = np.concatenate([m_train_pred, m_val_pred, m_test_pred])

        sets = np.concatenate([np.repeat("train", len(theta_train_pred)), np.repeat("val", len(theta_val_pred)), np.repeat("test", len(theta_test_pred))])

        results_dir = data_dir.split("/")
        results_dir = "{}/{}".format("SimulationResults", "/".join(results_dir[1:]))
        pd.DataFrame({"p_pred": p_pred, "theta_pred": theta_pred, "m_pred": m_pred, "set": sets}).to_csv(
            "{}/{}/{}/data_pred.csv".format(results_dir, distribution, i), index = False
        )
        
        # Save the model weights
        result["new_model"].save_model("{}/{}/{}/model.weights.h5".format(results_dir, distribution, i))

        # Save the piecewise exponential estimated parameters
        save_alpha_s(result["new_alpha"], s_t, filename = "{}/{}/{}/alpha_s.csv".format(results_dir, distribution, i))

        loss_values.append( result["loss_history"][-1] )
        loss_val_values.append( result["loss_val_history"][-1] )
        converged.append( result["converged"] )
        steps.append( result["steps"] )

    pd.DataFrame({"execution_times": execution_times, "loss_values": loss_values, "loss_val_values": loss_val_values, "converged": converged, "steps": steps}).to_csv(
        "{}/{}/sim_metadata.csv".format(results_dir, distribution, i), index = False
    )

if(__name__ == "__main__"):
    seed = 100
    data_dir = input("Digite a pasta com os dados da simulação (Ex: SimulationDataset/Scenario2/n500): ")
    distribution = input("Digite a distribuição a ser considerada: ")
    q = input("Digite o valor do parâmetro adicional q (Se não houver, digite None): ")

    if(q.lower() == "none"):
        q = None
    else:
        q = float(eval(q))

    run_scenario(data_dir, distribution, q, train_images, test_images, start_index = 1, seed = 1)





