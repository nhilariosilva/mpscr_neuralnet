import os
import sys
import json
import logging

import numpy as np
import pandas as pd

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from keras import models, layers, initializers, optimizers, losses
from keras.callbacks import Callback

from tqdm.keras import TqdmCallback
from tqdm import tqdm

from net_model import *
from custom_model import *
from mps_models import *
import pwexp
import mps

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

data_dir = "EM_data"

# Argumentos para o algoritmo EM
func_args = load_args(filename = "{}/EM_args.json".format(data_dir))

log_a_str = func_args["log_a_str"]
log_phi_str = func_args["log_phi_str"]
C_str = func_args["C_str"]
C_inv_str = func_args["C_inv_str"]
sup_str = func_args["sup_str"]
theta_min = func_args["theta_min"]
theta_max = func_args["theta_max"]

max_iterations = func_args["max_iterations"]
early_stopping_em = func_args["early_stopping_em"]
early_stopping_em_warmup = func_args["early_stopping_em_warmup"]
early_stopping_em_eps = func_args["early_stopping_em_eps"]
epochs = func_args["epochs"]
batch_size = func_args["batch_size"]
shuffle = func_args["shuffle"]
learning_rate = func_args["learning_rate"]
run_eagerly = func_args["run_eagerly"]
gradient_accumulation_steps = func_args["gradient_accumulation_steps"]
early_stopping_nn = func_args["early_stopping_nn"]
early_stopping_min_delta_nn = func_args["early_stopping_min_delta_nn"]
early_stopping_patience_nn = func_args["early_stopping_patience_nn"]
reduce_lr = func_args["reduce_lr"]
reduce_lr_steps = func_args["reduce_lr_steps"]
reduce_lr_factor = func_args["reduce_lr_factor"]
validation = func_args["validation"]
val_prop = func_args["val_prop"]
verbose = func_args["verbose"]
seed = func_args["seed"]
alpha_known = func_args["alpha_known"]

set_all_seeds(seed)

# Carregamento dos dados de treinamento e validação iniciais
data = np.load("{}/data.npz".format(data_dir))
x = tf.cast(data["x"], dtype = tf.float32)
t = data["t"]
delta = data["delta"]
m = data["m"]
# Carregamento dos dados de validação
x_val = data["x_val"]
t_val = data["t_val"]
delta_val = data["delta_val"]
m_val = data["m_val"]
# Arquivos .npz não reconhecem None, por isso a conversão
if(len(x_val.shape) == 0):
    x_val = None
    t_val = None
    delta_val = None
    m_val = None
else:
     x_val = tf.cast(x_val, dtype = tf.float64)

# Cria as funções referentes ao modelo MPS para inicializar a rede neural
log_a_tf = eval(log_a_str)
log_phi_tf = eval(log_phi_str)
C_tf = eval(C_str)
C_inv_tf = eval(C_inv_str)
sup_tf = eval(sup_str)

# Carregamento do modelo - Os pesos são os mesmos do modelo inicial recebido como argumento
model = MPScrModel(log_a_tf, log_phi_tf, C_tf, C_inv_tf, sup_tf, theta_min, theta_max, "logit", verbose = verbose)
model.define_structure(shape_input = x[0].shape)
model.load_model("{}/model.weights.h5".format(data_dir))

# Carregamento do parâmetro inicial alpha e nós s
alpha, s = load_alpha_s(filename = "{}/alpha_s.csv".format(data_dir))
alpha = alpha.to_numpy()
s = s.to_numpy()

m_pred = tf.constant(m, dtype = tf.float64, shape = (m.shape[0], 1))
m_history = [m]
m_val_history = [m_val]
alpha_history = [alpha]
distances = [0.0]

for i in range(max_iterations):
    # Reduz a taxa de aprendizado gradativamente ao longo das epochs
    if(reduce_lr and i > 0 and i % reduce_lr_steps == 0):
        learning_rate = learning_rate*reduce_lr_factor
        if(verbose >= 2):
            print("Learning rate reduzida para {}".format(learning_rate))
    
    if(verbose >= 2):
        print("Iniciando passo {}".format(i+1))
    
    # Por padrão, implementamos o EM considerando a redução da taxa de aprendizado ao longo de iterações EM e não necessariamente ao longo de épocas em cada passo M
    # Por isso, reduce_lr = False
    results = {}

    # Salva os pesos do modelo para o cálculo da distância percorrida após o treinamento
    vec_weights_model = vec_weights(model)
    
    # Executa o r-ésimo passo do algoritmo EM
    # A redução da learning_rate é controlada diretamente neste loop, não sendo necessária em cada passo do algoritmo
    r_step_results = EM_rstep(model = model, alpha = alpha, s = s,
                              x = x, t = t, delta = delta, m_r = m_pred,
                              epochs = epochs, batch_size = batch_size, shuffle = shuffle,
                              learning_rate = learning_rate, run_eagerly = run_eagerly, gradient_accumulation_steps = gradient_accumulation_steps,
                              early_stopping = early_stopping_nn, early_stopping_min_delta = early_stopping_min_delta_nn, early_stopping_patience = early_stopping_patience_nn,
                              reduce_lr = False, reduce_lr_factor = 0.5, reduce_lr_patience = 5, reduce_lr_warmup = 30,
                              validation = validation, val_prop = val_prop, x_val = x_val, m_val = m_val,
                              verbose = verbose, alpha_known = alpha_known)        
    # Modelo atualizado
    new_model = r_step_results["new_model"]
    # Vetor de parâmetros alpha atualizado
    new_alpha = r_step_results["new_alpha"]
    # Valores da perda para treinamento atualizado
    loss_values_train = r_step_results["new_model_history"]["loss"]
    
    # print("Novo modelo")
    # print(new_model.get_weights()[0][0])
    
    # Atualiza o vetor de variáveis latentes, M, para os dados de treino
    new_m = update_m_mps(new_model, new_alpha, s, x, t, delta)
    # Salva o novo vetor na lista de histórico
    m_history.append( new_m )
    
    # Se existem dados de validação, calcula o novo vetor de causas latentes para este conjunto também
    if(validation and m_val is not None):
        # Valores da perda para validação atualizado
        loss_values_val = r_step_results["new_model_history"]["loss_val"]
        # Atualiza o vetor de variáveis latentes, M, para os dados de validação
        new_m_val = update_m_mps(new_model, new_alpha, s, x_val, t_val, delta_val)
        # Salva o novo vetor na lista de histórico
        m_val_history.append(new_m_val)
        # Atualiza o vetor de variáveis latentes para o próximo passo do treinamento
        m_val = new_m_val.copy()
    
    # Salva os novos valores no histórico de treinamento
    new_model.save_model("{}/model_history/new_model_{}.weights.h5".format(data_dir, i+1))
    # Atualiza o novo vetor alppha na lista de histórico
    alpha_history.append( new_alpha.tolist() )
    # print( "new_m: {}".format(new_m[:20]) )
    
    # Cálculo da distância percorrida no passo
    vec_weights_new_model = vec_weights(new_model)
    
    # Escala as distâncias segundo a escala dos parâmetros no passo anterior
    distance_omega = np.mean( (vec_weights_new_model-vec_weights_model)**2 )
    distance_alpha = np.mean( (new_alpha-alpha)**2 )
    distance = (distance_omega + distance_alpha) / 2
    distances.append(distance)
    
    if(verbose >= 2):
        print("Distância Parâmetros Rede Neural: {}".format( distance_omega ))
        print("Distância Parâmetros Alpha: {}".format( distance_alpha ))
        print("Média das distâncias: {}".format(distance))
    
    # Verifica se o critério de parada já foi alcançado
    if(early_stopping_em and i >= early_stopping_em_warmup):
        if(distance < early_stopping_em_eps):
            if(verbose > 0):
                print("Algoritmo convergiu após {} iterações. Retornando.".format(i+1))
            new_model.save_model("{}/new_model.weights.h5".format(data_dir))
            alpha_history = np.array(alpha_history)
            m_history = np.array(m_history)
            m_val_history = np.array(m_val_history)
            distances = np.array(distances)
            np.savez("{}/EM_results.npz".format(data_dir), alpha_history = alpha_history, m_history = m_history, m_val_history = m_val_history, distances = distances, converged = True, steps = i+1, loss_history = loss_values_train, loss_val_history = loss_values_val)
            # Encerra o programa
            sys.exit()
    
    # model = new_model.copy()
    model = new_model
    alpha = new_alpha.copy()
    m_pred = new_m.copy()

if(verbose > 0):
    print("Algoritmo não convergiu após {} iterações. Retornando.".format(max_iterations))
new_model.save_model("{}/new_model.weights.h5".format(data_dir))
alpha_history = np.array(alpha_history)
m_history = np.array(m_history)
distances = np.array(distances)
np.savez("{}/EM_results.npz".format(data_dir), alpha_history = alpha_history, m_history = m_history, m_val_history = m_val_history, distances = distances, converged = False, steps = i+1, loss_history = loss_values_train, loss_val_history = loss_values_val)
# Encerra o programa
sys.exit()



    