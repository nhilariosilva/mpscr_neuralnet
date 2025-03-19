import os, shutil
import json
import subprocess

import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from keras import models, layers, initializers, optimizers, losses

from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback
from tqdm import tqdm

import mps
import pwexp

import os, shutil
from pathlib import Path

tf.keras.backend.set_floatx('float64')

def set_all_seeds(seed=42):
    '''
        Define todas as sementes de interesse para garantir que os resultados do keras sejam reproduzíveis.
    '''
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic operations on GPU
    tf.config.experimental.enable_op_determinism()

def vec_weights(model):
    '''
        Dado um modelo de redes neuras, achata todos os parâmetros da rede em um único vetor e o retorna.
    '''
    weights = model.get_weights()
    vec_weights = np.array([])
    for i in range(len(weights)):
        vec_weights = np.concatenate([vec_weights, weights[i].flatten()])
    return vec_weights

# Reestrutura um vetor de pesos da rede neural vetorizado na estrutura de pesos do modelo fornecido
def structure_weights(vec, model):
    '''
        Dado um vetor de parâmetros achatado e um modelo de redes neurais, utiliza a estrutura dos parâmetros do modelo para recuperar a estrutura dos parâmetros anterior
    '''
    structured_weights = []
    i = 0
    for layer in model.get_weights():
        if(len(layer.shape) == 2):
            pars_count = layer.shape[0] * layer.shape[1]
        else:
            pars_count = layer.shape[0]
        structured_weights.append(
            np.reshape( vec[i:(i+pars_count)], newshape = layer.shape ).astype(np.float64)
        )
        
        i += pars_count
    return structured_weights
        
class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    def on_train_begin(self, logs = None):
        self.model.training_started = True
    
class MPScrModelStructure(keras.models.Model):    
    def __init__(self, log_a, log_phi, C, C_inv, sup, theta_min = None, theta_max = None, link = "logit", verbose = 0):
        super().__init__()
        # Salva as funções referentes ao modelo MPScr
        self.log_a = log_a
        self.log_phi = log_phi
        self.C = C
        self.C_inv = C_inv
        self.sup = sup
        
        self.link = link
        if(link == "logit"):
            self.link_func = lambda eta : 1 / (1 + tf.math.exp(-eta))
        else:
            raise("Função de ligação não implementada.")
        
        # Caso existam, define as limitações do espaço paramétrico referentes ao parâmetro theta
        self.theta_min = theta_min
        self.theta_max = theta_max
        
        self.r_min = None
        self.r_max = None
        self.r_mid = None
        # Se o espaço paramétrico é limitado, obtém os limites de variação para r = C(theta)
        if(theta_min is not None and theta_max is not None):
            C_theta_min = float( self.C(tf.constant(self.theta_min, dtype = tf.float64)) )
            C_theta_max = float( self.C(tf.constant(self.theta_max, dtype = tf.float64)) )
            self.r_min = np.min([C_theta_min, C_theta_max])
            self.r_max = np.max([C_theta_min, C_theta_max])
            self.r_mid = (self.r_min + self.r_max)/2

            # Se o r mínimo for 1 e o máximo for infinito, então mesmo havendo uma restrição em theta, não há uma restrição na probabilidade de cura, seguindo normalmente o processo de estimação. Se r_min e r_max forem nan, também temos que não há restrições (é o caso em que a probabilidade de cura não está definida para os valores de theta nos limites do espaço paramétrico)
            if( (np.isnan(self.r_min) or np.isnan(self.r_max)) or ((self.r_min-1.0)<1.0e-12 and tf.math.is_inf(self.r_max)) ):
                self.r_min = None
                self.r_max = None
                self.r_mid = None
            else:
                # A probabilidade associada ao maior valor da razão r = a_0 / p_0 é a menor probabilidade e a associada ao menor r é a maior probabilidade (inversamente proporcional) 
                p_theta_max = tf.math.exp( self.log_a(tf.constant(0.0, dtype = tf.float64)) - tf.math.log(tf.constant(C_theta_min, dtype = tf.float64)) )
                # Obtém a probabilidade p associada ao maior valor de C(theta)
                p_theta_min = tf.math.exp( self.log_a(tf.constant(0.0, dtype = tf.float64)) - tf.math.log(tf.constant(C_theta_max, dtype = tf.float64)) )
                if(verbose > 0):
                    print("******* Warning: The cure probability for this model lies between {:.6f} and {:.6f} *******".format(float(p_theta_min), float(p_theta_max)))
        
    def define_structure(self, seed = 1):
        '''
            Define toda a estrutura da rede neural. Caso tenha o interesse em modificar a estrutura do modelo, deverá ser criada uma nova classe que herda MPScrModelStructure e atualizar essa função, além da função call e copy.
        '''
        # Por padrão, supõe que o modelo recebe vetores unidimensionais
        self.shape_input = (1,)
        dummy_input = np.zeros(self.shape_input)
        dummy_input = tf.expand_dims(dummy_input, -1)

        # Declara a estrutura do modelo de redes neurais.
        # Por padrão é um simples modelo denso com 10 neurônios na camada oculta.
        initializer = initializers.HeNormal(seed = seed)
        self.in_layer = layers.InputLayer(input_shape = (1,))
        self.dense1 = layers.Dense(10, activation = keras.activations.sigmoid, kernel_initializer = initializer)
        self.out_layer = layers.Dense(1, activation = None, use_bias = False, kernel_initializer = initializer)
        
        # Inicializa os pesos do modelo ao fornecer dados dummy
        self(dummy_input)
        
    def call(self, x_input):
        if(len(x_input.shape) == 1):
            x_input = tf.expand_dims(x_input, -1)  # Adds a new dimension at the end
        x = self.in_layer(x_input)
        x = self.dense1(x)
        x = self.out_layer(x)
        return x
    
    def copy(self):
        # Cria um objeto da mesma classe, passando os parâmetros do objeto atual como inputs
        new_model = MPScrModelStructure(self.log_a, self.log_phi, self.C, self.C_inv, self.sup, self.theta_min, self.theta_max, self.link)
        new_model.define_structure()
        new_model.set_weights(self.get_weights())
        return new_model
    
    def save_model(self, filename):
        self.save_weights(filename)

    def load_model(self, filename):
        self.load_weights(filepath = filename)
        return self

    def get_config(self):
        pass
    
    @tf.function
    def calculate_loss_weights_mean(self, r0, m, log_p0):
        '''
            A partir da razão r0 = a0/p0 e do vetor de variáveis latentes estimado, m, calcula
        '''
        # Recupera os valores estimados de theta de cada indivíduo
        C_inv_r = self.C_inv(r0)
        # Obtém a expressão da verossimilhança desejada
        loss_weights = m * self.log_phi(C_inv_r) + log_p0
        # Adiciona à média das perdas as penalizações para o espaço paramétrico
        loss_weights_mean = -tf.math.reduce_mean(loss_weights)
        return loss_weights_mean
    
    # Por alguma razão, inverte a ordem dos argumentos dada pelo data_generator
    @tf.function
    def loss_func(self, m, eta):
        log_a0 = self.log_a(tf.constant(0.0, dtype = tf.float64))
        p0 = self.link_func(eta)
        log_p0 = tf.math.log( p0 )
        
        log_r0 = log_a0 - log_p0
        r0 = tf.math.exp(log_r0)
        
        # Caso existam limitações no espaço paramétrico, verifica se estão sendo cumpridas pelo modelo
        parametric_space_penalty = tf.constant(0.0, dtype = tf.float64)
        # parametric_space_penalty = tf.constant(0.0, dtype = tf.float32)
        if(self.r_min is not None and self.r_max is not None):
            # Penalização para valores de p abaixo de p_0_min
            # When using dynamic batch_size there is no way to obtain the shape of the input objects by m.shape for example. That's why we use
            # the tf.ones_like instead of something like tf.constant(self.r_min, shape = (m.shape[0], 1))
            r_min_vector = tf.ones_like(m) * self.r_min
            
            # r_min_vector = tf.constant(self.r_min, dtype = tf.float32, shape = (m.shape[0], 1))
            # Toma max( 0, min_p_0 - p_0 )^2 (eleva ao quadrado para suavizar o gradiente)
            # We sum 1.0e-12 to avoid problems of zero over zero division, resulting in nan values. There is no risk of denominator being 1.0e-12 while numerator different than zero
            penalty_lower = ( tf.maximum(tf.constant(0.0, dtype = tf.float64), r_min_vector - r0) / (r_min_vector - r0 + 1.0e-12) * (self.r_mid - r0) )**2
            mean_penalty_lower = tf.math.reduce_mean(penalty_lower)

            # Penalização para valores de p abaixo de p_0_min
            r_max_vector = tf.ones_like(m) * self.r_max
            # r_max_vector = tf.constant(self.r_max, dtype = tf.float32, shape = (m.shape[0], 1))
            # Toma max( 0, min_p_0 - p_0 )^2 (eleva ao quadrado para suavizar o gradiente)
            # We sum 1.0e-12 to avoid problems of zero over zero division, resulting in nan values. There is no risk of denominator being 1.0e-12 while numerator different than zero
            penalty_upper = ( tf.maximum(tf.constant(0.0, dtype = tf.float64), r0 - r_max_vector) / (r0 - r_max_vector + 1.0e-12) * (r0 - self.r_mid) )**2
            mean_penalty_upper = tf.math.reduce_mean(penalty_upper)

            # Se ao menos uma das penalizações for maior que 0, significa que temos ao menos uma estimativa de p0 fora do espaço paramétrico
            parametric_space_penalty = mean_penalty_lower + mean_penalty_upper

        # Caso parametric_space_penalty = 0, calcula a perda com base na verossimilhança (primeiro lambda), caso contrário, usa a perda referente ao espaço paramétrico (segundo lambda)
        # loss_value = tf.cond(
        #     tf.equal(parametric_space_penalty, 0.0),  # Condition
        #     lambda: self.calculate_loss_weights_mean(r0, m, p0),  # True branch
        #     lambda: parametric_space_penalty # False branch
        # )
        loss_value = tf.cond(
            tf.equal(parametric_space_penalty, 0.0),  # Condition
            lambda: self.calculate_loss_weights_mean(r0, m, log_p0),  # True branch
            lambda: parametric_space_penalty + 100 # False branch (sums a big number to avoid the model being stuck trying to stay at the border of the parametric space)
        )
        # Notice that if the proper loss weights is positive, it is probabily greater than the loss related to the parametric space, because the probability values will be closer to
        # the parametric space border already. That way, when there are problems with parametric space, we sum high values for this loss to avoid the model preferring to stay in the border.
        # If model stays by the border, the loss is definetly lower than expected, which is reasonable for the weights to want to stabilize there. That's why we sum a big number to it
        
        return loss_value

    def train_step(self, data):
        x_train, m_train = data
        
        if(not self.training_started):
            return {"loss": 0.0, "loss_val": 0.0}
        
        # Cálculo da função perda
        with tf.GradientTape() as tape:
            # Compute the loss value
            eta_train = self(x_train)
            loss_train = self.loss_func(m = m_train, eta = eta_train)
        
        if(self.validation):
            eta_val = self(self.x_val)
            loss_val = self.loss_func(m = self.m_val, eta = eta_val)
        else:
            loss_val = loss_train
            
        # Ontenção dos gradientes da função perda
        gradients = tape.gradient(loss_train, sources = self.trainable_variables)
        # Atualiza os pesos do modelo
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss_train, "loss_val": loss_val}
    
    def compile_model(self, learning_rate = 0.001, run_eagerly = False, gradient_accumulation_steps = None):
        self.compile(
            optimizer = optimizers.Adam(learning_rate = learning_rate, gradient_accumulation_steps = None),
            loss = self.loss_func,
            run_eagerly = run_eagerly
        )
        
    def train_model(self, x, m, epochs = 10, batch_size = None, shuffle = False,
                    early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10,
                    reduce_lr = False, reduce_lr_factor = 0.5, reduce_lr_patience = 5, reduce_lr_warmup = 30,
                    verbose = 2,
                    validation = False, val_prop = 0.2, x_val = None, m_val = None):
        '''
            Organiza os conjunto de treino e validação e inicia o treinamento da rede neural
        '''
        self.validation = validation        
        
        # Formata as variáveis x e m para o padrão do tensorflow
        x = tf.cast(x, dtype = tf.float64)
        if(len(x.shape) == 1):
            x = tf.reshape( x, shape = (len(x), 1) )
        m = tf.reshape(m, (m.shape[0], 1))
        m = tf.cast(m, tf.float64)

        # Salva os dados originais
        self.x = x
        self.m = m

        if(self.validation):
            if(x_val is not None and m_val is not None):
                # Se forem fornecido ambos os dados x_val e m_val
                
                # Formata as variáveis x e m para o padrão do tensorflow
                x_val = tf.cast(x_val, dtype = tf.float64)
                # Caso o input seja vetorial, converte para um vetor coluna
                if(len(x.shape) == 1):
                    x_val = tf.reshape( x_val, shape = (len(x_val), 1) )
                m_val = tf.reshape(m_val, (m_val.shape[0], 1))
                m_val = tf.cast(m_val, tf.float64)
                
                self.x_val = x_val
                self.m_val = m_val
                self.x_train, self.m_train = self.x, self.m
            else: 
                # Se deseja validação e não foram fornecidos os dados, seleciona val_prop * 100% das observações como dados de validação
                
                # Embaralha os dados de treinamento para a seleção do conjunto de validação
                self.indexes_train = np.arange(x.shape[0])
                if(shuffle):
                    self.indexes_train = tf.random.shuffle( self.indexes_train )
                x_shuffled = tf.gather( x, self.indexes_train )
                # O vetor de variáveis latentes é np.array e não um tf.Tensor
                m_shuffled = tf.gather( m, self.indexes_train )

                # Seleciona um conjunto fixo para a validação durante o treinamento
                val_size = int(x.shape[0] * val_prop)
                self.x_val, self.m_val = x_shuffled[:val_size], m_shuffled[:val_size]
                self.x_train, self.m_train = x_shuffled[val_size:], m_shuffled[val_size:]
        else:
            # Caso não se deseja executar a validação, o conjunto de treinamento é o mesmo que o conjunto de validação
            self.x_train, self.m_train = self.x, self.m
            self.x_val, self.m_val = self.x, self.m
        
        # Declara os callbacks do modelo
        self.callbacks = [CustomCallback()]
        
        if(verbose >= 1):
            self.callbacks.append( TqdmCallback(verbose=0) )
        
        if(early_stopping):
            # Parada precoce (Early stopping) - Evita overfitting e agiliza o treinamento
            es = keras.callbacks.EarlyStopping(monitor = 'loss_val',
                                               mode = "min",
                                               min_delta = early_stopping_min_delta,
                                               patience = early_stopping_patience,
                                               restore_best_weights = True)
            self.callbacks.append(es)
        
        if(reduce_lr):
            # Redução dinâmica da taxa de aprendizado do modelo ao longo das épocas
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss_val', factor = reduce_lr_factor, patience = reduce_lr_patience, cooldown = reduce_lr_warmup)
            self.callbacks.append(reduce_lr)
            
        if(verbose >= 3):
            print("Iniciando treinamento:")
            print("Tamanho da amostra de treino: {}".format(self.x_train.shape[0]))
            print("Tamanho da amostra de validação: {}".format(self.x_val.shape[0]))

        # If batch_size is unspecified, set it to be the training size. Note that decreasing the batch size to smaller values, such as 500 for example, has previously lead the
        # model to converge too early, leading to a lot of time of investigation. When dealing with neural networks in statistical models context, we recommend to use a single
        # batch in training. Alternatives in the case that the sample is too big might be to consider a "gradient accumulation" approach.
        self.batch_size = self.x_train.shape[0]
        if(batch_size is not None):
            self.batch_size = batch_size

        self.fit(
            self.x_train, self.m_train,
            epochs = epochs,
            verbose = 0,
            callbacks = self.callbacks,
            batch_size = self.batch_size,
            shuffle = shuffle
        )
        
        self.final_history = self.history.history


def update_alpha(alpha, s, t, delta, m_r):
    # print("ATUALIZANDO ALPHA")
    # print("tipo s:", type(s))
    # Tinha esquecido do flatten e fiquei dias procurando o erro ;)
    m_r = m_r.numpy().flatten()
    # Novo vetor de parâmetros alpha
    new_alpha = alpha.copy()
    # Obtém os índices de quais intervalos definidos por s cada observação pertence
    ind_t_g = np.searchsorted(s, t)-1
    for g in range(len(alpha)-1):
        i = (ind_t_g == g)
        # print("g = {}... Número de obs nesse intervalo: {}".format(g, np.sum(i)))
        num = np.sum( delta[i] )
        den = np.sum( m_r[i] * (t[i] - s[g]) ) + np.sum( m_r[ind_t_g > g] * (s[g+1] - s[g]) )
        if(den == 0.0):
            new_alpha[g] = alpha[g]
        else:
            # Obtém numerador e denominador para a atualização do parâmetro alpha_g
            new_alpha[g] = num / den
    return new_alpha

def update_omega(model, m_r, x,
                 epochs = 100, batch_size = None, shuffle = True,
                 learning_rate = 0.01, run_eagerly = False, gradient_accumulation_steps = None,
                 early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 10,
                 reduce_lr = False, reduce_lr_factor = 0.5, reduce_lr_patience = 5, reduce_lr_warmup = 30,
                 verbose = 1, validation = False, val_prop = 0.2, x_val = None, m_val = None):
    # print("ATUALIZANDO OMEGA")
    # Não usa a função copy para evitar a alocação desnecessária de memória
    new_model = model
    new_model.compile_model(learning_rate, run_eagerly, gradient_accumulation_steps)
    new_model.train_model(x, m_r, epochs = epochs, batch_size = batch_size,
                          shuffle = shuffle,
                          early_stopping = early_stopping,
                          early_stopping_min_delta = early_stopping_min_delta, early_stopping_patience = early_stopping_patience,
                          reduce_lr = reduce_lr, reduce_lr_factor = reduce_lr_factor, reduce_lr_patience = reduce_lr_patience, reduce_lr_warmup = reduce_lr_warmup,
                          verbose = verbose,
                          validation = validation, val_prop = val_prop, x_val = x_val, m_val = m_val)
    return new_model

# Um único passo do modelo EM
def EM_rstep(model, alpha, s,
             x, t, delta, m_r,
             epochs = 100, batch_size = None, shuffle = True,
             learning_rate = 0.01, run_eagerly = False, gradient_accumulation_steps = None,
             early_stopping = True, early_stopping_min_delta = 0.0, early_stopping_patience = 5,
             reduce_lr = False, reduce_lr_factor = 0.5, reduce_lr_patience = 5, reduce_lr_warmup = 30,
             validation = False, val_prop = 0.2, x_val = None, m_val = None,
             verbose = 2, alpha_known = False):
    '''
        Um único passo do algoritmo EM para o treinamento do modelo MPScr com redes neurais
    '''
    
    # Inclui o zero no vetor de pontos de corte, caso ainda não tenha
    m_r = tf.reshape(m_r, shape = (m_r.shape[0], 1))
    
    # -------------- Atualização do vetor de parâmetros alpha --------------
    # Novo vetor de parâmetros alpha
    new_alpha = alpha.copy()
    if(not alpha_known):
        new_alpha = update_alpha(alpha, s, t, delta, m_r)

    # -------------- Atualização dos parâmetros da rede neural --------------
    new_model = update_omega(model, m_r, x,
                             epochs, batch_size, shuffle,
                             learning_rate, run_eagerly, gradient_accumulation_steps,
                             early_stopping, early_stopping_min_delta, early_stopping_patience,
                             reduce_lr, reduce_lr_factor, reduce_lr_patience, reduce_lr_warmup,
                             verbose, validation, val_prop, x_val, m_val)
    results = {
        "new_model": new_model,
        "new_alpha": new_alpha,
        "new_model_history": new_model.final_history
    }
    
    return results

# Função de sobrevivência base do modelo (exponencial por partes)
def S1(t, alpha, s, include_zero = False):
    return pwexp.cdf(t, alpha, s, lower_tail = False, include_zero = include_zero)

# Função densidade base do modelo (exponencial por partes)
def f1(t, alpha, s, include_zero = False):
    return pwexp.pdf(t, alpha, s, include_zero = include_zero)

# Função hazard base do modelo (exponencial por partes)
def h1(t, alpha, s, include_zero = False):
    return pwexp.h(t, alpha, s, include_zero = include_zero)

def update_m_mps(model, alpha, s, x, t, delta):
    '''
        Atualiza a estimativa do vetor de variáveis latentes segundo o novo modelo e novo vetor de parâmetros alpha
    '''
    # FIQUE ATENTO!! SE HOUVEREM PROBLEMAS DE FORMATAÇÃO NP / TF PROVAVELMENTE SERÁ AQUI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    log_a0 = model.log_a(tf.constant(0.0, dtype = tf.float64))
    eta_pred = model.predict(x, verbose = 0)
    
    log_p0_pred = np.log( model.link_func(eta_pred) )

    log_r_pred = log_a0 - log_p0_pred
    r_pred = tf.math.exp(log_r_pred)
    
    new_theta = model.C_inv(r_pred).numpy().flatten()
    
    Sis = S1(t, alpha, s, include_zero = False)
    # Necessário o reshape, pois na função mps.mpf é considerado o vetor theta como um vetor coluna
    Sis = np.reshape(Sis, (len(Sis), 1))
    
    new_log_phi = lambda theta : model.log_phi(theta) + np.log(Sis)
    
    # Para o cálculo da função para diferentes thetas, a função mps.pmf leva em conta shape broadcasting, comum ao numpy e ao tensorflow
    f_sup = mps.pmf(model.sup, model.log_a, new_log_phi, new_theta, model.sup)
    
    E_M = np.sum(f_sup * model.sup, axis = 1)
    E_M2 = np.sum(f_sup * model.sup**2, axis = 1)
    new_m = E_M.copy()
    new_m[delta == 1] = E_M2[delta == 1] / E_M[delta == 1]
    
    return new_m

def initialize_alpha_s(t, delta, n_cuts = 6):
    alpha0 = np.ones(n_cuts + 1)
    qs = np.linspace(0, 1, n_cuts+1)[1:]
    s = np.quantile(t[delta == 1], qs)
    s = np.concatenate([[0],s])
    return alpha0, s

def save_EM_args(filename,
                 log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max,
                 max_iterations = 30, early_stopping_em = True, early_stopping_em_warmup = 5, early_stopping_em_eps = 100,
                 epochs = 100, batch_size = None, shuffle = False,
                 learning_rate = 0.01, run_eagerly = False, gradient_accumulation_steps = None,
                 early_stopping_nn = True, early_stopping_min_delta_nn = 0.0, early_stopping_patience_nn = 5,
                 reduce_lr = True, reduce_lr_steps = 10, reduce_lr_factor = 0.1,
                 validation = False, val_prop = None,
                 verbose = 2, seed = 1, alpha_known = False):
    func_args = {
        "log_a_str": log_a_str,
        "log_phi_str": log_phi_str,
        "C_str": C_str,
        "C_inv_str": C_inv_str,
        "sup_str": sup_str,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "max_iterations": max_iterations,
        "early_stopping_em": early_stopping_em,
        "early_stopping_em_warmup": early_stopping_em_warmup,
        "early_stopping_em_eps": early_stopping_em_eps,
        "epochs": epochs,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "learning_rate": learning_rate,
        "run_eagerly": run_eagerly,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "early_stopping_nn": early_stopping_nn,
        "early_stopping_min_delta_nn": early_stopping_min_delta_nn,
        "early_stopping_patience_nn": early_stopping_patience_nn,
        "reduce_lr": reduce_lr,
        "reduce_lr_steps": reduce_lr_steps,
        "reduce_lr_factor": reduce_lr_factor,
        "validation": validation,
        "val_prop": val_prop,
        "verbose": verbose,
        "seed": seed,
        "alpha_known": alpha_known
    }
    with open(filename, "w") as args_file:
        json.dump(func_args, args_file)
        
def load_args(filename):
    with open(filename, "r") as args_file:
        func_args = json.load(args_file)
    return func_args

def save_alpha_s(alpha, s, filename):
    pd.DataFrame({"alpha": alpha, "s": s}).to_csv(filename, index = False)

def load_alpha_s(filename):
    alpha_s = pd.read_csv(filename)
    return alpha_s["alpha"], alpha_s["s"]

def save_m(m, filename):
    pd.DataFrame({"new_m": m}).to_csv(filename, index = False)

def save_data(x_train, t_train, delta_train, m_train, x_test, t_test, filename):
    np.savez(filename,
             x_train = x_train.numpy(), t_train = t_train, delta_train = delta_train, m_train = m_train,
             x_val = x_train.numpy(), t_val = t_test, delta_val = delta_test, m_valpup = m_test)

# Limpa os conteúdos de pastas recursivamente
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            # Se o objeto é um arquivo 
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # Se é um arquivo, o deleta
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                file_name = file_path.split("/")[-1]
                # Se é uma pasta oculta, a deleta
                if(file_name[0] == "."):
                    shutil.rmtree(file_path)
                else:
                    # Se é uma pasta normal, deleta todos os arquivos nela
                    clear_folder(file_path)
        except Exception as e:
            print("Failed to delete {}. Reason: {}".format(file_path, e))
                  
    
def call_EM(em_filename,
            log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max,
            dummy_model, alpha, s,
            x, t, delta, m,
            max_iterations = 30,
            early_stopping_em = True, early_stopping_em_warmup = 5, early_stopping_em_eps = 1.0e-6,
            epochs = 100, batch_size = None, shuffle = True,
            learning_rate = 0.001, run_eagerly = False, gradient_accumulation_steps = None,
            early_stopping_nn = True, early_stopping_min_delta_nn = 0.0, early_stopping_patience_nn = 15,
            reduce_lr = False, reduce_lr_steps = 30, reduce_lr_factor = 0.1,
            validation = False, val_prop = 0.2,
            x_val = None, t_val = None, delta_val = None, m_val = None,
            verbose = 1, seed = 1, alpha_known = False):
                  
    data_dir = "EM_data"
    Path("{}/model_history".format(data_dir)).mkdir(parents=True, exist_ok=True)
    
    # Limpa todos os arquivos da pasta EM_data
    clear_folder(data_dir)

    # Salva os pesos do modelo
    dummy_model.save_model("{}/model.weights.h5".format(data_dir))
    # Salva o parâmetro alpha e seus nós s
    save_alpha_s(alpha, s, filename = "{}/alpha_s.csv".format(data_dir))
    # Salva os argumentos no arquivo EM_data/EM_args.json
    save_EM_args("EM_data/EM_args.json",
                 log_a_str, log_phi_str, C_str, C_inv_str, sup_str, theta_min, theta_max,
                 max_iterations = max_iterations, early_stopping_em = early_stopping_em,
                 early_stopping_em_warmup = early_stopping_em_warmup, early_stopping_em_eps = early_stopping_em_eps,
                 epochs = epochs, batch_size = batch_size, shuffle = shuffle,
                 learning_rate = learning_rate, run_eagerly = run_eagerly, gradient_accumulation_steps = gradient_accumulation_steps,
                 early_stopping_nn = early_stopping_nn, early_stopping_min_delta_nn = early_stopping_min_delta_nn, early_stopping_patience_nn = early_stopping_patience_nn,
                 reduce_lr = reduce_lr, reduce_lr_steps = reduce_lr_steps, reduce_lr_factor = reduce_lr_factor,
                 validation = validation, val_prop = val_prop,
                 verbose = verbose, seed = seed, alpha_known = alpha_known)

    # Arquivos .npz não reconhecem o tipo None, por isso a conversão para a string "None"
    if(x_val is None):
        x_val = "None"
        t_val = "None"
        delta_val = "None"
        m_val = "None"
    np.savez("EM_data/data.npz",
             x = x, t = t, delta = delta, m = m,
             x_val = x_val, t_val = t_val, delta_val = delta_val, m_val = m_val)

    if(verbose > 0):
        subprocess_result = subprocess.run(
            ["python3", em_filename]
        )
    else:
        # If verbose = 0, supresses all possible outputs from the file
        subprocess_result = subprocess.run(
           ["python3", em_filename],
           stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL
        )

    model_history = []
    model_history_count = len([name for name in os.listdir("{}/model_history".format(data_dir))])

    # Percorre todos os modelos e salva os seus valores em variáveis
    for i in range(model_history_count):
        aux = dummy_model.copy()
        aux.load_model("{}/model_history/new_model_{}.weights.h5".format(data_dir, i+1))
        model_history.append(aux)

    if(verbose > 0):
        print("Número de arquivos no diretório: {}".format(model_history_count))

    results_alpha = np.load("{}/EM_results.npz".format(data_dir))
    alpha_history = results_alpha["alpha_history"]
    
    results = {
        "new_model": model_history[-1],
        "model_history": model_history,
        "new_alpha": alpha_history[-1],
        "alpha_history": alpha_history,
        "m_history": results_alpha["m_history"],
        "m_val_history": results_alpha["m_val_history"],
        "converged": results_alpha["converged"],
        "steps": results_alpha["steps"],
        "loss_history": results_alpha["loss_history"],
        "loss_val_history": results_alpha["loss_val_history"],
    }

    return results


def Spop_known_S1(S1_values, log_a, log_phi, theta, sup):
    '''
        Population survival function when the array of non cured survival probabilities have already been computed.
    '''
    # Reshape the object to a column vector
    S1_values = np.reshape(S1_values, (len(S1_values), 1))

    # Perform opperation S1^sup (for the probability generation function)
    pgf_coef = S1_values**sup
    # Reshape the probability generation function coefficients to meet the shape of the probabilities
    pgf_coef = pgf_coef.T[np.newaxis, :, :] # Shape: (1, <sup size>, <number of times>)

    f_sup = mps.pmf(sup, log_a, log_phi, theta, sup)
    # Reshape the probabilities to meet the pgf_coef shape
    f_sup = f_sup[:, :, np.newaxis] # Shape: (<theta size>, <sup_size>, 1)
    
    # # Obtain the actual terms from the pgf summation
    #pgf_terms = pgf_coef * f_sup # Shape: (<theta_size>, <sup_size>, <number of times>)
    # # Sum through the support, getting for each theta, all the respective population survival times
    # pgf_result = np.sum(pgf_terms, axis = 1) # Shape: (<theta size>, <number of times>) --- theta_size = number of times!
    # # The above code gets into memory trouble and is not very efficient! Leaving here for future updates

    # To avoid getting into memory trouble with intermediate results, we use the fact that the theta and the S1 dimension is actually the same
    # (because for each individual, we have exactly one theta and one time value!)
    # Using that fact, we will be using Einstein summation convention standard for tensors with the numpy function np.einsum

    # I'm telling numpy the following:
    # We have the matrices A = pgf_coef (1, <sup size>, <number of times>) and B = f_sup (<theta size>, <sup_size>, 1)
    # dim A indices = (i, j, k) ; dim B indices = (k, j, ell)
    # For each fixed pair (i, ell, k) which index both t and theta (the same dimension) ("->il") we multiply both matrices and sum only through the indices j, which are
    # correspondent to the support indices
    pgf_result = np.einsum("ijk,kjl->ilk", pgf_coef, f_sup) # Shape: (1, 1, <number of times>)
    
    return pgf_result[0,0,:]

# Função de sobrevivência base do modelo (exponencial por partes)
def Spop(t, alpha, s, log_a, log_phi, theta, sup):
    '''
        Population survival function when the array of non cured survival probabilities have not been computed yet.
    '''
    # Precompute the actual non cured survival probabilities
    S1_values = S1(t, alpha, s)
    # Just call the Spop with S1 known
    return Spop_known_S1(S1_values, log_a, log_phi, theta, sup)