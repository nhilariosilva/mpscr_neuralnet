import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from time import time

from scipy.special import comb, loggamma, lambertw

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp

import mps
import pwexp

# Valor máximo do suporte da distribuição
B = 10001
B_str = "10001"

# ---------------------------- Poisson (Pocr) ----------------------------
# Versões sem o tensorflow
log_a_poisson = lambda m : -loggamma(m+1)
log_phi_poisson = lambda theta : np.log(theta)
C_poisson = lambda theta : np.exp(theta)
C_inv_poisson = lambda u : np.log(u)
# Versões para o tensorflow
log_a_poisson_tf = lambda m : -tf.math.lgamma(m+1)
log_phi_poisson_tf = lambda theta : tf.math.log(theta)
C_poisson_tf = lambda theta : tf.math.exp(theta)
C_inv_poisson_tf = lambda u : tf.math.log(u)
sup_poisson = np.arange(0, B, 1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_poisson_str = "log_a_poisson_tf"
log_phi_poisson_str = "log_phi_poisson_tf"
C_poisson_str = "C_poisson_tf"
C_inv_poisson_str = "C_inv_poisson_tf"
sup_poisson_str = "sup_poisson"

theta_min_poisson = None
theta_max_poisson = None
def E_poisson(theta):
    return theta
# Variance is always equal to the mean
def Var_poisson(theta):
    return theta

# ---------------------------- Logarithmic (Locr) ----------------------------
# Versões sem o tensorflow
log_a_log = lambda m : -np.log(m+1)
log_phi_log = lambda theta : np.log(theta)
C_log = lambda theta : -np.log(1-theta)/theta
C_inv_log = lambda u : 1 + np.real(lambertw(-u*np.exp(-u))) / u
# Versões para o tensorflow
log_a_log_tf = lambda m : -tf.math.log(m+1)
log_phi_log_tf = lambda theta : tf.math.log(theta)
C_log_tf = lambda theta : -tf.math.log(1-theta)/theta
C_inv_log_tf = lambda u : 1 + tfp.math.lambertw(-u*tf.math.exp(-u)) / u
sup_log = np.arange(0, B, 1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_log_str = "log_a_log_tf"
log_phi_log_str = "log_phi_log_tf"
C_log_str = "C_log_tf"
C_inv_log_str = "C_inv_log_tf"
sup_log_str = "sup_log"

theta_min_log = 0
theta_max_log = 1
def E_log(theta):
    return -theta / (np.log(1-theta)*(1-theta)) - 1
# Overdispersion: in this case, variance is always greater than mean
def Var_log(theta):
    return -theta*(theta + np.log(1-theta)) / ((1-theta)**2*(np.log(1-theta))**2)

# ---------------------------- Binomial Negativa (+Geométrica) (NBcr + Gecr) ----------------------------
# Versões sem o tensorflow
def log_a_nb(q):
    return lambda m : loggamma(m+q) - loggamma(m+1) - loggamma(q)
def log_phi_nb(q):
    return lambda theta : np.log(1-theta)
def C_nb(q):
    return lambda theta : theta**(-q)
def C_inv_nb(q):
    return lambda u : u**(-1/q)
# Versões para o tensorflow
def log_a_nb_tf(q):
    return lambda m : tf.math.lgamma(m+q) - tf.math.lgamma(m+1) - tf.math.lgamma(q)
def log_phi_nb_tf(q):
    return lambda theta : tf.math.log(1-theta)
def C_nb_tf(q):
    return lambda theta : theta**(-q)
def C_inv_nb_tf(q):
    return lambda u : u**(-1/q)
sup_nb = np.arange(0, B, 1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_nb_str = "log_a_nb_tf({})"
log_phi_nb_str = "log_phi_nb_tf({})"
C_nb_str = "C_nb_tf({})"
C_inv_nb_str = "C_inv_nb_tf({})"
sup_nb_str = "sup_nb"

theta_min_nb = 0
theta_max_nb = 1
def E_nb(q, theta):
    return q*(1-theta)/theta
# Overdispersion: in this case, variance is always greater than mean
def Var_nb(q, theta):
    return q*(1-theta)/theta**2


# ---------------------------- Binomial Negativa (Mean-Variance parametrization) (MVNBcr) ----------------------------
# Versões sem o tensorflow
def log_a_mvnb(q):
    return lambda m : loggamma(1/q+m) - loggamma(1/q) - loggamma(m+1)
def log_phi_mvnb(q):
    return lambda theta : np.log(q*theta) - np.log(1+q*theta)
def C_mvnb(q):
    return lambda theta : (1 + q*theta)**(1/q)
def C_inv_mvnb(q):
    return lambda u : (u**q - 1)/q
# Versões para o tensorflow
def log_a_mvnb_tf(q):
    return lambda m : tf.math.lgamma(1/q+m) - tf.math.lgamma(1/q) - tf.math.lgamma(m+1)
def log_phi_mvnb_tf(q):
    return lambda theta : tf.math.log(q*theta) - tf.math.log(1+q*theta)
def C_mvnb_tf(q):
    return lambda theta : (1 + q*theta)**(1/q)
def C_inv_mvnb_tf(q):
    return lambda u : (u**q - 1) / q
sup_mvnb = np.arange(0, B, 1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_mvnb_str = "log_a_mvnb_tf({})"
log_phi_mvnb_str = "log_phi_mvnb_tf({})"
C_mvnb_str = "C_mvnb_tf({})"
C_inv_mvnb_str = "C_inv_mvnb_tf({})"
sup_mvnb_str = "sup_mvnb"

theta_min_mvnb = None
theta_max_mvnb = None
def E_mvnb(q, theta):
    return theta
# Overdispersion: in this case, variance is always greater than mean
def Var_mvnb(q, theta):
    return (1+q*theta)*theta

# ---------------------------- Binomial (+Bernoulli) (Bincr + Bercr) ----------------------------
# Versões sem o tensorflow
def log_a_bin(q):
    return lambda m : loggamma(q+1) - loggamma(m+1) - loggamma(q-m+1)
def log_phi_bin(q):
    return lambda theta : np.log(theta) - np.log(1-theta)
def C_bin(q):
    return lambda theta : (1-theta)**(-q)
def C_inv_bin(q):
    return lambda u : 1 - u**(-1/q)
# Versões para o tensorflow
def log_a_bin_tf(q):
    return lambda m : tf.math.lgamma(q+1) - tf.math.lgamma(m+1) - tf.math.lgamma(q-m+1)
def log_phi_bin_tf(q):
    return lambda theta : tf.math.log(theta) - tf.math.log(1-theta)
def C_bin_tf(q):
    return lambda theta : (1-theta)**(-q)
def C_inv_bin_tf(q):
    return lambda u : 1 - u**(-1/q)
def sup_bin(q):
    return np.arange(0, q+1, 1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_bin_str = "log_a_bin_tf({})"
log_phi_bin_str = "log_phi_bin_tf({})"
C_bin_str = "C_bin_tf({})"
C_inv_bin_str = "C_inv_bin_tf({})"
sup_bin_str = "sup_bin({})"

theta_min_bin = 0
theta_max_bin = 1
def E_bin(q, theta):
    return q*theta
# Underdispersion: in this case, variance is always lesser than mean
def Var_bin(q, theta):
    return q*theta*(1-theta)

# ---------------------------- Borel (Bocr) ----------------------------
# Versões sem o tensorflow
log_a_borel = lambda m : (m-1)*np.log(m+1) - loggamma(m+1)
log_phi_borel = lambda theta : np.log(theta) - theta
C_borel = lambda theta : np.exp(theta)
C_inv_borel = lambda u : np.log(u)
# Versões para o tensorflow
log_a_borel_tf = lambda m : (m-1)*tf.math.log(m+1) - tf.math.lgamma(m+1)
log_phi_borel_tf = lambda theta : tf.math.log(theta) - theta
C_borel_tf = lambda theta : tf.math.exp(theta)
C_inv_borel_tf = lambda u : tf.math.log(u)
sup_borel = np.arange(0, B, 1).astype(np.float64)

theta_min_borel = 0
theta_max_borel = 1
def E_borel(theta):
    return theta/(1-theta)
# Overdispersion: in this case, variance is always greater than mean
def Var_borel(theta):
    mu = theta/(1-theta)
    return mu*(1+mu)**2

# ---------------------------- Restricted Generalized Poisson (RGPcr) ----------------------------
# Versões sem o tensorflow
def log_a_rgp(q):
    return lambda m : (m-1)*np.log(1+q*m) - loggamma(m+1)
def log_phi_rgp(q):
    return lambda theta : np.log(theta) - q*theta
def C_rgp(q):
    return lambda theta : np.exp(theta)
def C_inv_rgp(q):
    return lambda u : np.log(u)
# Versões para o tensorflow
def log_a_rgp_tf(q):
    return lambda m : (m-1)*tf.math.log(1+q*m) - tf.math.lgamma(m+1)
def log_phi_rgp_tf(q):
    return lambda theta : tf.math.log(theta) - q*theta
def C_rgp_tf(q):
    return lambda theta : tf.math.exp(theta)
def C_inv_rgp_tf(q):
    return lambda u : tf.math.log(u)
def sup_rgp(q):
    if(q > 0):
        return np.arange(0, B, 1).astype(np.float64)
    else:
        if(q < -1):
            raise Exception("q value can't be less than -1")
        max_sup_candidates = np.arange(1, 101)
        max_sup = max_sup_candidates[(1 + max_sup_candidates*q) > 0][-1]
        return np.arange(max_sup+1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_rgp_str = "log_a_rgp_tf({})"
log_phi_rgp_str = "log_phi_rgp_tf({})"
C_rgp_str = "C_rgp_tf({})"
C_inv_rgp_str = "C_inv_rgp_tf({})"
sup_rgp_str = "sup_rgp({})"

theta_min_rgp = 0
# The RGP theta parameter must be lesser than q, otherwise its probabilities do not sum to one
def theta_max_rgp(q):
    return np.abs(1/q)
def E_rgp(q, theta):
    return theta/(1-q*theta)
# Overdispersion: in this case, variance is always greater than mean
def Var_rgp(q, theta):
    mu = theta/(1 - q*theta)
    return mu / (1-q*theta)**2


# ---------------------------- Haight (Catalan) (Cacr) - Geeta(q = 2) ----------------------------
# Versões sem o tensorflow
log_a_haight = lambda m : loggamma(2*m+2) - loggamma(m+2) - loggamma(m+1) - np.log(2*m+1)
log_phi_haight = lambda theta : np.log(theta) + np.log(1-theta)
C_haight = lambda theta : 1/(1-theta)
C_inv_haight = lambda u : 1 - 1/u
# Versões para o tensorflow
log_a_haight_tf = lambda m : tf.math.lgamma(2*m+2) - tf.math.lgamma(m+2) - tf.math.lgamma(m+1) - tf.math.log(2*m+1)
log_phi_haight_tf = lambda theta : tf.math.log(theta) + tf.math.log(1-theta)
C_haight_tf = lambda theta : 1/(1-theta)
C_inv_haight_tf = lambda u : 1 - 1/u
sup_haight = np.arange(0, B, 1).astype(np.float64)

theta_min_haight = 0
theta_max_haight = 0.5
def E_haight(theta):
    p = theta
    s = 1-p
    return s/(s-p) - 1
# Overdispersion: in this case, variance is always greater than mean
def Var_haight(theta):
    p = theta
    s = 1-p
    return p*s/(s-p)**2 + 2*p**2*s/(s-p)**3

# ---------------------------- Geeta (Gecr) ----------------------------
# Versões sem o tensorflow
def log_a_geeta(q):
    return lambda m : loggamma(q*m+q) - loggamma(m+2) - loggamma((q-1)*m+q-1) - np.log(q*m+q-1)
def log_phi_geeta(q):
    return lambda theta : np.log(theta) + (q-1)*np.log(1-theta)
def C_geeta(q):
    return lambda theta : (1-theta)**(1-q)
def C_inv_geeta(q):
    return lambda u : 1 - u**(1/(1-q))
# Versões para o tensorflow
def log_a_geeta_tf(q):
    return lambda m : tf.math.lgamma(q*m+q) - tf.math.lgamma(m+2) - tf.math.lgamma((q-1)*m+q-1) - tf.math.log(q*m+q-1)
def log_phi_geeta_tf(q):
    return lambda theta : tf.math.log(theta) + (q-1)*tf.math.log(1-theta)
def C_geeta_tf(q):
    return lambda theta : (1-theta)**(1-q)
def C_inv_geeta_tf(q):
    return lambda u : 1 - u**(1/(1-q))
sup_geeta = np.arange(0, B, 1).astype(np.float64)
# Versões para o EM - Invoca o nome das funções acima para ser carregada no arquivo EM.py
log_a_geeta_str = "log_a_geeta_tf({})"
log_phi_geeta_str = "log_phi_geeta_tf({})"
C_geeta_str = "C_geeta_tf({})"
C_inv_geeta_str = "C_inv_geeta_tf({})"
sup_geeta_str = "sup_geeta"

theta_min_geeta = 0
theta_max_geeta = lambda q : 1/q
def E_geeta(q, theta):
    p = theta
    s = 1-p
    return s/(s-p*(q-1)) - 1
def Var_geeta(q, theta):
    p = theta
    s = 1-p
    return p*s*(q-1)/(s-p*(q-1))**2 + p**2*s*(q-1)*q/(s-p*(q-1))**3


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
    elif(distribution == "binomial" or distribution == "bin"): 
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

    