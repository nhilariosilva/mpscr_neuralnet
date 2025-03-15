
import warnings

import numpy as np

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp

from scipy.stats import rv_discrete

def pmf(x, log_a, log_phi, theta, sup, force_broadcasting = False):
    # Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list, type(tf.constant(0.0))] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list, type(tf.constant(0.0))] ):
        theta = np.array([theta])
    # Se x é uma lista, o converte para np.array
    if(type(x) == list):
        x = np.array(x)
    # Se theta é uma lista, o converte para np.array
    if(type(theta) == list):
        theta = np.array(theta)
    
    # Garante um formato de colunas para theta para realizar o broadcasting, caso necessário
    theta = np.reshape(theta, (len(theta), 1))
    
    # Evita problemas nas funções log_a e log_phi
    sup = sup.astype("float64") 
    
    # Obtém os valores do núcleo para o suporte da distribuição
    Psup = np.exp( log_a(sup) + sup * log_phi(theta) )
    # Obtém os valores do núcleo para o vetor x desejado
    Px = np.exp( log_a(x) + x * log_phi(theta) )
    # Normaliza o vetor de probabilidades com base na soma das probabilidades do suporte
    Px = Px / np.sum(Psup, axis = 1).reshape((len(theta),1))
    
    # Se a matriz é quadrada, subentende-se que se deseja vetorizar o cálculo, considerando um valor de theta para cada valor de x
    if(len(x) == len(theta) and not force_broadcasting):
        return np.diag(Px)
    # Caso theta seja um único número, evita o retorno de uma matriz desnecessariamente
    if(len(theta) == 1):
        Px = Px[0,:]
    
    return Px

def cdf(x, log_a, log_phi, theta, sup, lower_tail = True, force_broadcasting = False):
    # Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list, type(tf.constant(0.0))] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list, type(tf.constant(0.0))] ):
        theta = np.array([theta])
    # Se x é uma lista, o converte para np.array
    if(type(x) == list):
        x = np.array(x)
    # Se theta é uma lista, o converte para np.array
    if(type(theta) == list):
        theta = np.array(theta)# Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list, type(tf.constant(0.0))] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list, type(tf.constant(0.0))] ):
        theta = np.array([theta])
    
    # Evita problemas nas funções log_a e log_phi
    sup = sup.astype("float64")
    # Probabilidades de cada elemento do suporte
    fsup = pmf(sup, log_a, log_phi, theta, sup)
    
    # Se len(theta) = 1, aumenta a dimensão do objeto para uma matriz de modo a facilitar as operações gerais
    if(len(theta) == 1):
        fsup = np.array([fsup.tolist()])
    
    # Probabilidades acumuladas de cada elemento do suporte
    fsup_cum = np.cumsum( fsup, axis = 1 )
    
    # Índices de cada c referente aos valores de theta
    i = np.repeat( np.arange(len(theta)), len(x) )
    # Índices de cada x referentes aos elementos do suporte
    j = np.tile( np.searchsorted(sup, x), len(theta) )
    
    fsup_cum_cdf = np.reshape( fsup_cum[i,j], (len(theta), len(x)) )
    
    if(not lower_tail):
        fsup_cum_cdf = 1-fsup_cum_cdf
    
    if(len(x) == len(theta) and not force_broadcasting):
        return np.diag(fsup_cum_cdf)
    
    # Se len(theta) = 1, retorna a matriz calculada para um vetor
    if(len(theta) == 1):
        fsup_cum_cdf = fsup_cum_cdf[0,:]
    
    return fsup_cum_cdf

def rvs(log_a, log_phi, theta, sup, size = 1):
    # If theta is not a list (i.e. it is a single digit) just sample the sample using np.random.choice with the required size
    # if( (type(theta) != list and type(theta) != type(np.array([]))) ):
    if( type(theta) not in [list, type(np.array([])), type(tf.constant(0.0))] ):
        return np.random.choice(sup, size = size, replace = True, p = pmf(sup, log_a, log_phi, theta, sup))

    # If theta is a vector of parameters obtain the matrix with all the cdf values for each sup value and each theta
    # Each theta corresponds to a sample (the argument size is not used in this context)
    cdf_sup = cdf(sup, log_a, log_phi, theta, sup, force_broadcasting = True)
    u = np.random.uniform(size = len(theta))
    u_T = np.reshape(u, (u.shape[0], 1))
    return (u_T > cdf_sup).sum(axis = 1)

def ppf(q, log_a, log_phi, theta, sup):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    Fs = cdf(sup, log_a, log_phi, theta, sup)
    i = np.searchsorted(Fs, q)
    return sup[i]

