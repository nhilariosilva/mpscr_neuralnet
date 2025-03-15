
import warnings

import numpy as np

def pmf(x, a, phi, theta, sup, force_broadcasting = False):
    # Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list] ):
        theta = np.array([theta])
    
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    
    # Garante um formato de colunas para theta para realizar o broadcasting, caso necessário
    theta = np.reshape(theta, (len(theta), 1))
    z_sup = np.log(a(sup)) + sup*np.log(phi(theta))
    c = np.max(z_sup, axis = 1) 
    c = np.reshape(c, (len(c), 1))
    
    log_sum_ezc = np.log( np.sum(np.exp(z_sup - c), axis = 1) )
    log_sum_ezc = np.reshape(log_sum_ezc, (len(log_sum_ezc),1) )
    
    # Log-sum-exp term
    lse = c + log_sum_ezc
    z = np.log(a(x)) + x*np.log(phi(theta))
    
    # Normalized probabilities
    Px = np.exp( z - lse )
    # Se a matriz é quadrada, subentende-se que se deseja vetorizar o cálculo, considerando um valor de theta para cada valor de x
    if(len(x) == len(theta) and not force_broadcasting):
        return np.diag(Px)
    
    if(len(theta) == 1):
        Px = Px[0,:]
    
    return Px

def cdf(x, a, phi, theta, sup, lower_tail = True, force_broadcasting = False):
    # Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list] ):
        theta = np.array([theta])
    
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    # Probabilidades de cada elemento do suporte
    fsup = pmf(sup, a, phi, theta, sup)
    
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

def rvs_single_theta(a, phi, theta, sup):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    probs = pmf(sup, a, phi, theta, sup)
    return np.random.choice(sup, size = 1, replace = True, p = probs)
    
def rvs(a, phi, theta, sup, size = 1):
    if( (type(theta) == list or type(theta) == type(np.array([]))) ):
        return np.array([rvs_single_theta(a, phi, the, sup) for the in theta]).flatten()
    
    probs = pmf(sup, a, phi, theta, sup)
    return np.random.choice(sup, size = size, replace = True, p = probs)

def ppf(q, a, phi, theta, sup):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    Fs = cdf(sup, a, phi, theta, sup)
    i = np.searchsorted(Fs, q)
    return sup[i]

# Dadas as funções a_m, phi da distribuição MPS e o suporte, obtém a função C(theta) = A(phi(theta))
def C_theta(a, phi, sup, vectorize = True):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    C_theta = lambda theta : np.sum( a(sup) * phi(theta)**sup )
    if(vectorize):
        return np.vectorize(C_theta)
    return C_theta

# Dadas as funções a_m, phi da distribuição MPS e o suporte, obtém a função C'(theta) = [ A(phi(theta)) ]'
def C_theta_prime(a, phi, phi_prime, sup, vectorize = True):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    sup = sup[1:]
    C_prime_theta = lambda theta : np.sum( sup * a(sup) * (phi(theta))**(sup-1) )
    if(vectorize):
        return np.vectorize(C_prime_theta)
    return C_prime_theta

# Inverte a função C numericamente a partir do método de Newton-Raphson
def C_inv(u, C, C_prime, theta0 = None, niter = 15, eps = 1.0e-6, verbose = 1):
    # Se o chute inicial é None, define como o ponto [1,1,...,1]
    if(theta0 is None):
        theta0 = np.ones_like(u)
    for i in range(niter):
        theta = theta0 - ( C(theta0) - u ) / C_prime(theta0)
        if( np.mean( (theta - theta0)**2 ) <= eps ):
            return theta
        theta0 = theta
    if(verbose == 1):
        warnings.warn("C_inv: Algoritmo não convergiu após {} iterações.".format(niter))
    return theta

