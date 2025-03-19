
import numpy as np

def get_lag_values(g, alpha, s):
    i = np.arange(0, g-1)
    return np.sum(
        alpha[i] * (s[i+1] - s[i])
    )

# Função risco (hazard)
def h(t, alpha, s, include_zero = True):
    if(include_zero):
        s = np.concatenate([[0], s])
    g = np.searchsorted(s, t)
    g[g == 0] = 1 # Caso exista algum valor t = 0, define g = 1 (veja que searchsorted resulta em 0)
    return alpha[g-1]

# Função risco acumulado (cumulative hazard)
def ch(t, alpha, s, include_zero = True):
    if(include_zero):
        s = np.concatenate([[0], s])
    g = np.searchsorted(s, t)
    g[g == 0] = 1 # Caso exista algum valor t = 0, define g = 1 (veja que searchsorted resulta em 0)
    lag_values = np.array(list(map(lambda g : get_lag_values(g, alpha, s), g)))
    return alpha[g-1] * (t - s[g-1]) + lag_values

# Função de distribuição
def cdf(t, alpha, s, include_zero = True, lower_tail = True):
    S = np.exp( -ch(t, alpha, s, include_zero) )
    if(lower_tail):
        return( 1 - S )
    return S

# Função densidade de probabilidade
def pdf(t, alpha, s, include_zero = True):
    return h(t, alpha, s, include_zero) * cdf(t, alpha, s, include_zero, lower_tail = False)

# Função quantil
def ppf(q, alpha, s, include_zero = True):
    if(include_zero):
        s = np.concatenate([[0], s])
        
    # Pega os pontos de corte para a função inversa de F
    s_inv = cdf(s, alpha, s, include_zero = False)
    
    g = np.searchsorted(s_inv, q)
    g[g == 0] = 1
    lag_values = np.array(list(map(lambda g : get_lag_values(g, alpha, s), g)))
    return -(np.log(1-q) + lag_values) / alpha[g-1] + s[g-1]

# Função de amostragem
def rvs(alpha, s, size = 1, include_zero = True):
    u = np.random.uniform(low = 0, high = 1, size = size)
    return ppf(u, alpha, s, include_zero)
