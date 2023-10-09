import numpy as np
import pandas as pd


WEIGHTS = {
    'mouse': {
        2000: 83,
        2828: 65,
        4000: 45,
        5656: 24,
        8000: 20,
        11313: 16,
        16000: 16,
        22627: 16,
        32000: 20,
        45254: 24,
        64000: 34,
    }
}


def load(freq, name):
    '''
    Returns weighting values for requested frequencies in dB

    Parameters
    ----------
    freq : array
        Frequencies to include weights for
    name : {'mouse'}
        Weights to load
    '''
    weights = pd.Series(WEIGHTS[name])
    w_freq = weights.index.values
    w_level = weights.values
    w_level -= w_level.min()
    return np.interp(freq, w_freq, w_level)
