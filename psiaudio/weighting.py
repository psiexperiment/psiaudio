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
    name : {'mouse', None, np.nan}
        Weights to load. If None or np.nan, then weights will be set to a null
        operation (0 dB) across all frequencies.

    Returns
    -------
    weights : array
        Weights for eacy frequency in `freq` in dB.
    '''
    # Short-circut if no weights desired (this is useful in cases where we want
    # to have simple code that can easily switch between weighted and
    # unweighted without having to add an if-statement in the calling code. Use
    # str(nan) to check value since np.isnan does not work on strings.
    if str(name) == 'nan' or name is None:
        return np.zeros_like(freq)
    weights = pd.Series(WEIGHTS[name])
    w_freq = weights.index.values
    w_level = weights.values
    w_level -= w_level.min()
    return np.interp(freq, w_freq, w_level)
