from collections import namedtuple

import numpy as np
from scipy import stats

'''
Some resources on Hotelling's T-Square test statistic
 * https://online.stat.psu.edu/stat505/lesson/7/7.1/7.1.15
'''

ht2_2samp_result = namedtuple('HT2_2_samp_result', ('T2', 'F', 'p', 'df'))


def ht2_individual(train, test=None):
    '''
    Computes the Hotelling T^2 statistic for each individual trial in the test
    set against the distribution in the train set.

    Parameters
    ----------
    train : 2D array or DataFrame
        Rows are individual trials. This is the reference dataset.
    test : {None, 2D array or DataFrame}
        Rows are individual trials. If None, then the train set is used as the
        test set.

    Returns
    -------
    t2 : 1D array
        T^2 statistic for each row in the test set.
    '''
    # Adapted from https://stackoverflow.com/questions/25412954/hotellings-t2-scores-in-python
    if test is None:
        test = train.copy()
    mean = train.mean(axis=0)
    cov = np.cov(train.T, ddof=1)
    test_centered = test - mean
    inverse_cov = np.linalg.pinv(cov)
    return np.einsum('ij,ij->i', test_centered @ inverse_cov, test_centered)


def ht2_2samp(x1, x2):
    '''
    Computes the two-sample Hotelling T^2 statistic representing the distance
    between x1 and x2.

    Implements Srivastava (2007), Multivariate Theory for Analyzing High
    Dimensional Data.

    Parameters
    ----------
    x1 : 2D array or DataFrame
        Rows are individual trials.
    x2 : {None, 2D array or DataFrame}
        Rows are individual trials.

    Returns
    -------
    result : namedtuple
        Named tuple (T2, F, p, df) where T2 is the T^2 statistic, F is the F
        value, p is the p-value as computed under the assumption that the
        F-statistic follows the chi-square distribution, and df is the degrees
        of freedom.
    '''
    # Number of samples in dimension (i.e., frequencies)
    p = x1.shape[-1]

    # N is the number of observations in group (i.e., probes)
    N1 = x1.shape[0]
    N2 = x2.shape[0]
    n = N1 + N2 - 2

    # Get sample covariance matrix
    #S = np.cov(x1.T, ddof=1)
    S = np.cov(x1.T)
    # Compute Moore-Penrose inverse of covariance matrix
    S_plus = np.linalg.pinv(S)

    x1_mean = x1.mean(axis=0)
    x2_mean = x2.mean(axis=0)
    xd = x1.mean(axis=0) - x2.mean(axis=0)

    # Implement Two-sample T+ test from section (3)
    t2 = (N1 * N2) / (N1 + N2) * xd.T @ S_plus @ xd

    # Implement Two-sample F+ test from section (3)
    Fp = (p - n + 1) / n**2 * (1/N1 + 1/N2)**-1 * xd.T @ S_plus @ xd

    # Convert F-statistic to a p value
    p_value = stats.chi2.sf(Fp, df=n)

    return ht2_2samp_result(t2, Fp, p_value, n)
