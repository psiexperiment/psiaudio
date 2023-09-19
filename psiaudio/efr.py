import numpy as np
import pandas as pd

from . import util


def efr_bs_verhulst(x, fs, fm, n_harmonics, n_bootstrap=20, n_draw=None,
                    rng=None, window=None):
    '''
    Calculate the EFR amplitude using a boostrapping algorithm

    Parameters
    ----------
    x : array
        Signal to compute EFR amplitude.
    fs : float
        Sampling rate of signal.
    fm : float
        Modulation frequency.
    n_harmonics : int
        Number of harmonics (including fundamental) to sum.
    n_bootstrap : int Number of bootstrap cycles.  n_draw : {None, int}
        Number of trials to draw (with replacement) on each bootstrap cycle. If
        None, set to the total number of trials provided.
    rng : instance of RandomState
        If provided, this will be used to drive the bootstrapping algorithm.
        Useful when you need to "freeze" results (e.g., for publication).
    window : {None, string}
        Type of window to use when calculating bootstrapped PSD.

    Result
    ------
    efr_amplitude : pd.Series
        Pandas series indexed by bootstrap cycle. Contains EFR amplitude
        defined as the peak to baseline amplitude.
    harmonics : pd.DataFrame
        Pandas DataFrame containing amplitude, noise floor, phase and
        normalized amplitude. Indexed by harmonic and bootstrap cycle. Norm
        amplitude is amplitude minus noise floor.
    psd : pd.DataFrame
        Pandas DataFrame containing the PSD of each bootstrap cycle. Columns
        are frequency and rows are bootstrap cycle.

    Notes
    -----
    This implements the EFR analysis algorithm first described in Vasilkov et
    al. 2021.
    '''
    if rng is None:
        rng = np.random.RandomState()
    if n_draw is None:
        n_draw =len(x)

    x = np.asarray(x)
    i = np.arange(len(x))

    freq_step = fs / x.shape[-1]

    # Make sure that our frequencies of interest fall on integer multiples of
    # the frequency spacing of the FFT. We could potentially work around this,
    # but it's better to plan experiments with this limitation in mind.
    if ((fm // freq_step) * freq_step) != fm:
        raise ValueError('fm must be a multiple of the frequency spacing')

    # Calculate average complex spectral density for each boostrap.
    csd_bs = []
    for _ in range(n_bootstrap):
        x_bs_csd = util.csd(x[rng.choice(i, n_draw)].mean(axis=0))
        csd_bs.append(x_bs_csd)
    csd_bs = np.vstack(csd_bs)

    # Indices of the noise bins relative to the bin containing the harmonic.
    # Delete the center value (0) so we don't include the harmonic in the
    # calculation of the noise floor.
    i_noise = np.delete(np.arange(-5, 6), 5)
    result = {}
    for harmonic in range(1, n_harmonics+1):
        i_h = int((fm * harmonic) // freq_step)
        i_nf = i_h + i_noise
        result[fm * harmonic] = pd.DataFrame({
            'amplitude': np.abs(csd_bs[:, i_h]),
            'noise_floor': np.abs(csd_bs[:, i_nf]).mean(axis=1),
            'phase': np.angle(csd_bs[:, i_h]),
        })

    result = pd.concat(result, names=['harmonic', 'bootstrap'])
    result['norm_amplitude'] = result.eval('amplitude - noise_floor').clip(lower=0)

    # Although the implementation in the paper divides the final calculation
    # (where the peak-to-peak amplitude is defined as the sum of the harmonics)
    # by n_fft (the length of the magnitude spectrum), we do not need to do this
    # here since the CSD caclulation is already normalized by multiplying by
    # 2 / n_time / sqrt(2). Note that n_fft = n_time / 2.
    ptp = result['norm_amplitude'] * np.exp(-1j * result['phase'])
    efr = np.abs(ptp.unstack('harmonic').sum(axis=1)) * np.sqrt(2)

    psd_bs = pd.DataFrame(
        np.abs(csd_bs),
        index=pd.Index(range(n_bootstrap), name='bootstrap'),
        columns=pd.Index(util.psd_freq(x, fs), name='frequency'),
    )
    return efr.rename('efr_amplitude'), result, psd_bs
