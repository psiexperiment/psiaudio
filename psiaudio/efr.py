import numpy as np
import pandas as pd

from . import util


def efr_bs_verhulst(arrays, fs, fm, n_harmonics=None, n_bootstrap=20, n_draw=None,
                    rng=None, window=None, max_harmonic=None):
    '''
    Calculate the EFR amplitude using a boostrapping algorithm

    Parameters
    ----------
    arrays : array or list of arrays
        Signal to compute PSD noise floor over. Must be trial x time. If a list
        of arrays is provided (e.g., one for each stimulus polarity), n_draw
        will be balanced across the arrays.
    fs : float
        Sampling rate of signal.
    fm : float
        Modulation frequency.
    n_harmonics : int
        Number of harmonics (including fundamental) to sum. If None, all
        harmonics up to `max_harmonic` are summed. An error is raised if both
        `n_harmonics` and `max_harmonic` is set to None.
    n_bootstrap : int Number of bootstrap cycles.  n_draw : {None, int}
        Number of trials to draw (with replacement) on each bootstrap cycle. If
        None, set to the total number of trials provided.
    rng : instance of RandomState
        If provided, this will be used to drive the bootstrapping algorithm.
        Useful when you need to "freeze" results (e.g., for publication).
    window : {None, string}
        Type of window to use when calculating bootstrapped PSD.
    max_harmonic : float
        Maximum harmonic frequency to compute up to. This can be set to the
        Nyquist frequency.

    Result
    ------
    efr_amplitude : pd.Series
        Pandas series indexed by bootstrap cycle. Contains EFR amplitude
        defined as the peak relative to the noise floor.
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
    if n_harmonics is None and max_harmonic is None:
        raise ValueError('Must provide n_harmonics or max_harmonic')
    elif n_harmonics is None:
        # Calculate number of harmonics to sum given max_harmonic and the
        # modulation frequency.
        n_harmonics = int(max_harmonic // fm)
        print(n_harmonics)

    if rng is None:
        rng = np.random.RandomState()
    if n_draw is None:
        n_draw =len(x)

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    if not isinstance(arrays, list):
        arrays = list(arrays)
    array_csd = [util.csd(x, window=window) for x in arrays]
    array_i = [np.arange(len(x)) for x in array_csd]

    n_draw_array = n_draw // len(arrays)
    if n_draw_array * len(arrays) != n_draw:
        raise ValueError('n_draw must be multiple of number of arrays')

    freq_step = fs / arrays[0].shape[-1]

    # Make sure that our frequencies of interest fall on integer multiples of
    # the frequency spacing of the FFT. We could potentially work around this,
    # but it's better to plan experiments with this limitation in mind.
    if ((fm // freq_step) * freq_step) != fm:
        raise ValueError('fm must be a multiple of the frequency spacing')

    # Calculate average complex spectral density for each boostrap.
    csd_bs = []
    for _ in range(n_bootstrap):
        # Draw equally from all arrays
        c_bs = []
        for i, c in zip(array_i, array_csd):
            c_bs.append(c[rng.choice(i, n_draw_array, replace=True)])
        c_bs = np.concatenate(c_bs, axis=0).mean(axis=0)

        csd_bs.append(c_bs.mean(axis=0))
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
        columns=pd.Index(util.psd_freq(arrays[0], fs), name='frequency'),
    )
    return efr.rename('efr_amplitude'), result, psd_bs
