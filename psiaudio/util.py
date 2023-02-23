import logging
log = logging.getLogger(__name__)

from fractions import math
from math import gcd

import numpy as np
import pandas as pd
from scipy import signal


def as_numeric(x):
    if not isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
        x = np.asanyarray(x)
    return x


def db(target, reference=1):
    target = as_numeric(target)
    reference = as_numeric(reference)
    with np.errstate(divide='ignore'):
        return 20*np.log10(target/reference)


def dbi(db, reference=1):
    db = as_numeric(db)
    return (10**(db/20))*reference


def dbtopa(db):
    '''
    Convert dB SPL to Pascal

    .. math:: 10^{dB/20.0}/(20\\cdot10^{-6})

    >>> round(dbtopa(94), 4)
    1.0024
    >>> dbtopa(100)
    2.0
    >>> dbtopa(120)
    20.0
    >>> patodb(dbtopa(94.0))
    94.0

    Will also take sequences:

    >>> print(dbtopa([80, 100, 120]))
    [ 0.2  2.  20. ]

    '''
    return dbi(db, 20e-6)


def patodb(pa):
    '''
    Convert Pascal to dB SPL

    .. math:: 20*log10(pa/20e-6)

    >>> round(patodb(1))
    94
    >>> patodb(2)
    100.0
    >>> patodb(0.2)
    80.0

    Will also take sequences:
    >>> print(patodb([0.2, 2.0, 20.0]))
    [ 80. 100. 120.]
    '''
    return db(pa, 20e-6)


def normalize_rms(waveform, out=None):
    '''
    Normalize RMS power to 1 (typically used when generating a noise waveform
    that will be scaled by a calibration factor)

    waveform : array_like
        Input array.
    out : array_like
        An array to store the output.  Must be the same shape as `waveform`.
    '''
    return np.divide(waveform, rms(waveform), out)


def csd(s, window=None, detrend='linear'):
    if detrend is not None:
        s = signal.detrend(s, type=detrend, axis=-1)
    n = s.shape[-1]
    if window is not None:
        w = signal.get_window(window, n)
        s = w/w.mean()*s
    scale = 2 / n / np.sqrt(2)
    return np.fft.rfft(s, axis=-1) * scale


def csd_to_signal(csd):
    n = 2 * (len(csd) - 1)
    scale = 2 / n / np.sqrt(2)
    return np.fft.irfft(csd / scale, axis=-1)


def _phase(csd, unwrap=True):
    p = np.angle(csd)
    if unwrap:
        p = np.unwrap(p)
    if isinstance(csd, pd.DataFrame):
        p = pd.DataFrame(p, index=csd.index, columns=csd.columns)
    elif isinstance(csd, pd.Series):
        p = pd.Series(p, index=csd.index)
    return p


def phase(s, fs, window=None, waveform_averages=None, unwrap=True):
    c = csd(s, window, waveform_averages)
    return _phase(c, unwrap)


def psd(s, fs, window=None, waveform_averages=None, trim_samples=True):
    s = np.asarray(s)
    if waveform_averages is None:
        waveform_averages = 1
    if trim_samples:
        n = (s.shape[-1] // waveform_averages) * waveform_averages
        s = s[..., :n]
    new_shape = s.shape[:-1] + (waveform_averages, -1)
    s = s.reshape(new_shape)
    c = csd(s, window=window)
    return np.abs(c).mean(axis=-2)


def psd_freq(s, fs):
    return np.fft.rfftfreq(s.shape[-1], 1.0/fs)


def csd_df(s, fs, *args, **kw):
    c = csd(s, *args, **kw)
    freqs = pd.Index(psd_freq(s, fs), name='frequency')
    if c.ndim == 1:
        name = s.name if isinstance(s, pd.Series) else 'psd'
        return pd.Series(c, index=freqs, name=name)
    else:
        index = s.index if isinstance(s, pd.DataFrame) else None
        return pd.DataFrame(c, columns=freqs, index=index)


def psd_df(s, fs, *args, waveform_averages=None, **kw):
    '''
    Compute PSD and return as a dataframe with columns indexed by frequency

    Parameters
    ----------
    s : {array, DataFrame}
        Data to compute PSD over. The last axis (columns for DataFrame) is
        always assumed to be time. If array, only 2D arrays are supported (for
        conversion into DataFrame).
    fs : float
        Sampling rate of data.
    window : {None, string}
        Applies given window to signal before computing FFT. If None, no
        windowing is performed. Name of window must be a valid window name that
        can be passed to `scipy.signal.get_window`.
    waveform_averages : {None, int}
        Number of chunks to segment time data into before computing PSD.
        Averaging is done after computing the PSD.
    trim_samples : bool
        If true, remove excess time samples so that waveform can be split into
        `waveform_averages` segments of equal size. If False and the number of
        timepoints is not an integer multiple of `waveform_averages`, an error
        will be raised.

    Additional arguments are passed to :func:`~psiaudio.util.psd`.

    Returns
    -------
    psd : pd.DataFrame
        DataFrame with frequency as columns. If ``s`` was a DataFrame, the
        original index of ``s`` will be preserved. If ``s`` was an array, psd
        will have a simple integer index.
    '''
    p = psd(s, fs, *args, waveform_averages=waveform_averages, **kw)
    n = s.shape[-1]
    if waveform_averages is not None:
        n = n // waveform_averages
    freqs = pd.Index(np.fft.rfftfreq(n, 1/fs), name='frequency')
    if p.ndim == 1:
        name = s.name if isinstance(s, pd.Series) else 'psd'
        return pd.Series(p, index=freqs, name=name)
    else:
        index = s.index if isinstance(s, pd.DataFrame) else None
        return pd.DataFrame(p, columns=freqs, index=index)


def tone_conv(s, fs, frequency, window=None):
    frequency_shape = tuple([Ellipsis] + [np.newaxis]*s.ndim)
    frequency = np.asarray(frequency)[frequency_shape]
    s = signal.detrend(s, type='linear', axis=-1)
    n = s.shape[-1]
    if window is not None:
        w = signal.get_window(window, n)
        s = w/w.mean()*s
    t = np.arange(n)/fs
    r = 2.0*s*np.exp(-1.0j*(2.0*np.pi*t*frequency))
    return np.mean(r, axis=-1)


def tone_power_conv(s, fs, frequency, window=None):
    r = tone_conv(s, fs, frequency, window)
    return np.abs(r)/np.sqrt(2.0)


def tone_phase_conv(s, fs, frequency, window=None):
    r = tone_conv(s, fs, frequency, window)
    return np.angle(r)


def tone_power_fft(s, fs, frequency, window=None):
    power = psd(s, fs, window)
    freqs = psd_freq(s, fs)
    flb, fub = freqs*0.9, freqs*1.1
    mask = (freqs >= flb) & (freqs < fub)
    return power[..., mask].max(axis=-1)


def tone_phase_fft(s, fs, frequency, window=None):
    p = phase(s, fs, window, unwrap=False)
    freqs = psd_freq(s, fs)
    flb, fub = freqs*0.9, freqs*1.1
    mask = (freqs >= flb) & (freqs < fub)
    return p[..., mask].max(axis=-1)


def tone_power_conv_nf(s, fs, frequency, window=None):
    samples = s.shape[-1]
    resolution = fs/samples
    frequencies = frequency+np.arange(-2, 3)*resolution
    magnitude = tone_power_conv(s, fs, frequencies, window)
    nf_rms = magnitude[(0, 1, 3, 4), ...].mean(axis=0)
    tone_rms = magnitude[2]
    return nf_rms, tone_rms


def analyze_mic_sens(ref_waveforms, exp_waveforms, vrms, ref_mic_gain,
                     exp_mic_gain, output_gain, ref_mic_sens, **kwargs):

    ref_data = analyze_tone(ref_waveforms, mic_gain=ref_mic_gain, **kwargs)
    exp_data = analyze_tone(exp_waveforms, mic_gain=exp_mic_gain, **kwargs)

    # Actual output SPL
    output_spl = ref_data['mic_rms']-ref_mic_sens-db(20e-6)
    # Output SPL assuming 0 dB gain and 1 VRMS
    norm_output_spl = output_spl-output_gain-db(vrms)
    # Exp mic sensitivity in dB(V/Pa)
    exp_mic_sens = exp_data['mic_rms']+ref_mic_sens-ref_data['mic_rms']

    result = {
        'output_spl': output_spl,
        'norm_output_spl': norm_output_spl,
        'exp_mic_sens': exp_mic_sens,
        'output_gain': output_gain,
    }
    shared = ('time', 'frequency')
    result.update({k: ref_data[k] for k in shared})
    t = {'ref_'+k: ref_data[k] for k, v in ref_data.items() if k not in shared}
    result.update(t)
    t = {'exp_'+k: exp_data[k] for k, v in exp_data.items() if k not in shared}
    result.update(t)
    return result


def thd(s, fs, frequency, harmonics=3, window=None):
    ph = np.array([tone_power_conv(s, fs, frequency*(i+1), window)[np.newaxis] \
                   for i in range(harmonics)])
    ph = np.concatenate(ph, axis=0)
    return (np.sum(ph[1:]**2, axis=0)**0.5)/ph[0]


def analyze_tone(waveforms, frequency, fs, mic_gain, trim=0, thd_harmonics=3):
    trim_n = int(trim*fs)
    waveforms = waveforms[:, trim_n:-trim_n]

    # Get average tone power across channels
    power = tone_power_conv(waveforms, fs, frequency, window='flattop')
    power = db(power).mean(axis=0)

    average_waveform = waveforms.mean(axis=0)
    time = np.arange(len(average_waveform))/fs

    # Correct for gains (i.e. we want to know the *actual* Vrms at 0 dB input
    # and 0 dB output gain).
    power -= mic_gain

    #max_harmonic = np.min(int(np.floor((fs/2.0)/frequency)), thd_harmonics)
    harmonics = []
    for i in range(thd_harmonics):
        f_harmonic = frequency*(i+1)
        p = tone_power_conv(waveforms, fs, f_harmonic, window='flattop')
        p_harmonic = db(p).mean(axis=0)
        harmonics.append({
            'harmonic': i+1,
            'frequency': f_harmonic,
            'mic_rms': p_harmonic,
        })

    harmonic_v = []
    for h_info in harmonics:
        harmonic_v.append(dbi(h_info['mic_rms']))
    harmonic_v = np.asarray(harmonic_v)[:thd_harmonics]
    thd = (np.sum(harmonic_v[1:]**2)**0.5)/harmonic_v[0]

    return {
        'frequency': frequency,
        'time': time,
        'mic_rms': power,
        'thd': thd,
        'mic_waveform': average_waveform,
        'harmonics': harmonics,
    }


def spectrum_to_band_level(spectrum_db, n):
    '''
    Convert spectrum level to overall level

    Parameters
    ----------
    spectrum_db : float
        Spectrum level in dB re. reference (e.g., dB SPL).
    n : int
        Number of bands. If you are trying to compare this to an FFT with 1 Hz
        resolution, this is equal to the bandwidth (in Hz). If you are
        comparing this to an FFT with a bin size of 500 Hz, then this is equal
        to the bandwidth / 500.

    Returns
    -------
    band_db : float
        Band level in same units as spectrum level.
    '''
    return spectrum_db + 10 * np.log10(n)


def band_to_spectrum_level(band_db, n):
    '''
    Convert overall band level to spectrum level

    Parameters
    ----------
    band_db : float
        Band level in dB re. reference (e.g., dB SPL).
    n : int
        Number of bands. If you are trying to compare this to an FFT with 1 Hz
        resolution, this is equal to the bandwidth (in Hz). If you are
        comparing this to an FFT with a bin size of 500 Hz, then this is equal
        to the bandwidth / 500.

    Returns
    -------
    spectrum_db : float
        Spectrum level in same units as band level.
    '''
    return band_db - 10 * np.log10(n)


def rms(s, detrend=False, axis=-1):
    if detrend:
        s = signal.detrend(s, axis=axis)
    return np.mean(s**2, axis=axis)**0.5


def rms_rfft(x):
    return np.sqrt(np.sum(np.abs(x) ** 2))


def rms_rfft_db(x, *args, **kw):
    return db(rms_rfft(dbi(x, *args, **kw)))


def golay_pair(n=15):
    '''
    Generate pair of Golay sequences
    '''
    a0 = np.array([1, 1])
    b0 = np.array([1, -1])
    for i in range(n):
        a = np.concatenate([a0, b0])
        b = np.concatenate([a0, -b0])
        a0, b0 = a, b
    return a.astype(np.float32), b.astype(np.float32)


def transfer_function(stimulus, response, fs):
    response = response[:len(stimulus)]
    h_response = np.fft.rfft(response, axis=-1)
    h_stimulus = np.fft.rfft(stimulus, axis=-1)
    freq = psd_freq(response, fs)
    return freq, 2*np.abs(h_response*np.conj(h_stimulus))


def golay_tf(a, b, a_signal, b_signal, fs):
    '''
    Estimate system transfer function from Golay sequence

    Implements algorithm as described in Zhou et al. 1992.
    '''
    a_signal = a_signal[..., :len(a)]
    b_signal = b_signal[..., :len(b)]
    ah_psd = np.fft.rfft(a_signal, axis=-1)
    bh_psd = np.fft.rfft(b_signal, axis=-1)
    a_psd = np.fft.rfft(a)
    b_psd = np.fft.rfft(b)
    h_omega = (ah_psd*np.conj(a_psd) + bh_psd*np.conj(b_psd))/(2*len(a))
    freq = psd_freq(a, fs)
    h_psd = np.abs(h_omega)
    h_phase = np.unwrap(np.angle(h_omega))
    return freq, h_psd, h_phase


def golay_ir(n, a, b, a_signal, b_signal):
    '''
    Estimate system impulse response from Golay sequence

    Implements algorithm described in Zhou et al. 1992
    '''
    a_signal = a_signal.mean(axis=0)
    b_signal = b_signal.mean(axis=0)
    a_conv = np.apply_along_axis(np.convolve, 1, a_signal, a[::-1], 'full')
    b_conv = np.apply_along_axis(np.convolve, 1, b_signal, b[::-1], 'full')
    return 1.0/(2.0*n)*(a_conv+b_conv)[..., -len(a):]


def summarize_golay(fs, a, b, a_response, b_response, waveform_averages=None):

    if waveform_averages is not None:
        n_epochs, n_time = a_response.shape
        new_shape = (waveform_averages, -1, n_time)
        a_response = a_response.reshape(new_shape).mean(axis=0)
        b_response = b_response.reshape(new_shape).mean(axis=0)

    time = np.arange(a_response.shape[-1])/fs
    freq, tf_psd, tf_phase = golay_tf(a, b, a_response, b_response, fs)
    tf_psd = tf_psd.mean(axis=0)
    tf_phase = tf_phase.mean(axis=0)

    return {
        'psd': tf_psd,
        'phase': tf_phase,
        'frequency': freq,
    }


def freq_smooth(frequency, power, bandwidth=20):
    '''
    Uses Konno & Ohmachi (1998) algorithm
    '''
    smoothed = []
    old = np.seterr(all='ignore')
    for f in frequency:
        if f == 0:
            # Special case for divide by 0
            k = np.zeros_like(frequency)
        else:
            r = bandwidth*np.log10(frequency/f)
            k = (np.sin(r)/r)**4
            # Special case for np.log10(0/frequency)
            k[0] = 0
            # Special case where ratio is 1 (log of ratio is set to 0)
            k[frequency == f] = 1
            # Equalize weights
            k /= k.sum(axis=0)
        smoothed.append(np.sum(power*k))
    np.seterr(**old)
    return np.array(smoothed)


def ir_iir(impulse_response, fs, smooth=None, *args, **kwargs):
    csd = np.fft.rfft(impulse_response)
    psd = np.abs(csd)/len(impulse_response)
    phase = np.unwrap(np.angle(csd))
    frequency = np.fft.rfftfreq(len(impulse_response), fs**-1)

    # Smooth in the frequency domain
    if smooth is not None:
        psd = dbi(freq_smooth(frequency, db(psd), smooth))
        phase = freq_smooth(frequency, phase, smooth)

    return iir(psd, phase, frequency, *args, **kwargs)


def iir(psd, phase, frequency, cutoff=None, phase_correction=None,
        truncate=None, truncate_spectrum=False, reference='mean'):
    '''
    Given the impulse response, compute the inverse impulse response.

    Parameters
    ----------
    # TODO

    Note
    ----
    Specifying the cutoff range is highly recommended to get a well-behaved
    function.
    '''
    # Equalize only a subset of the calibrated frequencies
    if cutoff is not None:
        lb, ub = cutoff
        m = (frequency >= lb) & (frequency < ub)
        inverse_psd = psd[m].mean()/psd
        inverse_psd[~m] = 1
    else:
        inverse_psd = psd.mean()/psd

    if phase_correction == 'linear':
        m, b = np.polyfit(frequency, phase, 1)
        inverse_phase = 2*np.pi*np.arange(len(frequency))*m+b
    elif phase_correction == 'subtract':
        inverse_phase = 2*np.pi-phase
    else:
        inverse_phase = phase

    filtered_spectrum = inverse_psd*np.exp(inverse_phase*1j)

    if truncate_spectrum:
        orig_ub = np.round(frequency[-1])
        ub = np.round(ub)
        filtered_spectrum = filtered_spectrum[frequency <= ub]
        iir = truncated_ifft(filtered_spectrum, orig_ub, ub)
    else:
        iir = np.fft.irfft(filtered_spectrum)

    if truncate:
        truncate_samples = int(truncate*fs)
        iir = iir[:truncate_samples]

    return iir


def truncated_ifft(spectrum, original_fs, truncated_fs):
    iir = np.fft.irfft(spectrum)
    lcm = original_fs*truncated_fs/math.gcd(original_fs, truncated_fs)
    up = lcm/truncated_fs
    down = lcm/original_fs
    iir = signal.resample_poly(iir, up, down)
    iir *= truncated_fs/original_fs
    return iir


def process_tone(fs, signal, frequency, min_snr=None, max_thd=None,
                 thd_harmonics=3, silence=None):
    '''
    Compute the RMS at the specified frequency. Check for distortion.

    Parameters
    ----------
    fs : float
        Sampling frequency of signal
    signal : ndarray
        Last dimension must be time. If more than one dimension, first
        dimension must be repetition.
    frequency : float
        Frequency (Hz) to analyze
    min_snr : {None, float}
        If specified, must provide a noise floor measure (silence). The ratio,
        in dB, of signal RMS to silence RMS must be greater than min_snr. If
        not, a CalibrationNFError is raised.
    max_thd : {None, float}
        If specified, ensures that the total harmonic distortion, as a
        percentage, is less than max_thd. If not, a CalibrationTHDError is
        raised.
    thd_harmonics : int
        Number of harmonics to compute. If you pick too many, some harmonics
        may be above the Nyquist frequency and you'll get an exception.
    thd_harmonics : int
        Number of harmonics to compute. If you pick too many, some harmonics
        may be above the Nyquist frequency and you'll get an exception.
    silence : {None, ndarray}
        Noise floor measurement. Required for min_snr. Shape must match signal
        in all dimensions except the first and last.

    Returns
    -------
    result : pandas Series or DataFrame
        Series will be indexed with RMS, SNR, THD and frequency. DataFrame will
        contain columns for RMS, SNR, THD and frequency. The return type will
        depend on the dimensionality of the input array.
    '''
    from .calibration import CalibrationTHDError, CalibrationNFError

    harmonics = frequency * (np.arange(thd_harmonics) + 1)

    # This returns an array of [harmonic, repetition, channel]. Here, rms[0] is
    # the rms at the fundamental frequency. rms[1:] is the rms at all the
    # harmonics.
    signal = np.atleast_2d(signal)
    rms = tone_power_conv(signal, fs, harmonics, 'flattop')
    phase = tone_phase_conv(signal, fs, frequency, 'flattop')

    # Compute the mean RMS at F0 across all repetitions
    rms = rms.mean(axis=1)
    freq_rms = rms[0]

    freq_phase = phase.mean(axis=0)
    freq_phase_deg = np.rad2deg(freq_phase)

    # Compute the harmonic distortion as a percent
    thd = np.sqrt(np.sum(rms[1:]**2))/freq_rms * 100

    # If a silent period has been provided, use this to estimat the signal to
    # noise ratio. As an alternative, could we just use the "sidebands"?
    if silence is not None:
        silence = np.atleast_2d(silence)
        noise_rms = tone_power_conv(silence, fs, frequency, 'flattop')
        noise_rms = noise_rms.mean(axis=0)
        freq_snr = db(freq_rms, noise_rms)
        if min_snr is not None:
            if np.any(freq_snr < min_snr):
                raise CalibrationNFError(frequency, freq_snr)
    else:
        freq_snr = np.full_like(freq_rms, np.nan)

    if max_thd is not None and np.any(thd > max_thd):
        raise CalibrationTHDError(frequency, thd)

    # Concatenate and return as a record array
    result = np.concatenate((freq_rms[np.newaxis], freq_snr[np.newaxis],
                             thd[np.newaxis]))

    data = {'rms': freq_rms, 'snr': freq_snr, 'thd': thd,
            'phase': freq_phase, 'phase_degrees': freq_phase_deg}

    if result.ndim == 1:
        return pd.Series(data)
    else:
        return pd.DataFrame(data)

################################################################################
# Octave functions (typically used for generating octave frequencies)
################################################################################
def octave_space(lb, ub, step, mode='nearest'):
    '''
    >>> print(octave_space(4, 32, 1.0))
    [ 4.  8. 16. 32.]

    >>> freq = octave_space(0.5, 50.0, 0.25, 'nearest')
    >>> print(round(min(freq), 2))
    0.5
    >>> print(round(max(freq), 2))
    53.82

    >>> freq = octave_space(0.5, 50.0, 0.25, 'bounded')
    >>> print(round(min(freq), 2))
    0.5
    >>> print(round(max(freq), 2))
    45.25
    '''
    if mode == 'nearest':
        lbi = round(np.log2(lb) / step) * step
        ubi = round(np.log2(ub) / step) * step
    elif mode == 'bounded':
        lbi = np.ceil(np.log2(lb) / step) * step
        ubi = np.floor(np.log2(ub) / step) * step
    x = np.arange(lbi, ubi+step, step)
    return 2.0**x


ordering_error_message = '''Unable to interleave {frequencies} so that adjacent
frequencies are spaced at least {octaves:.1f} octaves apart. This can often be
fixed by increasing the range of frequencies tested. For example, if you are
assessing frequencies spaced 0.5 octaves, you need at least four frequencies to
be able order them so that adjacent frequencies are at least 1.0 octaves apart.
'''

def interleave_octaves(freqs, min_octaves=1):
    '''
    Return correct ordering for frequencies in interleaved paradigm as per.
    Buran et al.

    This function works with both kHz and Hz.

    >>> interleave_octaves([2, 2.8, 4, 5.6, 8])
    [8, 4, 2, 5.6, 2.8]
    >>> interleave_octaves([2000, 2800, 4000, 5600, 8000])
    [8000, 4000, 2000, 5600, 2800]

    If a set of frequencies cannot appropriately be ordered, a ValueError is
    raised. In this example, the first and last frequences are within one
    octave.

    >>> interleave_octaves([2000, 2800, 4000]) # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    ValueError: Unable to interleave 4000, 2000, 2800 so that adjacent
    frequencies are spaced at least 1.0 octaves apart. This can often be fixed
    by increasing the range of frequencies tested. For example, if you are
    assessing frequencies spaced 0.5 octaves, you need at least four
    frequencies to be able order them so that adjacent frequencies are at least
    1.0 octaves apart.

    You can use a different octave spacing.

    >>> interleave_octaves([2000, 2800, 4000], 0.5)
    [4000, 2800, 2000]
    '''
    freqs = list(freqs).copy()
    freqs.sort()
    space = np.median(np.diff(np.log2(freqs)))
    freqs = freqs[::-1]
    n_groups = int(np.round(min_octaves/space))

    ordered = []
    for i in range(n_groups):
        ordered.extend(freqs[i::n_groups])
    if not check_interleaved_octaves(ordered, min_octaves):
        ordered = ', '.join(str(f) for f in ordered)
        m = ordering_error_message.format(frequencies=ordered,
                                          octaves=min_octaves)
        raise ValueError(m)
    return ordered


def check_interleaved_octaves(freqs, min_octaves=1):
    '''
    Ensure that frequencies are spaced at least an octave apart

    Parameters
    ----------
    freqs : ordered sequence
        Sequence of frequencies in the desired ordering.
    min_octaves : float
        Minimum octave spacing to enforce (this is multiplied by 0.95 to allow
        for some fudge factor if frequencies were rounded).

    Notes
    -----
    * If you are rounding frequencies to the nearest Hz, the actual octaves
      spacing may be slightly less or more than the desired spacing due to the
      rounding. We multiply `min_octaves` by 0.99 to allow for this small error
      in octave spacing.
    * This also checks that the last frequency is an octave from the first
      frequency.
    '''
    check = freqs.copy()
    # Ensure octave spacing between end of trial and beginning of next trial
    check += [check[0]]
    octaves = np.abs(np.diff(np.log2(check)))
    return not np.any(octaves < (min_octaves * 0.95))


################################################################################
# Misc. signal manipulation functions
################################################################################
def resample_fft(waveform, fs, target_fs):
    n = len(waveform)
    target_n = int(round(n * (target_fs / fs)))
    return signal.resample(waveform, target_n, axis=-1, window='boxcar')


def resample_poly(waveform, fs, target_fs):
    g = gcd(fs, target_fs)
    down = fs // g
    up = target_fs // g
    return signal.resample_poly(waveform, up, down, axis=-1)


def psd_bootstrap_vec(x, fs, n_draw=400, n_bootstrap=100, rng=None, window=None):
    '''
    Calculate the normalized PSD across trials using a boostrapping algorithm

    To estmate the noise floor, the CSD of each trial is computed and then the
    phases of the CSD are randomized.

    Parameters
    ----------
    x : array
        Signal to compute PSD noise floor over. Must be trial x time.
    fs : float
        Sampling rate of signal
    n_draw : int
        Number of trials to draw on each bootstrap cycle.
    n_bootstrap : int
        Number of bootstrap cycles.
    rng : instance of RandomState
        If provided, this will be used to drive the bootstrapping algorithm.
        Useful when you need to "freeze" results (e.g., for publication).
    window : {None, string}
        Type of window to use when calculating bootstrapped PSD.

    Result
    ------
    psd_bs : DataFrame
        Pandas DataFrame indexed by frequency. Columns include `psd_nf`, the
        noise floor as estimated by the bootstrapping algorithm.

    Notes
    -----
    TODO: Add citation (Bharadwaj).
    '''
    if rng is None:
        rng = np.random.RandomState()

    c = csd(x, window=window)
    i = np.arange(len(c))

    c_bs = c[rng.choice(i, n_draw * n_bootstrap)]
    c_bs.shape = (n_draw, n_bootstrap, -1)
    random_phases = rng.uniform(0, 2*np.pi, size=c_bs.shape)
    c_bs_rand = np.abs(c_bs) * np.exp(-1j * random_phases)

    mean_psd_bs = db(np.abs(c_bs.mean(axis=0)))
    mean_psd_bs_rand = db(np.abs(c_bs_rand.mean(axis=0)))

    angle_bs = np.angle(c_bs)
    plv_bs = np.abs(np.mean(np.exp(-1j*angle_bs), axis=0))
    plv = plv_bs.mean(axis=0)

    psd_nf = mean_psd_bs_rand.mean(axis=0)
    psd = mean_psd_bs.mean(axis=0)
    psd_norm = psd - psd_nf

    return pd.DataFrame({
            'psd_nf': psd_nf,
            'psd': psd,
            'psd_norm': psd_norm,
            'plv': plv,
        },
        index=pd.Index(psd_freq(x, fs), name='frequency'),
    )


def psd_bootstrap_loop(x, fs, n_draw=400, n_bootstrap=100, rng=None, window=None):
    '''
    Calculate the normalized PSD across trials using a boostrapping algorithm

    To estmate the noise floor, the CSD of each trial is computed and then the
    phases of the CSD are randomized.

    Parameters
    ----------
    x : array
        Signal to compute PSD noise floor over. Must be trial x time.
    fs : float
        Sampling rate of signal
    n_draw : int
        Number of trials to draw on each bootstrap cycle.
    n_bootstrap : int
        Number of bootstrap cycles.
    rng : instance of RandomState
        If provided, this will be used to drive the bootstrapping algorithm.
        Useful when you need to "freeze" results (e.g., for publication).
    window : {None, string}
        Type of window to use when calculating bootstrapped PSD.

    Result
    ------
    psd_bs : DataFrame
        Pandas DataFrame indexed by frequency. Columns include `psd_nf`, the
        noise floor as estimated by the bootstrapping algorithm.

    Notes
    -----
    TODO: Add citation (Bharadwaj).
    '''
    if rng is None:
        rng = np.random.RandomState()

    c = csd(x, window=window)
    i = np.arange(len(c))

    mean_psd_bs = []
    mean_psd_bs_rand = []
    plv_bs = []

    for b in range(n_bootstrap):
        c_bs = c[rng.choice(i, n_draw)]
        random_phases = rng.uniform(0, 2 * np.pi, size=c_bs.shape)
        c_bs_rand = np.abs(c_bs) * np.exp(-1j * random_phases)
        mean_psd_bs.append(db(np.abs(c_bs.mean(axis=0))))
        mean_psd_bs_rand.append(db(np.abs(c_bs_rand.mean(axis=0))))
        angle_bs = np.angle(c_bs)
        plv_bs.append(np.abs(np.mean(np.exp(-1j*angle_bs), axis=0)))

    plv = np.mean(plv_bs, axis=0)
    psd_nf = np.mean(mean_psd_bs_rand, axis=0)
    psd = np.mean(mean_psd_bs, axis=0)
    psd_norm = psd - psd_nf

    return pd.DataFrame({
            'psd_nf': psd_nf,
            'psd': psd,
            'psd_norm': psd_norm,
            'plv': plv,
        },
        index=pd.Index(psd_freq(x, fs), name='frequency'),
    )


################################################################################
# Multichannel functions
################################################################################
def diff_matrix(n_chan, reference, labels=None):
    if reference == 'all':
        matrix = np.full((n_chan, n_chan), -1/n_chan)
        di = np.diag_indices(n_chan)
        matrix[di] = 1 - 1/n_chan
    elif reference == 'raw':
        matrix = np.eye(n_chan, n_chan)
    else:
        if np.isscalar(reference):
            reference = [reference]

        if labels is not None:
            i_reference = [labels.index(r) for r in reference]
        else:
            i_reference = reference

        matrix = np.eye(n_chan, n_chan)

        for i in i_reference:
            col = matrix[:, i].copy()
            scale = 1 / len(i_reference)
            col[:i] = -scale
            col[i] -= scale
            col[i+1:] = -scale
            matrix[:, i] = col

    return matrix


################################################################################
# Binary (boolean/TTL) functions
################################################################################
# Many of these functions are copied over as-is from the binary_funcs.py file
# in NeuroBehavior (BSD-licensed).


def ts(TTL):
    return np.flatnonzero(TTL)


def edge_rising(TTL):
    return np.r_[0, np.diff(TTL.astype('i'))] == 1


def edge_falling(TTL):
    return np.r_[0, np.diff(TTL.astype('i'))] == -1


def epochs(x, pad=0):
    '''
    Given a boolean array, where 1 = epoch, return indices of epochs (first
    column is the index where x goes from 0 to 1 and second column is index
    where x goes from 1 to 0.
    '''
    start = ts(edge_rising(x))
    end = ts(edge_falling(x))
    for s in start:
        x[s-pad:s] = 1
    for e in end:
        x[e:e+pad] = 1
    start = ts(edge_rising(x))
    end = ts(edge_falling(x))

    # Handle various boundary conditions where some sort of task-related
    # activity is registered at very beginning or end of experiment.

    if len(end) == 0 and len(start) == 0:
        return np.array([]).reshape((0, 2))
    elif len(end) == 0 and len(start) == 1:
        end = np.r_[end, len(x)]
    elif len(end) == 1 and len(start) == 0:
        start = np.r_[0, start]

    if end[0] < start[0]:
        start = np.r_[0, start]
    if end[-1] < start[-1]:
        end = np.r_[end, len(x)]

    return np.c_[start, end]


def smooth_epochs(epochs):
    '''
    Given a 2D array of epochs in the format [[start time, end time], ...],
    identify and remove all overlapping epochs such that:

        [ epoch   ]        [ epoch ]
            [ epoch ]

    Will become:

        [ epoch     ]      [ epoch ]

    Epochs do not need to be ordered when provided; however, they will be
    returned ordered.
    '''
    if len(epochs) == 0:
        return epochs
    epochs = np.asarray(epochs)
    epochs.sort(axis=0)
    i = 0
    n = len(epochs)
    smoothed = []
    while i < n:
        lb, ub = epochs[i]
        i += 1
        while (i < n) and (ub >= epochs[i,0]):
            ub = epochs[i,1]
            i += 1
        smoothed.append((lb, ub))
    return np.array(smoothed)


def epochs_contain(epochs, ts):
    '''
    Returns True if ts falls within one of the epoch boundaries
    Epochs must be sorted.
    '''
    i = np.searchsorted(epochs[:,0], ts)
    j = np.searchsorted(epochs[:,1], ts)
    return i != j


def epochs_overlap(a, b):
    '''
    Returns True where `b` falls within boundaries of epoch in `a`
    Epochs must be sorted.
    '''
    i = np.searchsorted(a[:,0], b[:,0])
    j = np.searchsorted(a[:,1], b[:,1])
    return i != j


def debounce_epochs(epochs, debounce):
    '''
    Given a 2D array of epochs in the format [[start time, end time], ...],
    throw out all epochs that are shorter than the minimum sample duration.
    After discarding these epochs, combine remaining epochs if they are within
    the minimum sample duration of each other.
    '''
    # First, throw out all epochs that do not meet the minimum sample duration.
    keep = (epochs[:, 1] - epochs[:, 0]) >= debounce
    epochs = epochs[keep]

    # Now, we are going to use a trick to check to see if there are any epochs
    # within `debounce` samples of each other. If there are, we will just
    # combine them. The trick is to pad the upper edge of each epoch by
    # `debounce`, then use the `smooth_epochs` function (which combines epochs
    # that are touching or overlap) and then subtract the `debounce` value from
    # the upper edge of the returned epochs.
    epochs[:, 1] += debounce
    epochs = smooth_epochs(epochs)
    epochs[:, 1] -= debounce
    return epochs


def int_to_TTL(a, width):
    '''
    Converts a 1D array of integers to a 2D boolean array based on the binary
    representation of each integer.
    Primarily used in conjunction with TDT's `FromBits` component to reduce the
    overhead of storing and transferring TTL data.  Since a TTL can be
    represented as a single bit (0 or 1), it is wasteful to cast the TTL to an
    int32 before storing the data in a buffer.  `FromBits` combines up to 6 TTL
    channels into a single int32 word.  Since only the first 6 bits are used to
    store the 6 TTL channels, the data can be reduced further:

    1. Using `ShufTo8`, 24 TTL channels can be stored in a single index of a
       serial buffer.
    2. Using `CompTo8`, 4 consecutive samples of data from 6 TTL channels can be
       stored in a single index of a serial buffer.
    3. Combining `ShufTo16` and `CompTo16`, store 2 consecutive samples of data
       from 12 TTL channels in a single index of a serial buffer. Using this
       approach, the memory overhead and amount of data being transferred has
       been reduced by a factor of 24.  This function uses Numpy's bitshift and
       bitmask operators, so the algorithm should be pretty efficient.

    Parameters
    ==========
    a : array_like
        Sequence of integers to expand into the corresponding boolean array.
        The dtype (either int8, int16 or int32) of the array is used to figure
        out the size of the second dimension.  This will depend on your
        combination of `FromBits` and the shuffle/compression components.

    Returns
    =======
    bitfield : array
        2D boolean array repesenting the bits in little-endian order

    Example
    =======
    >>> int_to_TTL([4, 8, 5], width=6).T
    array([[False, False,  True, False, False, False],
           [False, False, False,  True, False, False],
           [ True, False,  True, False, False, False]])
    '''
    a = np.array(a)
    bitarray = [(a>>bit) & 1 for bit in range(width)]
    return np.array(bitarray, dtype=np.bool)


def bin_array(number, bits):
    '''Return binary representation of an integer as an integer array
    >>> bin_array(8, 4)
    [0, 0, 0, 1]
    >>> bin_array(3, 4)
    [1, 1, 0, 0]

    NOTE: This function has not been profiled for speed.
    '''
    return [(number>>bit)&1 for bit in range(bits)]
