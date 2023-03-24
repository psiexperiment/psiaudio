import logging
log = logging.getLogger(__name__)

from functools import partial
import itertools
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from . import util
from . import queue


def fast_cache(f):
    cache = {}
    kwd_marker = object()
    def wrapper(*args, **kw):
        key = args + (kwd_marker,) + tuple(sorted(kw.items()))
        if key not in cache:
            cache[key] = f(*args, **kw)
        return cache[key]
    return wrapper


################################################################################
# Base classes
################################################################################
class Waveform:

    def reset(self):
        raise NotImplementedError

    def next(self, samples):
        raise NotImplementedError

    def n_samples_remaining(self):
        raise NotImplementedError

    def n_samples(self):
        raise NotImplementedError

    def get_samples_remaining(self):
        samples = self.n_samples_remaining()
        if samples == np.inf:
            raise ValueError('Waveform does not have a finite duration')
        return self.next(samples)

    def get_duration(self):
        raise NotImplementedException

    def is_complete(self):
        raise NotImplementedException


class ContinuousWaveform(Waveform):

    def n_samples_remaining(self):
        return np.inf

    def n_samples(self):
        return np.inf

    def get_duration(self):
        return np.inf

    def is_complete(self):
        return False


class FixedWaveform(Waveform):

    def __init__(self, fs, waveform):
        self.fs = fs
        self.waveform = waveform
        self.reset()

    def reset(self):
        self.offset = 0
        self.complete = False

    def next(self, samples):
        samples = int(samples)
        waveform = self.waveform[self.offset:self.offset+samples]
        waveform_samples = waveform.shape[-1]
        if waveform_samples < samples:
            padding = samples-waveform_samples
            pad = np.zeros(padding)
            waveform = np.concatenate((waveform, pad), axis=-1)
        self.offset += samples
        return waveform

    def n_samples_remaining(self):
        remaining = len(self.waveform)-self.offset
        return np.clip(remaining, 0, np.inf)

    def n_samples(self):
        return len(self.waveform)

    def get_duration(self):
        return len(self.waveform)/self.fs

    def is_complete(self):
        return self.offset >= len(self.waveform)


class Carrier(Waveform):
    '''
    A continuous waveform
    '''

    def get_duration(self):
        return np.inf

    def n_samples_remaining(self):
        return np.inf

    def is_complete(self):
        return False


class Modulator(Waveform):
    '''
    Modulates an input waveform
    '''
    def get_duration(self):
        return self.input_factory.get_duration()

    def n_samples_remaining(self):
        return self.input_factory.n_samples_remaining()

    def is_complete(self):
        return self.input_factory.is_complete()

    def reset(self):
        self.offset = 0
        self.input_factory.reset()

    def next(self, samples):
        waveform = self.env(samples) * self.input_factory.next(samples)
        self.offset += len(waveform)
        return waveform

    def env(self, samples):
        raise NotImplementedError


class GateFactory(Modulator):

    def __init__(self, fs, start_time, duration, input_factory):
        vars(self).update(locals())
        self.start_samples = int(round(start_time * fs))
        self.duration_samples = int(round(self.duration * fs))
        self.total_samples = self.start_samples + self.duration_samples
        self.reset()

    def get_duration(self):
        return self.start_time + self.duration

    def n_samples_remaining(self):
        return max(self.total_samples - self.offset, 0)

    def n_samples(self):
        return self.total_samples

    def is_complete(self):
        return self.offset >= self.total_samples

    def next(self, samples):
        token = self.input_factory.next(samples)
        lb = self.start_samples - self.offset
        ub = lb + self.duration_samples
        if lb >= 0:
            token[:lb] = 0
        if ub > 0:
            token[ub:] = 0
        self.offset += samples
        return token


################################################################################
# Gated envelopes
################################################################################
@fast_cache
def envelope(window, fs, duration, rise_time=None, offset=0, start_time=0,
             samples='auto'):
    '''
    Generates envelope. Can handle generating fragments (i.e.,
    incomplete sections of the waveform).

    Parameters
    ----------
    window : {'cosine-squared', etc.}
        Name of window
    fs : float
        Sampling rate
    duration : float
        Duration of envelope (from rise onset to rise offset)
    rise_time : {None, float}
        Rise time of envelope. If None, then there is no plateau/steady-state
        portion of the envelope.
    offset : int
        Offset to begin generating waveform at (in samples relative to start)
    start_time : float
        Start time of envelope
    samples : int
        Number of samples to generate for envelope.
    '''
    i_env_lb = int(round(start_time * fs))
    i_duration = int(round(duration * fs))
    i_env_ub = i_env_lb + i_duration

    if samples == 'auto':
        samples = i_env_lb + i_duration

    if rise_time is None:
        i_rise_time = int(np.floor(i_duration / 2))
        rise_time = i_rise_time / fs
    else:
        i_rise_time = int(round(rise_time * fs))

    if i_duration < (i_rise_time * 2):
        m = f'Rise time ({rise_time}s) longer than envelope duration ({duration}s)'
        raise ValueError(m)

    if window == 'cosine-squared':
        ramp = cos2ramp(2 * i_rise_time)
    else:
        ramp = getattr(signal.windows, window)(2 * i_rise_time)

    # Maximum number of steady state samples possible to return. If it exceeds
    # samples, clip it.
    n_ss_max = i_duration - 2 * i_rise_time

    def get_i(offset, i_start):
        return max(offset - i_start, 0)

    def get_n(i_max, offset, i_start, max_n):
        return np.clip(i_max - (offset - i_start), 0, min(i_max, max_n))

    n_null_pre = get_n(i_env_lb, offset, 0, samples)
    samples -= n_null_pre

    i_onset = get_i(offset, i_env_lb)
    n_onset = get_n(i_rise_time, offset, i_env_lb, samples)
    samples -= n_onset

    n_ss = get_n(n_ss_max, offset, i_env_lb + i_rise_time, samples)
    samples -= n_ss

    i_offset = get_i(offset, i_env_ub - i_rise_time)
    n_offset = get_n(i_rise_time, offset, i_env_ub - i_rise_time, samples)
    samples -= n_offset

    n_null_post = samples

    return np.concatenate((
        np.zeros(n_null_pre),
        ramp[i_onset:i_onset+n_onset],
        np.ones(n_ss),
        ramp[i_rise_time+i_offset:i_rise_time+i_offset+n_offset],
        np.zeros(n_null_post),
    ), axis=-1)


def cos2ramp(m):
    return np.sin(np.pi * np.arange(m) / m)**2


@fast_cache
def cos2envelope(fs, duration, rise_time, offset=0, start_time=0,
                 samples='auto'):
    return envelope('cosine-squared', fs, duration, rise_time, offset,
                    start_time, samples)


class EnvelopeFactory(GateFactory):

    def __init__(self, envelope, fs, duration, rise_time, input_factory,
                 start_time=0):
        self.rise_time = rise_time
        self.envelope = envelope
        super().__init__(fs, start_time, duration, input_factory)

    def next(self, samples):
        token = self.input_factory.next(samples)
        env = envelope(window=self.envelope, fs=self.fs,
                       duration=self.duration, rise_time=self.rise_time,
                       offset=self.offset, start_time=self.start_time,
                       samples=samples)
        waveform = env*token
        self.offset += samples
        return waveform


class Cos2EnvelopeFactory(EnvelopeFactory):

    def __init__(self, fs, duration, rise_time, input_factory,
                 start_time=0):
        super().__init__('cosine-squared', fs, duration, rise_time,
                         input_factory, start_time)

    def max_amplitude(self):
        return self.input_factory.max_amplitude()


################################################################################
# SAM envelope
################################################################################
@fast_cache
def sam_eq_power(depth):
    return (3.0/8.0*depth**2.0-depth+1.0)**0.5


@fast_cache
def sam_eq_phase(delay, depth, direction):
    if depth == 0:
        return 0
    z = 2.0/depth*sam_eq_power(depth)-2.0/depth+1
    phi = np.arccos(z)
    return 2.0*np.pi-phi if direction == 1 else phi


@fast_cache
def _sam_envelope(offset, samples, fs, depth, fm, delay, eq_phase, eq_power):
    delay_n = np.clip(int(delay*fs)-offset, 0, samples)
    delay_n = int(np.round(delay_n))
    sam_n = samples-delay_n

    sam_offset = offset-delay_n
    t = (np.arange(sam_n, dtype=np.double) + sam_offset)/fs
    sam_envelope = depth/2.0*np.cos(2.0*np.pi*fm*t+eq_phase)+1.0-depth/2.0

    # Ensure that we scale the waveform so that the total power remains equal
    # to that of an unmodulated token.
    sam_envelope *= 1.0/eq_power

    delay_envelope = np.ones(delay_n)
    return np.concatenate((delay_envelope, sam_envelope))


@fast_cache
def sam_envelope(offset, samples, fs, depth, fm, delay, equalize):
    if equalize:
        eq_phase = sam_eq_phase(delay, depth, 1)
        eq_power = sam_eq_power(depth)
    else:
        eq_phase = eq_power = 0
    return _sam_envelope(offset, samples, fs, depth, fm, delay, eq_phase,
                         eq_power)


class SAMEnvelopeFactory(Modulator):

    def __init__(self, fs, depth, fm, delay, direction, input_factory):
        vars(self).update(locals())
        self.eq_phase = sam_eq_phase(delay, depth, direction)
        self.eq_power = sam_eq_power(depth)
        self.reset()

    def env(self, samples):
        return _sam_envelope(self.offset, samples, self.fs, self.depth,
                             self.fm, self.delay, self.eq_phase, self.eq_power)


################################################################################
# Square wave envelope
################################################################################
def square_wave(fs, offset, samples, depth, fm, duty_cycle):
    fm_samples = fs / fm
    duty_samples = int(round(duty_cycle * fm_samples))

    env = np.full(samples, 1-depth, dtype=np.double)
    fm_start = fm_samples * (offset // fm_samples) - offset
    for t in np.arange(fm_start, samples, fm_samples):
        t = int(np.round(t))
        lb = np.clip(t, 0, samples)
        ub = np.clip(t + duty_samples, 0, samples)
        env[lb:ub] = 1
    return env


class SquareWaveEnvelopeFactory(Modulator):

    def __init__(self, fs, depth, fm, duty_cycle, calibration, input_factory):
        vars(self).update(locals())

    def env(self, samples):
        return square_wave(self.fs, self.offset, samples, self.depth, self.fm,
                           self.duty_cycle)


################################################################################
# Broadband noise
################################################################################
class BroadbandNoiseFactory(Carrier):
    '''
    Factory for generating continuous bandlimited noise
    '''
    def __init__(self, fs, level, seed=1, equalize=False, polarity=1, calibration=None):
        self.fs = fs
        self.level = level
        self.seed = seed
        self.calibration = calibration
        self.polarity = polarity

        if equalize:
            raise ValueError('Equalization of broadband noise not implemented')

        if calibration is None:
            self.sf = level
        else:
            self.sf = calibration.get_mean_sf(0, fs, level)

        # The RMS value of noise drawn from a uniform distribution is
        # amplitude/sqrt(3). By setting the low and high to sqrt(3) and
        # multiplying by the scaling factors, we can ensure that the noise is
        # initially generated with the desired RMS. Correcting after generating
        # the noise (e.g., by dividing by the RMS of the generated sample) is
        # problematic due to the random nature of the noise itself. To the
        # extent possible, psiaudio expects reproducibility regardless of the
        # number of samples the noise is segmented into.
        self.low = -np.sqrt(3) * self.sf
        self.high = np.sqrt(3) * self.sf

        self.reset()

    def reset(self):
        self.state = np.random.RandomState(self.seed)

    def next(self, samples):
        return self.polarity * self.state.uniform(low=self.low, high=self.high, size=samples)


def broadband_noise(fs, level, duration, seed=1, equalize=False, polarity=1,
                    calibration=None):
    kwargs = locals()
    kwargs.pop('duration')
    factory = BroadbandNoiseFactory(**kwargs)
    samples = int(round(duration * fs))
    return factory.next(samples)


################################################################################
# Bandlimited noise
################################################################################
@fast_cache
def _calculate_bandlimited_noise_filter(fs, fl, fh, fls, fhs,
                                        passband_attenuation,
                                        stopband_attenuation):
    Wp = np.array([fl, fh])/(0.5*fs)
    Ws = np.array([fls, fhs])/(0.5*fs)
    b, a = signal.iirdesign(Wp, Ws, passband_attenuation, stopband_attenuation)
    if np.any(np.abs(np.roots(a)) >= 1):
        raise ValueError('Unstable filter coefficients')
    zi = signal.lfilter_zi(b, a)
    return b, a, zi


@fast_cache
def _calculate_bandlimited_noise_iir(fs, calibration, fl, fh):
    duration = 2.0/fl
    iir = calibration.get_iir(fs, fl, fh, duration)
    zi = signal.lfilter_zi(iir, [1])
    return iir, zi


class BandlimitedNoiseFactory(Carrier):
    '''
    Factory for generating continuous bandlimited noise using IIR filters.
    Equalization requires a calibration that was generated using Golay.

    See BandlimitedFIRNoiseFactory as an alternative.
    '''
    def __init__(self, fs, seed, level, fl, fh, filter_rolloff,
                 passband_attenuation, stopband_attenuation, equalize=False,
                 polarity=1, calibration=None, discard_initial_samples=True):

        self.fs = fs
        self.level = level
        self.seed = seed
        self.calibration = calibration
        self.filter_rolloff = filter_rolloff
        self.passband_attenuation = passband_attenuation
        self.stopband_attenuation = stopband_attenuation
        self.equalize = equalize
        self.polarity = polarity
        self.calibration = calibration
        self.fl = fl
        self.fh = fh
        self.discard_initial_samples = discard_initial_samples

        # Calculate the scaling factor for the noise
        pass_bandwidth = fh-fl
        if calibration is None:
            self.sf = level
        else:
            self.sf = calibration.get_mean_sf(fl, fh, level)

        # This was copied from the EPL CFTS. Need to figure out how this
        # equation works so we can document this better. But it works as
        # intended to scale the noise back to RMS=1.
        self.filter_sf = 1.0 / np.sqrt(pass_bandwidth * 2 / fs)

        # The RMS value of noise drawn from a uniform distribution is
        # amplitude/sqrt(3). By setting the low and high to sqrt(3) and
        # multiplying by the scaling factors, we can ensure that the noise is
        # initially generated with the desired RMS.
        self.low = -np.sqrt(3) * self.filter_sf * self.sf
        self.high = np.sqrt(3) * self.filter_sf * self.sf

        # Calculate the stop bandwidth as octaves above and below the passband.
        # Precompute the filter settings.
        fls, fhs = fl*(2**-filter_rolloff), fh*(2**filter_rolloff)
        self.b, self.a, self.initial_bp_zi = \
            _calculate_bandlimited_noise_filter(fs, fl, fh, fls, fhs,
                                                passband_attenuation,
                                                stopband_attenuation)

        # Calculate the IIR filter if we are equalizing the noise.
        if equalize:
            self.iir, self.initial_iir_zi = \
                _calculate_bandlimited_noise_iir(fs, calibration, fl, fh)
        else:
            self.iir = self.initial_iir_zi = None

        self.reset()

    def reset(self):
        self.state = np.random.RandomState(self.seed)
        self.iir_zi = self.initial_iir_zi
        self.bp_zi = self.initial_bp_zi
        self.next(int(np.ceil(self.fs)))

    def next(self, samples):
        waveform = self.state.uniform(low=self.low, high=self.high, size=samples)
        if self.equalize:
            waveform, self.iir_zi = signal.lfilter(self.iir, [1], waveform, zi=self.iir_zi)
        waveform, self.bp_zi = signal.lfilter(self.b, self.a, waveform, zi=self.bp_zi)
        return waveform * self.polarity


def bandlimited_noise(fs, level, fl, fh, duration, filter_rolloff=1,
                      passband_attenuation=1, stopband_attenuation=80,
                      equalize=False, polarity=1, seed=1, calibration=None):
    args = locals()
    args.pop('duration')
    factory = BandlimitedNoiseFactory(**args)
    samples = int(round(duration * fs))
    return factory.next(samples)


################################################################################
# Bandlimited FIR noise factory
################################################################################
class BandlimitedFIRNoiseFactory(Carrier):
    '''
    Factory for generating continuous shaped noise using FIR filters.

    This is similar to shaped noise, but with simpler inputs if all you want is
    bandlimited noise (i.e., no requirement to generate the dictionary of
    gains).
    '''
    def __init__(self, fs, fl, fh, level, ntaps=1001, window='hann',
                 polarity=1, seed=None, equalize=False, calibration=None):
        vars(self).update(locals())

        if calibration is None:
            raise NotImplemented

        # Calculate the gains for the shaped noise
        if equalize:
            freq = np.arange(fl, fh + 1)
            sf = calibration.get_sf(freq, level)
        else:
            freq = np.array([fl, fh])
            sf = calibration.get_mean_sf(fl, fh, level)
            sf = np.full_like(freq, fill_value=sf)

        freq = np.concatenate(([0, fl / 1.1], freq, [fh * 1.1, fs / 2]))
        sf = np.pad(sf, 2)

        self.taps = signal.firwin2(ntaps, freq=freq, gain=sf, window=window, fs=fs)
        self.initial_zi = signal.lfilter_zi(self.taps, [1])

        # The RMS value of noise drawn from a uniform distribution is
        # amplitude/sqrt(3). By setting the low and high to sqrt(3) and
        # multiplying by the scaling factors, we can ensure that the noise is
        # initially generated with the desired RMS.
        self.scale = np.sqrt(3)

        self.reset()

    def max_amplitude(self):
        return np.abs(self.taps).sum()

    def reset(self):
        self.zi = self.initial_zi
        self.state = np.random.RandomState(self.seed)
        self.next(len(self.initial_zi + 1))

    def next(self, samples):
        waveform = self.state.uniform(low=-self.scale, high=self.scale, size=samples)
        waveform, self.zi = signal.lfilter(self.taps, [1], waveform, zi=self.zi)
        return waveform * self.polarity


def bandlimited_fir_noise(fs, level, fl, fh, duration, ntaps=10001,
                          window='hann', polarity=1, seed=1,
                          calibration=None, equalize=True):
    '''
    Generate shaped noise using `scipy.signal.firwin2`.

    Parameters
    ----------
    fs : float
        Sampling rate
    level : float
        Level in units of calibration. If no calibration is provided, noise
        will be scaled such that `rms(noise) == level`.
    fl : float
        Lower bound of noise band (Hz).
    fh : float
        Upper bound of noise band (Hz).
    ntaps : int
        Number of taps to use for filter calculation. See
        `scipy.signal.firwin2` for hints on choosing a reasonable value.
    window : {string, (string, float), float, None}
        Window function to use. See `window` parameter of
        `scipy.signal.firwin2` for additional details on acceptable values.
    polarity : {-1, 1}
        Polarity of noise. Useful if you need to present two trials in inverted
        polarity to cancel out electrical artifacts.
    seed : int
        Seed to use for random number generator.
    calibration : {BaseCalibration, None}
        Instance of a `psiaudio.calibration.BaseCalibration` or subclass
        thereof. Used to determine scaling factor for noise amplitude.
    '''
    args = locals()
    args.pop('duration')
    factory = BandlimitedFIRNoiseFactory(**args)
    samples = int(round(duration * fs))
    return factory.next(samples)


################################################################################
# Shaped noise
################################################################################
def _calculate_firwin2_taps(gains, fs, window, ntaps):
    freqs = list(gains.keys())
    gains = util.dbi(list(gains.values()))
    taps = signal.firwin2(ntaps, freqs, gains, fs=fs, window=window)
    initial_zi = signal.lfilter_zi(taps, [1])
    return taps, initial_zi


class ShapedNoiseFactory(Carrier):
    '''
    Factory for generating continuous shaped noise using FIR filters.
    '''
    def __init__(self, fs, level, gains, ntaps=1001, window='hann',
                 polarity=1, seed=None, calibration=None):
        vars(self).update(locals())

        self.taps, self.initial_zi = _calculate_firwin2_taps(gains, fs, window, ntaps)
        self.sf = level if calibration is None else calibration.get_mean_sf(0, fs/2, level)

        # Calculate how much the filter attenuates a *broadband* (i.e., white
        # noise) signal. This calculation is obviously not accurate for other
        # types of signals.
        w, h = signal.freqz(self.taps, fs=fs)
        h_mean = np.mean(np.abs(h) ** 2) ** 0.5
        self.filter_sf = 1 / h_mean

        # The RMS value of noise drawn from a uniform distribution is
        # amplitude/sqrt(3). By setting the low and high to sqrt(3) and
        # multiplying by the scaling factors, we can ensure that the noise is
        # initially generated with the desired RMS.
        self.scale = np.sqrt(3) * self.filter_sf * self.sf
        self.reset()

    def reset(self):
        self.zi = self.initial_zi
        self.state = np.random.RandomState(self.seed)
        self.next(len(self.initial_zi + 1))

    def next(self, samples):
        waveform = self.state.uniform(low=-self.scale, high=self.scale, size=samples)
        waveform, self.zi = signal.lfilter(self.taps, [1], waveform, zi=self.zi)
        return waveform * self.polarity


def shaped_noise(fs, level, gains, duration, ntaps=10001, window='hann',
                 polarity=1, seed=1, calibration=None):
    '''
    Generate shaped noise using `scipy.signal.firwin2`.

    Parameters
    ----------
    fs : float
        Sampling rate
    level : float
        Level in units of calibration. If no calibration is provided, noise
        will be scaled such that `rms(noise) == level`.
    gains : dict
        Dictionary mapping frequency breakpoints (Hz) to gain (dB).
    ntaps : int
        Number of taps to use for filter calculation. See
        `scipy.signal.firwin2` for hints on choosing a reasonable value.
    window : {string, (string, float), float, None}
        Window function to use. See `window` parameter of
        `scipy.signal.firwin2` for additional details on acceptable values.
    polarity : {-1, 1}
        Polarity of noise. Useful if you need to present two trials in inverted
        polarity to cancel out electrical artifacts.
    seed : int
        Seed to use for random number generator.
    calibration : {BaseCalibration, None}
        Instance of a `psiaudio.calibration.BaseCalibration` or subclass
        thereof. Used to determine scaling factor for noise amplitude.
    '''
    args = locals()
    args.pop('duration')
    factory = ShapedNoiseFactory(**args)
    samples = int(round(duration * fs))
    return factory.next(samples)


################################################################################
# Shaped noise
################################################################################
def _calculate_firwin2_taps(gains, fs, window, ntaps):
    freqs = list(gains.keys())
    gains = util.dbi(list(gains.values()))
    taps = signal.firwin2(ntaps, freqs, gains, fs=fs, window=window)
    initial_zi = signal.lfilter_zi(taps, [1])
    return taps, initial_zi


class ShapedNoiseFactory(Carrier):
    '''
    Factory for generating continuous shaped noise using FIR filters.
    '''
    def __init__(self, fs, level, gains, ntaps=1001, window='hann',
                 polarity=1, seed=None, calibration=None):
        vars(self).update(locals())

        self.taps, self.initial_zi = _calculate_firwin2_taps(gains, fs, window, ntaps)
        self.sf = level if calibration is None else calibration.get_mean_sf(0, fs/2, level)

        # Calculate how much the filter attenuates a *broadband* (i.e., white
        # noise) signal. This calculation is obviously not accurate for other
        # types of signals.
        w, h = signal.freqz(self.taps, fs=fs)
        h_mean = np.mean(np.abs(h) ** 2) ** 0.5
        self.filter_sf = 1 / h_mean

        # The RMS value of noise drawn from a uniform distribution is
        # amplitude/sqrt(3). By setting the low and high to sqrt(3) and
        # multiplying by the scaling factors, we can ensure that the noise is
        # initially generated with the desired RMS.
        self.scale = np.sqrt(3) * self.filter_sf * self.sf
        self.reset()

    def reset(self):
        self.zi = self.initial_zi
        self.state = np.random.RandomState(self.seed)
        self.next(len(self.initial_zi + 1))

    def next(self, samples):
        waveform = self.state.uniform(low=-self.scale, high=self.scale, size=samples)
        waveform, self.zi = signal.lfilter(self.taps, [1], waveform, zi=self.zi)
        return waveform * self.polarity


def shaped_noise(fs, level, gains, duration, ntaps=10001, window='hann',
                 polarity=1, seed=1, calibration=None):
    '''
    Generate shaped noise using `scipy.signal.firwin2`.

    Parameters
    ----------
    fs : float
        Sampling rate
    level : float
        Level in units of calibration. If no calibration is provided, noise
        will be scaled such that `rms(noise) == level`.
    gains : dict
        Dictionary mapping frequency breakpoints (Hz) to gain (dB).
    ntaps : int
        Number of taps to use for filter calculation. See
        `scipy.signal.firwin2` for hints on choosing a reasonable value.
    window : {string, (string, float), float, None}
        Window function to use. See `window` parameter of
        `scipy.signal.firwin2` for additional details on acceptable values.
    polarity : {-1, 1}
        Polarity of noise. Useful if you need to present two trials in inverted
        polarity to cancel out electrical artifacts.
    seed : int
        Seed to use for random number generator.
    calibration : {BaseCalibration, None}
        Instance of a `psiaudio.calibration.BaseCalibration` or subclass
        thereof. Used to determine scaling factor for noise amplitude.
    '''
    args = locals()
    args.pop('duration')
    factory = ShapedNoiseFactory(**args)
    samples = int(round(duration * fs))
    return factory.next(samples)


################################################################################
# Tone
################################################################################
def tone(fs, frequency, level, phase=0, polarity=1, calibration=None,
         samples='auto', offset=0, duration=None):

    rms = level if calibration is None else calibration.get_sf(frequency, level)
    if samples == 'auto':
        if duration is None:
            raise ValueError('Must provide either duration or samples')
        samples = int(round(duration * fs))
    elif duration is not None:
        raise ValueError('Cannot specify duration if samples is provided')

    # Since the scaling factor is based on Vrms, we need to convert this to the
    # peak-to-peak scaling factor.
    t = (np.arange(samples, dtype=np.double) + offset)/fs
    return polarity * rms * np.sqrt(2) * np.cos(2 * np.pi * t * frequency + phase)


class ToneFactory(Carrier):

    def __init__(self, fs, frequency, level, phase=0, polarity=1,
                 calibration=None):
        vars(self).update(locals())
        self.reset()

    def reset(self):
        self.offset = 0

    def next(self, samples):
        # Note. At least for 5 msec tones it's faster to just compute the array
        # rather than cache the result.
        waveform = tone(self.fs, self.frequency, self.level, self.phase,
                        self.polarity, calibration=self.calibration,
                        offset=self.offset, samples=samples)
        self.offset += samples
        return waveform


################################################################################
# SAMTone
################################################################################
def sam_tone(fs, fc, fm, level, depth=1, phase=0, polarity=1, calibration=None,
             samples='auto', offset=0, duration=None, eq_power=True,
             equalize=True):
    '''
    Generates a SAM tone

    Unlike the alternate approach of combining a SAM envelope with a tone
    carrier, this is specially designed to handle speakers with nonlinear
    outputs as a function of frequency by adjusting the levels of the harmonics
    accordingly.

    Parameters
    ----------
    eq_power : bool
        If True, compensate for modualtion depth so the overall RMS is the same
        when varying modulation depth.  This is useful when doing AM detection
        studies as it minimizes loudness cues.
    equalize : bool
        If True and the calibration is provided, ensure that the sidebands
        (fc-fm, fc+fm) are adjusted to reflect the actual output of the speaker
        at those frequencies. If not provided, the calibration at the carrier
        frequency is used for both the carrier and modulation sidebands.

    # TODO
    '''
    frequencies = fc + fm * np.arange(-1, 2)

    if calibration is not None:
        if equalize:
            sf = calibration.get_sf(frequencies, level)
        else:
            sf = calibration.get_sf(fc, level)
    else:
        sf = level

    sf = sf * np.array([0.25, 0.5, 0.25])
    if eq_power:
        sf /= sam_eq_power(depth)

    if samples == 'auto':
        if duration is None:
            raise ValueError('Must provide either duration or samples')
        samples = int(round(duration * fs))
    elif duration is not None:
        raise ValueError('Cannot specify duration if samples is provided')

    # Since the scaling factor is based on Vrms, we need to convert this to the
    # peak-to-peak scaling factor.
    t = (np.arange(samples, dtype=np.double) + offset)/fs
    s = polarity * sf[..., np.newaxis] * np.sqrt(2) * \
        np.cos(2 * np.pi * t * frequencies[..., np.newaxis] + phase)
    return np.sum(s, axis=0)


class SAMToneFactory(Carrier):

    def __init__(self, fs, fc, fm, level, depth=1, phase=0, polarity=1,
                 eq_power=True, equalize=True, calibration=None):
        vars(self).update(locals())
        self.reset()

    def reset(self):
        self.offset = 0

    def next(self, samples):
        # Note. At least for 5 msec tones it's faster to just compute the array
        # rather than cache the result.
        waveform = sam_tone(
            fs=self.fs, fc=self.fc, fm=self.fm, level=self.level,
            depth=self.depth, phase=self.phase, polarity=self.polarity,
            eq_power=self.eq_power, equalize=self.equalize,
            calibration=self.calibration, offset=self.offset, samples=samples
        )
        self.offset += samples
        return waveform


################################################################################
# Silence
################################################################################
class SilenceFactory(Carrier):
    '''
    Generate silence

    All channels require at least one continuous output. If no token is
    specified for the continuous output, silence is used.

    Notes
    -----
    The fill_value can be set to a number other than zero for testing (e.g., to
    characterize the effect of a transformation).
    '''

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def next(self, samples):
        return np.full(samples, self.fill_value)

    def reset(self):
        pass


################################################################################
# Square waveform
################################################################################
class SquareWaveFactory(Carrier):

    def __init__(self, fs, level, frequency, duty_cycle):
        self.sf = level
        self.cycle_samples = int(round(fs/frequency))
        self.on_samples = int(round(self.cycle_samples * duty_cycle))
        self.reset()

    def reset(self):
        self.offset = 0

    def next(self, samples):
        waveform = np.zeros(samples)
        o = self.offset % self.cycle_samples
        while o < samples:
            waveform[o:o+self.on_samples] = self.sf
            o += self.cycle_samples
        return waveform


################################################################################
# Repeat
################################################################################
class RepeatFactory(FixedWaveform):

    def __init__(self, fs, n, skip_n, rate, delay, input_factory):
        vars(self).update(locals())
        self.reset()

    def get_duration(self):
        return (self.n + self.skip_n) / self.rate

    def reset(self):
        self.offset = 0
        self.input_factory.reset()
        waveform = self.input_factory.get_samples_remaining()
        self.waveform = repeat(waveform, self.fs, self.n, self.skip_n,
                               self.rate, self.delay)

    def max_amplitude(self):
        return self.input_factory.max_amplitude()


def repeat(waveform, fs, n, skip_n, rate, delay):
    s_period = int(round(fs / rate))
    s_waveform = len(waveform)
    s_delay = int(round(fs * delay))

    if s_waveform > (s_period - s_delay):
        t_waveform = s_waveform / fs
        t_period = s_period / fs
        t_delay = s_delay / fs
        raise ValueError('Waveform too long to repeat. '
                         f'Waveform is {t_waveform} s long starting at a delay '
                         f'of {t_delay}. Total repeat period is {t_period}.')

    result = np.zeros((n + skip_n, s_period))
    result[skip_n:, s_delay:s_delay+s_waveform] = waveform
    return result.ravel()


################################################################################
# Chirp
################################################################################
def chirp(fs, start_frequency, end_frequency, duration, level,
          calibration=None, window='boxcar', equalize=False):
    '''
    Notes
    -----
    Windowing algorithm was implemented as described in Neumann et al., 1994 to
    enable implementation of the Hann windowed chirp in the middel ear acoustic
    reflex assay described by Valero et al., 2016.
    '''

    # Compute the window and get the normalized integral. This is used to
    # adjust the instantaneous frequency so that we "dwell" longer on
    # frequencies that are attenuated at the boundaries of the window.
    n = int(fs * duration)
    w = signal.get_window(window, n)
    wi_norm = np.cumsum(w ** 2) / np.sum(w ** 2)
    ifreq = wi_norm * (end_frequency - start_frequency) + start_frequency

    # Now, we calculate the phase (the time integral of frequency). Divide by
    # fs is equivalent to multiplying by the period (i.e., dt).
    phase = np.cumsum(ifreq) / fs

    # Figure out scaling factor. If no calibration is provided, assume that
    # level is specified in Vrms.
    if calibration is None:
        if equalize:
            raise ValueError('Cannot equalize signal without calibration')
        sf = level
    else:
        if not equalize:
            sf = calibration.get_mean_sf(start_frequency, end_frequency, level)
        else:
            sf = calibration.get_sf(ifreq, level)

    # We need to normalize the window so that it has a RMS of 1. Then, we
    # multiply by the square root of 2 since we are using the sin function
    # (e.g. Vpeak = np.sqrt(2) * Vrms). Finally, multiply by the scaling factor
    # that gives us our desired level.
    w /= util.rms(w)
    return np.sqrt(2) * sf * w * np.sin(2 * np.pi * phase)


class ChirpFactory(FixedWaveform):

    def __init__(self, fs, start_frequency, end_frequency, duration, level,
                 calibration, window='boxcar', equalize=False):
        kwargs = locals()
        kwargs.pop('self')
        vars(self).update(kwargs)
        self.waveform = chirp(**kwargs)
        self.reset()

    def max_amplitude(self):
        return np.abs(self.waveform).max()


################################################################################
# Click
################################################################################
class ClickFactory(FixedWaveform):

    def __init__(self, fs, duration, level, polarity, calibration):
        vars(self).update(locals())
        n = int(fs*duration)
        sf = calibration.get_sf(0, level)
        self.waveform = polarity * sf * np.ones(n)
        self.reset()


################################################################################
# Bandlimited Click
################################################################################
def bandlimited_click(fs, flb, fub, window=0.1, level=1, calibration=None,
                      equalize=False):
    '''
    Generate bandlimited click.

    Parameters
    ----------

    The click waveform will be symmetric around the center of the window.
    '''

    n_window = int(round(window * fs))
    n = int(round(fs))
    freq = np.fft.rfftfreq(n, d=1/fs)
    psd = np.zeros_like(freq)
    m = (freq >= flb) & (freq < fub)
    psd[m] = 1

    if calibration is None:
        if equalize:
            raise ValueError('Cannot equalize signal without calibration')
        sf = level
    else:
        if not equalize:
            sf = calibration.get_sf(freq[m], level)
            psd[m] = psd[m] * sf / sf.mean()
        else:
            sf = calibration.get_sf(freq[m], level)

    sf = util.dbi(util.db(sf).mean())

    csd = psd * np.exp(-1j * freq * 2 * np.pi * 0.5)
    waveform = util.csd_to_signal(csd)
    waveform = waveform / waveform.ptp() * sf
    lb = int(round(n / 2 - n_window / 2))
    waveform = waveform[lb:lb+n_window]
    return waveform


class BandlimitedClickFactory(FixedWaveform):

    def __init__(self, fs, flb, fub, window, level, calibration):
        vars(self).update(locals())
        self.waveform = bandlimited_click(fs, flb, fub, window, level=level,
                                          calibration=calibration)
        self.reset()


################################################################################
# Wavfiles
################################################################################
@fast_cache
def load_wav(fs, filename, level=None, calibration=None, normalization=None):
    '''
    Load wav file, scale, and resample

    Parameters
    ----------
    fs : float
        Desired sampling rate for wav file. If wav file sampling rate is
        different, it will be resampled to the correct sampling rate using a
        FFT-based resampling algorithm.
    filename : {str, Path}
        Path to wav file
    level : {None, float}
        Level to present wav files at. If normalization is `'pe'`, level will
        be in units of peSPL (assuming calibration is in units of SPL). If
        normalization is in `'rms'`, level will be dB SPL RMS.
    calibration : {None, Calibration}
        Used to scale waveform to appropriate peSPL. If not provided,
        waveform is not scaled.
    normalization : {None, 'pe', 'rms'}
        Method for rescaling waveform. If None, no rescaling is done. If
        `'pe'`, rescales to peak-equivalent so the max value of the waveform
        matches the target level. If `'rms'`, rescales so that the RMS value of
        the waveform matches the target level.
    '''
    log.warning('Loading %s', filename)
    file_fs, waveform = wavfile.read(filename, mmap=True)

    # Rescale to range -1.0 to 1.0
    if waveform.dtype != np.float32:
        ii = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32)
        waveform = (waveform - ii.min) / (ii.max - ii.min) * 2 - 1

    if normalization is None:
        pass
    elif normalization == 'pe':
        waveform = waveform / waveform.max()
    elif normalization == 'rms':
        waveform = waveform / util.rms(waveform)
    else:
        raise ValueError(f'Unrecognized normalization: {normalization}')

    if calibration is not None:
        sf = calibration.get_sf(1e3, level)
        waveform *= sf

    # Resample if sampling rate does not match
    if fs != file_fs:
        waveform_resampled = util.resample_fft(waveform, file_fs, fs)
        return waveform_resampled

    return waveform


class WavFileFactory(FixedWaveform):

    def __init__(self, fs, filename, level=None, calibration=None,
                 normalization='pe'):
        self.fs = fs
        self.filename = filename
        self.level = level
        self.calibration = calibration
        self.normalization = normalization
        self.reset()

    @property
    def waveform(self):
        return load_wav(self.fs, self.filename, self.level, self.calibration,
                        normalization=self.normalization)


class WavSequenceFactory(ContinuousWaveform):

    def __init__(self, fs, path, level=None, calibration=None, duration=-1,
                 normalization='pe'):
        '''
        Parameters
        ----------
        fs : float
            Sampling rate of output. If wav file sampling rate is different, it
            will be resampled to the correct sampling rate using a FFT-based
            resampling algorithm.
        path : {str, Path}
            Path to directory containing wav files.
        level : float
            Level to present wav files at (currently peSPL due to how the
            normalization works).
        calibration : instance of Calibration
            Used to scale waveform to appropriate peSPL. If not provided,
            waveform is not scaled.
        duration : {None, float}
            Duration of each wav file. If None, the wav file is loaded at the
            beginning so that durations can be established. For large
            directories, this can slow down the startup time of the program.
            Knowing the exact duration may be important for some downstream
            operations. For example, epoch extraction relative to the
            presentation time of a particular wav file; estimating the overall
            duration of the entire wav sequence, etc.  If you don't have a need
            for these operations and want to speed up loading of wav files, set
            this value to -1 (the default).
        normalization : {'pe', 'rms'}
            Method for rescaling waveform. If `'pe'`, rescales to
            peak-equivalent so the max value of the waveform matches the target
            level. If `'rms'`, rescales so that the RMS value of the waveform
            matches the target level.
        '''
        self.wav_files = wavs_from_path(fs, path, level=level,
                                        calibration=calibration,
                                        normalization=normalization)
        self.fs = fs
        self.duration = duration
        self.reset()

    def reset(self):
        self.queue = queue.BlockedRandomSignalQueue(self.fs)
        metadata = [{'filename': fh.filename.name} for fh in self.wav_files]
        self.queue.extend(self.wav_files, np.inf, duration=self.duration, metadata=metadata)

    def next(self, samples):
        return self.queue.pop_buffer(samples)

    def connect(self, *args, **kwargs):
        return self.queue.connect(*args, **kwargs)


def wavs_from_path(fs, path, *args, **kwargs):
    return [WavFileFactory(fs, filename, *args, **kwargs) \
            for filename in Path(path).glob('*.wav')]


################################################################################
# Basic utility functions for the most common use-cases.
################################################################################
def ramped_tone(fs, frequency, level, duration, rise_time=None,
                window='cosine-squared', phase=0, calibration=None):
    carrier = tone(fs=fs, frequency=frequency, level=level, phase=phase,
                   calibration=calibration, duration=duration)
    env = envelope(window=window, fs=fs, rise_time=rise_time,
                   duration=duration)
    return carrier * env
