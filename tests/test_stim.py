import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import signal


from psiaudio import calibration
from psiaudio import stim
from psiaudio import util


def test_tone_factory():
    fs = 100e3
    frequency = 5e3
    # This is in dB!
    level = 0
    phase = 0
    polarity = 1
    cal = calibration.FlatCalibration.unity()

    tone = stim.ToneFactory(
        fs=100e3,
        frequency=frequency,
        level=level,
        phase=phase,
        polarity=polarity,
        calibration=cal
    )
    samples = 1000

    # Check initial segment
    waveform = tone.next(samples)
    t = np.arange(0, samples, dtype=np.double) / fs
    expected = np.sqrt(2) * np.cos(2 * np.pi * t * frequency)
    assert_array_equal(waveform, expected)

    # Check next segment
    t = np.arange(samples, samples*2, dtype=np.double) / fs
    waveform = tone.next(samples)
    expected = np.sqrt(2) * np.cos(2 * np.pi * t * frequency)
    assert_array_equal(waveform, expected)

    # Verify reset
    tone.reset()
    samples = 2000
    t = np.arange(0, samples, dtype=np.double) / fs
    waveform = tone.next(samples)
    expected = np.sqrt(2) * np.cos(2 * np.pi * t * frequency)
    assert_array_equal(waveform, expected)


def test_silence_factory():
    silence = stim.SilenceFactory()
    waveform = silence.next(100)
    expected = np.zeros(100)
    assert_array_equal(waveform, expected)

    silence.reset()
    waveform = silence.next(100)
    assert_array_equal(waveform, expected)


def test_cos2envelope():
    fs = 100e3
    offset = 0
    samples = 400000
    start_time = 0
    rise_time = 0.25
    duration = 4.0

    expected = np.ones(400000)
    t_samples = round(rise_time*fs)
    t_env = np.linspace(0, rise_time, t_samples, endpoint=False)

    ramp = np.sin(2*np.pi*t_env*1.0/rise_time*0.25)**2
    expected[:t_samples] = ramp

    ramp = np.sin(2*np.pi*t_env*1.0/rise_time*0.25 + np.pi/2)**2
    expected[-t_samples:] = ramp

    actual = stim.cos2envelope(fs, duration, rise_time, start_time, offset,
                               samples)

    assert_array_almost_equal(actual, expected, 4)


def test_cos2envelope_factory():
    fs = 100e3
    frequency = 5e3
    # This is in dB!
    level = 0
    phase = 0
    polarity = 1
    cal = calibration.FlatCalibration.unity()

    tone = stim.ToneFactory(
        fs=fs,
        frequency=frequency,
        level=level,
        phase=phase,
        polarity=polarity,
        calibration=cal
    )

    duration = 1
    rise_time = 0.5e-3

    ramped_tone = stim.Cos2EnvelopeFactory(
        fs=fs,
        duration=duration,
        rise_time=rise_time,
        input_factory=tone,
    )

    w1 = ramped_tone.get_samples_remaining()

    # Samples don't actually return to zero at the boundaries based on how we
    # do the calculations.
    assert w1[0] == pytest.approx(0, abs=1e-2)
    assert w1[-1] == pytest.approx(0, abs=1e-2)
    assert w1.shape == (int(fs),)
    assert_array_almost_equal(w1[1:1001], w1[-1000:][::-1])

    w2 = stim.ramped_tone(fs=fs, frequency=frequency, level=level,
                          calibration=cal, duration=duration,
                          rise_time=rise_time)

    assert_array_equal(w1, w2)


def test_envelope(fs, stim_window):
    actual = stim.envelope(stim_window, fs, duration=1, rise_time=0.5)
    if stim_window == 'cosine-squared':
        # The scipy window function calculates the window points at the bin
        # centers, whereas my approach is to calculate the window points at the
        # left edge of the bin.
        expected = signal.windows.cosine(len(actual)) ** 2
        assert_array_almost_equal(actual, expected, 4)
    else:
        expected = getattr(signal.windows, stim_window)(len(actual))
        assert_array_equal(actual, expected)


def _test_factory(factory_class, kwargs, chunksize, n_chunks, exact=True):
    factory = factory_class(**kwargs)
    chunked_samples = [factory.next(chunksize) for i in range(n_chunks)]
    chunked_samples = np.concatenate(chunked_samples, axis=-1)
    factory.reset()
    unchunked_samples = factory.next(chunksize * n_chunks)
    if exact:
        assert_array_equal(unchunked_samples, chunked_samples)
    else:
        assert_array_almost_equal(unchunked_samples, chunked_samples)


def _test_noise_helper(fn, kwargs, stim_level, stim_calibration):
    actual = fn(**kwargs)
    actual_level = stim_calibration.get_spl(1e3, util.rms(actual))
    assert actual_level == stim_level

    # Verify that this is frozen noise
    repeat = fn(**kwargs)
    assert_array_equal(actual, repeat)

    # Make sure a different seed behaves appropriately
    with pytest.raises(AssertionError):
        repeat = fn(**kwargs, seed=32)
        assert_array_equal(actual, repeat)


def test_broadband_noise(fs, stim_level, stim_duration, stim_calibration):
    kwargs = dict(fs=fs, level=stim_level, duration=stim_duration,
                  calibration=stim_calibration)
    _test_noise_helper(
        stim.broadband_noise,
        kwargs,
        pytest.approx(stim_level, abs=1),
        stim_calibration
    )


def test_broadband_noise_factory(fs, stim_level, stim_calibration, chunksize,
                                 n_chunks):
    kwargs = dict(fs=fs, level=stim_level, calibration=stim_calibration)
    _test_factory(stim.BroadbandNoiseFactory, kwargs, chunksize, n_chunks)


def test_bandlimited_noise(fs, stim_level, stim_duration, stim_fl, stim_fh,
                           stim_calibration):
    if fs in (25e3, 200e3):
        pytest.skip()
    if stim_duration == 0.001:
        abs_difference = 5
    elif stim_duration == 0.01:
        abs_difference = 2.5
    else:
        abs_difference = 1
    kwargs = dict(fs=fs, level=stim_level, fl=stim_fl, fh=stim_fh,
                  calibration=stim_calibration, duration=stim_duration)

    _test_noise_helper(
        stim.bandlimited_noise,
        kwargs,
        pytest.approx(stim_level, abs=abs_difference),
        stim_calibration
    )


def test_bandlimited_noise_factory(fs, stim_level, stim_fl, stim_fh,
                                   stim_calibration, chunksize, n_chunks):
    if fs in (25e3, 200e3):
        pytest.skip()
    kwargs = dict(fs=fs, level=stim_level, fl=stim_fl, fh=stim_fh,
                  calibration=stim_calibration, seed=1, filter_rolloff=1,
                  passband_attenuation=1, stopband_attenuation=80)
    _test_factory(stim.BandlimitedNoiseFactory, kwargs, chunksize, n_chunks)


@pytest.fixture
def shaped_noise_gains(fs, stim_fl, stim_fh):
    return {
        0: 0,
        stim_fl * 0.99: 0,
        stim_fl: -80,
        stim_fh: -80,
        stim_fh / 0.99: 0,
        fs / 2: 0,
    }


def test_shaped_noise(fs, stim_level, stim_duration, shaped_noise_gains, stim_calibration):
    if stim_duration == 0.001:
        abs_difference = 5
    elif stim_duration == 0.01:
        abs_difference = 2.5
    else:
        abs_difference = 1
    kwargs = dict(fs=fs, level=stim_level, gains=shaped_noise_gains,
                  calibration=stim_calibration, duration=stim_duration)
    _test_noise_helper(stim.shaped_noise,
                       kwargs,
                       pytest.approx(stim_level, abs=abs_difference),
                       stim_calibration)


def test_shaped_noise_factory(fs, stim_level, shaped_noise_gains,
                              stim_calibration, chunksize, n_chunks):
    kwargs = dict(fs=fs, level=stim_level, gains=shaped_noise_gains,
                  calibration=stim_calibration, seed=1)
    # Set exact to False. There seem to be some numerical precision issues, but
    # they are less than 1e-14.
    _test_factory(stim.ShapedNoiseFactory, kwargs, chunksize, n_chunks,
                  exact=False)
