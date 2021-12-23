import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import signal


from psiaudio import stim
from psiaudio import calibration


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


@pytest.mark.parametrize('window', ['cosine-squared', 'blackman'])
def test_envelope(fs, window):
    actual = stim.envelope(window, fs, duration=1, rise_time=0.5)
    if window == 'cosine-squared':
        # The scipy window function calculates the window points at the bin
        # centers, whereas my approach is to calculate the window points at the
        # left edge of the bin.
        expected = signal.windows.cosine(len(actual)) ** 2
        assert_array_almost_equal(actual, expected, 4)
    else:
        expected = getattr(signal.windows, window)(len(actual))
        assert_array_equal(actual, expected)
