import pytest

import numpy as np

from psiaudio.api import (
    CalibrationNFError, CalibrationTHDError, FlatCalibration,
    InterpCalibration, PointCalibration,
)

from psiaudio import util


@pytest.fixture
def relative_levels():
    # These are the levels, relative to unity gain (i.e., 0 dB) to equalize
    # these frequencies.
    return {
         300:  0.955,
         500: -0.682,
         700: -1.073,
        1000:  0.0,
        1500:  2.678,
        2000: -2.406,
        3000: -4.199,
        4000:  6.301,
        6000: -2.664,
    }


@pytest.fixture
def point_calibration(relative_levels):
    return PointCalibration.from_spl(relative_levels, vrms=1)


def make_tone(fs, f0, duration, phase=0):
    n = int(duration*fs)
    t = np.arange(n, dtype=np.double)/fs
    y = np.cos(2*np.pi*f0*t + phase)
    return y


@pytest.mark.benchmark(group='tone-phase')
def test_tone_phase_conv(benchmark):
    fs = 100e3
    f1 = 1e3
    p1 = 0
    t1 = make_tone(fs, f1, 1, p1)
    benchmark(util.tone_phase_conv, t1, fs, f1)


@pytest.mark.benchmark(group='tone-phase')
def test_tone_phase_fft(benchmark):
    fs = 100e3
    f1 = 1e3
    p1 = 0
    t1 = make_tone(fs, f1, 1, p1)
    benchmark(util.tone_phase_fft, t1, fs, f1)


@pytest.mark.benchmark(group='tone-power')
def test_tone_power_conv(benchmark):
    fs = 100e3
    f1 = 1e3
    p1 = 0
    t1 = make_tone(fs, f1, 1, p1)
    benchmark(util.tone_power_conv, t1, fs, f1)


@pytest.mark.benchmark(group='tone-power')
def test_tone_power_fft(benchmark):
    fs = 100e3
    f1 = 1e3
    p1 = 0
    t1 = make_tone(fs, f1, 1, p1)
    benchmark(util.tone_power_fft, t1, fs, f1)


def test_tone_util():
    fs = 100e3
    f1 = 1e3
    p1 = 0
    t1 = make_tone(fs, f1, 1, p1)

    f2 = 2e3
    p2 = np.pi/2
    t2 = make_tone(fs, f2, 1, p2)

    f3 = 32e3
    p3 = np.pi/4
    t3 = make_tone(fs, f3, 1, p3)

    rms = 1/np.sqrt(2)

    assert util.tone_power_conv(t1, fs, f1) == pytest.approx(rms)
    assert util.tone_power_conv(t2, fs, f2) == pytest.approx(rms)
    assert util.tone_power_conv(t3, fs, f3) == pytest.approx(rms)

    assert util.tone_phase_conv(t1, fs, f1) == pytest.approx(p1, abs=6)
    assert util.tone_phase_conv(t2, fs, f2) == pytest.approx(p2, abs=6)
    assert util.tone_phase_conv(t3, fs, f3) == pytest.approx(p3, abs=6)

    assert util.tone_power_fft(t1, fs, f1) == pytest.approx(rms)
    assert util.tone_power_fft(t2, fs, f2) == pytest.approx(rms)
    assert util.tone_power_fft(t3, fs, f3) == pytest.approx(rms)

    assert util.tone_phase_fft(t1, fs, f1) == pytest.approx(p1, abs=6)
    assert util.tone_phase_fft(t2, fs, f2) == pytest.approx(p2, abs=6)
    assert util.tone_phase_fft(t3, fs, f3) == pytest.approx(p3, abs=6)

    assert util.tone_power_conv(t1, fs, f1, window='flattop') == pytest.approx(rms)
    assert util.tone_power_conv(t2, fs, f2, window='flattop') == pytest.approx(rms)
    assert util.tone_power_conv(t3, fs, f3, window='flattop') == pytest.approx(rms)

    assert util.tone_phase_conv(t1, fs, f1, window='flattop') == pytest.approx(p1, abs=6)
    assert util.tone_phase_conv(t2, fs, f2, window='flattop') == pytest.approx(p2, abs=6)
    assert util.tone_phase_conv(t3, fs, f3, window='flattop') == pytest.approx(p3, abs=6)

    assert util.tone_power_fft(t1, fs, f1, window='flattop') == pytest.approx(rms)
    assert util.tone_power_fft(t2, fs, f2, window='flattop') == pytest.approx(rms)
    assert util.tone_power_fft(t3, fs, f3, window='flattop') == pytest.approx(rms)

    assert util.tone_phase_fft(t1, fs, f1, window='flattop') == pytest.approx(p1, abs=6)
    assert util.tone_phase_fft(t2, fs, f2, window='flattop') == pytest.approx(p2, abs=6)
    assert util.tone_phase_fft(t3, fs, f3, window='flattop') == pytest.approx(p3, abs=6)


def test_process_tone():
    fs = 100e3
    f1, p1 = 1e3, 0
    t1 = make_tone(fs, f1, 1, p1)

    f2, p2 = 500, np.pi/2
    t2 = make_tone(fs, f2, 1, p2)

    rms = 1/np.sqrt(2)

    # Build a 3D array of repetition x channel x time with two repetitions of
    # t1. The RMS power should be np.sqrt(2) by definition (e.g., if a tone's
    # peak to peak amplitude is X, then the RMS amplitude is X/np.sqrt(2)).
    signal = np.concatenate((t1[np.newaxis], t1[np.newaxis]))
    signal.shape = (2, 1, -1)
    result = util.process_tone(fs, signal, f1)
    assert np.allclose(result['rms'], rms)
    assert np.allclose(result['phase'], p1)
    assert result.shape == (1,5)

    # Build a 3D array of repetition x channel x time with two repetitions, but
    # designed such that the second repetition is t2 (and therefore will have 0
    # power at f1). This means that the average RMS power should be half.
    signal = np.concatenate((t1[np.newaxis], t2[np.newaxis]))
    signal.shape = (2, 1, -1)
    result = util.process_tone(fs, signal, f1)
    assert np.allclose(result['rms'], 0.5*rms)
    assert result.shape == (1,5)

    # Build a 3D array of repetition x channel x time with one repetition and
    # two channels (with channel 1 containing t1 and channel 2 containing t2).
    # This should return *two* numbers (one for each channel).
    signal = np.concatenate((t1[np.newaxis], t2[np.newaxis]))
    signal.shape = (1, 2, -1)
    result = util.process_tone(fs, signal, f1)
    assert np.allclose(result['rms'], [rms, 0])
    assert result.shape == (2,5)
    result = util.process_tone(fs, signal, f2)
    assert np.allclose(result['rms'], [0, rms])
    assert result.shape == (2,5)

    # Now, test the most simple case (e.g., single repetition, single channel).
    result = util.process_tone(fs, t1, f1)
    assert result['rms'] == pytest.approx(rms)

    # Now, test silence
    silence = np.random.normal(scale=1e-12, size=t1.shape)
    result = util.process_tone(fs, silence, f1)
    assert result['rms'] == pytest.approx(0)

    # Now, make sure we get an error for the noise floor.
    with pytest.raises(CalibrationNFError):
        result = util.process_tone(fs, silence, f1, min_snr=3, silence=silence)

    # Now, create a harmonic for t1 at 2*f1. This harmonic will have 0.1% the
    # power of t1.
    t1_harmonic = 1e-2*make_tone(fs, f1*2, 1)
    signal = t1 + t1_harmonic
    result = util.process_tone(fs, signal, f1, max_thd=2)
    assert result['rms'] == pytest.approx(rms)

    with pytest.raises(CalibrationTHDError):
        result = util.process_tone(fs, signal, f1, max_thd=1)


def test_unity_calibration():
    calibration = FlatCalibration.unity()
    assert calibration.get_sf(1000, 0) == 1
    assert calibration.get_sf(1000, 20) == 10
    assert calibration.get_sf(1000, 40) == 100
    assert calibration.get_spl(1000, 1) == 0
    assert calibration.get_spl(1000, 10) == 20
    assert calibration.get_spl(1000, 100) == 40


def test_flat_calibration():
    calibration = FlatCalibration(sensitivity=20*np.log10(1/20e-6))
    tests = {
         74:  0.1,
         94:  1.0,
        114: 10.0,
    }
    for level, expected_rms in tests.items():
        assert pytest.approx(expected_rms, rel=1e-2) == \
            calibration.get_sf(1e3, level)


def test_flat_calibration_from_spl():
    calibration = FlatCalibration.from_spl(spl=80, vrms=0.1)
    tests = {
         60: 0.01,
         80: 0.10,
        100: 1.00,
    }
    for level, expected_rms in tests.items():
        assert pytest.approx(expected_rms, abs=1e-2) == \
            calibration.get_sf(1e3, level)


def test_interp_calibration_from_spl_speaker():
    frequency =    np.array([500, 1000, 2000, 4000, 8000, 16000])
    measured_SPL = np.array([ 80,   90,  100,  100,   90,    80])
    calibration = InterpCalibration(frequency=frequency,
                                    sensitivity=measured_SPL)
    tests = {
          500: 3.16,
         1000: 1.00,
         2000: 0.32,
         4000: 0.32,
         8000: 1.00,
        16000: 3.16,
    }
    print(calibration.sensitivity)
    for frequency, expected_rms in tests.items():
        assert pytest.approx(expected_rms, abs=1e-2) == \
            calibration.get_sf(frequency, 90)


def test_interp_calibration_from_spl_mic():
    frequency =     np.array([500, 1000, 2000, 4000, 8000, 16000])
    measured_vrms = np.array([  3,   1,   0.3,  0.3,    1,     3])

    calibration = InterpCalibration.from_spl(frequency, spl=90,
                                             vrms=measured_vrms)

    tests = {
          500:  80.46,
         1000:  90.00,
         2000: 100.46,
         4000: 100.46,
         8000:  90.00,
        16000:  80.46,
    }
    for frequency, expected_spl in tests.items():
        assert pytest.approx(expected_spl, abs=1e-2) == \
            calibration.get_spl(frequency, 1)


def test_point_calibration(point_calibration, relative_levels):
    for frequency, level in relative_levels.items():
        expected = 10**(-level/20)
        assert expected == point_calibration.get_sf(frequency, 0)


def test_nd_point_calibration(point_calibration, relative_levels):
    frequencies = np.array(list(relative_levels.keys()))
    levels = np.array(list(relative_levels.values()))
    expected_rms = 10**(-levels/20)

    frequencies = point_calibration.frequency[:, np.newaxis]
    rms = point_calibration.get_sf(frequencies, 0)
    assert rms.shape == frequencies.shape
    np.testing.assert_array_equal(rms.ravel(), expected_rms)

    frequencies.shape = 3, 3
    rms = point_calibration.get_sf(frequencies, 0)
    assert rms.shape == frequencies.shape
    np.testing.assert_array_equal(rms.ravel(), expected_rms)
