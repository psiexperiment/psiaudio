import pytest

import numpy as np
import pandas as pd

from psiaudio import util
from psiaudio.calibration import CalibrationNFError, CalibrationTHDError


@pytest.fixture(scope='module', params=[-20, 20, 60, 100])
def spectrum_level(request):
    return request.param


@pytest.fixture(scope='module', params=[(2e3, 4e3), (1e3, 8e3), (0.5e3, 12e3)])
def noise_band(request):
    return request.param


@pytest.fixture
def noise_csd(fs, spectrum_level, noise_band):
    frequency = np.fft.rfftfreq(int(fs), 1/fs)
    phase = np.random.uniform(-np.pi, np.pi, size=len(frequency))
    flb, fub = noise_band
    noise_floor = -200

    power = np.full_like(frequency, util.dbtopa(noise_floor))
    mask = (frequency >= flb) & (frequency < fub)
    power[mask] = util.dbtopa(spectrum_level)
    return power * np.exp(1j * phase)


@pytest.fixture
def noise_signal(noise_csd):
    return util.csd_to_signal(noise_csd)


def make_tone(fs, f0, duration, phase=0):
    n = int(duration*fs)
    t = np.arange(n, dtype=np.double)/fs
    y = np.cos(2*np.pi*f0*t + phase)
    return y


def test_fft_tone():
    fs = 100e3
    frequency = 1e3
    duration = 1
    tone = make_tone(fs, frequency, duration)
    psd = util.psd_df(tone, fs)
    assert psd.max() == pytest.approx(1/np.sqrt(2))
    assert psd.idxmax() == 1000

    fs = 100e3
    frequency = 4e3
    duration = 1
    tone = 2 * make_tone(fs, frequency, duration)
    psd = util.psd_df(tone, fs)
    assert psd.max() == pytest.approx(2/np.sqrt(2))
    assert psd.idxmax() == 4000


def test_fft_tone_averages(fs):
    frequency = 1e3
    duration = np.round(fs * 4) / fs
    tone = make_tone(fs, frequency, duration)
    psd = util.psd_df(tone, fs, waveform_averages=4, trim_samples=True, window='flattop')
    # The absolute tolerance is required for the case where sampling rate is
    # 195312.5 Hz (i.e., on a TDT RZ6 system).
    assert psd.max() == pytest.approx(1 / np.sqrt(2), abs=1e-4)
    assert psd.idxmax() == pytest.approx(frequency, abs=1e-2)


def test_band_to_spectrum_level_roundtrip(spectrum_level, noise_band):
    n = noise_band[1] - noise_band[0]
    band_level = util.spectrum_to_band_level(spectrum_level, n)
    spectrum_level_rt = util.band_to_spectrum_level(band_level, n)
    assert spectrum_level == pytest.approx(spectrum_level_rt)


def test_signal_to_fft_roundtrip():
    signal = np.random.uniform(size=100000)
    signal_rt = util.csd_to_signal(util.csd(signal, detrend=None))
    np.testing.assert_array_almost_equal(signal, signal_rt)


def test_fft_to_signal_roundtrip():
    csd = np.random.random(size=10000) + np.random.random(size=10000) * 1j
    csd_rt = util.csd(util.csd_to_signal(csd), detrend=None)
    # Throw out first and last point due to boundary issues
    np.testing.assert_array_almost_equal(csd[1:-1], csd_rt[1:-1])


def test_rms(noise_csd, noise_signal, spectrum_level, noise_band):
    rms_fft = util.patodb(util.rms_rfft(noise_csd))
    rms_signal = util.patodb(util.rms(noise_signal))
    assert rms_fft == pytest.approx(rms_signal, 2)

    n = noise_band[1] - noise_band[0]
    expected_db = util.spectrum_to_band_level(spectrum_level, n)
    assert expected_db == pytest.approx(rms_fft)
    assert expected_db == pytest.approx(rms_signal)


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


def test_diff_matrix():
    matrix = util.diff_matrix(8, 'all')
    assert np.all(np.diag(matrix) == (1 - 1/8))
    assert np.all(matrix.sum(axis=1) == 0)

    matrix = util.diff_matrix(8, 'raw')
    assert np.all(np.diag(matrix) == 1)
    assert np.all(matrix.sum(axis=1) == 1)

    matrix = util.diff_matrix(8, [0, 1])
    assert np.all(np.abs(matrix[:, :1]) == 0.5)
    assert np.all(matrix.sum(axis=1) == 0)

    matrix = util.diff_matrix(8, 0)
    assert np.all(np.abs(matrix[:, :0]) == 1)
    assert np.all(matrix.sum(axis=1) == 0)

    matrix = util.diff_matrix(8, ['A', 'B'], labels='ABCDEFGH')
    assert np.all(np.abs(matrix[:, :1]) == 0.5)
    assert np.all(matrix.sum(axis=1) == 0)

    matrix = util.diff_matrix(8, 'A', labels='ABCDEFGH')
    assert np.all(np.abs(matrix[:, :0]) == 1)
    assert np.all(matrix.sum(axis=1) == 0)


def test_apply_diff_matrix():
    m = util.diff_matrix(4, [1, 2, 3])

    n_samples = 1000
    t = np.arange(n_samples) / n_samples
    s0 = np.sin(2 * np.pi * 10 * t)
    s1 = np.zeros_like(s0)
    s2 = np.zeros_like(s0)
    s3 = np.zeros_like(s0)
    noise = np.random.uniform(size=1000)

    data_clean = np.vstack((s0, s1, s2, s3))
    data_noisy = data_clean + noise
    data_cleaned = m @ data_noisy
    np.testing.assert_array_almost_equal(data_clean, data_cleaned)


def test_rms_rfft(spectrum_level, noise_band):
    psd = np.zeros(10000)
    lb, ub = noise_band
    lb = int(lb)
    ub = int(ub)
    psd[lb:ub] = util.dbi(spectrum_level)

    n = noise_band[1] - noise_band[0]
    expected_level = util.spectrum_to_band_level(spectrum_level, n)

    actual_level = util.db(util.rms_rfft(psd))
    assert expected_level == pytest.approx(actual_level, 2)


def test_psd_df():
    fs = 10e3
    tones = np.vstack((make_tone(fs, 1e3, 1), make_tone(fs, 2e3, 1)))
    psd = util.psd_df(tones, fs=fs)
    assert psd.columns.name == 'frequency'
    assert psd.index.values.tolist() == [0, 1]

    actual = psd[[1e3, 2e3]].values
    expected = np.array([[np.sqrt(2)/2, 0], [0, np.sqrt(2)/2]])
    np.testing.assert_array_almost_equal(actual, expected)

    t = np.arange(tones.shape[-1]) / fs
    columns = pd.Index(t, name='time')
    index = pd.MultiIndex.from_tuples([('A', '1kHz'), ('B', '2kHz')],
                                      names=['category', 'frequency'])
    tones = pd.DataFrame(tones, index=index, columns=columns)
    psd = util.psd_df(tones, fs=fs)
    assert psd.index.equals(tones.index)
    assert psd.columns.name == 'frequency'
    actual = psd[[1e3, 2e3]].values
    np.testing.assert_array_almost_equal(actual, expected)
