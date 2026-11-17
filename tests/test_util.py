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


def test_phase_tone():
    '''
    A tone built with a known phase offset reports that offset at its own bin.
    '''
    fs = 10e3
    for expected in (0.0, 1.0, -2.0, 3.0):
        tone = make_tone(fs, 1e3, 1, phase=expected)
        p = util.phase(tone, fs, unwrap=False, detrend=None)
        freqs = util.psd_freq(tone, fs)
        k = int(np.argmin(np.abs(freqs - 1e3)))
        assert p[k] == pytest.approx(expected, abs=1e-6)


def test_phase_2d():
    '''
    Each row of a 2D input keeps its own phase; the time axis is the last one.
    '''
    fs = 10e3
    tones = np.vstack((make_tone(fs, 1e3, 1, phase=0.5),
                       make_tone(fs, 1e3, 1, phase=-1.5)))
    p = util.phase(tones, fs, unwrap=False, detrend=None)
    freqs = util.psd_freq(tones, fs)
    k = int(np.argmin(np.abs(freqs - 1e3)))
    assert p.shape == (2, len(freqs))
    assert p[0, k] == pytest.approx(0.5, abs=1e-6)
    assert p[1, k] == pytest.approx(-1.5, abs=1e-6)


def test_phase_detrend_default():
    '''
    The default call must detrend linearly, matching ``csd`` and ``psd``.

    Regression test: ``phase`` used to forward ``waveform_averages`` into
    ``csd``'s ``detrend`` slot, so the default silently became
    ``detrend=None`` and disagreed with ``psd`` on the same input.
    '''
    fs = 10e3
    n = int(fs)
    trending = 0.01 * make_tone(fs, 1e3, 1) + np.linspace(0, 10, n)

    default = util.phase(trending, fs)
    assert default == pytest.approx(util.phase(trending, fs, detrend='linear'))
    assert default != pytest.approx(util.phase(trending, fs, detrend=None))


def test_phase_averages():
    '''
    ``waveform_averages`` splits the time axis and averages, as ``psd`` does.

    Regression test: any integer value used to land in ``csd``'s ``detrend``
    argument and raise.
    '''
    fs = 10e3
    tone = make_tone(fs, 1e3, 1, phase=1.0)

    for averages in (None, 1, 2, 4):
        p = util.phase(tone, fs, waveform_averages=averages, unwrap=False,
                       detrend=None)
        # Same segmenting as psd, so the two stay directly comparable.
        assert p.shape == util.psd(tone, fs, waveform_averages=averages).shape

        n = tone.shape[-1] // (averages if averages else 1)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        assert len(p) == len(freqs)
        k = int(np.argmin(np.abs(freqs - 1e3)))
        assert p[k] == pytest.approx(1.0, abs=1e-6)


def test_phase_averages_are_coherent():
    '''
    Averaging is done on the complex spectra, not on the angles.

    Two segments sitting either side of the +/-pi wrap average to pi. Averaging
    the angles themselves would give 0, which is the opposite direction.
    '''
    fs = 10e3
    segments = [make_tone(fs, 100.0, 0.5, phase=3.0),
                make_tone(fs, 100.0, 0.5, phase=-3.0)]
    signal = np.concatenate(segments)

    p = util.phase(signal, fs, waveform_averages=2, unwrap=False,
                   detrend=None)
    freqs = np.fft.rfftfreq(len(segments[0]), 1 / fs)
    k = int(np.argmin(np.abs(freqs - 100.0)))

    # Coherent average of exp(3j) and exp(-3j) is real and negative, so the
    # angle is +/-pi.
    assert abs(p[k]) == pytest.approx(np.pi, abs=1e-6)

    individual = [util.phase(s, fs, unwrap=False, detrend=None)[k]
                  for s in segments]
    assert np.mean(individual) == pytest.approx(0, abs=1e-6)


def test_phase_df_averages():
    '''
    ``phase_df`` indexes by the frequencies of the averaged segments.

    Regression test: this call used to raise, because ``_as_df`` forwards
    ``waveform_averages`` to ``phase`` by keyword.
    '''
    fs = 10e3
    tone = make_tone(fs, 1e3, 1, phase=1.0)

    p = util.phase_df(tone, fs, waveform_averages=4, unwrap=False,
                      detrend=None)
    assert p.index.name == 'frequency'
    assert len(p) == len(np.fft.rfftfreq(tone.shape[-1] // 4, 1 / fs))
    assert p.loc[1e3] == pytest.approx(1.0, abs=1e-6)


def test_psd_bootstrap_loop_phase_near_pi():
    # The response phase is pi, so noise scatters the bootstrapped phases on
    # both sides of the +/- pi wrap. An arithmetic mean of those phases would
    # land near 0 with a spread of about pi.
    fs = 1000
    n_samples = 1000
    n_trials = 200
    n_draw = 100
    f0 = 40

    rng = np.random.default_rng(1)
    t = np.arange(n_samples) / fs
    tone = np.cos(2 * np.pi * f0 * t + np.pi)

    def make_trials():
        noise = rng.normal(scale=2, size=(n_trials, n_samples))
        return tone + noise

    result = util.psd_bootstrap_loop(
        [make_trials(), make_trials()], fs, n_draw=n_draw, n_bootstrap=200,
        rng=np.random.RandomState(1), callback=None)
    row = result.loc[f0]

    # Circular distance from pi.
    assert np.abs(np.angle(np.exp(1j * (row['phase'] - np.pi)))) < 0.05
    assert 0 < row['phase_std'] < 0.2
    assert row['psd_norm'] > 10

    # Two arrays of n_trials, with n_draw / 2 drawn from each.
    n = n_draw // 2
    plv_nf_power = 2 * (1 / n_trials + 1 / n - 1 / (n * n_trials)) / 2 ** 2
    expected_plv_nf = np.sqrt(np.pi / 4 * plv_nf_power)
    np.testing.assert_allclose(result['plv_nf'], expected_plv_nf)
    # The closed-form noise floor should match the PLV observed at
    # frequencies with no response.
    noise_plv = result['plv'].drop(f0).loc[1:]
    assert noise_plv.mean() == pytest.approx(expected_plv_nf, rel=0.05)


def make_bootstrap_trials(rng, n_trials, amp_mean, amp_std, phase_mean,
                          phase_std, noise_std, fs, n_samples, f0):
    '''
    Generate trials of a tone whose amplitude and phase vary from trial to
    trial.

    Amplitude is gamma-distributed so it stays positive (a negative amplitude
    would flip that trial's phase by pi). Phase is normally distributed (i.e.,
    wrapped normal).
    '''
    if amp_std:
        shape = (amp_mean / amp_std) ** 2
        scale = amp_std ** 2 / amp_mean
        amplitude = rng.gamma(shape, scale, size=(n_trials, 1))
    else:
        amplitude = np.full((n_trials, 1), amp_mean)
    phase = rng.normal(phase_mean, phase_std, size=(n_trials, 1))
    t = np.arange(n_samples) / fs
    trials = amplitude * np.cos(2 * np.pi * f0 * t + phase)
    if noise_std:
        trials += rng.normal(scale=noise_std, size=trials.shape)
    return trials


def expected_bootstrap_stats(amp_mean, amp_std, phase_mean, phase_std,
                             noise_std, n_samples, n_draw, n_pool):
    '''
    Predict the output of psd_bootstrap_loop at the tone frequency.

    The CSD of A * cos(wt + phi) is A / sqrt(2) * exp(1j * phi). Rotating so
    the mean phase is 0, the per-trial CSD has a mean of `r` and variances of
    `v_par` and `v_perp` along and across the mean vector. `psd_std` and
    `phase_std` describe the spread of the mean of `n_draw` trials, so they
    scale as 1 / sqrt(n_draw). The pool of `n_pool` trials is itself a random
    sample, so its mean differs from the population's by the `*_pool_se`
    standard errors, which the bootstrap cannot remove.
    '''
    rho = np.exp(-phase_std ** 2 / 2)    # E[cos(dphi)]
    rho2 = np.exp(-2 * phase_std ** 2)   # E[cos(2 * dphi)]
    a2 = amp_mean ** 2 + amp_std ** 2    # E[A ** 2]
    # White noise with variance q ** 2 adds q ** 2 / n_samples to each
    # component of the CSD.
    v_noise = noise_std ** 2 / n_samples

    r = amp_mean * rho / np.sqrt(2)
    v_par = (a2 * (1 + rho2) / 2 - amp_mean ** 2 * rho ** 2) / 2 + v_noise
    v_perp = a2 * (1 - rho2) / 4 + v_noise
    db_per_neper = 20 / np.log(10)
    sample_weight = 1 / n_draw + 1 / n_pool

    return {
        # Averaging in dB adds a small second-order bias.
        'psd': util.db(r) + db_per_neper * (v_perp - v_par) / (2 * r ** 2) \
            * sample_weight,
        'phase': phase_mean,
        'psd_std': db_per_neper * np.sqrt(v_par / n_draw) / r,
        'phase_std': np.sqrt(v_perp / n_draw) / r,
        # Only valid without additive noise, which also perturbs the phase.
        # The PLV of the population is exp(-phase_std ** 2 / 2); finite
        # samples bias it upward.
        'plv': rho + (1 - rho2) / (4 * rho) * sample_weight,
        'psd_pool_se': db_per_neper * np.sqrt(v_par / n_pool) / r,
        'phase_pool_se': np.sqrt(v_perp / n_pool) / r,
        'plv_pool_se': np.sqrt(((1 + rho2) / 2 - rho ** 2) / n_pool),
    }


@pytest.mark.parametrize('n_arrays', [1, 2])
@pytest.mark.parametrize(
    'amp_mean, amp_std, phase_mean, phase_std, noise_std', [
        (1.0, 0.3, 0.5, 0.0, 0.0),      # amplitude jitter only
        (1.0, 0.0, 1.0, 0.3, 0.0),      # phase jitter only
        (1.0, 0.3, np.pi, 0.5, 0.0),    # phase straddles +/- pi
        (2.0, 1.0, -2.0, 1.0, 0.0),     # broad amplitude and phase spread
        (1.0, 0.3, 1.0, 0.3, 2.0),      # jitter plus additive noise
    ])
def test_psd_bootstrap_loop_recovers_trial_stats(
        amp_mean, amp_std, phase_mean, phase_std, noise_std, n_arrays):
    fs = 1000
    n_samples = 500
    f0 = 100
    n_trials = 500
    n_draw = 100

    rng = np.random.default_rng(0)
    arrays = [
        make_bootstrap_trials(rng, n_trials, amp_mean, amp_std, phase_mean,
                              phase_std, noise_std, fs, n_samples, f0)
        for _ in range(n_arrays)
    ]
    result = util.psd_bootstrap_loop(
        arrays, fs, n_draw=n_draw, n_bootstrap=500,
        rng=np.random.RandomState(0), callback=None).loc[f0]
    e = expected_bootstrap_stats(amp_mean, amp_std, phase_mean, phase_std,
                                 noise_std, n_samples, n_draw,
                                 n_trials * n_arrays)

    # Means should fall within the sampling error of the trial pool.
    assert result['psd'] == pytest.approx(e['psd'], abs=4 * e['psd_pool_se'])
    phase_error = np.angle(np.exp(1j * (result['phase'] - e['phase'])))
    # The linear detrend in csd shifts phase by up to ~1e-4 rad.
    assert abs(phase_error) <= 4 * e['phase_pool_se'] + 1e-3
    if noise_std == 0:
        assert result['plv'] == pytest.approx(
            e['plv'], abs=4 * e['plv_pool_se'] + 1e-6)

    # Spreads across bootstrap cycles should match the predicted standard
    # error of the mean of n_draw trials.
    assert result['psd_std'] == pytest.approx(e['psd_std'], rel=0.2)
    if e['phase_std'] == 0:
        assert result['phase_std'] < 1e-3
    else:
        assert result['phase_std'] == pytest.approx(e['phase_std'], rel=0.2)


@pytest.mark.parametrize('n_trials, n_arrays', [(100, 1), (200, 1), (100, 2)])
def test_psd_bootstrap_loop_noise_floor(n_trials, n_arrays):
    # With noise only, psd should match psd_nf. The legacy 'draw' mode misses
    # the residual mean of the fixed trial pool and reads high by
    # 10 * log10(1 + n_draw / n_pool).
    fs = 1000
    n_draw = 100
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(n_trials, 500)) for _ in range(n_arrays)]

    def mean_psd_norm(noise_floor):
        result = util.psd_bootstrap_loop(
            arrays, fs, n_draw=n_draw, n_bootstrap=200,
            rng=np.random.RandomState(0), callback=None,
            noise_floor=noise_floor)
        # Skip DC, which the linear detrend removes.
        return result['psd_norm'].loc[1:].mean()

    assert mean_psd_norm('pool') == pytest.approx(0, abs=0.15)
    legacy_bias = 10 * np.log10(1 + n_draw / (n_trials * n_arrays))
    assert mean_psd_norm('draw') == pytest.approx(legacy_bias, abs=0.15)


def expected_db_mean_bias(n_draw, n_pool):
    '''
    Expected dB of the mean power minus the mean of the dB power, across
    bootstrap cycles of `psd_bootstrap_loop`, at a frequency with only noise

    Each bootstrapped mean is ``mu + e``, where `mu` is the mean of the fixed
    pool of trials and `e` is the deviation of that cycle's draw from it.
    Both are circular complex Gaussian, with variances proportional to
    ``1 / n_pool`` and ``1 / n_draw``. `mu` is the same on every cycle, so
    across cycles the power is non-central chi-squared and, with ``x =
    |mu|**2 / var(e)``,

        E[ln |mu + e|**2] = ln |mu|**2 + E1(x)
        ln E[|mu + e|**2] = ln(|mu|**2 + var(e))

    where E1 is the exponential integral. Their difference,
    ``ln(1 + 1 / x) - E1(x)``, is averaged over frequencies, across which `x`
    is exponential with mean ``n_draw / n_pool``. With no pool term (`x` -> 0)
    this is Euler's gamma, the bias for exponential power.
    '''
    from scipy import integrate, special
    r = n_draw / n_pool

    def integrand(x):
        return (np.log1p(1 / x) - special.exp1(x)) * np.exp(-x / r) / r

    bias, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    return 10 / np.log(10) * bias


def test_psd_bootstrap_loop_power_noise_only():
    # With noise only, compare the new columns (dB of the mean power across
    # bootstrap cycles) to the old ones (mean of the dB power) and to each
    # other.
    fs = 1000
    n_trials = 200
    n_draw = 100
    rng = np.random.default_rng(0)
    arrays = rng.normal(size=(n_trials, 500))
    result = util.psd_bootstrap_loop(
        arrays, fs, n_draw=n_draw, n_bootstrap=500,
        rng=np.random.RandomState(0), callback=None)
    # Skip DC, which the linear detrend removes, and Nyquist, where the CSD is
    # real rather than complex and so has a different distribution. That
    # leaves 249 frequencies, which are independent for white noise. Each
    # check below averages across them, and its tolerance is 4 standard
    # errors of that average (estimated by repeating the test across seeds).
    result = result.loc[1:fs / 2 - 1]

    # `psd_nf` re-randomizes the phases of the whole pool on every cycle, so
    # the power of the bootstrapped mean is exponential across cycles. The
    # mean of the dB of exponential power is 10 * euler_gamma / ln(10) = 2.51
    # dB below the dB of the mean power. The standard error of the average
    # across frequencies is 0.01 dB.
    gamma_bias = 10 * np.euler_gamma / np.log(10)
    nf_bias = (result['psd_nf_power'] - result['psd_nf']).mean()
    assert nf_bias == pytest.approx(gamma_bias, abs=0.04)

    # `psd` draws from the fixed pool, whose mean is the same on every cycle,
    # so the power is less spread than exponential and the bias is smaller
    # (2.32 dB when n_draw / n_pool = 0.5). See `expected_db_mean_bias`. The
    # standard error of the average across frequencies is 0.02 dB.
    psd_bias = (result['psd_power'] - result['psd']).mean()
    assert psd_bias == pytest.approx(
        expected_db_mean_bias(n_draw, n_trials), abs=0.08)
    # The two biases differ by more than the tolerance, so the check above
    # can tell them apart.
    assert abs(psd_bias - gamma_bias) > 0.1

    # Where there is no response, subtracting the noise floor in power
    # should leave nothing. At each frequency, `psd_power` measures the
    # power of this particular pool, including its residual mean `mu`, while
    # `psd_nf_power` measures the power expected of a pool. These agree only
    # on average across frequencies. `|mu| ** 2` is exponential, and it makes
    # up 1 / n_pool of the 1 / n_draw + 1 / n_pool total (a third here), so
    # the excess at each frequency has a standard deviation of about 1 / 3 of
    # the noise floor. Across 249 frequencies, the standard error is 0.021.
    power = util.dbi(result['psd_power']) ** 2
    nf_power = util.dbi(result['psd_nf_power']) ** 2
    excess = (power - nf_power) / nf_power.mean()
    assert excess.mean() == pytest.approx(0, abs=0.1)


def test_psd_bootstrap_loop_power_recovers_tone():
    # Subtracting the noise floor in power should recover the power of a tone
    # in noise. The CSD of A * cos(wt) is A / sqrt(2), so the tone power is
    # A ** 2 / 2. The phase-randomized trials keep the magnitude of the tone,
    # so the noise floor includes 1 / n_draw + 1 / n_pool of the tone power
    # and the excess reads low by that share (3% here).
    fs = 1000
    n_samples = 500
    f0 = 100
    n_trials = 100
    n_draw = 50
    # SNR is about 8 dB at f0 (the noise floor is 0.3 of the tone power), so
    # leaving the noise floor in, or subtracting it in the wrong units, would
    # be off by tens of percent.
    amplitude = 0.03
    excess = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        arrays = make_bootstrap_trials(rng, n_trials, amplitude, 0, 0, 0, 1,
                                       fs, n_samples, f0)
        result = util.psd_bootstrap_loop(
            arrays, fs, n_draw=n_draw, n_bootstrap=100,
            rng=np.random.RandomState(seed), callback=None).loc[f0]
        excess.append(util.dbi(result['psd_power']) ** 2
                      - util.dbi(result['psd_nf_power']) ** 2)

    expected = (amplitude / np.sqrt(2)) ** 2         * (1 - 1 / n_draw - 1 / n_trials)
    # The noise in each pool of trials shifts its mean at f0, and the
    # bootstrap cannot remove that shift. Across pools, the excess has a
    # standard deviation of about 40% of the tone power, so a single pool
    # cannot test for bias. Averaging 30 pools brings the standard error down
    # to about 7%. The tolerance of 3 standard errors (about 22%) is computed
    # from the observed spread rather than fixed, so it tracks the noise if
    # the parameters above change. That is enough to detect gross errors,
    # such as leaving the noise floor in (+30%) or converting dB to amplitude
    # rather than power (off by a factor of about 50), but not the 3%
    # correction above.
    se = np.std(excess) / np.sqrt(len(excess))
    assert np.mean(excess) == pytest.approx(expected, abs=3 * se)


def test_psd_bootstrap_loop_noise_floor_invalid():
    x = np.random.default_rng(0).normal(size=(10, 100))
    with pytest.raises(ValueError, match='noise_floor'):
        util.psd_bootstrap_loop(x, 1000, n_draw=10, callback=None,
                                noise_floor='trial')
