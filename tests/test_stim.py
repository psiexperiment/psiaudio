import pytest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import signal

from psiaudio import calibration, stim, util

from conftest import assert_chunked_generation


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


@pytest.fixture(scope='module', params=[0, 1])
def silence_fill_value(request):
    return request.param


def test_silence_factory(silence_fill_value):
    silence = stim.SilenceFactory(fill_value=silence_fill_value)
    waveform = silence.next(100)
    expected = np.full(100, silence_fill_value)
    assert_array_equal(waveform, expected)

    silence.reset()
    waveform = silence.next(100)
    assert_array_equal(waveform, expected)


def test_cos2envelope_shape():
    fs = 100e3
    offset = 0
    samples = 400000
    start_time = 0
    rise_time = 1.0
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


@pytest.fixture(scope='module', params=[0.1e-3, 0.2e-3, 0.3e-3])
def env_start_time(request):
    return request.param


def test_cos2envelope_partial_generation(fs, env_start_time):
    duration = 10e-3
    tone_duration = 5e-3
    rise_time = 0.5e-3
    samples = int(round(duration*fs))

    # Make sure that the envelope is identical even if we delay the start
    y0 = stim.cos2envelope(fs, duration=tone_duration, rise_time=rise_time,
                           samples=samples)

    y1 = stim.cos2envelope(fs, offset=0, samples=samples,
                           start_time=env_start_time, rise_time=rise_time,
                           duration=tone_duration)
    n = int(round(env_start_time * fs))
    np.testing.assert_allclose(y0[:-n], y1[n:])

    # Now, test piecemeal generation
    partition_size = 0.1e-3
    partition_samples = round(partition_size * fs)
    n_partitions = int(np.ceil(samples / partition_samples))

    env = stim.Cos2EnvelopeFactory(fs, rise_time=rise_time,
                                    duration=tone_duration,
                                    input_factory=stim.SilenceFactory(fill_value=1),
                                    start_time=env_start_time)

    # Need to make sure y1 is the same length as y0 for funny sampling rates.
    y1 = [env.next(partition_samples) for i in range(n_partitions)]
    y1 = np.concatenate(y1)[:len(y0)]

    n = int(round(env_start_time * fs))
    print(n, y0.shape, y1.shape)
    if n > 0:
        np.testing.assert_allclose(y0[:-n], y1[n:])
    else:
        np.testing.assert_allclose(y0, y1)


@pytest.fixture(scope='module', params=['cosine-squared', 'blackman'])
def env_window(request):
    return request.param


@pytest.fixture(scope='module', params=[5e-3, 0.1, 1, 10])
def env_duration(request):
    return request.param


@pytest.fixture(scope='module', params=[None, 0.25e-3, 2.5])
def env_rise_time(request):
    return request.param


def test_envelope(fs, env_window, env_duration, env_rise_time, chunksize,
                  n_chunks):
    if (env_rise_time is not None) and (env_duration < (env_rise_time * 2)):
        mesg = 'Rise time longer than envelope duration'
        with pytest.raises(ValueError):
            actual = stim.envelope(env_window, fs, duration=env_duration,
                                   rise_time=env_rise_time)
        return

    actual = stim.envelope(env_window, fs, duration=env_duration,
                           rise_time=env_rise_time)

    if env_rise_time is None:
        n_window = (len(actual) // 2) * 2
    else:
        n_window = int(round(env_rise_time * fs)) * 2
    n_steady_state = len(actual) - n_window

    if env_window == 'cosine-squared':
        # The scipy window function calculates the window points at the bin
        # centers, whereas my approach is to calculate the window points at the
        # left edge of the bin.
        expected = signal.windows.cosine(n_window) ** 2
        if n_steady_state != 0:
            expected = np.concatenate((
                expected[:n_window//2],
                np.ones(n_steady_state),
                expected[n_window//2:],
            ), axis=-1)
        assert_array_almost_equal(actual, expected, 1)
    else:
        expected = getattr(signal.windows, env_window)(n_window)
        if n_steady_state != 0:
            expected = np.concatenate((
                expected[:n_window//2],
                np.ones(n_steady_state),
                expected[n_window//2:],
            ), axis=-1)
        assert_array_equal(actual, expected)

    kwargs = {
        'envelope': env_window,
        'fs': fs,
        'duration': env_duration,
        'rise_time': env_rise_time,
        'input_factory': stim.SilenceFactory(fill_value=1),
    }
    assert_chunked_generation(stim.EnvelopeFactory, kwargs, chunksize,
                              n_chunks)


@pytest.fixture(scope='module', params=[0, 0.2, 1.0])
def square_wave_duty_cycle(request):
    return request.param


def test_square_wave(fs, square_wave_duty_cycle):
    level = 5
    frequency = 10
    samples = int(round(fs / frequency))

    factory = stim.SquareWaveFactory(fs=fs, level=level, frequency=frequency,
                                     duty_cycle=square_wave_duty_cycle)
    waveform = factory.next(samples)
    if square_wave_duty_cycle == 0:
        assert set(waveform) == {0}
    elif square_wave_duty_cycle == 1:
        assert set(waveform) == {level}
    else:
        assert set(waveform) == {0, level}
    assert waveform.mean() == \
        pytest.approx(level * square_wave_duty_cycle, abs=2e-4)


################################################################################
# SAM envelope
################################################################################
@pytest.fixture(scope='module', params=[0, 0.25, 0.5, 1.0])
def mod_envelope_depth(request):
    return request.param


@pytest.fixture(scope='module', params=[5, 50, 500])
def mod_envelope_fm(request):
    return request.param


def test_sam_envelope(mod_envelope_depth, mod_envelope_fm):
    offset = 0
    samples = 400000
    fs = 100000
    delay = 1

    result = stim.sam_envelope(offset, samples, fs, mod_envelope_depth,
                               mod_envelope_fm, delay, equalize=True)
    assert util.rms(result) == pytest.approx(1)


################################################################################
# SAM tone
################################################################################
@pytest.fixture(scope='module', params=[4e3, 5.6e3, 8e3])
def mod_fc(request):
    return request.param


def test_sam_tone(fs, stim_level, mod_fc, mod_envelope_fm, mod_envelope_depth,
                  chunksize, n_chunks):

    cal = calibration.FlatCalibration.from_spl(94)
    kwargs = dict(fs=fs, fc=mod_fc, fm=mod_envelope_fm,
                  depth=mod_envelope_depth, level=stim_level,
                  calibration=cal, eq_power=False)

    if round(fs) != fs:
        # This causes some annoying indexing issues.
        pytest.skip()

    factory = stim.SAMToneFactory(**kwargs)
    s = factory.next(samples=int(fs * 30))
    s_psd = util.psd_df(s, fs, waveform_averages=30)
    s_spl = pd.Series(
        cal.get_db(s_psd.index, s_psd.values),
        index=s_psd.index
    )
    assert pytest.approx(s_spl.loc[mod_fc], abs=0.5) == (stim_level - 6)
    assert pytest.approx(s_spl.loc[mod_fc-mod_envelope_fm], abs=0.5) == (stim_level - 12)
    assert pytest.approx(s_spl.loc[mod_fc+mod_envelope_fm], abs=0.5) == (stim_level - 12)
    assert np.sum(s_spl > (stim_level-20)) == 3


def test_sam_tone_starship(fs, stim_level, mod_fc, mod_envelope_fm,
                           mod_envelope_depth, chunksize, n_chunks):

    # This ensures that the SAM tone generation properly compensates for
    # nonlinearities in the acoustic system. If the speaker output is lower at
    # a certain frequency, the SAM tone harmonic needs to be boosted to
    # compensate.
    cal = calibration.PointCalibration.from_spl(
        [mod_fc - mod_envelope_fm, mod_fc, mod_fc + mod_envelope_fm],
        [-6, 0, 6]
    )
    kwargs = dict(fs=fs, fc=mod_fc, fm=mod_envelope_fm,
                  depth=mod_envelope_depth, level=stim_level,
                  calibration=cal, eq_power=False)

    if round(fs) != fs:
        # This causes some annoying indexing issues.
        pytest.skip()

    factory = stim.SAMToneFactory(**kwargs)
    s = factory.next(samples=int(fs * 30))
    s_psd = util.db(util.psd_df(s, fs, waveform_averages=30))
    assert pytest.approx(s_psd.loc[mod_fc], abs=0.5) == (stim_level - 6)
    assert pytest.approx(s_psd.loc[mod_fc-mod_envelope_fm], abs=0.5) == (stim_level - 12 + 6)
    assert pytest.approx(s_psd.loc[mod_fc+mod_envelope_fm], abs=0.5) == (stim_level - 12 - 6)
    assert np.sum(s_psd > (stim_level-20)) == 3


def test_sam_tone_factory(fs, stim_level, mod_fc, mod_envelope_fm,
                          mod_envelope_depth, stim_calibration, chunksize,
                          n_chunks):
    kwargs = dict(fs=fs, fc=mod_fc, fm=mod_envelope_fm,
                  depth=mod_envelope_depth, level=stim_level,
                  calibration=stim_calibration)
    assert_chunked_generation(stim.SAMToneFactory, kwargs, chunksize,
                              n_chunks, exact=True)


################################################################################
# Square wave envelope
################################################################################
@pytest.fixture(scope='module', params=[0.1, 0.5, 0.9])
def square_wave_duty_cycle(request):
    return request.param


def test_square_wave_envelope(fs, mod_envelope_depth, mod_envelope_fm,
                              square_wave_duty_cycle, offset=0):
    result = stim.square_wave(fs, offset, int(fs * 2), mod_envelope_depth,
                              mod_envelope_fm, square_wave_duty_cycle)
    expected_average = square_wave_duty_cycle + \
        (1 - mod_envelope_depth) * (1 - square_wave_duty_cycle)

    # For sampling rates that are not a clean multiple of the modulation
    # frequency, we will not get *exactly* the expected average.
    if fs == 195312.5:
        expected_average = pytest.approx(expected_average, abs=0.005)
        expected_fm = pytest.approx(mod_envelope_fm, abs=0.005)
        expected_duty_cycle = pytest.approx(square_wave_duty_cycle, abs=0.005)
        # allow for 1-sample errors due to uneven sampling rates
        expected_start_jitter = pytest.approx(0, abs=1)
    else:
        expected_fm = mod_envelope_fm
        expected_duty_cycle = square_wave_duty_cycle
        expected_start_jitter = pytest.approx(0, abs=0)

    assert np.mean(result) == expected_average
    assert np.min(result) == (1 - mod_envelope_depth)
    assert np.max(result) == 1

    if mod_envelope_depth != 0:
        plateau = (result == 1).astype('i')
        starts = np.flatnonzero(np.diff(plateau) == 1)
        ends = np.flatnonzero(np.diff(plateau) == -1)

        starts_iti = np.diff(starts)
        ends_iti = np.diff(ends)

        assert (starts_iti - starts_iti[0]) == expected_start_jitter
        assert (ends_iti - ends_iti[0]) == expected_start_jitter

        assert (fs / starts_iti.mean()) == expected_fm
        assert (fs / ends_iti.mean()) == expected_fm

        # Discard first end. Since "offset" is 0, we don't have the ability to
        # detect the first start value.
        ends = ends[1:]

        plateau_samples = (ends - starts)
        assert plateau_samples == pytest.approx(plateau_samples[0], abs=0)
        duty_cycle = plateau_samples[0] / (fs / mod_envelope_fm)
        assert duty_cycle == expected_duty_cycle
