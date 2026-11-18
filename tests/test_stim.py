import pytest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import signal

from psiaudio import calibration, stim, util

from .helpers import assert_chunked_generation


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
    if mod_envelope_depth != 1.0:
        # Not currently supported
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
    if mod_envelope_depth != 1.0:
        # Not currently supported
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
    if mod_envelope_depth != 1.0:
        # Not currently supported
        pytest.skip()
    kwargs = dict(fs=fs, fc=mod_fc, fm=mod_envelope_fm,
                  depth=mod_envelope_depth, level=stim_level,
                  calibration=stim_calibration)
    assert_chunked_generation(stim.SAMToneFactory, kwargs, chunksize,
                              n_chunks, exact=True)


import matplotlib.pyplot as plt

################################################################################
# Square wave envelope
################################################################################
@pytest.fixture(scope='module', params=[0.1, 0.5, 0.9])
def square_wave_duty_cycle(request):
    return request.param


@pytest.fixture(scope='module', params=[0, 0.025, 0.5])
def square_wave_alpha(request):
    return request.param


def test_square_wave_envelope(fs, mod_envelope_depth, mod_envelope_fm,
                              square_wave_duty_cycle, square_wave_alpha,
                              offset=0):

    duty_samples = int(round(fs / mod_envelope_fm * square_wave_duty_cycle))
    w = signal.windows.tukey(duty_samples, square_wave_alpha)
    w = w * mod_envelope_depth + (1 - mod_envelope_depth)
    w_mean = np.mean(w)

    result = stim.square_wave(fs, offset, int(fs * 2), mod_envelope_depth,
                              mod_envelope_fm, square_wave_duty_cycle,
                              square_wave_alpha)
    expected_average = square_wave_duty_cycle * w_mean + \
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
        if square_wave_alpha != 0:
            expected_average = pytest.approx(expected_average, abs=1e-14)

    assert np.mean(result) == expected_average
    assert np.min(result) == (1 - mod_envelope_depth)
    assert np.max(result) == 1

    if mod_envelope_depth != 0:
        plateau = (result > (1 - mod_envelope_depth)).astype('i')
        starts = np.flatnonzero(np.diff(plateau) == 1)
        ends = np.flatnonzero(np.diff(plateau) == -1)

        starts_iti = np.diff(starts)
        ends_iti = np.diff(ends)

        assert (starts_iti - starts_iti[0]) == expected_start_jitter
        assert (ends_iti - ends_iti[0]) == expected_start_jitter

        assert (fs / starts_iti.mean()) == expected_fm
        assert (fs / ends_iti.mean()) == expected_fm

        # Discard first end if square_wave_alpha is 0 (i.e., perfectly
        # rectangular envelope). Since "offset" is 0, we don't have the ability
        # to detect the first start value.
        if square_wave_alpha == 0:
            ends = ends[1:]

        plateau_samples = (ends - starts)
        # Verify all plateaus found are the same duration
        assert plateau_samples == pytest.approx(plateau_samples[0], abs=0)

        plateau_samples = plateau_samples[0]
        if square_wave_alpha != 0:
            # Add 2 samples to correct for the fact that the Tukey window goes
            # to 0 at the ends (thus, making the "apparent" duty cycle shorter
            # than expected).
            plateau_samples += 2

        duty_cycle = plateau_samples / (fs / mod_envelope_fm)
        assert duty_cycle == expected_duty_cycle


def test_ram_efr_factory(fs, stim_level, mod_envelope_depth, mod_envelope_fm,
                         square_wave_duty_cycle, square_wave_alpha,
                         stim_calibration, chunksize, n_chunks):

    def make_factory(fs, fc, level, calibration, **kwargs):
        carrier = stim.ToneFactory(fs=fs, frequency=fc, level=level,
                                   phase=-np.pi/2, calibration=calibration)
        factory = stim.SquareWaveEnvelopeFactory(fs=fs,
                                                 calibration=calibration,
                                                 input_factory=carrier,
                                                 **kwargs)
        return factory

    kwargs = dict(fs=fs, fc=8e3, level=stim_level,
                  calibration=stim_calibration, fm=mod_envelope_fm,
                  depth=mod_envelope_depth, duty_cycle=square_wave_duty_cycle,
                  alpha=square_wave_alpha)
    assert_chunked_generation(make_factory, kwargs, chunksize, n_chunks)


################################################################################
# Envelope transform
################################################################################
def test_envelope_transform(fs, stim_calibration):
    carrier = stim.ToneFactory(fs=fs, frequency=8e3, level=94,
                                calibration=stim_calibration)
    env = stim.EnvelopeFactory(envelope='bartlett', duration=8,
                                rise_time=None,
                                fs=fs,
                                input_factory=carrier,
                                transform=lambda x: x ** 10)
    actual = env.get_samples_remaining()
    n_samples = len(actual)

    carrier = stim.ToneFactory(fs=fs, frequency=8e3, level=94,
                                calibration=stim_calibration)
    env = signal.windows.bartlett(n_samples) ** 10
    expected = carrier.next(n_samples) * env

    assert_array_equal(actual, expected)


def test_envelope_transform_factory(fs, chunksize, n_chunks, stim_calibration):
    def make_factory(**kwargs):
        carrier = stim.ToneFactory(fs=fs, frequency=8e3, level=94,
                                   calibration=stim_calibration)
        env = stim.EnvelopeFactory(envelope='bartlett', duration=8,
                                   rise_time=None,
                                   fs=fs,
                                   input_factory=carrier,
                                   transform=lambda x: x ** 10)
        return env
    assert_chunked_generation(make_factory, {}, chunksize, n_chunks)


################################################################################
# Gate
################################################################################
def test_gate_silences_after_gate_closes():
    # A block requested entirely after the gate has closed used to pass the
    # carrier straight through.
    fs = 1000
    carrier = stim.ToneFactory(fs=fs, frequency=50, level=1)
    gate = stim.GateFactory(fs=fs, start_time=0.01, duration=0.02,
                            input_factory=carrier)
    before = gate.next(10)
    during = gate.next(20)
    after = gate.next(20)
    assert_array_equal(before, 0)
    assert np.abs(during).max() > 0
    assert_array_equal(after, 0)


def test_gate_chunked_generation_matches_unchunked():
    fs = 1000
    def make_factory():
        carrier = stim.ToneFactory(fs=fs, frequency=50, level=1)
        return stim.GateFactory(fs=fs, start_time=0.015, duration=0.02,
                                input_factory=carrier)
    assert_chunked_generation(make_factory, {}, 7, 10)


################################################################################
# max_amplitude
################################################################################
def make_iir_noise(fs=25e3, **kwargs):
    return stim.BandlimitedNoiseFactory(
        fs=fs, seed=1, level=1, fl=1e3, fh=4e3, filter_rolloff=1,
        passband_attenuation=1, stopband_attenuation=80, **kwargs)


def peak(factory, samples):
    factory.reset()
    return np.abs(factory.next(samples)).max()


@pytest.mark.parametrize('make_factory', [
    lambda: make_iir_noise(),
    lambda: make_iir_noise(use_sos=True),
    lambda: make_iir_noise(polarity=-1),
    lambda: stim.BroadbandNoiseFactory(fs=25e3, level=1),
    lambda: stim.ShapedNoiseFactory(fs=25e3, level=1, seed=1,
                                    gains={0: -20, 1e3: 0, 4e3: 0, 12.5e3: -20}),
    lambda: stim.NotchFilterFactory(
        fs=25e3, notch_frequency=2e3, q=1.33,
        input_factory=stim.BroadbandNoiseFactory(fs=25e3, level=1)),
    lambda: stim.SAMEnvelopeFactory(
        fs=25e3, depth=1, fm=5, delay=0.1, direction='positive',
        input_factory=stim.ToneFactory(fs=25e3, frequency=1e3, level=1)),
], ids=['iir_noise', 'iir_noise_sos', 'iir_noise_negative', 'broadband_noise',
        'shaped_noise', 'notch_noise', 'sam_tone'])
def test_continuous_max_amplitude_is_an_upper_bound(make_factory):
    # Continuous waveforms (including noise, which used to have no
    # max_amplitude at all) answer from a closed form, which must never be
    # exceeded by what is actually played.
    factory = make_factory()
    bound = factory.max_amplitude()
    assert np.isfinite(bound)
    assert peak(factory, int(25e3 * 2)) <= bound


def test_broadband_noise_max_amplitude_is_exact():
    factory = stim.BroadbandNoiseFactory(fs=25e3, level=2)
    assert factory.max_amplitude() == pytest.approx(2 * np.sqrt(3))


def test_silence_max_amplitude():
    assert stim.SilenceFactory().max_amplitude() == 0
    assert stim.SilenceFactory(fill_value=-0.5).max_amplitude() == 0.5


def test_square_wave_max_amplitude():
    on = stim.SquareWaveFactory(fs=1000, level=-2, frequency=10, duty_cycle=0.5)
    off = stim.SquareWaveFactory(fs=1000, level=2, frequency=10, duty_cycle=0)
    assert on.max_amplitude() == 2
    assert off.max_amplitude() == 0


def test_gated_noise_uses_noise_bound():
    noise = make_iir_noise()
    gate = stim.Cos2EnvelopeFactory(fs=25e3, duration=0.1, rise_time=0.01,
                                    input_factory=noise)
    assert gate.max_amplitude() == noise.max_amplitude()
    assert np.abs(gate.get_samples_remaining()).max() <= gate.max_amplitude()


def test_gated_tone_max_amplitude_uses_closed_form():
    fs = 25e3
    tone = stim.ToneFactory(fs=fs, frequency=1e3, level=1)
    gate = stim.GateFactory(fs=fs, start_time=0, duration=0.1,
                            input_factory=tone)
    assert gate.max_amplitude() == tone.max_amplitude()


def make_transformed_envelope(fs=1000):
    carrier = stim.ToneFactory(fs=fs, frequency=50, level=1)
    # The transform can push the envelope above 1, so there is no closed
    # form and the waveform has to be measured.
    return stim.EnvelopeFactory(envelope='bartlett', fs=fs, duration=0.5,
                                rise_time=None, input_factory=carrier,
                                transform=lambda x: 3 * x)


def test_envelope_with_transform_is_measured():
    factory = make_transformed_envelope()
    expected = np.abs(make_transformed_envelope().get_samples_remaining()).max()
    assert factory.max_amplitude() == pytest.approx(expected)
    # Larger than the carrier's own closed form, which would be wrong here.
    assert factory.max_amplitude() > factory.input_factory.max_amplitude()


def test_measured_max_amplitude_leaves_factory_at_start():
    factory = make_transformed_envelope()
    factory.max_amplitude()
    assert_array_equal(factory.get_samples_remaining(),
                       make_transformed_envelope().get_samples_remaining())


def test_continuous_waveform_without_closed_form_raises():
    class Ramp(stim.Carrier):
        def reset(self):
            self.offset = 0

        def next(self, samples):
            result = self.offset + np.arange(samples, dtype=float)
            self.offset += samples
            return result

    with pytest.raises(ValueError):
        Ramp().max_amplitude()


def test_square_wave_chunked_generation_matches_unchunked():
    # The square wave used to restart its cycle at the start of every block.
    assert_chunked_generation(
        stim.SquareWaveFactory,
        dict(fs=1000, level=1, frequency=30, duty_cycle=0.25), 7, 20)


################################################################################
# SAM envelope without equalization
################################################################################
def test_sam_envelope_not_equalized():
    # Without equalization the envelope used to be scaled by 0 (silence).
    fs = 1000
    depth = 1
    env = stim.sam_envelope(0, fs, fs, depth, 5, 0, equalize=False)
    assert env.max() == pytest.approx(1)
    assert env.min() == pytest.approx(0, abs=1e-6)
    assert env.mean() == pytest.approx(1 - depth / 2)


def test_apply_sam_envelope_honors_equalize():
    fs = 1000
    waveform = np.ones(fs)
    for equalize in (True, False):
        expected = stim.sam_envelope(0, fs, fs, 1, 5, 0, equalize=equalize)
        actual = stim.apply_sam_envelope(waveform, fs, 1, 5, 0,
                                         equalize=equalize)
        assert_array_almost_equal(actual, expected)


################################################################################
# Wav files
################################################################################
def test_wav_to_float_signed_integers():
    for dtype in (np.int16, np.int32):
        ii = np.iinfo(dtype)
        samples = np.array([ii.min, 0, ii.max], dtype=dtype)
        result = stim.wav_to_float(samples, legacy_scaling=False)
        assert result[0] == -1
        # Zero must stay zero (legacy scaling adds a small offset).
        assert result[1] == 0
        assert result[2] == pytest.approx(1, abs=1e-4)


def test_wav_to_float_unsigned_8_bit():
    samples = np.array([0, 128, 255], dtype=np.uint8)
    result = stim.wav_to_float(samples, legacy_scaling=False)
    assert_array_almost_equal(result, [-1, 0, 127 / 128])


def test_wav_to_float_legacy_scaling_is_the_default():
    # Stimuli for data already collected were scaled this way, so it stays
    # the default: full integer range stretched onto -1 to 1.
    samples = np.array([-32768, 0, 32767], dtype=np.int16)
    expected = (samples.astype(np.float32) + 32768) / 65535 * 2 - 1
    assert_array_equal(stim.wav_to_float(samples), expected)
    assert_array_equal(stim.wav_to_float(samples, legacy_scaling=True),
                       expected)


def test_wav_to_float_float_is_writable_copy():
    samples = np.array([-0.5, 0.25], dtype=np.float32)
    samples.flags.writeable = False
    for legacy_scaling in (True, False):
        result = stim.wav_to_float(samples, legacy_scaling=legacy_scaling)
        assert result.flags.writeable
        assert_array_equal(result, samples)


def write_wav(tmp_path, name, samples, fs=1000):
    from scipy.io import wavfile
    filename = tmp_path / name
    wavfile.write(filename, fs, samples)
    return filename


def test_load_wav_peak_normalization(tmp_path):
    # Negative peak larger than the positive one.
    samples = np.array([0, 0.25, -0.5, 0.1], dtype=np.float32)
    filename = write_wav(tmp_path, 'peak.wav', samples)
    corrected = stim.load_wav(1000, filename, normalization='pe',
                              legacy_scaling=False)
    assert np.abs(corrected).max() == pytest.approx(1)
    assert corrected.min() == pytest.approx(-1)
    # Legacy scaling divides by the largest positive value (0.25), so the
    # negative peak ends up at -2.
    legacy = stim.load_wav(1000, filename, normalization='pe')
    assert legacy.max() == pytest.approx(1)
    assert legacy.min() == pytest.approx(-2)


def test_load_wav_float_file_with_calibration(tmp_path):
    # Float files are read as a read-only memory map; scaling them by the
    # calibration used to fail.
    samples = np.array([0, 0.5, -0.5], dtype=np.float32)
    filename = write_wav(tmp_path, 'float.wav', samples)
    cal = calibration.FlatCalibration.unity()
    result = stim.load_wav(1000, filename, level=6, calibration=cal)
    assert_array_almost_equal(result, samples * util.dbi(6), decimal=5)


def test_load_wav_int16_file(tmp_path):
    samples = np.array([0, 16384, -32768], dtype=np.int16)
    filename = write_wav(tmp_path, 'int16.wav', samples)
    result = stim.load_wav(1000, filename, legacy_scaling=False)
    assert_array_almost_equal(result, [0, 0.5, -1])


def test_wav_factories_pass_legacy_scaling(tmp_path):
    samples = np.array([0, 0.25, -0.5, 0.1], dtype=np.float32)
    write_wav(tmp_path, 'a.wav', samples)
    for legacy_scaling, expected in ((True, 2), (False, 1)):
        factory = stim.WavFileFactory(1000, tmp_path / 'a.wav',
                                      normalization='pe',
                                      legacy_scaling=legacy_scaling)
        assert factory.max_amplitude() == pytest.approx(expected)
        sequence = stim.WavSequenceFactory(1000, tmp_path,
                                           normalization='pe',
                                           legacy_scaling=legacy_scaling)
        assert sequence.max_amplitude() == pytest.approx(expected)


def test_wav_sequence_max_amplitude_is_loudest_file(tmp_path):
    write_wav(tmp_path, 'quiet.wav', np.array([0, 0.1], dtype=np.float32))
    write_wav(tmp_path, 'loud.wav', np.array([0, -0.4], dtype=np.float32))
    # No normalization, so each file keeps its own peak.
    sequence = stim.WavSequenceFactory(1000, tmp_path, normalization=None)
    assert sequence.max_amplitude() == pytest.approx(0.4)


################################################################################
# Duplicate definitions
################################################################################
def test_stim_has_no_duplicate_definitions():
    # A second definition with the same name silently replaces the first
    # (e.g., a dead SquareWaveFactory and a second copy of the shaped noise
    # code both used to be in this module).
    import ast
    import collections
    from pathlib import Path
    tree = ast.parse(Path(stim.__file__).read_text(encoding='utf-8'))
    names = collections.Counter(
        node.name for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    )
    assert [n for n, c in names.items() if c > 1] == []
