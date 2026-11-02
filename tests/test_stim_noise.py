import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from psiaudio import stim, util

from .helpers import assert_chunked_generation


@pytest.fixture(scope='module', params=[20, 80, 140])
def stim_level(request):
    return request.param


@pytest.fixture(scope='module', params=[0.001, 0.1, 10])
def stim_duration(request):
    return request.param


@pytest.fixture(scope='module', params=[1e3, 2e3])
def stim_fl(request):
    return request.param


@pytest.fixture(scope='module', params=[6e3, 8e3])
def stim_fh(request):
    return request.param


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
    assert_chunked_generation(stim.BroadbandNoiseFactory, kwargs, chunksize,
                              n_chunks)


def test_bandlimited_noise(fs, stim_level, stim_duration, stim_fl, stim_fh,
                           stim_calibration):
    if fs != 100e3:
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


@pytest.mark.parametrize('fs', [96e3, 195312.5])
@pytest.mark.parametrize('equalize', [False, True])
def test_bandlimited_fir_noise_level(fs, equalize, stim_calibration):
    fl, fh, level = 4e3, 40e3, 80
    kwargs = dict(fs=fs, level=level, fl=fl, fh=fh, duration=2, ntaps=1001,
                  calibration=stim_calibration, equalize=equalize)
    expected_shortfall = 10 * np.log10(2 * (fh - fl) / fs)

    def band_level(**extra):
        x = stim.bandlimited_fir_noise(**kwargs, **extra)
        return stim_calibration.get_spl(1e3, util.rms(x))

    # The legacy level is short by about the fraction of the white noise's
    # power in the band (a little less: the filter's skirts pass some more).
    assert band_level() == pytest.approx(level + expected_shortfall, abs=0.3)
    assert band_level(correct_level=True) == pytest.approx(level, abs=0.3)


def test_bandlimited_noise_factory(fs, stim_level, stim_fl, stim_fh,
                                   stim_calibration, chunksize, n_chunks):
    if fs != 100e3:
        pytest.skip()
    kwargs = dict(fs=fs, level=stim_level, fl=stim_fl, fh=stim_fh,
                  calibration=stim_calibration, seed=1, filter_rolloff=1,
                  passband_attenuation=1, stopband_attenuation=80)
    assert_chunked_generation(stim.BandlimitedNoiseFactory, kwargs, chunksize,
                              n_chunks)


@pytest.fixture(scope='module', params=[(1e3, 2e3), (4e3, 8e3)])
def shaped_noise_gains(fs, request):
    stim_fl, stim_fh = request.param
    return {
        0: 0,
        stim_fl * 0.99: 0,
        stim_fl: -80,
        stim_fh: -80,
        stim_fh / 0.99: 0,
        fs / 2: 0,
    }


@pytest.mark.slow
def test_shaped_noise(fs, stim_level, stim_duration, shaped_noise_gains,
                      stim_calibration):
    if stim_level == 80:
        pytest.skip()
    if stim_duration == 0.1:
        pytest.skip()

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


@pytest.mark.slow
def test_shaped_noise_factory(fs, stim_level, shaped_noise_gains,
                              stim_calibration, chunksize, n_chunks):
    kwargs = dict(fs=fs, level=stim_level, gains=shaped_noise_gains,
                  calibration=stim_calibration, seed=1)
    # Set exact to False. There seem to be some numerical precision issues, but
    # they are less than 1e-14.
    assert_chunked_generation(stim.ShapedNoiseFactory, kwargs, chunksize,
                              n_chunks, exact=False)
