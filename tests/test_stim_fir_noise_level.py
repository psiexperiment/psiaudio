'''
Level of bandlimited FIR noise through a real speaker calibration.

The calibration is a cftscal Golay measurement of a speaker in a noise-exposure
booth (see `load_golay_calibration`). Its sensitivity has deep, narrow
reflection notches. The noise used to be scaled per frequency from that
calibration, which played it several dB too loud. `legacy_level=True` keeps the
old scaling, and the goldens in `test_stim_golden.py` pin its waveforms.

Each level here is measured from the generated waveform, independently of how
the factory sets it: the Welch PSD of the noise is weighted by the
calibration's sensitivity at each frequency and summed, as a microphone at the
calibrated position would.
'''
import numpy as np
import pytest
from scipy import signal

from psiaudio import calibration, stim

from .fixtures.generate_fixtures import load_golay_calibration
from .helpers import assert_chunked_generation

FS = 100e3
LEVEL = 100

BANDS = [
    pytest.param(0.7e3, 1.4e3, id='0.7-1.4k'),
    pytest.param(1.4e3, 2.8e3, id='1.4-2.8k'),
    pytest.param(2.8e3, 5.6e3, id='2.8-5.6k'),
    pytest.param(8e3, 16e3, id='8-16k'),
    pytest.param(4e3, 45e3, id='4-45k'),
]


@pytest.fixture(scope='module')
def golay_calibration():
    return load_golay_calibration()


def make_factory(cal, fl, fh, **kwargs):
    # As the noise-exposure paradigm configures it.
    kwargs = dict(dict(fs=FS, fl=fl, fh=fh, level=LEVEL, ntaps=1001, seed=1,
                       max_correction=20, calibration=cal), **kwargs)
    return stim.BandlimitedFIRNoiseFactory(**kwargs)


def measured_level(x, fs, cal):
    '''
    Level of `x` through `cal`, from its Welch PSD (~1.5 Hz resolution, so the
    calibration's notches are resolved).
    '''
    f, pxx = signal.welch(x, fs=fs, window='hann', nperseg=2**16)
    f_cal = np.clip(f, cal.frequency.min(), cal.frequency.max())
    power = np.sum(pxx * 10 ** (cal.get_sens(f_cal) / 10)) * (f[1] - f[0])
    return 10 * np.log10(power)


@pytest.mark.parametrize('equalize', [True, False], ids=['eq', 'noeq'])
@pytest.mark.parametrize('fl,fh', BANDS)
def test_level_through_golay_calibration(golay_calibration, fl, fh, equalize):
    factory = make_factory(golay_calibration, fl, fh, equalize=equalize)
    x = factory.next(int(5 * FS))
    assert measured_level(x, FS, golay_calibration) == \
        pytest.approx(LEVEL, abs=0.25)


@pytest.mark.parametrize('fl,fh,equalize,excess', [
    pytest.param(1.4e3, 2.8e3, True, 4.0, id='1.4-2.8k-eq'),
    pytest.param(1.4e3, 2.8e3, False, 6.0, id='1.4-2.8k-noeq'),
    pytest.param(2.8e3, 5.6e3, True, 3.5, id='2.8-5.6k-eq'),
    pytest.param(2.8e3, 5.6e3, False, 7.9, id='2.8-5.6k-noeq'),
])
def test_legacy_level(golay_calibration, fl, fh, equalize, excess):
    '''
    Documents the level `legacy_level` keeps: short of `level` by the fraction
    of the white noise's power in the band, and louder than that by `excess`
    through this calibration's notches (via the FIR filter's smoothing when
    equalized, `get_mean_sf`'s amplitude average when not).
    '''
    factory = make_factory(golay_calibration, fl, fh, equalize=equalize,
                           legacy_level=True)
    x = factory.next(int(5 * FS))
    shortfall = 10 * np.log10(2 * (fh - fl) / FS)
    expected = LEVEL + shortfall + excess
    assert measured_level(x, FS, golay_calibration) == \
        pytest.approx(expected, abs=0.3)


@pytest.mark.parametrize('equalize', [True, False], ids=['eq', 'noeq'])
def test_fixed_level_only_rescales(golay_calibration, equalize):
    '''
    The fix changes only the noise's overall gain: the filter (so the
    spectrum, equalized or not) and the random samples are the legacy ones.
    '''
    legacy = make_factory(golay_calibration, 2.8e3, 5.6e3, equalize=equalize,
                          legacy_level=True)
    fixed = make_factory(golay_calibration, 2.8e3, 5.6e3, equalize=equalize)
    np.testing.assert_array_equal(fixed.taps, legacy.taps)
    x_legacy = legacy.next(10000)
    x_fixed = fixed.next(10000)
    np.testing.assert_allclose(x_fixed, x_legacy * fixed.level_correction,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('legacy_level', [False, True])
def test_chunked_generation(golay_calibration, legacy_level, chunksize,
                            n_chunks):
    kwargs = dict(fs=FS, fl=2.8e3, fh=5.6e3, level=LEVEL, seed=1,
                  max_correction=20, equalize=True,
                  calibration=golay_calibration, legacy_level=legacy_level)
    # Not exact: filtering in chunks rounds differently (by ~1e-17 here), as
    # for ShapedNoiseFactory.
    assert_chunked_generation(stim.BandlimitedFIRNoiseFactory, kwargs,
                              chunksize, n_chunks, exact=False)


@pytest.mark.parametrize('legacy_level', [False, True])
@pytest.mark.parametrize('equalize', [True, False], ids=['eq', 'noeq'])
def test_max_amplitude_bounds_waveform(golay_calibration, equalize,
                                       legacy_level):
    factory = make_factory(golay_calibration, 1.4e3, 2.8e3, equalize=equalize,
                           legacy_level=legacy_level)
    x = factory.next(int(FS))
    assert np.abs(x).max() <= factory.max_amplitude()


@pytest.mark.parametrize('fs', [96e3, 100e3, 195312.5])
def test_fir_noise_level_allpass(fs):
    '''
    Unit-variance white noise through an all-pass filter is at the
    calibration's sensitivity.
    '''
    cal = calibration.FlatCalibration.from_spl(94)
    level = stim.fir_noise_level(np.array([1.0]), fs, cal)
    assert level == pytest.approx(94, abs=1e-3)
