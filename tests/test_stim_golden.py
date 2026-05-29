"""
Golden-file regression tests for stim.py.

Each test regenerates a waveform with fixed parameters and compares it
sample-for-sample against a reference .npy file stored in tests/fixtures/.
Any change in output — intentional or not — will cause the test to fail.

To regenerate the reference files after an intentional change:
    python tests/fixtures/generate_fixtures.py
Commit the updated .npy files alongside the code change.
"""
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from psiaudio import calibration, stim, util

FIXTURE_DIR = Path(__file__).parent / 'fixtures'
CAL = calibration.FlatCalibration.from_spl(94)


def _load(name):
    return np.load(FIXTURE_DIR / f'{name}.npy')


@pytest.mark.parametrize('func,kwargs,fixture', [
    pytest.param(
        stim.tone,
        dict(fs=100e3, frequency=1e3, level=1.0, duration=0.1),
        'tone_1kHz_100ms_fs100k',
        id='tone_1kHz',
    ),
    pytest.param(
        stim.tone,
        dict(fs=100e3, frequency=8e3, level=1.0, duration=0.05, polarity=-1),
        'tone_8kHz_50ms_polneg_fs100k',
        id='tone_negative_polarity',
    ),
    pytest.param(
        stim.cos2envelope,
        dict(fs=100e3, duration=0.2, rise_time=10e-3),
        'cos2envelope_200ms_rise10ms_fs100k',
        id='cos2envelope',
    ),
    pytest.param(
        stim.envelope,
        dict(window='gaussian', fs=100e3, duration=0.2, rise_time=10e-3),
        'gaussian_envelope_200ms_rise10ms_fs100k',
        id='gaussian_envelope',
    ),
    pytest.param(
        stim.chirp,
        dict(fs=100e3, start_frequency=1e3, end_frequency=8e3,
             duration=0.01, level=1.0),
        'chirp_1k_8k_10ms_fs100k',
        id='chirp',
    ),
    pytest.param(
        stim.broadband_noise,
        dict(fs=100e3, level=1.0, duration=0.5, seed=1),
        'broadband_noise_500ms_seed1_fs100k',
        id='broadband_noise',
    ),
    pytest.param(
        stim.bandlimited_noise,
        dict(fs=100e3, level=80, fl=1e3, fh=8e3, duration=0.5, seed=1,
             calibration=CAL),
        'bandlimited_noise_1k_8k_500ms_seed1_fs100k',
        id='bandlimited_noise',
    ),
    pytest.param(
        stim.gap,
        dict(fs=100e3, fc=4e3, octaves=1, gap=5e-3, durations=[0.05, 0.05],
             rise_time=5e-3, level=80, calibration=CAL),
        'gap_noise_4k_1oct_5ms_2x50ms_fs100k',
        id='gap_noise_single',
    ),
    pytest.param(
        stim.gap,
        dict(fs=100e3, fc=4e3, octaves=0, gap=5e-3, durations=[0.05, 0.05],
             rise_time=5e-3, level=80, calibration=CAL),
        'gap_tone_4k_5ms_2x50ms_fs100k',
        id='gap_tone_single',
    ),
    pytest.param(
        stim.gap,
        dict(fs=100e3, fc=4e3, octaves=1, gap=5e-3,
             durations=[0.05, 0.05, 0.05], rise_time=5e-3, level=80,
             calibration=CAL),
        'gap_noise_4k_1oct_5ms_3x50ms_fs100k',
        id='gap_noise_multiple',
    ),
    pytest.param(
        stim.sam_envelope,
        dict(offset=0, samples=10000, fs=100e3, depth=1.0,
             fm=100, delay=0, equalize=True),
        'sam_envelope_depth1_fm100_eq_fs100k',
        id='sam_envelope',
    ),
])
def test_golden(func, kwargs, fixture):
    assert_array_equal(func(**kwargs), _load(fixture))


# ---------------------------------------------------------------------------
# Behavioral assertions not captured by the golden files
# ---------------------------------------------------------------------------

def test_tone_rms():
    actual = stim.tone(fs=100e3, frequency=1e3, level=1.0, duration=0.1)
    assert util.rms(actual) == pytest.approx(1.0, rel=1e-5)


def test_tone_polarity_inversion():
    pos = stim.tone(fs=100e3, frequency=8e3, level=1.0, duration=0.05)
    neg = stim.tone(fs=100e3, frequency=8e3, level=1.0, duration=0.05, polarity=-1)
    assert_array_equal(neg, -pos)


def test_cos2envelope_shape():
    actual = stim.cos2envelope(fs=100e3, duration=0.2, rise_time=10e-3)
    # cos2ramp uses arange(m) so endpoints are sin(π/m)² ≈ 2.5e-6, not exactly 0
    assert actual[0] == pytest.approx(0, abs=1e-4)
    assert actual[-1] == pytest.approx(0, abs=1e-4)
    assert actual[len(actual) // 2] == pytest.approx(1.0)


def test_gaussian_envelope_peak():
    actual = stim.envelope(window='gaussian', fs=100e3, duration=0.2, rise_time=10e-3)
    assert actual[len(actual) // 2] == pytest.approx(1.0)


def test_broadband_noise_rms():
    actual = stim.broadband_noise(fs=100e3, level=1.0, duration=0.5, seed=1)
    assert util.rms(actual) == pytest.approx(1.0, abs=0.01)


def test_sam_envelope_rms():
    actual = stim.sam_envelope(
        offset=0, samples=10000, fs=100e3, depth=1.0,
        fm=100, delay=0, equalize=True)
    assert util.rms(actual) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize('durations,gap_starts', [
    pytest.param(
        [0.05, 0.05],
        [int(0.05 * 100e3)],
        id='single_gap',
    ),
    pytest.param(
        [0.05, 0.05, 0.05],
        [int(0.05 * 100e3), int((0.05 + 5e-3 + 0.05) * 100e3)],
        id='multiple_gaps',
    ),
])
def test_gap_zeros(durations, gap_starts):
    actual = stim.gap(
        fs=100e3, fc=4e3, octaves=1, gap=5e-3, durations=durations,
        rise_time=5e-3, level=80, calibration=CAL)
    gap_n = int(5e-3 * 100e3)
    for start in gap_starts:
        assert_array_equal(actual[start:start + gap_n], 0)
