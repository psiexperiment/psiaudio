import pytest

import numpy as np
import pandas as pd

from psiaudio.calibration import (
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
def relative_phases():
    # Phase correction data in degrees.
    values = {
         300: 50.4,
         500: 30.1,
         700: 21.5,
        1000: 14.9,
        1500:  9.8,
        2000:  7.3,
        3000:  4.6,
        4000:  3.2,
        6000:  1.7,
    }
    frequencies = np.fromiter(values.keys(), 'double')
    phases = np.deg2rad(np.fromiter(values.values(), 'double'))
    return dict(zip(frequencies, phases))


@pytest.fixture
def point_calibration(relative_levels):
    frequencies = list(relative_levels.keys())
    levels = list(relative_levels.values())
    return PointCalibration.from_spl(frequencies, levels, vrms=1)


def test_unity_calibration():
    calibration = FlatCalibration.unity()
    assert calibration.get_sf(1000, 0) == 1
    assert calibration.get_sf(1000, 20) == 10
    assert calibration.get_sf(1000, 40) == 100
    assert calibration.get_db(1000, 1) == 0
    assert calibration.get_db(1000, 10) == 20
    assert calibration.get_db(1000, 100) == 40


def test_flat_calibration():
    calibration = FlatCalibration(sensitivity=20*np.log10(1/20e-6))
    tests = {
         74:  0.1,
         94:  1.0,
        114: 10.0,
    }
    for level, expected_rms in tests.items():
        sf = calibration.get_sf(1e3, level)
        assert isinstance(sf, float)
        assert pytest.approx(expected_rms, rel=1e-2) == sf

        # Also, verify that we are properly returning a numpy array if we pass
        # in a list of frequencies. This ensures compatibility with downstream
        # code that does not care whether they are using an interp or flat
        # calibration.
        sf = calibration.get_sf([1e3, 2e3], level)
        assert sf.shape == (2,)
        np.testing.assert_allclose(sf, expected_rms, rtol=1e-2)



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


def test_flat_calibration_from_mv_pa():
    calibration = FlatCalibration.from_mv_pa(1)
    assert calibration.get_sf(1000, 94) == pytest.approx(1e-3, rel=1e-2)
    assert calibration.get_db(1000, 1e-3) == pytest.approx(94, rel=1e-2)
    assert calibration.get_sf(1000, 114) == pytest.approx(10e-3, rel=1e-2)
    assert calibration.get_db(1000, 10e-3) == pytest.approx(114, rel=1e-2)


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


def test_phase_calculation(relative_levels, relative_phases):
    calibration = InterpCalibration.from_spl(
        frequency=list(relative_levels.keys()),
        spl=list(relative_levels.values()),
        phase=list(relative_phases.values()),
    )
    for frequency, phase in relative_phases.items():
        assert calibration.get_phase(frequency) == phase
    p1 = relative_phases[1000]
    p2 = relative_phases[1500]
    p_average = (p1 + p2) / 2
    assert calibration.get_phase(1250) == p_average
    assert calibration.get_phase(1251) != p_average
