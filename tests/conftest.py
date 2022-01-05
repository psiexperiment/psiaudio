import pytest

from psiaudio import calibration


@pytest.fixture(scope='module', params=[25e3, 50e3, 100e3, 200e3])
def fs(request):
    return request.param


@pytest.fixture(scope='module', params=[20, 60, 100, 140])
def stim_level(request):
    return request.param


@pytest.fixture(scope='module', params=[0.001, 0.1, 10])
def stim_duration(request):
    return request.param


@pytest.fixture(scope='module', params=['cosine-squared', 'blackman'])
def stim_window(request):
    return request.param


@pytest.fixture(scope='module', params=[1e3, 2e3])
def stim_fl(request):
    return request.param


@pytest.fixture(scope='module', params=[6e3, 8e3])
def stim_fh(request):
    return request.param


@pytest.fixture
def chunksize():
    return 1000

@pytest.fixture
def n_chunks():
    return 10


@pytest.fixture
def stim_calibration():
    return calibration.FlatCalibration.from_spl(94)
