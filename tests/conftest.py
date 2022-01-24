import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
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


@pytest.fixture(scope='module', params=[1e3, 2e3])
def stim_fl(request):
    return request.param


@pytest.fixture(scope='module', params=[6e3, 8e3])
def stim_fh(request):
    return request.param


@pytest.fixture(scope='module', params=[0, 1])
def silence_fill_value(request):
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


def assert_chunked_generation(factory_class, kwargs, chunksize, n_chunks,
                              exact=True):
    '''
    Test chunked generation yields same result as unchunked
    '''
    factory = factory_class(**kwargs)
    chunked_samples = [factory.next(chunksize) for i in range(n_chunks)]
    chunked_samples = np.concatenate(chunked_samples, axis=-1)
    factory.reset()
    unchunked_samples = factory.next(chunksize * n_chunks)
    if exact:
        assert_array_equal(unchunked_samples, chunked_samples)
    else:
        assert_array_almost_equal(unchunked_samples, chunked_samples)
