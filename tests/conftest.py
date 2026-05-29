import pytest

from psiaudio import calibration


def pytest_addoption(parser):
    parser.addoption('--slow', action='store_true', dest='slow', default=False,
                     help='Enable slow-running tests')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--slow'):
        skip_slow = pytest.mark.skip(reason='Test runs very slowly. Use --slow to run.')
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope='module', params=[25e3, 100e3, 195312.5])
def fs(request):
    return request.param


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
