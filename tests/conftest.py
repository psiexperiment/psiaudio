import pytest


@pytest.fixture(scope='module', params=[25e3, 50e3, 100e3, 200e3])
def fs(request):
    return request.param
