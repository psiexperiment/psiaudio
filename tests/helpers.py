import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


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
