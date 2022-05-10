import numpy as np
from psiaudio import pipeline


def test_capture_epoch():
    signal = np.random.uniform(size=100000)
    t0 = 12345
    samples = 54321
    expected = signal[t0:t0+samples]

    result = []
    cr = pipeline.capture_epoch(t0, samples, {}, result.append)

    o = 0
    while True:
        try:
            i = np.random.randint(low=10, high=100)
            cr.send((o, signal[o:o+i]))
            o += i
        except StopIteration:
            break

    assert len(result) == 1
    result = result[0]['signal']
    np.testing.assert_array_equal(result, expected)
