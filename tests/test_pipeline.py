import pytest
from functools import partial

import numpy as np
from scipy import signal

from psiaudio import pipeline


@pytest.fixture
def data(fs):
    md = {'foo': 'bar'}
    data = np.random.uniform(size=100000)
    return pipeline.PipelineData(data, fs, metadata=md)


def test_pipeline_data(data):
    fs = data.fs
    md = data.metadata.copy()

    assert data[::2].fs == fs / 2
    assert data[::3].fs == fs / 3
    assert data[::2][::3].fs == fs / 2 / 3

    assert data[5:].s0 == 5
    assert data[10:].s0 == 10
    assert data[10:][10:].s0 == (10 + 10)

    d = data[5::2][10::2]
    assert d.s0 == (5 + 10)
    assert d.fs == fs / 2 / 2

    assert data[:1].fs == fs
    assert data[1:1].fs == fs
    assert data[:1:1].fs == fs
    assert data[1:1:1].fs == fs
    assert data[:1].s0 == 0
    assert data[1:1].s0 == 1
    assert data[:1:1].s0 == 0
    assert data[1:1:1].s0 == 1
    assert data[:1].metadata == md
    assert data[1:1].metadata == md
    assert data[:1:1].metadata == md
    assert data[1:1:1].metadata == md

    n = len(data)
    assert data[n:].s0 == n


def test_pipeline_data_concat(fs):
    md = {'foo': 'bar'}

    o = 0
    data = []
    segments = []
    for i in range(10):
        n = np.random.randint(1, 10)
        samples = np.random.uniform(size=n)
        d = pipeline.PipelineData(samples, fs, o)
        data.append(d)
        segments.append(samples)
        o += n

    expected = np.concatenate(segments, axis=-1)
    actual = pipeline.concat(data)
    np.testing.assert_array_equal(actual, expected)


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
    np.testing.assert_array_equal(result[0], expected)


def feed_pipeline(cb, data):
    result = []
    o = 0
    cr = cb(result.append)
    while o < data.shape[-1]:
        i = np.random.randint(low=10, high=100)
        cr.send(data[o:o+i])
        o += i
    return result


def test_rms(fs):
    duration = 0.1
    chunk_samples = int(round(fs * duration))
    n_samples = chunk_samples * 10
    signal = np.arange(n_samples)

    signal = pipeline.PipelineData(signal, fs=fs, metadata={'test': True})
    x = signal.reshape((-1, chunk_samples))
    expected = np.mean(x ** 2, axis=-1) ** 0.5

    cb = partial(pipeline.rms, fs, duration)
    result = feed_pipeline(cb, signal)
    for r in result:
        assert r.fs == (fs / chunk_samples)

    actual = pipeline.concat(result, axis=-1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.fs == (fs / chunk_samples)
    assert actual.s0 == 0


def test_iirfilter(data, stim_fl, stim_fh):
    cb = partial(pipeline.iirfilter, data.fs, 1, (stim_fl, stim_fh), None,
                 None, 'band', 'butter')

    b, a = signal.iirfilter(1, (stim_fl, stim_fh), None, None, 'band',
                            ftype='butter', fs=data.fs)
    zi = signal.lfilter_zi(b, a)
    expected, _ = signal.lfilter(b, a, data, zi=zi * data[0])
    actual = feed_pipeline(cb, data)
    actual = pipeline.concat(actual)
    np.testing.assert_array_equal(actual, expected)


def test_blocked(data):
    cb = partial(pipeline.blocked, 100)
    actual = feed_pipeline(cb, data)
    for i, a in enumerate(actual):
        assert a.s0 == (i * 100)
        assert a.shape == (100,)

    actual = pipeline.concat(actual)
    np.testing.assert_array_equal(actual, data)
