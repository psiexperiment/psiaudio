import pytest

from collections import deque
from functools import partial

import numpy as np
from scipy import signal

from psiaudio import pipeline
from psiaudio.pipeline import normalize_index


@pytest.fixture
def data(fs):
    md = {'foo': 'bar'}
    data = np.random.uniform(size=100000)
    return pipeline.PipelineData(data, fs, metadata=md)


def test_normalize_index():
    assert normalize_index(np.s_[np.newaxis], 1) == (np.newaxis, slice(None))
    assert normalize_index(np.s_[...], 1) == (slice(None),)
    assert normalize_index(np.s_[...], 2) == (slice(None), slice(None))
    assert normalize_index(np.s_[..., :], 2) == (slice(None), slice(None))
    assert normalize_index(np.s_[::5], 1) == (slice(None, None, 5),)
    assert normalize_index(np.s_[::5], 2) == (slice(None, None, 5), slice(None))
    assert normalize_index(np.s_[:, ::5], 2) == (slice(None), slice(None, None, 5))
    assert normalize_index(np.s_[np.newaxis, ::5], 2) == (np.newaxis, slice(None, None, 5), slice(None))
    assert normalize_index(np.s_[np.newaxis, ..., ::5], 2) == (np.newaxis, slice(None), slice(None, None, 5))
    assert normalize_index(np.s_[np.newaxis, ::5, ...], 2) == (np.newaxis, slice(None, None, 5), slice(None))
    assert normalize_index(np.s_[..., ::5, :], 2) == (slice(None, None, 5), slice(None))
    assert normalize_index(np.s_[..., ::5, :], 3) == (slice(None), slice(None, None, 5), slice(None))
    assert normalize_index(np.s_[0, 1:], 2) == (0, slice(1, None, None))

    with pytest.raises(IndexError):
        normalize_index(np.s_[..., ...], 4)


def test_pipeline_data_1d(data):
    fs = data.fs
    md = data.metadata.copy()

    assert data.channel is None
    assert data.epochs is None

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


def test_pipeline_data_2d(data):
    # Upcast to 2D
    data = data[np.newaxis]
    fs = data.fs
    md = data.metadata.copy()

    assert data.channel == [None]
    assert data.epochs is None

    # Check basic time-slicing
    data[..., ::4]
    assert data[::4].fs == fs
    assert data[4:].s0 == 0

    assert data[:, ::2].fs == fs / 2
    assert data[:, ::3].fs == fs / 3
    assert data[:, ::2][:, ::3].fs == fs / 2 / 3

    assert data[:, 5:].s0 == 5
    assert data[:, 10:].s0 == 10
    assert data[:, 10:][:, 10:].s0 == (10 + 10)

    d = data[:, 5::2][:, 10::2]
    assert d.s0 == (5 + 10)
    assert d.fs == fs / 2 / 2

    assert data[:, :1].fs == fs
    assert data[:, 1:1].fs == fs
    assert data[:, :1:1].fs == fs
    assert data[:, 1:1:1].fs == fs
    assert data[:, :1].s0 == 0
    assert data[:, 1:1].s0 == 1
    assert data[:, :1:1].s0 == 0
    assert data[:, 1:1:1].s0 == 1
    assert data[:, :1].metadata == md
    assert data[:, 1:1].metadata == md
    assert data[:, :1:1].metadata == md
    assert data[:, 1:1:1].metadata == md

    n = len(data)
    assert data[:, n:].s0 == n

    # Now, check channel slicing
    assert data[0].channel == None
    assert data[0].fs == fs
    assert data[0].s0 == 0
    assert data[:1].channel == [None]
    assert data[:1].fs == fs
    assert data[:1].s0 == 0
    assert data[1:].channel == []
    assert data[1:].fs == fs
    assert data[1:].s0 == 0

    assert data[0, :1].fs == fs
    assert data[0, 1:1].fs == fs
    assert data[0, :1:1].fs == fs
    assert data[0, 1:1:1].fs == fs
    assert data[0, :1].s0 == 0
    assert data[0, 1:1].s0 == 1
    assert data[0, :1:1].s0 == 0
    assert data[0, 1:1:1].s0 == 1
    assert data[0, :1].metadata == md
    assert data[0, 1:1].metadata == md
    assert data[0, :1:1].metadata == md
    assert data[0, 1:1:1].metadata == md
    assert data[0, :1].channel == None
    assert data[0, 1:1].channel == None
    assert data[0, :1:1].channel == None
    assert data[0, 1:1:1].channel == None


def test_pipeline_data_3d(data):
    # Upcast to 2D
    md = data.metadata.copy()
    data = data[np.newaxis, np.newaxis]
    fs = data.fs
    assert data.channel == [None]
    assert data.metadata == [md]
    assert data[0].metadata == md
    assert data[0].channel == [None]
    assert data[0, 0].metadata == md
    assert data[0, 0].channel == None


def test_pipeline_data_concat_time(fs):
    md = {'foo': 'bar'}
    o = 0
    data = []
    segments = []
    for i in range(10):
        n = np.random.randint(1, 10)
        samples = np.random.uniform(size=n)
        d = pipeline.PipelineData(samples, fs, o, metadata=md)
        data.append(d)
        segments.append(samples)
        o += n

    expected = np.concatenate(segments, axis=-1)
    actual = pipeline.concat(data)
    np.testing.assert_array_equal(actual, expected)
    assert actual.s0 == 0
    assert actual.channel == None
    assert actual.metadata == md


def test_pipeline_data_concat_epochs(fs):
    o = 0
    data = []
    epochs = []
    md = []
    for i in range(10):
        samples = np.random.uniform(size=10)
        d = pipeline.PipelineData(samples, fs, 0, metadata={'epoch': i})
        data.append(d)
        md.append({'epoch': i})

    expected = np.concatenate([d[np.newaxis, np.newaxis, :] for d in data], axis=0)
    actual = pipeline.concat(data, axis=-3)
    np.testing.assert_array_equal(actual, expected)
    assert actual.s0 == 0
    assert actual.channel == [None]
    assert actual.metadata == md


def feed_pipeline(cb, data, include_offset=False):
    result = []
    o = 0
    cr = cb(result.append)
    while o < data.shape[-1]:
        try:
            i = np.random.randint(low=10, high=100)
            if include_offset:
                cr.send((o, data[o:o+i]))
            else:
                cr.send(data[o:o+i])
            o += i
        except StopIteration:
            break
    return result


def test_capture_epoch(data):
    s0 = 12345
    samples = 54321

    expected = data[s0:s0+samples]
    cb = partial(pipeline.capture_epoch, s0, samples, {})
    result = feed_pipeline(cb, data, True)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], expected)
    assert result[0].metadata == {}
    assert result[0].s0 == s0
    assert result[0].channel == None


def test_extract_epochs(data):
    queue = deque()
    epoch_size = 0.1
    epoch_samples = int(round(epoch_size * data.fs))
    cb = partial(pipeline.extract_epochs, data.fs, queue, epoch_size, 0, 0)
    queue.append({'t0': 0.1, 'metadata': {'epoch': 'A'}})
    queue.append({'t0': 0.2, 'metadata': {'epoch': 'B'}})
    queue.append({'t0': 0.3, 'metadata': {'epoch': 'C'}})
    result = pipeline.concat(feed_pipeline(cb, data), -3)
    assert result.shape == (3, 1, epoch_samples)
    #assert result.s0 == 0
    assert result.channel == [None]
    assert result.metadata == [{'epoch': 'A'}, {'epoch': 'B'}, {'epoch': 'C'}]


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
    assert actual.s0 == 0
    assert actual.fs == data.fs
    np.testing.assert_array_equal(actual, expected)


def test_blocked(data):
    cb = partial(pipeline.blocked, 100)
    actual = feed_pipeline(cb, data)
    for i, a in enumerate(actual):
        assert a.s0 == (i * 100)
        assert a.shape == (100,)
    actual = pipeline.concat(actual)
    assert actual.fs == data.fs
    assert actual.s0 == 0
    np.testing.assert_array_equal(actual, data)


@pytest.fixture(scope='module', params=[None, 'constant', 'linear'])
def detrend_mode(request):
    return request.param


def test_detrend(data, detrend_mode):
    cb = partial(pipeline.detrend, detrend_mode)
    with pytest.raises(ValueError):
        actual = feed_pipeline(cb, data)

    fs = data.fs
    epoch_size = 0.1
    epoch_samples = int(round(epoch_size * fs))

    queue = deque()
    expected = []
    for i, name in enumerate('ABC'):
        t0 = (i + 1) / 10
        queue.append({'t0': t0, 'metadata': {'epoch': name}})
        lb = int(round(t0 * fs))
        ub = lb + epoch_samples
        expected.append(data[np.newaxis, lb:ub])
    expected = pipeline.concat(expected, axis=-3)
    if detrend_mode is not None:
        expected = signal.detrend(expected, axis=-1, type=detrend_mode)

    def cb(target):
        nonlocal detrend_mode
        nonlocal queue
        nonlocal fs
        nonlocal epoch_size

        return \
            pipeline.extract_epochs(
                data.fs,
                queue,
                epoch_size,
                0,
                0,
                pipeline.detrend(
                    detrend_mode,
                    target
                ).send
            )

    actual = pipeline.concat(feed_pipeline(cb, data), -3)
    np.testing.assert_array_almost_equal(actual, expected)

    assert actual.shape == (3, 1, epoch_samples)
    assert actual.channel == [None]
    assert actual.metadata == [{'epoch': 'A'}, {'epoch': 'B'}, {'epoch': 'C'}]
