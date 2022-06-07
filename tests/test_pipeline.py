import pytest

from collections import deque
from functools import partial

import numpy as np
from scipy import signal

from psiaudio import pipeline
from psiaudio.pipeline import normalize_index
from psiaudio import util


@pytest.fixture
def data1d(fs):
    md = {'foo': 'bar'}
    data = np.random.uniform(size=100000)
    return pipeline.PipelineData(data, fs, metadata=md)


@pytest.fixture
def data2d(fs):
    md = {'foo': 'bar'}
    channel = ['channel1', 'channel2']
    data = np.random.uniform(size=(2, 100000))
    return pipeline.PipelineData(data, fs, channel=channel, metadata=md)


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


def test_pipeline_data_construct():
    d = np.random.uniform(size=10)
    d = pipeline.PipelineData(d, fs=1e3)
    assert d.s0 == 0
    assert d.channel == None
    assert d.metadata == {}

    d = np.random.uniform(size=(2, 10))
    d = pipeline.PipelineData(d, fs=1e3)
    assert d.s0 == 0
    assert d.channel == [None, None]
    assert d.metadata == {}

    d = np.random.uniform(size=(3, 2, 10))
    d = pipeline.PipelineData(d, fs=1e3)
    assert d.s0 == 0
    assert d.channel == [None, None]
    assert d.metadata == [{}, {}, {}]


def test_pipeline_data_1d(data1d):
    fs = data1d.fs
    md = data1d.metadata.copy()

    assert data1d.channel is None
    assert data1d.metadata == md

    assert data1d[::2].fs == fs / 2
    assert data1d[::3].fs == fs / 3
    assert data1d[::2][::3].fs == fs / 2 / 3

    assert data1d[5:].s0 == 5
    assert data1d[10:].s0 == 10
    assert data1d[10:][10:].s0 == (10 + 10)

    d = data1d[5::2][10::2]
    assert d.s0 == (5 + 10)
    assert d.fs == fs / 2 / 2

    assert data1d[:1].fs == fs
    assert data1d[1:1].fs == fs
    assert data1d[:1:1].fs == fs
    assert data1d[1:1:1].fs == fs
    assert data1d[:1].s0 == 0
    assert data1d[1:1].s0 == 1
    assert data1d[:1:1].s0 == 0
    assert data1d[1:1:1].s0 == 1
    assert data1d[:1].metadata == md
    assert data1d[1:1].metadata == md
    assert data1d[:1:1].metadata == md
    assert data1d[1:1:1].metadata == md

    n = len(data1d)
    assert data1d[n:].s0 == n


def test_pipeline_data_2d(data1d, data2d):
    # TODO: add data2d[..., 0] once we add support for dimensionality
    # reduction.

    s = data2d[..., :1]
    assert s.s0 == 0
    assert s.channel == ['channel1', 'channel2']
    assert data2d[0].channel == 'channel1'
    assert data2d[1].channel == 'channel2'

    # Upcast to 1D to 2D
    fs = data1d.fs
    md = data1d.metadata.copy()
    data = data1d[np.newaxis]

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


def test_pipeline_data_3d(data1d):
    # Upcast to 2D
    md = data1d.metadata.copy()
    fs = data1d.fs
    data = data1d[np.newaxis, np.newaxis]
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


def test_pipeline_attrs_1d(data1d):
    assert data1d.n_channels == 1
    assert data1d.n_time == 100000
    assert data1d[:500].n_channels == 1
    assert data1d[:500].n_time == 500
    assert data1d[::2].n_channels == 1
    assert data1d[::2].n_time == 50000
    assert data1d[np.newaxis].n_channels == 1
    assert data1d[np.newaxis].n_time == 100000


def test_pipeline_attrs_2d(data2d):
    assert data2d.n_channels == 2
    assert data2d.n_time == 100000
    assert data2d[..., :500].n_channels == 2
    assert data2d[..., :500].n_time == 500
    assert data2d[..., ::2].n_channels == 2
    assert data2d[..., ::2].n_time == 50000
    assert data2d[np.newaxis].n_channels == 2
    assert data2d[np.newaxis].n_time == 100000

    assert data2d[0].n_channels == 1
    assert data2d[:1].n_channels == 1


def feed_pipeline(cb, data, include_offset=False):
    result = []
    o = 0
    cr = cb(result.append)
    while o < data.shape[-1]:
        try:
            i = np.random.randint(low=10, high=100)
            if include_offset:
                cr.send((o, data[..., o:o+i]))
            else:
                cr.send(data[..., o:o+i])
            o += i
        except StopIteration:
            break
    return result


@pytest.mark.parametrize('data,', ['data1d', 'data2d'])
def test_capture_epoch(fs, data, request):
    data = request.getfixturevalue(data)
    s0 = 12345
    samples = 54321

    expected = data[..., s0:s0+samples]
    cb = partial(pipeline.capture_epoch, s0, samples, {})
    result = feed_pipeline(cb, data, True)
    assert len(result) == 1

    np.testing.assert_array_equal(result[0], expected)
    assert result[0].metadata == expected.metadata
    assert result[0].s0 == s0
    assert result[0].channel == expected.channel



@pytest.mark.parametrize('data_fixture,', ['data1d', 'data2d'])
def test_extract_epochs(fs, data_fixture, request):
    if data_fixture == 'data1d':
        n_channels = 1
        expected_channels = [None]
    elif data_fixture == 'data2d':
        n_channels = 2
        expected_channels = ['channel1', 'channel2']
    data = request.getfixturevalue(data_fixture)

    queue = deque()
    epoch_size = 0.1
    epoch_samples = int(round(epoch_size * data.fs))
    cb = partial(pipeline.extract_epochs, data.fs, queue, epoch_size, 0, 0)
    queue.append({'t0': 0.1, 'metadata': {'epoch': 'A'}})
    queue.append({'t0': 0.2, 'metadata': {'epoch': 'B'}})
    queue.append({'t0': 0.3, 'metadata': {'epoch': 'C'}})
    result = pipeline.concat(feed_pipeline(cb, data), -3)
    assert result.shape == (3, n_channels, epoch_samples)
    assert result.channel == expected_channels
    expected_md = [{**data.metadata, **{'epoch': e}} for e in 'ABC']
    assert result.metadata == expected_md


@pytest.mark.parametrize('data,', ['data1d', 'data2d'])
def test_rms(fs, data, request):
    data = request.getfixturevalue(data)

    duration = 0.1
    n_samples = int(round(data.fs * duration))
    n_chunks = data.shape[-1] // n_samples

    n = n_samples * n_chunks
    expected = data[..., :n]
    expected.shape = list(expected.shape[:-1]) + [n_chunks, n_samples]
    expected_rms = np.mean(expected ** 2, axis=-1) ** 0.5

    cb = partial(pipeline.rms, data.fs, duration)
    result = feed_pipeline(cb, data)
    for r in result:
        assert r.fs == (fs / n_samples)

    actual_rms = pipeline.concat(result, axis=-1)
    np.testing.assert_array_equal(actual_rms, expected_rms)
    assert actual_rms.fs == (fs / n_samples)
    assert actual_rms.s0 == 0


@pytest.mark.parametrize('data,', ['data1d', 'data2d'])
def test_iirfilter(fs, data, stim_fl, stim_fh, request):
    # Note, do not remove `fs` from the list of arguments. This seems to be
    # necessary to allow pytest to run. Not sure why, but
    # request.getfixturevalue(data) will fail if we do not pull in the fs
    # fixture here.
    data = request.getfixturevalue(data)
    cb = partial(pipeline.iirfilter, data.fs, 1, (stim_fl, stim_fh), None,
                 None, 'band', 'butter')

    b, a = signal.iirfilter(1, (stim_fl, stim_fh), None, None, 'band',
                            ftype='butter', fs=data.fs)
    zi = signal.lfilter_zi(b, a)
    expected, _ = signal.lfilter(b, a, data, zi=zi * data[..., :1], axis=-1)
    actual = feed_pipeline(cb, data)
    actual = pipeline.concat(actual)
    assert actual.s0 == 0
    assert actual.fs == data.fs
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('data,', ['data1d', 'data2d'])
def test_blocked(fs, data, request):
    data = request.getfixturevalue(data)
    cb = partial(pipeline.blocked, 100)
    actual = feed_pipeline(cb, data)
    for i, a in enumerate(actual):
        assert a.s0 == (i * 100)
        assert a.shape[-1] == 100
    actual = pipeline.concat(actual)
    assert actual.fs == data.fs
    assert actual.s0 == 0
    np.testing.assert_array_equal(actual, data)


@pytest.fixture(scope='module', params=[None, 'constant', 'linear'])
def detrend_mode(request):
    return request.param


@pytest.mark.parametrize('data_fixture', ['data1d', 'data2d'])
def test_detrend(fs, data_fixture, detrend_mode, request):
    if data_fixture == 'data1d':
        n_channels = 1
        expected_channels = [None]
    elif data_fixture == 'data2d':
        n_channels = 2
        expected_channels = ['channel1', 'channel2']
    data = request.getfixturevalue(data_fixture)
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
        expected.append(data[..., lb:ub])
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

    assert actual.shape == (3, n_channels, epoch_samples)
    assert actual.channel == expected_channels
    expected_md = [{**data.metadata, **{'epoch': e}} for e in 'ABC']
    assert actual.metadata == expected_md


@pytest.mark.parametrize('data_fixture', ['data1d', 'data2d'])
def test_transform(fs, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    cb = partial(pipeline.transform, lambda x: x * 2)
    actual = feed_pipeline(cb, data)
    actual = pipeline.concat(actual, axis=-1)

    assert actual.shape == data.shape
    assert actual.channel == data.channel
    assert actual.metadata == data.metadata
    np.testing.assert_array_equal(data * 2, actual)


def test_mc_reference(fs, data2d):
    matrix = util.diff_matrix(2, 'all')
    cb = partial(pipeline.mc_reference, matrix)
    actual = feed_pipeline(cb, data2d)
    actual = pipeline.concat(actual, axis=-1)
    assert actual.shape == data2d.shape
    assert actual.channel == data2d.channel
    assert actual.metadata == data2d.metadata
    np.testing.assert_array_equal(matrix @ data2d, actual)


# TODO TEST:
# * mc_select
# * accumulate
# * downsample, decimate, discard, threshold, capture, delay, events_to_info,
# reject-epochs, average
