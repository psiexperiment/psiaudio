import logging
log = logging.getLogger(__name__)

import pytest

from collections import deque
from functools import partial

import numpy as np
from scipy import signal

from psiaudio import pipeline
from psiaudio.pipeline import normalize_index
from psiaudio import util


################################################################################
# utility functions and fixtures
################################################################################
def assert_pipeline_data_equal(a, b):
    np.testing.assert_array_equal(a, b)
    assert a.channel == b.channel
    assert a.metadata == b.metadata


def assert_pipeline_data_almost_equal(a, b, *args, **kw):
    np.testing.assert_array_almost_equal(a, b, *args, **kw)
    assert a.channel == b.channel
    assert a.metadata == b.metadata


def queue_epochs(data):
    fs = data.fs
    epoch_size = 0.1
    epoch_samples = int(round(epoch_size * fs))

    queue = deque()
    expected = []
    expected_md = []
    for i, e in enumerate('ABC'):
        t0 = (i + 1) / 10
        queue.append({'t0': t0, 'metadata': {'epoch': e}})
        lb = int(round(t0 * fs))
        ub = lb + epoch_samples
        expected.append(data[..., lb:ub])
        new_md = {'epoch': e, 't0': t0, 'poststim_time': 0, 'epoch_size': 0.1,
                  'prestim_time': 0}
        expected_md.append({**data.metadata, **new_md})
    expected = pipeline.concat(expected, axis=-3)
    expected.metadata = expected_md
    return queue, expected, epoch_size


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


@pytest.fixture
def data3d(fs):
    md = [{'foo': 'bar'}, {'foo': 'baz'}, {'foo': 'biz'}]
    channel = ['channel1', 'channel2']
    data = np.random.uniform(size=(3, 2, 1000))
    return pipeline.PipelineData(data, fs, channel=channel, metadata=md)


################################################################################
# unit tests
################################################################################
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


def test_pipeline_data_astype():
    data = np.random.uniform(size=(2, 400))
    data_th = (data > 0.5).astype('int')
    pd = pipeline.PipelineData(data, channel=['A', 'B'], fs=1e3)
    pd_th = pipeline.PipelineData(data_th, channel=['A', 'B'], fs=1e3)
    assert_pipeline_data_equal((pd > 0.5).astype('int'), pd_th)
    assert_pipeline_data_equal((pd > 0.5), pd_th.astype('bool'))


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

    # Make sure negative slicing works
    assert data1d[-n:].n_time == n
    assert data1d[-n:].s0 == 0
    assert data1d[-10:].n_time == 10
    assert data1d[-10:].s0 == (n-10)
    assert_pipeline_data_equal(data1d[n-10:], data1d[-10:])

    assert data1d[10:][-10:].s0 == (n-10)


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
    assert data.metadata == md

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

    # Test to make sure that list-like preservation of channel is preserved if
    # we pull out a single channel but preserve dimensionality.
    assert data2d[[0]].channel == ['channel1']
    assert data2d[[0]].ndim == 2
    assert data2d[[0, 1]].channel == ['channel1', 'channel2']
    assert data2d[[0, 1]].ndim == 2

    n = data2d.n_time
    assert data2d[..., n:].s0 == n

    # Make sure negative slicing works
    assert data2d[..., -n:].n_time == n
    assert data2d[..., -n:].s0 == 0
    assert data2d[..., -10:].n_time == 10
    assert data2d[..., -10:].s0 == (n-10)
    assert_pipeline_data_equal(data2d[..., n-10:], data2d[..., -10:])

    assert data2d[..., 10:][..., -10:].s0 == (n-10)


def test_pipeline_data_3d(data1d, data3d):
    assert data3d[[1]].metadata == [data3d.metadata[1]]
    assert data3d[1].metadata == data3d.metadata[1]
    assert data3d[[0, 2]].metadata == [data3d.metadata[0], data3d.metadata[2]]
    assert data3d[[0, 2], [0]].channel == [data3d.channel[0]]

    # Upcast to 3D
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
    cr = cb(target=result.append)
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

    queue, expected, epoch_size = queue_epochs(data)
    epoch_samples = int(round(epoch_size * data.fs))
    cb = partial(pipeline.extract_epochs, fs=data.fs, queue=queue,
                 epoch_size=epoch_size, buffer_size=0)

    actual = pipeline.concat(feed_pipeline(cb, data), -3)
    assert_pipeline_data_equal(actual, expected)


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

    queue, expected, epoch_size = queue_epochs(data)
    epoch_samples = int(round(data.fs * epoch_size))

    if detrend_mode is not None:
        expected[:] = signal.detrend(expected, axis=-1, type=detrend_mode)

    def cb(target):
        nonlocal data
        nonlocal detrend_mode
        nonlocal queue
        nonlocal epoch_size

        return \
            pipeline.extract_epochs(
                fs=data.fs,
                queue=queue,
                epoch_size=epoch_size,
                target=pipeline.detrend(
                    detrend_mode,
                    target
                ).send
            )

    actual = pipeline.concat(feed_pipeline(cb, data), -3)
    assert_pipeline_data_almost_equal(actual, expected)


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


def test_basic_reject_epochs():
    def status_cb(n, v):
        print(f'{v} of {n} valid')

    result = []
    p = pipeline.reject_epochs(1, 'amplitude', status_cb, result.extend)

    for i in range(10):
        s = np.zeros((1, 1, 100))
        p.send(s)

    assert len(result) == 10
    for i in range(10):
        s = np.zeros((1, 1, 100))
        s[..., 0] = 2
        p.send(s)
    assert len(result) == 10

    s = np.zeros((10, 1, 100))
    p.send(s)
    assert len(result) == 20

    s = np.zeros((10, 1, 100))
    s[::2, :, 0] = 2
    p.send(s)
    assert len(result) == 25


@pytest.mark.parametrize('data_fixture', ['data1d', 'data2d'])
def test_reject_epochs(fs, data_fixture, detrend_mode, request):
    data = request.getfixturevalue(data_fixture)

    def cb(target):
        nonlocal data
        nonlocal queue
        nonlocal epoch_size

        return \
            pipeline.extract_epochs(
                fs=data.fs,
                queue=queue,
                epoch_size=epoch_size,
                target=pipeline.reject_epochs(
                    2,
                    'amplitude',
                    None,
                    target
                ).send
            )

    s0 = int(round(0.21 * fs))
    if data_fixture == 'data1d':
        data[s0] = 3
        queue, expected, epoch_size = queue_epochs(data)
        actual = pipeline.concat(feed_pipeline(cb, data), axis=-3)
        assert_pipeline_data_equal(actual, expected[[0, 2]])
    elif data_fixture == 'data2d':
        data[1, s0] = 3
        queue, expected, epoch_size = queue_epochs(data)
        with pytest.raises(ValueError):
            actual = pipeline.concat(feed_pipeline(cb, data), axis=-3)
        queue, expected, epoch_size = queue_epochs(data)
        actual = pipeline.concat(feed_pipeline(cb, data[1]), axis=-3)
        # Avoid weird numpy indexing behavior
        expected = expected[[0, 2]][:, [1]]
        assert_pipeline_data_equal(actual, expected)


# TODO TEST:
# * mc_select
# * accumulate
# * downsample, decimate, discard, threshold, capture, delay, events_to_info,
# average
