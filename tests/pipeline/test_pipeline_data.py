import pytest
import numpy as np

from psiaudio import pipeline
from psiaudio.pipeline import normalize_index
from psiaudio.testing import assert_pipeline_data_equal


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


def test_negative_s0():
    md = {'foo': 'bar'}
    d = np.random.uniform(size=100000)
    data = pipeline.PipelineData(d, 100e3, metadata=md, s0=-64)
    assert data[::8].s0 == -64
    assert data[64:].s0 == 0
    assert data[32:].s0 == -32


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


def test_pipeline_data_3d_slice(data3d):
    assert data3d[0].metadata == data3d.metadata[0]
    assert data3d[0].metadata == data3d.metadata[0]
    mask = [True, False, False]
    assert data3d[mask].metadata == [data3d.metadata[0]]
    mask = np.array([True, False, False])
    assert data3d[mask].metadata == [data3d.metadata[0]]
    mask = [True, False, True]
    assert data3d[mask].metadata == [data3d.metadata[0], data3d.metadata[2]]
    mask = np.array([True, False, True])
    assert data3d[mask].metadata == [data3d.metadata[0], data3d.metadata[2]]


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


