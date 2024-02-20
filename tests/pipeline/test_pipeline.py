import pytest

from collections import deque
from functools import partial

import numpy as np
import pandas as pd
from scipy import signal

from psiaudio import pipeline
from psiaudio import util

from psiaudio.testing import (
    assert_events_equal,
    assert_pipeline_data_equal,
    assert_pipeline_data_almost_equal,
)


################################################################################
# utility functions and fixtures
################################################################################
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


################################################################################
# unit tests
################################################################################
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


@pytest.mark.parametrize('data,', ['tone1d', 'tone2d'])
def test_rms_band(fs, data, request):
    data = request.getfixturevalue(data)

    b, a = signal.iirfilter(4, [1000, 2000], btype='band', ftype='butter',
                            fs=fs)

    duration = 0.5
    n_samples = int(round(data.fs * duration))
    n_chunks = data.shape[-1] // n_samples

    n = n_samples * n_chunks
    expected = signal.filtfilt(b, a, data[..., :n], axis=-1)
    expected.shape = list(expected.shape[:-1]) + [n_chunks, n_samples]
    expected_rms = np.mean(expected ** 2, axis=-1) ** 0.5


    cb = partial(pipeline.rms_band, data.fs, 1000, 2000, duration)
    result = feed_pipeline(cb, data)
    for r in result:
        assert r.fs == (fs / n_samples)

    actual_rms = pipeline.concat(result, axis=-1)
    np.testing.assert_array_almost_equal(actual_rms, expected_rms, 2)
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
    def status_cb(n):
        print(f'{n} valid')

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
        expected.add_metadata('reject_threshold', 2)
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
        expected.add_metadata('reject_threshold', 2)
        assert_pipeline_data_equal(actual, expected)


@pytest.mark.parametrize('data_fixture', ['data1d'])
def test_reject_epochs_update_th(fs, data_fixture, detrend_mode, request):
    data = request.getfixturevalue(data_fixture)

    # Lambda functions will pull the value reject_threshold from the namespace.
    th_cb = lambda: th

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
                    th_cb,
                    'amplitude',
                    None,
                    target
                ).send
            )

    queue, expected, epoch_size = queue_epochs(data)

    segments = []
    actual = []
    cr = cb(target=actual.append)
    o = 0

    thresholds = [2, 4, 2]
    for e, th in zip(expected, thresholds):
        # Now, add an artifact value of 3 to each.
        s0 = int(round(e.metadata['t0'] * e.fs))
        e0 = s0 + e.n_time
        i = np.random.randint(s0, e0)
        data[i] = 3
        cr.send(data[o:e0])
        o = e0
    cr.send(data[o:])

    # We need to re-run the standard epoch queue using the mutated data (where
    # we added the artifacts) so that the resulting epoch waveforms are
    # identical to the ones used for the artifact reject.
    _, expected, _ = queue_epochs(data)
    expected = expected[[1]]
    expected.add_metadata('reject_threshold', 4)
    actual = pipeline.concat(actual, axis=-3)
    assert_pipeline_data_equal(actual, expected)


def test_edges():
    # Supporting function to handle segmenting PipelineData into 10 chunks and
    # then feeding each chunk into the pipeline (useful for verifying that we
    # are properly addressing what we might expect to see in a full experiment
    # pipeline where data is incrementially obtained from the acquisitoin
    # process).
    def _test_pipeline(d, expected, debounce=2, initial_state=0,
                       detect='both'):
        actual = []
        pipe = pipeline.edges(min_samples=debounce, target=actual.append,
                              initial_state=initial_state, detect=detect)
        for i in range(4):
            lb = i * 100
            ub = lb + 100
            pipe.send(d[..., lb:ub])
        for a, e in zip(actual, expected):
            try:
                assert_events_equal(a, e)
            except:
                print(a, e)
                raise

    # Check what happens when we have no changes over the duration of the
    # signal (all zeros).
    d = np.zeros((1, 400))
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    expected = [
        pipeline.Events([], -2, 98, 1000),
        pipeline.Events([], 98, 198, 1000),
        pipeline.Events([], 198, 298, 1000),
        pipeline.Events([], 298, 398, 1000),
    ]
    _test_pipeline(d, expected)
    expected[0] = pipeline.Events([('falling', 0)], -2, 98, 1000)
    _test_pipeline(d, expected, debounce=2, initial_state=1)

    # Check what happens when we have no changes over the duration of the
    # signal (all ones).
    d = np.ones((1, 400))
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    expected = [
        pipeline.Events([('rising', 0)], -2, 98, 1000),
        pipeline.Events([], 98, 198, 1000),
        pipeline.Events([], 198, 298, 1000),
        pipeline.Events([], 298, 398, 1000),
    ]
    _test_pipeline(d, expected)
    expected[0] = pipeline.Events([], -2, 98, 1000)
    _test_pipeline(d, expected, initial_state=1)

    # Check change in first 10 samples.
    d = np.zeros((1, 400))
    d[:, :10] = 1
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    expected = [
        pipeline.Events([('rising', 0), ('falling', 10)], -2, 98, 1000),
        pipeline.Events([], 98, 198, 1000),
        pipeline.Events([], 198, 298, 1000),
        pipeline.Events([], 298, 398, 1000),
    ]
    _test_pipeline(d, expected)

    # Now, check proper edge is reported.
    expected[0] = pipeline.Events([('falling', 10)], -2, 98, 1000)
    _test_pipeline(d, expected, detect='falling')
    expected[0] = pipeline.Events([('rising', 0)], -2, 98, 1000)
    _test_pipeline(d, expected, detect='rising')

    # Check change in 10 samples bracketing a boundary condition.
    d = np.zeros((1, 400))
    d[:, 95:105] = 1
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    expected = [
        pipeline.Events([('rising', 95)], -2, 98, 1000),
        pipeline.Events([('falling', 105)], 98, 198, 1000),
        pipeline.Events([], 198, 298, 1000),
        pipeline.Events([], 298, 398, 1000),
    ]
    _test_pipeline(d, expected)

    # Check debounce in 10 samples bracketing a boundary condition. Verify what
    # happens when it brackets a block external to the pipeline.
    d = np.zeros((1, 400))
    d[:, 96:105] = 1
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    expected = [
        pipeline.Events([], -10,  90, 1000),
        pipeline.Events([],  90, 190, 1000),
        pipeline.Events([], 190, 290, 1000),
        pipeline.Events([], 290, 390, 1000),
    ]
    _test_pipeline(d, expected, debounce=10)

    # Check what happens if it brackets the block internal to the pipeline
    d = np.zeros((1, 400))
    d[:, 86:95] = 1
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    _test_pipeline(d, expected, debounce=10)

    # Verify debounce correctly ignores both stretches.
    d = np.zeros((1, 400))
    d[:, 87:95] = 1
    d[:, 97:105] = 1
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    _test_pipeline(d, expected, debounce=10)

    # Verify edge detection correctly handles rising/falling edges of TTL that
    # lasts over multiple segments.
    d = np.zeros((1, 400))
    d[:, 80:105] = 1
    d[:, 115:200] = 1
    expected = [
        pipeline.Events([('rising', 80)], -10,  90, 1000),
        pipeline.Events([],  90, 190, 1000),
        pipeline.Events([('falling', 200)], 190, 290, 1000),
        pipeline.Events([], 290, 390, 1000),
    ]
    d = pipeline.PipelineData(d, s0=0, fs=1000)
    _test_pipeline(d, expected, debounce=10)

    # Verify that if the initial chunk does nto have a s0 of 0, it will
    # properly report the *times* of the segments.
    expected = [
        pipeline.Events([('rising', 87)], -3,  97, 1000),
        pipeline.Events([],  97, 197, 1000),
        pipeline.Events([('falling', 207)], 197, 297, 1000),
        pipeline.Events([], 297, 397, 1000),
    ]
    d = pipeline.PipelineData(d, s0=7, fs=1000)
    _test_pipeline(d, expected, debounce=10)


def test_event_rate():
    events = [
        pipeline.Events(
            [
                ('rising', 10),
                ('rising', 30),
                ('rising', 80),
            ], 0, 100, 1000
        ),
        pipeline.Events(
            [
                ('rising', 190),
            ], 100, 200, 1000
        ),
        pipeline.Events(
            [
            ], 200, 300, 1000
        ),
        pipeline.Events(
            [
                ('rising', 310),
                ('rising', 330),
                ('rising', 380),
            ], 300, 400, 1000
        ),
    ]
    rate = []
    pipe = pipeline.event_rate(block_size=50, block_step=25,
                               target=rate.append)
    for event in events:
        pipe.send(event)
    expected = pipeline.PipelineData(
        [[40, 20, 20, 20, 0, 0, 20, 20, 0, 0, 0, 20, 40, 20]],
        s0=12.5, fs=4)
    assert_pipeline_data_equal(pipeline.concat(rate), expected)


@pytest.mark.parametrize('data_fixture', ['data1d', 'data2d'])
def test_discard(fs, data_fixture, request):
    data = request.getfixturevalue(data_fixture)

    def cb(target):
        return pipeline.discard(15, target)

    actual = pipeline.concat(feed_pipeline(cb, data), axis=-1)
    assert_pipeline_data_equal(actual, data[..., 15:])


@pytest.mark.parametrize('data_fixture', ['data1d', 'data2d'])
def test_derivative(fs, data_fixture, request):
    data = request.getfixturevalue(data_fixture)

    def cb(target):
        return pipeline.derivative(0, target)

    actual = pipeline.concat(feed_pipeline(cb, data), axis=-1)
    pad_shape = ((0, 0), (1, 0)) if data_fixture == 'data2d' else (1, 0)
    padded_data = np.pad(data, pad_shape, 'constant', constant_values=0)
    expected = pipeline.PipelineData(
        np.diff(padded_data) * data.fs,
        s0=1,
        fs=data.fs,
        channel=data.channel,
        metadata=data.metadata,
    )

    assert_pipeline_data_equal(actual, expected)


# TODO TEST:
# * mc_select
# * accumulate
# * downsample, decimate, discard, threshold, capture, delay, events_to_info,
# average
