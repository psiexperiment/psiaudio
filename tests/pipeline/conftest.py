import pytest

import numpy as np

from psiaudio import pipeline


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
def tone1d(fs):
    md = {'foo': 'bar'}
    t = np.arange(100000) / fs
    data = np.sin(2*np.pi*t*1500)
    return pipeline.PipelineData(data, fs, metadata=md)


@pytest.fixture
def tone2d(fs):
    md = {'foo': 'bar'}
    channel = ['channel1', 'channel2']
    t = np.arange(100000) / fs
    data = np.vstack([
        np.sin(2*np.pi*t*1500),
        np.sin(2*np.pi*t*4500),
    ])
    return pipeline.PipelineData(data, fs, channel=channel, metadata=md)


@pytest.fixture
def data3d(fs):
    md = [{'foo': 'bar'}, {'foo': 'baz'}, {'foo': 'biz'}]
    channel = ['channel1', 'channel2']
    data = np.random.uniform(size=(3, 2, 1000))
    return pipeline.PipelineData(data, fs, channel=channel, metadata=md)
