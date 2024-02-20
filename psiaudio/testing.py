import numpy as np
import pandas as pd


def assert_pipeline_data_equal(a, b):
    np.testing.assert_array_equal(a, b)
    assert a.channel == b.channel
    assert a.metadata == b.metadata


def assert_events_equal(a, b):
    pd.testing.assert_frame_equal(a.events, b.events)
    assert a.start == b.start
    assert a.end == b.end
    assert a.fs == b.fs


def assert_pipeline_data_almost_equal(a, b, *args, **kw):
    np.testing.assert_array_almost_equal(a, b, *args, **kw)
    assert a.channel == b.channel
    assert a.metadata == b.metadata
