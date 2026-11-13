import threading

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from psiaudio.gain import GainSchedule


FS = 1000


def render(schedule, n, blocksize):
    '''
    Render `n` samples of ones through `schedule`, a block at a time.
    '''
    out = []
    for offset in range(0, n, blocksize):
        block = np.ones(min(blocksize, n - offset))
        out.append(schedule.apply(offset, block))
    return np.concatenate(out)


def test_unchanged_until_set():
    '''
    Until the gain changes, the waveform is passed through as is (bit-exact).
    '''
    schedule = GainSchedule(FS)
    x = np.random.default_rng(0).normal(size=100)
    assert schedule.apply(0, x) is x


def test_initial_gain():
    schedule = GainSchedule(FS, gain_db=-6)
    assert_allclose(schedule.apply(0, np.ones(10)), 10 ** (-6 / 20))


def test_change_starts_at_first_unrendered_sample():
    schedule = GainSchedule(FS, ramp_time=0)
    schedule.apply(0, np.ones(250))
    assert schedule.set(6) == 250
    y = schedule.apply(250, np.ones(10))
    assert_allclose(y, 10 ** (6 / 20))


def test_samples_already_rendered_keep_their_gain():
    schedule = GainSchedule(FS, ramp_time=0)
    schedule.apply(0, np.ones(100))
    schedule.set(6)
    assert_array_equal(schedule.gain_at(np.arange(100)), 0)
    assert_array_equal(schedule.gain_at(np.arange(100, 110)), 6)


@pytest.mark.parametrize('ramp_time', [0, 0.05])
def test_ramp_shape(ramp_time):
    schedule = GainSchedule(FS, ramp_time=ramp_time)
    schedule.apply(0, np.ones(100))
    schedule.set(-10)
    gain = schedule.gain_at(np.arange(100, 200))
    n = int(round(ramp_time * FS))
    if n:
        # cos² from 0 to -10 dB over the ramp, monotonic, then -10 dB.
        assert gain[0] == 0
        assert np.all(np.diff(gain[:n + 1]) <= 0)
        assert gain[n // 2] == pytest.approx(-5)
    assert_array_equal(gain[n:], -10)


@pytest.mark.parametrize('blocksize', [1, 7, 64, 1000])
def test_blocksize_independent(blocksize):
    '''
    The gain of each sample depends only on its index, not on how the stream
    is split into blocks.
    '''
    def run(blocksize):
        schedule = GainSchedule(FS, ramp_time=0.03)
        out = []
        offset = 0
        for step, gain in [(300, 3), (200, -2), (500, None)]:
            block = np.ones(step)
            for i in range(0, step, blocksize):
                b = block[i:i + blocksize]
                out.append(schedule.apply(offset, b))
                offset += len(b)
            if gain is not None:
                schedule.set(gain)
        return np.concatenate(out)
    assert_allclose(run(blocksize), run(1000))


def test_change_during_ramp_is_continuous():
    '''
    A change made before the last ramp finished starts from wherever that
    ramp had got to, so the gain does not jump.
    '''
    schedule = GainSchedule(FS, ramp_time=0.1)
    schedule.apply(0, np.ones(100))
    schedule.set(10)
    schedule.apply(100, np.ones(50))  # halfway up the ramp
    schedule.set(0)
    gain = schedule.gain_at(np.arange(100, 300))
    assert np.max(np.abs(np.diff(gain))) < 0.5
    assert gain[-1] == 0


def test_set_before_rendering_replaces_pending_change():
    schedule = GainSchedule(FS, ramp_time=0)
    schedule.apply(0, np.ones(10))
    schedule.set(3)
    schedule.set(-3)
    assert schedule.changes == [(10, 0.0, -3.0)]
    assert schedule.gain_db == -3


class TestConstantOver:

    def make(self):
        # 0 dB for samples 0-99, a 20-sample ramp to 6 dB from 100, then 6 dB.
        schedule = GainSchedule(FS, ramp_time=0.02)
        schedule.apply(0, np.ones(100))
        schedule.set(6)
        schedule.apply(100, np.ones(400))
        return schedule

    def test_before_change(self):
        assert self.make().constant_over(0, 100) == 0

    def test_spanning_change(self):
        assert self.make().constant_over(50, 150) is None

    def test_during_ramp(self):
        assert self.make().constant_over(105, 115) is None

    def test_after_ramp(self):
        assert self.make().constant_over(120, 500) == 6

    def test_not_yet_rendered(self):
        '''
        Samples not yet rendered could still change, so they are not
        counted as constant.
        '''
        assert self.make().constant_over(120, 501) is None

    def test_empty(self):
        assert self.make().constant_over(10, 10) is None


def test_threads():
    '''
    Rendering and changing the gain from different threads keeps the record
    consistent with what was rendered.
    '''
    schedule = GainSchedule(FS, ramp_time=0)
    n_blocks, blocksize = 2000, 16
    out = np.empty(n_blocks * blocksize)

    def render_all():
        for i in range(n_blocks):
            s = slice(i * blocksize, (i + 1) * blocksize)
            out[s] = schedule.apply(s.start, np.ones(blocksize))

    t = threading.Thread(target=render_all)
    t.start()
    gains = iter(range(1, 1000))
    while t.is_alive():
        schedule.set(next(gains) % 7)
    t.join()
    expected = 10 ** (schedule.gain_at(np.arange(len(out))) / 20)
    assert_allclose(out, expected)
