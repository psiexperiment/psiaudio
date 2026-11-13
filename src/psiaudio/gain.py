'''
Gain changed while a stream of samples is playing.

`GainSchedule` scales a waveform that is rendered a block at a time (e.g., an
output topping up a hardware buffer) by a gain that can be changed while it
plays, and keeps a record of the gain each sample was rendered with. A
feedback loop uses that record to judge only the samples played at a single
gain (see `GainSchedule.constant_over`).

Samples are counted from the start of the stream (e.g., the hardware's sample
clock), and a change always starts at the first sample not yet rendered: the
gain of a sample already rendered never changes, so the record matches what
was played.
'''
import threading

import numpy as np


class GainSchedule:
    '''
    Gain (dB) applied to a stream of samples, changed as it plays.

    Each change ramps from the gain at the sample where it starts to the new
    gain, with a cos² shape over `ramp_time`, so it does not click.

    Parameters
    ----------
    fs : float
        Sampling rate of the stream (Hz).
    ramp_time : float
        Duration of the ramp to a new gain (s). 0 changes it in one step.
    gain_db : float
        Gain until the first change (dB).

    Notes
    -----
    `apply` (rendering) and `set` (changing the gain) may be called from
    different threads.
    '''
    def __init__(self, fs, ramp_time=0.1, gain_db=0):
        if ramp_time < 0:
            raise ValueError('ramp_time must not be negative')
        self.fs = fs
        self.ramp_samples = int(round(ramp_time * fs))
        self.initial_db = gain_db
        # Each change is (first sample, gain at that sample, new gain). Their
        # first samples never decrease.
        self._changes = []
        # First sample not yet rendered.
        self._rendered = 0
        self._lock = threading.Lock()

    @property
    def rendered(self):
        '''
        First sample not yet rendered: the gain of every sample before it is
        final.
        '''
        with self._lock:
            return self._rendered

    @property
    def changes(self):
        '''
        The changes so far, as (first sample, gain from, gain to) in dB.
        '''
        with self._lock:
            return list(self._changes)

    @property
    def gain_db(self):
        '''
        Gain (dB) the stream ends on once every scheduled ramp is done.
        '''
        with self._lock:
            return self._changes[-1][2] if self._changes else self.initial_db

    def _gain_db(self, samples):
        samples = np.asarray(samples)
        gain = np.full(samples.shape, float(self.initial_db))
        for start, from_db, to_db in self._changes:
            # Later changes override earlier ones from their first sample on
            # (a change may cut short the ramp of the one before it).
            m = samples >= start
            if not m.any():
                break
            if self.ramp_samples == 0:
                gain[m] = to_db
            else:
                frac = np.clip((samples[m] - start) / self.ramp_samples, 0, 1)
                w = np.sin(np.pi / 2 * frac) ** 2
                gain[m] = from_db + (to_db - from_db) * w
        return gain

    def gain_at(self, samples):
        '''
        Gain (dB) at each sample in `samples`.
        '''
        with self._lock:
            return self._gain_db(samples)

    def set(self, gain_db):
        '''
        Change the gain to `gain_db` (dB), starting at the first sample not
        yet rendered.

        Returns
        -------
        start : int
            The first sample of the change (the start of its ramp).
        '''
        with self._lock:
            start = self._rendered
            # Changes that have not started yet are replaced (first, so that
            # the gain they would have set is not taken as the gain at
            # `start`).
            self._changes = [c for c in self._changes if c[0] < start]
            # The ramp starts from wherever the gain is at `start`, which is
            # partway through a ramp if the last change hasn't finished.
            from_db = float(self._gain_db([start])[0])
            self._changes.append((start, from_db, float(gain_db)))
            return start

    def apply(self, offset, waveform):
        '''
        Scale `waveform`, whose first sample is sample `offset` of the stream,
        by the gain at each of its samples.

        Until the gain is first changed, the waveform is returned unchanged
        (not multiplied by 1), so it is bit-identical to the unscaled one.
        '''
        n = waveform.shape[-1]
        with self._lock:
            self._rendered = max(self._rendered, offset + n)
            if not self._changes and self.initial_db == 0:
                return waveform
            gain = self._gain_db(np.arange(offset, offset + n))
        return waveform * 10 ** (gain / 20)

    def constant_over(self, start, end):
        '''
        The gain (dB) if every sample from `start` up to (not including) `end`
        was rendered at that one gain, else None.

        None also if any of those samples has not been rendered yet, since its
        gain could still change.
        '''
        start = max(int(start), 0)
        end = int(end)
        with self._lock:
            if end > self._rendered:
                return None
            if end <= start:
                return None
            gain = self._gain_db([start, end - 1])
            if gain[0] != gain[1]:
                return None
            # Constant only if no change starts within (start, end) and any
            # change starting at or before `start` has finished its ramp.
            for c_start, from_db, to_db in self._changes:
                if start < c_start < end:
                    return None
                if c_start <= start < c_start + self.ramp_samples \
                        and from_db != to_db:
                    return None
            return float(gain[0])
