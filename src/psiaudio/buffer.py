import logging
log = logging.getLogger(__name__)

import threading
import numpy as np


class SignalBuffer:
    """
    Ringbuffer class that facilitates caching only what we need in memory.

    This was written primarily for continuous plotting of data. If the time
    segment includes uncached data (e.g., prior to the oldest sample in the
    cache or after the newest sample in the cache), the segment will be padded
    with `fill_value`.

    Parameters
    ----------
    fs : float
        Sampling rate of buffered signal.
    size : float
        Duration of buffer in seconds.
    fill_value : float, optional
        Value to fill missing or invalidated data with, by default np.nan.
    dtype : type, optional
        Data type of the buffer, by default np.double.
    n_channels : int, optional
        Number of channels for the buffer. If None, a 1D buffer is created.
        By default None.
    """

    def __init__(self, fs, size, fill_value=np.nan, dtype=np.double,
                 n_channels=None):
        log.debug('Creating signal buffer with fs=%f and size=%f', fs, size)
        self._lock = threading.RLock()
        self._buffer_fs = fs
        self._buffer_samples = int(np.ceil(fs*size))
        self._n_channels = n_channels
        self._size = size

        if n_channels is not None:
            shape = (self._n_channels, self._buffer_samples)
        else:
            shape = self._buffer_samples

        self._buffer = np.full(shape, fill_value, dtype=dtype)
        self._fill_value = fill_value
        self._samples = 0

        # This is an attribute that represents the current "start" of the
        # buffered data. We always "push" data into the array from the right,
        # so the newest samples will appear at the end of the array. In the
        # beginning, the "start" of the buffered data is at the very end of hte
        # array, but as we fill up the array, `_ilb` will eventually always be
        # 0.
        self._ilb = self._buffer_samples

    def resize(self, size):
        """
        Resize buffer to hold the specified number of seconds.

        Parameters
        ----------
        size : float
            New size of the buffer in seconds.

        Notes
        -----
        * This does not allow you to add/remove channels.
        * A request to decrease buffer size is ignored.
        """
        # Don't shrink buffer size. Not worth the effort.
        with self._lock:
            old_samples = self._buffer_samples
            self._buffer = self.get_latest(-size, fill_value=self._fill_value)
            self._buffer_samples = self._buffer.shape[-1]

            new_samples = self._buffer_samples
            # This corrects the lower bound index to point to the oldest data
            # in the buffer.
            self._ilb = max(0, self._ilb + (new_samples - old_samples))

    def time_to_samples(self, t):
        """
        Convert time to samples (re acquisition start).

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        int
            Number of samples.
        """
        return round(t*self._buffer_fs)

    def time_to_index(self, t):
        """
        Convert time to index in buffer.

        Note that the index may fall out of the buffered range.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        int
            Index corresponding to the time.
        """
        i = self.time_to_samples(t)
        return self.samples_to_index(i)

    def samples_to_index(self, i):
        """
        Convert absolute sample index to the index within the buffer.

        Note that the resulting index can fall outside the populated buffered range.

        Parameters
        ----------
        i : int
            Absolute sample index.

        Returns
        -------
        int
            Index in the buffer.
        """
        # Convert index to the index in the buffer. Note that the index can
        # fall outside the buffered range.
        return i - self._samples + self._buffer_samples

    def get_range_filled(self, lb, ub, fill_value):
        """
        Get buffered segment and pad missing data with `fill_value`.

        Parameters
        ----------
        lb : float
            Lower time bound in seconds.
        ub : float
            Upper time bound in seconds.
        fill_value : float
            Value used to pad regions if they fall outside the cached range.

        Returns
        -------
        ndarray
            Array slice representing the range requested.
        """
        # Index of requested range
        with self._lock:
            ilb = self.time_to_samples(lb)
            iub = self.time_to_samples(ub)
            # Index of buffered range
            slb = self.get_samples_lb()
            sub = self.get_samples_ub()
            lpadding = max(slb-ilb, 0)
            elb = max(slb, ilb)
            rpadding = max(iub-sub, 0)
            eub = min(sub, iub)
            data = self.get_range_samples(elb, eub)

            padding = (lpadding, rpadding)
            if data.ndim == 2:
                padding = ((0, 0), (lpadding, rpadding))
            else:
                padding = (lpadding, rpadding)

            return np.pad(data, padding, 'constant',
                         constant_values=fill_value)

    def get_range(self, lb=None, ub=None):
        """
        Get buffered segment corresponding to the absolute time bounds.

        Parameters
        ----------
        lb : float, optional
            Lower time bound in seconds. If None, uses the start time of the buffer.
        ub : float, optional
            Upper time bound in seconds. If None, uses the end time of the buffer.

        Returns
        -------
        ndarray
            Array slice for the requested time range.
        """
        with self._lock:
            if lb is None:
                lb = self.get_time_lb()
            if ub is None:
                ub = self.get_time_ub()
            ilb = None if lb is None else self.time_to_samples(lb)
            iub = None if ub is None else self.time_to_samples(ub)
            return self.get_range_samples(ilb, iub)

    def get_range_samples(self, lb=None, ub=None):
        """
        Get buffered segment corresponding to the absolute sample bounds.

        Parameters
        ----------
        lb : int, optional
            Lower bound in samples. If None, uses the first cached sample.
        ub : int, optional
            Upper bound in samples. If None, uses the most recent sample.

        Returns
        -------
        ndarray
            Array slice within the buffer.

        Raises
        ------
        IndexError
            If `lb` or `ub` fall completely outside the buffered segment.
        """
        with self._lock:
            if lb is None:
                lb = self.get_samples_lb()
            if ub is None:
                ub = self.get_samples_ub()
            ilb = self.samples_to_index(lb)
            iub = self.samples_to_index(ub)
            log.trace('Need range %d to %d.', ilb, iub)
            log.trace('Current lower bound is %d for %d', self._ilb,
                      self._buffer_samples)
            if ilb < self._ilb:
                raise IndexError
            elif iub > self._buffer_samples:
                raise IndexError
            return self._buffer[..., ilb:iub]

    def append_data(self, data):
        """
        Append new data into the right-side of the buffer.

        Parameters
        ----------
        data : ndarray
            The data to append. Must have shape matching `n_channels` if specified.

        Raises
        ------
        ValueError
            If dimensions or channels of appended data are mismatched.
        """
        if self._n_channels is not None:
            if data.ndim != 2:
                raise ValueError('Appended data must be two-dimensional')
            if data.shape[0] != self._n_channels:
                raise ValueError(f'Appended data must have {self._n_channels} channels.')
        else:
            if data.ndim != 1:
                raise ValueError('Appended data must be one-dimensional')

        with self._lock:
            samples = data.shape[-1]
            if samples > self._buffer_samples:
                self._buffer[..., :] = data[..., -self._buffer_samples:]
                self._ilb = 0
            else:
                self._buffer[..., :-samples] = self._buffer[..., samples:]
                self._buffer[..., -samples:] = data
                self._ilb = max(0, self._ilb - samples)
            self._samples += samples

    def _invalidate(self, i):
        # This is only called by invalidate or invalidate_samples, which are
        # already wrapped inside a lock block.
        if i <= 0:
            self._buffer[:] = self._fill_value
            self._ilb = self._buffer_samples
        else:
            self._buffer[..., -i:] = self._buffer[..., :i]
            self._buffer[..., :-i] = np.nan
            self._ilb = self._ilb + self._buffer_samples - i

    def invalidate(self, t):
        """
        Invalidate cached data after the specified time.

        Parameters
        ----------
        t : float
            Time in seconds beyond which data is invalidated.
        """
        with self._lock:
            self.invalidate_samples(self.time_to_samples(t))

    def invalidate_samples(self, i):
        """
        Invalidate cached data after the specified sample index.

        Parameters
        ----------
        i : int
            Sample index beyond which data is invalidated.
        """
        with self._lock:
            if i >= self._samples:
                return
            bi = self.samples_to_index(i)
            self._invalidate(bi)
            di = self.get_samples_ub() - i
            self._samples -= di

    def get_latest(self, lb, ub=0, fill_value=None):
        """
        Returns buffered data relative to the most recently-buffered sample.

        Parameters
        ----------
        lb : float
            Time in seconds relative to the most recently-buffered sample.
            Usually will be a negative value.
        ub : float, optional
            Time in seconds relative to the most recently-buffered sample.
            Usually will be 0 or a negative value. By default 0.
        fill_value : float, optional
            If not None, specifies the value to pad uncached array sections.
            By default None.

        Returns
        -------
        ndarray
            Array segment corresponding to the latest cached data.

        Examples
        --------
        Get the most recent 1 second of buffered data

        >>> buffer.get_latest(-1)

        Get the buffered data from -2 to -1 relative to current time.

        >>> buffer.get_latest(-2, -1)
        """
        with self._lock:
            log.trace('Converting latest %f to %f to absolute time', lb, ub)
            lb = lb + self.get_time_ub()
            ub = ub + self.get_time_ub()
            log.trace('Absolute time is %f to %f', lb, ub)
            if fill_value is None:
                return self.get_range(lb, ub)
            else:
                return self.get_range_filled(lb, ub, fill_value)

    def get_time_lb(self):
        """
        Get the lowest valid time bounding the currently cached sequence.

        Returns
        -------
        float
            Lowest lower-bound time in seconds.
        """
        return self.get_samples_lb()/self._buffer_fs

    def get_time_ub(self):
        """
        Get the highest valid time bounding the currently cached sequence.

        Returns
        -------
        float
            Highest upper-bound time in seconds.
        """
        with self._lock:
            return self.get_samples_ub()/self._buffer_fs

    def get_samples_lb(self):
        """
        Get the lower-bound index specifying the oldest cached sample.

        Returns
        -------
        int
            Lowest bounds sample index.
        """
        with self._lock:
            return self._samples - self._buffer_samples + self._ilb

    def get_samples_ub(self):
        """
        Get the upper-bound index specifying the most recent cached sample.

        Returns
        -------
        int
            Upper bounds sample index.
        """
        with self._lock:
            return self._samples


