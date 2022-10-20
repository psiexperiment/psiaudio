import logging
log = logging.getLogger(__name__)

from collections import deque
from copy import copy

import numpy as np
import pandas as pd
from scipy import signal

from . import util


################################################################################
# PipelineData
################################################################################
def normalize_index(index, ndim):
    """Expands an index into the same dimensionality as the array

    Parameters
    ----------
    index : {Ellipsis, None, slice, tuple}
        Index to normalize
    ndim : int
        The dimension of the object that is being indexed

    Returns
    -------
    norm_index : tuple
        The expanded index.
    """
    if index is np.newaxis:
        return tuple([np.newaxis] + [slice(None) for i in range(ndim)])
    if index is Ellipsis:
        return tuple(slice(None) for i in range(ndim))
    if isinstance(index, (slice, int)):
        return tuple([index] + [slice(None) for i in range(ndim - 1)])
    # If we've made it this far, we now have an indexing tuple.

    # Capture edge-case where we are dealing with `x[[n]]`.
    if isinstance(index, list):
        index = (np.s_[index],)

    n_ellipsis = sum(int(i is Ellipsis) for i in index)
    if n_ellipsis > 1:
        raise IndexError('More than one ... not supported')

    # Update for the number of new dimensions we are adding
    ndim += sum(int(i is np.newaxis) for i in index)

    norm_index = []
    for i in index:
        if isinstance(i, (slice, int, list)):
            norm_index.append(i)
        elif i is np.newaxis:
            norm_index.append(np.newaxis)
        elif i is Ellipsis:
            for _ in range(ndim - len(index) + 1):
                norm_index.append(slice(None))

    # Tack on remaining dimensions
    for _ in range(ndim - len(norm_index)):
        norm_index.append(slice(None))

    return tuple(norm_index)


class PipelineData(np.ndarray):

    @property
    def t(self):
        return np.arange(self.s0, self.s0 + self.n_time) / self.fs

    @property
    def n_time(self):
        return self.shape[-1]

    @property
    def n_channels(self):
        return 1 if self.ndim == 1 else self.shape[-2]

    @property
    def n_epochs(self):
        if self.ndim < 3:
            return None
        else:
            return self.shape[-3]

    def __new__(cls, arr, fs, s0=0, channel=None, metadata=None):
        obj = np.asarray(arr).view(cls)
        obj.fs = fs
        obj.s0 = s0

        if obj.ndim <= 2:
            if metadata is None:
                metadata = {}
        if obj.ndim > 1:
            if channel is None:
                channel = [None for i in range(obj.shape[-2])]
            elif len(channel) != obj.shape[-2]:
                raise ValueError(f'Length of channel must be {obj.shape[-2]}. Got {channel}.')
        if  obj.ndim > 2:
            if metadata is None:
                metadata = [{} for i in range(obj.shape[-3])]
            elif len(metadata) != obj.shape[-3]:
                raise ValueError(f'Length of metadata must be {obj.shape[-3]}')

        obj.channel = channel
        obj.metadata = metadata
        return obj

    def __getitem__(self, s):
        obj = super().__getitem__(s)

        # This will be the case when s is just an integer, not a slice.
        if not hasattr(obj, 'metadata'):
            return obj

        # Now, figure out the operations on our dimensions
        s = normalize_index(s, self.ndim)

        # Since np.newaxis is None, we need a different placeholder to indicate
        # that we have a no-op. We need to know whether a new axis was added so
        # that we can properly adjust the channel or metadata on the array.
        skip = object()
        if len(s) == 1:
            epoch_slice, channel_slice, (time_slice,) = skip, skip, s
        elif len(s) == 2:
            epoch_slice, (channel_slice, time_slice,) = skip, s
        elif len(s) == 3:
            epoch_slice, channel_slice, time_slice = s

        if isinstance(time_slice, int):
            # Before we implement this, we need to have some way of tracking
            # dimensionality (e.g., if ndim=1, what dimension has been
            # preserved, time, channel, etc.?).
            raise NotImplementedError
            obj.s0 += time_slice
        elif time_slice is np.newaxis:
            raise IndexError('Pipeline data cannot be recast this way')
        else:
            if time_slice.start is not None:
                if time_slice.start > 0:
                    obj.s0 += time_slice.start
                elif time_slice.start < 0:
                    obj.s0 = self.s0 + self.n_time + time_slice.start
            if time_slice.step is not None:
                obj.fs /= time_slice.step

        if channel_slice is np.newaxis:
            if not isinstance(obj.channel, list):
                obj.channel = [obj.channel]
            elif len(obj.channel) != 1:
                raise ValueError('Too many channels')
        elif channel_slice is not skip:
            if isinstance(channel_slice, list):
                obj.channel = [obj.channel[s] for s in channel_slice]
            elif isinstance(channel_slice, (int, slice)):
                obj.channel = obj.channel[channel_slice]
            else:
                raise ValueError(f'Unrecognized channel slice {channel_slice}')

        if epoch_slice is np.newaxis:
            if not isinstance(obj.metadata, list):
                obj.metadata = [obj.metadata]
            elif len(obj.metadata) != 1:
                raise ValueError('Too many entries for metadata')
        elif epoch_slice is not skip:
            if isinstance(epoch_slice, list):
                obj.metadata = [obj.metadata[s] for s in epoch_slice]
            elif isinstance(epoch_slice, (int, slice)):
                obj.metadata = obj.metadata[epoch_slice]
            else:
                raise ValueError(f'Unrecognized epoch slice {epoch_slice}')

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.fs = getattr(obj, 'fs', None)
        self.s0 = getattr(obj, 's0', None)
        self.metadata = copy(getattr(obj, 'metadata', {}))
        self.channel = copy(getattr(obj, 'channel', None))
        if self.channel is None and self.ndim > 1:
            self.channel = [None for i in range(self.shape[-2])]
        # TODO: Something about metadata?

    def mean(self, axis=None, *args, **kwargs):
        dim, axis = dim_axis(axis)
        if axis == -1:
            n = self.shape[-1]
            result = super().mean(axis, *args, **kwargs)
            result.fs /= n
            result.s0 /= n
            return result
        elif axis == -2:
            result = super().mean(axis, *args, **kwargs)
            result.channel = 'average'
            return result
        elif axis == -3:
            result = super().mean(axis, *args, **kwargs)
            result.metadata = None
            return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        result = f'Pipeline > s: {self.s0} to {self.s0+self.n_time}, ' \
            f'fs: {self.fs}, channel: {self.channel}, shape: {self.shape}'
        return result

    def add_metadata(self, key, value):
        if self.ndim > 2:
            for md in self.metadata:
                md[key] = value
        else:
            self.metadata[key] = value


def ensure_dim(arrays, dim):
    ndim = arrays[0].ndim
    if dim == 'channel' and ndim == 1:
        s = np.s_[np.newaxis, :]
    elif dim == 'epoch' and ndim == 1:
        s = np.s_[np.newaxis, np.newaxis, :]
    elif dim == 'epoch' and ndim == 2:
        s = np.s_[np.newaxis, :, :]
    else:
        s = np.s_[:]
    return [a[s] for a in arrays]


def dim_axis(axis):
    if axis == 'time':
        axis, dim = -1, 'time'
    if axis == 'channel':
        axis, dim = -2, 'channel'
    if axis == 'epoch':
        axis, dim = -3, 'epoch'
    elif axis == -1:
        dim = 'time'
    elif axis == -2:
        dim = 'channel'
    elif axis == -3:
        dim = 'epoch'
    else:
        raise ValueError(f'Axis not supported. Got {axis}')
    return dim, axis


def concat(arrays, axis=-1):
    dim, axis = dim_axis(axis)

    is_pipeline_data = [isinstance(a, PipelineData) for a in arrays]
    if not any(is_pipeline_data):
        return np.concatenate(arrays, axis=axis)
    if not all(is_pipeline_data):
        raise ValueError('Cannot concatenate pipeline and non-pipeline data')

    arrays = ensure_dim(arrays, dim)

    # Do consistency checks to ensure we can properly concatenate
    base_arr = arrays[0]
    for a in arrays[1:]:
        if a.ndim != base_arr.ndim:
            raise ValueError('Cannot concatenate inputs with different ndim')

    # First, make sure sampling rates match. We simply cannot deal with
    # variable sampling rates no matter what concat dimension we have.
    fs = base_arr.fs
    for a in arrays[1:]:
        if a.fs != base_arr.fs:
            raise ValueError('Cannot concatenate inputs with different sampling rates')

    # If we are concatenating across time, we need to make sure that we have
    # contiguous chunks of data.
    s0 = base_arr.s0
    if dim == 'time':
        current_s0 = s0 + arrays[0].shape[-1]
        for a in arrays[1:]:
            if a.s0 != current_s0:
                raise ValueError(f'first sample of each array is not aligned (expected {current_s0}, found {a.s0})')
            current_s0 += a.shape[-1]

    if dim != 'channel':
        channel = base_arr.channel
        for a in arrays[1:]:
            if a.channel != channel:
                raise ValueError('Cannot concatenate inputs with different channel')
    else:
        channel = [c for array in arrays for c in array.channel]

    if dim != 'epoch':
        metadata = base_arr.metadata
        for a in arrays[1:]:
            if a.metadata != metadata:
                raise ValueError(f'Cannot concatenate inputs with different metadata. Expected {metadata}. Found {a.metadata}.')
    else:
        metadata = []
        for a in arrays:
            if a.ndim >= 3:
                metadata.extend(a.metadata)
            else:
                metadata.append(a.metadata)

    result = np.concatenate(arrays, axis=axis)
    return PipelineData(result, fs=fs, s0=s0, channel=channel, metadata=metadata)


################################################################################
# Events
################################################################################
class Events:
    '''
    Collection of events occuring over a given span
    '''

    def __init__(self, events, start, end, fs=None):
        '''
        Parameters
        ----------
        events : list of tuples
            Each tuple must have two elements. The first is the event name and
            the second is the sample time of the event.
        start : int
            Starting sample of the detection range (inclusive)
        end : int
            Ending sample of the detection rnage (exclusive)
        fs : float
            Sampling rate of data. Optional. If not provided, `get_range` and
            `get_latest` will not work.
        '''
        self.events = pd.DataFrame(events, columns=['event', 'sample'])
        self.events['ts'] = self.events['sample'] / fs
        self.start = start
        self.end = end
        self.fs = fs

    def __str__(self):
        return f'Events (n={len(self.events)} between {self.start} and {self.end}, fs={self.fs})'

    def __repr__(self):
        return f'<{self}>'

    def get_range_samples(self, start, end):
        if (start < self.start) or (end > self.end):
            raise ValueError('Invalid range')
        m = (self.events['sample'] >= start) & (self.events['sample'] < end)
        return Events(self.events[m], start, end, self.fs)

    def get_range(self, lb, ub):
        '''
        Return a new `Events` instance containing the subset of events occuring
        within the given range (in seconds).

        Parameters
        ----------
        lb : float
            Starting time of range
        ub : float
            Ending time time of range (exclusive)
        '''
        if self.fs is None:
            raise ValueError('Unknown sampling rate. Use `get_range_samples`.')
        start = int(np.round(lb * self.fs))
        end = int(np.round(ub * self.fs))
        return self.get_range_samples(start, end)

    def get_latest_samples(self, lb, ub=0):
        start = lb + self.end
        end = ub + self.end
        return self.get_range_samples(start, end)

    def get_latest(self, lb, ub=0):
        '''
        Return a new `Events` instance containing the subset of events occuring
        within the given range (in seconds). Here, `lb` and `ub` are specified
        relative to the ending time of the block and should be negative.

        Parameters
        ----------
        lb : float
            Starting time of range
        ub : float
            Ending time time of range (exclusive)
        '''
        if self.fs is None:
            raise ValueError('Unknown sampling rate. Use `get_latest_samples`.')
        lb = int(np.round(lb * self.fs))
        ub = int(np.round(ub * self.fs))
        return self.get_latest_samples(lb, ub)

    @property
    def range_samples(self):
        return self.end - self.start

    @property
    def t0(self):
        return self.start / self.fs

    def rate(self):
        return len(self.events) / self.range_samples * self.fs


def combine_events(events):
    s0 = events[0].end
    for ed in events[1:]:
        if ed.start != s0:
            raise ValueError(f'Times of each Events collection are not aligned (expected {s0}, found {ed.start})')
        s0 = ed.end
        if ed.fs != events[0].fs:
            raise ValueError('Cannot concatenate Events with different sampling rates')

    return Events(
        pd.concat(ed.events for ed in events),
        events[0].start,
        events[-1].end,
        events[0].fs
    )


################################################################################
# Generic
################################################################################
def coroutine(func):
    '''Decorator to auto-start a coroutine.'''
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


@coroutine
def broadcast(*targets):
    '''
    Send the data to multiple targets
    '''
    while True:
        data = (yield)
        for target in targets:
            target(data)


@coroutine
def transform(function, target):
    '''
    Apply function to data and send return value to next target
    '''
    while True:
        data = (yield)
        target(function(data))


################################################################################
# Continuous data pipeline
################################################################################
@coroutine
def rms(fs, duration, target):
    n = int(round(fs * duration))
    data = [(yield)]
    samples = sum(d.shape[-1] for d in data)

    while True:
        if samples >= n:
            data = concat(data, axis=-1)
            n_blocks = data.shape[-1] // n
            n_samples = n_blocks * n

            shape = list(data.shape[:-1]) + [n_blocks, n]
            d = data[..., :n_samples]
            d.shape = shape
            result = np.mean(d ** 2, axis=-1) ** 0.5

            target(result)
            d = data[..., n_samples:]
            samples = d.shape[-1]
            data = [d]

        data.append((yield))
        samples += data[-1].shape[-1]


@coroutine
def iirfilter(fs, N, Wn, rp, rs, btype, ftype, target):
    b, a = signal.iirfilter(N, Wn, rp, rs, btype, ftype=ftype, fs=fs)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError('Unstable filter coefficients')

    # Initialize the state of the filter and scale it by y[0] to avoid a
    # transient.
    zi = signal.lfilter_zi(b, a)
    y = (yield)
    zo = zi * y[..., :1]

    while True:
        y_filt, zo = signal.lfilter(b, a, y, zi=zo, axis=-1)
        if isinstance(y, PipelineData):
            y_filt = PipelineData(y_filt, y.fs, y.s0)
        target(y_filt)
        y = (yield)


@coroutine
def blocked(block_size, target):
    data = []
    n = 0

    while True:
        d = (yield)
        if d is Ellipsis:
            data = []
            target(d)
            continue

        n += d.shape[-1]
        data.append(d)
        if n >= block_size:
            merged = concat(data, axis=-1)
            while merged.shape[-1] >= block_size:
                block = merged[..., :block_size]
                target(block)
                merged = merged[..., block_size:]
            data = [merged]
            n = merged.shape[-1]


@coroutine
def capture_epoch(epoch_s0, epoch_samples, info, target, fs=None,
                  auto_send=False):
    '''
    Coroutine to facilitate capture of a single epoch

    This was written as a supporting function for `extract_epochs` (i.e., it
    creates one `capture_epoch` for each epoch it is looking for), but can also
    be used stand-alone to capture single epochs.

    Parameters
    ----------
    epoch_s0 : float
        Starting sample of epoch.
    epoch_samples : int
        Number of samples to capture.
    info : dict
        Dictionary of metadata that will be passed along to downstream
        coroutines (i.e., the callback).
    target : callable
        Callable that receives a single argument. The argument will be an
        instance of PipelineData with three dimensions (epoch, channel, time).
    auto_send : bool
        If true, automatically send samples as they are acquired.
    '''
    # This coroutine will continue until it acquires all the samples it needs.
    # It then provides the samples to the callback function and exits the while
    # loop.
    accumulated_data = []
    current_s0 = epoch_s0

    # Info may be shared among multiple pipeline components. Make a copy so
    # that we don't accidentally affect other pipeline components.
    info = info.copy()
    md = info.pop('metadata', {})

    while True:
        slb, data = (yield)
        samples = data.shape[-1]

        if current_s0 < slb:
            # We have missed the start of the epoch. Notify the callback of this
            m = 'Missed samples for epoch of %d samples starting at %d'
            log.warning(m, epoch_samples, epoch_s0)
            target(PipelineData([], fs=fs, s0=epoch_s0, metadata=md))
            break

        elif current_s0 <= (slb + samples):
            # The start of the epoch is somewhere inside `data`. Find the start
            # `i` and determine how many samples `d` to extract from `data`.
            # It's possible that data does not contain the entire epoch. In
            # that case, we just pull out what we can and save it in
            # `accumulated_data`. We then update start to point to the last
            # acquired sample `i+d` and update duration to be the number of
            # samples we still need to capture.
            i = int(round(current_s0 - slb))
            d = int(round(min(epoch_samples, samples - i)))
            c = data[..., i:i + d]

            # TODO: Not in love with this approach. I don't like the idea of
            # squashing md and info into existing metadata, but I don't want to
            # add additional attributes to PipelineData.
            if hasattr(c, 'metadata'):
                c.metadata.update(md)
                c.metadata.update(info)

            accumulated_data.append(c)
            current_s0 += d
            epoch_samples -= d

            if auto_send:
                # TODO: Not tested
                accumulated_data = concat(accumulated_data, axis=-1)
                target(accumulated_data)
                accumulated_data = []
                if epoch_samples == 0:
                    break

            elif epoch_samples == 0:
                data = concat(accumulated_data, axis=-1)
                target(data)
                break


@coroutine
def extract_epochs(fs, queue, epoch_size, target, buffer_size=0,
                   empty_queue_cb=None, removed_queue=None, prestim_time=0,
                   poststim_time=0):
    '''
    Coroutine to facilitate extracting epochs from an incoming stream of data


    Parameters
    ----------
    fs : float
        Sampling rate of input stream. Used to convert parameters specified in
        seconds to number of samples.
    queue : deque
        Instance of the collections.deque class containing information about
        the epochs to extract. Must be a dictionary containing at least the
        `t0` key (indicating the starting time, in seconds, of the epoch). The
        `duration` key (indicating epoch duration, in seconds) is mandatory if
        the `epoch_size` parameter is None. Optional keys include `key` (a
        unique identifier for that epoch) and `metadata` (attributes that will
        be attached to the epoch).
    epoch_size : {None, float}
        Size of epoch to extract, in seconds. If None, than the dictionaries
        (provided via the queue) must contain a `duration` key.
    buffer_size : float
        Duration of samples to buffer in memory. If you anticipate needing to
        "look back" and capture some epochs after the samples have already been
        acquired, this value should be greater than 0.
    target : callable
        Callable that receives a list of epochs that were extracted. This is
        typically another coroutine in the pipeline.
    empty_queue_cb : {None, callable}
        Callback function taking no arguments. Called when there are no more
        epochs pending for capture and the queue is empty.
    removed_queue : deque
        Instance of the collections.deque class. Each entry in the queue must
        contain at least the `t0` key and the `key` (if originally provided via
        `queue`). If the epoch has not been fully captured yet, this epoch will
        be removed from the list of epochs to capture.
    prestim_time : float
        Additional time to capture before the specified epoch start.
    poststim_time : float
        Additional time to capture beyond the specified epoch size (or
        `duration`).
    '''
    # The variable `tlb` tracks the number of samples that have been acquired
    # and reflects the lower bound of `data`. For example, if we have acquired
    # 300,000 samples, then the next chunk of data received from (yield) will
    # start at sample 300,000 (remember that Python is zero-based indexing, so
    # the first sample has an index of 0).
    tlb = 0

    # This tracks the epochs that we are looking for. The key will be a
    # two-element tuple. key[0] is the starting time of the epoch to capture
    # and key[1] is a universally unique identifier. The key may be None, in
    # which case, you will not have the ability to capture two different epochs
    # that begin at the exact same time.
    epoch_coroutines = {}

    # Maintain a buffer of prior samples that can be used to retroactively
    # capture the start of an epoch if needed.
    prior_samples = []

    # How much historical data to keep (for retroactively capturing epochs)
    buffer_samples = round(buffer_size * fs)

    # Since we may capture very short, rapidly occurring epochs (at, say,
    # 80 per second), I find it best to accumulate as many epochs as possible before
    # calling the next target. This list will maintain the accumulated set.
    epochs = []

    # This is used for communicating events
    if removed_queue is None:
        removed_queue = deque()

    while True:
        # Wait for new data to become available
        data = (yield)
        prior_samples.append((tlb, data))

        # First, check to see what needs to be removed from epoch_coroutines.
        # If it doesn't exist, it may already have been captured.
        skip = []
        n_remove = 0
        n_pop = 0
        while removed_queue:
            info = removed_queue.popleft()

            # This is a unique identifier.
            key = info['t0'], info.get('key', None)
            if key not in epoch_coroutines:
                n_remove += 1
                skip.append(key)
            else:
                epoch_coroutines.pop(key)
                n_pop += 1

        if n_remove or n_pop:
            log.debug('Marked %d epochs for removal, removed %d epochs', n_remove, n_pop)

        # Send the data to each coroutine. If a StopIteration occurs,
        # this means that the epoch has successfully been acquired and has
        # been sent to the callback and we can remove it. Need to operate on
        # a copy of list since it's bad form to modify a list in-place.
        for key, epoch_coroutine in list(epoch_coroutines.items()):
            try:
                epoch_coroutine.send((tlb, data))
            except StopIteration:
                epoch_coroutines.pop(key)

        # Check to see if more epochs have been requested. Information will be
        # provided in seconds, but we need to convert this to number of
        # samples.
        n_queued = 0
        n_invalid = 0
        while queue:
            info = queue.popleft()
            key = info['t0'], info.get('key', None)
            if key in skip:
                skip.remove(key)
                n_invalid += 1
                continue
            n_queued += 1

            # Figure out how many samples to capture for that epoch
            info['prestim_time'] = prestim_time
            info['poststim_time'] = poststim_time
            info['epoch_size'] = epoch_size if epoch_size else info['duration']
            total_epoch_size = info['epoch_size'] + poststim_time + prestim_time
            epoch_samples = round(total_epoch_size * fs)
            t0 = round((info['t0'] - prestim_time) * fs)
            epoch_coroutine = capture_epoch(t0, epoch_samples, info,
                                            epochs.append, fs)

            try:
                # Go through the data we've been caching to facilitate
                # historical acquisition of data. If this completes without a
                # StopIteration, then we have not finished capturing the full
                # epoch.
                for prior_sample in prior_samples:
                    epoch_coroutine.send(prior_sample)
                if key in epoch_coroutines:
                    raise ValueError(f'Duplicate epochs not supported. Got {key}.')
                epoch_coroutines[key] = epoch_coroutine
            except StopIteration:
                pass

        if n_queued or n_invalid:
            log.debug('Queued %d epochs, %d were invalid', n_queued, n_invalid)

        tlb = tlb + data.shape[-1]

        # Once the new segment of data has been processed, pass all complete
        # epochs along to the next target.
        if len(epochs) != 0:
            if isinstance(epochs[0], PipelineData):
                merged = concat(epochs, axis=-3)
            else:
                merged = np.concatenate([e[np.newaxis] for e in epochs], axis=0)
            target(merged)
            epochs[:] = []

        # Check to see if any of the cached samples are older than the
        # specified buffer_samples and discard them.
        while True:
            oldest_samples = prior_samples[0]
            tub = oldest_samples[0] + oldest_samples[1].shape[-1]
            if tub < (tlb - buffer_samples):
                prior_samples.pop(0)
            else:
                break

        if not (queue or epoch_coroutines) and empty_queue_cb:
            # If queue and epoch coroutines are complete, call queue callback
            # once and only once.
            empty_queue_cb()
            empty_queue_cb = None


@coroutine
def accumulate(n, axis, newaxis, status_cb, target):
    data = []
    while True:
        d = (yield)
        if d is Ellipsis:
            data = []
            target(d)
            continue

        if newaxis:
            data.append(d[np.newaxis])
        else:
            data.append(d)
        if len(data) == n:
            data = concatenate(data, axis=axis)
            target(data)
            data = []

        if status_cb is not None:
            status_cb(len(data))


@coroutine
def downsample(q, target):
    y_remainder = None

    while True:
        if y_remainder is None:
            y = (yield)
        else:
            y_new = (yield)
            y = concat((y_remainder, y_new), axis=-1)

        remainder = y.shape[-1] % q
        if remainder != 0:
            y_remainder = y[..., -remainder:]
            y = y[..., :-remainder]
        else:
            y_remainder = None

        result = y[..., ::q]
        if len(result):
            target(result)


@coroutine
def decimate(q, target):
    b, a = signal.cheby1(4, 0.05, 0.8 / q)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError('Unstable filter coefficients')

    y = (yield)
    s0 = getattr(y, 's0', 0)

    # Create the initial state for the filter (we call it zf since we will just
    # repeatedly use this on each iteration).
    zf = signal.lfilter_zi(b, a)
    if y.ndim == 2:
        zf = zf[np.newaxis]

    while True:
        remainder = y.shape[-1] % q

        y_filt, zf = signal.lfilter(b, a, y, zi=zf, axis=-1)
        if isinstance(y, PipelineData):
            y_filt = PipelineData(y_filt, y.fs, y.s0, y.channel, y.metadata)

        if remainder != 0:
            y_remainder = y_filt[..., -remainder:]
            y_filt = y_filt[..., :-remainder]
        else:
            y_remainder = None

        result = y_filt[..., ::q]
        if isinstance(result, PipelineData):
            result.s0 = s0
        s0 += result.shape[-1]

        if result.shape[-1] > 0:
            target(result)

        if y_remainder is not None:
            y = concat((y_remainder, (yield)), axis=-1)
        else:
            y = (yield)


@coroutine
def discard(discard_samples, cb):
    '''
    Discard samples.

    discard_samples : int
        Number of samples to discard.
    '''
    to_discard = discard_samples
    while True:
        samples = (yield)
        if samples is Ellipsis:
            # Restart the pipeline
            to_discard = discard_samples
            cb(samples)
            continue

        if to_discard == 0:
            cb(samples)
        elif samples.shape[-1] <= to_discard:
            to_discard -= samples.shape[-1]
        elif samples.shape[-1] > to_discard:
            samples = samples[..., to_discard:]
            to_discard = 0
            cb(samples)


@coroutine
def capture(fs, queue, target):
    s0 = 0
    t_start = None  # Time, in seconds, of capture start
    s_next = None  # Sample number for capture

    while True:
        # Wait for new data to come in
        data = (yield)
        try:
            # We've recieved a new command. The command will either be None
            # (i.e., no more acquisition for a bit) or a floating-point value
            # (indicating when next acquisition should begin).
            info = queue.popleft()
            if info is not None:
                t_start = info['t0']
                s_next = round(t_start * fs)
                target(Ellipsis)
                log.error('Starting capture at %f', t_start)
            elif info is None:
                log.debug('Ending capture')
                s_next = None
            else:
                raise ValueError('Unsupported queue input %r', info)
        except IndexError:
            pass

        if (s_next is not None) and (s_next >= s0):
            i = s_next - s0
            if i < data.shape[-1]:
                d = data[..., i:]
                d.metadata['capture'] = t_start
                target(d)
                s_next += d.shape[-1]

        s0 += data.shape[-1]


@coroutine
def delay(n, target):
    data = np.full(n, np.nan)
    while True:
        target(data)
        data = (yield)


@coroutine
def edges(min_samples, target, initial_state=False, fs='auto', detect='both'):
    '''
    Find the rising and falling edges of a binary (boolean) input. The output
    is an instance of `Events`. Even if no edges are detected, an `Events`
    instance (containing zero events) is generated. This enables downstream
    steps to continue updating even if no events are detected.

    Parameters
    ----------
    initial_state : {int, bool}
        Initial state of the system. Required to ensure that we properly detect
        changes that may occur as soon as the pipeline begins.
    min_samples : int
        Minimum number of samples required before a change is registered. This
        is effectively a means of "debouncing" the signal.
    fs : {'auto', float}
        If `'auto'`, this must be part of a pipeline that processes
        `PipelineData` as fs will be derived from the first `PipelineData`
        segment.
    detect : {'both', 'rising', 'falling'}
        Edge to detect.

    Notes
    -----
    Incoming data is cast to boolean dtype. If the data is not already boolean,
    then the data must have been transformed such that the logical low has a
    value of 0. All non-zero values will be interpreted as a logical high.
    '''
    if min_samples < 1:
        raise ValueError('min_samples must be >= 1')


    # Get timestamp of first chunk
    new_samples = (yield).astype('bool')
    prior_samples = np.tile(initial_state, min_samples).astype('bool')

    if isinstance(new_samples, PipelineData):
        prior_samples = PipelineData(prior_samples,
                                     s0=new_samples.s0-min_samples,
                                     fs=new_samples.fs)
        s0 = prior_samples.s0
        fs = prior_samples.fs
    else:
        s0 = -min_samples
        fs = None

    while True:
        # Wait for new data to become available
        if new_samples.ndim == 1:
            pass
        elif (new_samples.ndim == 2) and (new_samples.shape[0] == 1):
            new_samples = new_samples[0]
        else:
            raise ValueError('Cannot handle N-dimensional data')

        n_samples = new_samples.shape[-1]
        samples = concat((prior_samples, new_samples), axis=-1)
        epochs = util.debounce_epochs(util.epochs(samples), min_samples)

        events = []
        for lb, ub in epochs:
            if detect in ('rising', 'both') and lb > 0:
                events.append(('rising', lb + s0))
            if detect in ('falling', 'both') and ub < samples.shape[-1]:
                events.append(('falling', ub + s0))

        events = Events(events, s0, s0 + n_samples, fs)
        target(events)

        s0 += new_samples.shape[-1]
        prior_samples = samples[..., -min_samples:]

        new_samples = (yield).astype('bool')


@coroutine
def average(n, target):
    data = (yield)
    axis = 0
    while True:
        while data.shape[axis] >= n:
            s = [Ellipsis] * data.ndim
            s[axis] = np.s_[:block_size]
            target(data[s].mean(axis=axis))
            s[axis] = np.s_[block_size:]
            data = data[s]
        new_data = (yield)
        data = np.concatenate((data, new_data), axis=axis)


@coroutine
def auto_th(n, baseline, target, fs='auto'):
    '''
    Automatically determine threshold based on input data standard deviation

    Parameters
    ----------
    n : int
        Number of standard deviations to fix threshold at.
    baseline : float
        Duration, in seconds, to use for calculating baseline standard
        deviation.
    fs : {'auto', float}
        Sampling rate of data. If `'auto'`, this coroutine must be part of a
        pipeline that is processing `PipelineData` as `fs` will be derived from
        the first `PipelineData` segment.

    Once the threshold has been determined, the portion of the input data
    collected for determining the threshold will then be thresholded and passed
    to the next stage in the pipeline.
    '''
    data = (yield)

    # Now that we have our first chunk of data, calculate some basics.
    if fs is None:
        fs = data.fs
    baseline_samples = int(np.round(baseline * fs))

    # Accumulate data until we have spooled enough to do our baseline estimate.
    # All downstream pipeline steps will appear to be unresponsive until this
    # step completes.
    while data.shape[-1] < baseline_samples:
        data = concat((data, (yield)), axis=-1)

    # Now, discard the pre-baseline samples.
    d = data[..., :baseline_samples].view(np.ndarray)

    th = np.std(d) * n
    log.info('auto_th set to %f', th)

    # Immediately send the data accumulated for the baseline (plus any extra
    # data that was captured), then wait for the next chunk of data.
    while True:
        target(data >= th)
        data = (yield)


################################################################################
# Events data
################################################################################
@coroutine
def event_rate(block_size, block_step, target, s0_mode='center'):
    '''
    Calculate the rate at which events occur using a sliding temporal window.

    Parameters
    ----------
    block_size : int
        Size of window, in samples, to calculate event rate over.
    block_step : float
        Increment, in samples, to advance window before calculating next event
        rate.
    s0_mode : {'center', 'left', 'right'}
        Controls how the initial s0 of the resulting PipelineData object is
        determined. If `'center'`, s0 will reflect the center of the block over
        which the event rate is calculated. If `'left'` or `'right'`, s0 will
        reflect either the beginning or end of the block, respectively, over
        which the event rate is calculated.

    This coroutine must be part of a pipeline in which the previous stage
    outputs an `Events` instance. As `Events` instances are acquired, they are
    merged and then split into discrete blocks of `block_size`. The starting
    time of each block is advanced by `block_step`. If `block_step` is less
    than `block_size`, then the blocks will overlap. This allows us to
    effectively apply a sliding window approach to calculating event rate over
    time.
    '''
    events = (yield)
    s0 = events.start + block_size * 0.5
    fs = events.fs / block_step

    while True:
        events = combine_events((events, (yield)))
        blocks = []
        while events.range_samples > block_size:
            block = events.get_range_samples(
                events.start,
                events.start + block_size
            )
            blocks.append(block)
            events = events.get_range_samples(
                events.start + block_step,
                events.end
            )
        if blocks:
            rate = [b.rate() for b in blocks]

            # In general, psiepxeriment prefers 2D data (even for singleton
            # channels), so upcast.
            data = PipelineData([rate], s0=s0, fs=fs)
            target(data)
            s0 += len(rate)


################################################################################
# Multichannel continuous data
################################################################################
@coroutine
def mc_reference(matrix, target):
    while True:
        data = matrix @ (yield)
        target(data)


@coroutine
def mc_select(channel, labels, target):
    if isinstance(channel, int):
        i = channel
    elif labels is not None:
        i = labels.index(channel)
    else:
        raise ValueError(f'Unsupported channel: {channel}')

    while True:
        data = (yield)
        if data.ndim != 2:
            raise ValueError('Input must be channel x time')
        target(data[i])


################################################################################
# Epoch pipelines
################################################################################
@coroutine
def detrend(mode, target):
    while True:
        data = (yield)
        if isinstance(data, PipelineData) and data.ndim != 3:
            raise ValueError('Cannot detrend')
        if mode is None:
            target(data)
        else:
            data_detrend = signal.detrend(data, axis=-1, type=mode)
            if isinstance(data, PipelineData):
                data_detrend = PipelineData(data_detrend, data.fs, data.s0,
                                            data.channel, data.metadata)
            target(data_detrend)


@coroutine
def events_to_info(trigger_edge, base_info, target):
    while True:
        events = (yield)
        results = []
        for e, ts in events:
            if e == trigger_edge:
                info = base_info.copy()
                info['t0'] = ts
                results.append(info)
        target(results)


@coroutine
def reject_epochs(reject_threshold, mode, status_cb, valid_target):
    '''
    Discard epochs containing artifacts

    Parameters
    ----------
    reject_threshold : {float, callable}
        If callble, this will be called each time this coroutine is invoked
        allowing threshold to be updated during long-running experiments.
    mode : {'absolute value', 'amplitude'}
        Specifies reject method:
        * 'absolute value': rejected if max(abs(signal)) > reject_threshold
        * 'amplitude': rejected if (max(s) - min(s)) > reject_threshold
    status_cb : {None, callable}
        If provided, must be a callback that takes two arguments, the number of
        epochs reviewed and the number of epochs rejected on that particular
        invocation of the callback.
    valid_target : {coroutine, callable}
        Target to send accepted (i.e., not rejected) epochs to.
    '''
    if mode == 'absolute value':
        accept = lambda s, th: np.max(np.abs(s), axis=-1) < th
    elif mode == 'amplitude':
        accept = lambda s, th: np.ptp(s, axis=-1) < th

    # This reduces the likelihood of name collisions (i.e., if the calling code
    # provides a lambda that uses a variable name, it's very unlikely to be
    # __th_cb__).
    if not callable(reject_threshold):
        __th_cb__ = lambda: reject_threshold
    else:
        __th_cb__ = reject_threshold

    while True:
        data = (yield)
        if isinstance(data, PipelineData):
            if data.n_channels != 1:
                raise ValueError('Not supported for multichannel data')
            if data.n_epochs is None:
                raise ValueError('Not supported for un-epoched data')
        else:
            if data.ndim != 3:
                raise ValueError('Must have 3 dimensions (epoch, channel, time)')
            elif data.shape[1] != 1:
                raise ValueError('Only one channel supported')

        # Check for valid epochs and send them if there are any
        th = __th_cb__()
        mask = accept(np.asarray(data), th)[:, 0]
        valid_data = data[mask]
        if isinstance(valid_data, PipelineData):
            valid_data.add_metadata('reject_threshold', th)

        n = len(data)
        n_accept = len(valid_data)

        if n_accept != 0:
            valid_target(valid_data)
        if status_cb is not None:
            status_cb(n, n_accept)
