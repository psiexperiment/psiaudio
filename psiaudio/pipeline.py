import logging
log = logging.getLogger(__name__)

from collections import deque

import numpy as np
from scipy import signal


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

    n_ellipsis = sum(int(i is Ellipsis) for i in index)
    if n_ellipsis > 1:
        raise IndexError('More than one ... not supported')

    # Update for the number of new dimensions we are adding
    ndim += sum(int(i is np.newaxis) for i in index)

    norm_index = []
    for i in index:
        if isinstance(i, (slice, int)):
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
                raise ValueError(f'Length of channel must be {obj.shape[-2]}')
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
            # preserved, time, channel, etc.?
            raise NotImplementedError
            obj.s0 += time_slice
        else:
            if time_slice.start is not None:
                obj.s0 += time_slice.start
            if time_slice.step is not None:
                obj.fs /= time_slice.step

        if channel_slice is np.newaxis:
            if not isinstance(obj.channel, list):
                obj.channel = [obj.channel]
            elif len(obj.channel) != 1:
                raise ValueError('Too many channels')
        elif channel_slice is not skip:
            obj.channel = obj.channel[channel_slice]

        if epoch_slice is np.newaxis:
            if not isinstance(obj.metadata, list):
                obj.metadata = [obj.metadata]
            elif len(obj.metadata) != 1:
                raise ValueError('Too many entries for metadata')
        elif epoch_slice is not skip:
            obj.metadata = obj.metadata[epoch_slice]

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.fs = getattr(obj, 'fs', None)
        self.s0 = getattr(obj, 's0', None)
        self.metadata = getattr(obj, 'metadata', {}).copy()

        if getattr(obj, 'channel', None) is not None:
            self.channel = getattr(obj, 'channel').copy()
        elif self.ndim > 1:
            self.channel = [None for i in range(self.shape[-2])]
        else:
            self.channel = None

        if getattr(obj, 'epochs', None) is not None:
            self.epochs = getattr(obj, 'epochs').copy()
        elif self.ndim > 2:
            self.epochs = [{} for e in range(self.shape[-3])]
        else:
            self.epochs = None


    def mean(self, axis=None, *args, **kwargs):
        if axis != -1:
            raise NotImplementedError('Cannot average along other axes yet')
        n = self.shape[-1]
        result = super().mean(axis, *args, **kwargs)
        result.fs /= n
        result.s0 /= n
        return result

    def __repr__(self):
        result = f'Pipeline > s0: {self.s0}, fs: {self.fs}, shape: {self.shape}'
        return result

    def __str__(self):
        result = f'Pipeline > s0: {self.s0}, fs: {self.fs}, shape: {self.shape}'
        return result


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


def concat(arrays, axis=-1):
    is_pipeline_data = [isinstance(a, PipelineData) for a in arrays]
    if not any(is_pipeline_data):
        return np.concatenate(arrays, axis=axis)
    if not all(is_pipeline_data):
        raise ValueError('Cannot concatenate pipeline and non-pipeline data')

    if axis == -1:
        dim = 'time'
    elif axis == -2:
        dim = 'channel'
    elif axis == -3:
        dim = 'epoch'
    else:
        raise ValueError('Axis not supported')

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
                raise ValueError('Cannot concatenate inputs with different metadata')
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
def capture_epoch(epoch_s0, epoch_samples, info, callback, fs=None,
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
    callback : callable
        Callable that receives a single argument. The argument will be a
        dictionary with two keys (`signal` and `info`) where `signal` is the
        epoch and `info` is information regarding the epoch.
    auto_send : bool
        If true, automatically send samples as they are acquired.
    '''
    # This coroutine will continue until it acquires all the samples it needs.
    # It then provides the samples to the callback function and exits the while
    # loop.
    accumulated_data = []
    current_s0 = epoch_s0
    print(info)
    md = info.pop('metadata', {})

    while True:
        slb, data = (yield)
        samples = data.shape[-1]

        if current_s0 < slb:
            # We have missed the start of the epoch. Notify the callback of this
            m = 'Missed samples for epoch of %d samples starting at %d'
            log.warning(m, epoch_samples, epoch_s0)
            callback(PipelineData([], fs=fs, s0=epoch_s0, metadata=md))
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
            c.metadata.update(md)
            accumulated_data.append(c)
            current_s0 += d
            epoch_samples -= d

            if auto_send:
                # TODO: Not tested
                accumulated_data = concat(accumulated_data, axis=-1)
                callback(accumulated_data)
                accumulated_data = []
                if epoch_samples == 0:
                    break

            elif epoch_samples == 0:
                data = concat(accumulated_data, axis=-1)
                callback(data)
                break


@coroutine
def extract_epochs(fs, queue, epoch_size, poststim_time, buffer_size, target,
                   empty_queue_cb=None, removed_queue=None):
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
    poststim_time : float
        Additional time to capture beyond the specified epoch size (or
        `duration`).
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

            # This is a uinique
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
            t0 = round(info['t0'] * fs)
            info['poststim_time'] = poststim_time
            info['epoch_size'] = epoch_size if epoch_size else info['duration']
            total_epoch_size = info['epoch_size'] + poststim_time
            epoch_samples = round(total_epoch_size * fs)
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
                    raise ValueError('Duplicate epochs not supported')
                epoch_coroutines[key] = epoch_coroutine
            except StopIteration:
                pass

        if n_queued or n_invalid:
            log.debug('Queued %d epochs, %d were invalid', n_queued, n_invalid)

        tlb = tlb + data.shape[-1]

        # Once the new segment of data has been processed, pass all complete
        # epochs along to the next target.
        if len(epochs) != 0:
            target(concat(epochs[:], axis=-3))
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
            # If queue and epoch coroutines are complete, call queue callback.
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
    y_remainder = np.array([])
    while True:
        y = np.r_[y_remainder, (yield)]
        remainder = len(y) % q
        if remainder != 0:
            y, y_remainder = y[:-remainder], y[-remainder:]
        else:
            y_remainder = np.array([])
        result = y[::q]
        if len(result):
            target(result)


@coroutine
def decimate(q, target):
    b, a = signal.cheby1(4, 0.05, 0.8 / q)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError('Unstable filter coefficients')
    zf = signal.lfilter_zi(b, a)
    y_remainder = np.array([])
    while True:
        y = np.r_[y_remainder, (yield)]
        remainder = len(y) % q
        if remainder != 0:
            y, y_remainder = y[:-remainder], y[-remainder:]
        else:
            y_remainder = np.array([])
        y, zf = signal.lfilter(b, a, y, zi=zf)
        result = y[::q]
        if len(result):
            target(result)


@coroutine
def discard(discard_samples, cb):
    to_discard = discard_samples
    while True:
        samples = (yield)
        if samples is Ellipsis:
            # Restart the pipeline
            to_discard = discard_samples
            cb(samples)
            continue

        samples.metadata['discarded'] = discard_samples
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
                d = data[i:]
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
def edges(initial_state, min_samples, fs, target):
    if min_samples < 1:
        raise ValueError('min_samples must be >= 1')
    prior_samples = np.tile(initial_state, min_samples)
    t_prior = -min_samples
    while True:
        # Wait for new data to become available
        new_samples = (yield)
        samples = np.r_[prior_samples, new_samples]
        ts_change = np.flatnonzero(np.diff(samples, axis=-1)) + 1
        ts_change = np.r_[ts_change, samples.shape[-1]]

        events = []
        for tlb, tub in zip(ts_change[:-1], ts_change[1:]):
            if (tub - tlb) >= min_samples:
                if initial_state == samples[tlb]:
                    continue
                edge = 'rising' if samples[tlb] == 1 else 'falling'
                initial_state = samples[tlb]
                ts = t_prior + tlb
                events.append((edge, ts / fs))
        if events:
            target(events)
        t_prior += new_samples.shape[-1]
        prior_samples = samples[..., -min_samples:]


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
def reject_epochs(reject_threshold, mode, status, valid_target):
    if mode == 'absolute value':
        accept = lambda s: np.max(np.abs(s)) < reject_threshold
    elif mode == 'amplitude':
        accept = lambda s: np.ptp(s) < reject_threshold

    while True:
        epochs = (yield)
        # Check for valid epochs and send them if there are any
        valid = [e for e in epochs if accept(e)]
        if len(valid):
            valid_target(valid)

        def update():
            # Update the status. Must be wrapped in a deferred call to ensure
            # that the update occurs on the GUI thread.
            status.total += len(epochs)
            status.rejects += len(epochs) - len(valid)
            status.reject_percent = status.rejects / status.total * 100

        deferred_call(update)
