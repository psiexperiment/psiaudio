import pytest

from collections import Counter, deque

import numpy as np
import time

from psiaudio.calibration import FlatCalibration
from psiaudio.pipeline import extract_epochs
from psiaudio.queue import FIFOSignalQueue, InterleavedFIFOSignalQueue
from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory

rate = 76.0
isi = np.round(1 / rate, 5)


def make_tone(fs, frequency=250, duration=5e-3):
    calibration = FlatCalibration.as_attenuation()
    tone = ToneFactory(fs=fs, level=0, frequency=frequency,
                       calibration=calibration)
    return Cos2EnvelopeFactory(fs=fs, start_time=0, rise_time=0.5e-3,
                               duration=duration, input_factory=tone)


def make_queue(fs, ordering, frequencies, trials, duration=5e-3, isi=isi):
    if ordering == 'FIFO':
        queue = FIFOSignalQueue(fs=fs)
    elif ordering == 'interleaved':
        queue = InterleavedFIFOSignalQueue(fs=fs)
    else:
        raise ValueError(f'Unrecognized queue ordering {ordering}')

    conn = deque()
    queue.connect(conn.append, 'added')
    removed_conn = deque()
    queue.connect(removed_conn.append, 'removed')

    keys = []
    tones = []
    for frequency in frequencies:
        t = make_tone(fs, frequency=frequency, duration=duration)
        delay = max(isi - duration, 0)
        md = {'frequency': frequency}
        k = queue.append(t, trials, delay, metadata=md)
        keys.append(k)
        tones.append(t)

    return queue, conn, removed_conn, keys, tones


def test_long_tone_queue(fs):
    queue, conn, rem_conn, _, _ = \
        make_queue(fs, 'interleaved', [1e3, 5e3], 5, duration=1, isi=1)

    waveforms = []
    n_pop = round(fs * 0.25)
    for i in range(16):
        w = queue.pop_buffer(n_pop)
        waveforms.append(w)

    waveforms = np.concatenate(waveforms, axis=-1)
    waveforms.shape = 4, -1
    assert waveforms.shape == (4, round(fs))
    assert np.all(waveforms[0] == waveforms[2])
    assert np.all(waveforms[1] == waveforms[3])
    assert np.any(waveforms[0] != waveforms[1])


def test_fifo_queue_pause_with_requeue(fs):
    # Helper function to track number of remaining keys
    def _adjust_remaining(k1, k2, n):
        nk1 = min(k1, n)
        nk2 = min(n - nk1, k2)
        return k1 - nk1, k2 - nk2

    queue, conn, rem_conn, (k1, k2), (t1, t2) = \
        make_queue(fs, 'FIFO', [1e3, 5e3], 100)
    extractor_conn = deque()
    extractor_rem_conn = deque()
    queue.connect(extractor_conn.append, 'added')
    queue.connect(extractor_rem_conn.append, 'removed')

    # Generate the waveform template
    n_t1 = t1.n_samples_remaining()
    n_t2 = t2.n_samples_remaining()
    t1_waveform = t1.next(n_t1)
    t2_waveform = t2.next(n_t2)

    waveforms = []
    extractor = extract_epochs(fs=fs,
                               queue=extractor_conn,
                               removed_queue=extractor_rem_conn,
                               poststim_time=0,
                               buffer_size=0,
                               epoch_size=15e-3,
                               target=waveforms.extend)
    time.sleep(0.1)

    # Track number of trials remaining
    k1_left, k2_left = 100, 100
    samples = int(round(fs))

    # Since the queue uses the delay (between offset and onset of
    # consecutive segments), we need to calculate the actual ISI since it
    # may have been rounded to the nearest sample.
    delay_samples = round((isi - t1.duration) * fs)
    duration_samples = round(t1.duration * fs)
    total_samples = duration_samples + delay_samples
    actual_isi = total_samples / fs

    ###########################################################################
    # First, queue up 2 seconds worth of trials
    ###########################################################################
    waveform = queue.pop_buffer(samples * 2)
    n_queued = np.floor(2 / actual_isi) + 1
    t1_lb = 0
    t2_lb = 100 * total_samples
    t2_lb = int(t2_lb)
    assert np.all(waveform[t1_lb:t1_lb + duration_samples] == t1_waveform)
    assert np.all(waveform[t2_lb:t2_lb + duration_samples] == t2_waveform)

    time.sleep(0.1)
    assert len(conn) == np.ceil(2 / actual_isi)
    assert len(rem_conn) == 0
    keys = [i['key'] for i in conn]
    assert set(keys) == {k1, k2}
    assert set(keys[:100]) == {k1}
    assert set(keys[100:]) == {k2}

    k1_left, k2_left = _adjust_remaining(k1_left, k2_left, n_queued)
    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    conn.clear()

    ###########################################################################
    # Now, pause
    ###########################################################################
    # Pausing should remove all epochs queued up after 0.5s. After sending
    # the first waveform to the extractor, we generate a new waveform to
    # verify that no additional trials are queued and send that to the
    # extractor.
    queue.pause(round(0.5 * fs) / fs)
    extractor.send(waveform[:round(0.5 * fs)])

    # We need to add 1 to account for the very first trial.
    n_queued = int(np.floor(2 / actual_isi)) + 1
    n_kept = int(np.floor(0.5 / actual_isi)) + 1

    # Now, fix the counters
    k1_left, k2_left = _adjust_remaining(100, 100, n_kept)

    # This is the total number that were removed when we paused.
    n_removed = n_queued - n_kept

    # Subtract 1 because we haven't fully captured the last trial that
    # remains in the queue because the epoch_size was chosen such that the
    # end of the epoch to be extracted is after 0.5s.
    n_captured = n_kept - 1
    assert len(waveforms) == n_captured

    # Doing this will capture the final epoch.
    waveform = queue.pop_buffer(samples)
    assert np.all(waveform == 0)
    extractor.send(waveform)
    assert len(waveforms) == (n_captured + 1)

    # Verify removal event is properly notifying the timestamp
    rem_t0 = np.array([i['t0'] for i in rem_conn])
    assert np.all(rem_t0 >= 0.5)
    assert (rem_t0[0] % actual_isi) == pytest.approx(0, 0.1 / fs)

    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    assert len(conn) == 0
    assert len(rem_conn) == n_removed

    rem_count = Counter(i['key'] for i in rem_conn)
    assert rem_count[k1] == 100 - n_kept
    assert rem_count[k2] == n_queued - 100
    conn.clear()
    rem_conn.clear()

    queue.resume(samples * 1.5 / fs)

    waveform = queue.pop_buffer(samples)
    n_queued = np.floor(1 / actual_isi) + 1
    k1_left, k2_left = _adjust_remaining(k1_left, k2_left, n_queued)

    extractor.send(waveform)
    time.sleep(0.1)

    assert len(conn) == np.floor(1 / actual_isi) + 1
    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    assert len(conn) == np.floor(1 / actual_isi) + 1
    keys += [i['key'] for i in conn]
    conn.clear()

    waveform = queue.pop_buffer(5 * samples)
    n_queued = np.floor(5 / actual_isi) + 1
    k1_left, k2_left = _adjust_remaining(k1_left, k2_left, n_queued)

    extractor.send(waveform)
    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    keys += [i['key'] for i in conn]

    # We requeued 1.5 second worth of trials so need to factor this because
    # keys (from conn) did not remove the removed keys.
    assert len(keys) == (200 + n_removed)

    # However, the extractor is smart enough to handle cancel appropriately
    # and should only have the 200 we originally intended.
    assert len(waveforms) == 200

    # This should capture the 1-sample bug that sometimes occurs when using
    # int() instead of round() with quirky sample rates (e.g., like with the
    # RZ6).
    n = len(t1_waveform)
    waveforms = np.vstack(waveforms)
    t1_waveforms = waveforms[:100]
    t2_waveforms = waveforms[100:]

    assert np.all(t1_waveforms[:, :n] == t1_waveform)
    assert np.all(t2_waveforms[:, :n] == t2_waveform)


def test_queue_isi_with_pause(fs):
    """
    Verifies that queue generates samples at the expected ISI and also verifies
    pause functionality works as expected.
    """
    queue, conn, _, _, (t1,) = make_queue(fs, 'FIFO', [250], 500)
    duration = 1
    samples = round(duration * fs)
    queue.pop_buffer(samples)
    time.sleep(0.1)
    expected_n = int(duration / isi) + 1
    assert len(conn) == expected_n

    # Pause is after `duration` seconds
    queue.pause()
    waveform = queue.pop_buffer(samples)
    assert np.sum(waveform ** 2) == 0
    assert len(conn) == int(duration / isi) + 1

    # Resume after `duration` seconds. Note that tokens resume *immediately*.
    queue.resume()
    queue.pop_buffer(samples)
    time.sleep(0.1)
    assert len(conn) == np.ceil(2 * duration / isi)
    queue.pop_buffer(samples)
    time.sleep(0.1)
    assert len(conn) == np.ceil(3 * duration / isi)

    times = [u['t0'] for u in conn]
    assert times[0] == 0
    all_isi = np.diff(times)

    # Since the queue uses the delay (between offset and onset of
    # consecutive segments), we need to calculate the actual ISI since it
    # may have been rounded to the nearest sample.
    actual_isi = round((isi - t1.duration) * fs) / fs + t1.duration

    # We paused the playout, so this means that we have a very long delay in
    # the middle of the queue. Check for this delay, ensure that there's only
    # one ISI with this delay and then verify that all other ISIs are the
    # expected ISI given the tone pip duration.
    expected_max_isi = round((duration + actual_isi) * fs) / fs
    assert all_isi.max() == expected_max_isi
    m = all_isi == all_isi.max()
    assert sum(m) == 1

    # Now, check that all other ISIs are as expected.
    expected_isi = round(actual_isi * fs) / fs
    np.testing.assert_almost_equal(all_isi[~m], expected_isi)


def test_fifo_queue_pause_resume_timing(fs):
    trials = 20
    samples = int(fs)
    queue, conn, _, _, _ = make_queue(fs, 'FIFO', (1e3, 5e3), trials)
    queue.pop_buffer(samples)
    conn.clear()
    queue.pause(0.1025)
    queue.pop_buffer(samples)
    queue.resume(0.6725)
    queue.pop_buffer(samples)
    time.sleep(0.2)
    t0 = [i['t0'] for i in conn]
    assert t0[0] == round(0.6725 * fs) / fs


def test_fifo_queue_ordering(fs):
    trials = 20
    samples = round(fs)

    queue, conn, _, (k1, k2), (t1, _) = \
        make_queue(fs, 'FIFO', (1e3, 5e3), trials)
    epoch_samples = round(t1.duration * fs)

    waveforms = []
    queue_empty = False

    def mark_empty():
        nonlocal queue_empty
        queue_empty = True

    extractor = extract_epochs(fs=fs,
                               queue=conn,
                               epoch_size=None,
                               poststim_time=0,
                               buffer_size=0,
                               target=waveforms.append,
                               empty_queue_cb=mark_empty)

    waveform = queue.pop_buffer(samples)
    extractor.send(waveform)
    time.sleep(0.1)
    assert queue_empty

    metadata = list(conn)
    for md in metadata[:trials]:
        assert k1 == md['key']
    for md in metadata[trials:]:
        assert k2 == md['key']

    waveforms = np.concatenate(waveforms, axis=0)
    assert waveforms.shape == (trials * 2, epoch_samples)
    for w in waveforms[:trials]:
        assert np.all(w == waveforms[0])
    for w in waveforms[trials:]:
        assert np.all(w == waveforms[trials])
    assert np.any(waveforms[0] != waveforms[trials])


def test_interleaved_fifo_queue_ordering(fs):
    samples = round(fs)
    trials = 20

    queue, conn, _, (k1, k2), (t1, _) = \
        make_queue(fs, 'interleaved', (1e3, 5e3), trials)
    epoch_samples = round(t1.duration * fs)

    waveforms = []
    queue_empty = False

    def mark_empty():
        nonlocal queue_empty
        queue_empty = True

    extractor = extract_epochs(fs=fs,
                               queue=conn,
                               epoch_size=None,
                               poststim_time=0,
                               buffer_size=0,
                               target=waveforms.append,
                               empty_queue_cb=mark_empty)

    waveform = queue.pop_buffer(samples)
    extractor.send(waveform)
    time.sleep(0.1)
    assert queue_empty

    # Verify that keys are ordered properly
    metadata = list(conn)
    for md in metadata[::2]:
        assert k1 == md['key']
    for md in metadata[1::2]:
        assert k2 == md['key']

    waveforms = np.concatenate(waveforms, axis=0)
    assert waveforms.shape == (trials * 2, epoch_samples)
    for w in waveforms[::2]:
        assert np.all(w == waveforms[0])
    for w in waveforms[1::2]:
        assert np.all(w == waveforms[1])
    assert np.any(waveforms[0] != waveforms[1])


def test_queue_continuous_tone(fs):
    """
    Test ability to work with continuous tones and move to the next one
    manually (e.g., as in the case of DPOAEs).
    """
    samples = round(1 * fs)
    queue, conn, _, _, (t1, t2) = make_queue(fs, 'FIFO', (1e3, 5e3), 1,
                                             duration=100)

    # Get samples from t1
    assert queue.get_max_duration() == 100
    assert np.all(queue.pop_buffer(samples) == t1.next(samples))
    assert np.all(queue.pop_buffer(samples) == t1.next(samples))

    # Switch to t2
    queue.next_trial()
    assert np.all(queue.pop_buffer(samples) == t2.next(samples))
    assert np.all(queue.pop_buffer(samples) == t2.next(samples))

    # Ensure timing information correct
    assert len(conn) == 2
    assert conn.popleft()['t0'] == 0
    assert conn.popleft()['t0'] == (samples * 2) / fs


def test_future_pause(fs):
    queue, conn, rem_conn, _, _ = make_queue(fs, 'FIFO', [1e3, 5e3], 100)
    queue.pop_buffer(1000)
    # This is OK
    queue.pause(1000 / fs)
    queue.resume(1000 / fs)
    # This is not
    with pytest.raises(ValueError):
        queue.pause(1001 / fs)


def test_queue_partial_capture(fs):
    queue, conn, rem_conn, _, (t1, t2) = \
        make_queue(fs, 'FIFO', [1e3, 5e3], 100)
    extractor_conn = deque()
    extractor_rem_conn = deque()
    queue.connect(extractor_conn.append, 'added')
    queue.connect(extractor_rem_conn.append, 'removed')

    waveforms = []
    extractor = extract_epochs(fs=fs,
                               queue=extractor_conn,
                               removed_queue=extractor_rem_conn,
                               poststim_time=0,
                               buffer_size=0,
                               epoch_size=15e-3,
                               target=waveforms.extend)

    samples = int(fs)
    tone_samples = t1.n_samples_remaining()
    w1 = queue.pop_buffer(int(tone_samples / 2))
    queue.pause(0.5 * tone_samples / fs)
    w2 = queue.pop_buffer(samples)
    extractor.send(w1)
    extractor.send(w2)
    time.sleep(0.1)

    assert len(waveforms) == 0


def test_remove_keys(fs):
    frequencies = (500, 1e3, 2e3, 4e3, 8e3)
    queue, conn, _, keys, tones = make_queue(fs, 'FIFO', frequencies, 100)
    queue.remove_key(keys[1])
    queue.pop_buffer(int(fs))
    queue.remove_key(keys[0])
    queue.pop_buffer(int(fs))
    time.sleep(0.1)
    counts = Counter(c['key'] for c in conn)
    assert counts[keys[0]] == int(rate)
    assert counts[keys[2]] == int(rate)

    # Should generate all remaining queued trials. Make sure it properly
    # exits the queue.
    queue.pop_buffer((int(5 * 100 / rate * fs)))
    counts = Counter(c['key'] for c in conn)
    assert keys[1] not in counts
    assert counts[keys[0]] == int(rate)
    for k in keys[2:]:
        assert counts[k] == 100


def test_remove_keys_with_no_auto_decrement(fs):
    frequencies = (500, 1e3, 2e3, 4e3, 8e3)
    queue, conn, _, keys, tones = make_queue(fs, 'FIFO', frequencies, 100)
    queue.remove_key(keys[1])
    queue.pop_buffer(10 * int(fs), decrement=False)
    queue.remove_key(keys[0])
    for key in keys[2:]:
        queue.remove_key(key)

    # Should generate all remaining queued trials. Make sure it properly
    # exits the queue.
    queue.pop_buffer((int(5 * 100 / rate * fs)), decrement=False)
    counts = Counter(c['key'] for c in conn)
    assert keys[1] not in counts
    assert counts[keys[0]] == 10 * int(rate)
    for k in keys[2:]:
        assert k not in counts


def test_get_closest_key(fs):
    frequencies = (500, 1e3, 2e3, 4e3, 8e3)
    queue, conn, _, keys, tones = make_queue(fs, 'FIFO', frequencies, 100)
    assert queue.get_closest_key(1) is None
    queue.pop_buffer(int(fs))
    assert queue.get_closest_key(1) == keys[0]
    queue.pop_buffer(int(fs))
    assert queue.get_closest_key(1) == keys[0]
    assert queue.get_closest_key(2) == keys[1]


def test_rebuffering(fs):
    from matplotlib import pyplot as plt
    frequencies = (500, 1e3, 2e3, 4e3, 8e3)
    trials = 200
    queue, conn, rem_conn, keys, tones = \
        make_queue(fs, 'FIFO', frequencies, trials)


    waveforms = []
    extractor_conn = deque()
    extractor_rem_conn = deque()
    queue.connect(extractor_conn.append, 'added')
    queue.connect(extractor_rem_conn.append, 'removed')
    extractor = extract_epochs(fs=fs,
                               queue=extractor_conn,
                               removed_queue=extractor_rem_conn,
                               poststim_time=0,
                               buffer_size=0,
                               epoch_size=8.5e-3,
                               target=waveforms.append)

    # Default tone duration is 5e-3
    tone_duration = tones[0].duration
    tone_samples = int(round(tone_duration * fs))

    # Remove 5e-3 sec of the waveform
    extractor.send(queue.pop_buffer(tone_samples))
    time.sleep(0.1)

    # Now, pause the queue at 5e-3 sec, remove 10e-3 worth of samples, and then
    # resume.
    queue.pause(tone_duration)
    extractor.send(queue.pop_buffer(tone_samples*2))
    time.sleep(0.1)

    queue.resume()
    time.sleep(0.1)

    # Pull off one additonal second.
    new_ts = queue.get_ts()

    # Since we will be pausing the queue at 1.005 sec, we need to make sure
    # that we do not actually deliver the samples after 1.005 sec to the
    # extractor (this simulates a DAQ where we have uploaded some samples to a
    # "buffer" but have not actually played them out).
    old_ts = queue.get_ts()
    keep = (tone_duration + 1.0) - old_ts
    keep_samples = int(round(keep * fs))
    w = queue.pop_buffer(int(fs))
    extractor.send(w[:keep_samples])
    time.sleep(0.1)
    assert queue.get_ts() == pytest.approx(1.015, 4)

    # This will result in pausing in the middle of a tone burst. This ensures
    # that we properly notify the extractor that a stimulus was cut-off halfway
    # (i.e., rem_conn will have an item in the queue).
    assert len(rem_conn) == 0
    queue.pause(tone_duration + 1.0)
    queue.resume(tone_duration + 1.0)
    time.sleep(0.1)
    assert len(rem_conn) == 1

    # Clear all remaining trials
    extractor.send(queue.pop_buffer(15 * int(fs)))

    # Check that we have the expected number of epochs acquired
    #assert len(waveforms) == (len(frequencies) * trials)

    epochs = np.concatenate(waveforms, axis=0)
    epochs.shape = len(frequencies), trials, -1

    # Make sure epochs 1 ... end are equal to epoch 0
    assert np.all(np.equal(epochs[:, [0]], epochs))


def test_queue_speed(fs, benchmark):

    def setup():
        n_trials = 512
        duration = 5e-3
        isi = 1/40
        queue, _, _, _, _ = \
            make_queue(fs, 'interleaved', [1e3, 2e3, 4e3, 8e3, 16e3], n_trials, duration=duration, isi=isi)

        block_size = 4096
        total_duration_sec = 5 * n_trials * (duration + isi)
        total_duration_samples = int(round(total_duration_sec * fs))
        n_blocks = total_duration_samples // block_size

        return (n_blocks, block_size, queue), {}

    def profiler(n_blocks, block_size, queue):
        segments = []
        for i in range(n_blocks):
            segments.append(queue.pop_buffer(block_size))
        return n_blocks, block_size, segments

    n_blocks, block_size, segments = benchmark.pedantic(profiler, setup=setup, rounds=5, iterations=1)

    assert len(segments) == n_blocks
    assert segments[0].shape[-1] == block_size
    assert segments[-1].shape[-1] == block_size
