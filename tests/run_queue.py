import os
os.environ['LINE_PROFILE'] = '1'

from test_queue import make_queue

fs = 96e3


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


args, kw = setup()
profiler(*args, **kw)
