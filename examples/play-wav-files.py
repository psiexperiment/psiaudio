'''
===================================================
Using a queue to play wav files through the speaker
===================================================

Randomly shuffle the six wav files and play them continuously using the queue.
'''
import argparse
import sys
import threading

import numpy as np
import sounddevice as sd

from psiaudio.stim import wavs_from_path
from psiaudio.queue import BlockedRandomSignalQueue


help_text = 'Output device (numeric ID). ' \
    'To find the sound device number, type `python -m sounddevice`.'
parser = argparse.ArgumentParser('play-wav-files')
parser.add_argument('-d', '--device', type=int, help=help_text)
args = parser.parse_args()


# Set up our random signal queue. This is the psiaudio specific part
blocksize = 2048
fs = 44100.0
base_path = 'wav-files'
wavfiles = wavs_from_path(fs, base_path)
queue = BlockedRandomSignalQueue(fs)


###############################################################################
# We can subscribe to a notification event for each time a new stimulus comes
# up in the queue. The notification event is a dictionary consisting of `t0`
# (the time the stimulus began), `duration` (the duration of the stimulus)`,
# `key` (the UUID of the stimulus), `metadata` (a user-provided value that
# helps identify the stimulus, and `decrement` (which indicates whether the
# number of trials was decremented by one).
def print_event(info):
    filename = info['metadata']
    duration = info['duration']
    t0 = info['t0']
    print(f'Now playing {filename}, a {duration} second long stimulus ' \
          f'starting at {t0} seconds')


queue.connect(print_event, 'added')

# Set up the metadata so that it contains the name of each wavfile that was
# played.
metadata = [w.filename.stem for w in wavfiles]
queue.extend(wavfiles, 1, metadata=metadata)

###############################################################################
# This callback is required by sounddevice to continuously update the audio
# card buffer with new samples. The rest of the code is specific to the
# sounddevice module.
def callback(outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)

    # Format of outdata is sample x channel
    data = queue.pop_buffer(frames)
    outdata[:] = data[:, np.newaxis]


event = threading.Event()
stream = sd.OutputStream(samplerate=fs, blocksize=blocksize, channels=1,
                         device=args.device, dtype='float32',
                         callback=callback, finished_callback=event.set)


try:
    with stream:
        event.wait()
except KeyboardInterrupt:
    pass
