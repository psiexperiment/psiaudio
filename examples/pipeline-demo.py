'''
Pipeline Demonstration
======================

This script demonstrates how to build a data processing pipeline using 
`psiaudio.pipeline`. It generates a signal containing tone pips embedded 
in broadband noise, and then processes this signal through two branches:

1.  A branch that band-pass filters the signal and calculates a running RMS.
2.  A branch that extracts individual tone pips (epochs) based on their 
    known presentation times.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque
from psiaudio import stim, calibration, pipeline

# 1. Configuration
fs = 100000  # Sampling rate
cal = calibration.FlatCalibration.as_attenuation()

# 2. Stimulus Generation Setup
# Define a 1kHz tone pip, 50ms duration, with 5ms rise/fall ramps.
tone_factory = stim.ToneFactory(fs=fs, frequency=1000, level=0, calibration=cal)
tone_pip = stim.Cos2EnvelopeFactory(fs=fs, duration=0.05, rise_time=0.005,
                                    input_factory=tone_factory)

# Define continuous broadband noise at a lower level (-20 dB).
noise_factory = stim.BroadbandNoiseFactory(fs=fs, level=-20, calibration=cal)

# Setup a FIFO queue to manage the presentation of 5 tone pips.
# We add random delays (inter-trial intervals) between the pips.
queue = stim.queue.FIFOSignalQueue(fs=fs)
delays = np.random.uniform(0.2, 0.6, 5)
queue.append(tone_pip, trials=5, delays=delays)

# 3. Pipeline Setup
# The epoch_queue will hold information about when each tone pip occurs.
epoch_info_queue = deque()

rms_capture = []
epoch_capture = []

def on_epoch_extracted(data):
    '''Callback for extracted epochs'''
    epoch_capture.append(data)

def on_rms_calculated(data):
    '''Callback for running RMS'''
    rms_capture.append(data)

# Branch 1: Band-pass filter -> RMS calculation
# We'll filter between 500Hz and 2000Hz to focus on the 1kHz tone.
rms_branch = pipeline.iirfilter(
    fs=fs, N=4, Wn=[500, 2000], rp=0.1, rs=40,
    btype='bandpass', ftype='cheby1',
    target=pipeline.rms(fs=fs, duration=0.1, target=on_rms_calculated).send
)

# Branch 2: Epoch extraction
# Extract 150ms epochs, starting 50ms before the trigger (t0).
epoch_branch = pipeline.extract_epochs(
    fs=fs,
    queue=epoch_info_queue,
    epoch_size=0.1,    # The tone is 50ms, we capture 100ms total (+prestim)
    #prestim_time=0.05, # 50ms pre-trigger
    target=on_epoch_extracted,
    buffer_size=5,
)

# Main broadcast point to send data to both branches.
main_pipeline = pipeline.broadcast(
    rms_branch.send,
    epoch_branch.send
)

# Connect the stimulus queue to the pipeline's epoch queue.
# When the queue 'adds' a stimulus to its internal generation schedule,
# it emits an event that we use to tell the pipeline to look for an epoch.
def on_stimulus_added(info):
    epoch_info_queue.append(info)

queue.connect(on_stimulus_added, 'added')

# 4. Execution Loop
# We simulate a real-time system by popping data in small chunks and
# pushing it through the pipeline.
chunk_size = int(fs * 0.1) # 100ms chunks
total_duration = 10.0
n_chunks = int(total_duration / 0.1)

print(f"Simulating {total_duration} seconds of audio processing...")
print("Tone pips are 1kHz, 50ms long. Noise is broadband at -20dB.\n")

for i in range(n_chunks):
    # Get the next chunk of tone pip signal (mostly silence if no pip is active)
    tone_samples = queue.pop_buffer(chunk_size)
    # Get the next chunk of continuous noise
    noise_samples = noise_factory.next(chunk_size)
    # Combine them
    combined_signal = tone_samples + noise_samples
    # Wrap in PipelineData to preserve timing and sampling rate metadata
    data = pipeline.PipelineData(combined_signal, fs=fs, s0=i*chunk_size)
    # Push to pipeline
    main_pipeline.send(data)

epoch_capture = pipeline.concat(epoch_capture, axis=-3)
epoch_info = pd.DataFrame(epoch_capture.metadata)
epoch_mean = epoch_capture.mean(axis='epoch')
plt.plot(epoch_mean.t, np.array(epoch_mean[0]))
plt.show()


