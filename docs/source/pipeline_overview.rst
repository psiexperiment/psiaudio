Pipeline Overview
=================

The ``psiaudio.pipeline`` module provides a framework for building real-time
and offline data processing pipelines. It is built around a push-based
architecture using Python coroutines and specialized data structures that
preserve metadata across processing steps.

Architecture
------------

The pipeline follows a "Research -> Strategy -> Execution" philosophy, where
individual processing components are implemented as coroutines decorated with
``@coroutine``. Data is passed through the pipeline using the ``.send()``
method.

Key Components
~~~~~~~~~~~~~~

* **Coroutines**: Functions decorated with ``@coroutine`` that automatically
  advance to the first ``yield`` statement. They typically receive data from
  an upstream source and push results to one or more downstream targets.
* **Broadcasting**: The ``broadcast`` coroutine allows a single data stream
  to be sent to multiple processing branches.
* **Transformation**: The ``transform`` coroutine applies a simple function
  to data before passing it along.

Data Structures
---------------

PipelineData
~~~~~~~~~~~~

The ``PipelineData`` class is a subclass of ``numpy.ndarray`` designed to carry
essential metadata along with the raw signal data. It supports:

* **Sampling Rate (fs)**: Persisted through slicing and many transformations.
* **Timing (s0, t0)**: Tracks the starting sample and time relative to the
  beginning of the stream.
* **Channels**: Maintains channel labels for multichannel data.
* **Metadata**: Dictionary or list of dictionaries (for epoched data) containing
  arbitrary key-value pairs (e.g., trial parameters).

Events
~~~~~~

The ``Events`` class manages time-stamped events occurring within a data
stream. It provides utilities for sub-selecting events by time or sample
range and calculating event rates.

Common Processing Steps
-----------------------

Continuous Data
~~~~~~~~~~~~~~~

* **Filtering**: ``iirfilter`` provides real-time IIR filtering with state
  preservation to avoid transients between data chunks.
* **Resampling**: ``downsample`` and ``decimate`` for changing sampling rates.
* **RMS Calculation**: ``rms`` and ``rms_band`` for power estimation over
  sliding windows.

Epoching
~~~~~~~~

The ``extract_epochs`` coroutine is a powerful tool for segmenting continuous
data into discrete trials (epochs) based on trigger times. It includes a
look-back buffer to capture the start of an epoch even if the trigger signal
arrives slightly after the data.

Multichannel Support
~~~~~~~~~~~~~~~~~~~~

* **mc_select**: Select individual channels or subsets of channels.
* **mc_reference**: Apply linear referencing matrices (e.g., Common Average
  Reference).

Example Usage
-------------

.. code-block:: python

    from psiaudio import pipeline

    @pipeline.coroutine
    def printer(data):
        while True:
            d = (yield)
            print(f"Received data with shape {d.shape}")

    # Build a simple pipeline: filter -> print
    target = printer()
    p = pipeline.iirfilter(fs=1000, N=2, Wn=50, rp=0.1, rs=40,
                           btype='lowpass', ftype='cheby1', target=target)

    # Push data into the pipeline
    import numpy as np
    data = pipeline.PipelineData(np.random.randn(100), fs=1000)
    p.send(data)

API Reference
-------------

For detailed information on specific functions and classes, see the
:doc:`api/psiaudio.pipeline`.
