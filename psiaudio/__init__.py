import logging

from .version import __version__


def add_logging_level(name, level):
    logging.addLevelName(level, name)
    # This step is required for coloredlogs to properly map the level name
    setattr(logging, name, level)
    def trace(self, message, *args, **kws):
        nonlocal level
        # Yes, logger takes its '*args' as 'args'.
        if self.isEnabledFor(level):
            self._log(level, message, args, **kws)
    logging.Logger.trace = trace


# Set up a verbose debugger level for tracing
add_logging_level('TRACE', 5)
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


try:
    # Ensure that we register the OctaveScale
    from . import plot
except ImportError:
    pass
