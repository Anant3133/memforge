"""
MemForge ML runtime.

A NullHandler is attached here and nothing else is configured: library code
that installs its own log handlers hijacks the logging of whatever imports
it. The application decides where records go -- see
`server/logging_config.py` for the server's choice.
"""

import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
