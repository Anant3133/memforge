"""
Structured logging for the MemForge server.

The server previously used `print()`, which is fine for watching a terminal and
useless for anything else: no levels, no timestamps, no way to filter, and
nothing a log aggregator can query. This is the remaining Phase F item from
`PROGRESS_AND_NEXT_PHASES.md` §8.

Two formats, chosen by `MEMFORGE_LOG_FORMAT`:

  * `text` (default) -- aligned, human-readable, what you want in a terminal.
  * `json` -- one object per line, for shipping somewhere. Extra fields passed
    via `logger.info(..., extra={...})` become top-level keys, so a query can
    filter on `seq_id` or `blocks_allocated` rather than regex a message.

**Layering matters here.** Only the *application* configures handlers;
`memforge_ml` code calls `logging.getLogger(__name__)` and nothing else. A
library that installs its own handlers hijacks the logging of whatever imports
it, which is why `memforge_ml/__init__.py` attaches a `NullHandler` and stops
there. `configure_logging()` is called from the server's lifespan, and anyone
embedding the engine keeps their own setup.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

#: Fields `logging` puts on every record. Anything outside this set came from an
#: `extra={...}` argument and is therefore worth emitting as structured data.
_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }
)

LOGGER_NAMES = ("memforge", "memforge_ml", "server")


class JsonFormatter(logging.Formatter):
    """
    One JSON object per line.

    Values that are not JSON-serialisable are stringified rather than raising:
    a logging call must never be the thing that brings down a request.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, default=str)
        except Exception:
            return json.dumps(
                {"ts": payload["ts"], "level": "ERROR", "logger": record.name,
                 "message": "log record was not serialisable"}
            )


class TextFormatter(logging.Formatter):
    """
    Human-readable, with structured extras appended as `key=value`.

    The extras are not dropped in text mode: a field worth emitting as JSON is
    worth seeing in a terminal, and silently showing less than the JSON format
    would make the two disagree about what was logged.
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
            base = f"{base}  [{rendered}]"
        return base


def configure_logging(
    level: str | None = None,
    log_format: str | None = None,
    stream: Any = None,
) -> logging.Logger:
    """
    Install a single handler on MemForge's loggers. Safe to call more than once.

    Reads `MEMFORGE_LOG_LEVEL` (default `INFO`) and `MEMFORGE_LOG_FORMAT`
    (`text` or `json`, default `text`) when the arguments are omitted.

    Writes to stderr, not stdout: stdout is where a CLI's actual output goes,
    and a benchmark piping results through it should not have log lines mixed in.

    `propagate` is left off on these loggers so records do not also reach the
    root handler uvicorn installs, which would print everything twice.
    """
    resolved_level = (level or os.environ.get("MEMFORGE_LOG_LEVEL", "INFO")).upper()
    resolved_format = (log_format or os.environ.get("MEMFORGE_LOG_FORMAT", "text")).lower()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if resolved_format == "json" else TextFormatter())
    handler.set_name("memforge")

    numeric = getattr(logging, resolved_level, logging.INFO)
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        # Replace rather than append: calling this twice (a reload, a test)
        # would otherwise duplicate every line for each extra handler.
        for existing in [h for h in logger.handlers if h.get_name() == "memforge"]:
            logger.removeHandler(existing)
        logger.addHandler(handler)
        logger.setLevel(numeric)
        logger.propagate = False

    return logging.getLogger("memforge")


def get_logger(name: str) -> logging.Logger:
    """A logger under the configured hierarchy. Handlers come from the app."""
    return logging.getLogger(name)
