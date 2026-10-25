"""
Tests for structured logging.

Logging is easy to get subtly wrong in ways nobody notices until they need the
logs: duplicated lines, dropped structured fields, a library stealing the host
application's handlers, or a non-serialisable value turning a log call into the
thing that fails a request. Each of those has a test here.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from server.logging_config import (
    LOGGER_NAMES,
    JsonFormatter,
    TextFormatter,
    configure_logging,
    get_logger,
)


@pytest.fixture
def capture():
    """Configure logging into a buffer and restore the handlers afterwards."""
    buffer = io.StringIO()
    saved = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in LOGGER_NAMES
    }
    yield buffer
    for name, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def lines(buffer: io.StringIO):
    return [line for line in buffer.getvalue().splitlines() if line.strip()]


# ------------------------------------------------------------------ json mode

def test_json_format_emits_one_object_per_line_with_extras(capture):
    configure_logging(level="INFO", log_format="json", stream=capture)
    get_logger("server.test").info(
        "engine ready", extra={"kv_pool_mb": 320.0, "num_blocks": 204}
    )

    emitted = lines(capture)
    assert len(emitted) == 1
    record = json.loads(emitted[0])
    assert record["level"] == "INFO"
    assert record["logger"] == "server.test"
    assert record["message"] == "engine ready"
    # Structured fields are top level, so a query can filter on them rather
    # than regex the message.
    assert record["kv_pool_mb"] == 320.0
    assert record["num_blocks"] == 204
    assert record["ts"].endswith("Z")


def test_json_format_includes_exceptions(capture):
    configure_logging(level="INFO", log_format="json", stream=capture)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("server.test").error("step failed", exc_info=True)

    record = json.loads(lines(capture)[0])
    assert "exception" in record
    assert "ValueError: boom" in record["exception"]


def test_json_format_survives_an_unserialisable_value(capture):
    """
    A logging call must never be the thing that fails a request.

    `default=str` handles the common case; the formatter also catches outright
    serialisation failure rather than letting it propagate into the caller.
    """
    configure_logging(level="INFO", log_format="json", stream=capture)

    class _Hostile:
        def __repr__(self):
            return "<hostile>"

    get_logger("server.test").info("odd value", extra={"thing": _Hostile()})
    record = json.loads(lines(capture)[0])
    assert record["thing"] == "<hostile>"


# ------------------------------------------------------------------ text mode

def test_text_format_shows_extras_rather_than_dropping_them(capture):
    """
    A field worth emitting as JSON is worth seeing in a terminal; silently
    showing less would make the two formats disagree about what was logged.
    """
    configure_logging(level="INFO", log_format="text", stream=capture)
    get_logger("server.test").info("engine ready", extra={"blocks": 204, "device": "cpu"})

    line = lines(capture)[0]
    assert "engine ready" in line
    assert "blocks=204" in line
    assert "device=cpu" in line
    assert "INFO" in line


def test_text_format_without_extras_has_no_empty_brackets(capture):
    configure_logging(level="INFO", log_format="text", stream=capture)
    get_logger("server.test").info("plain message")
    line = lines(capture)[0]
    assert "plain message" in line
    assert "[]" not in line


# --------------------------------------------------------------- configuration

def test_configure_is_idempotent(capture):
    """
    Calling it twice must not duplicate every line.

    A reload, a test, or an embedding application calling setup defensively
    would otherwise double or triple the output.
    """
    for _ in range(3):
        configure_logging(level="INFO", log_format="text", stream=capture)
    get_logger("server.test").info("once")
    assert len(lines(capture)) == 1


def test_level_is_honoured(capture):
    configure_logging(level="ERROR", log_format="text", stream=capture)
    logger = get_logger("server.test")
    logger.info("suppressed")
    logger.warning("also suppressed")
    logger.error("emitted")

    emitted = lines(capture)
    assert len(emitted) == 1
    assert "emitted" in emitted[0]


def test_level_comes_from_the_environment(capture, monkeypatch):
    monkeypatch.setenv("MEMFORGE_LOG_LEVEL", "warning")
    monkeypatch.setenv("MEMFORGE_LOG_FORMAT", "json")
    configure_logging(stream=capture)

    logger = get_logger("server.test")
    logger.info("suppressed")
    logger.warning("kept")

    emitted = lines(capture)
    assert len(emitted) == 1
    assert json.loads(emitted[0])["message"] == "kept"


def test_an_unknown_level_falls_back_to_info(capture):
    """A typo in an env var should not silence the server."""
    configure_logging(level="NOTALEVEL", log_format="text", stream=capture)
    get_logger("server.test").info("still visible")
    assert len(lines(capture)) == 1


def test_records_do_not_propagate_to_the_root_logger(capture):
    """
    uvicorn installs a root handler, so propagating would print everything
    twice -- once ours, once theirs.
    """
    configure_logging(level="INFO", log_format="text", stream=capture)
    for name in LOGGER_NAMES:
        assert logging.getLogger(name).propagate is False


def test_engine_logs_reach_the_configured_handler(capture):
    """
    The point of the layering: `memforge_ml` only asks for a logger, and the
    application's handler is what catches it.
    """
    configure_logging(level="INFO", log_format="json", stream=capture)
    logging.getLogger("memforge_ml.runtime.continuous_engine").warning(
        "block attention unavailable"
    )
    record = json.loads(lines(capture)[0])
    assert record["logger"] == "memforge_ml.runtime.continuous_engine"
    assert record["level"] == "WARNING"


def test_library_package_installs_only_a_null_handler():
    """
    Importing `memforge_ml` must install nothing but a NullHandler.

    Library code that attaches real handlers hijacks the logging of whatever
    imports it. Checked in a subprocess on purpose: `configure_logging()`
    deliberately *does* add a real handler to `memforge_ml` — that is the
    application configuring the library — so asserting this in-process only
    works if nothing else has configured logging yet, which makes the result
    depend on test ordering. A fresh interpreter is the only honest check.
    """
    import subprocess
    import sys

    probe = (
        "import logging, memforge_ml;"
        "hs = logging.getLogger('memforge_ml').handlers;"
        "print(len(hs), all(isinstance(h, logging.NullHandler) for h in hs))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    count, only_null = result.stdout.split()
    assert int(count) >= 1, "expected a NullHandler so library use is silent by default"
    assert only_null == "True", (
        f"memforge_ml installed a real handler on import: {result.stdout!r}"
    )


def test_formatters_are_usable_directly():
    """They are public enough to be reused, so they must not need the helper."""
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    assert "hello world" in TextFormatter().format(record)
    assert json.loads(JsonFormatter().format(record))["message"] == "hello world"
