import io
import json
import logging
import sys

from agentic_rag.observability.logging_setup import configure_json_logging


def _log(logger_name: str, message: str) -> None:
    logging.getLogger(logger_name).info(message)


def test_configure_json_logging_writes_to_the_given_stream():
    stream = io.StringIO()

    configure_json_logging("agentic_rag.test.alpha", stream=stream)
    _log("agentic_rag.test.alpha", json.dumps({"event": "alpha"}))

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "alpha"


def test_configure_json_logging_resolves_stdout_at_call_time(monkeypatch):
    replacement_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", replacement_stdout)

    configure_json_logging("agentic_rag.test.beta")
    _log("agentic_rag.test.beta", json.dumps({"event": "beta"}))

    payload = json.loads(replacement_stdout.getvalue().strip())
    assert payload["event"] == "beta"


def test_configure_json_logging_reconfiguring_does_not_duplicate_handlers():
    first_stream = io.StringIO()
    second_stream = io.StringIO()

    configure_json_logging("agentic_rag.test.gamma", stream=first_stream)
    configure_json_logging("agentic_rag.test.gamma", stream=second_stream)
    _log("agentic_rag.test.gamma", json.dumps({"event": "gamma"}))

    assert first_stream.getvalue() == ""
    lines = [line for line in second_stream.getvalue().splitlines() if line]
    assert len(lines) == 1


def test_configure_json_logging_tracks_different_loggers_independently():
    # Reconfiguring one logger must not disturb a handler already
    # attached to a different logger - each is tracked by its own name,
    # not a single shared "the one active handler" reference.
    delta_stream = io.StringIO()
    epsilon_stream = io.StringIO()

    configure_json_logging("agentic_rag.test.delta", stream=delta_stream)
    configure_json_logging("agentic_rag.test.epsilon", stream=epsilon_stream)
    configure_json_logging("agentic_rag.test.epsilon", stream=epsilon_stream)

    _log("agentic_rag.test.delta", json.dumps({"event": "delta"}))
    _log("agentic_rag.test.epsilon", json.dumps({"event": "epsilon"}))

    delta_lines = [line for line in delta_stream.getvalue().splitlines() if line]
    epsilon_lines = [line for line in epsilon_stream.getvalue().splitlines() if line]
    assert len(delta_lines) == 1
    assert len(epsilon_lines) == 1
