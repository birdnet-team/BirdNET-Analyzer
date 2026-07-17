import logging

import pytest

from birdnet_analyzer import cli, logs, settings


@pytest.fixture
def error_log(monkeypatch, tmp_path):
    log_path = tmp_path / "error_log.txt"
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(log_path))
    logs.setup_logging()

    yield log_path

    logs._remove_installed_handlers()


def _raised(message: str) -> Exception:
    """Returns an exception that carries a stacktrace, as an except block gets it."""
    try:
        raise RuntimeError(message)
    except RuntimeError as e:
        return e


def test_messages_reach_the_console_bare_like_the_print_calls_they_replaced(
    error_log, capsys
):
    logs.setup_logging()  # binds the stdout capsys sees, not the fixture-time one

    logging.getLogger("birdnet_analyzer.segments.utils").info(
        "Found 3 audio files with valid result file."
    )

    assert capsys.readouterr().out == "Found 3 audio files with valid result file.\n"
    assert not error_log.exists()


def test_errors_keep_the_console_clean_and_collect_the_stacktrace_in_the_file(
    error_log, capsys
):
    logs.setup_logging()  # binds the stdout capsys sees, not the fixture-time one

    logging.getLogger("birdnet_analyzer.train.utils").error(
        "Error: Cannot open audio file x.wav", exc_info=_raised("boom")
    )

    assert capsys.readouterr().out == "Error: Cannot open audio file x.wav\n"

    content = error_log.read_text(encoding="utf-8")

    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: boom" in content


def test_setup_can_be_called_again_without_duplicating_output(error_log, capsys):
    logs.setup_logging()
    logs.setup_logging()

    logs.logger.info("once")

    assert capsys.readouterr().out == "once\n"


def test_removing_the_handlers_also_resets_the_logger_level(error_log):
    assert logs.logger.level != logging.NOTSET

    logs._remove_installed_handlers()

    # Back to the library state: a host application configuring logging afterwards
    # decides which records get through, including DEBUG.
    assert logs.logger.level == logging.NOTSET


def test_unconfigured_library_use_writes_no_files(monkeypatch, tmp_path):
    log_path = tmp_path / "error_log.txt"
    monkeypatch.setattr(settings, "ERROR_LOG_FILE", str(log_path))

    settings.write_error_log(_raised("library mode"))

    assert not log_path.exists()


def test_long_stacktraces_are_shortened(monkeypatch, error_log):
    monkeypatch.setattr(settings, "MAX_ERROR_LOG_ENTRY_SIZE", 200)
    settings.write_error_log(_raised("x" * 5000 + "tail"))

    content = error_log.read_text(encoding="utf-8")

    assert len(content) < 500
    assert "characters omitted" in content
    # Both ends of the stacktrace survive, so the exception stays identifiable.
    assert content.startswith("[")
    assert content.rstrip().endswith("tail")


def test_error_log_is_rotated_when_it_gets_too_large(monkeypatch, error_log):
    monkeypatch.setattr(settings, "MAX_ERROR_LOG_SIZE", 1000)
    logs.setup_logging()  # picks up the lowered size limit
    error_log.write_text("old entry\n" + "x" * 1000, encoding="utf-8")

    settings.write_error_log(_raised("new entry"))

    backup = error_log.with_name("error_log.txt.1")

    assert backup.read_text(encoding="utf-8").startswith("old entry")

    content = error_log.read_text(encoding="utf-8")

    assert "new entry" in content
    assert "old entry" not in content


def test_error_log_is_not_rotated_while_it_is_small(error_log):
    settings.write_error_log(_raised("old entry"))
    settings.write_error_log(_raised("new entry"))

    content = error_log.read_text(encoding="utf-8")

    assert not error_log.with_name("error_log.txt.1").exists()
    assert "old entry" in content
    assert "new entry" in content


def test_verbose_flag_shows_stacktraces_on_the_console(error_log, capsys):
    args = cli.verbosity_args().parse_args(["--verbose"])

    logs.logger.error("Error: something broke", exc_info=_raised("boom"))

    out = capsys.readouterr().out

    assert "Error: something broke" in out
    assert "Traceback (most recent call last)" in out
    assert "RuntimeError: boom" in out
    # The flag configures logging directly and must not leak into the keyword
    # arguments the entry points pass into the feature functions.
    assert vars(args) == {}


def test_quiet_flag_hides_messages_but_still_collects_errors(error_log, capsys):
    args = cli.verbosity_args().parse_args(["--quiet"])

    logs.logger.info("Found 3 audio files.")
    logs.logger.error("Error: something broke", exc_info=_raised("boom"))

    assert capsys.readouterr().out == "Error: something broke\n"
    assert "RuntimeError: boom" in error_log.read_text(encoding="utf-8")
    assert vars(args) == {}


def test_verbose_and_quiet_are_mutually_exclusive(error_log, capsys):
    with pytest.raises(SystemExit):
        cli.verbosity_args().parse_args(["--verbose", "--quiet"])
