import logging

import pytest

from birdnet_analyzer import cli, logs, settings, utils


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

    assert capsys.readouterr() == ("Found 3 audio files with valid result file.\n", "")
    assert not error_log.exists()


def test_warnings_reach_the_console_with_the_other_messages(error_log, capsys):
    logs.setup_logging()  # binds the streams capsys sees, not the fixture-time ones

    logging.getLogger("birdnet_analyzer.utils").warning("1 of 3 files was skipped.")

    assert capsys.readouterr() == ("1 of 3 files was skipped.\n", "")


def test_errors_go_to_stderr_bare_and_leave_the_stacktrace_to_the_file(
    error_log, capsys
):
    logs.setup_logging()  # binds the streams capsys sees, not the fixture-time ones

    logging.getLogger("birdnet_analyzer.train.utils").error(
        "Error: Cannot open audio file x.wav", exc_info=_raised("boom")
    )

    assert capsys.readouterr() == ("", "Error: Cannot open audio file x.wav\n")

    content = error_log.read_text(encoding="utf-8")

    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: boom" in content


def test_setup_can_be_called_again_without_duplicating_output(error_log, capsys):
    logs.setup_logging()
    logs.setup_logging()

    logs.logger.info("once")

    assert capsys.readouterr().out == "once\n"


def test_a_handler_a_dependency_puts_on_the_root_logger_gets_nothing_of_the_app(
    error_log,
):
    seen = []

    class OfADependency(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = OfADependency()
    logging.getLogger().addHandler(handler)

    try:
        logs.logger.error("Error: something broke", exc_info=_raised("boom"))
        logs._remove_installed_handlers()
        logs.logger.error("as a library")
    finally:
        logging.getLogger().removeHandler(handler)

    assert seen == ["as a library"]


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

    err = capsys.readouterr().err

    assert "Error: something broke" in err
    assert "Traceback (most recent call last)" in err
    assert "RuntimeError: boom" in err
    # The flag configures logging directly and must not leak into the keyword
    # arguments the entry points pass into the feature functions.
    assert vars(args) == {}


def test_quiet_flag_hides_messages_but_still_collects_errors(error_log, capsys):
    args = cli.verbosity_args().parse_args(["--quiet"])

    logs.logger.info("Found 3 audio files.")
    logs.logger.error("Error: something broke", exc_info=_raised("boom"))

    assert capsys.readouterr() == ("", "Error: something broke\n")
    assert "RuntimeError: boom" in error_log.read_text(encoding="utf-8")
    assert vars(args) == {}


def test_verbose_and_quiet_are_mutually_exclusive(error_log, capsys):
    with pytest.raises(SystemExit):
        cli.verbosity_args().parse_args(["--verbose", "--quiet"])


@pytest.fixture
def failing_command(error_log, monkeypatch):
    monkeypatch.setattr(utils, "configure_runtime", lambda: None)

    def run(*flags, error=ValueError("Input path 'x' was not found.")):
        @utils.runtime_error_handler
        def main():
            cli.verbosity_args().parse_args(list(flags))
            raise error

        with pytest.raises(SystemExit) as exit_info:
            main()

        return exit_info.value.code

    return run


def test_a_failed_command_prints_its_error_and_leaves_the_stacktrace_to_the_file(
    failing_command, error_log, capsys
):
    assert failing_command() == 1

    assert capsys.readouterr() == (
        "",
        "Error: Input path 'x' was not found.\n"
        f"The stacktrace is in {error_log}; --verbose shows it here.\n",
    )
    assert "Traceback (most recent call last)" in error_log.read_text(encoding="utf-8")


def test_a_failed_verbose_command_prints_its_stacktrace_once(failing_command, capsys):
    assert failing_command("--verbose") == 1

    err = capsys.readouterr().err

    assert err.count("Traceback (most recent call last)") == 1
    assert "The stacktrace is in" not in err


@pytest.mark.parametrize(
    ("error", "line"),
    [
        (
            UnicodeDecodeError("utf-8", b"M\xf6nch", 1, 2, "invalid start byte"),
            "Error: 'utf-8' codec can't decode byte 0xf6 in position 1: "
            "invalid start byte",
        ),
        (EOFError(), "Error: EOFError()"),
    ],
    ids=["several arguments", "no message"],
)
def test_a_failed_command_names_the_reason_of_any_error(
    failing_command, capsys, error, line
):
    failing_command(error=error)

    assert capsys.readouterr().err.splitlines()[0] == line
