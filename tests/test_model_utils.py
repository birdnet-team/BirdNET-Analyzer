"""Tests for analysis session pause/cancel behavior in model_utils."""

from birdnet_analyzer import model_utils


class FakeSession:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_pause_cancels_sessions_without_latching_shutdown():
    session = FakeSession()
    model_utils._register_session(session)

    try:
        assert model_utils.pause_active_analyses() == 1
        assert session.cancelled
        assert not model_utils._SHUTDOWN.is_set(), "pause must not latch shutdown"

        # After a pause, newly registered sessions must not be auto-cancelled
        # (unlike after cancel_active_analyses), so the run can be continued.
        new_session = FakeSession()
        model_utils._register_session(new_session)
        assert not new_session.cancelled
        model_utils._unregister_session(new_session)
    finally:
        model_utils._unregister_session(session)
        model_utils._SHUTDOWN.clear()


def test_pause_with_no_active_sessions_is_a_noop():
    assert model_utils.pause_active_analyses() == 0
    assert not model_utils._SHUTDOWN.is_set()
