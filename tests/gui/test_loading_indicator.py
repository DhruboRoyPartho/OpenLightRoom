"""Tests for interface/gui/loading_indicator.py.

Driven via direct method calls on the real Qt objects rather than waiting
on real timers/event-loop polling - this sandbox's QTest.qWait()-based
waits have been unreliable under memory pressure in earlier sessions, and
calling the same internal methods a real timeout would call is an exact,
faster substitute.
"""

import pytest

from interface.gui.busy_tracker import BusyTracker
from interface.gui.loading_indicator import LoadingIndicator


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _shown_indicator():
    """A LoadingIndicator parented to a shown top-level container, so
    isVisible() reflects the indicator's own show/hide calls instead of
    always being False for lack of any shown ancestor (calling .show() on
    the indicator itself would force it visible regardless of its
    internal debounce state, which is exactly what these tests need to
    observe)."""
    from PySide6.QtWidgets import QWidget
    container = QWidget()
    indicator = LoadingIndicator(container)
    container.show()
    return container, indicator


def test_starts_hidden(app):
    indicator = LoadingIndicator()
    assert indicator.isVisible() is False


def test_show_now_and_hide_now_bypass_the_debounce(app):
    _container, indicator = _shown_indicator()

    indicator.show_now("Importing test.jpg...")
    assert indicator.isVisible() is True
    assert indicator._label.text() == "Importing test.jpg..."

    indicator.hide_now()
    assert indicator.isVisible() is False


def test_attach_does_not_show_immediately_for_a_brief_task(app):
    """The whole point of the debounce: a task that ends before the show-
    delay elapses must never make the indicator visible at all."""
    _container, indicator = _shown_indicator()
    tracker = BusyTracker()
    indicator.attach(tracker)

    tracker.begin("Rendering...")
    assert indicator.isVisible() is False  # show-delay timer hasn't fired yet
    tracker.end("Rendering...")
    assert indicator._show_timer.isActive() is False  # begin's pending show was cancelled
    assert indicator.isVisible() is False


def test_attach_shows_once_the_show_delay_elapses(app):
    _container, indicator = _shown_indicator()
    tracker = BusyTracker()
    indicator.attach(tracker)

    tracker.begin("Rendering...")
    indicator._show_now()  # simulate the show-delay timer firing
    assert indicator.isVisible() is True
    assert indicator._label.text() == "Rendering..."


def test_attach_updates_label_without_reshowing_while_already_visible(app):
    _container, indicator = _shown_indicator()
    tracker = BusyTracker()
    indicator.attach(tracker)

    tracker.begin("Rendering...")
    indicator._show_now()
    tracker.begin("Exporting...")
    assert indicator._label.text() == "Exporting..."
    assert indicator.isVisible() is True


def test_attach_schedules_hide_after_min_visible_once_shown(app):
    _container, indicator = _shown_indicator()
    tracker = BusyTracker()
    indicator.attach(tracker)

    tracker.begin("Rendering...")
    indicator._show_now()
    tracker.end("Rendering...")
    # Still visible immediately after end() - hide is deferred to protect
    # against a single-frame blink, not applied synchronously.
    assert indicator.isVisible() is True
    assert indicator._hide_timer.isActive() is True

    indicator._hide_now()  # simulate the hide-delay timer firing
    assert indicator.isVisible() is False
