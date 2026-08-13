"""Tests for interface/gui/busy_tracker.py."""

from interface.gui.busy_tracker import BusyTracker


def test_starts_idle():
    tracker = BusyTracker()
    assert tracker.is_busy() is False


def test_begin_makes_it_busy_and_emits_the_label():
    tracker = BusyTracker()
    events = []
    tracker.busyChanged.connect(lambda busy, label: events.append((busy, label)))

    tracker.begin("Rendering...")

    assert tracker.is_busy() is True
    assert events == [(True, "Rendering...")]


def test_end_makes_it_idle_and_emits_empty_label():
    tracker = BusyTracker()
    events = []
    tracker.begin("Rendering...")
    tracker.busyChanged.connect(lambda busy, label: events.append((busy, label)))

    tracker.end("Rendering...")

    assert tracker.is_busy() is False
    assert events == [(False, "")]


def test_overlapping_operations_stay_busy_until_all_end():
    tracker = BusyTracker()
    tracker.begin("Rendering...")
    tracker.begin("Exporting...")
    assert tracker.is_busy() is True

    tracker.end("Rendering...")
    assert tracker.is_busy() is True  # "Exporting..." still in flight

    tracker.end("Exporting...")
    assert tracker.is_busy() is False


def test_ending_reports_whichever_label_is_still_active():
    tracker = BusyTracker()
    events = []
    tracker.begin("Rendering...")
    tracker.begin("Exporting...")
    tracker.busyChanged.connect(lambda busy, label: events.append((busy, label)))

    tracker.end("Exporting...")

    assert events == [(True, "Rendering...")]


def test_end_with_no_matching_begin_is_a_no_op():
    tracker = BusyTracker()
    tracker.begin("Rendering...")
    tracker.end("SomethingElse")
    assert tracker.is_busy() is True  # the real "Rendering..." entry is untouched
