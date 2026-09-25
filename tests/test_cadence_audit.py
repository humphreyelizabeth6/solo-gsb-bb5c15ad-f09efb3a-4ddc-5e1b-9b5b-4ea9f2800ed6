"""Unit tests for the read-only sampling-cadence audit engine.

The audit generates expected instants anchored at the range start over a
half-open window, attributes only in-range events within tolerance to their
nearest expected instant (each instant at most once), and merges
consecutive missed instants into 缺采段.
"""

import pytest

from app.engine import (
    MAX_CADENCE_SLOTS,
    cadence_audit,
    expected_sample_times,
)

BASE = 1_000_000.0
P = 60.0


def beat(k, off=0.0):
    return BASE + k * P + off


def audit(events, end_beat=6, tolerance=10.0, start=BASE, period=P):
    return cadence_audit(events, start, start + end_beat * period, period, tolerance)


def ids_at(*spec):
    """events as (beat_index, offset, id) triples -> (ts, id) pairs."""
    return [(beat(k, off), eid) for k, off, eid in spec]


# --------------------------------------------------------------------------- #
# expected instants
# --------------------------------------------------------------------------- #


def test_expected_times_anchored_at_start():
    assert expected_sample_times(BASE, 60.0, BASE + 180.0) == [BASE, BASE + 60, BASE + 120]
    # anchor need not be a "nice" time
    assert expected_sample_times(BASE + 7, 60.0, BASE + 127.0) == [
        BASE + 7,
        BASE + 67,
    ]


def test_expected_times_half_open_drops_end_instant():
    # an instant landing exactly on end is excluded
    assert expected_sample_times(0.0, 60.0, 180.0) == [0.0, 60.0, 120.0]
    assert expected_sample_times(0.0, 60.0, 179.999) == [0.0, 60.0, 120.0]


def test_expected_times_rejects_bad_inputs():
    with pytest.raises(ValueError):
        expected_sample_times(10.0, 60.0, 10.0)  # empty window
    with pytest.raises(ValueError):
        expected_sample_times(0.0, 0.0, 60.0)
    with pytest.raises(ValueError):
        expected_sample_times(0.0, -5.0, 60.0)
    with pytest.raises(ValueError):
        expected_sample_times(0.0, 1.0, float(MAX_CADENCE_SLOTS) + 1.0)


# --------------------------------------------------------------------------- #
# hit / miss accounting
# --------------------------------------------------------------------------- #


def test_all_hit():
    r = audit(ids_at((0, 0, 1), (1, 0, 2), (2, 1, 3), (3, -2, 4), (4, 0, 5), (5, 0, 6)))
    assert r.expected_count == 6
    assert r.hit_count == 6
    assert r.miss_count == 0
    assert r.hit_rate == 1.0
    assert r.gaps() == []


def test_consecutive_misses_merge_into_one_gap():
    # beats 1,2,3 missing -> a single gap
    r = audit(ids_at((0, 0, 1), (4, 0, 2), (5, 0, 3)))
    gaps = r.gaps()
    assert len(gaps) == 1
    assert gaps[0].start == beat(1)
    assert gaps[0].end == beat(3)
    assert gaps[0].missed == 3
    assert r.hit_count == 3
    assert r.miss_count == 3
    assert r.hit_rate == pytest.approx(0.5)


def test_separate_miss_runs_stay_separate():
    r = audit(ids_at((0, 0, 1), (2, 0, 2), (4, 0, 3)))  # miss 1, 3, 5
    gaps = r.gaps()
    assert [(g.start, g.end, g.missed) for g in gaps] == [
        (beat(1), beat(1), 1),
        (beat(3), beat(3), 1),
        (beat(5), beat(5), 1),
    ]


def test_gap_first_and_last_for_long_run():
    r = audit(ids_at((0, 0, 1), (1, 0, 2), (8, 0, 3), (9, 0, 4)), end_beat=10)
    gaps = r.gaps()
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end, gaps[0].missed) == (beat(2), beat(7), 6)


def test_no_events_all_miss_one_gap():
    r = audit([])
    assert r.hit_count == 0
    assert r.hit_rate == 0.0
    gaps = r.gaps()
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end, gaps[0].missed) == (beat(0), beat(5), 6)


# --------------------------------------------------------------------------- #
# tolerance / nearest-instant attribution
# --------------------------------------------------------------------------- #


def test_deviation_within_tolerance_counts():
    # tolerance boundary is inclusive (10s)
    r = audit(ids_at((0, 10, 1), (1, -10, 2), (2, 5, 3)), end_beat=3)
    assert r.hit_count == 3
    assert sorted(h.deviation for h in r.hits) == [-10, 5, 10]


def test_event_outside_tolerance_not_counted():
    r = audit(ids_at((0, 11, 1)), end_beat=3, tolerance=10.0)
    assert r.hit_count == 0
    assert r.events_tolerated == 0
    assert r.events_in_range == 1
    assert r.gaps()[0].missed == 3


def test_nearest_instant_wins():
    # event 5s before beat 1 is 55s from beat 0 and 5s from beat 1
    r = audit(ids_at((1, -5, 1), (0, 0, 2)), end_beat=2)
    hit_slots = {round((h.expected_ts - BASE) / P) for h in r.hits}
    assert hit_slots == {0, 1}


def test_tolerance_must_be_under_half_period():
    with pytest.raises(ValueError):
        cadence_audit([], BASE, BASE + 120, 60.0, 30.0)
    with pytest.raises(ValueError):
        cadence_audit([], BASE, BASE + 120, 60.0, 45.0)
    with pytest.raises(ValueError):
        cadence_audit([], BASE, BASE + 120, 60.0, -1.0)
    # strictly below half a period is accepted
    cadence_audit([], BASE, BASE + 120, 60.0, 29.999)


# --------------------------------------------------------------------------- #
# one instant at most once / deterministic assignment
# --------------------------------------------------------------------------- #


def test_same_instant_counts_once_surplus_unassigned():
    # two events both nearest beat 0 and within tolerance -> one hit
    r = audit(ids_at((0, 1, 1), (0, 2, 2)), end_beat=2)
    assert r.hit_count == 1
    assert r.events_tolerated == 2
    assert r.events_unassigned == 1
    # closest event claims the instant
    assert r.hits[0].deviation == 1
    # beat 1 is still missed
    assert r.miss_count == 1


def test_closest_first_global_assignment():
    # precise beat-0 event must not lose its slot to a far (+9s) event
    r = audit(ids_at((0, 9, 1), (0, 0, 2), (1, 9, 3)), end_beat=3, tolerance=10.0)
    by_slot = {round((h.expected_ts - BASE) / P): h for h in r.hits}
    assert by_slot[0].deviation == 0
    assert by_slot[1].deviation == 9
    assert r.hit_count == 2


def test_duplicate_same_timestamp_event_one_hit():
    r = audit([(beat(0), 7), (beat(0), 8)], end_beat=2)
    assert r.hit_count == 1
    assert r.events_unassigned == 1


# --------------------------------------------------------------------------- #
# window / range semantics
# --------------------------------------------------------------------------- #


def test_half_open_window_end_event_excluded():
    r = audit([(beat(0), 1), (beat(6), 99)], end_beat=6)
    assert r.expected_count == 6
    assert r.hit_count == 1
    assert r.events_in_range == 1  # event exactly on end is out of range


def test_start_event_included():
    r = audit([(BASE, 1)], end_beat=1)
    assert r.expected_count == 1 and r.hit_count == 1


def test_events_outside_window_ignored():
    r = audit([(BASE - 60, 1), (BASE, 2), (BASE + 360, 3)], end_beat=6)
    assert r.events_in_range == 1
    assert r.hit_count == 1


def test_hit_record_fields():
    r = audit(ids_at((2, -3, 42)), end_beat=4)
    hit = next(h for h in r.hits if h.event_id == 42)
    assert hit.expected_ts == beat(2)
    assert hit.event_ts == beat(2, -3)
    assert hit.deviation == -3.0
