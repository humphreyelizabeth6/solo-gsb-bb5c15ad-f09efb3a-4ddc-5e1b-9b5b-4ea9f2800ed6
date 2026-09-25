"""End-to-end tests for the read-only sampling-cadence audit API.

The audit selects an existing channel, reads that channel's stored events
inside the half-open window and checks them against the expected beat.  It
never inserts/rewrites events and never touches fault results.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch

DAY = 86400.0
BASE = to_epoch("2026-09-23T00:00:00Z")
P = 60.0


def beat(k, off=0.0):
    return BASE + k * P + off


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


@pytest.fixture()
def populated(scratch_db):
    # pump-1/temp: beats 0,1,2 present; beat 3 missing; beats 4,5 present.
    # beat 5 is +10s (inside the ±10s tolerance).
    rows = [
        {"device": "pump-1", "metric": "temp", "ts": beat(0), "value": 60},
        {"device": "pump-1", "metric": "temp", "ts": beat(1, 1), "value": 61},
        {"device": "pump-1", "metric": "temp", "ts": beat(2, -2), "value": 62},
        {"device": "pump-1", "metric": "temp", "ts": beat(4), "value": 64},
        {"device": "pump-1", "metric": "temp", "ts": beat(5, 10), "value": 65},
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows), reevaluate=False)
    return rows


def _params(**over):
    p = {
        "device": "pump-1",
        "metric": "temp",
        "start": BASE,
        "end": BASE + 6 * P,
        "period": P,
        "tolerance": 10.0,
    }
    p.update(over)
    return p


# --------------------------------------------------------------------------- #
# service level
# --------------------------------------------------------------------------- #


def test_service_summary_and_gap(populated):
    res = services.query_cadence_audit("pump-1", "temp", BASE, BASE + 6 * P, P, 10.0)
    assert res["expected_count"] == 6
    assert res["hit_count"] == 5
    assert res["miss_count"] == 1
    assert res["hit_rate"] == pytest.approx(5 / 6)
    assert res["anchor"] == BASE
    assert res["period_seconds"] == P and res["tolerance_seconds"] == 10.0
    assert res["gaps"] == [{"start": beat(3), "end": beat(3), "missed": 1}]
    assert res["events_in_range"] == 5


def test_service_two_consecutive_gap_runs(scratch_db):
    rows = [
        {"device": "d", "metric": "m", "ts": beat(0), "value": 1},
        {"device": "d", "metric": "m", "ts": beat(3), "value": 4},
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows), reevaluate=False)
    res = services.query_cadence_audit("d", "m", BASE, BASE + 5 * P, P, 5.0)
    assert res["expected_count"] == 5
    assert res["hit_count"] == 2
    assert [(g["start"], g["end"], g["missed"]) for g in res["gaps"]] == [
        (beat(1), beat(2), 2),
        (beat(4), beat(4), 1),
    ]


def test_service_unknown_channel_raises_keyerror(populated):
    with pytest.raises(KeyError):
        services.query_cadence_audit("ghost", "temp", BASE, BASE + 6 * P, P, 10.0)


def test_service_validates_inputs(populated):
    good = dict(device="pump-1", metric="temp", start=BASE, end=BASE + 6 * P,
                period=P, tolerance=10.0)
    def expect_value_error(**over):
        kw = {**good, **over}
        with pytest.raises(ValueError):
            services.query_cadence_audit(**kw)

    expect_value_error(start=BASE + 6 * P, end=BASE)          # inverted
    expect_value_error(start=BASE, end=BASE)                 # zero length
    expect_value_error(period=0)
    expect_value_error(period=-10)
    expect_value_error(tolerance=-1)
    expect_value_error(tolerance=30)                         # >= half period
    expect_value_error(tolerance=P / 2)


def test_service_is_read_only(populated):
    event_count = db.query_one("SELECT COUNT(*) AS c FROM events")["c"]
    fault_count = db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"]
    services.query_cadence_audit("pump-1", "temp", BASE, BASE + 6 * P, P, 10.0)
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == event_count
    assert db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"] == fault_count


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


def test_api_audit_ok(populated, client):
    resp = client.get("/api/sampling-audit", params=_params())
    assert resp.status_code == 200
    body = resp.json()
    assert body["device"] == "pump-1" and body["metric"] == "temp"
    assert (body["expected_count"], body["hit_count"], body["miss_count"]) == (6, 5, 1)
    assert body["hit_rate"] == pytest.approx(5 / 6)
    assert body["gaps"] == [{"start": beat(3), "end": beat(3), "missed": 1}]


def test_api_accepts_iso_utc_times(populated, client):
    resp = client.get(
        "/api/sampling-audit",
        params=_params(
            start="2026-09-23T00:00:00Z",
            end="2026-09-23T00:06:00Z",
        ),
    )
    assert resp.status_code == 200
    assert resp.json()["hit_count"] == 5


def test_api_unknown_channel_404(populated, client):
    resp = client.get("/api/sampling-audit", params=_params(device="ghost"))
    assert resp.status_code == 404


def test_api_bad_window_400(populated, client):
    resp = client.get(
        "/api/sampling-audit",
        params=_params(end="2026-09-22T00:00:00Z"),
    )
    assert resp.status_code == 400


def test_api_tolerance_half_period_400(populated, client):
    resp = client.get("/api/sampling-audit", params=_params(tolerance=30))
    assert resp.status_code == 400
    assert "half" in resp.json()["detail"]


def test_api_nonpositive_period_400(populated, client):
    # period=0 parses as a float and is rejected by the audit validation (400)
    resp = client.get("/api/sampling-audit", params=_params(period=0))
    assert resp.status_code == 400
    assert "positive" in resp.json()["detail"]


def test_api_negative_tolerance_400(populated, client):
    resp = client.get("/api/sampling-audit", params=_params(tolerance=-3))
    assert resp.status_code == 400
    assert "non-negative" in resp.json()["detail"]


def test_api_missing_params_422(client):
    assert client.get("/api/sampling-audit").status_code == 422
    assert client.get(
        "/api/sampling-audit",
        params={"device": "pump-1", "metric": "temp"},
    ).status_code == 422


def test_api_bad_time_format_400(populated, client):
    resp = client.get(
        "/api/sampling-audit",
        params=_params(start="not-a-time"),
    )
    assert resp.status_code == 400


def test_api_is_read_only(populated, client):
    before_events = db.query_one("SELECT COUNT(*) AS c FROM events")["c"]
    before_faults = db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"]
    resp = client.get("/api/sampling-audit", params=_params())
    assert resp.status_code == 200
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == before_events
    assert db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"] == before_faults
