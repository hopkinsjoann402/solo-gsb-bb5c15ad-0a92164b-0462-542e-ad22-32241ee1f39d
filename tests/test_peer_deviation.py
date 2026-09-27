"""展柜同伴基线偏离查询：纯算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, CaseState, Reading, SessionLocal, engine, init_db
from app.main import app
from app.peer import compute_peer_deviation
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _sample(minute: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(seconds=60 * minute), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _compute(target, refs, dimension="temperature", *, start=BASE, end=None,
             step=60.0, max_gap=600.0, threshold=2.0, min_duration=0.0):
    return compute_peer_deviation(
        target,
        refs,
        dimension=dimension,
        start=start,
        end=end if end is not None else BASE + dt.timedelta(seconds=60 * 10),
        step_seconds=step,
        max_gap_seconds=max_gap,
        deviation_threshold=threshold,
        min_duration_seconds=min_duration,
    )


# ---------- 纯算法：基本判定与方向 ----------


def test_basic_above_period():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0) for i in range(11)]
    res = _compute(target, refs)
    # 基线 = median(20, 22) = 21；diff = 4 ≥ 2 → 全部 11 刻标记
    assert res.grid_point_count == 11
    assert res.target_valid_count == 11
    assert res.baseline_valid_count == 11
    assert res.evaluated_count == 11
    assert res.deviation_count == 11
    assert len(res.periods) == 1
    p = res.periods[0]
    assert p.start_ts == BASE
    assert p.end_ts == BASE + dt.timedelta(seconds=600)
    assert p.duration_seconds == 600.0
    assert p.point_count == 11
    assert p.max_deviation == pytest.approx(4.0)
    assert p.peak_ts == BASE  # 并列取最早
    assert p.direction == "above"


def test_below_direction():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    target = [_sample(i, 15.0) for i in range(11)]  # diff = -6
    res = _compute(target, refs)
    assert len(res.periods) == 1
    assert res.periods[0].direction == "below"
    assert res.periods[0].max_deviation == pytest.approx(6.0)


def test_threshold_inclusive():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0) for i in range(11)]  # |diff| = 4
    res = _compute(target, refs, threshold=4.0)  # 达到阈值即标记（含等于）
    assert res.deviation_count == 11
    res2 = _compute(target, refs, threshold=4.0001)
    assert res2.deviation_count == 0
    assert res2.periods == []


def test_odd_reference_count_median():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 21.0) for i in range(11)],
        [_sample(i, 24.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0) for i in range(11)]
    res = _compute(target, refs)
    # 基线 = median(20, 21, 24) = 21 → diff = 4
    assert res.periods[0].max_deviation == pytest.approx(4.0)


def test_humidity_dimension():
    refs = [
        [_sample(i, 20.0, 40.0) for i in range(11)],
        [_sample(i, 20.0, 44.0) for i in range(11)],
    ]
    target = [_sample(i, 20.0, 60.0) for i in range(11)]
    res = _compute(target, refs, "humidity", threshold=5.0)
    # 基线 = 42，diff = 18
    assert res.periods[0].max_deviation == pytest.approx(18.0)
    assert res.periods[0].direction == "above"
    # 温度维 diff = 0 → 无偏离
    res_t = _compute(target, refs, "temperature", threshold=5.0)
    assert res_t.periods == []


# ---------- 纯算法：基线有效性与断档 ----------


def test_fewer_than_two_valid_references_interrupts():
    # ref1 只覆盖 0..4 分钟；5..10 分钟只剩 1 个有效对照 → 那些刻中断
    refs = [
        [_sample(i, 20.0) for i in range(5)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0) for i in range(11)]
    res = _compute(target, refs)
    assert res.baseline_valid_count == 5  # 0..4 分钟
    assert res.evaluated_count == 5
    assert len(res.periods) == 1
    assert res.periods[0].end_ts == BASE + dt.timedelta(seconds=240)


def test_no_valid_references_returns_empty():
    refs = [[], []]  # 对照没有任何读数：不报错，只是无有效值
    target = [_sample(i, 25.0) for i in range(11)]
    res = _compute(target, refs)
    assert res.baseline_valid_count == 0
    assert res.evaluated_count == 0
    assert res.deviation_count == 0
    assert res.periods == []


def test_target_gap_interrupts_and_splits_periods():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    # 目标在 2→8 分钟之间有 6 分钟断档（360s > max_gap=300）
    target = [_sample(i, 25.0) for i in (0, 1, 2, 8, 9, 10)]
    res = _compute(target, refs, max_gap=300.0)
    # 目标在 3..7 分钟的网格点无效 → 切成 [0..2] 与 [8..10] 两段
    assert res.target_valid_count == 6
    assert len(res.periods) == 2
    assert res.periods[0].start_ts == BASE
    assert res.periods[0].end_ts == BASE + dt.timedelta(seconds=120)
    assert res.periods[1].start_ts == BASE + dt.timedelta(seconds=480)
    assert res.periods[1].end_ts == BASE + dt.timedelta(seconds=600)


def test_reference_gap_invalidates_baseline_moments():
    # ref1 在 0→10 分钟有 10 分钟断档（max_gap=300）：中间刻无基线
    refs = [
        [_sample(0, 20.0), _sample(10, 20.0)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0) for i in range(11)]
    res = _compute(target, refs, max_gap=300.0)
    # 仅 0 与 10 分钟两刻 ref1 有效（读数点本身），基线只在这两刻形成
    assert res.baseline_valid_count == 2
    assert res.evaluated_count == 2
    # 两个孤立标记点不合并：各成单点时段
    assert len(res.periods) == 2
    assert all(p.point_count == 1 and p.duration_seconds == 0.0 for p in res.periods)


def test_gap_equal_to_max_gap_still_continuous():
    refs = [
        [_sample(0, 20.0), _sample(5, 22.0)],
        [_sample(0, 22.0), _sample(5, 24.0)],
    ]
    target = [_sample(0, 25.0), _sample(5, 27.0)]
    res = _compute(
        target, refs,
        end=BASE + dt.timedelta(seconds=300), step=60.0, max_gap=300.0,
    )
    # 间隔 300s 恰好等于 max_gap：中间插值点全部有效
    assert res.grid_point_count == 6
    assert res.evaluated_count == 6
    assert len(res.periods) == 1
    assert res.periods[0].duration_seconds == 300.0


def test_grid_floored_to_end():
    refs = [[_sample(i, 20.0) for i in range(11)], [_sample(i, 22.0) for i in range(11)]]
    target = [_sample(i, 25.0) for i in range(11)]
    res = _compute(
        target, refs,
        end=BASE + dt.timedelta(seconds=630),  # 10.5 分钟：尾部不足一步不补点
    )
    assert res.grid_point_count == 11


# ---------- 纯算法：合并、最短持续与峰值 ----------


def test_min_duration_filters_short_periods():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    # 目标在 2→8 分钟断档：两段各持续 120 秒
    target = [_sample(i, 25.0) for i in (0, 1, 2, 8, 9, 10)]
    res = _compute(target, refs, max_gap=300.0, min_duration=120.0)
    assert len(res.periods) == 2  # 含等于：120 ≥ 120 保留
    res2 = _compute(target, refs, max_gap=300.0, min_duration=121.0)
    assert res2.periods == []
    # 计数字段不受 min_duration 过滤影响
    assert res2.deviation_count == 6


def test_single_point_period_zero_duration():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    # 目标只在第 5 分钟偏离（25.0），其余与基线持平（21.0）
    target = [_sample(i, 21.0) for i in range(11)]
    target[5] = _sample(5, 25.0)
    res = _compute(target, refs)
    assert len(res.periods) == 1
    p = res.periods[0]
    assert p.point_count == 1
    assert p.duration_seconds == 0.0
    assert p.start_ts == p.end_ts == BASE + dt.timedelta(seconds=300)
    assert p.max_deviation == pytest.approx(4.0)
    # min_duration > 0 时单点时段被过滤
    res2 = _compute(target, refs, min_duration=1.0)
    assert res2.periods == []


def test_peak_tie_takes_earliest():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    # 第 3、7 分钟同为最大偏离 6.0 → 峰值取最早的第 3 分钟
    values = [21.0] * 11
    values[3] = values[7] = 27.0
    values[5] = 24.0
    target = [_sample(i, values[i]) for i in range(11)]
    res = _compute(target, refs)
    # 3、5、7 分钟互不相邻 → 三个独立单点时段
    assert len(res.periods) == 3
    peaks = {p.peak_ts for p in res.periods}
    assert BASE + dt.timedelta(seconds=180) in peaks
    big = [p for p in res.periods if p.max_deviation == pytest.approx(6.0)]
    assert len(big) == 2


def test_peak_within_merged_period():
    refs = [
        [_sample(i, 20.0) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    # 连续偏离段 2..6 分钟，峰值在第 4 分钟
    values = [21.0] * 11
    for i in range(2, 7):
        values[i] = 24.0
    values[4] = 26.0
    target = [_sample(i, values[i]) for i in range(11)]
    res = _compute(target, refs)
    assert len(res.periods) == 1
    p = res.periods[0]
    assert p.start_ts == BASE + dt.timedelta(seconds=120)
    assert p.end_ts == BASE + dt.timedelta(seconds=360)
    assert p.point_count == 5
    assert p.max_deviation == pytest.approx(5.0)
    assert p.peak_ts == BASE + dt.timedelta(seconds=240)
    assert p.direction == "above"


def test_interpolation_between_readings():
    # 对照与目标都只在 0、10 分钟有读数：网格点全部走线性插值
    refs = [
        [_sample(0, 20.0), _sample(10, 30.0)],
        [_sample(0, 22.0), _sample(10, 32.0)],
    ]
    target = [_sample(0, 30.0), _sample(10, 40.0)]
    res = _compute(target, refs, threshold=5.0)
    # 基线(t) = 21 + t/60；目标(t) = 30 + t/60 → diff 恒为 9
    assert res.evaluated_count == 11
    assert res.periods[0].max_deviation == pytest.approx(9.0)


def test_unsorted_input_deterministic():
    refs = [
        [_sample(i, 20.0 + (i % 3)) for i in range(11)],
        [_sample(i, 22.0) for i in range(11)],
    ]
    target = [_sample(i, 25.0 - (i % 2)) for i in range(11)]
    r1 = _compute(target, refs)
    r2 = _compute(
        list(reversed(target)), [list(reversed(r)) for r in refs]
    )
    assert r1 == r2


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _minute_readings(sn, values, dimension_value_h=50.0):
    return [
        _reading(
            sn,
            (BASE + dt.timedelta(seconds=60 * i)).isoformat().replace("+00:00", "Z"),
            v,
            dimension_value_h,
        )
        for i, v in enumerate(values)
    ]


def _params(**over):
    p = {
        "reference_sn": ["CASE-R1", "CASE-R2"],
        "dimension": "temperature",
        "start": "2026-09-01T00:00:00Z",
        "end": "2026-09-01T00:10:00Z",
        "step_seconds": 60,
        "max_gap_seconds": 600,
        "deviation_threshold": 2.0,
    }
    p.update(over)
    return p


def _seed_basic(client):
    _ingest(client, "CASE-R1", _minute_readings("CASE-R1", [20.0] * 11))
    _ingest(client, "CASE-R2", _minute_readings("CASE-R2", [22.0] * 11))
    _ingest(client, "CASE-T", _minute_readings("CASE-T", [25.0] * 11))


def test_endpoint_basic(client):
    _seed_basic(client)
    r = client.get("/api/v1/cases/CASE-T/peer-deviation", params=_params())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-T"
    assert body["reference_sns"] == ["CASE-R1", "CASE-R2"]
    assert body["dimension"] == "temperature"
    assert body["grid_point_count"] == 11
    assert body["target_valid_count"] == 11
    assert body["baseline_valid_count"] == 11
    assert body["evaluated_count"] == 11
    assert body["deviation_count"] == 11
    assert body["period_count"] == 1
    p = body["periods"][0]
    assert p["start_ts"] == "2026-09-01T00:00:00Z"
    assert p["end_ts"] == "2026-09-01T00:10:00Z"
    assert p["duration_seconds"] == 600.0
    assert p["point_count"] == 11
    assert p["max_deviation"] == pytest.approx(4.0)
    assert p["peak_ts"] == "2026-09-01T00:00:00Z"
    assert p["direction"] == "above"


def test_endpoint_unknown_target_404(client):
    _seed_basic(client)
    r = client.get("/api/v1/cases/CASE-GHOST/peer-deviation", params=_params())
    assert r.status_code == 404


def test_endpoint_missing_reference_is_not_404(client):
    # 对照展柜没有读数：不报错，基线无法形成 → 空结果
    _ingest(client, "CASE-T2", _minute_readings("CASE-T2", [25.0] * 11))
    r = client.get(
        "/api/v1/cases/CASE-T2/peer-deviation",
        params=_params(reference_sn=["CASE-G1", "CASE-G2"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["baseline_valid_count"] == 0
    assert body["evaluated_count"] == 0
    assert body["periods"] == []
    assert body["period_count"] == 0


def test_endpoint_insufficient_references_empty_result(client):
    # 两个对照中一个无读数：每刻只剩 1 个有效对照值 → 无基线 → 空结果
    _ingest(client, "CASE-R1", _minute_readings("CASE-R1", [20.0] * 11))
    _ingest(client, "CASE-T3", _minute_readings("CASE-T3", [25.0] * 11))
    r = client.get(
        "/api/v1/cases/CASE-T3/peer-deviation",
        params=_params(reference_sn=["CASE-R1", "CASE-GHOST"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["baseline_valid_count"] == 0
    assert body["periods"] == []


def test_endpoint_no_deviation_empty_periods(client):
    _seed_basic(client)
    _ingest(client, "CASE-T4", _minute_readings("CASE-T4", [21.0] * 11))
    r = client.get("/api/v1/cases/CASE-T4/peer-deviation", params=_params())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evaluated_count"] == 11
    assert body["deviation_count"] == 0
    assert body["periods"] == []


def test_endpoint_duplicate_reference_422(client):
    _seed_basic(client)
    r = client.get(
        "/api/v1/cases/CASE-T/peer-deviation",
        params=_params(reference_sn=["CASE-R1", "CASE-R1"]),
    )
    assert r.status_code == 422


def test_endpoint_reference_contains_target_422(client):
    _seed_basic(client)
    r = client.get(
        "/api/v1/cases/CASE-T/peer-deviation",
        params=_params(reference_sn=["CASE-R1", "CASE-T"]),
    )
    assert r.status_code == 422


def test_endpoint_reference_count_bounds(client):
    _seed_basic(client)
    # 少于 2 个对照
    r = client.get(
        "/api/v1/cases/CASE-T/peer-deviation",
        params=_params(reference_sn=["CASE-R1"]),
    )
    assert r.status_code == 422
    # 多于 10 个对照
    r = client.get(
        "/api/v1/cases/CASE-T/peer-deviation",
        params=_params(reference_sn=[f"CASE-X{i}" for i in range(11)]),
    )
    assert r.status_code == 422
    # 缺少对照参数
    p = _params()
    del p["reference_sn"]
    assert client.get("/api/v1/cases/CASE-T/peer-deviation", params=p).status_code == 422


def test_endpoint_invalid_time_range_422(client):
    _seed_basic(client)
    r = client.get(
        "/api/v1/cases/CASE-T/peer-deviation",
        params=_params(start="2026-09-02T00:00:00Z", end="2026-09-01T00:00:00Z"),
    )
    assert r.status_code == 422


def test_endpoint_param_validation_422(client):
    _seed_basic(client)

    def status(**over):
        return client.get(
            "/api/v1/cases/CASE-T/peer-deviation", params=_params(**over)
        ).status_code

    assert status(step_seconds=0) == 422
    assert status(step_seconds=-60) == 422
    assert status(max_gap_seconds=0) == 422
    assert status(deviation_threshold=0) == 422
    assert status(deviation_threshold=-1) == 422
    assert status(min_duration_seconds=-1) == 422
    assert status(dimension="pressure") == 422
    # 缺少必填参数
    assert client.get("/api/v1/cases/CASE-T/peer-deviation").status_code == 422


def test_endpoint_min_duration_filter(client):
    _seed_basic(client)
    # 目标在 2→8 分钟断档（360s > 300）：两段各持续 120 秒
    rows = [
        _reading(
            "CASE-T5",
            (BASE + dt.timedelta(seconds=60 * i)).isoformat().replace("+00:00", "Z"),
            25.0,
            50.0,
        )
        for i in (0, 1, 2, 8, 9, 10)
    ]
    _ingest(client, "CASE-T5", rows)
    p = _params(max_gap_seconds=300, min_duration_seconds=120)
    body = client.get("/api/v1/cases/CASE-T5/peer-deviation", params=p).json()
    assert body["period_count"] == 2
    p["min_duration_seconds"] = 121
    body = client.get("/api/v1/cases/CASE-T5/peer-deviation", params=p).json()
    assert body["period_count"] == 0
    assert body["deviation_count"] == 6  # 标记计数不受过滤影响


def test_endpoint_is_read_only(client):
    """同伴基线偏离查询不得改变读数、分段与版本号。"""
    _seed_basic(client)
    db = SessionLocal()
    try:
        version_before = db.get(CaseState, "CASE-T").version
        count_before = db.query(Reading).filter(Reading.device_sn == "CASE-T").count()
    finally:
        db.close()

    url = "/api/v1/cases/CASE-T/peer-deviation"
    assert client.get(url, params=_params()).status_code == 200
    assert client.get(url, params=_params()).status_code == 200

    db = SessionLocal()
    try:
        db.expire_all()
        assert db.get(CaseState, "CASE-T").version == version_before
        assert db.query(Reading).filter(Reading.device_sn == "CASE-T").count() == count_before
    finally:
        db.close()


def test_endpoint_deterministic(client):
    _seed_basic(client)
    url = "/api/v1/cases/CASE-T/peer-deviation"
    first = client.get(url, params=_params()).json()
    second = client.get(url, params=_params()).json()
    assert first == second
