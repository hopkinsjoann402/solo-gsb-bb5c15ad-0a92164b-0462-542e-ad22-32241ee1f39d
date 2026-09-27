"""展柜同伴基线偏离查询（peer baseline deviation，纯计算，只读）。

以查询起点 ``start`` 为原点按固定步长 ``step_seconds`` 生成**等距时刻
网格**（``t_i = start + i·step``，``t_i ≤ end``）。目标展柜与每个对照
展柜各自的读数视为**分段线性**曲线：仅在相邻读数间隔不超过
``max_gap_seconds`` 时线性插值（恰好等于仍连续，否则该刻该柜无效；
落在覆盖范围外的时刻同样无效）。

每个网格时刻至少要有 **两个有效对照值** 才取其中位数（偶数取中间两数
均值）作为该刻基线，否则该刻**中断判定**；目标值无效的时刻同样中断。
目标值与基线之差的绝对值**达到** ``deviation_threshold``（含等于）即
标记偏离；连续标记按相邻网格时差不超过步长合并为时段（网格严格等距，
相邻标记时刻之差恒为步长，天然满足），仅返回持续时间 **≥**
``min_duration_seconds``（含等于）的时段，并给出时段内**最大偏离值**
（绝对偏差最大处，并列取最早）、**峰值时刻**与**方向**
（``above``：目标高于基线 / ``below``：目标低于基线）。

全部计算确定性：相同输入恒得相同输出；不修改任何输入。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass
from statistics import median
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

# 偏离方向
ABOVE = "above"  # 峰值处目标值高于基线
BELOW = "below"  # 峰值处目标值低于基线

# 每个网格时刻参与中位数基线的有效对照值下限
MIN_REFERENCE_VALUES = 2


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class DeviationPeriod:
    """一段连续偏离时段（起止均为被标记的网格时刻）。"""

    start_ts: dt.datetime  # 段内首个被标记的网格时刻
    end_ts: dt.datetime  # 段内末个被标记的网格时刻
    duration_seconds: float  # end_ts − start_ts
    point_count: int  # 段内被标记的网格时刻数
    max_deviation: float  # 段内 |目标值 − 基线| 的最大值
    peak_ts: dt.datetime  # 最大偏离对应的网格时刻（并列取最早）
    direction: str  # 峰值处目标相对基线的方向：above / below


@dataclass(frozen=True)
class PeerDeviationResult:
    """同伴基线偏离计算结果。"""

    grid_point_count: int  # 网格时刻总数
    target_valid_count: int  # 目标展柜插值有效的时刻数
    baseline_valid_count: int  # 有效对照值 ≥ 2、可形成基线的时刻数
    evaluated_count: int  # 目标与基线同时有效、实际参与判定的时刻数
    deviation_count: int  # 被标记偏离的时刻数
    periods: list[DeviationPeriod]


def _dimension_value(sample: _SampleLike, dimension: str) -> float:
    if dimension == TEMPERATURE:
        return sample.temperature
    if dimension == HUMIDITY:
        return sample.humidity
    raise ValueError(f"未知维度: {dimension}")  # pragma: no cover - 接口层已枚举校验


def _series(
    samples: list[_SampleLike], dimension: str
) -> tuple[list[dt.datetime], list[float]]:
    ordered = sorted(samples, key=lambda s: s.ts)
    return [s.ts for s in ordered], [_dimension_value(s, dimension) for s in ordered]


def _interpolate_at(
    times: list[dt.datetime],
    values: list[float],
    ts: dt.datetime,
    max_gap_seconds: float,
) -> float | None:
    """在分段线性曲线上求 ``ts`` 处的值。

    - ``ts`` 落在覆盖范围外返回 ``None``；
    - 所依赖的相邻读数间隔大于 ``max_gap_seconds``（断档）返回 ``None``；
    - 恰好落在读数时刻直接取该读数；断档判定含等于（等于 max_gap 连续）。
    """
    n = len(times)
    if n == 0 or ts < times[0] or ts > times[-1]:
        return None
    i = bisect_left(times, ts)
    if i < n and times[i] == ts:
        return values[i]
    # i >= 1（ts > times[0]）且 i-1, i 构成包围 ts 的相邻读数
    t0, t1 = times[i - 1], times[i]
    if (t1 - t0).total_seconds() > max_gap_seconds:
        return None
    v0, v1 = values[i - 1], values[i]
    frac = (ts - t0).total_seconds() / (t1 - t0).total_seconds()
    return v0 + (v1 - v0) * frac


def _build_grid(
    start: dt.datetime, end: dt.datetime, step_seconds: float
) -> list[dt.datetime]:
    """以 ``start`` 为原点生成不超过 ``end`` 的等距网格（含两端边界）。"""
    if end < start:
        return []
    step_us = int(round(step_seconds * 1_000_000))
    span_us = (end - start) // dt.timedelta(microseconds=1)
    n = span_us // step_us
    return [start + dt.timedelta(microseconds=step_us * k) for k in range(n + 1)]


def compute_peer_deviation(
    target_samples: list[_SampleLike],
    reference_samples: list[list[_SampleLike]],
    *,
    dimension: str,
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: float,
    max_gap_seconds: float,
    deviation_threshold: float,
    min_duration_seconds: float,
) -> PeerDeviationResult:
    """计算目标展柜相对对照组中位数基线的偏离时段（只读纯计算）。

    ``reference_samples`` 与请求中的对照展柜一一对应（可为空序列，
    空对照在任何时刻都不贡献有效值）。参数合法性（时间范围、步长、
    断档、阈值、最短持续、对照数量与去重）由接口层校验，本函数假定
    入参合法；samples 无需预先排序。
    """
    grid = _build_grid(start, end, step_seconds)

    tgt_times, tgt_values = _series(list(target_samples), dimension)
    ref_series = [_series(list(samples), dimension) for samples in reference_samples]

    target_grid = [
        _interpolate_at(tgt_times, tgt_values, t, max_gap_seconds) for t in grid
    ]
    ref_grids = [
        [_interpolate_at(times, values, t, max_gap_seconds) for t in grid]
        for times, values in ref_series
    ]

    # 逐刻求基线并判定：diff[i] 仅在目标与基线同时有效时有值，
    # 否则为 None（该刻中断判定，前后标记绝不跨它合并）
    diffs: list[float | None] = []
    baseline_valid_count = 0
    for i in range(len(grid)):
        valid_refs = [g[i] for g in ref_grids if g[i] is not None]
        baseline: float | None = None
        if len(valid_refs) >= MIN_REFERENCE_VALUES:
            baseline = float(median(valid_refs))
            baseline_valid_count += 1
        target = target_grid[i]
        if baseline is None or target is None:
            diffs.append(None)
        else:
            diffs.append(target - baseline)

    evaluated = [d for d in diffs if d is not None]
    marks = [d is not None and abs(d) >= deviation_threshold for d in diffs]

    # 连续标记合并：网格严格等距，相邻标记时刻之差恒为步长（≤ 步长），
    # 直接按下标连续归并；中断刻（diff 为 None）天然切开两侧
    periods: list[DeviationPeriod] = []
    n = len(grid)
    i = 0
    while i < n:
        if not marks[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and marks[j + 1]:
            j += 1
        start_ts, end_ts = grid[i], grid[j]
        duration = (end_ts - start_ts).total_seconds()
        if duration >= min_duration_seconds:
            # 段内最大偏离：升序扫描 + 严格不等号，并列取最早
            peak = i
            for k in range(i + 1, j + 1):
                if abs(diffs[k]) > abs(diffs[peak]):  # type: ignore[arg-type]
                    peak = k
            peak_diff = diffs[peak]
            assert peak_diff is not None
            periods.append(
                DeviationPeriod(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration_seconds=duration,
                    point_count=j - i + 1,
                    max_deviation=abs(peak_diff),
                    peak_ts=grid[peak],
                    direction=ABOVE if peak_diff > 0 else BELOW,
                )
            )
        i = j + 1

    return PeerDeviationResult(
        grid_point_count=n,
        target_valid_count=sum(v is not None for v in target_grid),
        baseline_valid_count=baseline_valid_count,
        evaluated_count=len(evaluated),
        deviation_count=sum(marks),
        periods=periods,
    )
