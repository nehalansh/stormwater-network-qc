"""Pipe slopes from DEMs, constrained by full-flow velocity limits.

Without invert elevations, pipe slopes are estimated from the ground
elevation of the pipe ends and corrected so that the full-flow velocity
(Manning's equation, circular pipe) lies within design limits, by default
1.5-4.6 m/s (Drainage Design Manual for Maricopa County - Hydraulics).
SI units: diameter in m, velocity in m/s.
"""
import numpy as np
import pandas as pd
from rasterio.transform import rowcol


def full_flow_velocity(diameter, slope, n=0.013):
    """Manning full-flow velocity of a circular pipe (NaN for slope <= 0)."""
    slope = np.asarray(slope, float)
    v = (1.0 / n) * (np.asarray(diameter) / 4.0) ** (2 / 3) * np.sqrt(np.where(slope > 0, slope, np.nan))
    return v


def slope_for_velocity(diameter, velocity, n=0.013):
    """Slope giving ``velocity`` at full flow."""
    return (np.asarray(velocity) * n / (np.asarray(diameter) / 4.0) ** (2 / 3)) ** 2


def full_flow_discharge(diameter, slope, n=0.013):
    """Manning full-flow discharge (m3/s) of a circular pipe."""
    return full_flow_velocity(diameter, slope, n) * np.pi * np.asarray(diameter) ** 2 / 4


def diameter_for_discharge(discharge, slope, n=0.013):
    """Diameter carrying ``discharge`` at full flow on ``slope``."""
    k = 4 ** (5 / 3) / np.pi  # = 3.208 (Q = D^(8/3) S^(1/2) / (k n))
    return (np.asarray(discharge) * k * n / np.sqrt(np.asarray(slope))) ** (3 / 8)


def sample_dem(junctions, dem, transform):
    """DEM value at each junction (NaN outside the raster)."""
    r, c = rowcol(transform, junctions.geometry.x.to_numpy(), junctions.geometry.y.to_numpy())
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < dem.shape[0]) & (c >= 0) & (c < dem.shape[1])
    z = np.full(len(r), np.nan)
    z[ok] = dem[r[ok], c[ok]]
    return pd.Series(z, index=junctions.node_id.to_numpy())


def _miss(v, vmin, vmax):
    """Distance of a velocity to [vmin, vmax]; inf if undefined."""
    v = np.asarray(v, float)
    return np.where(np.isnan(v), np.inf, np.maximum(vmin - v, 0) + np.maximum(v - vmax, 0))


def estimate_slopes(segments, junctions, dem, transform, dem_coarse=None,
                    transform_coarse=None, inlet_nodes=(), n=0.013, vmin=1.5, vmax=4.6,
                    max_iter=100):
    """Estimate a slope for every segment.

    Requires ``us_node``/``ds_node`` oriented in the flow direction, ``diameter``
    (m) and ``parent_id`` (the original pipe each segment comes from).

    1. Slope of the whole original pipe: ground elevation of its most upstream
       node minus that of its most downstream node, over its length (fine
       DEM). Long pipes average out local terrain variability.
    2. If the velocity is outside the limits, the same with the coarse DEM,
       then segment by segment (fine, then coarse DEM); a new estimate is kept
       only if it brings the velocity closer to the limits.
    3. Remaining segments outside the limits (or with slope <= 0):
       * connector pipes (starting at an inlet, nothing upstream) get the
         median velocity of the connector pipes within the limits;
       * other pipes get the largest (above vmax or slope <= 0) or smallest
         (below vmin) within-limit velocity among their upstream and
         downstream pipes, or the limit itself if there is none.
       The slope is recovered from the assigned velocity.

    Returns segments with ``slope``, ``velocity`` and ``slope_source``.
    """
    seg = segments.copy()
    L = seg.length.to_numpy()
    D = seg.diameter.to_numpy()
    zf = sample_dem(junctions, dem, transform)
    zc = sample_dem(junctions, dem_coarse, transform_coarse) if dem_coarse is not None else None

    def try_update(mask, new_slope, label):
        nonlocal slope, source
        new_v = full_flow_velocity(D, new_slope, n)
        better = mask & (_miss(new_v, vmin, vmax) < _miss(full_flow_velocity(D, slope, n), vmin, vmax))
        slope = np.where(better, new_slope, slope)
        source = np.where(better, label, source)

    # 1-2: whole original pipes
    slope = np.full(len(seg), np.nan)
    source = np.full(len(seg), "none", dtype=object)
    groups = seg.groupby("parent_id").indices
    parent_ends = {}
    for pid, idx in groups.items():
        sub = seg.iloc[idx]
        heads = set(sub.us_node) - set(sub.ds_node)
        tails = set(sub.ds_node) - set(sub.us_node)
        if len(heads) == 1 and len(tails) == 1:
            parent_ends[pid] = (heads.pop(), tails.pop(), L[idx].sum())
    for label, z in (("parent_fine_dem", zf), ("parent_coarse_dem", zc)):
        if z is None:
            continue
        s = np.full(len(seg), np.nan)
        for pid, (a, b, length) in parent_ends.items():
            s[groups[pid]] = (z[a] - z[b]) / length
        out = _miss(full_flow_velocity(D, slope, n), vmin, vmax) > 0
        try_update(out, s, label)
    for label, z in (("segment_fine_dem", zf), ("segment_coarse_dem", zc)):
        if z is None:
            continue
        s = (z.loc[seg.us_node].to_numpy() - z.loc[seg.ds_node].to_numpy()) / L
        out = _miss(full_flow_velocity(D, slope, n), vmin, vmax) > 0
        try_update(out, s, label)

    # 3: velocity rules
    sid = seg.segment_id.to_numpy()
    pos = {s_: i for i, s_ in enumerate(sid)}
    up = seg.groupby("ds_node").segment_id.apply(list)
    down = seg.groupby("us_node").segment_id.apply(list)
    neighbours = [[pos[x] for x in up.get(u, []) + down.get(d, []) if x != s_]
                  for s_, u, d in zip(sid, seg.us_node, seg.ds_node)]
    inlet_nodes = set(inlet_nodes)
    connector = np.array([u in inlet_nodes and not up.get(u) for u in seg.us_node])
    for _ in range(max_iter):
        v = full_flow_velocity(D, slope, n)
        ok = (v >= vmin) & (v <= vmax)
        bad = np.flatnonzero(~ok)
        if not len(bad):
            break
        conn_median = np.nanmedian(v[connector & ok]) if (connector & ok).any() else vmin
        target = np.empty(len(bad))
        label = np.empty(len(bad), dtype=object)
        for k, i in enumerate(bad):
            if connector[i]:
                target[k], label[k] = conn_median, "connector_median"
                continue
            nb = [v[j] for j in neighbours[i] if ok[j]]
            high = np.isnan(v[i]) or v[i] > vmax
            if nb:
                target[k], label[k] = (max(nb) if high else min(nb)), "neighbour_velocity"
            else:
                target[k], label[k] = (vmax if v[i] > vmax else vmin), "velocity_limit"
        slope[bad] = slope_for_velocity(D[bad], target, n)
        source[bad] = label
    seg["slope"] = slope
    seg["velocity"] = full_flow_velocity(D, slope, n)
    seg["slope_source"] = source
    return seg
