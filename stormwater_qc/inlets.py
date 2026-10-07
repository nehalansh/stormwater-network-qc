"""Stormwater inlets: selection, snapping, clustering and merging.

In SWMM every sub-catchment drains to one junction. Inlets in municipal
layers are often a few metres off the pipe they feed, attached to pipes that
were removed, or grouped at street corners where each would otherwise get a
tiny catch basin. These functions place each inlet on a junction and reduce
each group of nearby inlets to one representative inlet, preserving the
capacity of the connector pipes that are removed.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from .slopes import diameter_for_discharge, full_flow_discharge


def select_inlets(inlets, segments, distance=1.0):
    """Inlets within ``distance`` of a pipe (Buffer + Select By Location).

    Returns (selected inlets, dropped inlets).
    """
    hit = gpd.sjoin_nearest(inlets[["geometry"]], segments[["geometry"]],
                            max_distance=distance, how="inner")
    keep = inlets.index.isin(hit.index)
    return inlets[keep].copy(), inlets[~keep].copy()


def snap_inlets(inlets, junctions, review_distance=2.0, max_distance=None):
    """Move every inlet to its nearest junction.

    Adds ``node_id``, ``snap_distance`` and ``needs_review`` (displacement larger
    than ``review_distance``, to be checked visually). Inlets farther than
    ``max_distance`` from any junction are left unassigned (node_id = -1).
    """
    jxy = np.c_[junctions.geometry.x, junctions.geometry.y]
    d, k = cKDTree(jxy).query(np.c_[inlets.geometry.x, inlets.geometry.y])
    out = inlets.copy()
    out["node_id"] = junctions.node_id.to_numpy()[k]
    out["snap_distance"] = d
    out["needs_review"] = d > review_distance
    if max_distance is not None:
        far = d > max_distance
        out.loc[far, "node_id"] = -1
    else:
        far = np.zeros(len(d), bool)
    out["geometry"] = [g if f else Point(x, y)
                       for g, f, (x, y) in zip(inlets.geometry, far, jxy[k])]
    return out


def cluster_inlets(inlets, eps=25.0, min_samples=2):
    """DBSCAN clusters of inlets (``cluster_id``); unclustered inlets get their own ID."""
    xy = np.c_[inlets.geometry.x, inlets.geometry.y]
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    single = lab == -1
    lab[single] = lab.max() + 1 + np.arange(single.sum())
    out = inlets.copy()
    out["cluster_id"] = lab + 1
    return out


def cluster_points(inlets, field="cluster_id"):
    """One point per cluster: centroid of the minimum-area rectangle around it

    (ArcGIS Minimum Bounding Geometry, rectangle by area, + Feature To Point).
    """
    hulls = inlets.dissolve(by=field).geometry.minimum_rotated_rectangle()
    return gpd.GeoDataFrame({field: hulls.index.to_numpy()},
                            geometry=hulls.centroid.to_numpy(), crs=inlets.crs)


def representative_inlets(inlets, basins=None, field="cluster_id", basin_field="gridcode"):
    """Choose the inlet each cluster's catch basin drains to.

    With ``basins`` (polygons whose ``basin_field`` equals the cluster ID): the
    single inlet inside the basin if there is exactly one; otherwise the inlet
    farthest from the basin centroid, among those inside or, if none is
    inside, among all inlets of the cluster. Without basins, the inlet
    farthest from the cluster centroid. Adds a boolean ``representative``.
    """
    out = inlets.copy()
    out["representative"] = False
    basins = basins.set_index(basin_field) if basins is not None else None
    for cid, grp in out.groupby(field):
        if basins is not None and cid in basins.index:
            poly = basins.geometry.loc[cid]
            inside = grp[grp.within(poly)]
            centre = poly.centroid
            pool = inside if len(inside) else grp
            if len(inside) == 1:
                out.loc[inside.index, "representative"] = True
                continue
        else:
            centre = grp.union_all().centroid
            pool = grp
        far = pool.geometry.distance(centre).idxmax()
        out.loc[far, "representative"] = True
    return out


def merge_connectors(segments, inlets, n=0.013, field="cluster_id"):
    """Remove the connector pipes of non-representative inlets, preserving capacity.

    For each cluster, the connector pipe (the pipe starting at the inlet's
    junction) of every non-representative inlet is removed if that junction
    has no other pipe; the inlet is kept otherwise, so that the network is not
    broken. The full-flow discharges of the removed connectors are added to
    that of the representative inlet's connector, whose diameter is
    recomputed with Manning's equation at its slope.

    Requires ``slope`` (m/m) and ``diameter`` (m) on the segments and
    ``node_id``/``representative`` on the inlets. Returns (segments, inlets
    with ``role``: representative, kept_for_connectivity or removed).
    """
    seg = segments.copy()
    inl = inlets.copy()
    degree = pd.concat([seg.us_node, seg.ds_node]).value_counts()
    inl["role"] = np.where(inl.representative, "representative", "kept_for_connectivity")
    drop_rows = []
    for cid, grp in inl.groupby(field):
        rep = grp[grp.representative]
        if not len(rep):
            continue
        rep_rows = seg.index[seg.us_node == rep.node_id.iloc[0]]
        extra_q = 0.0
        for idx, row in grp[~grp.representative].iterrows():
            rows = seg.index[seg.us_node == row.node_id]
            if degree.get(row.node_id, 0) == 1 and len(rows) == 1:
                r = rows[0]
                extra_q += float(full_flow_discharge(seg.at[r, "diameter"], seg.at[r, "slope"], n))
                drop_rows.append(r)
                inl.at[idx, "role"] = "removed"
        if extra_q > 0 and len(rep_rows):
            for r in rep_rows:
                q = float(full_flow_discharge(seg.at[r, "diameter"], seg.at[r, "slope"], n))
                seg.at[r, "diameter"] = float(diameter_for_discharge(q + extra_q, seg.at[r, "slope"], n))
    return seg.drop(index=drop_rows), inl
