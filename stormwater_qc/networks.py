"""Independent drainage networks: identification, classification, cleaning.

A network is a set of pipes connected through shared end points. Real
systems drain to outfalls; small networks that do not are usually stubs
along local streets left in the GIS layer, and larger ones may be parts of
the system separated from it by a digitizing gap.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString


def _ends(segments):
    return np.array([[*g.coords[0][:2], *g.coords[-1][:2]] for g in segments.geometry])


def identify_networks(segments, tolerance=0.1):
    """Network ID of each 2-vertex segment.

    Two segments are connected when any of their end points are within
    ``tolerance``. IDs start at 1 and are numbered in order of the first
    segment of each network (the order of MATLAB's ``conncomp``).
    """
    ends = _ends(segments)
    n = len(ends)
    pts = np.vstack([ends[:, :2], ends[:, 2:]])
    owner = np.r_[np.arange(n), np.arange(n)]
    pairs = cKDTree(pts).query_pairs(tolerance, output_type="ndarray")
    i, j = owner[pairs[:, 0]], owner[pairs[:, 1]]
    keep = i != j
    adj = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    _, first = np.unique(lab, return_index=True)
    remap = np.empty(lab.max() + 1, dtype=int)
    remap[np.argsort(first)] = np.arange(1, lab.max() + 2)
    return pd.Series(remap[lab], index=segments.index, name="network_id")


def classify_networks(segments, outfalls, network_id=None, outfall_tolerance=1.0,
                      small=4, large=60):
    """Summary table of the networks and a suggested action for each.

    category:
      ``outfall`` - an end point lies within ``outfall_tolerance`` of an outfall
      ``large``   - more than ``large`` segments (keep)
      ``inspect`` - ``small`` < segments <= ``large`` and no outfall: check in GIS,
                    it may be part of the system cut off by a gap
      ``small``   - at most ``small`` segments and no outfall (remove)
    """
    net = identify_networks(segments) if network_id is None else network_id
    ends = _ends(segments)
    tree = cKDTree(np.c_[outfalls.geometry.x, outfalls.geometry.y])
    d = np.minimum(tree.query(ends[:, :2])[0], tree.query(ends[:, 2:])[0])
    table = pd.DataFrame({"network_id": net.values, "near_outfall": d <= outfall_tolerance,
                          "length": segments.length.values})
    table = table.groupby("network_id").agg(n_segments=("length", "size"),
                                            length=("length", "sum"),
                                            has_outfall=("near_outfall", "any"))
    table["category"] = np.select(
        [table.has_outfall, table.n_segments > large, table.n_segments > small],
        ["outfall", "large", "inspect"], "small")
    return table


def remove_isolated_networks(segments, outfalls, keep_ids=(), drop_ids=(), **kwargs):
    """Keep networks draining to an outfall, large networks and ``keep_ids``.

    Networks in ``drop_ids`` are removed regardless of their category.
    Returns (kept segments, removed segments, classification table).
    """
    net = identify_networks(segments)
    table = classify_networks(segments, outfalls, net, **kwargs)
    keep = set(table.index[table.category.isin(["outfall", "large"])]) | set(keep_ids)
    keep -= set(drop_ids)
    mask = net.isin(keep).values
    return segments[mask].copy(), segments[~mask].copy(), table


def reconnect_networks(segments, outfalls, max_gap=5.0, tolerance=0.1, **kwargs):
    """Close digitizing gaps between networks that do not reach an outfall.

    For each network without an outfall, the dangling end (an end shared by no
    other segment) closest to an end point of a network that reaches an
    outfall is found; if the gap is at most ``max_gap``, the dangling end is
    moved onto that point.
    Repeats until no gap can be closed. Returns the edited segments and a list
    of the moves as (segment_id, gap in m).
    """
    segments = segments.copy()
    moves = []
    while True:
        net = identify_networks(segments, tolerance)
        table = classify_networks(segments, outfalls, net, **kwargs)
        ends = _ends(segments)
        pts = np.vstack([ends[:, :2], ends[:, 2:]])
        owner = np.r_[np.arange(len(ends)), np.arange(len(ends))]
        which = np.r_[np.zeros(len(ends), int), np.ones(len(ends), int)]
        counts = cKDTree(pts).query_ball_point(pts, tolerance, return_length=True)
        best = None
        draining = np.isin(net.values[owner], table.index[table.has_outfall])
        others = np.flatnonzero(draining)
        for nid in table.index[~table.has_outfall]:
            inside = net.values[owner] == nid
            dangling = np.flatnonzero(inside & (counts == 1))
            if not len(dangling) or not len(others):
                continue
            d, k = cKDTree(pts[others]).query(pts[dangling])
            m = int(np.argmin(d))
            if d[m] <= max_gap and (best is None or d[m] < best[0]):
                best = (d[m], dangling[m], others[k[m]])
        if best is None:
            return segments, moves
        gap, a, b = best
        row = segments.index[owner[a]]
        c = list(segments.at[row, "geometry"].coords)
        c[0 if which[a] == 0 else -1] = tuple(pts[b])
        segments.at[row, "geometry"] = LineString(c)
        sid = segments.at[row, "segment_id"] if "segment_id" in segments else row
        moves.append((sid, round(float(gap), 3)))


def remove_duplicates(segments, decimals=6):
    """Drop segments sharing both end points (in either order) with an earlier one.

    Returns (unique segments, removed duplicates).
    """
    ends = np.round(_ends(segments), decimals)
    key = [tuple(sorted([tuple(e[:2]), tuple(e[2:])])) for e in ends]
    dup = pd.Series(key, index=segments.index).duplicated(keep="first").values
    return segments[~dup].copy(), segments[dup].copy()
