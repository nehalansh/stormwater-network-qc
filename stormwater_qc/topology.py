"""Network topology: junctions, end nodes, flow direction and connectivity."""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString


def build_nodes(segments, tolerance=0.1):
    """Junctions at segment end points and the end nodes of every segment.

    End points within ``tolerance`` of each other become one junction, placed
    at the first of them. Junctions are numbered 1..n in order of first
    appearance. Returns (segments with ``us_node``/``ds_node`` taken from the
    digitized direction, junctions GeoDataFrame with ``node_id``).

    Equivalent to creating junctions from the unique end points and running
    PCSWMM's Auto-Connect.
    """
    ends = np.array([[*g.coords[0][:2], *g.coords[-1][:2]] for g in segments.geometry])
    n = len(ends)
    pts = ends.reshape(-1, 2)  # start0, end0, start1, end1, ...
    pairs = cKDTree(pts).query_pairs(tolerance, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(2 * n, 2 * n))
    _, lab = connected_components(adj, directed=False)
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(first)
    node_of_label = np.empty(lab.max() + 1, dtype=int)
    node_of_label[order] = np.arange(1, lab.max() + 2)
    node = node_of_label[lab].reshape(n, 2)
    xy = pts[np.sort(first)]
    junctions = gpd.GeoDataFrame({"node_id": np.arange(1, len(xy) + 1)},
                                 geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
                                 crs=segments.crs)
    out = segments.copy()
    out["us_node"], out["ds_node"] = node[:, 0], node[:, 1]
    # place both ends exactly on their junction
    out["geometry"] = [LineString([xy[a - 1], xy[b - 1]]) for a, b in node]
    return out, junctions


def outfall_nodes(junctions, outfalls, tolerance=1.0):
    """IDs of the junctions within ``tolerance`` of an outfall point."""
    tree = cKDTree(np.c_[junctions.geometry.x, junctions.geometry.y])
    hits = tree.query_ball_point(np.c_[outfalls.geometry.x, outfalls.geometry.y], tolerance)
    return sorted({int(junctions.node_id.iloc[i]) for h in hits for i in h})


def orient_to_outfalls(segments, junctions, outfalls, tolerance=1.0):
    """Point every segment toward the outfall it reaches along the network.

    Distances along the network (by pipe length) from every junction to the
    nearest outfall junction are computed; each segment is oriented from its
    farther to its nearer end. Segments of networks without an outfall, and
    segments whose two ends are equally far (loops), keep their digitized
    direction. Returns (oriented segments with a ``reversed`` flag, distance
    to outfall of each junction).
    """
    nn = len(junctions)
    u = segments.us_node.to_numpy() - 1
    v = segments.ds_node.to_numpy() - 1
    w = segments.length.to_numpy()
    graph = coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(nn, nn)).tocsr()
    sources = np.array(outfall_nodes(junctions, outfalls, tolerance)) - 1
    dist = (dijkstra(graph, directed=False, indices=sources, min_only=True)
            if len(sources) else np.full(nn, np.inf))
    flip = (dist[u] < dist[v]) & np.isfinite(dist[u])
    out = segments.copy()
    out["reversed"] = flip
    out.loc[flip, ["us_node", "ds_node"]] = out.loc[flip, ["ds_node", "us_node"]].to_numpy()
    out.loc[flip, "geometry"] = [LineString(list(g.coords)[::-1]) for g in out.geometry[flip]]
    return out, pd.Series(dist, index=junctions.node_id, name="distance_to_outfall")


def connectivity(segments):
    """Upstream and downstream segments of every segment.

    Returns a DataFrame indexed by ``segment_id`` with list columns
    ``upstream`` (segments ending at its upstream node) and ``downstream``
    (segments starting at its downstream node).
    """
    sid = segments.segment_id.to_numpy()
    by_ds = segments.groupby("ds_node").segment_id.apply(list)
    by_us = segments.groupby("us_node").segment_id.apply(list)
    up = [by_ds.get(n, []) for n in segments.us_node]
    down = [by_us.get(n, []) for n in segments.ds_node]
    up = [[x for x in a if x != s] for a, s in zip(up, sid)]
    down = [[x for x in a if x != s] for a, s in zip(down, sid)]
    return pd.DataFrame({"upstream": up, "downstream": down}, index=pd.Index(sid, name="segment_id"))
