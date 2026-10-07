"""Geometric cleaning of pipe layers.

Municipal pipe layers often store a pipe as a polyline spanning several
junctions, and pipes that should share an end point miss each other by
millimetres to centimetres. These functions turn such a layer into
2-vertex segments whose shared end points coincide exactly.
"""
import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point


def _vertices(pipes):
    """All vertices: coordinates, owning row, position and end flag."""
    xy, owner, pos, is_end = [], [], [], []
    for row, g in enumerate(pipes.geometry):
        c = np.asarray(g.coords)[:, :2]
        xy.append(c)
        owner.append(np.full(len(c), row))
        pos.append(np.arange(len(c)))
        e = np.zeros(len(c), bool)
        e[[0, -1]] = True
        is_end.append(e)
    return (np.vstack(xy), np.concatenate(owner), np.concatenate(pos),
            np.concatenate(is_end))


def snap_endpoints(pipes, tolerance=0.1):
    """Make pipe ends that nearly meet coincide exactly.

    1. Vertices closer than ``tolerance`` to each other, at least one of them a
       pipe end, are grouped. Every end in a group is moved to an interior
       vertex of the group if there is one (a lateral meeting a trunk at a
       vertex), otherwise to the first end of the group.
    2. An end still not touching another pipe but within ``tolerance`` of the
       interior of a segment of another pipe is moved onto that segment, and
       a vertex is inserted there (a lateral meeting a trunk mid-segment).

    Returns a copy of ``pipes`` and the number of ends moved in each stage.
    """
    pipes = pipes.reset_index(drop=True)
    coords = [np.asarray(g.coords)[:, :2].copy() for g in pipes.geometry]
    xy, owner, pos, is_end = _vertices(pipes)
    pairs = cKDTree(xy).query_pairs(tolerance, output_type="ndarray")
    pairs = pairs[(owner[pairs[:, 0]] != owner[pairs[:, 1]])
                  & (is_end[pairs[:, 0]] | is_end[pairs[:, 1]])]
    n = len(xy)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, group = connected_components(adj, directed=False)
    moved_1 = 0
    for g in np.unique(group[pairs.ravel()]) if len(pairs) else []:
        members = np.flatnonzero(group == g)
        interior = members[~is_end[members]]
        target = xy[interior[0]] if len(interior) else xy[members[is_end[members]][0]]
        for m in members[is_end[members]]:
            if not np.array_equal(coords[owner[m]][pos[m]], target):
                coords[owner[m]][pos[m]] = target
                moved_1 += 1

    # stage 2: ends near the interior of another pipe's segment
    lines = [LineString(c) for c in coords]
    sindex = gpd.GeoSeries(lines).sindex
    vertex_set = {tuple(p) for c in coords for p in c}
    moved_2 = 0
    for row, c in enumerate(coords):
        for end in (0, len(c) - 1):
            p = Point(c[end])
            own = {tuple(v) for v in c}
            # other pipes, ignoring exact duplicates of this one
            cand = [j for j in sindex.query(p.buffer(tolerance))
                    if j != row and {tuple(v) for v in coords[j]} != own]
            if not cand or any(tuple(c[end]) in map(tuple, coords[j]) for j in cand):
                continue
            j = min(cand, key=lambda k: lines[k].distance(p))
            if lines[j].distance(p) > tolerance:
                continue
            q = lines[j].interpolate(lines[j].project(p))
            q = (q.x, q.y)
            # insert q as a vertex of every pipe passing through it (overlapping
            # duplicates must stay identical), in the segment that contains it
            for k in cand:
                if lines[k].distance(Point(q)) > 1e-6 or q in map(tuple, coords[k]):
                    continue
                ck = coords[k]
                seg = int(np.argmin([LineString(ck[s:s + 2]).distance(Point(q))
                                     for s in range(len(ck) - 1)]))
                coords[k] = np.insert(ck, seg + 1, q, axis=0)
                lines[k] = LineString(coords[k])
            c[end] = q
            vertex_set.add(q)
            moved_2 += 1
    out = pipes.copy()
    out["geometry"] = [LineString(c) for c in coords]
    return out, {"snapped_to_vertex": moved_1, "snapped_to_segment": moved_2}


def extend_lines(pipes, search_distance=0.1, tol=1e-6):
    """Extend dangling ends along their last segment to the nearest pipe.

    Equivalent of the ArcGIS "Extend Line" tool: an end that touches no other
    pipe is extended, in the direction of its last segment, to the first pipe
    it meets within ``search_distance``; the intersection is added as a new
    vertex.
    """
    geoms = list(pipes.geometry)
    sindex = pipes.sindex
    out = []
    for i, line in enumerate(geoms):
        coords = [c[:2] for c in line.coords]
        for end in (0, -1):
            p = np.array(coords[end])
            q = np.array(coords[1] if end == 0 else coords[-2])
            others = [j for j in sindex.query(Point(p).buffer(search_distance)) if j != i]
            if not others or min(geoms[j].distance(Point(p)) for j in others) <= tol:
                continue
            direction = p - q
            norm = np.hypot(*direction)
            if norm == 0:
                continue
            ray = LineString([p, p + direction / norm * search_distance])
            best = None
            for j in others:
                hit = ray.intersection(geoms[j])
                if hit.is_empty:
                    continue
                for h in getattr(hit, "geoms", [hit]):
                    h = h if h.geom_type == "Point" else Point(h.coords[0])
                    d = Point(p).distance(h)
                    if d > tol and (best is None or d < best[0]):
                        best = (d, (h.x, h.y))
            if best is not None:
                coords.insert(0, best[1]) if end == 0 else coords.append(best[1])
        out.append(LineString(coords))
    result = pipes.copy()
    result["geometry"] = out
    return result


def split_at_vertices(pipes, id_field="pipe_id"):
    """Split every polyline into 2-vertex segments.

    Segments keep the attributes of their parent; ``parent_id`` holds the
    parent's ``id_field`` and ``segment_id`` numbers the segments 1..n.
    Zero-length segments are dropped.
    """
    rows, segments = [], []
    for idx, line in zip(pipes.index, pipes.geometry):
        c = [p[:2] for p in line.coords]
        for a, b in zip(c[:-1], c[1:]):
            if a != b:
                rows.append(idx)
                segments.append(LineString([a, b]))
    out = pipes.loc[rows].drop(columns="geometry").reset_index(drop=True)
    out = gpd.GeoDataFrame(out, geometry=segments, crs=pipes.crs)
    out.insert(0, "parent_id", out[id_field] if id_field in out else rows)
    out.insert(0, "segment_id", np.arange(1, len(out) + 1))
    return out


def clean_geometry(pipes, tolerance=0.1, id_field="pipe_id"):
    """Snap near-miss ends, then split into 2-vertex segments."""
    pipes = pipes.explode(index_parts=False).reset_index(drop=True)
    snapped, report = snap_endpoints(pipes, tolerance)
    segments = split_at_vertices(snapped, id_field)
    report["segments"] = len(segments)
    return segments, report
