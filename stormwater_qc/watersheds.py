"""Catch-basin delineation: one basin per inlet cluster.

Equivalent of the ArcGIS chain Fill -> Flow Direction (D8) -> Flow
Accumulation -> Snap Pour Point -> Watershed -> Raster to Polygon ->
Dissolve. Every DEM cell is assigned the ID of the first snapped pour point
found downstream along its D8 flow path. Requires pysheds.
"""
import heapq

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

# pysheds D8 codes (ESRI convention) -> (row, col) offsets
D8 = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
      4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def downstream_index(fdir):
    """Flat index of the D8 receiver of each cell (the cell itself if none)."""
    nr, nc = fdir.shape
    idx = np.arange(nr * nc)
    nxt = idx.copy()
    flat = fdir.ravel()
    r, c = np.divmod(idx, nc)
    for code, (dr, dc) in D8.items():
        m = flat == code
        rr, cc = r[m] + dr, c[m] + dc
        ok = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
        nxt[idx[m][ok]] = rr[ok] * nc + cc[ok]
    return nxt


def route_unresolved(fdir, elev, keep=None):
    """Route the cells whose D8 flow path never leaves the grid.

    resolve_flats does not resolve every flat of a large urban DEM, and the
    cells around such a flat drain into it. All cells whose flow path ends
    in an undefined interior cell are re-routed by a priority flood (Barnes
    et al., 2014) seeded from the cells that do drain: cells are visited in
    order of increasing (filled) elevation and each is pointed to the
    neighbour from which it was reached. This cannot create flow loops.
    Cells in ``keep`` (e.g. cells already draining to a pour point) keep
    their direction.
    """
    nr, nc = fdir.shape
    fdir = fdir.copy()
    interior = np.zeros(fdir.shape, bool)
    interior[1:-1, 1:-1] = True
    undefined = ((fdir <= 0) & interior).ravel()
    term = downstream_index(fdir)
    for _ in range(int(np.ceil(np.log2(fdir.size))) + 1):
        term = term[term]
    stuck = undefined[term]  # path ends in an undefined cell
    if keep is not None:
        stuck &= ~keep.ravel()
    if not stuck.any():
        return fdir
    z = elev.ravel()
    flat = fdir.ravel()
    offsets = [(code, dr, dc) for code, (dr, dc) in D8.items()]
    # seeds: draining cells adjacent to a stuck cell
    stuck2 = stuck.reshape(nr, nc)
    near = np.zeros_like(stuck2)
    for _, dr, dc in offsets:
        near |= np.roll(np.roll(stuck2, dr, 0), dc, 1)
    seeds = np.flatnonzero(near.ravel() & ~stuck)
    heap = [(float(z[i]), int(i)) for i in seeds]
    heapq.heapify(heap)
    visited = ~stuck
    while heap:
        e, i = heapq.heappop(heap)
        r, c = divmod(i, nc)
        for code, dr, dc in offsets:
            rr, cc = r - dr, c - dc  # neighbour whose direction 'code' points to i
            if 0 <= rr < nr and 0 <= cc < nc:
                j = rr * nc + cc
                if not visited[j]:
                    visited[j] = True
                    flat[j] = code
                    heapq.heappush(heap, (max(float(z[j]), e), j))
    return flat.reshape(nr, nc)


def flow_routing(dem_path):
    """Filled DEM, D8 flow direction and flow accumulation."""
    from pysheds.grid import Grid  # optional dependency

    dem_path = str(dem_path)
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    filled = grid.fill_depressions(grid.fill_pits(dem))
    inflated = grid.resolve_flats(filled)
    fdir = grid.flowdir(inflated, routing="d8")
    acc = grid.accumulation(fdir, routing="d8")
    return np.asarray(fdir), np.asarray(acc), np.asarray(filled)


def snap_pour_points(points, acc, transform, snap_distance):
    """Move each point to the highest-accumulation cell within snap_distance."""
    cell = abs(transform.a)
    r = int(np.ceil(snap_distance / cell))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    footprint = (xx ** 2 + yy ** 2) * cell ** 2 <= snap_distance ** 2
    rows, cols = rasterio.transform.rowcol(transform, points.geometry.x, points.geometry.y)
    rows, cols = np.asarray(rows), np.asarray(cols)
    out_r, out_c = rows.copy(), cols.copy()
    for k, (i, j) in enumerate(zip(rows, cols)):
        i0, j0 = max(i - r, 0), max(j - r, 0)
        win = acc[i0:i + r + 1, j0:j + r + 1]
        fp = footprint[i0 - (i - r):i0 - (i - r) + win.shape[0],
                       j0 - (j - r):j0 - (j - r) + win.shape[1]]
        masked = np.where(fp, win, -np.inf)
        a, b = np.unravel_index(np.argmax(masked), masked.shape)
        out_r[k], out_c[k] = i0 + a, j0 + b
    return out_r, out_c


def label_watersheds(fdir, pour_rows, pour_cols, labels):
    """Assign every cell the label of the first pour point downstream of it."""
    nr, nc = fdir.shape
    n = nr * nc
    nxt = downstream_index(fdir)
    nxt[nxt == np.arange(n)] = -1  # cells with no receiver end their path
    lab = np.zeros(n, dtype=np.int64)
    lab[pour_rows * nc + pour_cols] = labels
    nxt[lab > 0] = -1  # pour points terminate the flow paths above them
    # pointer jumping: resolve the first labelled cell downstream of each cell
    for _ in range(int(np.ceil(np.log2(n))) + 2):
        todo = (lab == 0) & (nxt >= 0)
        if not todo.any():
            break
        t = nxt[todo]
        lab[todo] = lab[t]
        nxt_new = nxt.copy()
        still = todo.copy()
        still[todo] = lab[t] == 0
        nxt_new[still] = nxt[nxt[still]]
        nxt = nxt_new
    return lab.reshape(nr, nc)


def delineate(dem_path, points, snap_distance=30.0, field="cluster_id"):
    """Catch basins (``gridcode`` = ``field`` of the pour point) from a DEM GeoTIFF."""
    fdir, acc, filled = flow_routing(dem_path)
    with rasterio.open(dem_path) as src:
        transform, crs = src.transform, src.crs
    pr, pc = snap_pour_points(points, acc, transform, snap_distance)
    ids = points[field].to_numpy().astype(np.int64)
    lab = label_watersheds(fdir, pr, pc, ids)
    # cells whose path ends in an unresolved flat before reaching a pour
    # point are re-routed, and labelled again
    fdir = route_unresolved(fdir, filled, keep=lab > 0)
    lab = label_watersheds(fdir, pr, pc, ids)
    polys = [(shape(g), int(v)) for g, v in
             shapes(lab.astype(np.int32), mask=lab > 0, transform=transform)]
    gdf = gpd.GeoDataFrame({"gridcode": [v for _, v in polys]},
                           geometry=[g for g, _ in polys], crs=crs)
    out = gdf.dissolve(by="gridcode").reset_index()
    out["Shape_Area"] = out.geometry.area
    return out
