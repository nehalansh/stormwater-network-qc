"""Synthetic stormwater network with the defects found in municipal GIS data.

``make_synthetic_network`` builds a small street-grid drainage system and a
DEM, then injects the problems commonly met in municipal infrastructure
layers, recording which features were altered so that each correction can
be checked:

=====================  ====================================================
defect                 how it is injected
=====================  ====================================================
near-miss ends         pipe ends moved 0.1 mm - 8 cm away from the pipe
                       they should join
multi-vertex pipes     trunks stored as polylines spanning several blocks,
                       with laterals meeting them at interior vertices
mid-segment junctions  connectors ending in the middle of a trunk segment
                       (no vertex there)
isolated fragments     1-4-pipe networks along local streets, not connected
                       to anything, each with an inlet
disconnected network   the upper part of one trunk separated from the rest
                       of the system by a 3-m gap
separate system        a second network draining to its own outfall
duplicates             pipes digitized twice (sometimes in reverse)
reversed pipes         polylines digitized against the flow direction
missing diameters      diameter = 0 on single pipes and runs of pipes
depth in diameter      the depth value entered as diameter
displaced inlets       inlets on their connector but about 1-6 m off its end
remote inlets          inlets far from any pipe
noisy DEM              terrain noise, pits and a flat area, so DEM slopes
                       between nodes are often negative or unrealistic
=====================  ====================================================

Coordinates are metres in UTM Zone 12N (EPSG:32612); diameters and depths
are metres.
"""
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

CRS = "EPSG:32612"
INLET_DIAMETER = 0.381  # 15 in
X0, Y0 = 400_000.0, 3_700_000.0


@dataclass
class SyntheticNetwork:
    pipes: gpd.GeoDataFrame        # pipe_id, diameter, depth, geometry
    inlets: gpd.GeoDataFrame       # inlet_id, geometry
    outfalls: gpd.GeoDataFrame     # outfall_id, geometry
    dem: np.ndarray                # fine DEM (1 m)
    dem_transform: object
    dem_coarse: np.ndarray         # coarse DEM (5 m, block mean of the fine DEM)
    dem_coarse_transform: object
    truth: dict = field(default_factory=dict)
    crs: str = CRS

    def write_dem(self, path, coarse=False):
        """Write the fine (or coarse) DEM to a GeoTIFF and return the path."""
        data, transform = ((self.dem_coarse, self.dem_coarse_transform) if coarse
                           else (self.dem, self.dem_transform))
        with rasterio.open(path, "w", driver="GTiff", height=data.shape[0],
                           width=data.shape[1], count=1, dtype="float32", nodata=-9999.0,
                           crs=self.crs, transform=transform) as dst:
            dst.write(data.astype("float32"), 1)
        return path


def _jitter(rng, point, low, high):
    angle = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(low, high)
    return (point[0] + d * np.cos(angle), point[1] + d * np.sin(angle))


def make_synthetic_network(nx=6, ny=6, block=200.0, seed=0, dem_resolution=1.0):
    """Build the synthetic network, inlets, outfalls and DEMs.

    nx, ny : number of north-south trunks and of east-west streets
    block  : street spacing (m)
    """
    rng = np.random.default_rng(seed)
    xs = X0 + 300 + block * np.arange(nx)
    ys = Y0 + 300 + block * np.arange(ny)
    pipes, inlets = [], []   # dicts: geometry + attributes
    truth = {k: set() for k in ("isolated", "duplicate", "reversed", "zero_diameter",
                                "depth_as_diameter", "displaced_inlet", "remote_inlet",
                                "near_miss", "mid_segment", "disconnected")}
    truth["true_diameter"] = {}

    def add_pipe(coords, diameter, kind, **extra):
        pid = len(pipes) + 1
        pipes.append(dict(pipe_id=pid, geometry=LineString(coords), diameter=diameter,
                          depth=round(rng.uniform(2.5, 6.0), 2), kind=kind, **extra))
        truth["true_diameter"][pid] = diameter
        return pid

    def add_inlet(xy, pipe_id=None):
        iid = len(inlets) + 1
        inlets.append(dict(inlet_id=iid, geometry=Point(xy), true_xy=xy, pipe_id=pipe_id))
        return iid

    # --- main system: trunks (north -> south) into an interceptor (east -> west)
    k = ny // 2
    trunk_upper, trunk_lower = {}, {}
    for i, x in enumerate(xs):
        upper = [(x, ys[j]) for j in range(ny - 1, k - 1, -1)]
        lower = [(x, ys[j]) for j in range(k, -1, -1)]
        trunk_upper[i] = add_pipe(upper, 0.610, "trunk")
        trunk_lower[i] = add_pipe(lower, 0.762, "trunk")
    interceptor = add_pipe([(x, ys[0]) for x in xs[::-1]], 1.524, "interceptor")
    outfall_main = (xs[0] - 150.0, ys[0])
    add_pipe([(xs[0], ys[0]), outfall_main], 1.829, "outfall")

    # --- separate system east of the grid with its own outfall
    xe = xs[-1] + 260.0
    sep = add_pipe([(xe, ys[-1]), (xe, ys[-2]), (xe, ys[-3])], 0.610, "trunk")
    outfall_sep = (xe + 160.0, ys[-3])
    add_pipe([(xe, ys[-3]), outfall_sep], 0.914, "outfall")

    # --- inlets and 15-in connectors at street intersections
    nodes = [(x, y) for x in xs for y in ys] + [(xe, ys[-1]), (xe, ys[-2]), (xe, ys[-3])]
    for node in nodes:
        corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for c in rng.choice(4, size=rng.integers(1, 4), replace=False):
            sx, sy = corners[c]
            xy = (node[0] + sx * rng.uniform(6, 12), node[1] + sy * rng.uniform(5, 10))
            pid = add_pipe([xy, node], INLET_DIAMETER, "connector")
            add_inlet(xy, pid)

    # connectors meeting a trunk in the middle of a segment (no vertex there)
    for i in rng.choice(nx, size=3, replace=False):
        y_mid = ys[rng.integers(1, k)] - block / 2
        node = (xs[i], y_mid)
        xy = (xs[i] + 9.0, y_mid + 4.0)
        pid = add_pipe([xy, node], INLET_DIAMETER, "connector")
        add_inlet(xy, pid)
        truth["mid_segment"].add(pid)

    truth_lines = {p["pipe_id"]: list(p["geometry"].coords) for p in pipes}

    # --- defect: isolated fragments along local streets, each with an inlet
    for _ in range(6):
        # inside a block, 55-75 m from the nearest trunk and street
        cx = xs[rng.integers(0, nx - 1)] + rng.uniform(55, 75)
        cy = ys[rng.integers(1, ny)] - rng.uniform(55, 75)
        pts = [(cx, cy)]
        for _ in range(rng.integers(1, 5)):
            pts.append((pts[-1][0] + rng.uniform(-8, 8), pts[-1][1] - rng.uniform(12, 20)))
        first = None
        for a, b in zip(pts[:-1], pts[1:]):
            pid = add_pipe([a, b], INLET_DIAMETER, "isolated")
            truth["isolated"].add(pid)
            first = first or pid
        add_inlet(pts[0], first)

    # --- defect: upper part of trunk 1 separated by a 3-m gap
    up = pipes[trunk_upper[1] - 1]
    c = list(up["geometry"].coords)
    c[-1] = (c[-1][0] + 3.0, c[-1][1] + 0.5)
    up["geometry"] = LineString(c)
    truth["disconnected"].add(trunk_upper[1])

    # --- defect: near-miss ends (connectors and trunk joints)
    for p in pipes:
        if p["kind"] == "connector" and rng.random() < 0.4:
            c = list(p["geometry"].coords)
            c[-1] = _jitter(rng, c[-1], 0.005, 0.08)
            p["geometry"] = LineString(c)
            truth["near_miss"].add(p["pipe_id"])
    for i in range(nx):
        if i == 1:
            continue
        p = pipes[trunk_lower[i] - 1]
        c = list(p["geometry"].coords)
        c[0] = _jitter(rng, c[0], 1e-4, 5e-4)
        p["geometry"] = LineString(c)
        truth["near_miss"].add(p["pipe_id"])

    # --- defect: missing diameters and depth entered as diameter
    connectors = [p for p in pipes if p["kind"] == "connector"
                  and p["pipe_id"] not in truth["mid_segment"]]
    for p in rng.choice(connectors, size=4, replace=False):
        p["diameter"] = 0.0
        truth["zero_diameter"].add(p["pipe_id"])
    for pid in (trunk_upper[3], trunk_lower[4]):
        pipes[pid - 1]["diameter"] = 0.0
        truth["zero_diameter"].add(pid)
    for p in rng.choice([q for q in connectors if q["diameter"] > 0], size=2, replace=False):
        p["diameter"] = p["depth"]
        truth["depth_as_diameter"].add(p["pipe_id"])

    # --- defect: polylines digitized against the flow direction
    for p in pipes:
        if p["kind"] in ("trunk", "connector") and rng.random() < 0.15:
            p["geometry"] = LineString(list(p["geometry"].coords)[::-1])
            truth["reversed"].add(p["pipe_id"])

    # --- defect: duplicated pipes
    originals = rng.choice([p for p in pipes if p["kind"] == "connector"], size=3, replace=False)
    originals = list(originals) + [pipes[trunk_lower[2] - 1]]
    for p in originals:
        coords = list(p["geometry"].coords)
        if rng.random() < 0.5:
            coords = coords[::-1]
        pid = add_pipe(coords, p["diameter"], p["kind"])
        truth["duplicate"].add(pid)

    # --- defect: displaced and remote inlets
    for inl in inlets:
        if inl["pipe_id"] in truth["isolated"]:
            continue
        if inl["pipe_id"] is not None and rng.random() < 0.25:
            # slid along its connector (up to 40% of its length) and up to
            # 0.5 m sideways: still on the pipe, but off its end vertex
            line = truth_lines[inl["pipe_id"]]
            a, b = np.array(inl["true_xy"]), np.array(line[1] if np.allclose(line[0], inl["true_xy"]) else line[0])
            u = (b - a) / np.hypot(*(b - a))
            xy = a + u * rng.uniform(0.1, 0.4) * np.hypot(*(b - a)) \
                + np.array([-u[1], u[0]]) * rng.uniform(-0.5, 0.5)
            inl["geometry"] = Point(xy)
            truth["displaced_inlet"].add(inl["inlet_id"])
    all_lines = gpd.GeoSeries([p["geometry"] for p in pipes])
    while len(truth["remote_inlet"]) < 5:
        xy = (rng.uniform(xs[0], xs[-1]), rng.uniform(ys[0], ys[-1]))
        if all_lines.distance(Point(xy)).min() > 40.0:
            truth["remote_inlet"].add(add_inlet(xy))
    truth["isolated_inlets"] = {i["inlet_id"] for i in inlets if i["pipe_id"] in truth["isolated"]}

    pipes_gdf = gpd.GeoDataFrame(
        [{k2: v for k2, v in p.items() if k2 != "kind"} for p in pipes], crs=CRS)
    inlets_gdf = gpd.GeoDataFrame(
        [{"inlet_id": i["inlet_id"], "geometry": i["geometry"]} for i in inlets], crs=CRS)
    outfalls = gpd.GeoDataFrame({"outfall_id": [1, 2]},
                                geometry=[Point(outfall_main), Point(outfall_sep)], crs=CRS)
    truth["inlet_true_xy"] = {i["inlet_id"]: i["true_xy"] for i in inlets}
    truth["n_systems"] = 2

    dem, transform = _make_dem(rng, xs, ys, xe, dem_resolution)
    coarse, ctransform = _block_mean(dem, transform, 5)
    return SyntheticNetwork(pipes_gdf, inlets_gdf, outfalls, dem, transform,
                            coarse, ctransform, truth)


def _make_dem(rng, xs, ys, xe, res):
    """Terrain sloping south-west, with smoothed noise, pits and a flat area."""
    x_min, x_max = X0, xe + 400.0
    y_min, y_max = Y0, ys[-1] + 300.0
    ncol = int(np.ceil((x_max - x_min) / res))
    nrow = int(np.ceil((y_max - y_min) / res))
    transform = from_origin(x_min, y_max, res, res)
    cols = x_min + (np.arange(ncol) + 0.5) * res
    rows = y_max - (np.arange(nrow) + 0.5) * res
    X, Y = np.meshgrid(cols, rows)
    z = 340.0 + 0.004 * (Y - Y0) + 0.0015 * (X - X0)
    # smooth noise: sum of random sinusoids (amplitude ~0.3 m, wavelength 20-80 m)
    for _ in range(12):
        kx, ky = rng.uniform(2 * np.pi / 80, 2 * np.pi / 20, size=2) * rng.choice([-1, 1], 2)
        z += rng.uniform(0.05, 0.12) * np.sin(kx * X + ky * Y + rng.uniform(0, 2 * np.pi))
    z += rng.normal(0, 0.05, size=z.shape)
    # pits
    for _ in range(4):
        px, py = rng.uniform(x_min + 200, x_max - 200), rng.uniform(y_min + 200, y_max - 200)
        z -= 1.0 * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * 12.0 ** 2))
    # flat area (e.g. a parking lot graded level)
    fx, fy = xs[len(xs) // 2] + 60, ys[len(ys) // 2] + 60
    flat = (abs(X - fx) < 40) & (abs(Y - fy) < 40)
    z[flat] = z[flat].min()
    return z, transform


def _block_mean(dem, transform, factor):
    nr, nc = (dem.shape[0] // factor) * factor, (dem.shape[1] // factor) * factor
    coarse = dem[:nr, :nc].reshape(nr // factor, factor, nc // factor, factor).mean(axis=(1, 3))
    t = transform
    return coarse, from_origin(t.c, t.f, t.a * factor, -t.e * factor)


def summary(net):
    """Number of features affected by each injected defect."""
    t = net.truth
    rows = {k: len(t[k]) for k in ("near_miss", "mid_segment", "isolated", "disconnected",
                                   "duplicate", "reversed", "zero_diameter",
                                   "depth_as_diameter", "displaced_inlet", "remote_inlet")}
    return pd.Series(rows, name="injected")


def evaluate(net, result, vmin=1.5, vmax=4.6):
    """Compare a ``workflow.run_workflow`` result with the injected defects.

    Returns a DataFrame with, for each defect, the number injected, the
    number corrected and whether the correction matches the ground truth.
    """
    t, log = net.truth, result.log
    seg = result.steps["segments_before_merge"]
    rows = []

    def add(defect, injected, fixed, ok, note=""):
        rows.append(dict(defect=defect, injected=injected, corrected=fixed, matches_truth=ok, note=note))

    snapped = log["clean_geometry"]["snapped_to_vertex"] + log["clean_geometry"]["snapped_to_segment"]
    from .networks import identify_networks
    n_net = identify_networks(seg).nunique()
    add("near-miss ends", len(t["near_miss"]) + len(t["mid_segment"]), snapped,
        n_net == t["n_systems"], f"{n_net} connected systems left (truth: {t['n_systems']})")
    removed = set(log["removed_isolated_pipes"])
    add("isolated fragments", len(t["isolated"]), len(removed), removed == t["isolated"])
    add("disconnected network", len(t["disconnected"]), len(log["reconnected_segments"]),
        len(log["reconnected_segments"]) == len(t["disconnected"]))
    dups = set(log["removed_duplicate_pipes"])
    add("duplicate pipes", len(t["duplicate"]), len(dups), dups == t["duplicate"])
    rev = set(seg.loc[seg.reversed, "parent_id"])
    add("reversed pipes", len(t["reversed"]), len(rev), rev == t["reversed"])
    flagged = set(log["diameter_outlier_pipes"])
    add("depth entered as diameter", len(t["depth_as_diameter"]), len(flagged),
        flagged == t["depth_as_diameter"])
    bad = t["zero_diameter"] | t["depth_as_diameter"]
    filled = seg[seg.parent_id.isin(bad)].drop_duplicates("parent_id")
    exact = (np.isclose(filled.diameter, filled.parent_id.map(t["true_diameter"]))).sum()
    add("missing diameters", len(bad), int((filled.diameter > 0).sum()), bool((filled.diameter > 0).all()),
        f"{exact} of {len(filled)} equal to the true diameter")
    inl = result.inlets
    jx = result.steps["junctions_all"]
    tree_xy = np.c_[jx.geometry.x, jx.geometry.y]
    from scipy.spatial import cKDTree
    true_node = {i: int(jx.node_id.iloc[cKDTree(tree_xy).query(t["inlet_true_xy"][i])[1]])
                 for i in t["displaced_inlet"]}
    got = inl.set_index("inlet_id").node_id
    n_ok = sum(got.get(i) == n_ for i, n_ in true_node.items())
    add("displaced inlets", len(t["displaced_inlet"]), n_ok, n_ok == len(true_node),
        "snapped to the junction of their own connector")
    dropped = set(log["inlets_dropped"])
    expect = t["remote_inlet"] | t["isolated_inlets"]
    add("remote / orphaned inlets", len(expect), len(dropped), dropped == expect)
    v = seg.velocity
    inside = (v >= vmin - 1e-9) & (v <= vmax + 1e-9)
    needed = seg.slope_source != "parent_fine_dem"   # DEM slope of the whole pipe was not usable
    add("DEM slopes outside limits", int(needed.sum()), int((needed & inside).sum()), bool(inside.all()),
        f"final velocities {v.min():.2f}-{v.max():.2f} m/s")
    return pd.DataFrame(rows)
