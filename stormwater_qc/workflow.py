"""The complete workflow, from raw GIS layers to SWMM-ready layers.

``run_workflow`` checks the inputs, applies every step in order and returns
the final layers, all intermediate products and a log of what each step
changed. Decisions that need judgement (which mid-size networks to keep,
inlets moved far, diameters no rule can fill) are reported in the log rather
than taken silently.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import rasterio

from . import diameters, geometry, inlets as inl, networks, slopes, swmm, topology


@dataclass
class WorkflowResult:
    segments: object = None      # final conduits
    junctions: object = None     # final junctions with depth and inverts
    inlets: object = None        # inlets with node, cluster and role
    basins: object = None        # catch basins (gridcode = cluster_id)
    steps: dict = field(default_factory=dict)   # intermediate products
    log: dict = field(default_factory=dict)


def _read_dem(path):
    with rasterio.open(path) as src:
        dem = src.read(1).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        dem[np.abs(dem) > 1e30] = np.nan
        return dem, src.transform, src.crs


def _check_inputs(pipes, inlets, outfalls, dem_crs):
    if pipes.crs is None or not pipes.crs.is_projected:
        raise ValueError("pipes must be in a projected CRS in metres (e.g. a UTM zone); "
                         f"got {pipes.crs}")
    for name, layer in (("inlets", inlets), ("outfalls", outfalls)):
        if layer.crs != pipes.crs:
            raise ValueError(f"{name} CRS ({layer.crs}) differs from the pipes CRS ({pipes.crs})")
    if dem_crs is not None and rasterio.crs.CRS.from_user_input(pipes.crs) != dem_crs:
        raise ValueError(f"DEM CRS ({dem_crs}) differs from the pipes CRS ({pipes.crs})")
    if "diameter" not in pipes.columns:
        raise KeyError("pipes need a 'diameter' column in metres (0 or NaN = missing)")
    d = pipes["diameter"]
    if (d > 0).any() and d[d > 0].median() > 3:
        warnings.warn("median pipe diameter is above 3: diameters must be in metres "
                      "(convert from inches or millimetres first)", stacklevel=3)


def _ids(table, col):
    """Values of ``col`` if the table has it, else its index labels."""
    return sorted(table[col].tolist()) if col in table.columns else sorted(table.index.tolist())


def run_workflow(pipes, inlets, outfalls, dem_path, dem_coarse_path=None,
                 tolerance=0.1, max_gap=5.0, outfall_tolerance=1.0, inlet_distance=1.0,
                 cluster_eps=25.0, connector_diameter=0.381, n=0.013, vmin=1.5, vmax=4.6,
                 cover=0.9144, snap_distance=30.0, keep_networks=(), drop_networks=(),
                 depth_field=None, delineate=True):
    """Run every step, from raw pipe and inlet layers to SWMM-ready layers.

    Parameters
    ----------
    pipes : GeoDataFrame
        Pipe lines (LineString or MultiLineString) in a projected CRS in
        metres, with a ``diameter`` column in metres (0 or NaN = missing). An
        identifier column ``pipe_id`` is optional and is kept as ``parent_id``.
    inlets, outfalls : GeoDataFrame
        Points, same CRS. An ``inlet_id`` column is optional (used in the log).
        Outfalls must lie within ``outfall_tolerance`` of a pipe end.
    dem_path, dem_coarse_path : str
        GeoTIFF ground elevation (m), same CRS. The fine DEM gives slopes and
        rim elevations; the optional coarse DEM is a second slope estimate and
        is used for catch-basin delineation.
    tolerance : float
        Distance (m) under which pipe ends are snapped together.
    max_gap : float
        Largest digitizing gap (m) closed between a network without an outfall
        and one with an outfall.
    outfall_tolerance : float
        Distance (m) within which an outfall point matches a pipe end.
    inlet_distance : float
        Inlets farther than this (m) from every pipe are dropped.
    cluster_eps : float
        Neighbourhood radius (m) for clustering inlets (DBSCAN).
    connector_diameter : float
        Diameter (m) given to missing-diameter pipes that start at an inlet.
    n, vmin, vmax : float
        Manning's roughness and full-flow velocity limits (m/s) for slopes.
    cover : float
        Soil cover (m) above the largest pipe at each junction.
    snap_distance : float
        Distance (m) within which cluster points are moved to the highest
        flow accumulation before delineating catch basins.
    keep_networks, drop_networks : sequence of int
        ``network_id`` values (from ``result.log["networks"]``) to keep or
        remove regardless of their category. Typical use: run once, inspect
        the networks listed in ``result.log["networks_to_inspect"]`` in GIS,
        and run again with ``keep_networks=[...]``.
    depth_field : str, optional
        Name of a pipe depth column; diameters equal to the depth are treated
        as missing (a depth typed in the diameter field).
    delineate : bool
        Delineate catch basins (requires pysheds).

    Returns
    -------
    WorkflowResult
        ``segments`` (conduits), ``junctions``, ``inlets``, ``basins``,
        ``steps`` (intermediate products) and ``log`` (what each step changed;
        keys ending in ``_pipes`` hold your ``pipe_id`` values, keys ending in
        ``_segments`` hold ``segment_id`` values of ``steps["segments_raw"]``,
        inlet lists hold ``inlet_id`` values or input index labels).
    """
    dem, transform, dem_crs = _read_dem(dem_path)
    _check_inputs(pipes, inlets, outfalls, dem_crs)
    if delineate:
        try:
            from . import watersheds  # noqa: F401
            import pysheds  # noqa: F401
        except ImportError as exc:
            raise ImportError("catch-basin delineation needs pysheds: "
                              "pip install 'stormwater-network-qc[watersheds]', "
                              "or pass delineate=False") from exc
    r = WorkflowResult()
    log = r.log

    seg, log["clean_geometry"] = geometry.clean_geometry(pipes, tolerance)
    r.steps["segments_raw"] = seg
    seg, log["reconnected_segments"] = networks.reconnect_networks(
        seg, outfalls, max_gap, tolerance, outfall_tolerance=outfall_tolerance)
    seg, removed, table = networks.remove_isolated_networks(
        seg, outfalls, keep_ids=keep_networks, drop_ids=drop_networks,
        outfall_tolerance=outfall_tolerance)
    log["networks"] = table
    log["networks_to_inspect"] = table.index[table.category == "inspect"].tolist()
    log["removed_isolated_segments"] = removed.segment_id.tolist()
    log["removed_isolated_pipes"] = sorted(set(removed.parent_id))
    seg, dups = networks.remove_duplicates(seg)
    log["removed_duplicate_segments"] = dups.segment_id.tolist()
    log["removed_duplicate_pipes"] = sorted(set(dups.parent_id))

    seg, junctions = topology.build_nodes(seg, tolerance)
    matched = topology.outfall_nodes(junctions, outfalls, outfall_tolerance)
    if len(matched) < len(outfalls):
        warnings.warn(f"{len(outfalls) - len(matched)} outfall(s) lie farther than "
                      f"outfall_tolerance={outfall_tolerance} m from every pipe end; their "
                      "networks are not oriented and may be removed", stacklevel=2)
    seg, dist = topology.orient_to_outfalls(seg, junctions, outfalls, outfall_tolerance)
    log["reversed_segments"] = seg.loc[seg.reversed, "segment_id"].tolist()
    log["reversed_pipes"] = sorted(set(seg.loc[seg.reversed, "parent_id"]))
    r.steps["junctions_all"] = junctions

    sel, dropped = inl.select_inlets(inlets, seg, inlet_distance)
    log["inlets_dropped"] = _ids(dropped, "inlet_id")
    sel = inl.snap_inlets(sel, junctions)
    log["inlets_to_review"] = _ids(sel[sel.needs_review], "inlet_id")
    inlet_nodes = set(sel.node_id)

    conn = topology.connectivity(seg)
    seg, flagged = diameters.flag_diameter_outliers(seg, conn, depth_field=depth_field)
    log["diameter_outlier_segments"] = flagged
    log["diameter_outlier_pipes"] = sorted(set(seg.loc[seg.segment_id.isin(flagged), "parent_id"]))
    seg = diameters.reconstruct_diameters(seg, inlet_nodes, connector_diameter)
    log["diameter_source"] = seg.diameter_source.value_counts().to_dict()

    coarse = ctrans = None
    if dem_coarse_path:
        coarse, ctrans, _ = _read_dem(dem_coarse_path)
    seg = slopes.estimate_slopes(seg, junctions, dem, transform, coarse, ctrans,
                                 inlet_nodes, n, vmin, vmax)
    log["slope_source"] = seg.slope_source.value_counts().to_dict()
    r.steps["segments_before_merge"] = seg

    sel = inl.cluster_inlets(sel, cluster_eps)
    points = inl.cluster_points(sel)
    r.steps["cluster_points"] = points
    if delineate:
        from . import watersheds
        r.basins = watersheds.delineate(dem_coarse_path or dem_path, points, snap_distance)
    sel = inl.representative_inlets(sel, r.basins)
    seg, sel = inl.merge_connectors(seg, sel, n)
    log["inlet_roles"] = sel.role.value_counts().to_dict()

    junctions = swmm.drop_orphan_junctions(junctions, seg)
    junctions = swmm.node_depths(seg, junctions, cover)
    junctions = swmm.node_inverts(junctions, dem, transform)
    seg = swmm.conduit_offsets(seg, junctions)
    log["max_slope_error"] = float(np.nanmax(np.abs(swmm.swmm_slope(seg, junctions) - seg.slope)))
    r.segments, r.junctions, r.inlets = seg, junctions, sel
    return r
