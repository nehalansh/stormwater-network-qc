"""Junction depths, invert elevations and conduit offsets for SWMM.

SWMM does not take a conduit slope as input: it computes it from the invert
elevations of the end nodes and the conduit offsets,

    slope = [(invert_us + offset_in) - (invert_ds + offset_out)] / length.

These functions derive junction depths and inverts from pipe diameters and
the ground surface, and the offsets that make every conduit reproduce its
estimated slope.
"""
import numpy as np
import pandas as pd

from .slopes import sample_dem


def node_depths(segments, junctions, cover=0.9144):
    """Junction depth (m) = ``cover`` + largest diameter of the connected pipes.

    The default cover is 36 in. Junctions with no pipe get ``cover``.
    """
    d = pd.concat([segments[["us_node", "diameter"]].rename(columns={"us_node": "node"}),
                   segments[["ds_node", "diameter"]].rename(columns={"ds_node": "node"})])
    dmax = d.groupby("node").diameter.max()
    out = junctions.copy()
    out["depth"] = cover + dmax.reindex(out.node_id).fillna(0.0).to_numpy()
    return out


def node_inverts(junctions, dem, transform):
    """Rim elevation from the DEM and invert = rim - depth (needs ``depth``)."""
    out = junctions.copy()
    out["rim_elev"] = sample_dem(junctions, dem, transform).to_numpy()
    out["invert_elev"] = out.rim_elev - out.depth
    return out


def conduit_offsets(segments, junctions, slope_field="slope"):
    """Inlet/outlet offsets (m) that impose each conduit's slope.

    offset_in = slope * length - invert_us + invert_ds; where this is
    negative it is applied at the outlet end instead (offset_out = -offset_in,
    offset_in = 0), so node elevations are not changed.
    """
    inv = junctions.set_index("node_id").invert_elev
    out = segments.copy()
    z_us = inv.loc[out.us_node].to_numpy()
    z_ds = inv.loc[out.ds_node].to_numpy()
    off = out[slope_field].to_numpy() * out.length.to_numpy() - z_us + z_ds
    out["offset_in"] = np.where(off < 0, 0.0, off)
    out["offset_out"] = np.where(off < 0, -off, 0.0)
    return out


def swmm_slope(segments, junctions):
    """Slope SWMM will compute from inverts and offsets (for checking)."""
    inv = junctions.set_index("node_id").invert_elev
    z_us = inv.loc[segments.us_node].to_numpy() + segments.offset_in.to_numpy()
    z_ds = inv.loc[segments.ds_node].to_numpy() + segments.offset_out.to_numpy()
    return (z_us - z_ds) / segments.length.to_numpy()


def drop_orphan_junctions(junctions, segments):
    """Junctions not connected to any segment removed."""
    used = set(segments.us_node) | set(segments.ds_node)
    return junctions[junctions.node_id.isin(used)].copy()
