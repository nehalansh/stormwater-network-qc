"""Pipe diameters: implausible values and missing values.

Diameters are in metres. A missing diameter is 0 or NaN.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def flag_diameter_outliers(segments, conn, factor=1.5, depth_field=None):
    """Set implausible diameters to missing (0).

    A diameter is implausible when it exceeds ``factor`` times the largest
    diameter among the pipes immediately upstream and downstream of it, or,
    if ``depth_field`` is given, when it equals the depth of the pipe (the
    depth typed in the diameter field). Returns (segments, flagged ids).
    """
    d = segments.set_index("segment_id").diameter
    flagged = []
    for sid, row in conn.iterrows():
        nb = d.reindex(row.upstream + row.downstream).dropna()
        nb = nb[nb > 0]
        if d[sid] > 0 and len(nb) and d[sid] > factor * nb.max():
            flagged.append(sid)
    if depth_field:
        same = segments.loc[np.isclose(segments.diameter, segments[depth_field]), "segment_id"]
        flagged = sorted(set(flagged) | set(same))
    out = segments.copy()
    out.loc[out.segment_id.isin(flagged), "diameter"] = 0.0
    return out, flagged


def reconstruct_diameters(segments, inlet_nodes=(), connector_diameter=0.381):
    """Fill missing diameters from the neighbouring pipes (Shrestha et al., 2022).

    Segments of the same original pipe (``parent_id``), and missing pipes in
    pure series, are treated as one group. Pipes flowing into the head of the group are upstream, pipes
    leaving its end downstream:

    1. if some diameter appears both upstream and downstream, the largest such
       diameter is assigned to the group;
    2. otherwise the largest upstream diameter is assigned;
    3. otherwise, if the group starts at an inlet (a node in ``inlet_nodes``),
       ``connector_diameter`` is assigned;
    4. otherwise the group is left missing and reported for manual review.

    Requires ``us_node``/``ds_node`` oriented in the flow direction. Returns
    segments with filled diameters and a ``diameter_source`` column.
    """
    out = segments.copy()
    out["diameter"] = out.diameter.fillna(0.0)
    out["diameter_source"] = np.where(out.diameter > 0, "data", "missing")
    miss = out.index[out.diameter <= 0]
    if not len(miss):
        return out
    # groups: segments of the same original pipe (they share one diameter),
    # joined with missing pipes in pure series (a node with exactly one pipe
    # in and one out). A lateral meeting a run of missing pipes stays separate.
    sub = out.loc[miss]
    k = len(sub)
    n_in = out.ds_node.value_counts()
    n_out = out.us_node.value_counts()
    starts = pd.Series(np.arange(k), index=sub.us_node.to_numpy())
    a, b = [], []
    for i, dn in enumerate(sub.ds_node):
        if n_in.get(dn, 0) != 1 or n_out.get(dn, 0) != 1:
            continue
        for j in np.atleast_1d(starts.get(dn, [])):
            a.append(i)
            b.append(int(j))
    if "parent_id" in sub:
        for idx in sub.reset_index(drop=True).groupby(sub.parent_id.to_numpy()).indices.values():
            a.extend(idx[:-1])
            b.extend(idx[1:])
    adj = coo_matrix((np.ones(len(a)), (a, b)), shape=(k, k))
    _, lab = connected_components(adj, directed=False)
    known = out[out.diameter > 0]
    inlet_nodes = set(inlet_nodes)
    for g in np.unique(lab):
        members = sub.index[lab == g]
        grp = out.loc[members]
        heads = set(grp.us_node) - set(grp.ds_node)   # where the group starts
        tails = set(grp.ds_node) - set(grp.us_node)   # where it ends
        up = known.loc[known.ds_node.isin(heads), "diameter"].round(6)
        down = known.loc[known.us_node.isin(tails), "diameter"].round(6)
        common = set(up) & set(down)
        if common:
            value, src = max(common), "common_up_down"
        elif len(up):
            value, src = up.max(), "max_upstream"
        elif heads & inlet_nodes:
            value, src = connector_diameter, "inlet_connector"
        else:
            continue
        out.loc[members, "diameter"] = value
        out.loc[members, "diameter_source"] = src
    out.loc[out.diameter <= 0, "diameter_source"] = "unresolved"
    return out
