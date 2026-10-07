"""Each injected defect must be detected and corrected (ground truth from the generator)."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from stormwater_qc import diameters, geometry, networks, slopes, swmm, topology
from stormwater_qc.synthetic import evaluate, make_synthetic_network
from stormwater_qc.workflow import run_workflow

SEEDS = range(5)


@pytest.fixture(scope="module", params=SEEDS)
def run(request, tmp_path_factory):
    net = make_synthetic_network(seed=request.param)
    tmp = tmp_path_factory.mktemp(f"seed{request.param}")
    fine = net.write_dem(str(tmp / "dem.tif"))
    coarse = net.write_dem(str(tmp / "dem5.tif"), coarse=True)
    result = run_workflow(net.pipes, net.inlets, net.outfalls, fine, coarse,
                          depth_field="depth", delineate=False)
    return net, result


def test_every_defect_corrected(run):
    report = evaluate(*run)
    assert report.matches_truth.all(), report.to_string()


def test_final_network_is_valid(run):
    net, r = run
    seg = r.segments
    assert networks.identify_networks(seg).nunique() == net.truth["n_systems"]
    assert (seg.diameter > 0).all()
    assert (seg.slope > 0).all()
    assert np.allclose(swmm.swmm_slope(seg, r.junctions), seg.slope)
    assert (r.junctions.depth > 0).all()
    # every conduit drains to an outfall following its direction
    outs = set(topology.outfall_nodes(r.junctions, net.outfalls))
    nxt = dict(zip(seg.us_node, seg.ds_node))
    for node in seg.us_node:
        seen = set()
        while node in nxt and node not in seen:
            seen.add(node)
            node = nxt[node]
        assert node in outs


def test_snap_endpoints_near_miss_and_t_junction():
    trunk = LineString([(0, 0), (100, 0)])
    near = LineString([(100.03, 0.02), (150, 0)])      # 3.6 cm gap at a trunk end
    lateral = LineString([(50, 10), (50, 0.05)])       # ends 5 cm off the trunk, mid-segment
    gdf = gpd.GeoDataFrame({"pipe_id": [1, 2, 3]}, geometry=[trunk, near, lateral])
    seg, report = geometry.clean_geometry(gdf, tolerance=0.1)
    assert report == {"snapped_to_vertex": 1, "snapped_to_segment": 1, "segments": 4}
    assert networks.identify_networks(seg, tolerance=1e-9).nunique() == 1


def test_duplicates_in_either_direction():
    a = LineString([(0, 0), (10, 0)])
    gdf = gpd.GeoDataFrame({"segment_id": [1, 2, 3]},
                           geometry=[a, LineString(a.coords[::-1]), LineString([(10, 0), (20, 0)])])
    kept, dup = networks.remove_duplicates(gdf)
    assert list(kept.segment_id) == [1, 3] and list(dup.segment_id) == [2]


def test_reconstruct_from_inlet_and_upstream():
    # inlet node 1 -> 2 -> 3; segment 1 missing, segment 2 known (0.6 m)
    seg = gpd.GeoDataFrame({"segment_id": [1, 2], "parent_id": [1, 2],
                            "us_node": [1, 2], "ds_node": [2, 3],
                            "diameter": [0.0, 0.6]},
                           geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])])
    out = diameters.reconstruct_diameters(seg, inlet_nodes={1}, connector_diameter=0.381)
    assert out.diameter.iloc[0] == pytest.approx(0.381)
    assert out.diameter_source.iloc[0] == "inlet_connector"


def test_manning_round_trip():
    s = slopes.slope_for_velocity(0.6, 2.0)
    assert slopes.full_flow_velocity(0.6, s) == pytest.approx(2.0)
    q = slopes.full_flow_discharge(0.6, s)
    assert slopes.diameter_for_discharge(q, s) == pytest.approx(0.6)


def test_watersheds(tmp_path):
    pytest.importorskip("pysheds")
    net = make_synthetic_network(seed=0)
    fine = net.write_dem(str(tmp_path / "dem.tif"))
    coarse = net.write_dem(str(tmp_path / "dem5.tif"), coarse=True)
    r = run_workflow(net.pipes, net.inlets, net.outfalls, fine, coarse, depth_field="depth")
    n_clusters = r.inlets.cluster_id.nunique()
    assert 0.9 * n_clusters <= len(r.basins) <= n_clusters
    assert (r.inlets.groupby("cluster_id").representative.sum() == 1).all()
