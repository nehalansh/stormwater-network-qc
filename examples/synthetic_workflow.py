"""Run the full workflow on the synthetic network and report the corrections.

    python examples/synthetic_workflow.py [--seed 0] [--figure docs/img/synthetic_workflow.png]

Builds a synthetic street-grid drainage system with the defect types commonly found
in municipal infrastructure data, runs ``run_workflow`` on it,
prints how each injected defect was corrected compared with the ground
truth, and draws the network before and after.
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from stormwater_qc.synthetic import evaluate, make_synthetic_network
from stormwater_qc.workflow import run_workflow

# categorical slots 1-6 of the reference palette, in their validated order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#8a897f", "#d9d8d2", "#fcfcfb"


def plot(net, result, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = net.truth
    raw = net.pipes
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # before: raw layers with each defect highlighted
    ax = axes[0]
    raw.plot(ax=ax, color=GRID, linewidth=1.2)
    groups = [("isolated fragment", t["isolated"]), ("duplicate", t["duplicate"]),
              ("digitized against flow", t["reversed"]),
              ("missing / implausible diameter", t["zero_diameter"] | t["depth_as_diameter"])]
    handles = []
    for (label, ids), color in zip(groups, COLORS):
        raw[raw.pipe_id.isin(ids)].plot(ax=ax, color=color, linewidth=2)
        handles.append(Line2D([], [], color=color, linewidth=2, label=label))
    disp = net.inlets[net.inlets.inlet_id.isin(t["displaced_inlet"])]
    remote = net.inlets[net.inlets.inlet_id.isin(t["remote_inlet"])]
    disp.plot(ax=ax, color=COLORS[4], markersize=22, marker="o", edgecolor=SURFACE, linewidth=1.5)
    remote.plot(ax=ax, color=COLORS[5], markersize=30, marker="D", edgecolor=SURFACE, linewidth=1.5)
    handles += [Line2D([], [], color=COLORS[4], marker="o", linestyle="", markersize=7,
                       label="inlet off its pipe end"),
                Line2D([], [], color=COLORS[5], marker="D", linestyle="", markersize=7,
                       label="remote inlet")]
    gap = raw[raw.pipe_id.isin(t["disconnected"])].geometry.iloc[0].coords[-1]
    ax.annotate("3-m gap", gap, xytext=(gap[0] - 95, gap[1] + 45), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    net.outfalls.plot(ax=ax, color=INK, marker="^", markersize=50)
    ax.set_title("Raw layers: injected defects", color=INK, loc="left", fontsize=12)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2,
              frameon=False, fontsize=9, labelcolor=INK)

    # after: cleaned network, catch basins, representative inlets
    ax = axes[1]
    if result.basins is not None:
        result.basins.boundary.plot(ax=ax, color=GRID, linewidth=0.6)
    seg = result.segments
    seg.plot(ax=ax, color=INK, linewidth=0.6 + 2.2 * seg.diameter / seg.diameter.max())
    rep = result.inlets[result.inlets.role == "representative"]
    rep.plot(ax=ax, color=COLORS[0], markersize=14, edgecolor=SURFACE, linewidth=1)
    net.outfalls.plot(ax=ax, color=INK, marker="^", markersize=50)
    handles = [Line2D([], [], color=INK, linewidth=2, label="conduit (width ~ diameter)"),
               Line2D([], [], color=GRID, linewidth=1, label="catch basin"),
               Line2D([], [], color=COLORS[0], marker="o", linestyle="", markersize=6,
                      label="representative inlet"),
               Line2D([], [], color=INK, marker="^", linestyle="", markersize=7, label="outfall")]
    ax.set_title("After QC: connected, oriented, attribute-complete", color=INK,
                 loc="left", fontsize=12)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2,
              frameon=False, fontsize=9, labelcolor=INK)
    # same extent in both panels
    frames = [raw.total_bounds, net.inlets.total_bounds]
    if result.basins is not None:
        frames.append(result.basins.total_bounds)
    xmin = min(f[0] for f in frames) - 30
    ymin = min(f[1] for f in frames) - 30
    xmax = max(f[2] for f in frames) + 30
    ymax = max(f[3] for f in frames) + 30
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure", help="save a before/after figure to this path")
    ap.add_argument("--no-basins", action="store_true", help="skip delineation (no pysheds)")
    args = ap.parse_args()

    net = make_synthetic_network(seed=args.seed)
    print(f"Synthetic network: {len(net.pipes)} pipes, {len(net.inlets)} inlets, "
          f"{len(net.outfalls)} outfalls\n")
    with tempfile.TemporaryDirectory() as tmp:
        fine = net.write_dem(Path(tmp) / "dem_1m.tif")
        coarse = net.write_dem(Path(tmp) / "dem_5m.tif", coarse=True)
        result = run_workflow(net.pipes, net.inlets, net.outfalls, fine, coarse,
                              depth_field="depth", delineate=not args.no_basins)
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(evaluate(net, result).to_string(index=False))
    print(f"\nFinal network: {len(result.segments)} conduits, {len(result.junctions)} junctions, "
          f"{0 if result.basins is None else len(result.basins)} catch basins; "
          f"largest SWMM-slope error {result.log['max_slope_error']:.1e}")
    if args.figure:
        plot(net, result, args.figure)
        print(f"Figure written to {args.figure}")


if __name__ == "__main__":
    main()
