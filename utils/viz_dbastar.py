#!/usr/bin/env python3
"""Visualize the db-A* search tree.

1. Run dbastar with debug on, which writes /tmp/dynoplan/dbastar_debug.yaml:

     ./main_dbastar --env_file ENV --motionsFile MOTIONS \
         --models_base_path ../dynobench/models/ --debug 1

2. Open the viewer:

     python3 utils/viz_dbastar.py --env ENV

Interactive controls:
  slider / left-right arrows   replay the search (shift: x10, ctrl/up-down: x100)
  home / end                   first / last expansion
  click a node                 print its info and highlight its path to the root
  t                            toggle tree edges

Only the first two state coordinates (x, y) are drawn. Paths to the root use the
*final* parent pointers, so a node that was rewired later shows its final path.

Headless: --save out.png (final state, or --step K) or --animate out.gif.
"""

import argparse
import sys

import numpy as np
import yaml

try:
    Loader = yaml.CSafeLoader
except AttributeError:
    Loader = yaml.SafeLoader


def load_debug(path):
    with open(path) as f:
        d = yaml.load(f, Loader=Loader)
    nodes = d["nodes"]
    n = len(nodes)
    X = np.array([nd["x"] for nd in nodes], dtype=float)
    out = {
        "meta": {k: v for k, v in d.items() if k not in ("nodes", "expansions")},
        "X": X,
        "parent": np.array([nd["parent"] for nd in nodes], dtype=int),
        "g": np.array([nd["g"] for nd in nodes], dtype=float),
        "h": np.array([nd["h"] for nd in nodes], dtype=float),
        "f": np.array([nd["f"] for nd in nodes], dtype=float),
        "created": np.array([nd["created_at"] for nd in nodes], dtype=int),
        "first_exp": np.array(
            [nd["expanded_at"][0] if nd["expanded_at"] else -1 for nd in nodes],
            dtype=int,
        ),
        "num_exp": np.array([len(nd["expanded_at"]) for nd in nodes], dtype=int),
        "rewires": np.array([nd["rewires"] for nd in nodes], dtype=int),
        "motion": np.array([nd["motion"] for nd in nodes], dtype=int),
        "nodes": nodes,
        "n": n,
    }
    # expansions: [node, num_motions, num_valid, num_novel, num_rewired]
    E = np.array(d["expansions"], dtype=int).reshape(-1, 5)
    out["E"] = E
    trajs = []
    for nd in nodes:
        t = np.array(nd["traj"], dtype=float)
        if t.size == 0:
            trajs.append(None)
        else:
            trajs.append(t[:, :2])
    out["trajs"] = trajs
    return out


def draw_env(ax, env_file):
    import matplotlib.patches as mpatches

    with open(env_file) as f:
        env = yaml.load(f, Loader=Loader)
    e = env["environment"]
    lo, hi = e["min"], e["max"]
    ax.add_patch(
        mpatches.Rectangle(
            (lo[0], lo[1]),
            hi[0] - lo[0],
            hi[1] - lo[1],
            fill=False,
            ec="k",
            lw=1.5,
        )
    )
    for o in e.get("obstacles", []) or []:
        c, s = o["center"], o["size"]
        if o["type"] == "box":
            ax.add_patch(
                mpatches.Rectangle(
                    (c[0] - s[0] / 2, c[1] - s[1] / 2),
                    s[0],
                    s[1],
                    fc="0.45",
                    ec="0.2",
                    zorder=1,
                )
            )
        elif o["type"] == "sphere":
            r = s[0] if isinstance(s, list) else s
            ax.add_patch(mpatches.Circle((c[0], c[1]), r, fc="0.45", ec="0.2"))
    pad = 0.05 * max(hi[0] - lo[0], hi[1] - lo[1])
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")


def path_to_root(D, i):
    path = []
    seen = set()
    while i >= 0 and i not in seen:
        seen.add(i)
        path.append(i)
        i = D["parent"][i]
    return path[::-1]


def path_xy(D, idxs):
    pts = []
    for i in idxs:
        t = D["trajs"][i]
        if t is not None:
            pts.append(t)
        pts.append(D["X"][i : i + 1, :2])
    return np.vstack(pts) if pts else np.zeros((0, 2))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--env", required=True, help="dynobench env yaml (obstacles)")
    p.add_argument("--debug", default="/tmp/dynoplan/dbastar_debug.yaml")
    p.add_argument(
        "--color",
        default="f",
        choices=["f", "g", "h", "created", "expanded", "rewires"],
        help="node color",
    )
    p.add_argument("--arrows", action="store_true", help="draw x[2] as heading")
    p.add_argument("--no-tree", action="store_true", help="hide motion edges")
    p.add_argument("--step", type=int, default=None, help="initial step")
    p.add_argument("--save", help="save a png of --step (default: last) and exit")
    p.add_argument("--animate", help="save an animation (gif/mp4) and exit")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    import matplotlib

    if args.save or args.animate:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import Slider

    print(f"loading {args.debug} ...", flush=True)
    D = load_debug(args.debug)
    meta, X, E = D["meta"], D["X"], D["E"]
    n_exp = len(E)
    last = max(n_exp - 1, 0)
    print(
        f"{meta['problem']} ({meta['robot']}): {meta['status']}, "
        f"{D['n']} nodes, {n_exp} expansions"
    )

    # color values
    cvals = {
        "f": D["f"],
        "g": D["g"],
        "h": D["h"],
        "created": D["created"].astype(float),
        "expanded": np.where(D["first_exp"] >= 0, D["first_exp"], np.nan).astype(
            float
        ),
        "rewires": D["rewires"].astype(float),
    }[args.color]
    finite = cvals[np.isfinite(cvals)]
    vmin, vmax = (np.percentile(finite, 1), np.percentile(finite, 99)) if len(
        finite
    ) else (0, 1)
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin, vmax)

    # order nodes by creation / first expansion -> prefix slicing per step
    order_c = np.argsort(D["created"], kind="stable")
    created_sorted = D["created"][order_c]
    expd = np.where(D["first_exp"] >= 0)[0]
    order_e = expd[np.argsort(D["first_exp"][expd], kind="stable")]
    exp_sorted = D["first_exp"][order_e]

    segs = [D["trajs"][i] for i in order_c]
    has_seg = np.array([s is not None for s in segs])

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[2.2, 1], height_ratios=[1, 1, 0.08], hspace=0.35
    )
    ax = fig.add_subplot(gs[0:2, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax_f)
    ax_s = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f"db-A*  {meta['problem']} ({meta['robot']})  {meta['status']}  "
        f"delta={meta['delta']:g} alpha={meta['alpha']:g}  "
        f"{D['n']} nodes, {n_exp} expansions",
        fontsize=11,
    )
    draw_env(ax, args.env)
    start, goal = np.array(meta["start"]), np.array(meta["goal"])
    r_goal = meta["delta"] * meta["delta_factor_goal"]
    ax.add_patch(
        plt.Circle(goal[:2], r_goal, fill=False, ec="r", ls="--", lw=1, zorder=5)
    )
    ax.plot(*start[:2], "o", c="lime", mec="k", ms=10, zorder=6, label="start")
    ax.plot(*goal[:2], "*", c="r", mec="k", ms=14, zorder=6, label="goal")

    tree = LineCollection([], lw=0.5, alpha=0.5, zorder=2)
    tree.set_cmap(cmap)
    tree.set_norm(norm)
    ax.add_collection(tree)
    open_sc = ax.scatter(
        np.zeros(0), np.zeros(0), s=10, facecolors="none", lw=0.7, zorder=3
    )
    open_sc.set_label("open (created)")
    closed_sc = ax.scatter(
        np.zeros(0), np.zeros(0), c=np.zeros(0), s=9, cmap=cmap, norm=norm,
        zorder=4,
    )
    closed_sc.set_label("closed (expanded)")
    quiv = None
    (cur_path_ln,) = ax.plot([], [], "-", c="orange", lw=2, zorder=7)
    (cur_pt,) = ax.plot([], [], "o", c="orange", mec="k", ms=9, zorder=8)
    (sol_ln,) = ax.plot([], [], "-", c="red", lw=2.5, zorder=7, label="solution" if meta["status"] == "SOLVED" else "closest to goal")
    (sel_ln,) = ax.plot([], [], "-", c="magenta", lw=2, zorder=9)
    (sel_pt,) = ax.plot([], [], "s", c="magenta", mec="k", ms=9, zorder=10)
    sol_xy = path_xy(D, path_to_root(D, meta["solution_node"]))
    fig.colorbar(closed_sc, ax=ax, fraction=0.035, pad=0.02, label=args.color)
    from matplotlib.lines import Line2D

    handles, _ = ax.get_legend_handles_labels()
    handles = [h for h in handles if h not in (open_sc, closed_sc)] + [
        Line2D([], [], ls="", marker="o", mfc="none", mec="tab:blue",
               label="open (created)"),
        Line2D([], [], ls="", marker="o", c="tab:blue",
               label="closed (expanded)"),
        Line2D([], [], c="orange", lw=2, label="path to popped node"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.8)

    # f/g/h of the popped node per expansion
    steps = np.arange(n_exp)
    popped = E[:, 0]
    ax_f.plot(steps, D["f"][popped], ".", ms=1.5, label="f")
    ax_f.plot(steps, D["g"][popped], ".", ms=1.5, label="g")
    ax_f.plot(steps, D["h"][popped], ".", ms=1.5, label="h")
    ax_f.plot(steps, np.maximum.accumulate(D["f"][popped]), "k-", lw=0.8,
              label="max f so far")
    ax_f.set_title("scores of popped node (final values)", fontsize=9)
    ax_f.legend(fontsize=7, markerscale=5, loc="upper left")
    vl_f = ax_f.axvline(0, c="orange")

    # branching statistics per expansion (moving average)
    def smooth(a, w=max(1, n_exp // 200)):
        return np.convolve(a, np.ones(w) / w, mode="same")

    ax_b.plot(steps, smooth(E[:, 1]), lw=1, label="motions tried")
    ax_b.plot(steps, smooth(E[:, 2]), lw=1, label="valid")
    ax_b.plot(steps, smooth(E[:, 3]), lw=1, label="novel (new node)")
    ax_b.plot(steps, smooth(E[:, 4]), lw=1, label="rewired")
    ax_b.set_title("per expansion (moving avg)", fontsize=9)
    ax_b.set_xlabel("expansion step")
    ax_b.legend(fontsize=7, loc="upper right")
    vl_b = ax_b.axvline(0, c="orange")

    slider = Slider(ax_s, "step", 0, last, valinit=last, valstep=1)
    show_tree = [not args.no_tree]

    def update(k):
        nonlocal quiv
        k = int(k)
        nc = np.searchsorted(created_sorted, k, side="right")
        ne = np.searchsorted(exp_sorted, k, side="right")
        idx_c = order_c[:nc]
        idx_e = order_e[:ne]
        if show_tree[0]:
            m = has_seg[:nc]
            tree.set_segments([s for s, ok in zip(segs[:nc], m) if ok])
            tree.set_array(cvals[idx_c][m])
        else:
            tree.set_segments([])
        is_closed = np.zeros(D["n"], bool)
        is_closed[idx_e] = True
        idx_o = idx_c[~is_closed[idx_c]]
        open_sc.set_offsets(X[idx_o, :2] if len(idx_o) else np.zeros((0, 2)))
        open_sc.set_edgecolors(cmap(norm(cvals[idx_o])))
        closed_sc.set_offsets(X[idx_e, :2] if len(idx_e) else np.zeros((0, 2)))
        closed_sc.set_array(cvals[idx_e])
        if args.arrows and X.shape[1] >= 3:
            if quiv is not None:
                quiv.remove()
            L = 0.6 * meta["delta"]
            quiv = ax.quiver(
                X[idx_c, 0], X[idx_c, 1], L * np.cos(X[idx_c, 2]),
                L * np.sin(X[idx_c, 2]), angles="xy", scale_units="xy",
                scale=1, width=0.0015, alpha=0.4, zorder=3,
            )
        if n_exp:
            cur = E[k, 0]
            xy = path_xy(D, path_to_root(D, cur))
            cur_path_ln.set_data(xy[:, 0], xy[:, 1])
            cur_pt.set_data([X[cur, 0]], [X[cur, 1]])
            st = E[k]
            ax.set_title(
                f"step {k}/{last}: popped node {cur} (f={D['f'][cur]:.2f} "
                f"g={D['g'][cur]:.2f} h={D['h'][cur]:.2f})\n"
                f"motions tried {st[1]}, valid {st[2]}, new {st[3]}, "
                f"rewired {st[4]} | open {len(idx_o)}, closed {ne}",
                fontsize=9,
            )
        if k == last:
            sol_ln.set_data(sol_xy[:, 0], sol_xy[:, 1])
        else:
            sol_ln.set_data([], [])
        vl_f.set_xdata([k, k])
        vl_b.set_xdata([k, k])
        fig.canvas.draw_idle()

    def node_info(i):
        nd = D["nodes"][i]
        x = ", ".join(f"{v:.3f}" for v in nd["x"])
        return (
            f"node {i}: x=[{x}] f={nd['f']:.3f} g={nd['g']:.3f} h={nd['h']:.3f} "
            f"parent={nd['parent']} motion={nd['motion']} "
            f"created_at={nd['created_at']} expanded_at={nd['expanded_at']} "
            f"open={nd['open']} rewires={nd['rewires']} "
            f"intermediate={nd['intermediate']}"
        )

    def on_click(ev):
        if ev.inaxes is not ax or ev.button != 1:
            return
        k = int(slider.val)
        vis = order_c[: np.searchsorted(created_sorted, k, side="right")]
        if not len(vis):
            return
        d = np.hypot(X[vis, 0] - ev.xdata, X[vis, 1] - ev.ydata)
        i = vis[np.argmin(d)]
        path = path_to_root(D, i)
        xy = path_xy(D, path)
        sel_ln.set_data(xy[:, 0], xy[:, 1])
        sel_pt.set_data([X[i, 0]], [X[i, 1]])
        print(node_info(i))
        print(f"  path to root ({len(path)} nodes): {path}")
        fig.canvas.draw_idle()

    def on_key(ev):
        k = int(slider.val)
        step = {"right": 1, "left": -1, "shift+right": 10, "shift+left": -10,
                "up": 100, "down": -100, "ctrl+right": 100, "ctrl+left": -100}
        if ev.key in step:
            slider.set_val(int(np.clip(k + step[ev.key], 0, last)))
        elif ev.key == "home":
            slider.set_val(0)
        elif ev.key == "end":
            slider.set_val(last)
        elif ev.key == "t":
            show_tree[0] = not show_tree[0]
            update(slider.val)

    slider.on_changed(update)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    k0 = last if args.step is None else int(np.clip(args.step, 0, last))
    slider.set_val(k0)
    update(k0)

    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"saved {args.save}")
        return
    if args.animate:
        from matplotlib.animation import FuncAnimation

        ks = np.unique(np.linspace(0, last, args.frames).astype(int))
        anim = FuncAnimation(fig, lambda k: slider.set_val(k), frames=ks)
        anim.save(args.animate, fps=args.fps, dpi=90)
        print(f"saved {args.animate}")
        return
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
