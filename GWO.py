"""LEACH‑based WSN Simulator
=================================
Simulates Wireless Sensor Networks using the classic LEACH protocol for
cluster‑head (CH) selection.  Designed to exhaustively (or randomly) explore
parameter spaces defined in a *parameters_variations.json* file and record rich
per‑node, per‑cluster and network‑level telemetry.

Outputs per simulation run (saved in its own sub‑folder):
-----------------------------------------------------------------
* **wsn_initial.png**     – node & sink layout at round 0
* **wsn_first_dead.png**  – network state when first node dies
* **wsn_half_dead.png**   – network state when half the nodes have died
* **parameter.json**      – the specific parameter values of this run
* **node_info.parquet**   – static node metadata
* **node_round.parquet**  – per‑round node telemetry
* **cluster_round.parquet** – per‑round cluster telemetry
* **network_round.parquet** – per‑round aggregated network telemetry

Usage
-----
```bash
python leach_simulator.py \
       --param_json parameters_variations.json \
       --sample 10                 # (optional) randomly simulate 10 combos
       --out_folder leach_runs
```

Requires: numpy, pandas, matplotlib, pyarrow (or fastparquet).
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.cm as cm          # <- add this import near the other mpl imports
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import os, time, concurrent.futures, multiprocessing as mp
from functools import partial

# ---------------------------------------------------------------------------
# Energy model constants (adapt as needed)
# ---------------------------------------------------------------------------
E_ELEC  = 50e-9      # J/bit for transmitter & receiver electronics
E_AMP   = 100e-12    # J/bit/m^2 for transmitter amplifier
E_AGG   = 5e-9       # J/bit for data aggregation at CH
PKT_LEN = 4000   # 4000 bytes → bits
P_CH    = 0.05       # desired CH probability per round

# Matplotlib sizing (6 in × 9 in preferred by user)
FIGSIZE = (6, 9)

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------
class Node:
    """Represents a single sensor node."""

    def __init__(self, node_id: int, x: float, y: float, init_energy: float):
        self.id: int = node_id
        self.pos: np.ndarray = np.array([x, y], dtype=float)
        self.energy: float = init_energy
        self.is_CH: bool = False
        self.cluster_id: int | None = None
        self.G: int = 0          # rounds left before eligible for CH again

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.energy > 0.0

    def dist_to(self, other: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos - other))

    # ------------------------------------------------------------------
    # Energy bookkeeping
    # ------------------------------------------------------------------
    def consume_tx(self, bits: int, dist: float):
        """Transmit *bits* over distance *dist* (m)."""
        self.energy -= bits * (E_ELEC + E_AMP * dist**2)

    def consume_rx(self, bits: int):
        """Receive *bits*."""
        self.energy -= bits * E_ELEC

    def consume_agg(self, bits: int):
        """Aggregate *bits*."""
        self.energy -= bits * E_AGG

# ────────────────────────────────────────────────────────────────
# Grey Wolf Optimizer (continuous version, adapted to WSN CH pick)
# ────────────────────────────────────────────────────────────────
class GreyWolfOptimizer:
    """Min-imiser – returns the best position vector found."""

    def __init__(self, fitness_fn, dim, lb, ub,
                 n_wolves=20, max_iter=30, rng=None):
        self.fitness_fn, self.dim = fitness_fn, dim
        self.lb, self.ub = np.asarray(lb), np.asarray(ub)
        self.n_wolves, self.max_iter = n_wolves, max_iter
        self.rng = np.random.default_rng() if rng is None else rng

    def _init_pack(self):
        return self.rng.uniform(self.lb, self.ub,
                                size=(self.n_wolves, self.dim))

    def optimise(self):
        X = self._init_pack()                     # (n_wolves, dim)
        fitness = np.apply_along_axis(self.fitness_fn, 1, X)
        alpha, beta, delta = self._rank(X, fitness)

        for t in range(self.max_iter):
            a = 2 - 2 * (t / self.max_iter)      # linearly ↓ 2→0
            for i in range(self.n_wolves):
                for d in range(self.dim):
                    r1, r2 = self.rng.random(2)
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1*alpha[d] - X[i, d])
                    X1 = alpha[d] - A1*D_alpha

                    r1, r2 = self.rng.random(2)
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2*beta[d] - X[i, d])
                    X2 = beta[d] - A2*D_beta

                    r1, r2 = self.rng.random(2)
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3*delta[d] - X[i, d])
                    X3 = delta[d] - A3*D_delta

                    X[i, d] = np.clip((X1+X2+X3)/3,
                                      self.lb[d], self.ub[d])

            fitness = np.apply_along_axis(self.fitness_fn, 1, X)
            alpha, beta, delta = self._rank(X, fitness)

        return alpha, self.fitness_fn(alpha)

    def _rank(self, X, fitness):
        idx = np.argsort(fitness)
        return X[idx[0]], X[idx[1]], X[idx[2]]

# ---------------------------------------------------------------------------
# Core simulation logic
# ---------------------------------------------------------------------------
class LeachSimulation:
    """Runs a single LEACH simulation with the given parameter set."""

    def __init__(self, params: Dict, run_dir: Path):
        self.params = params
        self.run_dir = run_dir
        self.rng = np.random.default_rng(params.get("seed", 42))

        # Unpack parameters -------------------------------------------------
        self.node_count          = params["network_level"]["node_count"]
        self.area_side           = params["network_level"]["deployment_area_m2"]
        self.node_placement      = params["network_level"]["node_placement"]
        self.sink_location       = params["network_level"]["sink_location"]
        self.initial_energy      = params["node_level"]["initial_energy_joule"]
        self.energy_model        = params["node_level"]["energy_model"]
        self.node_failure_rate   = params["environmental"]["node_failure_rate_percent"] / 100.0
        self.event_freq          = params["environmental"]["event_frequency_per_round"]

        # Create output directory and write parameter snapshot -------------
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "parameter.json").open("w") as f:
            json.dump(params, f, indent=2)

        # Initialise network ----------------------------------------------
        self.sink_pos = self._init_sink()
        self.nodes: List[Node] = self._init_nodes()

        # Telemetry containers -------------------------------------------
        self.node_round_records: list[dict] = []
        self.cluster_round_records: list[dict] = []
        self.network_round_records: list[dict] = []

        # Milestone trackers ---------------------------------------------
        self.first_dead_round: int | None = None
        self.half_dead_round: int | None = None

        # Plot initial layout --------------------------------------------
        self._plot_network(self.nodes, title="Initial deployment", fname="wsn_initial.png")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_sink(self) -> np.ndarray:
        side = self.area_side
        if self.sink_location == "central":
            return np.array([side / 2, side / 2])
        # "edge" – place at middle of top edge
        return np.array([side / 2, side])

    def _init_nodes(self) -> List[Node]:
        side = self.area_side
        n    = self.node_count
        nodes: list[Node] = []

        if self.node_placement == "random":
            xs = self.rng.uniform(0, side, n)
            ys = self.rng.uniform(0, side, n)
        else:  # "manual_clustered" – 4 clusters at quadrants
            cluster_centers = np.array([[side*0.25, side*0.25],
                                         [side*0.75, side*0.25],
                                         [side*0.25, side*0.75],
                                         [side*0.75, side*0.75]])
            xs, ys = [], []
            for i in range(n):
                cx, cy = cluster_centers[i % 4]
                xs.append(self.rng.normal(cx, side*0.05))
                ys.append(self.rng.normal(cy, side*0.05))
            xs, ys = np.clip(xs, 0, side), np.clip(ys, 0, side)

        for idx, (x, y) in enumerate(zip(xs, ys)):
            nodes.append(Node(idx, x, y, self.initial_energy))

        # Persist node_info ------------------------------------------------
        pd.DataFrame({
            "node_id": [n.id for n in nodes],
            "x": [n.pos[0] for n in nodes],
            "y": [n.pos[1] for n in nodes],
            "initial_energy": [self.initial_energy]*len(nodes)
        }).to_parquet(self.run_dir / "node_info.parquet", index=False)

        return nodes

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self, max_rounds: int = 10000):
        round_idx = 0
        while any(n.alive for n in self.nodes) and round_idx < max_rounds:
            round_idx += 1
            self._simulate_round(round_idx)

        # Final plots (if needed)
        if self.first_dead_round is None:
            print("Warning: simulation ended before any node died – increase max_rounds.")
        if self.half_dead_round is None:
            print("Warning: simulation ended before half nodes died – increase max_rounds.")

        # Persist parquet logs -----------------------------------------
        pd.DataFrame(self.node_round_records).to_parquet(self.run_dir / "node_round.parquet", index=False)
        pd.DataFrame(self.cluster_round_records).to_parquet(self.run_dir / "cluster_round.parquet", index=False)
        pd.DataFrame(self.network_round_records).to_parquet(self.run_dir / "network_round.parquet", index=False)

    # ------------------------------------------------------------------
    # Round‑level operations
    # ------------------------------------------------------------------
    def _simulate_round(self, r: int):
        alive_nodes = [n for n in self.nodes if n.alive]
        n_alive = len(alive_nodes)

        # Record first/half/last dead milestones ------------------------
        if n_alive < self.node_count and self.first_dead_round is None:
            self.first_dead_round = r
            self._plot_network(self.nodes, title=f"First node dead (Round {r})", fname="wsn_first_dead.png")
        if n_alive <= self.node_count // 2 and self.half_dead_round is None:
            self.half_dead_round = r
            self._plot_network(self.nodes, title=f"50% nodes dead (Round {r})", fname="wsn_half_dead.png")

        # Drop out nodes due to random failure -------------------------
        for n in alive_nodes:
            if self.rng.random() < self.node_failure_rate:
                n.energy = 0.0
        alive_nodes = [n for n in self.nodes if n.alive]
        if not alive_nodes:
            # All died due to failure this round
            self._record_network_round(r)
            return

        # Reset node roles -------------------------------------------
        for n in self.nodes:
            n.is_CH = False
            n.cluster_id = None

        # LEACH CH election -------------------------------------------
        ch_nodes = self._elect_CHs(r, alive_nodes)
        clusters = self._form_clusters(ch_nodes, alive_nodes)

        # Data transmission & energy accounting -----------------------
        self._data_exchange(clusters, ch_nodes, r)

        # Record telemetry -------------------------------------------
        self._record_node_round(r)
        self._record_cluster_round(r, clusters)
        self._record_network_round(r)

    # ------------------------------------------------------------------
    # GWO-based CH election (replaces probabilistic LEACH variant)
    # ------------------------------------------------------------------
    def _elect_CHs(self, r: int, alive_nodes: List[Node]) -> List[Node]:
        """Pick CHs with Grey Wolf Optimizer each round.

        Decision variables = k × 2D coordinates (continuous);
        After optimisation, we map each coordinate to the *nearest
        alive node with above-median residual energy* so we always
        end up with actual nodes as CHs.

        Tunables
        - k (desired CH count) – right now ⌈P_CH × alive⌉. Pass a custom ratio or a fixed integer.
        - Fitness weights – the 0.1 coefficient balances “short distances” vs “pick high-energy nodes”.
        - n_wolves & max_iter – more iterations/wolves ↑ solution quality but ↑ runtime.
        """
        k = max(1, int(P_CH * len(alive_nodes)))          # desired CH count

        coords      = np.array([n.pos for n in alive_nodes])
        energies    = np.array([n.energy for n in alive_nodes])
        side        = self.area_side
        high_energy = energies >= np.median(energies)

        # -------- fitness: weighted sum of (mean dist + CH energy term) --------
        def fitness(vec):
            # vec = [x1,y1,x2,y2,...,xk,yk]
            centres = vec.reshape(k, 2)
            # distance from each node to nearest centre
            d = np.min(
                np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=2),
                axis=1
            )
            # low avg distance good; high CH energy good
            dist_term   = d.mean()
            # encourage centres near high-energy nodes
            ch_ids = np.argmin(
                np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=2),
                axis=0
            )
            energy_term = -energies[ch_ids].mean()        # minus → maximise
            return dist_term + 0.1 * energy_term          # 0.1 = trade-off weight

        # -------- run GWO in the continuous (0,side)² space -----------
        dim  = 2 * k
        lb   = np.zeros(dim)
        ub   = np.full(dim, side)
        gwo  = GreyWolfOptimizer(fitness, dim, lb, ub,
                                n_wolves=15, max_iter=25, rng=self.rng)
        best_pos, _ = gwo.optimise()
        centres = best_pos.reshape(k, 2)

        # -------- map each centre to a real node (nearest & hi-energy) ---------
        ch_nodes: list[Node] = []
        taken_ids = set()
        for c in centres:
            dists = np.linalg.norm(coords - c, axis=1)
            # prefer high-energy nodes not already taken
            candidate_ids = np.argsort(dists)
            for idx in candidate_ids:
                nid = alive_nodes[idx].id
                if nid in taken_ids:
                    continue
                if high_energy[idx] or len(ch_nodes) == 0:  # always accept one
                    taken_ids.add(nid)
                    alive_nodes[idx].is_CH = True
                    ch_nodes.append(alive_nodes[idx])
                    break

        # Fallback: guarantee at least one CH
        if not ch_nodes:
            chosen = self.rng.choice(alive_nodes)
            chosen.is_CH = True
            ch_nodes.append(chosen)
        return ch_nodes

    def _form_clusters(self, ch_nodes: List[Node], alive_nodes: List[Node]):
        clusters: dict[int, list[Node]] = {ch.id: [] for ch in ch_nodes}
        for n in alive_nodes:
            if n.is_CH:
                n.cluster_id = n.id
                continue
            # Assign to nearest CH
            dists = [n.dist_to(ch.pos) for ch in ch_nodes]
            assigned_ch = ch_nodes[int(np.argmin(dists))]
            n.cluster_id = assigned_ch.id
            clusters[assigned_ch.id].append(n)
        return clusters

    def _data_exchange(self, clusters: Dict[int, List[Node]], ch_nodes: List[Node], r: int):
        # Event‑driven nodes may generate data only with some probability
        for ch in ch_nodes:
            members = clusters[ch.id]
            total_bits_from_members = 0
            # Member transmission
            for m in members:
                if self.energy_model == "event_driven" and self.rng.random() > self.event_freq:
                    bits_generated = 0
                else:
                    bits_generated = PKT_LEN
                if bits_generated > 0 and m.alive:
                    dist = m.dist_to(ch.pos)
                    m.consume_tx(bits_generated, dist)
                    ch.consume_rx(bits_generated)
                total_bits_from_members += bits_generated

            # CH aggregates & transmits to sink (even if no members)
            if total_bits_from_members > 0 and ch.alive:
                ch.consume_agg(total_bits_from_members)
                dist_sink = ch.dist_to(self.sink_pos)
                ch.consume_tx(PKT_LEN, dist_sink)  # aggregated packet same size

    # ------------------------------------------------------------------
    # Telemetry collectors
    # ------------------------------------------------------------------
    def _record_node_round(self, r: int):
        for n in self.nodes:
            self.node_round_records.append({
                "round": r,
                "node_id": n.id,
                "residual_energy": n.energy,
                "is_CH": n.is_CH,
                "cluster_id": n.cluster_id,
                "dist_to_CH": (0.0 if n.is_CH or n.cluster_id is None
                                 else n.dist_to(self._node_by_id(n.cluster_id).pos)),
                "dist_to_sink": n.dist_to(self.sink_pos),
                "tx_energy": None,         # populated if needed
                "agg_energy": None,        # populated if needed
                "data_generated": None     # populated if needed
            })

    def _record_cluster_round(self, r: int, clusters: Dict[int, List[Node]]):
        for ch in clusters:
            members = clusters[ch]
            dists = [m.dist_to(self._node_by_id(ch).pos) for m in members] if members else [0]
            self.cluster_round_records.append({
                "round": r,
                "cluster_id": ch,
                "CH_id": ch,
                "member_ids": [m.id for m in members],
                "avg_dist_to_CH": float(np.mean(dists)),
                "total_data_collected": len(members) * PKT_LEN,
                "CH_energy_usage": self._node_by_id(ch).energy
            })

    def _record_network_round(self, r: int):
        alive = [n for n in self.nodes if n.alive]
        dead  = len(self.nodes) - len(alive)
        self.network_round_records.append({
            "round": r,
            "total_alive": len(alive),
            "total_dead": dead,
            "total_residual_energy": sum(n.energy for n in alive),
            "avg_residual_energy": 0 if not alive else sum(n.energy for n in alive)/len(alive),
            "number_of_CH": sum(1 for n in alive if n.is_CH),
            "energy_used_this_round": None,   # can be filled with diff if desired
            "data_delivered_to_sink": None,   # optional
            "is_first_node_dead": self.first_dead_round == r,
            "is_half_node_dead": self.half_dead_round == r,
            "is_last_node_dead": len(alive) == 0
        })

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _plot_network(self, nodes: List[Node], title: str, fname: str):
        """Draw the WSN; colour members by their cluster-head ID."""
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # ───────────────────────── Gather node groups ─────────────────────────
        alive = [n for n in nodes if n.alive]
        dead  = [n for n in nodes if not n.alive]

        # Build a reproducible colour map: cluster_id → RGB
        # ◾ We want the same CH to keep the same colour across rounds.
        cluster_ids = sorted({n.cluster_id for n in alive if n.cluster_id is not None})
        n_clusters  = max(len(cluster_ids), 1)
        cmap        = cm.get_cmap('tab20', n_clusters)   # tab20 gives 20 visually distinct hues
        colour_of   = {cid: cmap(i) for i, cid in enumerate(cluster_ids)}

        # ───────────────────────── Plot alive MEMBER nodes ────────────────────
        for n in alive:
            if n.is_CH:
                continue
            c = colour_of.get(n.cluster_id, "grey")   # grey fallback (e.g. before first round)
            ax.scatter(n.pos[0], n.pos[1], s=15, color=c, alpha=0.9)

        # ───────────────────────── Plot cluster-heads ─────────────────────────
        for n in alive:
            if n.is_CH:
                c = colour_of.get(n.id, "k")          # CH uses its own cluster colour
                ax.scatter(n.pos[0], n.pos[1], marker="^", s=50, color=c, edgecolors='k')

        # ───────────────────────── Plot dead nodes & sink ─────────────────────
        ax.scatter([n.pos[0] for n in dead],  [n.pos[1] for n in dead],
                marker="x", color="k", label="Dead")
        ax.scatter(self.sink_pos[0], self.sink_pos[1],
                marker="s", s=60, color="gold", edgecolors='k', label="Sink")

        # ───────────────────────── Final cosmetics ────────────────────────────
        ax.set_xlim(0, self.area_side)
        ax.set_ylim(0, self.area_side)
        ax.set_aspect("equal")
        ax.set_title(title)

        # Create a simple legend entry for CHs / Dead / Sink only
        ch_patch   = plt.Line2D([], [], marker="^", linestyle="None", color="w",
                                markerfacecolor="lightgrey", markeredgecolor='k',
                                markersize=8, label="Cluster-Head")
        dead_patch = plt.Line2D([], [], marker="x", linestyle="None", color="k",
                                markersize=8, label="Dead")
        sink_patch = plt.Line2D([], [], marker="s", linestyle="None", color="gold",
                                markeredgecolor='k', markersize=8, label="Sink")
        ax.legend(handles=[ch_patch, dead_patch, sink_patch], loc="upper right", framealpha=0.9)

        plt.tight_layout()
        fig.savefig(self.run_dir / fname, dpi=300)
        plt.close(fig)

    def _node_by_id(self, nid: int) -> Node:
        return self.nodes[nid]


# ---------------------------------------------------------------------------
# Parameter space exploration driver
# ---------------------------------------------------------------------------

def build_param_grid(variations: Dict) -> List[Dict]:
    """Expand the JSON variations into a list of concrete parameter dicts."""
    # Flatten JSON tree into ordered lists per level
    keys_vals = []
    for level_key, level_dict in variations.items():
        for param, meta in level_dict.items():
            keys_vals.append(((level_key, param), meta["variation"]))

    combos = list(itertools.product(*[v for _, v in keys_vals]))
    print(f'Number of params combinations: {len(combos)}')

    param_dicts: List[Dict] = []
    for combo in combos:
        d: Dict = {"seed": random.randint(0, 99999)}  # randomise seed for run diversity
        for ((level, param), _), value in zip(keys_vals, combo):
            d.setdefault(level, {})[param] = value
        param_dicts.append(d)

    random.shuffle(param_dicts)
    return param_dicts

def load_params_already_ran(runs_folder: str):
    from glob import glob
    ran_params = []
    all_rans = glob(f'{runs_folder}/run_*/parameter.json')
    for ran in all_rans:
        ran_param = json.loads(Path(ran).read_text())
        del ran_param['seed']
        ran_params.append(ran_param)
    return ran_params

def _run_single(params: Dict, run_dir: Path):
    """
    Worker function executed in a separate process.
    It builds and runs a single LeachSimulation instance.
    """
    try:
        # Matplotlib must be in non-interactive mode inside subprocesses
        import matplotlib
        matplotlib.use("Agg")

        sim = LeachSimulation(params, run_dir)
        sim.run()                      # ← long-running work
        return f"FINISHED {run_dir.name}"
    except Exception as e:             # catch *everything* so the pool keeps going
        return f"ERROR    {run_dir.name}: {e!r}"

def run_experiments(param_file: Path,
                    sample_n: int | None,
                    runs_folder: str | None,
                    max_workers: int | None = None):
    """
    Read the variation JSON, build the parameter grid, then farm out each
    parameter set to its own Python process via ProcessPoolExecutor.
    """
    tic = time.time()
    variations = json.loads(Path(param_file).read_text())
    grid       = build_param_grid(variations)

    # ───────────────────────── Optional random sampling ─────────────────────────
    if sample_n is not None and sample_n < len(grid):
        grid = random.sample(grid, sample_n)

    # ───────────────────────── Output root folder handling ───────────────────────
    if not runs_folder:
        for i in range(1, 10_000):
            tentative = f"runs_{i:02d}"
            if not os.path.exists(tentative):
                runs_folder = tentative
                break
    root_out = Path(runs_folder)
    root_out.mkdir(exist_ok=True)

    # ───────────────────────── Skip combos already completed ─────────────────────
    ran_params = load_params_already_ran(root_out) if root_out.exists() else []
    pending    = []
    run_dirs   = []
    idx_start  = len(ran_params) + 1
    for params in grid:
        if {k: v for k, v in params.items() if k != "seed"} in ran_params:
            continue
        run_id   = f"run_{idx_start:04d}_{uuid.uuid4().hex[:6]}"
        run_dir  = root_out / run_id
        pending.append(params)
        run_dirs.append(run_dir)
        idx_start += 1

    if not pending:
        print("Nothing to do – every combination already simulated.")
        return

    # ───────────────────────── Launch pool ───────────────────────────────────────
    max_workers = max_workers or mp.cpu_count()
    print(f"Launching {len(pending)} simulations with {max_workers} workers…")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for msg in pool.map(_run_single, pending, run_dirs):
            print(msg)

    # ───────────────────────── Timing summary ────────────────────────────────────
    h, rem = divmod(int(time.time() - tic), 3600)
    m, s   = divmod(rem, 60)
    print(f"\nTotal running time: {h:02d}:{m:02d}:{s:02d} (hh:mm:ss)")

# ---------------------------------------------------------------------------
# CLI entry-point with an extra --workers flag
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="LEACH-based WSN parameter-sweep simulator (parallel)")
    p.add_argument("--param_json", type=str, default="src/parameters_variations.json",
                   help="Path to JSON file defining parameter variations")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly run this many parameter combinations instead of exhaustive sweep")
    p.add_argument("--out_folder", type=str, default=None,
                   help="Root folder to save runs; auto-increments if omitted")
    p.add_argument("--workers", type=int, default=None,
                   help="Processes to launch (default = CPU-count)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_experiments(Path(args.param_json),
                    args.sample,
                    args.out_folder,
                    args.workers)
