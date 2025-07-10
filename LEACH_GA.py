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
       --param_json src/parameters_variations.json \
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
# Simple Genetic Algorithm for CH subset selection
# ────────────────────────────────────────────────────────────────
class GeneticCHSelector:
    """
    GA that evolves a binary chromosome of length N (alive nodes);
    a '1' bit means the node is chosen as a CH.
    Exactly k ones are enforced in every chromosome.

    Tweakable
    30 individuals, 25 generations, single-point crossover (0.8), bit-flip mutation (0.05) | Fast (≈ few ms per round) but effective; tweak as needed.
    """

    def __init__(self, coords, energies, k,
                 pop_size=30, n_gen=25, pc=0.8, pm=0.1, rng=None):
        """
        coords  : (N, 2) array of node positions
        energies: (N,) residual energies
        k       : desired number of CHs
        """
        self.coords, self.energies, self.k = coords, energies, k
        self.pop_size, self.n_gen, self.pc, self.pm = pop_size, n_gen, pc, pm
        self.N = coords.shape[0]
        self.rng = np.random.default_rng() if rng is None else rng

    # ----------- fitness = low mean member distance – 0.1 × CH energy
    def _fitness(self, chromo):
        ch_idx = np.where(chromo)[0]
        ch_coords = self.coords[ch_idx]

        d = np.min(np.linalg.norm(self.coords[:, None, :] - ch_coords[None], axis=2), axis=1)
        dist_term = d.mean()
        energy_term = -self.energies[ch_idx].mean()
        return dist_term + 0.1 * energy_term

    # ----------- genetic operators
    def _tournament(self, pop, fit):
        i, j = self.rng.integers(0, self.pop_size, size=2)
        return pop[i] if fit[i] < fit[j] else pop[j]

    def _crossover(self, p1, p2):
        """Single-point crossover with feasibility repair."""
        # If crossover not chosen OR chromosome too short, just copy parents
        if self.rng.random() > self.pc or self.N <= 2:
            return p1.copy(), p2.copy()

        # safe: here self.N >= 3, so 1 <= cut < self.N-1
        cut = self.rng.integers(1, self.N - 1)
        c1 = np.concatenate([p1[:cut], p2[cut:]])
        c2 = np.concatenate([p2[:cut], p1[cut:]])
        return self._repair(c1), self._repair(c2)

    def _mutate(self, c):
        for i in range(self.N):
            if self.rng.random() < self.pm:
                c[i] ^= 1  # flip bit
        return self._repair(c)

    # ensure exactly k ones
    def _repair(self, chromo):
        ones = np.where(chromo)[0]
        zeros = np.where(~chromo)[0]
        if len(ones) > self.k:
            turn_off = self.rng.choice(ones, len(ones) - self.k, replace=False)
            chromo[turn_off] = 0
        elif len(ones) < self.k:
            turn_on = self.rng.choice(zeros, self.k - len(ones), replace=False)
            chromo[turn_on] = 1
        return chromo

    # ----------- main loop
    def evolve(self):
        pop = np.array([self._repair(self.rng.choice([0, 1], self.N)) for _ in range(self.pop_size)])
        fitness = np.array([self._fitness(c) for c in pop])

        for _ in range(self.n_gen):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])
            pop = np.array(new_pop[:self.pop_size])
            fitness = np.array([self._fitness(c) for c in pop])

        best = pop[np.argmin(fitness)]
        return np.where(best)[0]          # indices of chosen CHs

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
    # GA-based CH election (LEACH-GA)
    # ------------------------------------------------------------------
    def _elect_CHs(self, r: int, alive_nodes: List[Node]) -> List[Node]:
        """Pick cluster-heads via a genetic algorithm every round."""
        k = max(1, int(P_CH * len(alive_nodes)))      # desired #CH

        coords   = np.array([n.pos for n in alive_nodes])
        energies = np.array([n.energy for n in alive_nodes])

        ga = GeneticCHSelector(coords, energies, k,
                            pop_size=30, n_gen=25,
                            pc=0.8, pm=0.05, rng=self.rng)
        ch_indices = ga.evolve()

        ch_nodes: list[Node] = []
        for idx in ch_indices:
            node = alive_nodes[idx]
            node.is_CH = True
            ch_nodes.append(node)

        # safety net
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
    print('Sleep for 2hours before start run')
    time.sleep(2*60*60)
    run_experiments(Path(args.param_json),
                    args.sample,
                    args.out_folder,
                    args.workers)
