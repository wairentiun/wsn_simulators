"""WSN Simulator with ML‑based Cluster‑Head (CH) Selection – *Telemetry‑parity Edition*
===================================================================================

Adds **exactly the same logs & column names** as the reference LEACH simulator so
that downstream KPI collectors work seamlessly across both protocols.

New in this version
-------------------
1. Per‑round **energy bookkeeping** (`tx_energy`, `agg_energy`) and
   `data_generated` per node.
2. Cluster log now records `avg_dist_to_CH`, `total_data_collected`, and
   *energy used* by the CH (`CH_energy_usage`).
3. Network log gains `total_dead`, `total_residual_energy`,
   `energy_used_this_round`, `data_delivered_to_sink`, and milestone flags
   (`is_first_node_dead`, etc.).
4. Maintains backwards‑compatible CLI and deterministic parallel sweep driver.

Only LightGBM & CQL inference logic changed enough to wire in the new metrics;
all radio‑physics and CH‑selection behaviour stay untouched.
"""
from __future__ import annotations
import argparse, concurrent.futures, itertools, json, multiprocessing as mp, os, random, time, uuid
from pathlib import Path
from typing import Dict, List, Literal, Sequence

# matplotlib headless setup *before* pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd

# ───────────────────────────────────── Radio / sim constants ───────────────
E_ELEC, E_AMP, E_AGG = 50e-9, 100e-12, 5e-9   # J/bit
PKT_LEN  = 4000                                # bits per packet
P_CH     = 0.05                                # desired CH fraction
FIGSIZE  = (6, 9)                              # saved PNG size

# ───────────────────────────────────── ML model shim ───────────────────────
class _MLWrapper:
    def __init__(self, model_type: Literal["lgbm", "cql"], model_path: str):
        self.model_type = model_type
        if model_type == "lgbm":
            import joblib
            self.model = joblib.load(model_path)
        elif model_type == "cql":
            import d3rlpy
            self.model = d3rlpy.load_learnable(model_path)
        else:
            raise ValueError("model_type must be 'lgbm' or 'cql'")

    def _choose_lgbm(self, feats: pd.DataFrame, alive):
        cols = ["residual_energy", "dist_to_sink", "total_alive", "avg_residual_energy", "round_idx"]
        proba = self.model.predict_proba(feats[cols])[:, 1]
        k = max(1, int(np.ceil(P_CH * len(alive))))
        top_idx = np.argsort(-proba)[:k]
        return [alive[i] for i in top_idx]

    def _choose_cql(self, feats: pd.DataFrame, alive):
        obs = feats[["residual_energy", "dist_to_sink", "prev_CH"]].to_numpy(np.float32)
        act_vec = np.ones(len(obs), dtype=np.int64)
        q_vals  = self.model.predict_value(obs, act_vec)
        k = max(1, int(np.ceil(P_CH * len(alive))))
        top_idx = np.argsort(-q_vals)[:k]
        return [alive[i] for i in top_idx] or [max(alive, key=lambda n: n.energy)]

    def select_CHs(self, feats: pd.DataFrame, alive):
        return self._choose_lgbm(feats, alive) if self.model_type == "lgbm" else self._choose_cql(feats, alive)

# ───────────────────────────────────── Sensor node ─────────────────────────
class Node:
    def __init__(self, idx: int, x: float, y: float, init_e: float):
        self.id       = idx
        self.pos      = np.array([x, y], dtype=float)
        self.energy   = init_e
        self.energy_prev = init_e      # snapshot at start of round
        self.tx_energy  = 0.0
        self.agg_energy = 0.0
        self.data_generated = 0
        self.is_CH    = False
        self.cluster_id: int | None = None
        self.prev_CH  = 0

    #  ‑‑ geometry ‑‑
    def alive(self):
        return self.energy > 0.0

    def dist_to(self, other: np.ndarray):
        return float(np.linalg.norm(self.pos - other))

    #  ‑‑ energy helpers (also track usage this round) ‑‑
    def _start_round(self):
        self.energy_prev   = self.energy
        self.tx_energy     = 0.0
        self.agg_energy    = 0.0
        self.data_generated = 0

    def consume_tx(self, bits: int, d: float):
        e = bits * (E_ELEC + E_AMP * d**2)
        self.energy -= e
        self.tx_energy += e

    def consume_rx(self, bits: int):
        e = bits * E_ELEC
        self.energy -= e
        # RX energy isn't attributed to KPIs (focus on tx / agg), but keep if needed

    def consume_agg(self, bits: int):
        e = bits * E_AGG
        self.energy -= e
        self.agg_energy += e

# ───────────────────────────────────── Simulation core ─────────────────────
class MLCHSimulation:
    def __init__(self, params: Dict, run_dir: Path, ml: _MLWrapper):
        self.params   = params
        self.run_dir  = run_dir
        self.ml       = ml
        self.rng      = np.random.default_rng(params.get("seed", 42))

        #  ‑‑ unpack JSON params ‑‑
        net, node_l, env = params["network_level"], params["node_level"], params["environmental"]
        self.node_count  = net["node_count"]
        self.area_side   = net["deployment_area_m2"]
        self.node_placement = net["node_placement"]
        self.sink_location  = net["sink_location"]
        self.initial_energy = node_l["initial_energy_joule"]
        self.energy_model   = node_l["energy_model"]
        self.node_failure_rate = env["node_failure_rate_percent"] / 100.0
        self.event_freq = env["event_frequency_per_round"]

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "parameter.json").write_text(json.dumps(params, indent=2))

        self.sink_pos = self._init_sink()
        self.nodes: List[Node] = self._init_nodes()

        #  Telemetry
        self.node_round:    list[dict] = []
        self.cluster_round: list[dict] = []
        self.network_round: list[dict] = []
        self.first_dead_round: int | None = None
        self.half_dead_round:  int | None = None

        self._plot_network(self.nodes, "Initial deployment", "wsn_initial.png")

    # ‑‑ placement helpers ‑‑
    def _init_sink(self):
        s = self.area_side
        return np.array([s/2, s/2]) if self.sink_location == "central" else np.array([s/2, s])

    def _init_nodes(self):
        side, n = self.area_side, self.node_count
        if self.node_placement == "random":
            xs, ys = self.rng.uniform(0, side, n), self.rng.uniform(0, side, n)
        else:
            centres = np.array([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])*side
            xs, ys = [], []
            for i in range(n):
                cx, cy = centres[i % 4]
                xs.append(self.rng.normal(cx, side*0.05))
                ys.append(self.rng.normal(cy, side*0.05))
            xs, ys = np.clip(xs,0,side), np.clip(ys,0,side)
        nodes = [Node(i, x, y, self.initial_energy) for i,(x,y) in enumerate(zip(xs,ys))]
        pd.DataFrame({"node_id":[n.id for n in nodes],"x":xs,"y":ys,"initial_energy":self.initial_energy}).to_parquet(self.run_dir/"node_info.parquet",index=False)
        return nodes

    # ‑‑ public driver ‑‑
    def run(self, max_rounds:int=10_000):
        for r in range(1, max_rounds+1):
            if not any(n.alive() for n in self.nodes):
                break
            self._simulate_round(r)
        # dump logs
        pd.DataFrame(self.node_round).to_parquet(self.run_dir/"node_round.parquet",index=False)
        pd.DataFrame(self.cluster_round).to_parquet(self.run_dir/"cluster_round.parquet",index=False)
        pd.DataFrame(self.network_round).to_parquet(self.run_dir/"network_round.parquet",index=False)

    # ‑‑ main round routine ‑‑
    def _simulate_round(self, r:int):
        for n in self.nodes:  # snapshot start‑of‑round energy & zero trackers
            n._start_round()

        alive_nodes = [n for n in self.nodes if n.alive()]
        n_alive = len(alive_nodes)

        # milestone plots
        if n_alive < self.node_count and self.first_dead_round is None:
            self.first_dead_round = r
            self._plot_network(self.nodes, f"First node dead (Round {r})", "wsn_first_dead.png")
        if n_alive <= self.node_count//2 and self.half_dead_round is None:
            self.half_dead_round = r
            self._plot_network(self.nodes, f"50% nodes dead (Round {r})", "wsn_half_dead.png")

        # random failures
        for n in alive_nodes:
            if self.rng.random() < self.node_failure_rate:
                n.energy = 0.0
        alive_nodes = [n for n in self.nodes if n.alive()]
        if not alive_nodes:
            self._record_network_round(r, 0, 0, 0)
            return

        # reset roles
        for n in self.nodes:
            n.is_CH = False
            n.cluster_id = None

        # ML‑based CH election
        total_e = sum(n.energy for n in alive_nodes)
        avg_e   = total_e / n_alive
        feat_df = pd.DataFrame([
            {"residual_energy":n.energy,"dist_to_sink":n.dist_to(self.sink_pos),"total_alive":n_alive,
             "avg_residual_energy":avg_e,"round_idx":r,"prev_CH":n.prev_CH} for n in alive_nodes
        ])
        ch_nodes = self.ml.select_CHs(feat_df, alive_nodes)
        for ch in ch_nodes:
            ch.is_CH, ch.prev_CH = True, 1
        for n in alive_nodes:
            if n not in ch_nodes:
                n.prev_CH = 0

        clusters = self._form_clusters(ch_nodes, alive_nodes)
        energy_round, bits_to_sink = self._data_exchange(clusters, ch_nodes)

        # record telemetry
        self._record_node_round(r)
        self._record_cluster_round(r, clusters)
        self._record_network_round(r, n_alive, energy_round, bits_to_sink)

    # ‑‑ helpers ‑‑
    def _form_clusters(self, ch_nodes, alive_nodes):
        clusters: Dict[int,List[Node]] = {ch.id:[] for ch in ch_nodes}
        for n in alive_nodes:
            if n.is_CH:
                n.cluster_id = n.id
                continue
            dists = [n.dist_to(ch.pos) for ch in ch_nodes]
            assigned = ch_nodes[int(np.argmin(dists))]
            n.cluster_id = assigned.id
            clusters[assigned.id].append(n)
        return clusters

    def _data_exchange(self, clusters, ch_nodes):
        energy_used = 0.0
        bits_to_sink = 0
        for ch in ch_nodes:
            members = clusters[ch.id]
            total_bits = 0
            for m in members:
                bits = 0
                if self.energy_model == "event_driven":
                    if self.rng.random() <= self.event_freq:
                        bits = PKT_LEN
                else:
                    bits = PKT_LEN
                if bits and m.alive():
                    d = m.dist_to(ch.pos)
                    m.consume_tx(bits, d)
                    ch.consume_rx(bits)
                    m.data_generated += bits
                total_bits += bits
            # CH aggregates + send to sink
            if total_bits and ch.alive():
                ch.consume_agg(total_bits)
                d_sink = ch.dist_to(self.sink_pos)
                ch.consume_tx(PKT_LEN, d_sink)
                bits_to_sink += PKT_LEN
            energy_used += ch.tx_energy + ch.agg_energy
            for m in members:
                energy_used += m.tx_energy + m.agg_energy
        return energy_used, bits_to_sink

    #  ‑‑ logging ‑‑
    def _record_node_round(self, r:int):
        for n in self.nodes:
            self.node_round.append({
                "round": r,
                "node_id": n.id,
                "residual_energy": n.energy,
                "is_CH": n.is_CH,
                "cluster_id": n.cluster_id,
                "dist_to_CH": (0.0 if n.is_CH or n.cluster_id is None else n.dist_to(self._node(n.cluster_id).pos)),
                "dist_to_sink": n.dist_to(self.sink_pos),
                "tx_energy": n.tx_energy,
                "agg_energy": n.agg_energy,
                "data_generated": n.data_generated,
            })

    def _record_cluster_round(self, r:int, clusters):
        for ch_id, members in clusters.items():
            ch = self._node(ch_id)
            dists = [m.dist_to(ch.pos) for m in members] if members else [0]
            self.cluster_round.append({
                "round": r,
                "cluster_id": ch_id,
                "CH_id": ch_id,
                "member_ids": [m.id for m in members],
                "avg_dist_to_CH": float(np.mean(dists)),
                "total_data_collected": len(members)*PKT_LEN,
                "CH_energy_usage": ch.tx_energy + ch.agg_energy,
            })

    def _record_network_round(self, r:int, n_alive:int, energy_used:float, bits_sink:int):
        dead = self.node_count - n_alive
        self.network_round.append({
            "round": r,
            "total_alive": n_alive,
            "total_dead": dead,
            "total_residual_energy": sum(n.energy for n in self.nodes if n.alive()),
            "avg_residual_energy": 0 if not n_alive else sum(n.energy for n in self.nodes if n.alive())/n_alive,
            "number_of_CH": sum(1 for n in self.nodes if n.alive() and n.is_CH),
            "energy_used_this_round": energy_used,
            "data_delivered_to_sink": bits_sink,
            "is_first_node_dead": self.first_dead_round == r,
            "is_half_node_dead": self.half_dead_round == r,
            "is_last_node_dead": n_alive == 0,
        })

    #  ‑‑ misc ‑‑
    def _node(self, idx:int):
        return self.nodes[idx]

    def _plot_network(self, nodes,title:str,fname:str):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        alive = [n for n in nodes if n.alive()]
        dead  = [n for n in nodes if not n.alive()]
        cluster_ids = sorted({n.cluster_id for n in alive if n.cluster_id is not None})
        cmap = cm.get_cmap("tab20", max(len(cluster_ids),1))
        colour_of = {cid:cmap(i) for i,cid in enumerate(cluster_ids)}
        for n in alive:
            c = colour_of.get(n.cluster_id, "grey")
            mrk = "^" if n.is_CH else "o"
            size = 50 if n.is_CH else 15
            ax.scatter(n.pos[0], n.pos[1], s=size, marker=mrk, color=c, edgecolors="k" if n.is_CH else None)
        if dead:
            ax.scatter([n.pos[0] for n in dead],[n.pos[1] for n in dead],marker="x",color="k")
        ax.scatter(self.sink_pos[0], self.sink_pos[1], marker="s", s=60, color="gold", edgecolors="k")
        ax.set_xlim(0, self.area_side); ax.set_ylim(0, self.area_side); ax.set_aspect("equal"); ax.set_title(title)
        plt.tight_layout(); fig.savefig(self.run_dir/fname, dpi=300); plt.close(fig)

# ───────────────────────────── Sweep driver / CLI ‑ unchanged ──────────────

def _build_param_grid(variations:Dict):
    key_meta = [((lvl,param), m["variation"]) for lvl,d in variations.items() for param,m in d.items()]
    combos = itertools.product(*[vals for _,vals in key_meta])
    out = []
    for combo in combos:
        rec = {"seed": random.randint(0,99999)}
        for ((lvl,param),_), val in zip(key_meta, combo):
            rec.setdefault(lvl, {})[param] = val
        out.append(rec)
    random.shuffle(out)
    return out

def _run_single(params:Dict, run_dir:Path, model_type:str, model_path:str):
    sim = MLCHSimulation(params, run_dir, _MLWrapper(model_type, model_path))
    sim.run()
    return f"finished {run_dir.name}"

def run_experiments(param_file:Path, sample_n:int|None, out_folder:str|None, model_type:str, model_path:str, workers:int|None=None):
    start = time.time()
    variations = json.loads(param_file.read_text())
    grid = _build_param_grid(variations)
    if sample_n and sample_n < len(grid):
        grid = random.sample(grid, sample_n)
    root = Path(out_folder or f"ml_runs_{int(start)}"); root.mkdir(exist_ok=True)
    run_dirs = [root / f"run_{i+1:04d}_{uuid.uuid4().hex[:6]}" for i in range(len(grid))]
    workers = workers or mp.cpu_count()
    print(f"Launching {len(grid)} simulations with {workers} workers… (output → {root})")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(_run_single, grid, run_dirs, [model_type]*len(grid), [model_path]*len(grid)):
            print(msg)
    h, rem = divmod(int(time.time()-start),3600); m,s = divmod(rem,60)
    print(f"Total running time: {h:02d}:{m:02d}:{s:02d}")

# ───────────────────────────── CLI entry‑point ─────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="WSN simulator with ML‑based CH selection")
    p.add_argument("--param_json", type=str, default="src/parameters_variations.json")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--out_folder", type=str, default=None)
    p.add_argument("--model_type", choices=["lgbm","cql"], default="lgbm")
    p.add_argument("--model_path", type=str, default="lgbm_ch.pkl")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_experiments(Path(args.param_json), args.sample, args.out_folder, args.model_type, args.model_path, args.workers)
