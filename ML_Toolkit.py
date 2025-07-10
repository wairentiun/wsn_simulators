#!/usr/bin/env python
"""
WSN Cluster‑Head Learning Toolkit
================================
Supports **multiple simulation folders** — e.g. separate runs for *LEACH*,
*LEACH‑GA*, *GWO* — and unifies them into one training buffer.

Usage examples
--------------
```bash
# LightGBM behaviour‑cloning on three folders
python wsn_ch_learning_toolkit.py \
    --sim_roots leach_runs leach_ga_runs gwo_runs \
    --mode lgbm

# Offline RL with CQL, 100 epochs
python wsn_ch_learning_toolkit.py \
    --sim_roots leach_runs leach_ga_runs gwo_runs \
    --mode cql --epochs 100
```

Dependencies: lightgbm, pandas, numpy, pyarrow, torch, torch_geometric,
               d3rlpy, joblib
"""
from __future__ import annotations

import argparse, glob, math, os, time
from pathlib import Path
from typing import List
import pickle
import random

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow as pa

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm                        # ↘ live progress bar
import multiprocessing as _mp

def ensure_numpy_dtypes(df):
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number) and df[col].dtype != bool:
            # Attempt to promote to float; if that fails, try int
            try:
                df[col] = df[col].astype('float64')
            except Exception:
                df[col] = df[col].astype('int64')
    return df

# ---------------------------------------------------------------------------
P_CH = 0.05  # keep consistent with simulator settings
# ---------------------------------------------------------------------------
# Helper: gather all run_*/ folders from a list of roots
# ---------------------------------------------------------------------------

def get_run_dirs(roots: List[str | Path]):
    run_dirs = []
    for root in roots:
        run_dirs.extend(sorted(glob.glob(os.path.join(root, "run_*"))))
    if not run_dirs:
        raise FileNotFoundError("No run_* folders found in the provided roots")
    return run_dirs

# =============================================================================
# 1 ───────────────────────────────────── Dataset extraction helpers ──────────
# =============================================================================

NEEDED_NODE = [
    "round", "node_id", "is_CH",
    "residual_energy", "dist_to_sink"
]
NEEDED_NET  = [
    "round", "total_alive", "total_dead",
    "avg_residual_energy", "number_of_CH"
]

def _load_run(run_dir: str):
    node_path = Path(run_dir) / "node_round.parquet"
    net_path  = Path(run_dir) / "network_round.parquet"
    node_tb = pq.read_table(node_path, columns=NEEDED_NODE)
    net_tb  = pq.read_table(net_path , columns=NEEDED_NET )
    # stay in Arrow for now
    return node_tb, net_tb

# -- Supervised dataset -------------------------------------------------------

def _sup_worker(run_dir: str):
    node_tb, net_tb = _load_run(run_dir)
    # ❶ Arrow join on the 'round' key
    joined = node_tb.join(net_tb, keys='round', right_keys='round',
                          use_threads=True)
    # ❷ Add label column once for all rows
    joined = joined.append_column(
        'is_CH_label',
        pc.equal(joined['is_CH'], True)         # bool arrow array
    )
    # ❸ Add round_idx ≡ round (already present, rename if you prefer)
    joined = joined.rename_columns(
        list(joined.schema.names)
    )
    # ❹ Convert to Pandas only once, *after* all zero-copy work is done
    return joined

def extract_supervised_dataset(run_dirs: List[str], max_workers: int | None = None):
    max_workers = max_workers or _mp.cpu_count()
    pieces = []
    print(f"Building supervised dataset with {max_workers} workers …")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_sup_worker, rd): rd for rd in run_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Runs"):
            res = fut.result()
            if res is not None:
                pieces.append(res)
    all_tb = pa.concat_tables(pieces)
    all_df = all_tb.to_pandas(self_destruct=True)
    return all_df


# -- RL replay buffer ---------------------------------------------------------

# ───────────────────────── _replay_worker (optimised) ──────────────────────
def _replay_worker(run_dir: str):
    # --- 1. Load just once into Pandas (Arrow → NumPy zero-copy when possible)
    node_tb, net_tb = _load_run(run_dir)
    df = node_tb.to_pandas()                          # already only needed cols

    # --- 2. Sort so that each node’s rounds are contiguous
    df.sort_values(['node_id', 'round'], inplace=True, ignore_index=True)

    # NumPy aliases (no extra copies)
    resid  = df['residual_energy'].to_numpy(dtype=np.float32,  copy=False)
    dist   = df['dist_to_sink'   ].to_numpy(dtype=np.float32,  copy=False)
    is_ch  = df['is_CH'          ].to_numpy(dtype=np.float32,  copy=False)
    nodeid = df['node_id'        ].to_numpy(copy=False)

    # --- 3. Identify the LAST row for each node (where next-state is invalid)
    last_mask = np.empty_like(nodeid, dtype=bool)
    last_mask[:-1] = nodeid[1:] != nodeid[:-1]
    last_mask[-1]  = True                          # last row of the whole df

    # --- 4. Build state, next_state with one cheap roll()
    state      = np.stack([resid, dist, is_ch], axis=-1)
    next_state = np.roll(state, -1, axis=0)        # shift up by one row

    # --- 5. Alive flags & rewards (vectorised)
    alive_now  = resid > 0.0
    alive_next = np.roll(alive_now, -1)

    # --- global penalty: how far the *round* deviates from P_CH -------------
    net_df     = net_tb.to_pandas()              # cols: round, number_of_CH …
    k_target_r = (net_df["total_alive"] * P_CH).round().clip(lower=1)
    ratio_pen  = -np.abs(net_df["number_of_CH"] - k_target_r) / net_df["total_alive"]
    ratio_pen.index = net_df["round"]            #  map by round #

    # base survival reward + penalty (broadcast to every node of that round)
    reward = (alive_next.astype(np.float32) - alive_now.astype(np.float32))
    reward += ratio_pen.loc[df["round"]].to_numpy(np.float32)

    done       = ~alive_next                       # episode ends when node dies

    # --- 6. Keep only “valid” transitions (not last row per node)
    valid = ~last_mask

    # --- 7. Package in one shot (no Python loop)  →  list[dict] expected upstream
    # NOTE: zip() is C-level; negligible overhead compared with Python loop
    return [
        {'state': s, 'action': int(a), 'reward': r,
         'next_state': ns, 'done': d}
        for s, a, r, ns, d in zip(
            state[valid],
            is_ch[valid],
            reward[valid],
            next_state[valid],
            done[valid],
        )
    ]

def build_replay_buffer(run_dirs: list[str], max_workers: int | None = None):
    max_workers = max_workers or _mp.cpu_count()
    print(f"Building replay buffer with {max_workers} workers …")
    buffer = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_replay_worker, rd): rd for rd in run_dirs}
        # once = True
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Runs"):
            # if once:
            #     print(fut.result())
            #     once=False
            buffer.extend(fut.result())
    return buffer

# =============================================================================
# 2 ───────────────────────────────────── LightGBM supervised scorer ──────────
# =============================================================================

def train_lightgbm(X: pd.DataFrame, y: pd.Series, model_path: str = "src/lgbm_ch.pkl"):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        colsample_bytree=0.8,
        subsample=0.8,
    )
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    print(f"LightGBM saved → {model_path}")
    return clf

# =============================================================================
# 3 ───────────────────────────────────── Offline RL with CQL ────────────────
# =============================================================================

import torch
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos      import DiscreteCQLConfig 
from d3rlpy.optimizers import AdamFactory
from d3rlpy.logging import FileAdapterFactory
from d3rlpy.constants import LoggingStrategy
from d3rlpy.metrics.evaluators import (
    TDErrorEvaluator,
    DiscountedSumOfAdvantageEvaluator,
    AverageValueEstimationEvaluator,
    InitialStateValueEstimationEvaluator,
    DiscreteActionMatchEvaluator,
)

def transitions_to_mdp(tr: list[dict]) -> MDPDataset:
    """
    Vectorised and dtype-safe.
      • observations remain float32
      • actions are int64  → d3rlpy sees DISCRETE(2)
    """
    obs       = np.stack([t['state']       for t in tr], dtype=np.float32)
    next_obs  = np.stack([t['next_state']  for t in tr], dtype=np.float32)
    actions   = np.asarray([t['action']    for t in tr], dtype=np.int64)
    rewards   = np.asarray([t['reward']    for t in tr], dtype=np.float32)
    terminals = np.asarray([t['done']      for t in tr], dtype=bool)

    return MDPDataset(
        observations       = obs,
        actions            = actions,      # int64 → DISCRETE
        rewards            = rewards,
        terminals          = terminals,
    )


def train_discrete_cql_policy(buffer: list[dict],
                              epochs: int = 50,
                              model_path: str = "src/cql_ch.d3",
                              logdir: str = "cql_logs"):

    dataset = transitions_to_mdp(buffer)

    # shared Adam optimiser factory (lr will be set in the config)
    adam = AdamFactory(weight_decay=0.0)

    cfg = DiscreteCQLConfig(
        batch_size     = 4096,
        learning_rate  = 3e-4,     # ← single LR
        gamma          = 0.99,
        optim_factory  = adam,
        n_critics      = 2,
        alpha          = 1.0,
        target_update_interval = 8000,
    )

    algo   = cfg.create(device=torch.cuda.is_available())
    algo.build_with_dataset(dataset)

    steps_per_epoch = 100
    total_steps     = epochs * steps_per_epoch

    print(f"Training on device: {algo._device}")
    algo.fit(
        dataset,
        n_steps            = total_steps,
        n_steps_per_epoch  = steps_per_epoch,
        experiment_name    = "cql",
        with_timestamp     = False,
        logging_steps      = 500,
        logging_strategy   = LoggingStrategy.EPOCH,
        logger_adapter     = FileAdapterFactory(root_dir=logdir),
        show_progress      = True,
        save_interval      = 100,          # save at the end; change if you want
        evaluators         = None,
    )

    algo.save(model_path)
    print(f"Discrete CQL policy saved → {model_path}")
    return algo

# =============================================================================
# 4 ───────────────────────────────────── CLI entry‑point ─────────────────────
# =============================================================================

def main(sim_roots: List[str], mode: str, epochs: int, sample_runs: int | None):
    t0 = time.time()
    run_dirs = get_run_dirs(sim_roots)

    if sample_runs is not None and sample_runs < len(run_dirs):
        run_dirs = random.sample(run_dirs, sample_runs)
        print(f"Using a random subset of {sample_runs} runs out of {len(run_dirs)}")

    if mode == "lgbm":
        if os.path.exists('src/supervised_dataset.parquet'):
            print('Supervised dataset already exist, loading... src/supervised_dataset.parquet')
            all_df = pd.read_parquet('src/supervised_dataset.parquet')
        else:
            print('No supervised dataset found, extract supervised dataset from simulations')
            all_df = extract_supervised_dataset(run_dirs)
            all_df.to_parquet('src/supervised_dataset.parquet')
        
        feature_cols = [
            "residual_energy",
            "dist_to_sink",
            "total_alive",
            "avg_residual_energy",
            "round",
        ]
        X = all_df[feature_cols]
        y = all_df["is_CH_label"]
        train_lightgbm(X, y)

    elif mode == "cql":
        if os.path.exists('src/replay_buffer.pkl'):
            print('Replay buffer already exist, loading... src/replay_buffer.pkl')
            with open('src/replay_buffer.pkl', 'rb') as f:
                buf = pickle.load(f) # deserialize using load()
        else:
            print('No replay buffer found, building replay buffer from simulations')
            buf = build_replay_buffer(run_dirs)
            with open('src/replay_buffer.pkl', 'wb') as f:  # open a text file
                pickle.dump(buf, f) # serialize the list
        train_discrete_cql_policy(buf, epochs=epochs)

    else:
        raise ValueError("mode must be 'lgbm' or 'cql'")

    print(f"Finished in {(time.time()-t0):.1f} s")


if __name__ == "__main__":
    # python .\src\WSN_ML_Toolkit.py --sim_roots .\runs_LEACH_2\ .\runs_GWO_2\ .\runs_LEACH_GA_2\ --mode lgbm --sample_runs 150
    # python .\src\WSN_ML_Toolkit.py --sim_roots .\runs_LEACH_2\ .\runs_GWO_2\ .\runs_LEACH_GA_2\ --mode cql --sample_runs 150
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sim_roots",
        nargs="+",
        default=["sim_runs"],
        help="One or more folders containing run_* sub‑folders",
    )
    ap.add_argument("--mode", choices=["lgbm", "cql"], default="lgbm")
    ap.add_argument("--epochs", type=int, default=50, help="CQL epochs")
    ap.add_argument("--sample_runs", type=int, help="Randomly pick N run_* folders from all sim_roots")
    args = ap.parse_args()
    main(args.sim_roots, args.mode, args.epochs, args.sample_runs)
