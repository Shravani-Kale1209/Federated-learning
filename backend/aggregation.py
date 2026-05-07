"""
Layer-wise federated aggregation: weighted FedAvg plus robust pipelines
(Multi-Krum selection + coordinate-wise trimmed mean on weight deltas).

Pure NumPy; works with pickled Keras weight lists (aligned layer shapes).
"""

from __future__ import annotations

import math
from typing import Literal, Sequence, Tuple

import numpy as np

AggregateMode = Literal["weighted_fedavg", "krum_trimmed_mean", "trimmed_mean"]


def subtract_weight_lists(
    a: Sequence[np.ndarray], b: Sequence[np.ndarray]
) -> list[np.ndarray]:
    return [np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64) for x, y in zip(a, b)]


def add_weight_lists(
    a: Sequence[np.ndarray], delta: Sequence[np.ndarray]
) -> list[np.ndarray]:
    out = []
    for wa, wd in zip(a, delta):
        wa64 = np.asarray(wa, dtype=np.float64)
        out.append((wa64 + np.asarray(wd, dtype=np.float64)).astype(wa.dtype, copy=False))
    return out


def flatten_weight_list(weights: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(w).ravel() for w in weights])


def shapes_from_weights(weights: Sequence[np.ndarray]) -> list[Tuple[int, ...]]:
    return [tuple(np.asarray(w).shape) for w in weights]


def unflatten_to_weight_list(vec: np.ndarray, shapes: list[Tuple[int, ...]]) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    offset = 0
    total = vec.size
    for shp in shapes:
        n_el = math.prod(shp)
        chunk = vec[offset : offset + n_el].reshape(shp)
        layers.append(chunk)
        offset += n_el
    if offset != total:
        raise ValueError("unflatten: shape/size mismatch")
    return layers


# ALGORITHM: Federated Averaging (FedAvg)
# DESCRIPTION: A standard federated learning algorithm that computes a weighted average of the local models based on their dataset sizes.
def weighted_fedavg(
    updates: Sequence[Tuple[Sequence[np.ndarray], float]],
) -> list[np.ndarray]:
    """
    Convex combination sum_k (n_k / sum n_j) * W^(k)_layer.
    """
    if not updates:
        raise ValueError("weighted_fedavg: empty updates")

    n_layers = len(updates[0][0])
    total_n = float(sum(n for _, n in updates))
    if total_n <= 0:
        raise ValueError("weighted_fedavg: total sample count must be positive")

    new_global_weights: list[np.ndarray] = []
    for layer_idx in range(n_layers):
        tensors = [w[layer_idx] for (w, _) in updates]
        out_dtype = np.asarray(tensors[0]).dtype
        accum = np.zeros_like(np.asarray(tensors[0]), dtype=np.float64)
        for (_, ns), t in zip(updates, tensors):
            accum += np.asarray(t, dtype=np.float64) * (float(ns) / total_n)
        new_global_weights.append(accum.astype(out_dtype, copy=False))
    return new_global_weights


def _pairwise_sq_distances(points: np.ndarray) -> np.ndarray:
    """(n,n) Euclidean distances squared."""
    # (n,n) norms |x_i - x_j|^2 = |xi|^2 + |xj|^2 - 2 xi.xj
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("points expected (n_clients, dim)")
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = sq + sq.T - 2 * (x @ x.T)
    d = np.maximum(d, 0.0)
    return d


# ALGORITHM: Multi-Krum (Byzantine-Robust Aggregation)
# DESCRIPTION: An algorithm that selects a subset of local models (multi-k) that have the smallest sum of squared distances to their closest neighbors, effectively filtering out malicious or poisoned updates.
def multi_krum_indices(
    flat_updates: np.ndarray,
    *,
    multi_k: int,
    neighbor_m: int | None,
) -> np.ndarray:
    """
    Return indices (length min(multi_k, n)) for clients selected by ascending Krum score.
    When n_clients < 4, callers should skip Krum — this still returns deterministic indices.
    """
    n = flat_updates.shape[0]
    if n <= 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    d = np.sqrt(_pairwise_sq_distances(flat_updates))
    np.fill_diagonal(d, np.inf)
    if neighbor_m is None:
        neighbor_m = max(1, n - 3)
    neighbor_m = min(neighbor_m, max(1, n - 1))

    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = np.sort(d[i])
        scores[i] = float(np.sum(row[:neighbor_m]))
    ranked = np.argsort(scores)
    k_pick = max(1, min(multi_k, n))
    return ranked[:k_pick]


# ALGORITHM: Coordinate-wise Trimmed Mean
# DESCRIPTION: A robust aggregation technique that sorts the parameter values for each coordinate, drops a specified fraction of the highest and lowest values (outliers), and computes the mean of the remaining values.
def coordinate_trimmed_mean(
    points: np.ndarray,
    beta_per_tail: float,
) -> np.ndarray:
    """
    Row-wise trimmed mean coordinate-wise across clients.
    points shape (num_clients, dim).
    Drops floor(beta * n_points) smallest and largest samples per coordinate
    ; if insufficient clients, averages all coordinates.
    """
    x = np.asarray(points, dtype=np.float64)
    n_clients, dim = x.shape
    if n_clients == 1:
        return x[0].copy()

    trimmed = beta_per_tail
    if trimmed <= 0 or trimmed >= 0.49:
        return np.mean(x, axis=0)

    trim = int(math.floor(n_clients * trimmed))
    trim = max(0, trim)
    if trim * 2 >= n_clients:
        return np.median(x, axis=0)

    sorted_x = np.sort(x, axis=0)
    trimmed_x = sorted_x[trim : n_clients - trim, :]
    
    if trimmed_x.shape[0] > 0:
        return np.mean(trimmed_x, axis=0)
    else:
        return np.median(x, axis=0)


def aggregate_updates(
    global_weights: Sequence[np.ndarray],
    clients: Sequence[Tuple[Sequence[np.ndarray], float]],
    mode: AggregateMode,
    *,
    min_clients_for_krum: int,
    krum_multi_k: int,
    krum_neighbor_m: int | None,
    trim_beta_per_tail: float,
) -> tuple[list[np.ndarray], dict]:
    """
    Returns (new_weight_list, audit_dict).

    Robust modes aggregate **deltas** (local − global): trimmed mean yields an aggregate
    delta applied back to ``global_weights`` (FedAvg-style equivalence on benign data).
    """
    if not clients:
        raise ValueError("aggregate_updates: empty clients")

    audit: dict = {"mode": mode, "clients_in": len(clients)}

    if mode == "weighted_fedavg":
        nw = weighted_fedavg([(w, n) for w, n in clients])
        audit["krum_selected"] = list(range(len(clients)))
        audit["clients_dropped"] = []
        return nw, audit

    shapes = shapes_from_weights(global_weights)
    flat_deltas: list[np.ndarray] = []
    for w, _n in clients:
        delta_tensors = subtract_weight_lists(w, global_weights)
        flat_deltas.append(flatten_weight_list(delta_tensors))
    stacked = np.stack(flat_deltas, axis=0)
    n_clients = stacked.shape[0]

    if mode == "krum_trimmed_mean" and n_clients >= min_clients_for_krum:
        idx = multi_krum_indices(stacked, multi_k=krum_multi_k, neighbor_m=krum_neighbor_m)
        stacked_used = stacked[idx]
        sel = set(idx.tolist())
        audit["krum_selected"] = [int(i) for i in idx]
        audit["neighbor_m_used"] = krum_neighbor_m
        audit["clients_dropped"] = [int(i) for i in range(n_clients) if i not in sel]
    else:
        stacked_used = stacked
        audit["krum_selected"] = list(range(n_clients))
        audit["clients_dropped"] = []
        if mode == "krum_trimmed_mean" and n_clients < min_clients_for_krum:
            audit["krum_fallback"] = f"n_clients<{min_clients_for_krum}"

    if mode in ("trimmed_mean", "krum_trimmed_mean"):
        delta_flat = coordinate_trimmed_mean(stacked_used, trim_beta_per_tail)
    else:
        raise ValueError(f"unknown mode {mode}")

    audit["trim_beta_used"] = trim_beta_per_tail
    agg_delta_vectors = unflatten_to_weight_list(delta_flat, shapes)
    new_w = add_weight_lists(global_weights, agg_delta_vectors)
    return new_w, audit
