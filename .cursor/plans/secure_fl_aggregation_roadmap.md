# Secure federated aggregation roadmap (updated)

**Status:** Sample-weighted FedAvg is implemented in `backend/admin_server.py` and `backend/hospital_node.py`.

## Baseline aggregation (included)

### Sample-weighted FedAvg (`n`-weighted)

Hospitals may train on unequal amounts of local data. The server must **not** use a plain `mean` of weight tensors unless all clients intentionally equal.

- Each client \(k\) sends trained weights \(W^{(k)}\) and **`num_samples`** \(n_k\) (training samples used this round).
- Aggregation per layer \(l\):

\[
W_{\text{global}}^{(l)} = \sum_k \frac{n_k}{\sum_j n_j} \, W_{(k)}^{(l)}
\]

- **Backward compatibility:** payloads that are a raw pickled list (weights only) are treated as **`num_samples = 1`**, restoring equal weight per client (matches old behavior when all omit counts).
- Later robust phases (Multi-Krum, trimmed mean, RECESS) apply **on top of or instead of** this baseline; adversarial contexts may **cap or ignore** claimed \(n_k\).

## Subsequent phases

See prior roadmap: modular `backend/aggregation.py`, Multi-Krum, EE trimmed mean (+ trust weights), FedProx on clients, RECESS-style verification, dashboard observability.
