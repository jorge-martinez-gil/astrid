"""Paper 2 (E5): dimension ablation / attribution of detection.
Uses the per-dimension score contributions recorded in the Balanced detection
grid. (1) Attribution matrix: score-points lost in each dimension per fault.
(2) Leave-one-dimension-out: recompute the composite with each dimension removed
(weights renormalized) and measure how much of the fault-induced score drop
survives -- proving each fault is driven by its aligned dimension and that no
single dimension dominates across faults. No new audits, no model trained.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DIMS = ["quality", "security", "reliability", "robustness", "fairness"]
W = {"quality": .35, "security": .25, "reliability": .20, "robustness": .10, "fairness": .10}
ALIGN = {"missingness": "quality", "duplicates": "quality", "split_leakage": "quality",
         "drift": "reliability", "pii": "security", "fairness": "fairness"}
FAULTS = ["missingness", "duplicates", "split_leakage", "drift", "pii", "fairness"]

def comp_score(contribs, drop_dim=None):
    if drop_dim is None:
        return sum(contribs[d] for d in DIMS)
    keep = [d for d in DIMS if d != drop_dim]
    renorm = sum(W[d] for d in keep)
    # contrib_d = 100*w_d*s_d  ->  s_d = contrib_d/(100*w_d); new score = 100*sum(w_d/renorm * s_d)
    return sum((W[d] / renorm) * (contribs[d] / W[d]) for d in keep)

def main():
    out = ROOT/"experiments"/"results"/"p2_audit"
    df = pd.read_csv(out/"detection_grid_balanced.csv")
    g = df.groupby(["fault", "severity"])
    # ---- attribution matrix at high severity (0.4) ----
    attr = {}
    for f in FAULTS:
        hi = df[(df.fault == f) & (df.severity == 0.40)]
        attr[f] = {d: float(hi["drop_"+d].mean()) for d in DIMS}
    A = pd.DataFrame(attr).T[DIMS]
    A.to_csv(out/"ablation_attribution.csv")
    print("== Attribution: score-points lost per dimension (alpha=0.4) ==")
    print(A.round(1).to_string())
    # ---- leave-one-dimension-out composite drop ----
    recs = []
    for f in FAULTS:
        clean = df[(df.fault == f) & (df.severity == 0.0)]
        hi = df[(df.fault == f) & (df.severity == 0.40)]
        cc = {d: clean["c_"+d].mean() for d in DIMS}
        ch = {d: hi["c_"+d].mean() for d in DIMS}
        full = comp_score(cc) - comp_score(ch)
        row = {"fault": f, "aligned": ALIGN[f], "full_drop": full}
        for d in DIMS:
            drop_wo = comp_score(cc, d) - comp_score(ch, d)
            row["retain_wo_"+d] = (drop_wo/full) if full else 0.0
        recs.append(row)
    L = pd.DataFrame(recs)
    L.to_csv(out/"ablation_leaveoneout.csv", index=False)
    print("\n== Fraction of composite score-drop retained when a dimension is removed ==")
    show = L[["fault", "aligned", "full_drop"] + ["retain_wo_"+d for d in DIMS]]
    print(show.round(2).to_string(index=False))
    # summary: removing aligned dim vs avg of others
    al = [L.loc[i, "retain_wo_"+L.loc[i, "aligned"]] for i in L.index]
    others = [np.mean([L.loc[i, "retain_wo_"+d] for d in DIMS if d != L.loc[i, "aligned"]]) for i in L.index]
    print(f"\nMean drop retained: remove ALIGNED dim = {np.mean(al):.2f}; remove a NON-aligned dim = {np.mean(others):.2f}")

if __name__ == "__main__":
    raise SystemExit(main())
