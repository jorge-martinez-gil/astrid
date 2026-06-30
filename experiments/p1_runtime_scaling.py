import warnings, time, sys, json; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from astrid_core import analyze_tabular_dataframe
import experiments.fault_injection_tabular as FI
ROOT = Path(".")
rows = []
rng = np.random.default_rng(0)
# scaling vs n_rows (fixed cols) and vs n_cols (fixed rows)
for n in [500, 1000, 2000, 4000, 8000, 16000]:
    df = FI.make_clean_dataset(n_rows=n, seed=7)
    ts = []
    for _ in range(3):
        t = time.time(); analyze_tabular_dataframe(df, dataset_name="r", preset="Balanced (recommended)", mode="Quick Scan"); ts.append(time.time()-t)
    rows.append(dict(axis="rows", size=n, ncols=df.shape[1], runtime_s=float(np.median(ts))))
    print("rows", n, round(np.median(ts),3))
base = FI.make_clean_dataset(n_rows=4000, seed=7)
import numpy as np
for k in [5, 10, 20, 40, 80]:
    extra = pd.DataFrame(rng.normal(size=(len(base), k)), columns=[f"x{i}" for i in range(k)])
    df = pd.concat([base.reset_index(drop=True), extra], axis=1)
    ts = []
    for _ in range(3):
        t = time.time(); analyze_tabular_dataframe(df, dataset_name="r", preset="Balanced (recommended)", mode="Quick Scan"); ts.append(time.time()-t)
    rows.append(dict(axis="cols", size=df.shape[1], ncols=df.shape[1], runtime_s=float(np.median(ts))))
    print("cols", df.shape[1], round(np.median(ts),3))
pd.DataFrame(rows).to_csv("experiments/results/p1_det/runtime_scaling.csv", index=False)
print("SCALING DONE")
