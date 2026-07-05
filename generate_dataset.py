"""
Parameter sweep over the heat sink model.

Ranges required by the assessment:
    TDP:        30 W to 250 W
    Air velocity: 0.5 m/s to 15 m/s
    k_tim:      1 to 12 W/m.K

The sweep is a full factorial grid so the surrogate model sees the whole
operating space, including all cross-combinations of the three inputs.
"""

import os
import numpy as np
import pandas as pd

from heat_sink_model import compute_thermals

# grid resolution: 23 x 30 x 12 = 8280 design points
tdp_values = np.arange(30, 251, 10)
velocity_values = np.arange(0.5, 15.01, 0.5)
k_tim_values = np.arange(1, 13, 1)

rows = []
for tdp in tdp_values:
    for v in velocity_values:
        for k in k_tim_values:
            rows.append(compute_thermals(TDP=float(tdp), V_air=float(v), k_tim=float(k)))

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "heat_sink_dataset.csv")
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print("\nInput ranges covered:")
print(df[["TDP_W", "V_air_m_s", "k_tim_W_mK"]].describe().loc[["min", "max"]])
print("\nOutput ranges:")
print(df[["R_total_C_W", "T_junction_C"]].describe().loc[["min", "max", "mean"]])
print("\nReynolds number range (all laminar if max < 2300):")
print(df["Re"].min(), "to", df["Re"].max())
