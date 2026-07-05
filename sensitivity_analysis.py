"""
Correlation and sensitivity analysis for the heat sink dataset.

Three views, because they answer slightly different questions:
- Pearson correlation: linear association between each input and output
- Spearman correlation: monotonic association (fairer to nonlinear physics)
- Permutation importance on a Random Forest: how much test error grows
  when one input is shuffled, i.e. true predictive contribution including
  interactions
"""

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

FEATURES = ["TDP_W", "V_air_m_s", "k_tim_W_mK"]
TARGETS = ["R_total_C_W", "T_junction_C"]

df = pd.read_csv(os.path.join("data", "heat_sink_dataset.csv"))
os.makedirs("results", exist_ok=True)

# ---- correlations ----
pearson = df[FEATURES + TARGETS].corr(method="pearson").loc[FEATURES, TARGETS]
spearman = df[FEATURES + TARGETS].corr(method="spearman").loc[FEATURES, TARGETS]

print("Pearson correlation (inputs vs outputs):")
print(pearson.round(3).to_string())
print("\nSpearman correlation (inputs vs outputs):")
print(spearman.round(3).to_string())

pearson.to_csv(os.path.join("results", "correlation_pearson.csv"))
spearman.to_csv(os.path.join("results", "correlation_spearman.csv"))

# ---- permutation importance per target ----
importance_rows = []
for target in TARGETS:
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[target], test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    for feature, mean_imp in zip(FEATURES, perm.importances_mean):
        importance_rows.append(
            {"target": target, "feature": feature, "importance": mean_imp}
        )

importances = pd.DataFrame(importance_rows)
importances.to_csv(os.path.join("results", "permutation_importance.csv"), index=False)
print("\nPermutation importance (Random Forest):")
print(importances.round(4).to_string(index=False))

# ---- plots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, target in zip(axes, TARGETS):
    sub = importances[importances["target"] == target]
    ax.bar(sub["feature"], sub["importance"], color="steelblue")
    ax.set_title(f"Permutation importance: {target}")
    ax.set_ylabel("Importance (R2 drop when shuffled)")
fig.tight_layout()
fig.savefig(os.path.join("results", "permutation_importance.png"), dpi=150)

# physical trend plots that make the sensitivity story visible
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sub = df[(df["k_tim_W_mK"] == 4) & (df["TDP_W"] == 150)]
axes[0].plot(sub["V_air_m_s"], sub["T_junction_C"], "o-")
axes[0].set_xlabel("Air velocity (m/s)")
axes[0].set_ylabel("T_junction (degC)")
axes[0].set_title("T_j vs air velocity (TDP=150 W, k_tim=4)")

sub = df[(df["k_tim_W_mK"] == 4) & (df["V_air_m_s"] == 1.0)]
axes[1].plot(sub["TDP_W"], sub["T_junction_C"], "o-")
axes[1].set_xlabel("TDP (W)")
axes[1].set_ylabel("T_junction (degC)")
axes[1].set_title("T_j vs TDP (V=1 m/s, k_tim=4)")

sub = df[(df["TDP_W"] == 150) & (df["V_air_m_s"] == 1.0)]
axes[2].plot(sub["k_tim_W_mK"], sub["R_total_C_W"], "o-")
axes[2].set_xlabel("k_tim (W/m.K)")
axes[2].set_ylabel("R_total (degC/W)")
axes[2].set_title("R_total vs k_tim (TDP=150 W, V=1 m/s)")

fig.tight_layout()
fig.savefig(os.path.join("results", "sensitivity_trends.png"), dpi=150)
print("\nSaved results/permutation_importance.png and results/sensitivity_trends.png")
