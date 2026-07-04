# -*- coding: utf-8 -*-
"""
ML surrogate models for the heat sink dataset.

Targets:  R_total (degC/W) and T_junction (degC)
Features: TDP, air velocity, k_tim

Two models are compared, chosen deliberately as opposite ends of the
bias/variance spectrum:

1. Linear Regression - interpretable baseline. If the physics were linear
   in the inputs this would be enough. It is not (h ~ V^(1/3),
   R_tim ~ 1/k_tim, and T_j has a TDP x resistance interaction), so we
   expect it to underfit and that gap tells us how nonlinear the problem is.

2. Random Forest - nonparametric, captures nonlinearity and interactions
   without feature engineering. Expected to fit well inside the sweep
   ranges; known weakness is extrapolation outside them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["TDP_W", "V_air_m_s", "k_tim_W_mK"]
TARGETS = ["R_total_C_W", "T_junction_C"]

df = pd.read_csv(os.path.join("data", "heat_sink_dataset.csv"))
X = df[FEATURES]
y = df[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

os.makedirs("results", exist_ok=True)
metrics_rows = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test[target], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred[:, i]))
        r2 = r2_score(y_test[target], y_pred[:, i])
        metrics_rows.append(
            {"model": name, "target": target, "MAE": mae, "RMSE": rmse, "R2": r2}
        )

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv(os.path.join("results", "model_metrics.csv"), index=False)
print(metrics.to_string(index=False))

# predicted vs actual plots (test set)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for col, (name, y_pred) in enumerate(predictions.items()):
    for row, target in enumerate(TARGETS):
        ax = axes[row, col]
        ax.scatter(y_test[target], y_pred[:, row], s=6, alpha=0.4)
        lims = [y_test[target].min(), y_test[target].max()]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(f"{name} - {target}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
fig.tight_layout()
fig.savefig(os.path.join("results", "predicted_vs_actual.png"), dpi=150)
print("\nSaved results/model_metrics.csv and results/predicted_vs_actual.png")
