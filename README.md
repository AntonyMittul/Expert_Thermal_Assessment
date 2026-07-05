# Expert Thermal / XThermal — AI/ML Assessment

Heat sink thermal model parameter sweep + ML surrogate models, based on the
calculation script provided by Expert Thermal.

**🌡️ Live interactive demo:** https://expertthermalassessment-kudzreqkcmxp85bietq7vk.streamlit.app/

Move the TDP / airflow / TIM sliders and compare the physics model against
the ML surrogate side by side — no installation needed.

## Repository structure

| File | Purpose |
|---|---|
| `heat_sink_model.py` | Base thermal model refactored into a parameterized function (physics unchanged) |
| `generate_dataset.py` | Full-factorial parameter sweep → `data/heat_sink_dataset.csv` (8,280 points) |
| `train_surrogate.py` | Trains and evaluates two surrogate models (Linear Regression, Random Forest) |
| `sensitivity_analysis.py` | Pearson/Spearman correlations + permutation importance + trend plots |
| `app.py` | Interactive Streamlit demo — live physics vs. surrogate comparison |
| `REPORT.md` | Summary report: assumptions, results, model comparison, limitations |

Written responses for Tasks 2–4 are submitted separately via email.

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
python generate_dataset.py       # creates data/heat_sink_dataset.csv
python train_surrogate.py        # creates results/model_metrics.csv + plots
python sensitivity_analysis.py   # creates correlation/importance CSVs + plots
streamlit run app.py             # interactive demo (http://localhost:8501)
```

## Key results at a glance

- Sanity check: default case (TDP=150 W, V=1 m/s, k_tim=4) reproduces the
  original script: R_total = 0.369 °C/W, T_j = 80.38 °C.
- Random Forest surrogate: R² = 0.9967 (R_total), 0.9998 (T_junction).
- Junction temperature is driven overwhelmingly by TDP; thermal resistance
  is driven by air velocity, with diminishing returns above ~5 m/s.

See `REPORT.md` for the full write-up.
