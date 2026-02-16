# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Congratulations, You Have a DAG. Now What?" — A PyMC-based causal inference demonstration for the ADIA Lab Causal Discovery Challenge. The project proves that discovering a causal DAG is insufficient for profitable decisions; the *functional form* of causal relationships matters enormously. Three competing Bayesian models (linear, regime-switching, nonlinear) are fit to synthetic data generated from a Bernanke-Kuttner monetary policy causal framework.

## Commands

Dependencies are managed with **pixi** (conda-based). Python 3.11 to <3.13.

```bash
pixi run run-demo    # Execute the full pipeline (run_demo_dag1.py)
pixi run lab         # Launch JupyterLab for interactive work
```

On Windows PowerShell: `./run_demo.ps1`

**Pre-commit hooks** (ruff, mypy, nbstripout, YAML/TOML validators):
```bash
pre-commit run --all-files
```

## Architecture

### Causal Structure (Bernanke-Kuttner)

```
Macro news (MN) ──────────────────────────┐
     │                                     │
     ▼                                     ▼
MP_surprise ──► Discount rates (DR) ──► Equity return (R_eq)
     │                                     ▲
     └──────► Expected cash flows (ECF) ───┘
                                           ▲
Risk sentiment (RS) ──────────────────────┘
```

### Pipeline (`run_demo_dag1.py`)

1. Generate 500 synthetic observations from the regime-switching DGP (ground truth)
2. Fit three competing PyMC models (linear, regime, nonlinear) via MCMC
3. Compare models using LOO (Expected Log Predictive Density)
4. Run posterior predictive checks and trading strategy P&L simulations
5. Output 5 PNG figures + `loo_comparison.csv` + `results.json` to `figures/`

### Three DGP Functions

- `dgp_mp_linear()` — symmetric linear relationships
- `dgp_mp_regime()` — regime-switching with stress amplification (ground truth)
- `dgp_mp_nonlinear()` — smooth tanh-based saturation

### Key Entry Points

- `run_demo_dag1.py` — main batch pipeline (all figures and results)
- `demo_dag1.ipynb` — interactive Jupyter notebook version
- `src/congrats_dag/` — package module
- `figures/` — generated outputs (PNGs, CSV, JSON)
