"""
Run the DAG 1 demo end-to-end: generate data, fit models, compare, plot, save figures.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

RANDOM_SEED = 42
N = 500
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")


# ── DGP Functions ─────────────────────────────────────────────────────────

def dgp_mp_linear(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    RS = rng.normal(0, 1, n)
    MN = rng.normal(0, 1, n)
    MP_surp = 0.3 * MN + rng.normal(0, 1, n)
    DR = 0.8 * MP_surp + 0.6 * RS + rng.normal(0, 1, n)
    ECF = -0.4 * MP_surp + 0.7 * MN + rng.normal(0, 1, n)
    R_eq = -0.6 * DR + 0.8 * ECF + 0.5 * RS + rng.normal(0, 1, n)
    return {"RS": RS, "MN": MN, "MP_surp": MP_surp, "DR": DR, "ECF": ECF, "R_eq": R_eq}


def dgp_mp_regime(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    RS = rng.normal(0, 1, n)
    MN = rng.normal(0, 1, n)
    MP_surp = 0.5 * MN + 1.5 * rng.normal(0, 1, n)
    tightening = MP_surp > 0
    DR = np.where(tightening, 1.8 * MP_surp, 0.3 * MP_surp) + 0.6 * RS + 0.7 * rng.normal(0, 1, n)
    ECF = np.where(tightening, -0.2 * MP_surp, -1.0 * MP_surp) + 0.7 * MN + 0.7 * rng.normal(0, 1, n)
    Z = rng.normal(0, 1, n)          # independent contagion shock (unobserved driver)
    stress = Z > 1.0                 # ~16% of events, uncorrelated with RS/MN/MP_surp
    R_eq = (
        -0.6 * DR + 0.8 * ECF + 0.5 * RS
        + np.where(stress, -5.0 * np.abs(MP_surp), 0.0)
        + 0.8 * rng.normal(0, 1, n)
    )
    return {"RS": RS, "MN": MN, "MP_surp": MP_surp, "DR": DR, "ECF": ECF, "R_eq": R_eq,
            "stress": stress.astype(float)}


def dgp_mp_nonlinear(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    RS = rng.normal(0, 1, n)
    MN = rng.normal(0, 1, n)
    MP_surp = 0.3 * MN + rng.normal(0, 1, n)
    DR = (
        1.0 * np.tanh(0.8 * MP_surp)
        + 0.7 * np.tanh(0.6 * RS)
        + 0.4 * MP_surp * RS
        + rng.normal(0, 1, n)
    )
    ECF = -0.6 * np.tanh(0.7 * MP_surp) + 0.8 * np.tanh(0.6 * MN) + rng.normal(0, 1, n)
    R_eq = -0.7 * DR + 0.9 * ECF + 0.6 * RS + rng.normal(0, 1, n)
    return {"RS": RS, "MN": MN, "MP_surp": MP_surp, "DR": DR, "ECF": ECF, "R_eq": R_eq}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"figure.figsize": (10, 5), "font.size": 12})
    az.style.use("arviz-darkgrid")

    # ── 1. Generate data (ground truth = regime-switching) ────────────────
    print("Generating data from regime-switching DGP...")
    data_true = dgp_mp_regime(n=N, seed=RANDOM_SEED)
    df = pd.DataFrame(data_true)

    mp_vals = df["MP_surp"].values
    rs_vals = df["RS"].values
    mn_vals = df["MN"].values
    dr_vals = df["DR"].values
    ecf_vals = df["ECF"].values
    req_vals = df["R_eq"].values
    tight = (mp_vals > 0).astype(float)
    tightening_bool = mp_vals > 0

    # ── 2. Figure 1: Exploratory scatter plots ────────────────────────────
    print("Plotting exploratory scatter...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, mask, color in [("Tightening", tightening_bool, "#d62728"), ("Easing", ~tightening_bool, "#1f77b4")]:
        axes[0].scatter(df["MP_surp"][mask], df["DR"][mask], alpha=0.15, s=10, label=label, color=color)
        axes[1].scatter(df["MP_surp"][mask], df["ECF"][mask], alpha=0.15, s=10, label=label, color=color)

    axes[0].set(xlabel="Monetary Policy Surprise", ylabel="Change in Discount Rate (ΔDR)", title="Discount-Rate Channel")
    axes[1].set(xlabel="Monetary Policy Surprise", ylabel="Change in Expected Cash Flows (ΔECF)", title="Cash-Flow Channel")
    for ax in axes:
        ax.legend(fontsize=10, framealpha=0.9)
        ax.axvline(0, color="grey", ls="--", alpha=0.4, lw=0.8)
    fig.suptitle("Asymmetric slopes: tightening hits harder than easing relieves", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_scatter_asymmetry.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig1_scatter_asymmetry.png saved")

    # ── 3. Build PyMC models ──────────────────────────────────────────────
    print("Building PyMC models...")

    # Model A: Linear
    with pm.Model() as model_linear:
        a1 = pm.Normal("a1", mu=0, sigma=2)
        a2 = pm.Normal("a2", mu=0, sigma=2)
        sigma_dr = pm.HalfNormal("sigma_dr", sigma=2)
        pm.Normal("DR", mu=a1 * mp_vals + a2 * rs_vals, sigma=sigma_dr, observed=dr_vals)

        b1 = pm.Normal("b1", mu=0, sigma=2)
        b2 = pm.Normal("b2", mu=0, sigma=2)
        sigma_ecf = pm.HalfNormal("sigma_ecf", sigma=2)
        pm.Normal("ECF", mu=b1 * mp_vals + b2 * mn_vals, sigma=sigma_ecf, observed=ecf_vals)

        c1 = pm.Normal("c1", mu=0, sigma=2)
        c2 = pm.Normal("c2", mu=0, sigma=2)
        c3 = pm.Normal("c3", mu=0, sigma=2)
        sigma_req = pm.HalfNormal("sigma_req", sigma=2)
        pm.Normal("R_eq", mu=c1 * dr_vals + c2 * ecf_vals + c3 * rs_vals, sigma=sigma_req, observed=req_vals)

    # Model B: Regime-switching
    with pm.Model() as model_regime:
        a1_t = pm.Normal("a1_tight", mu=0, sigma=2)
        a1_e = pm.Normal("a1_ease", mu=0, sigma=2)
        a2 = pm.Normal("a2", mu=0, sigma=2)
        sigma_dr = pm.HalfNormal("sigma_dr", sigma=2)
        dr_mu = (a1_t * tight + a1_e * (1 - tight)) * mp_vals + a2 * rs_vals
        pm.Normal("DR", mu=dr_mu, sigma=sigma_dr, observed=dr_vals)

        b1_t = pm.Normal("b1_tight", mu=0, sigma=2)
        b1_e = pm.Normal("b1_ease", mu=0, sigma=2)
        b2 = pm.Normal("b2", mu=0, sigma=2)
        sigma_ecf = pm.HalfNormal("sigma_ecf", sigma=2)
        ecf_mu = (b1_t * tight + b1_e * (1 - tight)) * mp_vals + b2 * mn_vals
        pm.Normal("ECF", mu=ecf_mu, sigma=sigma_ecf, observed=ecf_vals)

        c1 = pm.Normal("c1", mu=0, sigma=2)
        c2 = pm.Normal("c2", mu=0, sigma=2)
        c3 = pm.Normal("c3", mu=0, sigma=2)
        c_stress = pm.Normal("c_stress", mu=0, sigma=5)
        p_stress = pm.Beta("p_stress", alpha=2, beta=10)
        sigma_req = pm.HalfNormal("sigma_req", sigma=2)
        base_mu = c1 * dr_vals + c2 * ecf_vals + c3 * rs_vals
        pm.Mixture(
            "R_eq",
            w=pt.stack([1 - p_stress, p_stress]),
            comp_dists=[
                pm.Normal.dist(mu=base_mu, sigma=sigma_req),
                pm.Normal.dist(mu=base_mu + c_stress * np.abs(mp_vals), sigma=sigma_req),
            ],
            observed=req_vals,
        )

    # Model C: Nonlinear
    with pm.Model() as model_nonlinear:
        a1 = pm.Normal("a1", mu=0, sigma=2)
        a2 = pm.Normal("a2", mu=0, sigma=2)
        a3 = pm.Normal("a3", mu=0, sigma=2)
        a4 = pm.Normal("a4", mu=0, sigma=1)  # quadratic term
        a_s1 = pm.HalfNormal("a_scale1", sigma=2)
        a_s2 = pm.HalfNormal("a_scale2", sigma=2)
        sigma_dr = pm.HalfNormal("sigma_dr", sigma=2)
        dr_mu = (a1 * pm.math.tanh(a_s1 * mp_vals) + a2 * pm.math.tanh(a_s2 * rs_vals) + 
                 a3 * mp_vals * rs_vals + a4 * mp_vals**2)
        pm.Normal("DR", mu=dr_mu, sigma=sigma_dr, observed=dr_vals)

        b1 = pm.Normal("b1", mu=0, sigma=2)
        b2 = pm.Normal("b2", mu=0, sigma=2)
        b3 = pm.Normal("b3", mu=0, sigma=1)  # interaction
        b4 = pm.Normal("b4", mu=0, sigma=1)  # quadratic
        b_s1 = pm.HalfNormal("b_scale1", sigma=2)
        b_s2 = pm.HalfNormal("b_scale2", sigma=2)
        sigma_ecf = pm.HalfNormal("sigma_ecf", sigma=2)
        ecf_mu = (b1 * pm.math.tanh(b_s1 * mp_vals) + b2 * pm.math.tanh(b_s2 * mn_vals) +
                  b3 * mp_vals * mn_vals + b4 * mp_vals**2)
        pm.Normal("ECF", mu=ecf_mu, sigma=sigma_ecf, observed=ecf_vals)

        c1 = pm.Normal("c1", mu=0, sigma=2)
        c2 = pm.Normal("c2", mu=0, sigma=2)
        c3 = pm.Normal("c3", mu=0, sigma=2)
        c4 = pm.Normal("c4", mu=0, sigma=1)  # interaction
        sigma_req = pm.HalfNormal("sigma_req", sigma=2)
        pm.Normal("R_eq", mu=c1 * dr_vals + c2 * ecf_vals + c3 * rs_vals + c4 * dr_vals * ecf_vals, 
                  sigma=sigma_req, observed=req_vals)

    # ── 4. Sample all models (cores=1 for Windows compatibility) ──────────
    models = {"Linear": model_linear, "Regime": model_regime, "Nonlinear": model_nonlinear}
    traces = {}

    for name, model in models.items():
        print(f"Sampling {name}...")
        with model:
            traces[name] = pm.sample(
                draws=500, tune=200, chains=2, cores=1,
                random_seed=RANDOM_SEED, target_accept=0.95,
            )
            pm.compute_log_likelihood(traces[name])
        print(f"  -> {name} done.")

    # ── 5. Figure 2: LOO comparison (on R_eq, the target variable) ──────
    print("Computing LOO comparison...")
    comparison = az.compare(traces, ic="loo", var_name="R_eq")
    print(comparison.to_string())
    comparison.to_csv(os.path.join(FIG_DIR, "loo_comparison.csv"))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    az.plot_compare(comparison, ax=ax, insample_dev=False)
    ax.set_title("LOO Model Comparison: Same DAG, Three Functional Forms", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_loo_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig2_loo_comparison.png saved")

    # ── 6. Figure 3: Posterior predictive checks ─────────────────────────
    print("Generating posterior predictive checks...")
    ppc_data = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for i, (name, model) in enumerate(models.items()):
        with model:
            ppc = pm.sample_posterior_predictive(traces[name], random_seed=RANDOM_SEED)
        ppc_data[name] = ppc
        az.plot_ppc(ppc, var_names=["R_eq"], ax=axes[i], num_pp_samples=200, kind="cumulative")
        axes[i].set_title(f"{name}", fontsize=12)
    fig.suptitle("Posterior Predictive Checks: Equity Returns (R_eq)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ppc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig3_ppc.png saved")

    # ── 7. Figure 4: Trading strategy P&L ────────────────────────────────
    print("Computing trading strategy P&L...")

    def compute_strategy_pnl(ppc_obj, actual_req, pnl_scale=1.0):
        pred = ppc_obj.posterior_predictive["R_eq"].mean(dim=["chain", "draw"]).values
        signal = np.sign(pred)
        size = np.clip(np.abs(pred) / np.std(pred), 0, 2.0)
        pnl = signal * size * actual_req * pnl_scale
        return np.cumsum(pnl), pred

    pnl_scale = 0.02 / np.std(req_vals)   # normalise to 2% std dev per event
    results = {}
    for name in models:
        results[name] = compute_strategy_pnl(ppc_data[name], req_vals, pnl_scale=pnl_scale)

    stress_mask = df["stress"].values.astype(bool)
    n_stress = int(stress_mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_map = {"Linear": "#1f77b4", "Regime": "#2ca02c", "Nonlinear": "#ff7f0e"}
    for name, (cum_pnl, _) in results.items():
        axes[0].plot(cum_pnl, label=name, alpha=0.9, color=colors_map[name], lw=1.5)
        # Add label at the end of each line
        axes[0].text(len(cum_pnl)-1, cum_pnl[-1], f'{cum_pnl[-1]:.1f}%', 
                     fontsize=9, ha='left', va='center', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    axes[0].set(xlabel="Event index", ylabel="Cumulative Return (%)", title="Cumulative Return by Model")
    axes[0].legend(fontsize=11)
    axes[0].axhline(0, color="black", ls="--", alpha=0.3)

    stress_pnl = {}
    for name, (_, pred) in results.items():
        signal = np.sign(pred)
        size = np.clip(np.abs(pred) / np.std(pred), 0, 2.0)
        pnl_stress = (signal * size * req_vals * pnl_scale)[stress_mask].sum()
        stress_pnl[name] = float(pnl_stress)
        bar = axes[1].bar(name, pnl_stress, color=colors_map[name], alpha=0.85, edgecolor="black", lw=0.5)
        # Add label on top of each bar
        axes[1].text(bar[0].get_x() + bar[0].get_width()/2, pnl_stress, 
                     f'{pnl_stress:.1f}%', ha='center', 
                     va='bottom' if pnl_stress > 0 else 'top', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    axes[1].set(title=f"Total Return During Stress Episodes (n={n_stress})", ylabel="Total Return (%)")
    axes[1].axhline(0, color="black", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_pnl.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig4_pnl.png saved")

    # ── 8. Figure 5: Posterior parameter recovery (regime model) ──────────
    print("Plotting posterior parameter recovery for regime model...")
    true_params = {
        "a1_tight": 1.8, "a1_ease": 0.3, "a2": 0.6,
        "b1_tight": -0.2, "b1_ease": -1.0, "b2": 0.7,
        "c1": -0.6, "c2": 0.8, "c3": 0.5, "c_stress": -5.0,
        "p_stress": 0.159,
    }
    param_names = list(true_params.keys())
    posterior = traces["Regime"].posterior

    fig, axes = plt.subplots(2, 6, figsize=(18, 5.5))
    axes = axes.flatten()
    for ax in axes[len(param_names):]:
        ax.set_visible(False)
    for i, pname in enumerate(param_names):
        samples = posterior[pname].values.flatten()
        axes[i].hist(samples, bins=50, density=True, alpha=0.7, color="#2ca02c", edgecolor="none")
        axes[i].axvline(true_params[pname], color="red", ls="--", lw=2, label=f"True = {true_params[pname]}")
        axes[i].axvline(np.mean(samples), color="black", ls="-", lw=1.5, label=f"Post. mean = {np.mean(samples):.2f}")
        axes[i].set_title(pname, fontsize=10)
        axes[i].legend(fontsize=7, loc="upper right")
        axes[i].tick_params(labelsize=8)
    fig.suptitle("Regime Model: Posterior Parameter Recovery vs Ground Truth", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_param_recovery.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig5_param_recovery.png saved")

    # ── 9. Save summary results as JSON ──────────────────────────────────
    summary_results = {
        "loo_ranking": comparison.index.tolist(),
        "loo_elpd": {k: float(v) for k, v in comparison["elpd_loo"].items()},
        "loo_se": {k: float(v) for k, v in comparison["se"].items()},
        "stress_pnl": stress_pnl,
        "final_cum_pnl": {name: float(cum_pnl[-1]) for name, (cum_pnl, _) in results.items()},
        "n_obs": N,
        "n_stress": n_stress,
    }
    with open(os.path.join(FIG_DIR, "results.json"), "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n=== RESULTS SUMMARY ===")
    print(f"LOO ranking: {summary_results['loo_ranking']}")
    print(f"LOO ELPD: {summary_results['loo_elpd']}")
    print(f"Final cumulative P&L: {summary_results['final_cum_pnl']}")
    print(f"Stress P&L: {summary_results['stress_pnl']}")
    print(f"N stress events: {n_stress}/{N}")
    print("\nAll figures saved to:", FIG_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
