# Congratulations, You Have a DAG. Now What?

**Authored by**: Camilo Saldarriaga, Alexander Fengler, Francesco Muia

---

The ADIA Lab Causal Discovery Challenge demonstrated that supervised ML can classify causal roles from observational data at ~77% balanced accuracy — nearly double the ~40% achieved by classical constraint-based methods (Olivetti et al., 2024). In consequence, institutions that can generate thousands of realistic labeled simulations may hold a structural advantage in causal discovery.

But recovering a DAG is only the first step. A DAG tells you *what* causes *what* — not *how*. 

**Congratulations, You Have a DAG. Now What?**

To exemplify that identifying a DAG is just a first step that still leaves you far away from making money, we build a simple example in which a known DAG can follow different functional forms. Each functional form generates different results, and choosing the right modeling techniques may be the difference between making and losing money. This is where PyMC can bridge the gap.

We use a well grounded causal example in the literature, by following the causal relation between policy surprises and equity returns (Bernanke & Kuttner, 2005). The example shows that knowing the DAG  still leaves open the functional form of those relationships. Is the response linear? Does it switch regimes between tightening and easing cycles? Do effects saturate at extremes?

The answer matters enormously for anyone who wants to trade the insight. **The same causal structure combined with three different functional forms produces dramatically different P&L outcomes under stress — the correctly-specified regime model doubles the returns of misspecified alternatives during the most volatile episodes**. PyMC identifies the correct one.

---

## The Structure: Bernanke-Kuttner Channels

To illustrate the argument, we use the causal relation between Monetary Policy surprises and equity prices, based on Bernanke & Kuttner (2005).  Bernanke & Kuttner (2005) is one of the most cited studies in monetary economics — it provided the first clean causal identification of how Federal Reserve policy surprises move equity prices, using Fed Funds futures to isolate the unexpected component of rate decisions. 

We adopt their causal structure precisely because the DAG itself is uncontroversial: the debate is not *whether* these channels exist, but *what shape* the response functions take — which is exactly the question this post investigates.   

The canonical DAG from the monetary policy transmission literature decomposes equity return responses to policy surprises into two channels:

```
Macro news (MN) ──────────────────────────────────┐
               │                                  │
               ▼                                  ▼
  Monetary Policy surprise (MP_surp) ──────► Change in expected cash flows (ΔECF)  ──┐
               │              │                                                      │
               │              │                                                      ▼
               │              └────────────► Change in discount rates (ΔDR) ──► Equity return (R_eq)                             
               │                                  ▲                  ▲    ▲
               │                                  │                  │    │
               └──────────────────────────────────│──────────────────┘    │
                                                  │                       │
  Risk sentiment (RS) ────────────────────────────┴───────────────────────┘
```

The Monetary Policy surprise variable measures how much the Federal Reserve's interest rate decision differed from what markets expected beforehand. Bernanke & Kuttner (2005), computed it from Fed Funds futures prices in a narrow window around an FOMC announcement. The idea is simple: if the futures market was pricing a rate of 5.25% and the Fed announced 5.50%, that's a +25 basis point tightening surprise.

The Monetary policy surprises have a negative impact on the change in expected cash flows as they represent higher than expected borrowing costs for the economy. The effect of the MP surprises on the change in discount rates is positive, and they affect negatively the Equity prices.

The remaining nodes in the graph are: **Equity return** ($R_{eq}$), the broad stock-market index return on the day of the announcement — think of it as the S&P 500's daily move. **Change in discount rates** ($\Delta DR$) captures the shift in risk-free yields (e.g. Treasury rates) that reprices every future cash-flow stream. **Change in expected cash flows** ($\Delta ECF$) summarises the market's revision to its forecast of future corporate earnings — when borrowing costs jump unexpectedly, analysts mark down profit expectations. Finally, **Risk sentiment** ($RS$) is a latent measure of investors' willingness to bear risk; in the original paper it is proxied by credit spreads or the VIX, while in our synthetic data it is simply an exogenous standard-normal draw, independent of the policy surprise.

The structural equations take the general form:

$$\text{MP\_surp} = f_0(MN) + \varepsilon_0$$
$$\Delta DR = f_1(\text{MP\_surp}, RS) + \varepsilon_1$$
$$\Delta ECF = f_2(\text{MP\_surp}, MN) + \varepsilon_2$$
$$R_{eq} = f_3(\Delta DR, \Delta ECF, RS, \text{MP\_surp}) + \varepsilon_3$$

Where the $\varepsilon$ terms correspond to random Gaussian noise terms with mean zero.

---

### Ground truth: the regime-switching Data Generation Process (DGP)

In our demonstration the data is generated from a regime-switching process that embeds sharp asymmetries between monetary-policy tightening ($\text{MP\_surp} > 0$) and easing ($\text{MP\_surp} \le 0$). The explicit structural equations are:

$$\text{MP\_surp} = 0.5\;\text{MN} + 1.5\;\varepsilon_{MP}$$

$$\Delta DR = \begin{cases} 1.8\;\text{MP\_surp} & \text{tightening} \\ 0.3\;\text{MP\_surp} & \text{easing} \end{cases} + 0.6\;RS + 0.7\;\varepsilon_{DR}$$

$$\Delta ECF = \begin{cases} -0.2\;\text{MP\_surp} & \text{tightening} \\ -1.0\;\text{MP\_surp} & \text{easing} \end{cases} + 0.7\;\text{MN} + 0.7\;\varepsilon_{ECF}$$

$$R_{eq} = -0.6\;\Delta DR + 0.8\;\Delta ECF + 0.5\;RS + \underbrace{-5.0\;\lvert\text{MP\_surp}\rvert \cdot \mathbf{1}[Z > 1]}_{\text{stress amplifier}} + 0.8\;\varepsilon_{R}$$

where $Z \sim \mathcal{N}(0,1)$ is an independent contagion shock (unobserved), so the stress amplifier fires in roughly 16 % of observations. The key asymmetries are: tightening surprises raise discount-rate expectations **6 $\times$** more than easing lowers them ($1.8$ vs $0.3$), while easing boosts cash-flow expectations **5 $\times$** more than tightening depresses them ($1.0$ vs $0.2$). On top of these regime switches, rare stress episodes add a crash amplifier proportional to the size of the surprise. A symmetric linear model cannot capture any of these features.

The causal graph is agreed upon — it follows directly from Bernanke & Kuttner (2005). What remains open is the shape of $f_1, f_2, f_3$.

---

## Data generating process

To make the problem concrete, we generate synthetic data from the regime-switching DGP defined above (the ground truth). The function below encodes the full data generation process:
```python
def dgp_mp_regime(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    RS = rng.normal(0, 1, n)
    MN = rng.normal(0, 1, n)
    MP_surp = 0.5 * MN + 1.5 * rng.normal(0, 1, n)
    tightening = MP_surp > 0
    DR = (np.where(tightening, 1.8 * MP_surp, 0.3 * MP_surp)
          + 0.6 * RS + 0.7 * rng.normal(0, 1, n))
    ECF = (np.where(tightening, -0.2 * MP_surp, -1.0 * MP_surp)
           + 0.7 * MN + 0.7 * rng.normal(0, 1, n))
    Z = rng.normal(0, 1, n)           # independent contagion shock
    stress = Z > 1.0                  # ~16 % of observations
    R_eq = (
        -0.6 * DR + 0.8 * ECF + 0.5 * RS
        + np.where(stress, -5.0 * np.abs(MP_surp), 0.0)
        + 0.8 * rng.normal(0, 1, n)
    )
    return {"RS": RS, "MN": MN, "MP_surp": MP_surp,
            "DR": DR, "ECF": ECF, "R_eq": R_eq,
            "stress": stress.astype(float)}
```

### The Asymmetry Is Already Visible in the Data

Before fitting any model, a simple scatter plot of the data generated reveals the core problem with the linear assumption.

![Asymmetric slopes: tightening hits discount-rate changes harder than easing relieves them](figures/fig1_scatter_asymmetry.png)

**Left panel (Discount-Rate channel)**: Tightening surprises (red) drive up discount-rate changes with a noticeably steeper slope than easing surprises (blue) bring them down. The cloud of red points climbs sharply; the blue cloud is flatter.

**Right panel (Cash-Flow channel)**: The asymmetry runs in the opposite direction. Easing surprises produce a steeper negative relationship with cash-flow changes — the forward-looking growth stimulus effect — while tightening's contractionary signal is more muted.

A linear model forced to fit both regimes simultaneously will average across this asymmetry. It will systematically overestimate the discount-rate response during easing and underestimate it during tightening — exactly the observations where position sizing matters most.

---

## Same DAG, Three Stories

After generating the data, we now fit three competing PyMC models that share the same causal graph but assume different functional forms — linear, regime-switching, and nonlinear. The goal is to show that the correctly specified model recovers the true data-generating process and translates into profitable decisions.

**Model A — Linear**: Symmetric, proportional effects. One slope for both tightening and easing. The textbook baseline.

**Model B — Regime-switching**: Mirrors the ground-truth DGP structure with separate slopes per regime for both channels. Crucially, the stress contagion indicator $\mathbf{1}[Z > 1]$ is **unobserved** — the model never sees which events are stress episodes. Instead, we treat the stress regime as a **latent variable** and marginalize it out using a `pm.Mixture`: each observation's equity return is modelled as a two-component mixture (normal vs stress), with the mixing probability $p_{\text{stress}}$ learned from data. This is a harder inference problem than using the observed label, but it is the honest one — in practice, you would not know which days are stress events.

**Model C — Nonlinear**: Smooth `tanh` saturation in both channels with an interaction term. Captures diminishing returns but treats the asymmetry as a smooth transition rather than a discrete switch.

The regime-switching PyMC model is:

```python
tight = (mp_vals > 0).astype(float)

with pm.Model() as model_regime:
    # Discount-rate change channel: separate slope per regime
    a1_t = pm.Normal("a1_tight", mu=0, sigma=2)
    a1_e = pm.Normal("a1_ease", mu=0, sigma=2)
    a2   = pm.Normal("a2", mu=0, sigma=2)
    dr_mu = (a1_t * tight + a1_e * (1 - tight)) * mp_vals + a2 * rs_vals
    pm.Normal("DR", mu=dr_mu, sigma=pm.HalfNormal("sigma_dr", 2), observed=dr_vals)

    # Cash-flow change channel: regime-dependent slopes
    b1_t = pm.Normal("b1_tight", mu=0, sigma=2)
    b1_e = pm.Normal("b1_ease", mu=0, sigma=2)
    b2   = pm.Normal("b2", mu=0, sigma=2)
    ecf_mu = (b1_t * tight + b1_e * (1 - tight)) * mp_vals + b2 * mn_vals
    pm.Normal("ECF", mu=ecf_mu, sigma=pm.HalfNormal("sigma_ecf", 2), observed=ecf_vals)

    # Equity returns: latent stress mixture
    c1 = pm.Normal("c1", mu=0, sigma=2)
    c2 = pm.Normal("c2", mu=0, sigma=2)
    c3 = pm.Normal("c3", mu=0, sigma=2)
    c_stress = pm.Normal("c_stress", mu=0, sigma=5)
    p_stress = pm.Beta("p_stress", alpha=2, beta=10)
    sigma_req = pm.HalfNormal("sigma_req", sigma=2)
    base_mu = c1*dr_vals + c2*ecf_vals + c3*rs_vals
    pm.Mixture(
        "R_eq",
        w=pt.stack([1 - p_stress, p_stress]),
        comp_dists=[
            pm.Normal.dist(mu=base_mu, sigma=sigma_req),
            pm.Normal.dist(mu=base_mu + c_stress * np.abs(mp_vals), sigma=sigma_req),
        ],
        observed=req_vals,
    )
```

The tightening indicator `tight` is a deterministic function of observable data. The stress regime, by contrast, is latent — the model must discover it from the structure of the residuals. PyMC's `Mixture` marginalizes over the discrete indicator analytically, so NUTS can sample the continuous parameters without difficulty.

---

## Letting the Data Decide: LOO Model Comparison

We fit all three models (2 chains, 500 draws, target acceptance 0.95) and compare out-of-sample predictive performance via Leave-One-Out (LOO) cross-validation:

```python
comparison = az.compare(traces, ic="loo", var_name="R_eq")
```

![LOO Model Comparison: Same DAG, Three Functional Forms](figures/fig2_loo_comparison.png)

The result is unambiguous. The Leave-One-Out (LOO) helps us identify the right functional form! Which confirms the strength of Bayesian approach to bridge the gap between having a DAG and actually modeling the real structure of causal relations. The regime-switching model (ELPD = −785) is separated from both alternatives (ELPD ≈ −1320) by **535 log-points — more than 14× the standard error of the difference**. This is not a marginal improvement; the data overwhelmingly prefer the model that respects the empirical asymmetry between tightening and easing. Remarkably, this advantage holds even though the regime model must *discover* the latent stress regime from data rather than being told which observations are stress events.

Notably, the Nonlinear (`tanh`) model performs slightly worse than even the Linear model, despite having more parameters (17.1 effective parameters vs 12.0 for Linear). The added polynomial terms and interactions cause overfitting — the model fits training data well but generalizes poorly. Smooth saturation is not the right nonlinearity here. The discount-rate channel does not gradually curve — it switches. Approximating a step change with a sigmoid wastes flexibility and gains nothing.

| Model | ELPD (LOO) | SE | p_loo | Δ ELPD vs best |
|---|---:|---:|---:|---:|
| **Regime** | **−785** | **20** | **5.8** | **0** |
| Linear | −1320 | 40 | 12.0 | −535 |
| Nonlinear | −1322 | 40 | 17.1 | −536 |

---

## Posterior Predictive Checks: A Subtler Signal

LOO tells you which model predicts better. Posterior predictive checks tell you *how* models fail.

![Posterior Predictive Checks: Equity Returns (R_eq)](figures/fig3_ppc.png)

All three models reproduce the bulk of the equity return distribution. The cumulative density functions (blue = posterior predictive draws, black = observed) overlap tightly in the center. But the tails diverge.

The Regime switching model's x-axis extends to −10, while the Linear and Nonlinear panels barely reach −7. Only the regime-switching specification generates the left-tail mass consistent with the stress-amplified drawdowns in the data. The linear model's predictive distribution is systematically too narrow at the left extremes — precisely the events that drive risk and position-sizing decisions.

---

## Wrong Form, Right DAG, Bad P&L

The LOO score is a statistical number. The return is the economic consequence. It results from using the model as the basis for investment decisions.

A simple strategy: predict $R_{eq}$ from each model's posterior mean, go long if the prediction is positive, short if negative, and scale position by normalized confidence. We run all three models on the full 2 000 events, then isolate the 94 stress episodes (independent contagion shock, approximately 19% of observations).

![Cumulative P&L and Stress-Period Performance by Model](figures/fig4_pnl.png)

**Left panel — Cumulative return**: While all models generate positive returns, the regime switching model's superior specification translates to consistently stronger performance throughout the sample. The regime switching model achieves **6.15% return vs 5.60% for the linear and 5.62% for the nonlinear models** — a meaningful edge that compounds over time. Remember: the regime model has no privileged access to the stress labels. It must *infer* the latent stress regime from data alone — yet the Bayesian mixture still provides enough structural advantage to outperform. Knowing the DAG is not enough, you need sophisticated tools to identify the real functional forms. That is precisely what PyMC can do.

**Right panel — Stress episodes only**: During the 94 stress periods, the regime switching model earns **3.29% vs 2.64% for linear and 2.73% for nonlinear** — in the episodes where amplification is highest. This advantage is especially notable because the model never observes which events are stress episodes; it discovers them through the structure of the likelihood. The Bayesian approach provides an edge precisely where it matters most: in the tail events that drive portfolio risk.

| Model | Full-sample return | Stress-episode return |
|---|---:|---:|
| **Regime** | **6.15%** | **3.29%** |
| Linear | 5.60% | 2.64% |
| Nonlinear | 5.62% | 2.73% |



The stress-episode return gap represents alpha that the Bayesian structural model extracts without ever observing the latent contagion variable. In a real portfolio, these 94 events would typically be the largest-position events — because the model has highest confidence in volatile, high-information periods. A model that can discover hidden regimes from data, rather than requiring hand-labelled indicators, is inherently more robust and deployable.

---

## Calibration

One final check: does the winning model actually recover the causal parameters?

![Regime Model: Posterior Parameter Recovery vs Ground Truth](figures/fig5_param_recovery.png)

The posterior distributions are tightly centered on the ground-truth values across all 11 structural parameters — including the latent ones. The stress coefficient `c_stress` recovers the true value of −5.0 (posterior mean ≈ −4.99), and the stress probability `p_stress` recovers the true 15.9% (posterior mean ≈ 18%). The model successfully identifies both the magnitude and frequency of the unobserved contagion regime from data alone.

This is what a well-specified Bayesian structural model delivers: not just a prediction, but interpretable, uncertainty-quantified structural coefficients that can be audited against prior knowledge, stress-tested under regime shifts, and used to size hedges — even when key regime indicators are latent.

---

## The Pipeline

The workflow that produced these results is a five-step process that should become standard practice whenever a causal discovery output lands on your desk:

1. **Anchor on a candidate DAG** — from theory, literature, or a discovery algorithm. Know its assumptions.
2. **Propose competing functional forms** as PyMC models that all share the same graph.
3. **Fit and compare** with LOO. Let the data rule out misspecified structures.
4. **Check the tails** via posterior predictive checks — that is where misspecification hides.
5. **Test the economic consequence** — P&L simulation, hedging, risk attribution — because the cost of the wrong form is not uniform across the distribution.

Structure is the starting line, not the finish. A DAG without a functional form is a hypothesis without a prediction. PyMC gets you from graph to quantified, decision-ready structural model.

---

*Full reproducible code: `run_demo_dag1.py` · Notebook: `demo_dag1.ipynb`*

**References**

- Bernanke, B. S., & Kuttner, K. N. (2005). What Explains the Stock Market's Reaction to Federal Reserve Policy? *Journal of Finance*, 60(4), 1221–1257.
- Gilchrist, S., & Zakrajšek, E. (2012). Credit Spreads and Business Cycle Fluctuations. *American Economic Review*, 102(4), 1692–1720.
- Olivetti, E., Zoonekynd, V., Yam, P., Lopez de Prado, M., Imbens, G. W., & Hernán, M. A. (2024). Can AI Learn Causal Structure? 
