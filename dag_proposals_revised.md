# DAG Proposals (Literature-Grounded): Causal Structures for the Blogpost Demo

This document replaces the previously *invented* DAGs with **causal structures anchored in well-known, highly cited finance/macroeconomics papers**. The goal is not to claim “the” true DAG of markets, but to (a) use **canonical, well-studied causal channels**, and (b) show how PyMC can turn an accepted *structural story* into **estimable, uncertainty-aware causal models**.

We keep the original blogpost intent: **“Congratulations, you have a DAG. Now what?”**—but now the DAGs correspond to widely used identification narratives.

---

## DAG 1: Monetary Policy Surprises → Equity Returns (Discount-Rate vs Cash-Flow Channels)

**Core references (empirical identification & decomposition):**
- Bernanke & Kuttner (2005), *What Explains the Stock Market’s Reaction to Federal Reserve Policy?* (NBER / JF version).
- Rigobon & Sack (2004/2005) and related “high-frequency identification” work on policy surprises (often used to isolate exogenous policy shocks in narrow windows).
- (Related event-study measures) Kuttner (2001) on futures-implied policy surprises (widely used; Bernanke & Kuttner build on this).

### Causal Structure (Conceptual DAG)

We model the **surprise** component of policy (measured from Fed funds futures in a narrow window) as the key “as-if exogenous” driver, and separate the two textbook channels to equity prices:

1) **Discount-rate channel**: policy surprise → risk-free rates/discount rates → equity returns  
2) **Cash-flow channel**: policy surprise → expected real activity/inflation → expected cash flows → equity returns

```
          ┌───────────────────────┐
          │  Macro news / outlook  │  (MN)
          └──────────┬────────────┘
                     │
                     ├───────────────┐
                     │               │
                     ▼               ▼
        ┌──────────────────┐   ┌──────────────────┐
        │ Policy decision  │   │ Expected cash     │
        │ & comms          │   │ flows (ECF)       │
        └────────┬─────────┘   └─────────┬────────┘
                 │                       │
                 ▼                       ▼
        ┌──────────────────┐   ┌──────────────────┐
        │ Policy surprise   │   │ Equity returns   │
        │ (MP_surp)         │   │ (R_eq)           │
        └────────┬─────────┘   └──────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Discount rates /  │
        │ yields (DR)       │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Equity returns    │
        │ (R_eq)            │
        └──────────────────┘

      + Risk sentiment / risk premia (RS) affect DR and R_eq
```

### Variables (Blog Demo Version)

| Variable | Symbol | Interpretation (demo) | Typical proxy in practice |
|---|---:|---|---|
| Monetary policy surprise | MP_surp | Futures-implied surprise in target rate (event window) | Fed funds futures (Kuttner-style); intraday rates |
| Discount-rate state | DR | Change in risk-free discounting / term structure | 2y/10y yields, OIS, term premium proxies |
| Expected cash flows | ECF | Cash-flow news / expected earnings/dividends | Analyst revisions, dividend futures, macro nowcasts |
| Risk sentiment / risk premia | RS | Time-varying risk appetite / required returns | VIX, credit spreads, intermediary constraints |
| Equity returns | R_eq | Equity index return (event window) | S&P 500, CRSP VW, etc. |
| Macro news / outlook | MN | Information that moves both policy and cash flows | Inflation/output surprises, surveys |

> **Identification note:** High-frequency event windows are used precisely to reduce confounding by MN and other slow-moving factors. In a blog demo we still keep MN/RS nodes to show what “can go wrong” if you relax that assumption.

---

### Functional Form A: Linear (Baseline)

```python
import numpy as np

def dgp_mp_linear(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    # Exogenous / slow-moving states
    RS = rng.normal(0, 1, n)          # risk sentiment
    MN = rng.normal(0, 1, n)          # macro news/outlook

    # Policy surprise: mostly driven by policy reaction + noise, but in event windows
    # we treat the remaining surprise as "as-if exogenous" conditional on MN.
    MP_surp = 0.3 * MN + rng.normal(0, 1, n)

    # Discount-rate channel
    DR = 0.8 * MP_surp + 0.6 * RS + rng.normal(0, 1, n)

    # Cash-flow channel (policy affects expected activity/cash flows)
    ECF = -0.4 * MP_surp + 0.7 * MN + rng.normal(0, 1, n)

    # Equity return combines both channels + direct risk-premia shifts
    R_eq = -0.6 * DR + 0.8 * ECF + 0.5 * RS + rng.normal(0, 1, n)

    return {"RS": RS, "MN": MN, "MP_surp": MP_surp, "DR": DR, "ECF": ECF, "R_eq": R_eq}
```

---

### Functional Form B: Regime Switching (Tightening vs Easing Asymmetry)

Empirically, markets often respond asymmetrically to tightening vs easing surprises (and the risk-premia component can dominate in stress).

```python
def dgp_mp_regime(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    RS = rng.normal(0, 1, n)
    MN = rng.normal(0, 1, n)

    MP_surp = 0.3 * MN + rng.normal(0, 1, n)

    tightening = MP_surp > 0

    # Discount-rate response is stronger in tightening surprises
    DR = np.where(tightening, 1.2 * MP_surp, 0.5 * MP_surp) + 0.6 * RS + rng.normal(0, 1, n)

    # Cash-flow response differs by regime too (e.g., easing supports growth expectations more)
    ECF = np.where(tightening, -0.3 * MP_surp, -0.7 * MP_surp) + 0.7 * MN + rng.normal(0, 1, n)

    # Equity returns: risk-off episodes amplify discount-rate channel
    stress = RS > 1.0
    R_eq = (-0.6 * DR + 0.8 * ECF + 0.5 * RS
            + np.where(stress, -1.0 * np.abs(MP_surp), 0.0)
            + rng.normal(0, 1, n))

    return {"RS": RS, "MN": MN, "MP_surp": MP_surp, "DR": DR, "ECF": ECF, "R_eq": R_eq}
```

---

### Functional Form C: Nonlinear Interaction (Risk Premia × Policy Surprise)

```python
def dgp_mp_nonlinear(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    RS = rng.normal(0, 1, n)
    MN = rng.normal(0, 1, n)
    MP_surp = 0.3 * MN + rng.normal(0, 1, n)

    # Saturating discount-rate response + interaction with risk sentiment
    DR = 1.0 * np.tanh(0.8 * MP_surp) + 0.7 * np.tanh(0.6 * RS) + 0.4 * MP_surp * RS + rng.normal(0, 1, n)

    # Cash-flow channel with diminishing returns
    ECF = (-0.6 * np.tanh(0.7 * MP_surp) + 0.8 * np.tanh(0.6 * MN) + rng.normal(0, 1, n))

    R_eq = -0.7 * DR + 0.9 * ECF + 0.6 * RS + rng.normal(0, 1, n)

    return {"RS": RS, "MN": MN, "MP_surp": MP_surp, "DR": DR, "ECF": ECF, "R_eq": R_eq}
```

---

### “How the Wrong Functional Form Loses Money” (Policy-event strategy)

A toy event strategy: predict equity reaction to *policy surprises* and trade intraday.

- Go long if predicted return > 0; short if < 0  
- Position size scales with |prediction| but is capped

Key failure mode: a linear model *averages across regimes* and misses the **interaction with risk sentiment**, leading to the biggest mistakes exactly when it matters (stress).

---

## DAG 2: Excess Bond Premium, Intermediary Risk-Bearing Capacity, and the Macro Cycle

**Core references (measurement + VAR evidence):**
- Gilchrist & Zakrajšek (2012), *Credit Spreads and Business Cycle Fluctuations* (AER): constructs the **excess bond premium (EBP)** and shows shocks to it predict declines in activity and asset prices.
- The “risk-bearing capacity of intermediaries” interpretation is explicit in Gilchrist & Zakrajšek and connected to intermediary constraint models (e.g., He & Krishnamurthy-type mechanisms referenced therein).

### Causal Structure (Conceptual DAG)

This DAG matches the narrative: **a deterioration in intermediary balance-sheet capacity** raises the EBP, tightens credit supply, weakens real activity, and ultimately worsens default risk—while spreads embed both default and non-default components.

```
   ┌──────────────────────────────┐
   │ Intermediary balance-sheet    │
   │ / risk-bearing capacity (IBC) │
   └───────────────┬──────────────┘
                   │
                   ▼
       ┌──────────────────────────┐
       │ Excess bond premium (EBP) │
       └──────────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ Credit supply     │    │ Credit spreads    │
│ conditions (CS)   │    │ (S)               │
└────────┬─────────┘    └────────┬─────────┘
         │                        │
         ▼                        │
┌──────────────────┐              │
│ Real activity     │              │
│ (Y_real)          │              │
└────────┬─────────┘              │
         │                        │
         ▼                        │
┌──────────────────┐              │
│ Default risk (DR) │──────────────┘
└──────────────────┘

 + Monetary policy (MP) responds to Y_real and affects risk-free rates embedded in S
```

### Variables (Blog Demo Version)

| Variable | Symbol | Interpretation (demo) | Typical proxy in practice |
|---|---:|---|---|
| Intermediary capacity shock | IBC | Risk-bearing capacity / dealer health | broker-dealer leverage, dealer profits, funding spreads |
| Excess bond premium | EBP | Non-default component of spreads | GZ EBP series |
| Credit supply conditions | CS | Tightness of credit supply | bank lending standards, credit growth |
| Real activity | Y_real | Output/investment proxy | IP, GDP nowcasts, investment |
| Default risk | DR | Expected default / EDF-like | issuer EDF, leverage, distance-to-default |
| Credit spreads | S | Corporate spreads vs Treasuries | OAS, Baa-Aaa, etc. |
| Monetary policy | MP | Policy stance/response | FFR, shadow rate |

---

### Functional Form A: Linear (Baseline)

```python
def dgp_ebp_linear(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    IBC = rng.normal(0, 1, n)  # intermediary capacity shocks
    MP  = rng.normal(0, 1, n)  # simplified policy stance shocks

    EBP = 0.8 * (-IBC) + rng.normal(0, 1, n)             # worse capacity -> higher premium
    CS  = -1.0 * EBP + 0.3 * MP + rng.normal(0, 1, n)    # tighter credit when EBP high
    Y_real = 0.9 * CS + rng.normal(0, 1, n)              # credit supply -> activity
    DR  = -0.7 * Y_real + rng.normal(0, 1, n)            # weak activity -> higher default risk

    # Spreads embed both default risk + nondefault premium + policy-rate component
    S = 1.2 * DR + 1.0 * EBP + 0.4 * MP + rng.normal(0, 1, n)

    return {"IBC": IBC, "MP": MP, "EBP": EBP, "CS": CS, "Y_real": Y_real, "DR": DR, "S": S}
```

---

### Functional Form B: Convex Stress Dynamics (Spreads “blow out” when DR is high)

```python
def dgp_ebp_convex(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    IBC = rng.normal(0, 1, n)
    MP  = rng.normal(0, 1, n)

    EBP = 0.9 * (-IBC) + rng.normal(0, 1, n)
    CS  = -1.2 * EBP + 0.3 * MP + rng.normal(0, 1, n)
    Y_real = 1.0 * CS + rng.normal(0, 1, n)

    # Default risk bounded in [0,1] via logistic mapping
    DR_latent = -0.8 * Y_real + rng.normal(0, 1, n)
    DR = 1 / (1 + np.exp(-DR_latent))

    # Convex spread response to default risk (stress convexity)
    S = 0.8 * EBP + 0.4 * MP + 1.0 * DR + 2.0 * (DR ** 2) + rng.normal(0, 0.5, n)

    return {"IBC": IBC, "MP": MP, "EBP": EBP, "CS": CS, "Y_real": Y_real, "DR": DR, "S": S}
```

---

### Functional Form C: Threshold Crisis Regime (Credit supply collapses past an EBP level)

```python
def dgp_ebp_threshold(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    IBC = rng.normal(0, 1, n)
    MP  = rng.normal(0, 1, n)

    EBP = 0.9 * (-IBC) + rng.normal(0, 1, n)

    crisis = EBP > 1.0

    CS = np.where(crisis, -2.0 * EBP, -1.0 * EBP) + 0.3 * MP + rng.normal(0, 1, n)
    Y_real = 1.0 * CS + np.where(crisis, -1.0, 0.0) + rng.normal(0, 1, n)

    DR = np.clip(0.2 + (-0.6 * Y_real) + 0.3 * crisis + rng.normal(0, 0.2, n), 0, 1)

    # Spreads jump with a crisis premium
    S = 1.0 * EBP + 1.2 * DR + 0.4 * MP + np.where(crisis, 2.0, 0.0) + rng.normal(0, 0.5, n)

    return {"IBC": IBC, "MP": MP, "EBP": EBP, "CS": CS, "Y_real": Y_real, "DR": DR, "S": S}
```

---

### “How the Wrong Functional Form Loses Money” (Credit strategy)

A toy strategy: predict **spread changes** (or expected excess returns on credit) from EBP/DR and go long credit if “cheap.”

Failure modes that match the literature intuition:
- Using a linear model when reality is **convex/threshold** → systematically underprices tail risk and “catches falling knives.”
- Ignoring the **non-default component** (EBP) → misattributes spread widening to default and makes wrong hedges.

---

## How PyMC Fits Into the Blogpost

For both DAGs, the blogpost can show:

1) **DAG as a structural scaffold** (from literature)
2) Competing **functional forms** (linear vs nonlinear/regime)
3) PyMC models that encode each form, then use **LOO**/**WAIC** + PPCs to pick the best out-of-sample specification
4) An “investor mistake” simulation: correct DAG, wrong functional form ⇒ bad P&L

---

## References (Primary)

- Bernanke, B. S., & Kuttner, K. N. (2005). *What Explains the Stock Market’s Reaction to Federal Reserve Policy?* (NBER working paper / Journal of Finance version).  
- Gilchrist, S., & Zakrajšek, E. (2012). *Credit Spreads and Business Cycle Fluctuations.* American Economic Review.  
- Coval, J., & Stafford, E. (2007). *Asset Fire Sales (and Purchases) in Equity Markets.* Journal of Financial Economics. (Optional extension if we want a third DAG focused on flows/price pressure.)  
