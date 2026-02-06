# DAG Proposals: Ground Truth Causal Structures for Blogpost Demo

This document defines two causal DAGs with their data generating processes, three functional forms each, mechanisms for losing money, and how PyMC identifies the correct specification.

---

## DAG 1: Interest Rate Transmission to Asset Returns

### Causal Structure

```
                    ┌─────────────────┐
                    │ Market Sentiment│
                    │       (S)       │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │Interest Rate│   │ Investment  │   │   Asset     │
    │  Change (R) │──▶│ Flow (F)    │──▶│ Returns (Y) │
    └─────────────┘   └─────────────┘   └─────────────┘
              │                                 ▲
              └─────────────────────────────────┘
                      (direct effect)
```

### Variables

| Variable | Symbol | Description | Units |
|----------|--------|-------------|-------|
| Interest Rate Change | R | Fed rate decision (basis points) | -50 to +50 bps |
| Market Sentiment | S | Aggregate risk appetite index | -1 to +1 (normalized) |
| Investment Flow | F | Net fund flows into equities | $ billions |
| Asset Returns | Y | Monthly equity index return | % |

### Causal Relationships (Edges)

1. **S → R**: Sentiment influences Fed decisions (hawkish when optimistic)
2. **S → F**: Optimistic sentiment drives fund inflows
3. **S → Y**: Sentiment directly affects returns (animal spirits)
4. **R → F**: Rate hikes reduce investment flows
5. **R → Y**: Direct rate effect on discount rates/valuations
6. **F → Y**: Flows drive prices (demand effect)

---

### Functional Form A: Linear (Baseline)

**Data Generating Process:**
```python
import numpy as np

def dgp_linear(n=1000, seed=42):
    np.random.seed(seed)

    # Exogenous
    S = np.random.normal(0, 0.3, n)  # Sentiment

    # Structural equations (LINEAR)
    R = 0.5 * S + np.random.normal(0, 0.1, n)  # Rate change
    F = 2.0 * S - 1.5 * R + np.random.normal(0, 0.5, n)  # Flows
    Y = 0.3 * S - 0.8 * R + 0.4 * F + np.random.normal(0, 0.2, n)  # Returns

    return {"S": S, "R": R, "F": F, "Y": Y}
```

**Characteristics:**
- All effects are proportional and additive
- Easy to estimate with OLS
- Symmetric responses to positive/negative shocks

---

### Functional Form B: Threshold/Regime-Switching

**Data Generating Process:**
```python
def dgp_threshold(n=1000, seed=42):
    np.random.seed(seed)

    S = np.random.normal(0, 0.3, n)
    R = 0.5 * S + np.random.normal(0, 0.1, n)

    # THRESHOLD: Flows respond differently to rate hikes vs cuts
    F = np.where(
        R > 0,  # Rate HIKE regime
        2.0 * S - 3.0 * R,  # Strong negative response to hikes
        2.0 * S - 0.5 * R   # Weak response to cuts (asymmetric)
    ) + np.random.normal(0, 0.5, n)

    # THRESHOLD: Returns crash when sentiment turns negative
    Y = np.where(
        S < -0.2,  # PANIC regime
        -1.5 + 0.1 * S - 1.5 * R + 0.2 * F,  # Regime shift down
        0.3 * S - 0.8 * R + 0.4 * F          # Normal regime
    ) + np.random.normal(0, 0.2, n)

    return {"S": S, "R": R, "F": F, "Y": Y}
```

**Characteristics:**
- Markets react asymmetrically to rate hikes vs cuts
- Panic regime when sentiment drops below threshold
- A linear model will average across regimes and miss both

---

### Functional Form C: Non-linear (Saturating + Interaction)

**Data Generating Process:**
```python
def dgp_nonlinear(n=1000, seed=42):
    np.random.seed(seed)

    S = np.random.normal(0, 0.3, n)
    R = 0.5 * S + np.random.normal(0, 0.1, n)

    # NON-LINEAR: Flows saturate (tanh) - can't go infinitely negative
    F = 2.0 * np.tanh(3 * S) - 1.5 * np.tanh(2 * R) + np.random.normal(0, 0.5, n)

    # INTERACTION: Rate effect depends on sentiment
    # When sentiment is positive, rate hikes hurt MORE (overconfidence correction)
    # When sentiment is negative, rate cuts help MORE (relief rally)
    interaction = -0.5 * R * S  # Amplification term

    Y = (0.3 * S
         - 0.8 * R
         + interaction
         + 0.4 * np.tanh(F)  # Diminishing flow effect
         + np.random.normal(0, 0.2, n))

    return {"S": S, "R": R, "F": F, "Y": Y}
```

**Characteristics:**
- Saturating effects (bounded by tanh)
- Interaction: rate impact modulated by sentiment
- Linear model will systematically under/over-predict at extremes

---

### How Wrong Functional Forms Lose Money
*To develop*

Example of using the wrong functional form and estimation

### PyMC Solution: Identifying the Correct Functional Form
*To develop*


---

## DAG 2: Corporate Credit Risk Transmission

### Causal Structure

```
    ┌─────────────┐         ┌─────────────┐
    │  Macro      │         │  Industry   │
    │  Shock (M)  │         │  Leverage(L)│
    └──────┬──────┘         └──────┬──────┘
           │                       │
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │  Firm       │◀────────│  Contagion  │
    │  Default(D) │         │  Risk (C)   │
    └──────┬──────┘         └──────┬──────┘
           │                       │
           └───────────┬───────────┘
                       ▼
                ┌─────────────┐
                │Credit Spread│
                │  Change (Y) │
                └─────────────┘
```

### Variables

| Variable | Symbol | Description | Units |
|----------|--------|-------------|-------|
| Macro Shock | M | GDP growth surprise | % deviation |
| Industry Leverage | L | Sector debt/equity ratio | ratio |
| Firm Default Prob | D | Expected default frequency | 0-1 probability |
| Contagion Risk | C | Network spillover measure | 0-1 index |
| Credit Spread Change | Y | Corporate bond spread vs treasury | bps |

### Causal Relationships

1. **M → D**: Macro shocks affect default probability
2. **L → D**: Higher leverage increases default risk
3. **L → C**: Leveraged sectors create contagion risk
4. **C → D**: Contagion amplifies defaults (feedback)
5. **D → Y**: Default expectations widen spreads
6. **C → Y**: Systemic risk commands premium

---

### Functional Form A: Linear

```python
def dgp_credit_linear(n=1000, seed=42):
    np.random.seed(seed)

    # Exogenous
    M = np.random.normal(0, 1, n)  # Macro shock
    L = np.random.uniform(0.5, 2.0, n)  # Leverage

    # Structural equations
    C = 0.3 * L + np.random.normal(0, 0.1, n)  # Contagion
    D = -0.2 * M + 0.15 * L + 0.1 * C + np.random.normal(0, 0.05, n)
    D = np.clip(D, 0, 1)  # Probability bounds
    Y = 50 * D + 30 * C + np.random.normal(0, 10, n)  # Spread in bps

    return {"M": M, "L": L, "C": C, "D": D, "Y": Y}
```

---

### Functional Form B: Exponential Contagion (Network Effects)

```python
def dgp_credit_exponential(n=1000, seed=42):
    np.random.seed(seed)

    M = np.random.normal(0, 1, n)
    L = np.random.uniform(0.5, 2.0, n)

    # EXPONENTIAL: Contagion explodes with leverage
    C = 0.1 * np.exp(L) - 0.1 + np.random.normal(0, 0.1, n)
    C = np.clip(C, 0, 1)

    # NON-LINEAR: Default probability with floor and ceiling
    D_linear = -0.2 * M + 0.15 * L + 0.3 * C
    D = 1 / (1 + np.exp(-5 * (D_linear - 0.3)))  # Logistic transform
    D = D + np.random.normal(0, 0.02, n)
    D = np.clip(D, 0.01, 0.99)

    # CONVEX: Spread explodes non-linearly with default prob
    Y = 20 + 100 * D + 200 * D**2 + 30 * C + np.random.normal(0, 10, n)

    return {"M": M, "L": L, "C": C, "D": D, "Y": Y}
```

**Key Feature:** The D² term means spreads explode super-linearly as default risk rises.

---

### Functional Form C: Threshold with Contagion Cascade

```python
def dgp_credit_cascade(n=1000, seed=42):
    np.random.seed(seed)

    M = np.random.normal(0, 1, n)
    L = np.random.uniform(0.5, 2.0, n)

    # Normal contagion
    C_base = 0.3 * L + np.random.normal(0, 0.1, n)

    # CASCADE: When macro shock exceeds threshold AND leverage high
    # → Contagion explodes (systemic crisis)
    crisis = (M < -1.5) & (L > 1.5)
    C = np.where(crisis, 0.8 + 0.1 * L, C_base)
    C = np.clip(C, 0, 1)

    # Default jumps in crisis
    D_normal = -0.2 * M + 0.15 * L + 0.1 * C
    D = np.where(crisis, 0.5 + 0.3 * L, D_normal)
    D = np.clip(D, 0, 1) + np.random.normal(0, 0.02, n)
    D = np.clip(D, 0.01, 0.99)

    # Spread with crisis premium
    crisis_premium = np.where(crisis, 200, 0)
    Y = 50 * D + 30 * C + crisis_premium + np.random.normal(0, 10, n)

    return {"M": M, "L": L, "C": C, "D": D, "Y": Y}
```

**Key Feature:** The interaction M×L creates a crisis regime that the linear model completely misses.

---

### How Wrong Functional Forms Lose Money


#### Trading Strategy: Credit Spread Mean Reversion

Buy corporate bonds when spread is "too wide" (model predicts narrowing).

```python
def credit_trading_strategy(Y_true, Y_predicted, capital=10_000_000):
    """
    Mean reversion: Buy when spread wider than predicted (expect narrowing)
    """
    mispricing = Y_true - Y_predicted  # Positive = spread wider than expected

    # Position: Long credit when spread "too wide"
    position = np.sign(mispricing) * np.minimum(np.abs(mispricing) / 50, 1)

    # P&L: Make money if spread narrows after we buy
    spread_change = np.diff(Y_true, prepend=Y_true[0])
    pnl = -position * spread_change * capital / 100  # Negative because we're long

    return np.cumsum(pnl)
```

#### Loss Scenarios

| True DGP | Model Used | Failure Mode | Financial Impact |
|----------|------------|--------------|------------------|
| Exponential | Linear | Underestimates spread at high D | Catches falling knife in stress |
| Cascade | Linear | Misses crisis regime entirely | Catastrophic loss in tail event |
| Cascade | Exponential | Gradual vs sudden transition | Premature buying before cascade |

**Concrete Example:**

True DGP is **Cascade**, analyst uses **Linear**:
- In normal times (95% of data): Linear model works fine
- Pre-crisis: M = -1.3, L = 1.6 → Linear predicts Y = 80 bps
- True spread jumps to Y = 350 bps (cascade triggered)
- Analyst is long credit expecting mean reversion → 270 bps loss × leverage = BLOW UP

---

### PyMC Solution for DAG 2

#### Competing Model Specifications

```python
# Model 1: Linear
with pm.Model() as credit_linear:
    beta_d = pm.Normal("beta_d", 50, 20)
    beta_c = pm.Normal("beta_c", 30, 20)
    sigma = pm.HalfNormal("sigma", 20)

    mu = beta_d * D + beta_c * C
    Y_obs = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)

# Model 2: Quadratic in D (captures convexity)
with pm.Model() as credit_quadratic:
    beta_d = pm.Normal("beta_d", 50, 20)
    beta_d2 = pm.Normal("beta_d2", 100, 50)  # Convexity
    beta_c = pm.Normal("beta_c", 30, 20)
    sigma = pm.HalfNormal("sigma", 20)

    mu = beta_d * D + beta_d2 * D**2 + beta_c * C
    Y_obs = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)

# Model 3: Regime-switching with interaction
with pm.Model() as credit_regime:
    # Normal regime
    beta_d_normal = pm.Normal("beta_d_normal", 50, 20)
    beta_c_normal = pm.Normal("beta_c_normal", 30, 20)

    # Crisis regime
    crisis_premium = pm.HalfNormal("crisis_premium", 100)

    # Crisis indicator (learned from data)
    crisis_threshold_m = pm.Normal("threshold_m", -1.5, 0.5)
    crisis_threshold_l = pm.Normal("threshold_l", 1.5, 0.3)

    crisis = (M < crisis_threshold_m) & (L > crisis_threshold_l)

    mu = pm.math.switch(
        crisis,
        beta_d_normal * D + beta_c_normal * C + crisis_premium,
        beta_d_normal * D + beta_c_normal * C
    )

    sigma = pm.HalfNormal("sigma", 20)
    Y_obs = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)
```

#### Model Comparison

```python
# Fit all models
traces = {
    "linear": pm.sample(model=credit_linear),
    "quadratic": pm.sample(model=credit_quadratic),
    "regime": pm.sample(model=credit_regime)
}

# LOO comparison
comparison = az.compare(traces, ic="loo")

# Plot comparison
az.plot_compare(comparison)
```

#### Key PyMC Diagnostic: Posterior Predictive by Regime

```python
# Critical check: How does each model perform in crisis vs normal?
def evaluate_by_regime(trace, model, M, L, Y):
    crisis_mask = (M < -1.5) & (L > 1.5)

    with model:
        ppc = pm.sample_posterior_predictive(trace)

    Y_pred = ppc.posterior_predictive["Y"].mean(dim=["chain", "draw"])

    rmse_normal = np.sqrt(np.mean((Y[~crisis_mask] - Y_pred[~crisis_mask])**2))
    rmse_crisis = np.sqrt(np.mean((Y[crisis_mask] - Y_pred[crisis_mask])**2))

    return {"normal": rmse_normal, "crisis": rmse_crisis}

# Linear model will have:
# - rmse_normal ≈ 15 (acceptable)
# - rmse_crisis ≈ 150 (catastrophic!)

# Regime model will have:
# - rmse_normal ≈ 15
# - rmse_crisis ≈ 25 (captures the jump)
```

---

## Summary: PyMC Tools for Functional Form Selection

| Tool | Purpose | How It Helps |
|------|---------|--------------|
| `az.compare()` | LOO-CV model comparison | Ranks models by predictive accuracy |
| `az.plot_compare()` | Visualize model weights | Shows confidence in ranking |
| `pm.sample_posterior_predictive()` | Generate predictions | Visual check against data |
| `az.plot_ppc()` | Posterior predictive check | Identifies systematic misfits |
| `pm.set_data()` | Out-of-sample prediction | True test on held-out data |
| `az.loo_pit()` | Probability integral transform | Detects model misspecification |

### The Punchline for the Blogpost

**"The ADIA Lab Challenge can tell you the DAG. PyMC tells you everything else."**

1. **Structure** (from causal discovery): R → F → Y ✓
2. **Functional form** (from PyMC model comparison): Threshold, not linear ✓
3. **Parameters** (from PyMC posterior): β = -3.0 when R > 0 ✓
4. **Uncertainty** (from PyMC credible intervals): 95% CI for stress scenarios ✓
5. **Validation** (from PyMC PPC): Model captures regime shifts ✓

Without steps 2-5, you have a diagram. With them, you have an edge.

---

## References

- [PyMC Model Comparison Documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html)
- [ArviZ compare API](https://python.arviz.org/en/latest/api/generated/arviz.compare.html)
- [Stochastic Volatility Example](https://www.pymc.io/projects/examples/en/latest/time_series/stochastic_volatility.html)
- Vehtari, A., Gelman, A., Gabry, J. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC


### How Wrong Functional Forms Lose Money
*To develop*

Example of using the wrong functional form and estimation

### PyMC Solution: Identifying the Correct Functional Form
*To develop*