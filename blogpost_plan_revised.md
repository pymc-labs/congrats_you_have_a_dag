# Blogpost Development Plan: From Causal Discovery to Causal Inference with PyMC (Finance)

**Response to:** ADIA Lab Causal Discovery Challenge + Milos Maričić’s LinkedIn post (quant fund with 25,000 labeled DAGs)  
**Audience:** Quant finance practitioners, causal inference users, applied ML teams

---

## Core Message

**“Congratulations, you have a DAG. Now what?”**

The ADIA Lab Challenge shows that, *in a supervised setting with many labeled simulations*, ML can classify causal roles and recover structure far better than classical constraint-based methods (top solutions ~77% balanced accuracy vs ~40% baselines). The practical finance question is what comes next: **structure is not yet an estimand, an intervention, or a tradable edge**.

PyMC’s sweet spot is the next step: **turning a candidate DAG into an estimable structural model**, comparing plausible functional forms, and carrying uncertainty into decisions.

(We’ll cite the ADIA Lab report directly and keep the tone respectful but realistic about real-market constraints like confounding, nonstationarity, and regime shifts.)

---

## Key Adjustment Based on Feedback

The earlier draft used invented finance DAGs. This revision uses **literature-grounded causal structures** from highly cited finance/macro papers, and then builds toy SCMs/DGPs that instantiate those channels for demonstration.

**Selected DAGs for the demo (with canonical references):**
1) **Monetary policy surprises → equity returns via discount-rate vs cash-flow channels** (Bernanke & Kuttner 2005; related high-frequency identification work).  
2) **Excess bond premium / intermediary risk-bearing capacity → credit supply → macro activity → default risk → credit spreads** (Gilchrist & Zakrajšek 2012).

(An optional third, if space allows: mutual-fund outflows → forced selling → price pressure → return reversals, per Coval & Stafford 2007.)

---

## Blogpost Structure

### 1) Hook: The ADIA Lab Challenge
- Summarize the setup: 25k labeled datasets with ground-truth DAGs; node-role classification task; best methods were supervised/hybrid.
- Make the transition: in real finance, we rarely get labeled DAGs—so discovery outputs are *candidates*, not truth.

### 2) Why a DAG Isn’t Enough
- A DAG gives conditional independencies and adjustment sets, but not:
  - functional form (linear/nonlinear, thresholds),
  - dynamics and lags,
  - measurement error / proxying,
  - or decision-relevant uncertainty.

### 3) Demo 1 (Literature DAG): Monetary Policy Surprise → Equity Returns
- Present the canonical DAG (policy surprise → discount rates + expected cash flows → equity returns, with risk sentiment as a modifier).
- Fit **three competing PyMC models**: linear, regime-switching, nonlinear interaction.
- Use **LOO/WAIC** + posterior predictive checks to select the best specification.
- Show “how you lose money” if you keep the right DAG but choose the wrong functional form.

### 4) Demo 2 (Literature DAG): Excess Bond Premium → Macro → Credit Risk → Spreads
- Present the canonical DAG (intermediary capacity → EBP → credit supply → activity → default risk; spreads embed both default and non-default components).
- Same modeling workflow: competing structural forms, model comparison, stress-period validation.

### 5) PyMC Takeaway: A Practical Pipeline
1. Causal discovery → candidate structures (possibly an equivalence class)
2. PyMC → estimate structural relationships under competing functional forms
3. Model comparison → select what predicts and generalizes
4. Stress tests → validate in regimes that matter
5. Decision layer → interventions/hedges with uncertainty-aware sizing

### 6) Close: What We’d Do in a Real Engagement
- Use domain constraints + economic theory to propose a small set of plausible DAGs.
- Encode measurement models for proxies.
- Fit and compare structural models; validate under nonstationarity.
- Translate posterior uncertainty into risk limits and capital allocation.

---

## Assets / Appendix Ideas
- A one-page diagram: “DAG → structural equations → posterior → decision”
- A short code block per model class (linear vs regime vs nonlinear)
- A caution box: “Why discovery from observational finance data is hard (unmeasured confounding, reflexivity, policy endogeneity).”

