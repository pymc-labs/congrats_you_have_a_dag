# Blogpost Development Plan: Causal Discovery to Causal Inference with PyMC

**Response to**: ADIA Lab Causal Discovery Challenge & Milos Maricic's LinkedIn post
**Target audience**: Quant finance professionals, causal inference practitioners

---

## Core Message

**"Congratulations, you have a DAG. Now what?"**

The ADIA Lab Challenge proved supervised learning can discover causal structures (~77% accuracy vs ~40% for classical methods). But a DAG alone doesn't make money. PyMC bridges the gap from structure discovery to actionable causal inference.

---

## Blogpost Structure

### 1. Introduction 
- Reference the ADIA Lab Challenge results and LinkedIn discussion
- Acknowledge the achievement: supervised learning on labeled simulations works
- Introduce the gap: structure != functional relationships != profits
- Position PyMC Labs' perspective

### 2. The Missing Piece: Functional Forms 
- A DAG shows A causes B, but not HOW (linear? non-linear? temporal lags?)
- Same DAG can produce wildly different predictions depending on functional form
- Visual example: one DAG, three functional forms, three different investment outcomes

### 3. Code Demo: Same DAG, Different Functional Forms

**Setup:**
```python
# Define a simple financial DAG:
# interest_rate -> investment_behavior -> returns
# macro_sentiment -> investment_behavior
# macro_sentiment -> returns
```

**Demonstrate:**
- Generate synthetic data from ground-truth DAG with known functional form
- Show that discovering the DAG structure is only step one
- Apply three different functional assumptions to same DAG
- Compare predictive performance and "investment returns"

### 4. PyMC as the Solution 
- Probabilistic programming allows specifying AND inferring functional relationships
- Model comparison to select best functional form
- Uncertainty quantification for risk management
- Out-of-sample validation as the true test

**Code example:**
```python
import pymc as pm

with pm.Model() as causal_model:
    # Encode DAG structure
    # Specify flexible functional forms with priors
    # Let the data inform the relationships
```


### 6. Conclusion: The Complete Pipeline
1. Causal discovery (ADIA Lab methods) -> candidate DAGs
2. PyMC modeling -> infer functional relationships
3. Model comparison -> select best specification
4. Out-of-sample testing -> validate causal claims
5. Decision-making -> interventions with quantified uncertainty
