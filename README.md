# Congratulations, You Have a DAG. Now What?

A technical demonstration showing why causal structure discovery alone is insufficient for making profitable decisions, and how PyMC bridges the gap to actionable causal inference.

## Context

This project responds to the [ADIA Lab Causal Discovery Challenge](https://hub.crunchdao.com/competitions/causality-discovery) results, which showed that supervised learning on labeled simulations achieves ~77% accuracy in causal structure discovery vs ~40% for classical methods.

**The key insight:** Discovering that A causes B doesn't tell you *how* A causes B. The functional form matters enormously for predictions and decisions.

## What This Demo Shows

1. **Same DAG, Different Functional Forms**: Generate synthetic financial data from one causal structure with three different functional specifications (linear, threshold, non-linear)

2. **Wrong Assumptions Lose Money**: Demonstrate how using a linear model when the true DGP has regime-switching leads to catastrophic losses

3. **PyMC Identifies the Correct Form**: Use Bayesian model comparison (LOO-CV) to select the best functional specification

4. **Uncertainty Quantification**: Show how PyMC provides credible intervals for risk management

## Repository Structure

```
congrats_you_have_a_dag/
├── README.md
├── pixi.toml                 # Environment configuration
├── .pre-commit-config.yaml   # Code quality hooks
├── src/
│   └── congrats_dag/
│       ├── __init__.py
│       ├── data_generation.py    # Synthetic data from known DGPs
│       ├── models.py             # PyMC model specifications
│       └── evaluation.py         # Model comparison & trading sim
├── notebooks/
│   └── demo.ipynb            # Main demonstration notebook
└── tests/
    └── test_dgp.py
```

## Quick Start

```bash
# Install pixi if needed
curl -fsSL https://pixi.sh/install.sh | bash

# Setup environment
pixi install

# Run the demo notebook
pixi run jupyter lab notebooks/demo.ipynb
```

## Key Dependencies

- **PyMC** (>=5.10): Probabilistic programming
- **ArviZ**: Bayesian model diagnostics and comparison
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## References

- Olivetti et al. "Can AI Learn Causal Structure?: Evidence from ADIA Lab's Causal Discovery Challenge"
- [PyMC Model Comparison Documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html)
- [LinkedIn discussion by Milos Maricic](https://www.linkedin.com/posts/milosmaricic_a-quant-fund-with-25000-realistic-causal-share-7424065120277880832-UDvb/)

## License

MIT

## Authors

PyMC Labs
