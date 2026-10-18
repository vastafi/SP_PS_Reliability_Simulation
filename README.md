# SP–PS Reliability Simulation

This repository contains the Python implementation used for the statistical
simulation and maximum likelihood estimation of serial-parallel (SP) and
parallel-serial (PS) reliability models with discrete uniform component
lifetimes.

## Model

Let

X_ij ~ DiscreteUniform{0, 1, ..., K}.

The reliability models considered are:

SP:
U = min_i max_j X_ij

PS:
V = max_i min_j X_ij

The experiments use the configuration:

- K = 9
- N = 3 units per sub-network
- M = 2 sub-networks
- 
## Repository contents

- `sp_ps_core.py` – core mathematical model, probability functions,
  simulation procedures and maximum likelihood estimation.
- `simulari.py` – simulation experiments.
- `run_experiments.py` – experiment orchestration.
- `mle_inversCDF.py` – MLE based on inverse-CDF simulation.
- `mle_struct_vs_invCDF.py` – comparison of structural and inverse-CDF
  approaches.
- `mle_structurala(min-max).py` – structural simulation approach.
- `extragere_n_optim.py` – extraction and analysis of optimal sample sizes.
- `data/` – input datasets.
- `figures/` – generated figures.
- `results/` – numerical results.

## Methods

The implementation includes:

1. Analytical probability distributions for SP and PS networks.
2. Direct structural simulation.
3. Inverse-CDF simulation.
4. Exact sampling from the theoretical PMF.
5. Complete-data maximum likelihood estimation.
6. Censored-data maximum likelihood estimation.
7. Sample-size determination using the central limit theorem.
8. Statistical comparison of simulation results.

## Reproducibility

The experiments are implemented in Python using NumPy and SciPy.

Install the required dependencies:

```bash
pip install -r requirements.txt

python run_experiments.py