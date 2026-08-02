---
editor_options: 
  markdown: 
    wrap: 72
---

# Overcomplete Learning

Blind and semi-blind source separation in the **overcomplete** regime
(more sources than observations, `nsrc > nobs`). Given observations
`X = A S + noise`, the goal is to recover the mixing matrix `A` and the
sources `S`. The package implements a variational EM algorithm and a set
of baseline methods (ICA, OverICA, OLS, semi-blind EM), plus a parallel
simulation harness for benchmarking them on synthetic data.

------------------------------------------------------------------------

## Directory layout

```         
sim/Python/
├── src/overcomplete_learning/   # the package
│   ├── em.py                    # variational EM + all run_EM_* methods
│   ├── data.py                  # synthetic data generation & matrix utilities
│   ├── metrics.py               # error metrics (A_err, S_err_MCC/MSE, X_err, ...)
│   ├── run_study.py             # parallel study files
│   ├── fastICA.py, sdp_approach.py, semiblind_em.py, ...   # baselines
│   └── plotting.py              # helper functions for plotting
├── scripts/
│   ├── study.py                 # entry point: configure & launch a study
├── results/                     # csv files containing all simulated data used in thesis
├── notebooks/                   # analysis & figure notebooks (see below)
```

------------------------------------------------------------------------
## Installation
 Navigate to folder install exactly the same packages to get the vitural environment
``` bash
git clone https://github.com/otto-groen-roepstorff/SB-VEM.git
```

Navigate to folder and run

``` bash
uv venv
uv pip install -r requirements_exact_1.txt
uv pip install -e .
```


## Running a study (`scripts/study.py`)

`study.py` sweeps a set of methods and problem settings, runs each in
parallel over `(known_src_range × n_inits × n_data_sets)`, and writes
one CSV per configuration.

``` bash
python scripts/study.py
```

### Configure it

All knobs live in the `StudyConfig` dataclass at the top of
`scripts/study.py`. Edit the defaults, and/or the sweep lists in the
`__main__` block (`methods`, `nreps_range`, `nsrc_range`, ...). Key
fields:

| Field | Meaning | Default |
|------------------------|------------------------|------------------------|
| `nreps` | number of samples | 5000 |
| `nobs` | observation dimension (rows of `A`) | 10 |
| `nsrc` | source dimension (cols of `A`) — overcomplete when `> nobs` | 15 |
| `S_scale`, `A_scale` | scale of sources / mixing matrix | 1 |
| `sn_ratio` | signal-to-noise ratio (`err_sd` is derived from it in `__main__`) | 100 |
| `correlation_type` | noise structure (`'iid'`) | `'iid'` |
| `seed` | reproducibility seed | 1 |
| `EM_iters` | max inner EM iterations | 2500 |
| `convergence_threshold` | inner EM stop tolerance (relative ELBO change / nobs) | 1e-5 |
| `outer_iters`, `outer_tol` | outer debiasing loop cap / tolerance (OLS_debias only) | 10 / 1e-4 |
| `mu` | OverICA regularizer | 5 |
| `n_inits` | random initializations per dataset | 10 |
| `n_data_sets` | independent synthetic datasets | 30 |
| `method` | estimation method (see table below) | `'standard'` |
| `normalize_A_col` | normalize unknown columns of `A` | False |
| `whiten_data` | pre-whiten observations | False |
| `known_src_step` | granularity of the known-source sweep | 1 |
| `n_workers` | parallel workers (`None` → n_cores − 1) | None |
| `results_dir`, `log_dir` | output locations | `results/...` |

### Methods (`cfg.method`)

Registered in `EM_METHOD_REGISTRY` (`run_study.py`):

| key | description |
|------------------------------------|------------------------------------|
| `standard` | SB-VEM (full covariance noise) |
| `standard_iid` | SB-VEM, isotropic noise |
| `standard_wrong` | SB-VEM without updating posterior covariance (misspecified) |
| `OLS` | LS-B-VEM |
| `OLS_iid` | LS-B-VEM with isotropic noise |
| `OLS_debias` | LS-B-VEM+ |
| `OLS_debias_iid` | LS-B-VEM\_ with isotropic noise |
| `fast_ICA` | FastICA with whitening/dimension reduction (needs `n_unknown ≤ nobs`) |
| `OverICA` | overcomplete ICA via SDP (Podosinnikova et al. 2019) |
| `semiblind` | semi-blind EM (Lin, Xu & Liang 2009) |
| `random` | random-baseline floor |

------------------------------------------------------------------------

## Output

Each configuration produces a CSV in `results_dir`. One row per
`(n_known_src, init, dataset)`, with columns including:

-   **Identifiers:** `n_known_src`, `init`, `init_seed`, `data_seed`,
    `method`, `nobs`, `nsrc`, `nreps`, `sd_err`, `sn_ratio`,
    `normalize_A`, `err_correlation_type`
-   **Convergence:** `converged`, `convergence_iteration`
-   **Error metrics** (see `metrics.py`; lower is better except
    `S_err_MCC` where 1 is best): `A_err` (column-scaled MSE),
    `A_err_angle_mean`, `A_err_angle_max`, `S_err_MSE`, `S_err_MCC`,
    `X_err`, plus `Au_coherence`.

CSV filenames encode the configuration, e.g.
`em_study_nobs10_nsrc15_n5000_snratio100_errsd0.24_seed1_..._methodstandard_iid_....csv`.

------------------------------------------------------------------------

## Notebooks

| Notebook | Purpose |
|----------------------------|------------------------------------|
| `01_SB_VEM.ipynb` | A toy notebook to see the convergence of SB-VEM under different dimension|
| `02_visualize_effect.ipynb` | visualize recovery behaviour |
| `03_thesis_plots.ipynb` | generate report/thesis figures from study CSVs |
| `04_whitening_and_ica.ipynb` | whitening + FastICA plots |
| `05_error_metrics.ipynb` | error-metric behaviour and correlations across simulated studies |

------------------------------------------------------------------------

## Notes & caveats

-   **Overcomplete constraint:** `fast_ICA` require `n_unknown ≤ nobs`;

-   **`convergence_iteration` for `OLS_debias`:** reported as the
    *summed* inner iterations across outer passes, so it can exceed
    `EM_iters` and scales with `outer_iters` (the outer loop is
    currently capped by `outer_iters` ). Fix `outer_iters` when
    comparing debias runs.

-   **Reproducibility:** data and initialization seeds derive from
    `cfg.seed`; fix it when comparing runs.
