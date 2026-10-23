# Cosmological Parameter Estimation for Extended Universe Models (Pantheon+SH0ES)

A high-performance Python computing pipeline designed to investigate the **Hubble Tension**. The project implements both the standard Flat **$\Lambda$CDM** model and the extended **$w_0w_a\text{CDM}$** cosmological model featuring dynamic dark energy (CPL parametrization). Calculations are based on the full Pantheon+SH0ES observational dataset (1,701 Type Ia supernova light curves) and a full $1,657 \times 1,657$ covariance matrix preserving the STAT+SYS structure.

This research project was developed for the **Junior Academy of Sciences of Ukraine (JAS)**.

---

## 🌌 Mathematical Processing and Cosmological Analysis

The program performs joint precision estimation of cosmological parameters using the affine-invariant ensemble MCMC sampler `emcee`.

* **Distance Ladder Calibration Integration:**
For the joint fit, the standardized apparent magnitude `m_b_corr`, supernova absolute magnitude $M_B$, independent Cepheid distances `CEPH_DIST`, and the binary flag `IS_CALIBRATOR` are used. The model calculates the observational distance modulus as:
$$\mu_i = m_{b,\mathrm{corr},i} - M_B$$


For Cepheid calibrators, the residual in the likelihood function is computed as the difference between the derived distance modulus and the direct Cepheid distance:
$$\Delta\mu_i = \mu_i - \mu_{\mathrm{Cepheid},i}$$


For Hubble flow objects, the residual is compared against the theoretical Friedmann expansion curve:
$$\Delta\mu_i = \mu_i - \mu_{\mathrm{th}}(z_{\mathrm{HD}}, z_{\mathrm{HEL}}; H_0, \Omega_m, w_0, w_a)$$


This approach completely breaks the degeneracy between $M_B$ and $H_0$, allowing Cepheid anchors to directly fix the expansion curve zero-point. The `MU_SH0ES` column is a derived value and is not included in the likelihood calculations.
* **Official Cosmological Data Masking:**
The script reads all 1,701 data rows and the full $1,701 \times 1,701$ covariance matrix. The program automatically reproduces the official Pantheon+ selection cuts: objects are included in the analysis only if redshift $z_{\mathrm{HD}} > 0.01$ or if flagged as a Cepheid calibrator (`IS_CALIBRATOR == 1`). The corresponding covariance submatrix is sliced using NumPy indexing (`np.ix_`), strictly preserving the original row order. Multiple observations under the same Candidate ID (`CID`) are preserved to accurately account for inter-survey systematic correlations.
* **Matrix Algebra and Likelihood Function:**
Diagonal error columns (`m_b_corr_err_DIAG` and `MU_SH0ES_ERR_DIAG`) are used exclusively for plotting error bars in visualizations. The Gaussian log-likelihood function evaluates the **full covariance matrix**, accounting for all inter-survey systematic correlations and calibration uncertainties. To optimize the calculation of the quadratic form in the $\chi^2$ minimization instead of performing slow direct matrix inversion, fast Cholesky decomposition (`scipy.linalg.cho_factor`) is implemented.
* **Parallelization and Vectorized Integration:**
To calculate the comoving distance from Friedmann equations at each MCMC step, vectorized 64-node Gauss–Legendre quadrature is applied. Computations are vectorized across the entire dataset using NumPy axis expansion (`z[:, None]`), eliminating slow Python loops. The code supports multi-threaded parallel processing (Multiprocessing CPU Pool) with an isolated `spawn` context for stable performance on Windows and Unix architectures.

---

## 🚀 Installation

Running this computational pipeline requires **Python 3.10** or newer. Run the following commands in your terminal (PowerShell) from the project root directory:

```powershell
# 1. Create and activate an isolated virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade package manager and install data analysis libraries
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy emcee corner matplotlib

```

---

## 🏃 Execution

1. Ensure that the official data archive `DataRelease-main.zip` is unzipped into the root directory of the project. The program automatically scans the relative path: `DataRelease-main/Pantheon+_Data/4_DISTANCES_AND_COVAR/`.
2. Run the pipeline directly from the root folder:

```powershell
# Run the standard LCDM model (w0=-1, wa=0 fixed)
python pantheon_mcmc_H0+PLANCK.py --model lcdm --steps 5000

# Run the extended w0waCDM model (all 5 cosmological parameters free to fit)
python pantheon_mcmc_H0+PLANCK.py --model w0wacdm --steps 5000

```

### Key CLI Arguments:

* `--model`: Select cosmological scenario (`lcdm` or `w0wacdm`).
* `--steps`: Number of production MCMC steps per walker (recommended: 3000 to 5000).
* `--walkers`: Number of independent ensemble Markov chains (default: `32`).
* `--data` / `--covariance`: Explicitly pass custom file paths if located in a different directory.

---

## 📊 Results and Reporting

The script automatically generates and saves the following files to the working directory (appended with the chosen model suffix):

1. `pantheon_hubble_diagram_[model].png` — Publication-quality Hubble diagram showing distance curves on the top panel and fit residuals on the bottom panel.
2. `pantheon_confidence_contours_[model].png` — Multidimensional corner plot showing marginalized 1D parameter distributions and 2D confidence interval contours (68% and 95% levels).
3. `pantheon_posterior_samples_[model].npz` — Compressed NumPy array archive containing full flattened posterior chains for downstream analysis.
4. `pantheon_mcmc_summary_[model].json` — Structured high-precision summary report: medians, quantiles, Maximum A Posteriori (MAP) point, minimum $\chi^2$, degrees of freedom (`dof`), and reduced chi-squared ($\chi^2_{\mathrm{red}}$).

### Flat Priors:

* $50.0 \lt H_0 \lt 90.0 \ \text{km/s/Mpc}$
* $0.0 \lt \Omega_m \lt 1.0$
* $-3.0 \lt w_0 \lt 1.0$ (only for `w0wacdm`)
* $-3.0 \lt w_a \lt 3.0$ (only for `w0wacdm`)
* $-20.5 \lt M_B \lt -18.0 \ \text{mag}$

---

## 📚 References

* **Data Release:** [Official Pantheon+SH0ES Repository](https://github.com)
* **Cosmology Methodology:** [Brout et al. (2022), *The Pantheon+ Analysis: Cosmological Constraints*, ApJ 938 110](https://doi.org)
* **Distance Ladder:** [Riess et al. (2022), *A Comprehensive Measurement of the Local Value of the Hubble Constant with SNe Ia from Pantheon+ and Space Telescopes*, ApJL 934 L7](https://doi.org)
