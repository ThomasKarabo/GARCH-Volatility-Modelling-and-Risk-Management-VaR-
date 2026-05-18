# Volatility Forecasting and Market Risk Quantification (VaR) using GARCH Frameworks

A Python notebook implementing a conditional volatility pipeline for 1-day ahead Value at Risk (VaR) estimation, validated through statistical backtesting — aligned with Basel regulatory frameworks for market risk internal models.

---

## Overview

This project applies a GARCH(1,1) model with Student's t-distributed errors to the FTSE 100 index to forecast conditional volatility and derive time-varying parametric VaR estimates. Model accuracy is evaluated using **Kupiec's Proportion of Failure (POF) test**.

The workflow follows the standard quant risk pipeline:

```
Data Acquisition → ARCH Effects Testing → GARCH Fitting → VaR Calculation → Backtesting
```

---

## Methodology

### 1. Data Acquisition
Daily FTSE 100 (`^FTSE`) closing prices are downloaded via `yfinance` (2020–2026). Log returns are computed as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Returns are scaled by 100 to improve GARCH optimisation convergence.

### 2. Testing for ARCH Effects
A **Ljung-Box test** on squared returns is used to detect autocorrelation in variance (conditional heteroskedasticity) — a prerequisite for GARCH modelling. Both lag-5 and lag-10 tests yield p-values far below 0.05, confirming the presence of volatility clustering.

### 3. GARCH(1,1) Model Fitting
The conditional variance equation:

$$\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

The model is estimated with a **Student's t-distribution** to account for the leptokurtic (fat-tailed) nature of financial returns. All core parameters ($\omega$, $\alpha$, $\beta$, $\nu$) are statistically significant, with $\alpha + \beta = 0.9474 < 1$, confirming **stationarity and mean reversion** of conditional variance.

### 4. Value at Risk Estimation
Parametric GARCH-VaR at confidence level $1 - \alpha$:

$$\text{VaR}_{t} = -\left(\mu_t + t_{\nu,\,\alpha} \times \sigma_t\right)$$

VaR is computed at both **95%** and **99%** confidence levels using the t-distribution quantile scaled by the GARCH conditional volatility.

### 5. Backtesting — Kupiec's POF Test
Model validity is assessed using the **Likelihood Ratio test**:

$$LR_{POF} = -2\ln\left[\frac{(1-\alpha)^{N-x}\,\alpha^x}{(1-\hat{p})^{N-x}\,\hat{p}^x}\right] \sim \chi^2(1)$$

A breach occurs when the realised loss exceeds the predicted VaR. The test checks whether the observed breach rate is statistically consistent with the model's assumed confidence level.

---

## Results Summary

| Confidence Level | Breaches | POF p-value | Result |
|:---:|:---:|:---:|:---:|
| 95% | — | < 0.05 | ❌ Reject |
| 99% | — | < 0.05 | ❌ Reject |

The GARCH(1,1) model was rejected at both confidence levels, indicating systematic **underestimation of tail risk** in the FTSE 100 over the sample period. This motivates extensions such as GJR-GARCH, EGARCH, or filtered historical simulation approaches.

---

## Requirements

```bash
pip install numpy pandas yfinance matplotlib arch scipy statsmodels
```

| Package | Purpose |
|---|---|
| `arch` | GARCH model estimation |
| `yfinance` | Market data download |
| `scipy.stats` | t-distribution quantiles and chi-squared test |
| `statsmodels` | Ljung-Box diagnostic test |
| `matplotlib` | Visualisation |

---

## Usage

Open the notebook and run cells sequentially:

```bash
jupyter notebook Volatility_Forecasting_and_Market_Risk_Quantification__VaR__using_GARCH_Frameworks.ipynb
```

To change the asset, update the ticker on the data acquisition cell:

```python
ticker = "^FTSE"   # Replace with any yfinance-compatible ticker
```

---

## Project Structure

```
├── Volatility_Forecasting_and_Market_Risk_Quantification__VaR__using_GARCH_Frameworks.ipynb
└── README.md
```

---

## References

- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987–1007.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2), 73–84.
- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk.*
