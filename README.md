# Volatility Forecasting and Market Risk Quantification (VaR) using GARCH Frameworks

A Python notebook implementing a conditional volatility pipeline for 1-day ahead Value at Risk (VaR) estimation, validated through statistical backtesting — aligned with Basel regulatory frameworks for market risk internal models.

The project evaluates three models from the GARCH family (**Standard GARCH**, **GJR-GARCH**, and **EGARCH**), ultimately selecting **EGARCH(1,1)** as the preferred model based on both statistical fit (AIC/BIC) and backtesting validity (Kupiec's POF test).

---

## Overview

Daily FTSE 100 log returns are modelled using a conditional heteroskedasticity framework. After diagnosing ARCH effects, three GARCH-family models are fit and compared. Time-varying parametric VaR is derived from each model's conditional volatility, and validity is confirmed via Kupiec's Proportion of Failure (POF) test.

```
Data Acquisition → ARCH Effects Testing → GARCH Fitting → VaR Calculation → Backtesting → Model Selection
```

---

## Methodology

### 1. Data Acquisition
Daily FTSE 100 (`^FTSE`) closing prices are downloaded via `yfinance` (2020–2026). Log returns are computed as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Returns are scaled by 100 to improve GARCH optimisation convergence.

### 2. Testing for ARCH Effects
A **Ljung-Box test** on squared returns confirms conditional heteroskedasticity — a prerequisite for GARCH modelling. Both lag-5 and lag-10 tests yield p-values far below 0.05 ($3.73 \times 10^{-79}$ and $4.10 \times 10^{-140}$ respectively), confirming the presence of volatility clustering.

### 3. Model Fitting

Three GARCH-family models are estimated with **Student's t-distributed errors** to account for the leptokurtic nature of financial returns:

**Standard GARCH(1,1)**

$$\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

**GJR-GARCH(1,1)** — extends GARCH with an asymmetric term $\gamma$ that amplifies volatility specifically after negative shocks:

$$\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \gamma\epsilon_{t-1}^2 I_{t-1} + \beta\sigma_{t-1}^2$$

**EGARCH(1,1)** — models the log of conditional variance, ensuring non-negativity and explicitly separating the size and sign of market shocks:

$$\ln(\sigma_t^2) = \omega + \alpha\left(\left|\frac{\epsilon_{t-1}}{\sigma_{t-1}}\right| - E\left[\left|\frac{\epsilon_{t-1}}{\sigma_{t-1}}\right|\right]\right) + \gamma\frac{\epsilon_{t-1}}{\sigma_{t-1}} + \beta\ln(\sigma_{t-1}^2)$$

### 4. Value at Risk Estimation
Parametric GARCH-VaR at confidence level $1 - \alpha$:

$$\text{VaR}_{t} = -\left(\mu_t + t_{\nu,\,\alpha} \times \sigma_t\right)$$

VaR is computed at both **95%** and **99%** confidence levels using the t-distribution quantile scaled by the model's conditional volatility.

### 5. Backtesting — Kupiec's POF Test
Model validity is assessed using the **Likelihood Ratio test**:

$$LR_{POF} = -2\ln\left[\frac{(1-\alpha)^{N-x}\,\alpha^x}{(1-\hat{p})^{N-x}\,\hat{p}^x}\right] \sim \chi^2(1)$$

A breach occurs when the realised loss exceeds the predicted VaR. The test checks whether the observed breach rate is statistically consistent with the model's assumed confidence level. Models with a p-value $\geq 0.05$ are accepted.

---

## Results

| Model | AIC | BIC | Kupiec p-value (99%) | Status |
|:---|:---:|:---:|:---:|:---:|
| Standard GARCH(1,1) | — | — | < 0.05 | ❌ Rejected |
| GJR-GARCH(1,1) | — | — | < 0.05 | ❌ Rejected |
| **EGARCH(1,1)** | **3607.91** | **3639.84** | **0.1578** | ✅ Accepted |

The **EGARCH(1,1)** model is selected as the preferred specification. It is the only model to pass the Kupiec POF test at the 99% confidence level, and achieves the lowest AIC and BIC values among all candidates — indicating superior fit with appropriate complexity. Its asymmetric log-variance formulation better captures the leverage effect characteristic of equity index returns.

---

## Requirements

```bash
pip install arch numpy pandas yfinance matplotlib scipy statsmodels
```

| Package | Purpose |
|---|---|
| `arch` | GARCH, GJR-GARCH, and EGARCH model estimation |
| `yfinance` | Market data download |
| `scipy.stats` | t-distribution quantiles and chi-squared test |
| `statsmodels` | Ljung-Box diagnostic test |
| `matplotlib` | Visualisation |

---

## Usage

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
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779–1801.
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347–370.
- Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2), 73–84.
- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk.*
