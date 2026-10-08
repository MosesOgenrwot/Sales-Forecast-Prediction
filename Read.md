# Sales Forecast Prediction using Python & XGBoost

A machine learning project demonstrating end-to-end time series sales forecasting using Python, pandas, and XGBoost. The model generates lag-based temporal features from historical sales transactions to predict aggregate daily demand.

---

## Executive Summary & Key Findings

* **Predictive Accuracy:** The XGBoost model achieved a **Root Mean Squared Error (RMSE) of 735.58** on unseen test data. Given the high scale and variance of total aggregate daily sales, this deviation demonstrates strong predictive capabilities.
* **Temporal Patterns:** Feature engineering via **5-day lagged sales** effectively captured short-term temporal dependencies without introducing data leakage.
* **Model Convergence:** XGBoost ($100$ estimators, learning rate $0.1$, tree depth $5$) proved highly effective at fitting non-linear daily sales fluctuations without severe overfitting.

---

## Project Setup & Installation

Install all required dependencies via `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost