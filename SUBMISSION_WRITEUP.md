# STAT 5243 Project 4 Submission Writeup

**Project Title:** End-to-End Machine Learning on Chocolate Sales (2023-2024)  
**Course:** STAT 5243  
**Team Members:** Freya Chen (yc4684), Tong An (ta2829), Zhe Lin (zl3613), Jason Qin (jq2394)
**Submission Date:** May 4, 2026 
**Repository:** https://github.com/UkigawaSaeko/STAT5243-Project4
**Bonus web app:** https://stat5243-project4.streamlit.app/

---

## Executive Summary

This project implements a complete data science pipeline on a large transactional chocolate sales dataset (~1M orders) by integrating five relational tables (`sales`, `products`, `stores`, `customers`, `calendar`).  

We:
- cleaned and merged multi-table data,
- performed EDA with unsupervised analysis (clustering/PCA-style structure checks),
- engineered leakage-aware features,
- trained and compared supervised models for predicting order `quantity`,
- selected and retrained a final model,
- and delivered an interactive dashboard/web app as the optional bonus extension.

Main result: all supervised models performed similarly to the mean baseline (RMSE ~1.414), indicating limited incremental predictive signal for `quantity` beyond current features.

---

## 1) Data Collection & Preparation

### Data source and complexity
- Dataset: Chocolate Sales Dataset (2023-2024) from Kaggle.
- Tables used:
  - `sales.csv` (fact table; transaction-level)
  - `products.csv`
  - `stores.csv`
  - `customers.csv`
  - `calendar.csv`
- Size: ~1,000,000 transaction rows in the main table, plus four dimension tables.

### Preparation workflow
- Standardized column names and date formats (`order_date`, `join_date`, `date`).
- Checked duplicates and missingness at table level and post-merge.
- Performed key-based merges:
  - `sales` + `products` on `product_id`
  - `+ stores` on `store_id`
  - `+ customers` on `customer_id`
  - `+ calendar` on `order_date = date`
- Retained only modeling-safe rows where required model fields exist.

### Why this meets rubric expectations
- The project handles multi-table integration and realistic transactional data rather than a toy flat file.
- Data quality checks and preparation decisions are explicitly documented in notebook Section 1.

---

## 2) Exploratory Data Analysis (EDA) + Unsupervised Learning

### Core EDA analyses
- Time trends:
  - Revenue by date/month
- Segment and business cuts:
  - Top products by revenue
  - Revenue by country
  - Revenue by loyalty membership
  - Average revenue by age
- Correlation checks among numeric variables (`quantity`, `unit_price`, `discount`, `revenue`, `profit`)

### Unsupervised component
- KMeans-style clustering on selected transaction/customer features.
- PCA-oriented dimensionality checks to inspect feature structure and reduce geometric redundancy in visualization.

### Key EDA takeaway
- `quantity` is bounded/discrete and centered near ~3 with SD ~1.41, which already suggests a strong constant-mean baseline.
- Revenue/profit are mechanically tied to quantity/price/discount, so careful target/feature choices were needed to avoid trivial learning.

---

## 3) Data Preprocessing

### Methods used
- Missing-value handling and row filtering for required modeling fields.
- Date parsing and derived calendar features.
- One-hot encoding for categorical variables.
- Train-test split before scaling.
- Standardization fit on `X_train` only, then applied to `X_test` (leakage control).

### Leakage control decisions
- Did **not** use deterministic leakage columns when predicting `quantity`:
  - `revenue`, `cost`, `profit` excluded for supervised training because they are directly derived from quantity and price terms.

---

## 4) Feature Engineering

### Engineered features
- Temporal features from order date/calendar:
  - `year`, `month`, `day`, `week`, `day_of_week`
- Business/customer/product context:
  - customer demographics (`age`, `gender`, `loyalty_member`)
  - product descriptors (`brand`, `category`, `cocoa_percent`, `weight_g`)
  - store descriptors (`store_type`, `country`)
- Encoded categorical features with dummy variables.

### Rationale
- Feature set balances interpretability and predictive utility while respecting leakage constraints.
- Engineering reflects EDA findings and business relevance (seasonality, customer profile, product/store mix).

---

## 5) Supervised Modeling

### Prediction setup
- Task: supervised regression  
- Target: `quantity` (expected units per order)
- Validation design: common held-out 20% test split + CV/hyperparameter search where applicable.

### Models trained
1. **Baseline:** training-set mean predictor  
2. **Ridge Regression (`RidgeCV`)**  
3. **Random Forest Regressor** (tuned on subsample, refit on full train)  
4. **LightGBM / HistGradientBoosting fallback**  
5. **Stacking Regressor** (Ridge + RF + boosting base learners, Ridge meta-learner)

### Why this meets rubric expectations
- Includes >=3 distinct supervised models.
- Uses validation and tuning strategy appropriate to scale (~1M rows).
- Includes both performance and computational tradeoff reporting.

---

## 6) Model Evaluation & Selection

### Metrics
- RMSE (primary), MAE, R², training time, and RMSE lift vs baseline.

### Test-set summary (from notebook)

| Model | RMSE | MAE | R² | Train time (s) | Lift vs baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.4139 | 1.2004 | ~0 | ~0 | 0% |
| Ridge | 1.4139 | 1.2004 | ~-0.0001 | ~11.7 | ~0% |
| Random Forest | 1.4140 | 1.2005 | ~-0.0001 | ~90.2 | ~0% |
| LightGBM | 1.4142 | 1.2019 | ~-0.0004 | ~19.4 | ~-0.02% |
| Stacking | 1.4139 | 1.1995 | ~0 | ~23.7 | ~0% |

### Selection logic
- Rule-based choice in notebook:
  - rank by RMSE after removing baseline,
  - prefer simpler model when differences are negligible,
  - consider robustness + deployment cost.
- Since all models are effectively tied, a simpler model is justified for deployment.

### Residual diagnostics
- Residuals centered near zero with expected discrete-target pattern (`quantity` is integer-valued).
- Diagnostics support conclusion that current feature space provides limited extra signal over baseline.

---

## 7) Communication & Interpretation

### Narrative conclusions
- The pipeline is reproducible end-to-end and mirrors real practice: ingestion -> cleaning -> EDA -> feature design -> modeling -> comparison -> deployment artifact.
- Strong practical insight: model complexity did not meaningfully improve error relative to naive baseline, which is an important negative result and informs future data strategy.

### Business interpretation
- For `quantity` prediction, current variables mostly support mean-level forecasting.
- Highest leverage for future gains likely comes from richer demand drivers (promotions, stockouts, competitor pricing, holiday/event data, customer history windows).

### Reproducibility
- Code and dependencies are documented in repository (`README.md`, `requirements.txt`).
- Notebook + app are runnable locally and suitable for demonstration.

---

## 8) Creativity & Depth of Analysis

### Added depth beyond minimum requirements
- Multi-table integration at scale (~1M rows).
- Leakage-aware reframing from `revenue` to `quantity`.
- Multiple model families plus stacking.
- Feature-importance and residual diagnostic analysis.
- Runtime vs performance tradeoff discussion.
- Optional interactive app for communicating findings.

---

## Optional Bonus Artifact (Web App)

The repository includes a deployable interactive dashboard:
- `streamlit_app.py` (main app)
- `app.py` (deployment entrypoint)
- `.streamlit/config.toml`

The website is deployed to a streamlit server. 

This bonus artifact presents workflow outputs and a supervised prediction playground in a user-friendly interactive format.

---

## Team Contributions


- **Tong An:** table merging, cleaning pipeline, EDA, clustering/PCA analysis, Feature engineering
- **Freya Chen:** data aquisition, Feature engineering, Presentation, supervised model training/tuning  
- **Zhe Lin:** data aquisition, presentation, and writeup
- **Jason Qin:** presentation, writeup, web application

---

## Limitations and Future Work

### Limitations
- Limited uplift beyond baseline for `quantity`.
- Potentially missing key causal demand signals (promotions, inventory, campaign exposure, macro factors).
- Cross-sectional transaction modeling may underuse customer-level sequence/history effects.

### Future work
- Add lag/history features and customer-product interaction histories.
- Use time-aware validation for forecasting-style robustness.
- Include external event/holiday/promotions data.
- Evaluate probabilistic models or count models for discrete demand.

