# DriftSense 🛰️
### An Integrated AI Framework for Industrial Risk Management

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen?style=flat&logo=pytest)](tests/)
[![Dissertation](https://img.shields.io/badge/Dissertation-MSRUAS_2025-1F3864?style=flat)](docs/dissertation.pdf)

> **Predicting Raw Material Price Regimes and Carbon Compliance Risk Using Concept Drift Detection** — undergraduate dissertation project, M.S. Ramaiah University of Applied Sciences (B.Sc. Hons. Data Science & Analytics, 2025)

---

## What Is This?

DriftSense is a **production-grade, open-source industrial risk intelligence system** built entirely on Python open-source tooling. It solves a problem that two separate teams — finance and environmental — in any raw-material-intensive manufacturer are currently solving badly and in isolation:

| Problem | Current State | DriftSense |
|---|---|---|
| Commodity price regime shifts | Gut feel + lagging procurement data | GMM + XGBoost classifier, 21-day forward signal |
| Carbon compliance risk | Annual tick-box reporting | Real-time Random Forest classifier with threshold proximity |
| Model degradation over time | Nobody notices until it's too late | ADWIN + Page-Hinkley drift detection with auto-recalibration |

The insight driving the architecture: **price regime transitions and carbon compliance stress are correlated.** When energy prices enter a crisis regime, energy-intensive processes simultaneously face higher input costs *and* increased emission intensity — a compound risk that siloed tools cannot surface. DriftSense monitors both on a single adaptive platform.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DriftSense Framework                      │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │  MODULE 1         │    │  MODULE 2         │    │  MODULE 3  │  │
│  │  Price Regime     │    │  Carbon           │    │  Concept   │  │
│  │  Predictor        │    │  Compliance       │    │  Drift     │  │
│  │                   │    │  Classifier       │    │  Detection │  │
│  │  GMM (unsup.)     │    │                   │    │            │  │
│  │  → regime labels  │    │  Random Forest    │    │  ADWIN     │  │
│  │  XGBoost (sup.)   │───▶│  200 trees        │    │  +         │  │
│  │  → 21-day fcast   │    │  calibrated probs │    │  Page-     │  │
│  │                   │    │  SMOTE balanced   │    │  Hinkley   │  │
│  └──────────────────┘    └──────────────────┘    └────────────┘  │
│           │                        │                    │          │
│           └────────────────────────┴────────────────────┘          │
│                                    │                               │
│                    ┌───────────────▼───────────────┐               │
│                    │     Streamlit Dashboard         │               │
│                    │  Price Regime Panel             │               │
│                    │  Compliance Risk Panel          │               │
│                    │  Drift Monitoring Panel         │               │
│                    └───────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Results Summary

> All results are from out-of-sample evaluation on held-out data (2022–2023 for price regime; full synthetic dataset with 2 drift events for compliance). Full methodology and numbers are in [the dissertation](docs/dissertation.pdf).

### Price Regime Prediction

| Model | Macro F1 | Regime Transition Precision | Recall |
|---|---|---|---|
| **Adaptive XGBoost (DriftSense)** | **0.81** | **0.78** | **0.82** |
| Static XGBoost (no drift response) | 0.67 | 0.58 | 0.54 |
| Naive persistence baseline | 0.52 | — | — |

The adaptive model issued 18 regime-change alerts over the 2022–2023 evaluation window; 14 were confirmed within a 21-day horizon. The static baseline missed the 2022 energy market crisis transition entirely until 142 trading days after onset; the adaptive model detected it in **68 trading days**.

### Carbon Compliance Classification

| Model | AUC-ROC | Non-compliant F1 |
|---|---|---|
| **Adaptive Random Forest (DriftSense)** | **0.89** | **0.81** |
| Static Random Forest (no recalibration) | 0.77 | 0.62 |

After the synthetic regulatory threshold drift at observation 2,500, the static model's AUC-ROC fell from 0.79 → 0.63. The adaptive model detected the drift at observation 2,547 (47-obs delay) and recovered to AUC-ROC 0.85 within 100 post-recalibration observations.

### Drift Detection Performance

| Detector | Use Case | Detection Delay | False Positive Rate |
|---|---|---|---|
| ADWIN (δ=0.002) | Gradual regime shifts | 68 trading days | 2.3% |
| Page-Hinkley (λ=50) | Abrupt threshold changes | 31 observations | 4.1% |

The two detectors are deliberately complementary: ADWIN handles gradual distribution change; Page-Hinkley handles sharp mean shifts. Running both in parallel provides coverage across the full range of drift types seen in industrial data.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | `yfinance`, EIA public API, Quandl API |
| ML — regime | `scikit-learn` (GMM), `xgboost`, `pandas`, `numpy` |
| ML — compliance | `scikit-learn` (RandomForest), `imbalanced-learn` (SMOTE) |
| Drift detection | `river` (ADWIN, Page-Hinkley streaming API) |
| Dashboard | `streamlit`, `plotly` |
| NLP (future) | `FinBERT` (sentiment signal, see roadmap) |
| Testing | `pytest`, `pytest-cov` |
| CI/CD | GitHub Actions |

---

## Repo Structure

```
DriftSense/
├── src/
│   ├── regime_predictor/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # EIA/Quandl ingestion, preprocessing
│   │   ├── feature_engineering.py  # Log-returns, rolling stats, RSI, MACD, BBW
│   │   ├── gmm_regime_labeller.py  # Unsupervised GMM, BIC selection
│   │   └── xgboost_classifier.py   # Rolling-window XGBoost, 21-day forecast
│   ├── compliance_classifier/
│   │   ├── __init__.py
│   │   ├── data_generator.py       # Synthetic BEE PAT calibrated dataset
│   │   ├── feature_engineering.py  # Threshold proximity, rolling trend, energy mix
│   │   └── rf_classifier.py        # Random Forest, SMOTE, calibrated probs
│   ├── drift_detection/
│   │   ├── __init__.py
│   │   ├── adwin_monitor.py        # ADWIN wrapper, drift event logger
│   │   ├── page_hinkley_monitor.py # Page-Hinkley wrapper, recalibration trigger
│   │   └── recalibration.py        # Auto-retrain protocols on drift signal
│   ├── dashboard/
│   │   ├── app.py                  # Streamlit entry point
│   │   ├── regime_panel.py         # Price regime visualisation panel
│   │   ├── compliance_panel.py     # Compliance risk panel
│   │   └── drift_panel.py          # Drift monitoring timeline panel
│   └── utils/
│       ├── config.py               # Hyperparameters, thresholds, API keys
│       └── logger.py               # Structured logging for drift events
├── tests/
│   ├── unit/
│   │   ├── test_feature_engineering.py
│   │   ├── test_gmm_labeller.py
│   │   ├── test_rf_classifier.py
│   │   └── test_drift_detectors.py
│   └── integration/
│       ├── test_pipeline_end_to_end.py
│       └── test_dashboard_data_contracts.py
├── data/
│   ├── raw/                        # Downloaded price series (gitignored)
│   ├── processed/                  # Feature-engineered datasets
│   └── synthetic/                  # Generated compliance dataset
├── notebooks/
│   ├── 01_regime_exploration.ipynb
│   ├── 02_compliance_features.ipynb
│   └── 03_drift_sensitivity_analysis.ipynb
├── docs/
│   └── dissertation.pdf            # Full dissertation (MSRUAS, May 2025)
├── scripts/
│   ├── generate_synthetic_data.py
│   └── run_evaluation.py
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint + pytest on push
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Chandan218-cell/DriftSense.git
cd DriftSense
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API keys (for live data)

```bash
cp .env.example .env
# Add your EIA API key and Quandl API key to .env
# Both are free: https://www.eia.gov/opendata/  |  https://data.nasdaq.com
```

### 3. Generate the synthetic compliance dataset

```bash
python scripts/generate_synthetic_data.py
# Outputs: data/synthetic/compliance_dataset.csv (5000 obs, BEE PAT calibrated)
```

### 4. Run the full evaluation pipeline

```bash
python scripts/run_evaluation.py
# Prints: F1, AUC-ROC, drift detection delay, false positive rates for all modules
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Unit tests only
pytest tests/unit/ -v
```

---

## Key Design Decisions (and why)

**Why GMM for regime identification, not HMM?**
Gaussian Mixture Models recover economically meaningful market states from unlabelled data without requiring a parametric state-transition structure. The BIC selection procedure (fitting GMMs from 2–6 components) chose 3 components consistently across seeds, and the resulting labels align with qualitatively recognisable market periods: the 2003–07 supercycle, the 2008 crisis, the 2014–16 oil glut, and the 2020 COVID shock. HMMs would impose Markov constraints on regime transitions that may not hold.

**Why Random Forest for compliance, not XGBoost or LSTM?**
Two reasons: (1) RF produces well-calibrated probability outputs that suit threshold-based alerting better than the raw outputs of gradient boosted trees; (2) RF retrains on 1,000 observations in under a second, making automated drift-response recalibration genuinely practical. LSTM networks would require GPU time for fine-tuning that prices the system out of the SMB deployment context it targets.

**Why run ADWIN and Page-Hinkley in parallel?**
They are complementary. ADWIN maintains an adaptive sliding window and is best suited to gradual distributional change. Page-Hinkley monitors cumulative deviation from a running mean and is more sensitive to abrupt mean shifts. Real industrial data contains both types. Running both and logging which detector fires first provides diagnostic information about the *nature* of the drift, not just its presence.

**Why Streamlit, not a React/FastAPI stack?**
This is a research prototype demonstrating feasibility. Streamlit is Python-native, requires no translation layer between model and presentation, and can be used by non-programmers. A production deployment would separate the model inference layer from the frontend, introduce session management, and connect to real-time data feeds. The modular architecture of DriftSense makes this a straightforward engineering task rather than a redesign.

---

## Roadmap

- [ ] **FinBERT sentiment integration** — add NLP signal from commodity news and regulatory announcements to the price regime module to reduce detection lag (identified as the most valuable extension in dissertation Chapter 5)
- [ ] **Aluminium / titanium price series** — replace WTI/copper with aerospace-relevant materials using LME data
- [ ] **Federated drift detection** — distribute the drift layer across multi-site industrial deployments without centralising raw data
- [ ] **Cross-module feature engineering** — feed price regime state as a feature input into the compliance classifier (empirical basis: Regime 3 observations were 34% more likely to be associated with elevated compliance risk)
- [ ] **FastAPI inference layer** — expose regime and compliance predictions as REST endpoints for integration with ERP/MES systems
- [ ] **Real PAT scheme validation** — partner with BEE or an industrial PAT-scheme member to validate the compliance module against actual emission monitoring data

---

## Industrial Context

This framework was designed around a real operational problem observed during an internship at **Motherson Aerospace / CIM Tools Private Limited** (Bengaluru), a Boeing Tier 1 and Airbus Tier 2 precision aerospace component manufacturer. The data gaps and analytical needs described in the dissertation — energy consumption data collected but never used for proactive compliance management, procurement decisions made without regime-awareness — are not hypothetical. The framework is calibrated to the BEE PAT scheme regulatory context and is directly transferable to aerospace materials (aluminium, titanium) with parameter re-calibration.

Full context and internship reflection: [docs/dissertation.pdf](docs/dissertation.pdf), Chapter 1 and Annexure.

---

## Academic Reference

> Chandan D. (2025). *An Integrated AI Framework for Industrial Risk Management: Predicting Raw Material Price Regimes and Carbon Compliance Risk Using Concept Drift Detection.* B.Sc. (Hons.) Dissertation. Department of Data Sciences and Analytics, M.S. Ramaiah University of Applied Sciences, Bengaluru.
> Supervisor: Dr. Sharmistha Rakshit.

Turnitin similarity: **6%** | 0 integrity flags | Submitted: May 2026

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Chandan D** — Data Science & Analytics, MSRUAS Bengaluru  
[GitHub](https://github.com/Chandan218-cell) · Open to roles in Data Science / ML Engineering / Analytics (Bengaluru, Hyderabad, Pune, Delhi NCR, Chennai)
