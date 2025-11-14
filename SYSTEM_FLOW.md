# System Flow Diagram

## Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT LAYER                                │
│  • Raw customer data (tenure, charges, contract type, etc.)             │
│  • Can be: DataFrame, dict, or real-time stream                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING MODULE                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐       │
│  │ Handle Missing  │→ │ Encode Features  │→ │ Scale Features  │       │
│  │    Values       │  │  (Label Encoder) │  │ (StandardScaler)│       │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘       │
│  • Numeric: Fill with median                                            │
│  • Categorical: Fill with mode & encode                                 │
│  • Standardize all features                                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENSEMBLE MODELS                                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐   │
│  │    XGBoost        │  │     LightGBM      │  │ Neural Network   │   │
│  │  (Weight: 40%)    │  │   (Weight: 35%)   │  │  (Weight: 25%)   │   │
│  │                   │  │                   │  │                  │   │
│  │ • 100 estimators  │  │ • 100 estimators  │  │ • 4 layers       │   │
│  │ • Max depth: 6    │  │ • Max depth: 6    │  │ • 64→32→16→2     │   │
│  │ • Learning: 0.1   │  │ • Learning: 0.1   │  │ • Dropout: 0.3   │   │
│  │                   │  │                   │  │ • Softmax output │   │
│  └────────┬──────────┘  └────────┬──────────┘  └────────┬─────────┘   │
│           │                      │                       │              │
│           └──────────────────────┼───────────────────────┘              │
│                                  ▼                                       │
│                    ┌──────────────────────────┐                         │
│                    │  Weighted Aggregation    │                         │
│                    │  P = Σ(weight_i × prob_i)│                         │
│                    └──────────────────────────┘                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FINAL CLASSIFICATION                                │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  Prediction: Churn (1) or No Churn (0)                   │           │
│  │  Confidence: Probability score [0.0 - 1.0]               │           │
│  │  Individual Model Scores: XGB, LGB, NN predictions       │           │
│  └──────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Parallel Explainability Path

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SHAP EXPLAINABILITY MODULE                            │
│                         (Parallel Processing)                            │
│                                                                          │
│  Input Data → Split into Chunks                                         │
│                   │                                                      │
│         ┌─────────┼─────────┬─────────┬─────────┐                      │
│         ▼         ▼         ▼         ▼         ▼                      │
│    ┌────────┐┌────────┐┌────────┐┌────────┐                           │
│    │Worker 1││Worker 2││Worker 3││Worker 4│  (ThreadPoolExecutor)      │
│    └───┬────┘└───┬────┘└───┬────┘└───┬────┘                           │
│        │         │         │         │                                  │
│        │    Compute SHAP Values      │                                  │
│        │    (KernelExplainer)         │                                  │
│        │         │         │         │                                  │
│        └─────────┴─────────┴─────────┘                                  │
│                   │                                                      │
│                   ▼                                                      │
│         ┌──────────────────────┐                                        │
│         │  Aggregate Results   │                                        │
│         │  • Feature importance│                                        │
│         │  • Per-prediction    │                                        │
│         │  • Waterfall plots   │                                        │
│         └──────────────────────┘                                        │
│                                                                          │
│  Output: Top 5 contributing features with SHAP values                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Background PSI Monitoring

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PSI MONITORING SERVICE                                 │
│                   (Background Thread)                                    │
│                                                                          │
│  ┌────────────────────────────────────────────────┐                     │
│  │  Initialization: Fit reference distributions  │                     │
│  │  • Create bins for each feature                │                     │
│  │  • Calculate reference distribution            │                     │
│  └────────────────┬───────────────────────────────┘                     │
│                   │                                                      │
│                   ▼                                                      │
│  ┌────────────────────────────────────────────────┐                     │
│  │         Monitoring Loop (Every N seconds)      │                     │
│  │  ┌──────────────────────────────────────────┐  │                     │
│  │  │ 1. Get current data from data source     │  │                     │
│  │  └──────────────────┬───────────────────────┘  │                     │
│  │                     ▼                           │                     │
│  │  ┌──────────────────────────────────────────┐  │                     │
│  │  │ 2. Calculate PSI for each feature        │  │                     │
│  │  │    PSI = Σ(current - ref) × ln(c/r)     │  │                     │
│  │  └──────────────────┬───────────────────────┘  │                     │
│  │                     ▼                           │                     │
│  │  ┌──────────────────────────────────────────┐  │                     │
│  │  │ 3. Check against thresholds              │  │                     │
│  │  │    • PSI < 0.1  → Stable                 │  │                     │
│  │  │    • PSI < 0.25 → Warning ⚠️              │  │                     │
│  │  │    • PSI ≥ 0.25 → Critical 🚨            │  │                     │
│  │  └──────────────────┬───────────────────────┘  │                     │
│  │                     ▼                           │                     │
│  │  ┌──────────────────────────────────────────┐  │                     │
│  │  │ 4. Generate alerts if thresholds exceed  │  │                     │
│  │  │    • Log to console                      │  │                     │
│  │  │    • Store in history                    │  │                     │
│  │  │    • Send to dashboard                   │  │                     │
│  │  └──────────────────────────────────────────┘  │                     │
│  │                                                 │                     │
│  └─────────────────────────────────────────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Admin Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADMIN DASHBOARD                                   │
│                      (Flask Web Server)                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                     WEB INTERFACE                              │     │
│  │  ┌──────────────────────────────────────────────────────────┐ │     │
│  │  │  📊 SYSTEM STATUS                                         │ │     │
│  │  │  • Model Status: ✓ Loaded                               │ │     │
│  │  │  • Total Predictions: 1,234                             │ │     │
│  │  │  • Active Alerts: 2                                     │ │     │
│  │  └──────────────────────────────────────────────────────────┘ │     │
│  │                                                               │     │
│  │  ┌──────────────────────────────────────────────────────────┐ │     │
│  │  │  📈 CURRENT DRIFT STATUS                                 │ │     │
│  │  │  Overall: STABLE ✓                                      │ │     │
│  │  │                                                          │ │     │
│  │  │  Feature PSI Values:                                    │ │     │
│  │  │  ✓ tenure: 0.0123 (stable)                             │ │     │
│  │  │  ⚠ monthly_charges: 0.1523 (warning)                   │ │     │
│  │  │  ✓ contract_type: 0.0456 (stable)                      │ │     │
│  │  └──────────────────────────────────────────────────────────┘ │     │
│  │                                                               │     │
│  │  ┌──────────────────────────────────────────────────────────┐ │     │
│  │  │  🔔 RECENT ALERTS                                        │ │     │
│  │  │  [Refresh] button                                       │ │     │
│  │  │                                                          │ │     │
│  │  │  ⚠️ WARNING: monthly_charges                            │ │     │
│  │  │     PSI: 0.1523 | Time: 14:30:45                       │ │     │
│  │  │                                                          │ │     │
│  │  │  ⚠️ WARNING: payment_method                             │ │     │
│  │  │     PSI: 0.1156 | Time: 14:30:45                       │ │     │
│  │  └──────────────────────────────────────────────────────────┘ │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                       REST API                                 │     │
│  │  GET  /api/alerts              → Get all alerts               │     │
│  │  GET  /api/drift/status        → Current drift status         │     │
│  │  GET  /api/drift/history       → Historical drift data        │     │
│  │  GET  /api/predictions/history → Prediction log               │     │
│  │  POST /api/predict             → Make prediction              │     │
│  │  GET  /api/system/info         → System information           │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Auto-refresh: Every 30 seconds                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Complete System Integration

```
                     ┌──────────────────────┐
                     │   User / API Client  │
                     └──────────┬───────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
         ┌────────────────────┐    ┌──────────────────┐
         │  Prediction Request │    │  Dashboard Access│
         └──────────┬──────────┘    └────────┬─────────┘
                    │                        │
                    ▼                        ▼
    ┌──────────────────────────┐   ┌──────────────────────┐
    │  ChurnPredictionPipeline │   │   AdminDashboard     │
    │                          │   │   (Flask Server)     │
    │  • Preprocessing         │   │                      │
    │  • Ensemble Models       │◄──┤  • Status Display    │
    │  • SHAP Explainer        │   │  • Alert Management  │
    │  • PSI Monitor           │   │  • API Endpoints     │
    └───────┬──────────────────┘   └──────────────────────┘
            │                                │
            │                                │
            ▼                                ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Trained Models  │          │  Background      │
    │  • XGBoost       │          │  PSI Monitor     │
    │  • LightGBM      │          │  (Thread)        │
    │  • Neural Net    │          └──────────────────┘
    └──────────────────┘                   │
            │                              │
            └──────────┬───────────────────┘
                       │
                       ▼
            ┌────────────────────┐
            │  Response/Alerts   │
            │  • Prediction      │
            │  • Explanation     │
            │  • Drift Status    │
            └────────────────────┘
```

## Data Flow Example

```
Example: Customer wants to know churn risk

1. INPUT
   ┌────────────────────────────────────────┐
   │ customer = {                           │
   │   'tenure': 6,                         │
   │   'monthly_charges': 89.50,            │
   │   'contract_type': 'Month-to-month',   │
   │   'internet_service': 'Fiber optic',   │
   │   ...                                  │
   │ }                                      │
   └────────────────────────────────────────┘
                    │
                    ▼
2. PREPROCESSING
   ┌────────────────────────────────────────┐
   │ Encoded & Scaled:                      │
   │ [0.16, 0.89, 1.0, 2.0, ...]           │
   └────────────────────────────────────────┘
                    │
                    ▼
3. ENSEMBLE PREDICTION
   ┌────────────────────────────────────────┐
   │ XGBoost:  0.7898 (40% × 0.7898)       │
   │ LightGBM: 0.8023 (35% × 0.8023)       │
   │ NeuralNet: 0.7808 (25% × 0.7808)      │
   │ ─────────────────────────────────      │
   │ FINAL: 0.7919 (79.19% churn risk)     │
   └────────────────────────────────────────┘
                    │
                    ├─────────────────────────┐
                    │                         │
                    ▼                         ▼
4. EXPLANATION              5. DRIFT CHECK
   ┌──────────────────┐        ┌──────────────────┐
   │ Top Features:    │        │ PSI Check:       │
   │ • tenure: +0.45  │        │ • All stable ✓   │
   │ • contract: +0.32│        │ • No alerts      │
   │ • charges: +0.28 │        └──────────────────┘
   └──────────────────┘
                    │
                    ▼
6. OUTPUT
   ┌────────────────────────────────────────┐
   │ {                                      │
   │   "prediction": 1,                     │
   │   "churn_probability": 0.7919,         │
   │   "model_predictions": {               │
   │     "xgboost": 0.7898,                 │
   │     "lightgbm": 0.8023,                │
   │     "neural_network": 0.7808           │
   │   },                                   │
   │   "top_features": [...]                │
   │ }                                      │
   └────────────────────────────────────────┘
```

## Deployment Flow

```
┌─────────────────────────────────────────────────────────────┐
│ DEVELOPMENT                                                 │
│  1. python main.py --demo          # Train models           │
│  2. python -m unittest discover    # Run tests              │
│  3. python example_usage.py        # Validate examples      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGING                                                     │
│  1. Load models: pipeline.load()                           │
│  2. Test with real data                                    │
│  3. Monitor PSI metrics                                    │
│  4. Launch dashboard                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ PRODUCTION                                                  │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ API Server      │  │ PSI Monitor  │  │ Dashboard     │ │
│  │ (Gunicorn)      │  │ (Background) │  │ (Nginx)       │ │
│  └─────────────────┘  └──────────────┘  └───────────────┘ │
│           │                   │                  │          │
│           └───────────────────┴──────────────────┘          │
│                              │                               │
│                   ┌──────────▼──────────┐                   │
│                   │   Load Balancer     │                   │
│                   └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

This diagram illustrates the complete architecture and data flow of the implemented system.
