# Customer Retention Engine

Customer Retention Engine is an end-to-end churn intelligence project that predicts whether a customer will churn and recommends the smallest feasible intervention to retain that customer.

It includes:
- A **FastAPI backend** for prediction and recommendation endpoints.
- A **Streamlit frontend** for interactive scoring.
- Pretrained **scikit-learn model artifacts** and preprocessing pipeline.
- Notebooks and a full report covering EDA, modeling, and evaluation.

## Features

- Predicts churn (`Churn` / `Not Churn`) with probability scores.
- Returns actionable retention plans with one of four statuses:
  - `NO_ACTION`
  - `SINGLE_CHANGE`
  - `MULTI_CHANGE`
  - `HIGH_RISK`
- Uses a counterfactual-style recommendation engine to find minimal feature changes.

## Project Structure

```text
Customer-Retention-Engine/
├─ backend/
│  └─ app/
│     ├─ main.py          # FastAPI app and routes
│     ├─ schemas.py       # Request/response models
│     └─ service.py       # Model inference and recommendation logic
├─ frontend/
│  └─ streamlit_app.py    # Streamlit UI
├─ data/
│  └─ customer_churn_dataset-training-master.csv
├─ models/
│  ├─ best_model.joblib
│  ├─ preprocessing_pipeline.joblib
│  ├─ column_info.joblib
│  └─ other trained models
├─ notebooks/
│  ├─ experiment.ipynb    # Training and model comparison
│  └─ recommend.ipynb     # Recommendation logic experiments
├─ figures/
├─ report/
│  └─ main.tex            # Full technical report
└─ requirements.txt
```

## Tech Stack

- Python
- scikit-learn, pandas, numpy
- FastAPI + Uvicorn
- Streamlit
- joblib

## Setup

1. Create and activate a virtual environment.
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies.
   ```powershell
   pip install -r requirements.txt
   ```

## Run the Application

1. Start the backend API (from project root):
   ```powershell
   uvicorn backend.app.main:app --reload
   ```
2. Start the Streamlit frontend in a second terminal:
   ```powershell
   streamlit run frontend\streamlit_app.py
   ```
3. Open the UI:
   - Streamlit: `http://localhost:8501`
   - FastAPI docs: `http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info and selected model name |
| GET | `/health` | Health check |
| GET | `/metadata` | Categories and numeric ranges for UI/form generation |
| POST | `/predict` | Churn prediction + embedded recommendation |
| POST | `/recommend` | Recommendation only |

### Example Request (`POST /predict`)

```json
{
  "gender": "Female",
  "tenure": 27,
  "usage_frequency": 6,
  "support_calls": 10,
  "payment_delay": 20,
  "subscription_type": "Premium",
  "contract_length": "Quarterly",
  "total_spend": 300,
  "last_interaction": 14,
  "age_group": "Adult"
}
```

## Model Notes

- The report and notebooks compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Based on F1 score, **Random Forest** is selected as the production model and saved as `models\best_model.joblib`.

## Data

- Dataset file: `data\customer_churn_dataset-training-master.csv`
- Original source referenced in the report: Kaggle Customer Churn Dataset by Muhammad Shahid Azeem.

## Troubleshooting

- If `/metadata` is temporarily unavailable, the Streamlit app uses built-in fallback metadata to render the form.
- Prediction/recommendation requests still require the backend API to be running.
- If model files are missing in `models\`, regenerate them from the notebooks before starting the API.
