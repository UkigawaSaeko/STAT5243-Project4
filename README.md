# STAT5243-Project4

End-to-end machine learning project for STAT 5243 using a chocolate sales dataset.

## Main project artifact

- `Project4_Chocolate_Sales.ipynb`: core workflow covering data preparation, EDA, feature engineering, supervised models, and model comparison.

## Bonus feature: Interactive dashboard

This repository includes an optional web application (bonus feature from Project 4):

- `streamlit_app.py`: interactive dashboard with:
  - filterable KPI and EDA views,
  - unsupervised segment-style visualization,
  - supervised model playground for quantity prediction.

## How to run

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the dashboard:
   - `streamlit run streamlit_app.py`
3. Open the local URL shown in your terminal (usually `http://localhost:8501`).

## Free web deployment options

### Option A: Streamlit Community Cloud (recommended, free)

1. Push this project to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and select:
   - Repository: your GitHub repo
   - Branch: your branch (usually `main`)
   - Main file path: `app.py` (or `streamlit_app.py`)
4. Deploy.

The platform auto-installs dependencies from `requirements.txt`.

### Option B: Hugging Face Spaces (free tier)

1. Create a new Space using **Streamlit** SDK.
2. Upload this project files to the Space.
3. Ensure `app.py` is present as entrypoint and `requirements.txt` is included.
4. Space builds automatically and hosts your app publicly.
