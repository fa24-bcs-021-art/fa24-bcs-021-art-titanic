# Titanic Survival Predictor (FA24-BCS-021)

Files: app.py (Streamlit app, trains model on startup), titanic.csv (data), requirements.txt, titanic_submission.ipynb.

## Deploy (Streamlit Community Cloud)
1. Create a PUBLIC GitHub repo and upload app.py, titanic.csv, requirements.txt (repo root).
2. Go to share.streamlit.io -> Create app -> pick the repo, branch main, main file app.py.
3. Open Advanced settings -> App URL, and enter: fa24-bcs-021-titanic
4. Deploy. Live URL: https://fa24-bcs-021-titanic.streamlit.app

Run locally: pip install -r requirements.txt && streamlit run app.py
