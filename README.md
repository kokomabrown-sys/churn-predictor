# Customer Churn Predictor (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
```bash
git init
git add .
git commit -m "Churn predictor"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
Then at https://share.streamlit.io: **Create app** → select the repo, branch `main`, file `app.py` → **Deploy**.

`scikit-learn` is pinned to 1.6.1, the version the model was trained with. Pickled models can break on other versions.
