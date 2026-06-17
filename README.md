# Birthrate AI Platform

Run the analysis script from the project virtual environment:

```powershell
.\.venv\Scripts\python.exe fetch_birthrate.py
```

The script fetches World Bank birth-rate and population data, prints a short
analysis, saves the analysed dataset as `NGA_birth_rate_analysis.csv`, and saves
a visualisation as `NGA_birth_rate_analysis.png`.

If you are using a fresh environment, install the dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Run the web app:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
