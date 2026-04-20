# Spirometry Dashboard Project

This repository contains multiple Python dashboard and model scripts for spirometry data analysis.

## Included files
- `combined_dashboard.py`
- `combined_dashboard_v2.py`
- `combined_dashboard_v3.py`
- `combined_dashboard_v4.py` (latest UI version)
- `spirometer_dashboard.py`
- `xgboost_spirometer_model.py`
- Dataset CSV files used by the dashboards

## 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

## 2. Create and activate a virtual environment
### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Run the app
Run the latest dashboard:
```bash
python combined_dashboard_v4.py
```

You can also run older/alternate scripts:
```bash
python combined_dashboard_v3.py
python combined_dashboard_v2.py
python combined_dashboard.py
python spirometer_dashboard.py
python xgboost_spirometer_model.py
```

## 5. Notes for your friend
- Use Python 3.10+.
- If Tkinter is missing on Linux, install system package `python3-tk`.
- Keep CSV files in the project root unless you change script paths.
- If model training is slow, close other heavy apps and use the latest script (`combined_dashboard_v4.py`).

## 6. Updating from your side
After new changes:
```bash
git add .
git commit -m "Update project"
git push
```
