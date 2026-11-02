# Open in VS Code
1. File > Open Folder... > select this `ai-traffic-forecasting` folder.
2. Open a terminal (Ctrl+`) and run (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
(If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`.)
3. Results from my run are already included (reports/, data/processed/, models/*.pt), so every page works immediately.
4. To retrain the models yourself: `python run_pipeline.py --epochs 30`. To test: `pytest`.
Or use Run and Debug (Ctrl+Shift+D): "Streamlit app", "Run full pipeline", "Run tests".
Your dataset is already at data/raw/traffic.csv.
