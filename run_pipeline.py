"""One-command pipeline: audit/EDA -> forecasting (incl. LSTM) -> routing experiments -> research report.
Usage:  python run_pipeline.py [--skip-train] [--resume] [--epochs 30]
"""
import argparse

from src import config, eda, report, routing_evaluation, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    config.setup_logging()
    eda.run()
    if not a.skip_train:
        train.main(epochs=a.epochs, resume=a.resume)
    routing_evaluation.run_all()
    report.write()
    print("Done. Start the app with:  streamlit run app.py")
