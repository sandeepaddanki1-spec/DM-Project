"""Paths and constants. Nothing here is a model result; everything modelled is read from the artifacts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"

DATASET_PATH = DATA_DIR / "food_wastage_data.csv"
METRICS_PATH = DATA_DIR / "model_metrics.csv"
REG_PRED_PATH = DATA_DIR / "regression_predictions.csv"
CLS_PRED_PATH = DATA_DIR / "classification_predictions.csv"
KMEANS_PATH = DATA_DIR / "kmeans_clusters.csv"
HIER_PATH = DATA_DIR / "hierarchical_clusters.csv"
DBSCAN_PATH = DATA_DIR / "dbscan_clusters.csv"
DBSCAN_NOISE_PATH = DATA_DIR / "dbscan_noise_records.csv"

TARGET_COL = "Wastage Food Amount"
CATEGORY_ORDER = ["Low", "Medium", "High"]  # labels used in the notebook's pd.qcut(q=3)
CATEGORY_COLORS = {"Low": "#4F9D7F", "Medium": "#E0A526", "High": "#C8553D"}

# scikit-learn version recorded inside the .joblib files (found in the pickle bytes)
SAVED_SKLEARN_VERSION = "1.6.1"

# Evaluation protocol read from the final notebook cell
TEST_SIZE = 0.2
RANDOM_STATE = 42

APP_TITLE = "Restaurant Food Wastage Prediction & Management System"
APP_SUBTITLE = "AI-powered prediction and decision support for reducing restaurant food waste."
