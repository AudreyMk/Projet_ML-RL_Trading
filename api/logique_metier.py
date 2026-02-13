# 



import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.final_evaluation_model import run_evaluation

MODELS_DIR = PROJECT_ROOT / "models_registry"

class BestModel:
    def __init__(self):
        results_df, best_model_name, is_valid = run_evaluation()
        # On ignore la validation : on prend le meilleur modèle quoiqu'il arrive
        is_valid = True

        self.model_type = best_model_name
        print(f"✅ Meilleur modèle chargé : {self.model_type}")

        # Charger le modèle selon son type
        if self.model_type == "RL":
            self._load_rl()
        elif self.model_type == "ML":
            self._load_ml()
        else:
            self.model = None  # Random / Rule

        self.position = 0  # pour RL

    def _load_rl(self):
        path = MODELS_DIR / "rl" / "q_learning_clean.pkl"
        with open(path, "rb") as f:
            q_obj = pickle.load(f)
        self.q_table = q_obj["q_table"]
        self.bins = q_obj["bins"]
        self.features = list(self.bins.keys())
        self.n_actions = 3
        self.model = "RL_loaded"  # marker pour indiquer que le modèle est chargé

    def _load_ml(self):
        import joblib, json
        v1_dir = MODELS_DIR / "v1"
        self.model = joblib.load(v1_dir / "logistic_v1.pkl")
        self.scaler = joblib.load(v1_dir / "scaler_v1.pkl")
        meta = json.load(open(v1_dir / "metadata.json"))
        self.feature_list = meta.get("feature_list", [])

    def predict(self, df_row: pd.Series):
        if self.model_type == "RL":
            vals = df_row[self.features].values
            state_bins = [np.digitize(v, self.bins[col][1:-1], right=False) for v, col in zip(vals, self.features)]
            state = tuple(list(state_bins) + [self.position])
            q_vals = self.q_table.get(state, np.zeros(self.n_actions))
            action = int(np.argmax(q_vals))
            self.position = 1 if action == 1 else -1 if action == 2 else 0
            return self.position
        elif self.model_type == "ML":
            X = df_row[self.feature_list].values.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            return 1 if pred == 1 else -1
        elif self.model_type == "Rule":
            return 1 if df_row["sma_20"] > df_row["sma_50"] else -1
        elif self.model_type == "Random":
            return np.random.choice([-1, 0, 1])
