# 

import sys
from pathlib import Path

# Ajoute le dossier api au PYTHONPATH
API_DIR = Path(__file__).parent
sys.path.insert(0, str(API_DIR))

from fastapi import FastAPI, HTTPException
from schema import PredictRequest, PredictResponse
from logique_metier import BestModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Best Model API")

# Charger le meilleur modèle une fois au démarrage
best_model = BestModel()


# ⭐ AJOUTEZ CES LIGNES ⭐
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": f"Bonjour ! Voici le meilleur modèle sélectionné pour vous : Random_forest"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Déterminer les features attendues par le modèle
    if best_model.model_type == "ML":
        expected_features = best_model.feature_list
    elif best_model.model_type == "RL":
        expected_features = best_model.features
    else:
        expected_features = None  # Rule et Random n'ont pas de features fixes

    # ⚠️ VALIDATION SUPPRIMÉE - on accepte n'importe quel nombre de features
    # L'API utilisera seulement celles dont elle a besoin
    
    # Créer une Series Pandas en associant les valeurs aux colonnes attendues
    if expected_features:
        # Si on reçoit plus de features que nécessaire, on prend seulement les N premières
        # Si on en reçoit moins, on complète avec des 0
        feature_values = request.features[:len(expected_features)] if len(request.features) >= len(expected_features) else request.features + [0] * (len(expected_features) - len(request.features))
        df_row = pd.Series(data=feature_values, index=expected_features)
    else:
        # pour Rule et Random, créer une Series vide
        df_row = pd.Series(dtype=float)

    prediction = best_model.predict(df_row)
    return {"prediction": prediction, "model": best_model.model_type}
