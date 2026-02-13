# 

from fastapi import FastAPI, HTTPException
from schema import PredictRequest, PredictResponse
from logique_metier import BestModel
import pandas as pd

app = FastAPI(title="Best Model API")

# Charger le meilleur modèle une fois au démarrage
best_model = BestModel()


@app.get("/")
def read_root():
    return {
        "message": f"Bonjour ! Voici le meilleur modèle sélectionné pour vous : {best_model.model_type}"
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

    # Vérifier que le nombre de valeurs correspond
    if expected_features and len(request.features) != len(expected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu {len(expected_features)}, reçu {len(request.features)}"
        )

    # Créer une Series Pandas en associant les valeurs aux colonnes attendues
    if expected_features:
        df_row = pd.Series(data=request.features, index=expected_features)
    else:
        # pour Rule et Random, créer une Series vide
        df_row = pd.Series(dtype=float)

    prediction = best_model.predict(df_row)
    return {"prediction": prediction, "model": best_model.model_type}
