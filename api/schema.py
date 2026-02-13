from pydantic import BaseModel
from typing import List

# class PredictRequest(BaseModel):
#     data: List[dict]  # Chaque dict correspond à une ligne de features

# class PredictResponse(BaseModel):
#     signals: List[int]  # -1, 0 ou 1 pour chaque ligne


from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    features: List[float]  # liste de valeurs pour toutes les features

class PredictResponse(BaseModel):
    prediction: int        # -1, 0, 1
    model: str             # type de modèle utilisé
