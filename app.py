import numpy as np
from typing import Optional,List
from fastapi import FastAPI
from pydantic import BaseModel
import imblearn
import lightgbm
import pickle
import pandas as pd
import shap
import uvicorn  

# Charger la pipeline enregistrée
pipeline_path = "model.pkl"
with open(pipeline_path, 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Créer une instance FastAPI
app = FastAPI()

# Définir la classe de données d'entrée
class InputData(BaseModel):
    SK_ID_CURR: int
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float
    EXT_SOURCE_3: float
    DAYS_ID_PUBLISH: float
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    EXT_SOURCE_2: float
    EXT_SOURCE_1: float
    NAME_EDUCATION_TYPE_Highereducation: int
    NAME_CONTRACT_TYPE_Cashloans: int
    HOUR_APPR_PROCESS_START: int
    NAME_FAMILY_STATUS_Married: int
    FLAG_PHONE: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    DAYS_REGISTRATION: float
    INCOME_CREDIT_PERC: float
    FLAG_DOCUMENT_3: int
    EMERGENCYSTATE_MODE_No: int
    WALLSMATERIAL_MODE_Panel: int

# Fonction de prétraitement pour supprimer 'SK_ID_CURR' (si nécessaire)
def preprocess_input(input_data):
    # Supprimer la colonne 'SK_ID_CURR'
    del input_data['SK_ID_CURR']
    return input_data

# Modifier la fonction predict pour retourner SK_ID_CURR avec la prédiction
def predict(input_data, threshold=0.55):
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_input(input_df)
    prediction_proba = loaded_pipeline.predict_proba(input_df)[:, 1]
    prediction = (prediction_proba > threshold).astype(int)
    return int(input_data.get('SK_ID_CURR', -1)), int(prediction[0])
    
# Route pour la page d'accueil
@app.get("/")
async def root():
    return {"message": "Welcome to your FastAPI application!"}
    
# Modifier la route de prédiction pour retourner SK_ID_CURR avec la prédiction
@app.post("/predict/")
async def get_prediction(input_data: InputData, threshold: float = 0.52):
    sk_id_curr, prediction_result = predict(input_data.dict(), threshold)
    return {"SK_ID_CURR": sk_id_curr, "prediction": prediction_result}

@app.post("/predict_proba/")
def predict_proba(input_data: List[InputData]):
    # Convertir les données d'entrée en DataFrame pandas
    input_df = pd.DataFrame([record.dict() for record in input_data])
    
    # Supprimer la colonne SK_ID_CURR
    input_df.drop(columns=['SK_ID_CURR'], inplace=True)
    
    # Obtenir les probabilités prédites du modèle chargé
    predicted_proba = loaded_pipeline.predict_proba(input_df)[:, 1]  # Utilisez [:, 1] pour obtenir les probabilités de la classe positive
    
    # Récupérer les identifiants SK_ID_CURR
    sk_id_curr_list = [record.SK_ID_CURR for record in input_data]
    
    # Créer la réponse formatée
    formatted_response = {"SK_ID_CURR": sk_id_curr_list, "predicted_proba": predicted_proba.tolist()}
    
    return formatted_response
@app.post("/shap/")
def shap_analysis(input_data: List[InputData]):
    # Convertir les données d'entrée en DataFrame pandas
    input_df = pd.DataFrame([record.dict() for record in input_data])
    
    # Supprimer la colonne SK_ID_CURR
    input_df.drop(columns=['SK_ID_CURR'], inplace=True)
    
    # Extraire le modèle spécifique de la pipeline pour SHAP
    loaded_model_LGBMC2 = loaded_pipeline.named_steps['classifier']
    
    # Créer l'explainer SHAP avec le modèle extrait
    explainer2 = shap.TreeExplainer(loaded_model_LGBMC2)
    
    # Calculer les valeurs SHAP
    shap_values = explainer2.shap_values(input_df)
    
    # Récupérer les noms de variables
    feature_names = list(input_df.columns)
    
    # Récupérer les identifiants SK_ID_CURR
    sk_id_curr_list = [record.SK_ID_CURR for record in input_data]
    
    # Créer la réponse formatée
    formatted_response = []
    for i in range(len(input_data)):
        shap_dict = {"SK_ID_CURR": sk_id_curr_list[i], "Feature Names and SHAP Values": {}}
        for j, feature_name in enumerate(feature_names):
            shap_dict["Feature Names and SHAP Values"][feature_name] = shap_values[i][j]
        formatted_response.append(shap_dict)
    
    return formatted_response


# Exécuter l'application FastAPI avec uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



