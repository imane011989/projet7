
import pytest
from fastapi.testclient import TestClient
from app import app

# Créer un client de test pour l'application FastAPI
client = TestClient(app)

# Définir les données de test pour chaque endpoint
test_input_data = {
    "SK_ID_CURR": 12345,
    "DAYS_EMPLOYED": -2000,
    "DAYS_BIRTH": -15000,
    "EXT_SOURCE_3": 0.7,
    "DAYS_ID_PUBLISH": -1000,
    "CODE_GENDER": 0,
    "FLAG_OWN_CAR": 1,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_1": 0.5,
    "NAME_EDUCATION_TYPE_Highereducation": 1,
    "NAME_CONTRACT_TYPE_Cashloans": 0,
    "HOUR_APPR_PROCESS_START": 10,
    "NAME_FAMILY_STATUS_Married": 1,
    "FLAG_PHONE": 1,
    "AMT_INCOME_TOTAL": 50000,
    "AMT_CREDIT": 200000,
    "DAYS_REGISTRATION": -2000,
    "INCOME_CREDIT_PERC": 0.25,
    "FLAG_DOCUMENT_3": 1,
    "EMERGENCYSTATE_MODE_No": 1,
    "WALLSMATERIAL_MODE_Panel": 1
}

# Tester la route d'accueil
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to your FastAPI application!"}

# Tester la route de prédiction
def test_predict():
    response = client.post("/predict/", json=test_input_data)
    assert response.status_code == 200
    assert "SK_ID_CURR" in response.json()
    assert "prediction" in response.json()

# Tester la route de prédiction de probabilités
def test_predict_proba():
    response = client.post("/predict_proba/", json=[test_input_data])
    assert response.status_code == 200
    assert "SK_ID_CURR" in response.json()
    assert "predicted_proba" in response.json()

# Tester la route d'analyse SHAP
def test_shap_analysis():
    response = client.post("/shap/", json=[test_input_data])
    assert response.status_code == 200
    assert "SK_ID_CURR" in response.text
    assert "Feature Names and SHAP Values" in response.text
