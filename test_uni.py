
import pytest
from fastapi.testclient import TestClient
import api

client = TestClient(api)

# Test de la route "/predict/"
def test_predict():
    # Entrée de test
    test_input = {
          "SK_ID_CURR": 123,
            "DAYS_EMPLOYED": 1000,
            "DAYS_BIRTH": 12000,
            "EXT_SOURCE_3": 0.7,
            "DAYS_ID_PUBLISH": 2000,
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_1": 0.5,
            "NAME_EDUCATION_TYPE_Highereducation": 1,
            "NAME_CONTRACT_TYPE_Cashloans": 1,
            "HOUR_APPR_PROCESS_START": 10,
            "NAME_FAMILY_STATUS_Married": 1,
            "FLAG_PHONE": 0,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "DAYS_REGISTRATION": 3000,
            "INCOME_CREDIT_PERC": 0.5,
            "FLAG_DOCUMENT_3": 1,
            "EMERGENCYSTATE_MODE_No": 1,
            "WALLSMATERIAL_MODE_Panel": 1
    }
    response = client.post("/predict/", json=test_input)
    assert response.status_code == 200
    data = response.json()
    assert "SK_ID_CURR" in data
    assert "prediction" in data
    # Vérifie que SK_ID_CURR dans la réponse correspond à celui de l'entrée
    assert data["SK_ID_CURR"] == test_input["SK_ID_CURR"]
    # Vérifie que la prédiction est soit 0 ou 1
    assert data["prediction"] in [0, 1]

# Test de la route "/predict_proba/"
def test_predict_proba():
    # Entrée de test avec plusieurs enregistrements
    test_input = [
        {
              "SK_ID_CURR": 123,
            "DAYS_EMPLOYED": 1000,
            "DAYS_BIRTH": 12000,
            "EXT_SOURCE_3": 0.7,
            "DAYS_ID_PUBLISH": 2000,
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_1": 0.5,
            "NAME_EDUCATION_TYPE_Highereducation": 1,
            "NAME_CONTRACT_TYPE_Cashloans": 1,
            "HOUR_APPR_PROCESS_START": 10,
            "NAME_FAMILY_STATUS_Married": 1,
            "FLAG_PHONE": 0,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "DAYS_REGISTRATION": 3000,
            "INCOME_CREDIT_PERC": 0.5,
            "FLAG_DOCUMENT_3": 1,
            "EMERGENCYSTATE_MODE_No": 1,
            "WALLSMATERIAL_MODE_Panel": 1
        },
        {
             "SK_ID_CURR": 123,
            "DAYS_EMPLOYED": 1000,
            "DAYS_BIRTH": 12000,
            "EXT_SOURCE_3": 0.7,
            "DAYS_ID_PUBLISH": 2000,
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_1": 0.5,
            "NAME_EDUCATION_TYPE_Highereducation": 1,
            "NAME_CONTRACT_TYPE_Cashloans": 1,
            "HOUR_APPR_PROCESS_START": 10,
            "NAME_FAMILY_STATUS_Married": 1,
            "FLAG_PHONE": 0,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "DAYS_REGISTRATION": 3000,
            "INCOME_CREDIT_PERC": 0.5,
            "FLAG_DOCUMENT_3": 1,
            "EMERGENCYSTATE_MODE_No": 1,
            "WALLSMATERIAL_MODE_Panel": 1
        }
    ]
    response = client.post("/predict_proba/", json=test_input)
    assert response.status_code == 200
    data = response.json()
    assert "SK_ID_CURR" in data
    assert "predicted_proba" in data
    assert len(data["SK_ID_CURR"]) == len(test_input)
    assert len(data["predicted_proba"]) == len(test_input)

# Test de la route "/shap/"
def test_shap_analysis():
    # Entrée de test avec un seul enregistrement
    test_input = [
        {
              "SK_ID_CURR": 123,
            "DAYS_EMPLOYED": 1000,
            "DAYS_BIRTH": 12000,
            "EXT_SOURCE_3": 0.7,
            "DAYS_ID_PUBLISH": 2000,
            "CODE_GENDER": 1,
            "FLAG_OWN_CAR": 0,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_1": 0.5,
            "NAME_EDUCATION_TYPE_Highereducation": 1,
            "NAME_CONTRACT_TYPE_Cashloans": 1,
            "HOUR_APPR_PROCESS_START": 10,
            "NAME_FAMILY_STATUS_Married": 1,
            "FLAG_PHONE": 0,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "DAYS_REGISTRATION": 3000,
            "INCOME_CREDIT_PERC": 0.5,
            "FLAG_DOCUMENT_3": 1,
            "EMERGENCYSTATE_MODE_No": 1,
            "WALLSMATERIAL_MODE_Panel": 1
        }
    ]
    response = client.post("/shap/", json=test_input)
    assert response.status_code == 200
    data = response.text
    # Vérifie que les valeurs SHAP sont présentes dans la réponse
    assert "SK_ID_CURR" in data
    assert "Feature Names and SHAP Values" in data
