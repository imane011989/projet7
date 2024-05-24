**Introduction** 

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.). De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner. Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

**Les données**

Les données utilisées sont disponibles à cette adresse : https://www.kaggle.com/c/home-credit-default-risk/data

**La mission**

Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
Mettre en production le modèle de scoring de prédiction à l’aide d’une API

**Organisation**

notebooks 
 
cleaning.ipynb # Notebook pour l'exploration des données
modelistaion.ipynb # Notebook pour l'entraînement du modèle
test_api.ipynb # Notebook pour faire des prédictions avec le modèle entraîné via api 

data_drift_mlflow

data_drif.html # Rapport généré automatiquement sur la dérive des données
captures_ecran # Dossier contenant des captures d'écran des différentes expériences suivies avec MLflow

**Utilisation**

https://github.com/imane011989/projet7
https://modelia.azurewebsites.net/

