# %% [markdown]
# # Projet 7 : Implémentez un modèle de scoring

# %% [markdown]
# Suite du projet 7. Implémentation du modèle de scoring via API

# %% [markdown]
# ## Import des librairies

# %%
# File system manangement
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# Affichage en ligne des graphiques
# %matplotlib inline

# Options pour améliorer l'affichage des données
pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 100)
sns.set(style="whitegrid")

# %% [markdown]
# ## Chargement des fichiers Pickle et des Clients

# %%
import pickle

# Vérifier que le fichier pickle existe et n'est pas vide
pickle_file_path = 'best_lightgbm_model.pkl'
if os.path.exists(pickle_file_path):
    file_size = os.path.getsize(pickle_file_path)
    if file_size == 0:
        print("Le fichier pickle est vide.")
    else:
        print(f"Le fichier pickle a une taille de {file_size} octets.")
        print(__name__)
else:
    print("Le fichier pickle n'existe pas.")

# Charger le modèle LightGBM depuis le fichier pickle
with open('best_lightgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Vérifiez que le modèle est chargé
# print(model)

# Charger les données des clients (X) à partir du CSV
client_data = pd.read_csv('client_data.csv')
# print(client_data.sample())

# %%
client_data.num__SK_ID_CURR.sample(10)

# %% [markdown]
# ## Flask API

# %%
from flask import Flask, jsonify, request
import random

# Créer l'application Flask
app = Flask(__name__)

@app.route("/")
def home():
    random_client_id = random.choice(client_data['num__SK_ID_CURR'])
    return f"Bienvenue sur la page d'accueil! Un client ID : {random_client_id}"

@app.route('/predict', methods=['GET'])
def predict():
    # Récupérer l'ID du client passé en paramètre de la requête
    client_id = request.args.get('id', type=int)

    # Log pour déboguer
    print(f"Received client_id: {client_id}")

    # Vérifier si l'ID est manquant
    if client_id is None:
        return jsonify({'error': 'Missing client id'}), 400

    # Vérifier si l'ID du client existe dans les données
    if client_id not in client_data['num__SK_ID_CURR'].values:
        return jsonify({'error': f'Client ID {client_id} non trouvé'}), 404
    
    # Sélectionner les données du client
    client_info = client_data[client_data['num__SK_ID_CURR'] == client_id]

    # Effectuer la prédiction
    prediction = model.predict(client_info)

    # Effectuer la probabilité de classe
    prediction_proba = model.predict_proba(client_info)

    # Retourner la prédiction sous forme de JSON
    return jsonify({
        'client_id': client_id,
        'prediction': int(prediction[0]),
        'prediction_proba': float(prediction_proba[0][1])
    })

import pytest
import re

# Tests
@pytest.fixture
def client():
    # Créer un client de test (fourni par Flask) pour accéder aux routes
    with app.test_client() as client:
        yield client

# Test de la page d'accueil
# Prends la route principale / de l'application Flask.
def test_home(client):
    response = client.get('/') # Effectue une requête HTTP de type GET sur la route /
    assert response.status_code == 200 # Vérifie que le code de statut HTTP renvoyé par la route / est bien 200, ce qui indique que la requête a été traitée avec succès.
    assert b"Bienvenue sur la page d'accueil!" in response.data # Vérifie que le contenu de la réponse contient bien la chaîne "Bienvenue sur la page d'accueil!".
    assert re.search(r'\d+', response.data.decode())  # Vérifie qu'il y a un numéro dans la réponse
    # assert len(response.data.split()) > 3  # Vérifie qu'un ID de client est inclus
    
# Test de la route /predict avec un ID valide
def test_predict_valid_id(client):
    valid_client_id = 356810  # ID d'un client dans le dico factice client_data
    response = client.get(f'/predict?id={valid_client_id}')
    assert response.status_code == 200
    data = response.get_json() # Récupère le contenu de la réponse en format JSON, contient un dictionnaire avec comme info : client_id, prediction et predicion_proba
    assert 'client_id' in data # Vérifie que le champ client_id est bien dans data
    assert 'prediction' in data
    assert isinstance(data['prediction'], int) # Vérifie que la valeur dans prediction est bien de type int
    assert 'prediction_proba' in data 
    assert isinstance(data['prediction_proba'], float)

# Test de la route /predict avec un ID invalide
def test_predict_invalid_id(client):
    invalid_client_id = 567  # ID qui n'existe pas dans le dico factice client_data
    response = client.get(f'/predict?id={invalid_client_id}')
    assert response.status_code == 404
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == f'Client ID {invalid_client_id} non trouvé'

# Test de la gestion de l'absence de l'ID dans la requête
def test_predict_missing_id(client):
    response = client.get('/predict?id={}')  # ID manquant
    assert response.status_code == 400  # Vérifie que le statut est 400 (Bad Request)
    data = response.get_json()
    assert 'error' in data  # Vérifie que le champ 'error' est présent dans la réponse
    assert data['error'] == 'Missing client id'  # Vérifie que l'erreur est bien "Missing client id"

# Test de la structure de la réponse JSON
def test_predict_response_structure(client):
    valid_client_id = 356810
    response = client.get(f'/predict?id={valid_client_id}')
    data = response.get_json()
    assert 'client_id' in data
    assert 'prediction' in data
    assert 'prediction_proba' in data
    assert isinstance(data['client_id'], int)
    assert isinstance(data['prediction'], int)
    assert isinstance(data['prediction_proba'], float)

# Test de performance (vitesse de réponse)
import time

def test_predict_performance(client):
    start_time = time.time()
    valid_client_id = 356810
    response = client.get(f'/predict?id={valid_client_id}')
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1  # Temps de réponse < 1 seconde

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
