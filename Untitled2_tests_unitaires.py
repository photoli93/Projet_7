import pytest
import re
from flask import Flask, jsonify, request
import random
import pandas as pd
from unittest.mock import MagicMock

# Données factices pour simuler client_data
client_data = pd.DataFrame({
    'num__SK_ID_CURR': [1001, 1002, 1003],
    'feature_1': [0.5, 0.7, 0.8],
    'feature_2': [0.3, 0.4, 0.6]
})

# Modèle factice
class FakeModel:
    def predict(self, data):
        return [1]  # Exemple de prédiction
    def predict_proba(self, data):
        return [[0.3, 0.7]]  # Probabilité de la classe (0.3 pour la classe 0, 0.7 pour la classe 1)

# Créer l'application Flask
app = Flask(__name__)

# Initialisation d'un modèle factice
model = FakeModel()

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
    valid_client_id = 1001  # ID d'un client dans le dico factice client_data
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
    valid_client_id = 1001
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
    response = client.get('/predict?id=1001')
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1  # Temps de réponse < 1 seconde
