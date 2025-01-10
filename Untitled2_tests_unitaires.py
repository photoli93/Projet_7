import pytest
import re
from Untitled2 import app  # Importer l'application Flask

@pytest.fixture
def client():
    # Créer un client de test (fourni par Flask) pour accéder aux routes
    with app.test_client() as client:
        yield client

# Test de la page d'accueil
def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Bienvenue sur la page d'accueil!" in response.data
    assert re.search(r'\d+', response.data.decode())

# Test de la route /predict avec un ID valide
def test_predict_valid_id(client):
    valid_client_id = 356810
    response = client.get(f'/predict?id={valid_client_id}')
    assert response.status_code == 200
    data = response.get_json()
    assert 'client_id' in data
    assert 'prediction' in data
    assert isinstance(data['prediction'], int)
    assert 'prediction_proba' in data
    assert isinstance(data['prediction_proba'], float)

# Test de la route /predict avec un ID invalide
def test_predict_invalid_id(client):
    invalid_client_id = 567
    response = client.get(f'/predict?id={invalid_client_id}')
    assert response.status_code == 404
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == f'Client ID {invalid_client_id} non trouvé'

# Test de la gestion de l'absence de l'ID dans la requête
def test_predict_missing_id(client):
    response = client.get('/predict?id={}')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'Missing client id'

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

# Test de performance
import time

def test_predict_performance(client):
    start_time = time.time()
    valid_client_id = 356810
    response = client.get(f'/predict?id={valid_client_id}')
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1  # Temps de réponse < 1 seconde
