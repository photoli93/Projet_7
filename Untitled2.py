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


if __name__ == '__main__':
    app.run(port=5002, debug=True, use_reloader=False)


