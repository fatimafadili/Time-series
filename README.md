📊 Analyse des Séries Temporelles - Projet Complet
🎯 Description du Projet
Ce projet présente une analyse complète des techniques de prévision des séries temporelles, allant des méthodes statistiques traditionnelles aux modèles avancés de Deep Learning (RNN, LSTM, CNN, Transformers). L'objectif est de fournir une boîte à outils complète pour la prédiction de séries temporelles avec des exemples pratiques et des benchmarks comparatifs.

📁 Structure du Projet
text
time-series-project/
├── README.md
├── time_series_analysis.ipynb
├── requirements.txt
├── data/
│   ├── generated_series.npy
│   └── time_series.csv
├── models/
│   ├── naive_model.pkl
│   ├── nn_model.h5
│   ├── lstm_model.h5
│   └── transformer_model.h5
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   └── utils.py
└── results/
    ├── predictions/
    └── visualizations/
🚀 Installation
Prérequis
Python 3.8+

pip ou conda

Installation des dépendances
bash
pip install -r requirements.txt
Dépendances principales
txt
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
tensorflow==2.13.0
statsmodels==0.14.0
scikit-learn==1.3.0
jupyter==1.0.0
📊 Données
Génération de données synthétiques
Le projet utilise des données synthétiques avec tendance, saisonnalité et bruit :

python
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
time = np.arange(4 * 365 + 1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Génération de la série
series = baseline + slope * time
series += amplitude * np.sin(time / 365 * 2 * np.pi)
series += np.random.normal(scale=noise_level, size=len(time))
Division des données
Données d'entraînement : 1000 premiers points

Données de validation : points restants

🧠 Modèles Implémentés
1. Méthodes Statistiques
Prévision naïve : Dernière valeur observée

Moyenne mobile : Fenêtres glissantes

Différenciation : Élimination tendance/saisonnalité

2. Réseaux de Neurones
NN simple : 1 couche dense (MSE: 46.99)

Deep NN : 2 couches 10→10 (MSE: ~25-30)

RNN/LSTM/GRU : Modèles récurrents

3. Architectures Avancées
CNN 1D : Convolutions temporelles

Transformer : Attention multi-têtes

📈 Résultats
Comparaison des performances
Modèle	MSE	MAE	Temps d'entraînement
Prévision naïve	50.63	5.61	< 1s
Moyenne mobile	31.45	4.44	< 1s
NN simple	46.99	4.97	~2 min
Deep NN	25-30	4-5	~5 min
LSTM	Variable	Variable	~10 min
CNN	357.95	14.89	~8 min
🔧 Utilisation
1. Exécution du notebook
bash
jupyter notebook time_series_analysis.ipynb
2. Entraînement d'un modèle
python
from src.models import TimeSeriesModel

# Initialisation du modèle
model = TimeSeriesModel(window_size=30)

# Entraînement
model.train(x_train, epochs=100, validation_data=x_valid)

# Prédiction
predictions = model.predict(x_valid)
3. Visualisation des résultats
python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(x_valid, label='Valeurs réelles')
plt.plot(predictions, label='Prédictions', alpha=0.7)
plt.legend()
plt.title('Comparaison prédictions vs réalité')
plt.show()
🎯 Fonctionnalités Clés
✅ Pré-traitement intelligent
Différenciation pour stationnarité

Fenêtrage glissant configurable

Normalisation automatique

✅ Modèles variés
7 architectures différentes

Hyperparamètres optimisables

Sauvegarde/chargement des modèles

✅ Évaluation complète
Métriques MSE et MAE

Visualisations interactives

Comparaison des performances

✅ Production ready
Code modulaire

Documentation complète

Facile à étendre

📝 Exemple de Code
Création d'un modèle LSTM
python
import tensorflow as tf
from tensorflow.keras import layers

def create_lstm_model(window_size):
    model = tf.keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model
Pipeline d'entraînement complet
python
# 1. Chargement des données
data = load_time_series('data/time_series.csv')

# 2. Pré-traitement
processed_data = preprocess_data(data, window_size=30)

# 3. Division train/validation
x_train, x_valid = split_data(processed_data, split_time=1000)

# 4. Création du modèle
model = create_lstm_model(window_size=30)

# 5. Entraînement
history = model.fit(
    x_train,
    epochs=100,
    validation_data=x_valid,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
)

# 6. Évaluation
mse, mae = model.evaluate(x_valid)
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
📊 Visualisations Disponibles
Série temporelle originale

Décomposition (tendance, saisonnalité, résidus)

Autocorrélation

Prédictions vs réalité

Courbes d'apprentissage

Comparaison des modèles

🚀 Déploiement
Exécution rapide
bash
# Clonez le dépôt
git clone https://github.com/votre-utilisateur/time-series-project.git

# Installez les dépendances
cd time-series-project
pip install -r requirements.txt

# Exécutez le notebook
jupyter notebook time_series_analysis.ipynb
Docker (optionnel)
dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
🤝 Contribution
Les contributions sont les bienvenues ! Voici comment contribuer :

Fork le projet

Créez une branche (git checkout -b feature/AmazingFeature)

Committez vos changements (git commit -m 'Add some AmazingFeature')

Push vers la branche (git push origin feature/AmazingFeature)

Ouvrez une Pull Request

📚 Documentation Supplémentaire
Concepts théoriques couverts :
Stationnarité et tests ADF

Autocorrélation et autocorrélation partielle

Différenciation et désaisonnalisation

Validation croisée temporelle

Métriques d'évaluation spécifiques aux séries temporelles

Techniques avancées :
Learning rate scheduling

Early stopping dynamique

Regularisation pour séries temporelles

Ensembling de modèles

Features engineering temporel

