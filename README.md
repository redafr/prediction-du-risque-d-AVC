# Prédiction du Risque d'AVC à l'aide de Modèles Machine Learning

Ce projet propose une solution basée sur l'apprentissage automatique pour prédire la probabilité qu'un patient soit victime d’un accident vasculaire cérébral (AVC).  
Il combine des techniques modernes de traitement de données, de rééquilibrage de classes (CTGAN, SMOTEENN) et plusieurs algorithmes de classification performants.

L’objectif est de fournir à la fois un outil de prédiction fiable et une interface interactive permettant de tester différents profils patients en temps réel via Streamlit.

## Modèles utilisés

- XGBoostClassifier
- LightGBMClassifier
- RandomForestClassifier

Tous les modèles ont été entraînés sur un jeu de données enrichi par génération synthétique (CTGAN) et équilibré manuellement.

## Interface utilisateur

Une interface graphique a été développée avec **Streamlit** pour :
- Saisir des données patients simulées
- Obtenir une prédiction (risque ou non)
- Visualiser les probabilités et interpréter les résultats

## Comment exécuter l’application

```bash
pip install -r requirements.txt
streamlit run app.py
