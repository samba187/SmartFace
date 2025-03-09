# SmartFace

## Description
SmartFace est une application basée sur des réseaux de neurones convolutionnels (CNN) permettant de classifier le genre et l'âge à partir d'images de visages. L'application repose sur plusieurs modèles d'apprentissage automatique et offre une interface interactive via **Streamlit**. Le projet est déployé sur **Hugging Face Spaces**.

## Fonctionnalités
- **Classification du genre** (homme/femme) à l’aide d’un CNN.
- **Prédiction de l’âge** via une approche de régression.
- **Classification simultanée du genre et de l’âge** avec un modèle multitâche.
- **Utilisation du transfert d’apprentissage** pour améliorer la précision.
- **Interface utilisateur interactive** avec **Streamlit**.
- **Déploiement en ligne** sur **Hugging Face Spaces**.

## Jeu de Données
Le projet utilise le dataset [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new), qui contient des images de visages annotées avec des informations sur le genre et l'âge.

## Technologies utilisées
- **Python** : Langage de programmation principal.
- **TensorFlow & Keras** : Pour la construction et l'entraînement des CNN.
- **Streamlit** : Pour l’interface utilisateur interactive.
- **Hugging Face Spaces** : Pour le déploiement en ligne.

## Structure du projet
```
📂 SmartFace
│── README.md               # Documentation du projet
│── requirements.txt        # Dépendances Python
│── app.py                  # Code de l'application et interface Streamlit
│── create_model1.py        # Modèle de classification de genre
│── create_model3.py        # Modèle combiné genre + âge
│── create_model4.py        # Modèle avec transfert d’apprentissage
```

## Installation et Exécution

### 1. Cloner le projet
```bash
git clone https://github.com/ton-utilisateur/SmartFace.git
cd SmartFace
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

## Déploiement
L'application est déployée sur **Hugging Face Spaces**. Vous pouvez l’utiliser directement en ligne ici :  
🔗 **[Lien du projet sur Hugging Face](https://huggingface.co/spaces/sam100jsp/SmartFace)** 

## Auteur
Développé par **SORBO Samba**, **CHRIMNI Younes**, **FATHI Iliyes**,  **TEMIZ Arda** 
Encadré par **Mr Faye & Mme Azzag** (SAE 1 - BUT 3)
