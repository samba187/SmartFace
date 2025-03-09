import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import re
import os
from mtcnn import MTCNN

st.set_page_config(page_title="App de Classification & Régression", layout="centered")
custom_css = """
    <style>
        body {
            background-color: #F5F5F5;
            color: #333333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            font-size: 16px;
        }
        .stSelectbox label {
            font-size: 18px;
            font-weight: bold;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---- Paramètres ----
AGE_NORMALIZATION_FACTOR = 100
IMG_SIZE = (200, 200)

MODELS = {
    "Modèle 1 - Classification de Genre": {
        "path": "MODELE_1.keras",
        "type": "genre"
    },
    "Modèle 2 - Classification d'Âge": {
        "path": "MODELE_2.keras",
        "type": "age"
    },
    "Modèle 3 - Classification Simultanée ": {
        "path": "MODELE_3.keras",
        "type": "joint_inv"  
    },
    "Modèle 4 - Classification Simultanée (Transfert d'apprentissage)": {
        "path": "MODELE_4.keras",
        "type": "joint"
    }
}

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def crop_face(image, margin=0.2):
    """
    Détecte le visage dans l'image à l'aide de MTCNN et retourne un recadrage
    (PIL) incluant une marge. Retourne None si aucun visage n'est détecté.
    """
    detector = MTCNN()
    cv_img = np.array(image.convert("RGB"))  
    results = detector.detect_faces(cv_img)

    if len(results) == 0:
        return None
    
    face = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = face['box']
    
    x = max(0, x)
    y = max(0, y)
    
    m_w = int(margin * w)
    m_h = int(margin * h)
    x1 = max(0, x - m_w)
    y1 = max(0, y - m_h)
    x2 = x + w + m_w
    y2 = y + h + m_h
    
    cropped = cv_img[y1:y2, x1:x2]
    return Image.fromarray(cropped)

def predict_gender(image, model, target_size=(200, 200)):
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    score = preds[0][0]
    if score >= 0.5:
        label = "Femme"
        confidence = score
    else:
        label = "Homme"
        confidence = 1.0 - score
    return label, float(confidence)

def predict_age(image, model, target_size=(200, 200)):
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    age = preds[0][0] * AGE_NORMALIZATION_FACTOR
    return round(age)

def predict_joint_standard(image, model, target_size=(200, 200)):
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    genre_pred, age_pred = model.predict(img_array)
    score = genre_pred[0][0]
    if score >= 0.5:
        gender_label = "Femme"
        gender_conf = score
    else:
        gender_label = "Homme"
        gender_conf = 1.0 - score
    age = age_pred[0][0] * AGE_NORMALIZATION_FACTOR
    return gender_label, float(gender_conf), round(age)

def predict_joint_inversed(image, model, target_size=(200, 200)):
    image = image.convert("RGB").resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    genre_pred, age_pred = model.predict(img_array)
    score = genre_pred[0][0]
    if score >= 0.5:
        gender_label = "Homme"
        gender_conf = score
    else:
        gender_label = "Femme"
        gender_conf = 1.0 - score
    age = age_pred[0][0] * AGE_NORMALIZATION_FACTOR
    return gender_label, float(gender_conf), round(age)

def main():
    st.title("Application de Classification & Régression sur Visages")
    
    model_choice = st.selectbox("Choisissez le modèle :", list(MODELS.keys()))
    model_info = MODELS[model_choice]
    model_path = model_info["path"]
    model_type = model_info["type"]
    
    model = load_model(model_path)
    st.markdown(f"<h3 style='color:#4CAF50;'>{model_choice}</h3>", unsafe_allow_html=True)
    
    option = st.radio("Méthode :", ["Importer une image", "Prendre une photo"])
    
    if option == "Importer une image":
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Image importée", use_column_width=True)
            if st.button("Analyser"):
                cropped_face = crop_face(img, margin=0.2)
                if cropped_face is None:
                    st.warning("Aucun visage détecté, essayez une autre image.")
                else:
                    st.image(cropped_face, caption="Visage recadré", use_column_width=True)
                    if model_type == "genre":
                        label, conf = predict_gender(cropped_face, model)
                        st.markdown(f"<h4 style='color:#FF5722;'>Genre prédit : {label} (Confiance : {conf:.2f})</h4>", unsafe_allow_html=True)
                    elif model_type == "age":
                        age = predict_age(cropped_face, model)
                        st.markdown(f"<h4 style='color:#3F51B5;'>Âge prédit : {age} ans</h4>", unsafe_allow_html=True)
                    elif model_type == "joint":
                        gender_label, gender_conf, age_val = predict_joint_standard(cropped_face, model)
                        st.markdown(f"<h4 style='color:#FF5722;'>Genre : {gender_label} (Confiance : {gender_conf:.2f})</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#3F51B5;'>Âge : {age_val} ans</h4>", unsafe_allow_html=True)
                    elif model_type == "joint_inv":
                        gender_label, gender_conf, age_val = predict_joint_inversed(cropped_face, model)
                        st.markdown(f"<h4 style='color:#FF5722;'>Genre  : {gender_label} (Confiance : {gender_conf:.2f})</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#3F51B5;'>Âge : {age_val} ans</h4>", unsafe_allow_html=True)
                        
    elif option == "Prendre une photo":
        picture = st.camera_input("Prenez une photo")
        if picture is not None:
            img = Image.open(picture)
            st.image(img, caption="Photo capturée", use_column_width=True)
            if st.button("Analyser"):
                cropped_face = crop_face(img, margin=0.2)
                if cropped_face is None:
                    st.warning("Aucun visage détecté, essayez une autre photo.")
                else:
                    st.image(cropped_face, caption="Visage recadré", use_column_width=True)
                    if model_type == "genre":
                        label, conf = predict_gender(cropped_face, model)
                        st.markdown(f"<h4 style='color:#FF5722;'>Genre prédit : {label} (Confiance : {conf:.2f})</h4>", unsafe_allow_html=True)
                    elif model_type == "age":
                        age = predict_age(cropped_face, model)
                        st.markdown(f"<h4 style='color:#3F51B5;'>Âge prédit : {age} ans</h4>", unsafe_allow_html=True)
                    elif model_type == "joint":
                        gender_label, gender_conf, age_val = predict_joint_standard(cropped_face, model)
                        st.markdown(f"<h4 style='color:#FF5722;'>Genre : {gender_label} (Confiance : {gender_conf:.2f})</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#3F51B5;'>Âge : {age_val} ans</h4>", unsafe_allow_html=True)
                    elif model_type == "joint_inv":
                        gender_label, gender_conf, age_val = predict_joint_inversed(cropped_face, model)
                        st.markdown(f"<h4 style='color:#FF5722;'>Genre  : {gender_label} (Confiance : {gender_conf:.2f})</h4>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='color:#3F51B5;'>Âge : {age_val} ans</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()