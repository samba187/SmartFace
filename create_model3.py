import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import random
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model

##############################################################################
# 1. Paramètres
##############################################################################
IMG_SIZE = (200, 200)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4

train_dir = "/kaggle/working/data/train"
val_dir   = "/kaggle/working/data/val"
test_dir  = "/kaggle/working/data/test"


train_steps = 18966 // BATCH_SIZE
val_steps   = 2370  // BATCH_SIZE
test_steps  = 2372  // BATCH_SIZE

NORMALIZE_AGE = True

##############################################################################
# 2. Fonctions utilitaires
##############################################################################
def extract_age_from_filename(filename):
    """
    Extrait l'âge depuis le nom du fichier.
    Exemple : '24_0_3_20170117150025331.jpg.chip.jpg' -> 24
    """
    match = re.search(r'(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 0

def augment_image(image):
    """
    Data augmentation pour l'entraînement :
    - flip horizontal aléatoire
    """
    image = tf.image.random_flip_left_right(image)
    return image

def generate_data_from_directory(directory, batch_size=32, img_size=(200,200), 
                                 augment=False):

    image_paths, genre_labels, age_labels = [], [], []
    for folder_name in ['male', 'female']:
        folder_path = os.path.join(directory, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                full_path = os.path.join(folder_path, filename)
                image_paths.append(full_path)
                # Genre
                genre = 1 if folder_name == 'male' else 0
                genre_labels.append(genre)
                # Âge
                age = extract_age_from_filename(filename)
                if NORMALIZE_AGE:
                    age = age / 100.0
                age_labels.append(age)
    
    num_samples = len(image_paths)
    while True:
        indexes = np.arange(num_samples)
        np.random.shuffle(indexes)
        for i in range(0, num_samples, batch_size):
            batch_indexes = indexes[i:i+batch_size]
            batch_paths = [image_paths[k] for k in batch_indexes]
            batch_genre = [genre_labels[k] for k in batch_indexes]
            batch_age   = [age_labels[k]   for k in batch_indexes]
            
            images = []
            for path in batch_paths:
                img = plt.imread(path)
                img = tf.image.resize(img, img_size)
                img = img / 255.0
                if augment:
                    img = augment_image(img)
                images.append(img)
            
            images = np.array(images, dtype=np.float32)
            
            yield images, {
                'genre_output': np.array(batch_genre, dtype=np.float32),
                'age_output':   np.array(batch_age,   dtype=np.float32)
            }

##############################################################################
# 3. Construction du modèle (4 blocs)
##############################################################################
def build_model():
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Bloc 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Bloc 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Bloc 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Bloc 4
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Dense
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # Sortie genre 
    genre_output = Dense(1, activation='sigmoid', name='genre_output')(x)

    # Sortie âge 
    age_output = Dense(1, activation='linear', name='age_output')(x)

    model = Model(inputs=inputs, outputs=[genre_output, age_output])
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={'genre_output': 'binary_crossentropy', 'age_output': 'mse'},
    metrics={'genre_output': 'accuracy', 'age_output': 'mae'}
)
model.summary()

##############################################################################
# 4. Générateurs (train avec augmentation, val/test sans)
##############################################################################
train_generator = generate_data_from_directory(
    directory=train_dir,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=True  
)

val_generator = generate_data_from_directory(
    directory=val_dir,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False
)

##############################################################################
# 5. Callbacks
##############################################################################
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

##############################################################################
# 6. Entraînement
##############################################################################
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks
)

##############################################################################
# 7. Sauvegarde du modèle réentraîné
##############################################################################
model_save_path = "Modele_3.keras"
model.save(model_save_path)
print(f"Modèle sauvegardé sous '{model_save_path}'.")

##############################################################################
# 8. Visualisation des courbes
##############################################################################
plt.figure(figsize=(12, 4))

# Accuracy (genre_output)
plt.subplot(1, 2, 1)
plt.plot(history.history['genre_output_accuracy'], label='Train Accuracy', c='blue')
if 'val_genre_output_accuracy' in history.history:
    plt.plot(history.history['val_genre_output_accuracy'], label='Val Accuracy', c='orange')
plt.title("Courbe d'accuracy (Réentraînement)")
plt.xlabel('Époques')
plt.ylabel('Accuracy')
plt.legend()

# Perte globale
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', c='blue')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val Loss', c='orange')
plt.title("Courbe de perte (Réentraînement)")
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("Clés de l'historique :", history.history.keys())
