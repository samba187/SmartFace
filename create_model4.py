import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import tensorflow.keras.regularizers as reg
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import re

##############################################################################
# 1. Paramètres et configuration
##############################################################################
IMG_SIZE = (200, 200)    # Taille d'entrée
BATCH_SIZE = 64
EPOCHS = 30
INITIAL_LR = 1e-4        # LR initial

TRAIN_DIR = "/kaggle/working/data/train"
VAL_DIR   = "/kaggle/working/data/val"
TEST_DIR  = "/kaggle/working/data/test"

NORMALIZE_AGE = True  # Diviser l'âge par 100 si nécessaire

##############################################################################
# 2. Fonctions utilitaires
##############################################################################
def get_file_list(root_dir):
    file_list = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            for fname in os.listdir(subdir_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_list.append(os.path.join(subdir, fname))
    return file_list

def parse_utkface_filename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) >= 2:
        age = int(parts[0])
        gender = int(parts[1])
    else:
        age, gender = 0, 0
    return age, gender

def utkface_generator(root_dir, file_list, batch_size=64, img_size=(200,200)):
    num_samples = len(file_list)
    while True:
        random.shuffle(file_list)
        for i in range(0, num_samples, batch_size):
            batch_paths = file_list[i:i+batch_size]
            images = []
            genre_labels = []
            age_labels = []
            for rel_path in batch_paths:
                full_path = os.path.join(root_dir, rel_path)
                img = cv2.imread(full_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0
                images.append(img)
                
                fname = os.path.basename(rel_path)
                age, gender = parse_utkface_filename(fname)
                if NORMALIZE_AGE:
                    age = age / 100.0
                age_labels.append(age)
                genre_labels.append(gender)
            images = np.array(images)
            yield images, {
                "genre_output": np.array(genre_labels, dtype=np.float32),
                "age_output":   np.array(age_labels,   dtype=np.float32)
            }

##############################################################################
# 3. Récupération des fichiers
##############################################################################
train_files = get_file_list(TRAIN_DIR)
val_files   = get_file_list(VAL_DIR)
test_files  = get_file_list(TEST_DIR)

print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

train_steps = len(train_files) // BATCH_SIZE
val_steps   = len(val_files) // BATCH_SIZE
test_steps  = len(test_files) // BATCH_SIZE

train_gen = utkface_generator(TRAIN_DIR, train_files, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_gen   = utkface_generator(VAL_DIR,   val_files,   batch_size=BATCH_SIZE, img_size=IMG_SIZE)
test_gen  = utkface_generator(TEST_DIR,  test_files,  batch_size=BATCH_SIZE, img_size=IMG_SIZE)

##############################################################################
# 4. Construction du modèle (MobileNetV2 + régularisation extrême)
##############################################################################
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dropout(0.8)(x)

x = Dense(128, 
          activation='relu', 
          kernel_regularizer=reg.l2(1e-2))(x)

x = Dropout(0.7)(x)

# Sorties avec régularisation L2
genre_output = Dense(1, 
                     activation='sigmoid', 
                     kernel_regularizer=reg.l2(1e-2),
                     name='genre_output')(x)
age_output   = Dense(1, 
                     activation='linear', 
                     kernel_regularizer=reg.l2(1e-2),
                     name='age_output')(x)

model = Model(inputs=base_model.input, outputs=[genre_output, age_output])

for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss={'genre_output': 'binary_crossentropy', 'age_output': 'mse'},
    metrics={'genre_output': 'accuracy', 'age_output': 'mae'}
)

model.summary()

##############################################################################
# 5. Plan de LR + EarlyStopping
##############################################################################
def manual_lr_schedule(epoch, lr):

    if epoch > 10:
        return 1e-5
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(manual_lr_schedule, verbose=1)

reduce_lr_plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,   
    patience=3,   
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

callbacks = [lr_scheduler, reduce_lr_plateau, early_stopping]

##############################################################################
# 6. Entraînement
##############################################################################
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks
)

##############################################################################
# 7. Sauvegarde du modèle
##############################################################################
model_save_path = "model_4.keras"
model.save(model_save_path)
print(f"Modèle sauvegardé sous '{model_save_path}'.")

##############################################################################
# 8. Évaluation sur le jeu de test
##############################################################################
results = model.evaluate(test_gen, steps=test_steps, verbose=1)
print("Résultats sur le jeu de test :", results)

##############################################################################
# 9. Visualisation des courbes
##############################################################################
plt.figure(figsize=(12, 4))

# Accuracy (genre_output)
plt.subplot(1, 2, 1)
plt.plot(history.history['genre_output_accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_genre_output_accuracy'], label='Val Accuracy', color='orange')
plt.title("Courbe d'accuracy (Genre)")
plt.xlabel('Époques')
plt.ylabel('Accuracy')
plt.legend()

# Perte globale
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title("Courbe de perte")
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()