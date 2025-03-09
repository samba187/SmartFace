import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization, Activation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

#########################
# 1. Construction du Modèle
#########################
model = Sequential()

model.add(Input(shape=(200, 200, 3)))  # Images 200x200

# Bloc 1
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Bloc 2
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Bloc 3
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Bloc 4
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten + Dense
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

# Sortie binaire
model.add(Dense(1, activation='sigmoid'))

# Compilation
lr = 1e-3  # Taux d'apprentissage
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#########################
# 2. Générateurs de Données
#########################
# Chemins des répertoires d'entraînement et de validation sur Kaggle
train_dir = "/kaggle/working/data/train"
val_dir   = "/kaggle/working/data/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),  # Correspond à l'input_shape=(200, 200, 3)
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

#########################
# 3. Callbacks
#########################
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='/kaggle/working/best_model_200.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

#########################
# 4. Entraînement
#########################
epochs = 60

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks
)

#########################
# 5. Sauvegarde Finale
#########################
model.save("/kaggle/working/MODELE_1.keras")

#########################
# 6. Visualisation des Courbes
#########################
plt.figure(figsize=(12, 4))

# Courbe d’accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', c='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', c='orange')
plt.title("Courbe d'accuracy")
plt.xlabel('Époques')
plt.ylabel('Accuracy')
plt.legend()

# Courbe de perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', c='blue')
plt.plot(history.history['val_loss'], label='Val Loss', c='orange')
plt.title("Courbe de perte")
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
