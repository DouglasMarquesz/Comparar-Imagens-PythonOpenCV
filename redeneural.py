import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

# Parâmetros
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
DATASET_DIR = 'dataset'  # caminho para suas pastas de contorno

# Precisa ter no mino 20 imagens por classe (pastas)
# O nome da pastas será o mesmo do que a classe

# Gerador de imagens com aumento de dados
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% para validação
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',  # contornos são em grayscale
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation'
)

# Criando a CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compilando
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Treino:", train_gen.samples, "imagens")
print("Validação:", val_gen.samples, "imagens")
print("Classes:", train_gen.class_indices)

# Treinando
model.fit(train_gen, validation_data=val_gen, epochs=200)

# Salvando modelo
model.save('modelo_contornos.h5')
print("Modelo salvo com sucesso!")
