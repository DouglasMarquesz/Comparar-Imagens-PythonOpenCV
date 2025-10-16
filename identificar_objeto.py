import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from redeneural import train_gen

# Carrega o modelo
model = load_model('modelo_contornos.h5')

# Caminho da nova imagem
img_path = 'imagem_entrada.jpg'

# Carrega a imagem
img = image.load_img(img_path, target_size=(128,128), color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predição
pred = model.predict(img_array)
classe_idx = np.argmax(pred)

# Mapear índice para nome da classe
nomes_classes = list(train_gen.class_indices.keys())
print("Objeto previsto:", nomes_classes[classe_idx])
input('Aperte enter para finalizar')