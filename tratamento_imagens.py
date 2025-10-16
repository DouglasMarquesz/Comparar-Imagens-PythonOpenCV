import cv2
import numpy as np

# Carrega a imagem em cinza
img_gray = cv2.imread('imagens/entrada.jpg', cv2.IMREAD_GRAYSCALE)

# Redimensiona a imagem
img_gray = cv2.resize(img_gray, (1280, 720))  # largura x altura

# Aplica blur para reduzir ruído
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Ajusta contraste e brilho
img_enhanced = cv2.convertScaleAbs(img_blur, alpha=1.5, beta=-90)

# Filtro de nitidez
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
img_sharp = cv2.filter2D(img_enhanced, -1, kernel)

# Detecta bordas
bordas = cv2.Canny(img_sharp, 50, 150)

# Encontra contornos
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Cria imagem branca para desenhar os contornos
img_result = np.ones_like(img_gray) * 255  # fundo branco

# Desenha apenas os contornos em preto
cv2.drawContours(img_result, contornos, -1, 0, 4)  # 0 = preto, 2 = espessura

# Mostra e salva a imagem final
cv2.imshow('Contornos', img_result)
cv2.imwrite("dados/imagem_final.jpg", img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
