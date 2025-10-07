#Algoritmo para tratar imagens e cadastrar no banco de dados
import cv2
import numpy as np

# Carrega a imagem e converter para cinza
img_gray = cv2.imread('imagens\entrada.jpg', cv2.IMREAD_GRAYSCALE)  # coloque o caminho correto

# Alterar formato da imagem para 300x200
img_gray = cv2.resize(img_gray, (1280, 720))  # largura x altura

img = cv2.GaussianBlur(img_gray,(3,3),0)

# Melhorar o constraste
img = cv2.convertScaleAbs(img, alpha=1.5, beta=-90)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])  # Kernel de nitidez

img = cv2.filter2D(img, -1, kernel)

bordas = cv2.Canny(img, 50, 150)
# Encontra os contornos
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenha os contornos (espessura 3, cor preta)
cv2.drawContours(img, contornos, -1, (0, 0, 0), 3)

# Deixar o cinza mais proximo ao preto
# Aplica um threshold adaptativo para separar objeto do fundo
# Pixels claros (fundo) se tornam 255, pixels escuros (objeto) se tornam 0
_, binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)

# Cria a imagem resultado: objeto preto, fundo branco
img_result = np.zeros_like(img)
img_result[:] = 255  # fundo branco
img_result[binary == 255] = 0  # objeto preto

# Mostra a imagem em uma janela
cv2.imshow('Imagem Final',img_result)

# Salvar imagem
# Salva o resultado
cv2.imwrite("dados\imagem_final.jpg", img_result)

# Espera até que uma tecla seja pressionada
cv2.waitKey(0)

# Fecha todas as janelas
cv2.destroyAllWindows()