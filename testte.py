import cv2
from matplotlib import pyplot as plt

# --- Função segmentar_mao (a mesma que você já tem) ---
import numpy as np
from skimage.segmentation import chan_vese

def segmentar_mao(img, iteracoes=200, mu=0.25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    cv_result = chan_vese(
        gray,
        mu=mu,
        lambda1=1,
        lambda2=1,
        tol=1e-3,
        max_num_iter=iteracoes,
        dt=0.5,
        init_level_set="checkerboard",
        extended_output=True
    )
    mask = cv_result[0].astype(np.uint8)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result, mask

# --- Teste em uma única imagem ---
caminho_img = "exemplo.jpg"  # altere para o caminho da sua imagem
img = cv2.imread(caminho_img)

if img is None:
    raise FileNotFoundError("Não foi possível carregar a imagem!")

resultado, mascara = segmentar_mao(img)

# --- Exibir resultados ---
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Imagem original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Máscara Chan-Vese")
plt.imshow(mascara, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Segmentação aplicada")
plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
 