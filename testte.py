# import os

# def contar_imagens(path):
#     total = 0
#     por_classe = {}
#     for classe in os.listdir(path):
#         classe_path = os.path.join(path, classe)
#         if not os.path.isdir(classe_path):
#             continue
#         imgs = [f for f in os.listdir(classe_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         por_classe[classe] = len(imgs)
#         total += len(imgs)
#     return total, por_classe

# train_total, train_por_classe = contar_imagens("dataset/train")
# test_total, test_por_classe = contar_imagens("dataset/test")

# print(f"Número total de imagens de treino: {train_total}")
# print("Por classe (treino):", train_por_classe)

# print(f"Número total de imagens de teste: {test_total}")
# print("Por classe (teste):", test_por_classe)
import cv2
import numpy as np
import os

def segmentar_mao(img):
    """
    Segmenta a mão com base na cor da pele (YCrCb).
    """
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# Caminho da imagem de teste
img_path = "dataset/test/A/1.png"  # Substitua pelo caminho da sua imagem
img = cv2.imread(img_path)

if img is not None:
    # Pré-processamento
    img_segmentada = segmentar_mao(img)
    img_segmentada_gray = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2GRAY)
    img_segmentada_gray = cv2.resize(img_segmentada_gray, (128, 128))

    # Redimensiona e converte original para grayscale para concatenar
    original_gray = cv2.cvtColor(cv2.resize(img, (128,128)), cv2.COLOR_BGR2GRAY)

    # Concatenar horizontalmente
    combined = cv2.hconcat([original_gray, img_segmentada_gray])

    # Mostrar
    cv2.imshow("Original (esq) vs Pré-processada (dir)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Imagem não encontrada. Verifique o caminho.")
