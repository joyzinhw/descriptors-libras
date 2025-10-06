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

def carregar_imagens(base_path):
    """
    Carrega imagens, aplica segmentação e converte para escala de cinza.
    Retorna: lista de imagens processadas e suas labels.
    """
    imagens, labels = [], []
    for classe in os.listdir(base_path):
        classe_path = os.path.join(base_path, classe)
        if not os.path.isdir(classe_path):
            continue
        for fname in os.listdir(classe_path):
            img_path = os.path.join(classe_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = segmentar_mao(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))
            imagens.append(img)
            labels.append(classe)
    return imagens, labels
