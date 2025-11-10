import cv2
import numpy as np
import os
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

def carregar_imagens(base_path):
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
            result, mask = segmentar_mao(img)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            x, y, w, h = cv2.boundingRect(mask)
            crop = gray[y:y+h, x:x+w]
            crop = cv2.equalizeHist(crop)
            crop = cv2.GaussianBlur(crop, (3, 3), 0)
            imagens.append(crop)
            labels.append(classe)
    return imagens, labels
