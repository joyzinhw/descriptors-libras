import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
# from radiomics import glrlm

def extrair_glcm(img):
    glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'ASM').mean()
    ]

# def extrair_glrlm(img):
#     img_uint8 = img.astype(np.uint8)
#     rl = glrlm.RLMFeatures2D(img_uint8, np.ones_like(img_uint8))
#     rl.enableAllFeatures()
#     rl.calculateFeatures()
#     values = [float(v) for v in rl.featureValues.values()]
#     return values[:10]

def extrair_hog(img):
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

def extrair_lbp(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist
