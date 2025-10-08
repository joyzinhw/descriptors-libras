import os

def contar_imagens(path):
    total = 0
    por_classe = {}
    for classe in os.listdir(path):
        classe_path = os.path.join(path, classe)
        if not os.path.isdir(classe_path):
            continue
        imgs = [f for f in os.listdir(classe_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        por_classe[classe] = len(imgs)
        total += len(imgs)
    return total, por_classe

train_total, train_por_classe = contar_imagens("dataset/train")
test_total, test_por_classe = contar_imagens("dataset/test")

print(f"Número total de imagens de treino: {train_total}")
print("Por classe (treino):", train_por_classe)

print(f"Número total de imagens de teste: {test_total}")
print("Por classe (teste):", test_por_classe)

