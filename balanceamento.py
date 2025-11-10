import os
import shutil
import random

origem = "dataset"
destino_train = "dataset_dividido/train"
destino_test = "dataset_dividido/test"

classes = ['A', 'E', 'I', 'O', 'U']

# Criar pastas de destino
os.makedirs(destino_train, exist_ok=True)
os.makedirs(destino_test, exist_ok=True)

for c in classes:
    origem_classe = os.path.join(origem, c)
    imgs = [os.path.join(origem_classe, f) for f in os.listdir(origem_classe)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    random.shuffle(imgs)

    n_total = len(imgs)
    n_treino = int(0.8 * n_total)
    n_teste = n_total - n_treino

    imgs_treino = imgs[:n_treino]
    imgs_teste = imgs[n_treino:]

    os.makedirs(os.path.join(destino_train, c), exist_ok=True)
    os.makedirs(os.path.join(destino_test, c), exist_ok=True)

    for img in imgs_treino:
        shutil.copy(img, os.path.join(destino_train, c))
    for img in imgs_teste:
        shutil.copy(img, os.path.join(destino_test, c))

    print(f"Classe {c}: {len(imgs_treino)} treino, {len(imgs_teste)} teste")

print("✅ Dataset dividido em 80% treino e 20% teste criado em 'dataset_dividido'")
