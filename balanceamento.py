import os
import shutil
import random

origem_train = "dataset/train"
origem_test = "dataset/test"
destino_train = "dataset_reduzido/train"
destino_test = "dataset_reduzido/test"

classes = ['A', 'E', 'I', 'O', 'U']

n_treino = 500
n_teste = int(n_treino * 0.2)

os.makedirs(destino_train, exist_ok=True)
os.makedirs(destino_test, exist_ok=True)

for c in classes:
    os.makedirs(os.path.join(destino_train, c), exist_ok=True)
    os.makedirs(os.path.join(destino_test, c), exist_ok=True)

    imgs_train = [os.path.join(origem_train, c, f) for f in os.listdir(os.path.join(origem_train, c)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    imgs_test = [os.path.join(origem_test, c, f) for f in os.listdir(os.path.join(origem_test, c)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    imgs_total = imgs_train + imgs_test
    random.shuffle(imgs_total)

    imgs_total = imgs_total[:n_treino + n_teste]

    imgs_treino = imgs_total[:n_treino]
    imgs_teste = imgs_total[n_treino:n_treino + n_teste]

    for img in imgs_treino:
        shutil.copy(img, os.path.join(destino_train, c))
    for img in imgs_teste:
        shutil.copy(img, os.path.join(destino_test, c))

    print(f"Classe {c}: {len(imgs_treino)} treino, {len(imgs_teste)} teste")

print("✅ Dataset balanceado e reduzido criado em 'dataset_reduzido'")
