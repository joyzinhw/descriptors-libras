from preprocessamento import carregar_imagens
from descritores import extrair_glcm, extrair_hog, extrair_lbp
from classificadores import treinar_classificadores
import traceback

train_path = "dataset/train"
test_path = "dataset/test"

nomes_classes = ["A", "E", "I", "O", "U"]

print("🔹 Carregando imagens...")
X_train_imgs, y_train = carregar_imagens(train_path)
X_test_imgs, y_test = carregar_imagens(test_path)

descritores = {
    "GLCM": extrair_glcm,
    "HOG": extrair_hog,
    "LBP": extrair_lbp
}

resultados_finais = {}

for nome_descritor, func in descritores.items():
    try:
        print(f"\n=== Avaliando descritor: {nome_descritor} ===")
        resultados = treinar_classificadores(
            X_train_imgs, X_test_imgs, y_train, y_test,
            func, nome_descritor, nomes_classes=nomes_classes
        )
        resultados_finais[nome_descritor] = resultados
    except Exception as e:
        print(f" Erro ao avaliar {nome_descritor}: {e}")
        traceback.print_exc()

print("\n🏁 Comparação entre Descritores e Classificadores:")
for descritor, resultados in resultados_finais.items():
    print(f"\n🔹 Descritor: {descritor}")
    for clf, metricas in resultados.items():
        print(f"  📊 {clf} -> "
              f"Acurácia: {metricas['Acurácia']:.4f}, "
              f"Precisão: {metricas['Precisão']:.4f}, "
              f"F1-score: {metricas['F1-score']:.4f}, "
              f"Fisher Score: {metricas['Fisher Score']:.4f}")
