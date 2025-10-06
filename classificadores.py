import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
from tqdm import tqdm

def fisher_score(X, y):
    classes = np.unique(y)
    overall_mean = np.mean(X, axis=0)
    numerator = 0
    denominator = 0
    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        numerator += n_c * (mean_c - overall_mean) ** 2
        denominator += np.sum((X_c - mean_c) ** 2, axis=0)
    fs = np.mean(numerator / (denominator + 1e-8))
    return fs

def treinar_classificadores(X_train_imgs, X_test_imgs, y_train, y_test, extrator, nome_descritor, nomes_classes=None):
    # Extrair descritores
    X_train, X_test = [], []

    print(f"\n🔹 Extraindo descritores {nome_descritor}...")
    for img in tqdm(X_train_imgs, desc=f"Treino {nome_descritor}"):
        X_train.append(extrator(img))
    for img in tqdm(X_test_imgs, desc=f"Teste {nome_descritor}"):
        X_test.append(extrator(img))

    X_train, X_test = np.array(X_train), np.array(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classificadores = {
        "SVM": SVC(kernel='linear', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
    }

    resultados = {}
    for nome_clf, clf in classificadores.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        fs = fisher_score(X_train, np.array(y_train))

        resultados[nome_clf] = {
            "Acurácia": acc,
            "Precisão": prec,
            "F1-score": f1,
            "Fisher Score": fs
        }

        print(f"\n📊 {nome_descritor} - {nome_clf}")
        print(f"Acurácia: {acc:.4f}, Precisão: {prec:.4f}, F1-score: {f1:.4f}, Fisher Score: {fs:.4f}")


        print("\nResultados para {}:".format(nome_descritor))
        print(classification_report(y_test, y_pred, target_names=nomes_classes))

    return resultados
