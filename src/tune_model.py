"""
Model Tuning - Dataset de Deteção de Fraude em Cartões de Crédito
Projeto IA III - ISTEC

Otimiza 7 algoritmos de Machine Learning (scikit-learn) com GridSearchCV e treina 1 rede
neuronal (TensorFlow/Keras), guardando tudo em models/ (*_opt.joblib, tensorflow_mlp_opt.keras).

Decisões (trade-offs):
- Métrica de otimização: Average Precision (PR-AUC) - não premeia a classe maioritária
  e não depende do limiar de decisão.
- Validação cruzada temporal (TimeSeriesSplit) nos modelos treinados com todo o treino.
- SVM, KNN e Gradient Boosting não escalam para 228k linhas com 2 CPUs: são otimizados
  e treinados numa amostra (TODAS as fraudes + N legítimas, ver config.SAMPLE_LEGIT),
  mas avaliados no conjunto de teste completo.
"""
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_SPLITS, MODELS_DIR, RANDOM_STATE, SAMPLE_LEGIT, TEST_FILE, TRAIN_FILE
from src.train_model import calcular_metricas, carregar_preparados

warnings.filterwarnings("ignore")

# nome -> (estimador, grelha, usa_escala, ficheiro)
def definir_grelhas() -> dict:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            True, "logistic_regression_opt.joblib"),
        "SVM": (
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale"], "class_weight": [None, "balanced"]},
            True, "svm_opt.joblib"),
        "Random Forest": (
            RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
            {"n_estimators": [50, 100], "max_depth": [8, 12], "class_weight": ["balanced_subsample"]},
            False, "random_forest_opt.joblib"),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, 12], "criterion": ["gini", "entropy"],
             "class_weight": [None, "balanced"], "min_samples_leaf": [1, 10]},
            False, "decision_tree_opt.joblib"),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1], "max_depth": [3, 4], "subsample": [0.8]},
            False, "gradient_boosting_opt.joblib"),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": list(np.logspace(-9, -3, 4))},
            True, "naive_bayes_opt.joblib"),
        "KNN": (
            KNeighborsClassifier(n_jobs=-1),
            {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
            True, "knn_opt.joblib"),
    }


def amostrar_treino(X, y, n_legitimas: int):
    """Todas as fraudes + n_legitimas aleatórias (preserva as fraudes raras)."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx_f = np.where(y.values == 1)[0]
    idx_l = rng.choice(np.where(y.values == 0)[0], size=n_legitimas, replace=False)
    idx = np.sort(np.concatenate([idx_f, idx_l]))
    return X.iloc[idx], y.iloc[idx]


def otimizar_modelos_ml(verbose: bool = True):
    """GridSearchCV a cada modelo de ML; devolve (tabela de resultados, previsões/scores por modelo)."""
    X_tr, y_tr, X_te, y_te = carregar_preparados()
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_te_s = scaler.transform(X_te)

    resultados, scores = [], {}
    for nome, (estimador, grelha, escalado, ficheiro) in definir_grelhas().items():
        t0 = time.time()
        amostra = nome in SAMPLE_LEGIT
        Xa, ya = amostrar_treino(X_tr, y_tr, SAMPLE_LEGIT[nome]) if amostra else (X_tr, y_tr)
        Xa_in = scaler.transform(Xa) if escalado else Xa
        Xte_in = X_te_s if escalado else X_te
        cv = StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE) if amostra \
            else TimeSeriesSplit(n_splits=CV_SPLITS)

        gs = GridSearchCV(estimador, grelha, scoring="average_precision", cv=cv, n_jobs=1 if nome == "Random Forest" else 2)
        gs.fit(Xa_in, ya)
        melhor = gs.best_estimator_
        if nome == "SVM":  # probabilidades só no modelo final (evita CV interna em cada combinação)
            melhor = SVC(**{**gs.best_params_, "kernel": "rbf"}, probability=True, random_state=RANDOM_STATE).fit(Xa_in, ya)

        score = melhor.predict_proba(Xte_in)[:, 1]
        pred = (score >= 0.5).astype(int)
        joblib.dump(melhor, MODELS_DIR / ficheiro)
        scores[nome] = score
        resultados.append({"Modelo": nome, "Tipo": "Machine Learning", "Ficheiro": ficheiro, "Escalado": escalado,
                           "Treino": f"amostra ({len(ya)})" if amostra else f"completo ({len(ya)})",
                           **calcular_metricas(y_te, pred, score),
                           "Melhores Hiperparâmetros": str(gs.best_params_)})
        if verbose:
            print(f"[{nome}] CV AP={gs.best_score_:.4f} | teste PR-AUC={resultados[-1]['PR-AUC (%)']}% "
                  f"| {time.time() - t0:.0f}s | {gs.best_params_}")
    return pd.DataFrame(resultados), scores


def treinar_tensorflow(verbose: bool = True):
    """MLP em TensorFlow/Keras: Dense(64->32->16) + Dropout, sigmoid, class_weight e EarlyStopping."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    X_tr, y_tr, X_te, y_te = carregar_preparados()
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    n0, n1 = np.bincount(y_tr.values)
    pesos = {0: len(y_tr) / (2 * n0), 1: len(y_tr) / (2 * n1)}

    modelo = Sequential([
        Input(shape=(Xs_tr.shape[1],)),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(32, activation="relu"), Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008), loss="binary_crossentropy",
                   metrics=[tf.keras.metrics.AUC(name="auc", curve="PR")])
    paragem = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
    modelo.fit(Xs_tr, y_tr.values, epochs=60, batch_size=1024, validation_split=0.2,  # últimos 20% (cronológico)
               class_weight=pesos, callbacks=[paragem], verbose=2 if verbose else 0)

    score = modelo.predict(Xs_te, verbose=0).ravel()
    pred = (score >= 0.5).astype(int)
    modelo.save(MODELS_DIR / "tensorflow_mlp_opt.keras")
    linha = {"Modelo": "TensorFlow Neural Net", "Tipo": "Deep Learning", "Ficheiro": "tensorflow_mlp_opt.keras",
             "Escalado": True, "Treino": f"completo ({len(y_tr)})", **calcular_metricas(y_te, pred, score),
             "Melhores Hiperparâmetros": "Dense(64->32->16), Dropout(0.3/0.2), Adam(lr=0.0008), sigmoid, class_weight"}
    return pd.DataFrame([linha]), score


def executar_otimizacao():
    """Corre ML + TensorFlow e devolve (tabela, scores)."""
    tab_ml, scores = otimizar_modelos_ml()
    tab_tf, score_tf = treinar_tensorflow()
    scores["TensorFlow Neural Net"] = score_tf
    tabela = pd.concat([tab_ml, tab_tf], ignore_index=True)
    return tabela, scores


if __name__ == "__main__":
    tabela, _ = executar_otimizacao()
    print(tabela.drop(columns=["Melhores Hiperparâmetros"]).to_string(index=False))
