"""
Modelling (baseline) - Dataset de Deteção de Fraude em Cartões de Crédito
Projeto IA III - ISTEC

Usa os dados já preparados para treinar um baseline com dois modelos:
- Regressão Logística
- Random Forest

Como o dataset é extremamente desequilibrado (0.17% fraudes) usamos class_weight="balanced"
e avaliamos com Precision/Recall/F1 da classe fraude, ROC-AUC e PR-AUC (Average Precision)
em vez de accuracy (um modelo que diga sempre "legítima" teria 99.8% de accuracy).
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from src.config import FEATURES, RANDOM_STATE, TARGET, TEST_FILE, TRAIN_FILE


def carregar_preparados():
    """Devolve X_train, y_train, X_test, y_test (features sem escala)."""
    treino, teste = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
    return treino[FEATURES], treino[TARGET], teste[FEATURES], teste[TARGET]


def calcular_metricas(y_true, y_pred, y_score) -> dict:
    """Métricas orientadas à classe minoritária (fraude = 1)."""
    return {
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision Fraude (%)": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall Fraude (%)": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1 Fraude (%)": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "ROC-AUC (%)": round(roc_auc_score(y_true, y_score) * 100, 2),
        "PR-AUC (%)": round(average_precision_score(y_true, y_score) * 100, 2),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def treinar_baseline(verbose: bool = True) -> pd.DataFrame:
    from src.data_preparation import ajustar_scaler  # evita import circular em notebooks
    X_tr, y_tr, X_te, y_te = carregar_preparados()
    scaler = ajustar_scaler(pd.read_csv(TRAIN_FILE))
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    modelos = {
        "Regressão Logística": (LogisticRegression(class_weight="balanced", max_iter=1000), True),
        "Random Forest": (RandomForestClassifier(class_weight="balanced_subsample", n_estimators=100,
                                                 max_depth=10, n_jobs=-1, random_state=RANDOM_STATE), False),
    }
    linhas = []
    for nome, (modelo, escalado) in modelos.items():
        a, b = (X_tr_s, X_te_s) if escalado else (X_tr, X_te)
        modelo.fit(a, y_tr)
        pred, score = modelo.predict(b), modelo.predict_proba(b)[:, 1]
        linhas.append({"Modelo": nome, **calcular_metricas(y_te, pred, score)})
        if verbose:
            print("=" * 60, f"\n{nome}\n", "=" * 60, sep="")
            print("Matriz de confusão (linhas=real, colunas=previsto):")
            print(confusion_matrix(y_te, pred))
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    print(treinar_baseline().to_string(index=False))
