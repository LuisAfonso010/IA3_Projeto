"""
Evaluation - Dataset de Deteção de Fraude em Cartões de Crédito
Projeto IA III - ISTEC (CRISP-ML: Quality Assurance / Evaluation)

Carrega todos os modelos persistidos em models/, avalia-os no conjunto de teste (20% mais
recente), escolhe o modelo campeão e escreve o manifesto models/models_manifest.json,
que é o contrato entre a fase de modelação e o deployment (src/predict.py).

Critério do campeão: maior PR-AUC (Average Precision) - independente do limiar e adequada
a classes muito desequilibradas. O F1 da fraude (limiar 0.5) serve de desempate/leitura.
"""
import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from src.config import DOCS_DIR, FEATURES, MANIFEST_FILE, MODELS_DIR
from src.train_model import calcular_metricas, carregar_preparados
from src.tune_model import definir_grelhas

FICHEIROS_MODELOS = {nome: (ficheiro, escalado) for nome, (_, _, escalado, ficheiro) in definir_grelhas().items()}
FICHEIROS_MODELOS["TensorFlow Neural Net"] = ("tensorflow_mlp_opt.keras", True)


def hiperparametros(nome: str, modelo) -> str:
    if nome == "TensorFlow Neural Net":
        return "Dense(64->32->16), Dropout(0.3/0.2), Adam(lr=0.0008), sigmoid, class_weight"
    chaves = definir_grelhas()[nome][1].keys()
    return str({k: v for k, v in modelo.get_params().items() if k in chaves})


def prever_scores(modelo, X, is_tf: bool):
    return modelo.predict(X, verbose=0).ravel() if is_tf else modelo.predict_proba(X)[:, 1]


def avaliar_modelos_guardados():
    """Devolve (df_eval ordenado por PR-AUC, dict de scores por modelo, y_test)."""
    _, _, X_te, y_te = carregar_preparados()
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_te_s = scaler.transform(X_te)

    linhas, scores = [], {}
    for nome, (ficheiro, escalado) in FICHEIROS_MODELOS.items():
        caminho = MODELS_DIR / ficheiro
        if not caminho.exists():
            print(f"[AVISO] modelo em falta: {ficheiro}")
            continue
        is_tf = ficheiro.endswith(".keras")
        if is_tf:
            import tensorflow as tf
            modelo = tf.keras.models.load_model(caminho)
        else:
            modelo = joblib.load(caminho)
        score = prever_scores(modelo, X_te_s if escalado else X_te, is_tf)
        scores[nome] = score
        linhas.append({"Modelo": nome, "Tipo": "Deep Learning" if is_tf else "Machine Learning",
                       "Ficheiro": ficheiro, "Escalado": escalado,
                       **calcular_metricas(y_te, (score >= 0.5).astype(int), score),
                       "Melhores Hiperparâmetros": hiperparametros(nome, modelo)})
    df_eval = pd.DataFrame(linhas).sort_values(["PR-AUC (%)", "F1 Fraude (%)"], ascending=False).reset_index(drop=True)
    return df_eval, scores, y_te


def guardar_manifesto(df_eval: pd.DataFrame, limiar: float = 0.5) -> dict:
    melhor = df_eval.iloc[0]
    meta = {
        "data_avaliacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "melhor_modelo": {"nome": melhor["Modelo"], "ficheiro": melhor["Ficheiro"],
                          "escalado": bool(melhor["Escalado"]), "pr_auc": float(melhor["PR-AUC (%)"]),
                          "f1_fraude": float(melhor["F1 Fraude (%)"]), "recall_fraude": float(melhor["Recall Fraude (%)"]),
                          "precision_fraude": float(melhor["Precision Fraude (%)"])},
        "criterio_selecao": "maior PR-AUC no conjunto de teste temporal",
        "limiar_decisao": limiar,
        "classes": [0, 1],
        "rotulos": {"0": "Legítima", "1": "Fraude"},
        "features": FEATURES,
        "resumo_modelos": df_eval.drop(columns=["Treino"], errors="ignore").to_dict(orient="records"),
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"Manifesto exportado para: {MANIFEST_FILE}")
    print(f"Modelo Campeão: {melhor['Modelo']} ({melhor['Ficheiro']})")
    return meta


def graficos_avaliacao(df_eval, scores, y_te, guardar: bool = True):
    """Curvas Precision-Recall de todos os modelos + matriz de confusão do campeão."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for nome, score in scores.items():
        PrecisionRecallDisplay.from_predictions(y_te, score, name=nome, ax=ax)
    ax.set_title("Curvas Precision-Recall (conjunto de teste)")
    plt.tight_layout()
    if guardar:
        fig.savefig(DOCS_DIR / "pr_curves.png", dpi=120)

    campeao = df_eval.iloc[0]["Modelo"]
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, (scores[campeao] >= 0.5).astype(int), ax=ax2,
                                            display_labels=["Legítima", "Fraude"], cmap="Blues")
    ax2.set_title(f"Matriz de confusão - {campeao}")
    plt.tight_layout()
    if guardar:
        fig2.savefig(DOCS_DIR / "confusion_matrix_champion.png", dpi=120)


if __name__ == "__main__":
    tab, sc, y = avaliar_modelos_guardados()
    print(tab.drop(columns=["Melhores Hiperparâmetros"]).to_string(index=False))
    guardar_manifesto(tab)
    graficos_avaliacao(tab, sc, y)
