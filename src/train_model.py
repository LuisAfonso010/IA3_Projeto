"""
Modelling - Dataset de Previsão de Pulsares (HTRU2)
Projeto IA III - ISTEC

Treina e compara todos os modelos de classificação estudados em IA I e IA II:
- Scikit-Learn: Regressão Logística, Árvore de Decisão, Random Forest,
  Gradient Boosting, KNN, SVM, Naive Bayes e MLP.
- Deep Learning: Rede Neuronal Densa (MLP) em TensorFlow/Keras.

Guarda o melhor modelo Scikit-Learn e o modelo TensorFlow na pasta 'models/'.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Fixar sementes para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)


def get_project_paths():
    """Resolve automaticamente os caminhos independentemente de correr em src/ ou notebooks/."""
    base_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
    if base_dir.name == "notebooks":
        base_dir = base_dir.parent
    data_path = base_dir / "data" / "processed" / "pulsar_data_train_prepared.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return data_path, models_dir


def build_tensorflow_model(input_dim: int) -> tf.keras.Model:
    """Cria uma Rede Neuronal Densa em TensorFlow/Keras para classificação binária."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_and_evaluate_all(data_path=None, models_dir=None):
    """Função principal que carrega dados, treina todos os modelos e guarda os melhores."""
    if data_path is None or models_dir is None:
        default_data, default_models = get_project_paths()
        data_path = data_path or default_data
        models_dir = models_dir or default_models

    # 1. Carregar os dados já preparados
    train = pd.read_csv(data_path)
    X = train.drop(columns=["target_class"])
    y = train["target_class"].astype(int)

    # 2. Divisão estratificada (80% treino / 20% validação)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Treino: {X_train.shape[0]} linhas | Validação: {X_val.shape[0]} linhas\n")

    # 3. Dicionário com os modelos Scikit-Learn (IA I e IA II)
    sklearn_models = {
        "Regressão Logística": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Árvore de Decisão": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "SVM (RBF)": SVC(class_weight="balanced", probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "MLP (Scikit-Learn)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42),
    }

    results = []
    best_f1 = -1.0
    best_model_name = ""
    best_model_obj = None

    # 4. Treinar e avaliar cada modelo Scikit-Learn
    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        auc = roc_auc_score(y_val, proba)

        results.append({
            "Modelo": name,
            "Biblioteca": "Scikit-Learn",
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        })

        print("=" * 60)
        print(f"MODELO: {name}")
        print("=" * 60)
        print(classification_report(y_val, preds, target_names=["não-pulsar", "pulsar"]))
        print("Matriz de confusão:\n", confusion_matrix(y_val, preds))
        print(f"ROC-AUC: {auc:.4f}\n")

        # Guardar referência do melhor modelo segundo o F1-Score (ideal para classes desequilibradas)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    # 5. Treinar e avaliar o modelo Deep Learning (TensorFlow / Keras)
    print("=" * 60)
    print("MODELO: Rede Neuronal Densa (TensorFlow / Keras)")
    print("=" * 60)

    # Calcular pesos das classes para compensar o desequilíbrio (90.8% vs 9.2%)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    tf_model = build_tensorflow_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    tf_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0,
    )

    tf_proba = tf_model.predict(X_val, verbose=0).ravel()
    tf_preds = (tf_proba >= 0.5).astype(int)

    tf_f1 = f1_score(y_val, tf_preds)
    tf_prec = precision_score(y_val, tf_preds)
    tf_rec = recall_score(y_val, tf_preds)
    tf_auc = roc_auc_score(y_val, tf_proba)

    results.append({
        "Modelo": "Rede Neuronal (MLP)",
        "Biblioteca": "TensorFlow",
        "Precision": round(tf_prec, 4),
        "Recall": round(tf_rec, 4),
        "F1-Score": round(tf_f1, 4),
        "ROC-AUC": round(tf_auc, 4),
    })

    print(classification_report(y_val, tf_preds, target_names=["não-pulsar", "pulsar"]))
    print("Matriz de confusão:\n", confusion_matrix(y_val, tf_preds))
    print(f"ROC-AUC: {tf_auc:.4f}\n")

    # 6. Guardar modelos na pasta 'models/'
    best_sk_path = models_dir / "best_sklearn_model.joblib"
    tf_path = models_dir / "tensorflow_pulsar_model.keras"

    joblib.dump(best_model_obj, best_sk_path)
    tf_model.save(tf_path)

    # Guardar também os nomes das colunas para os agentes usarem depois
    joblib.dump(list(X.columns), models_dir / "feature_names.joblib")

    summary_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("=" * 60)
    print("RESUMO COMPARATIVO DE TODOS OS MODELOS")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nMelhor modelo Scikit-Learn ({best_model_name}) guardado em: {best_sk_path}")
    print(f"Modelo TensorFlow guardado em: {tf_path}")

    return summary_df


if __name__ == "__main__":
    train_and_evaluate_all()