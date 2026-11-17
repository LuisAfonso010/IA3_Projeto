"""
Model Tuning - Dataset de Previsão de Pulsares (HTRU2)
Projeto IA III - ISTEC

Melhora o Random Forest do train_model.py afinando os seus hiperparâmetros
com GridSearchCV: testa várias combinações automaticamente e escolhe a
melhor, usando validação cruzada (cross-validation) em vez de uma única
divisão treino/validação, para um resultado mais fiável.

Métrica escolhida: F1-score da classe "pulsar" (não accuracy), porque o
dataset é desequilibrado.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Carregar os dados já preparados --------------------------------------
train = pd.read_csv("pulsar_data_train_prepared.csv")

X = train.drop(columns=["target_class"])
y = train["target_class"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Definir as combinações de parâmetros a testar --------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
}

base_model = RandomForestClassifier(class_weight="balanced", random_state=42)

# cv=5 -> divide o treino em 5 partes e testa cada combinação 5 vezes,
# trocando qual parte serve de validação, para um resultado mais robusto.
grid_search = GridSearchCV(
    base_model,
    param_grid,
    scoring="f1",  # foca-se na classe pulsar (label 1)
    cv=5,
    n_jobs=-1,
)

print("A testar combinações de parâmetros (pode demorar um pouco)...")
grid_search.fit(X_train, y_train)

print("\n" + "=" * 60)
print("MELHORES PARÂMETROS ENCONTRADOS")
print("=" * 60)
print(grid_search.best_params_)
print(f"Melhor F1-score (validação cruzada): {grid_search.best_score_:.4f}")

# 3. Avaliar o melhor modelo no conjunto de validação -----------------------
best_model = grid_search.best_estimator_
preds = best_model.predict(X_val)
proba = best_model.predict_proba(X_val)[:, 1]

print("\n" + "=" * 60)
print("RESULTADO DO MELHOR MODELO (conjunto de validação)")
print("=" * 60)
print(classification_report(y_val, preds, target_names=["não-pulsar", "pulsar"]))
print("Matriz de confusão (linhas=real, colunas=previsto):")
print(confusion_matrix(y_val, preds))
print(f"ROC-AUC: {roc_auc_score(y_val, proba):.4f}")
