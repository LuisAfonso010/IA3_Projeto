"""
EDA (Exploratory Data Analysis) - Dataset de Previsão de Pulsares (HTRU2)
Projeto IA III - ISTEC

Este script faz a análise exploratória inicial do dataset de treino:
- Estrutura e tipos de dados
- Valores em falta
- Desequilíbrio de classes (pulsar vs não-pulsar)
- Estatísticas descritivas por classe
- Correlação entre as features
- Distribuição de cada feature por classe
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados -------------------------------------------------
# os CSVs devem estar na mesma pasta que este script
train = pd.read_csv("pulsar_data_train.csv")
test = pd.read_csv("pulsar_data_test.csv")

# limpar espaços nos nomes das colunas (o CSV original vem com espaços à frente)
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

print("=" * 60)
print("FORMA DOS DATASETS")
print("=" * 60)
print(f"Treino: {train.shape[0]} linhas, {train.shape[1]} colunas")
print(f"Teste:  {test.shape[0]} linhas, {test.shape[1]} colunas")

# 2. Valores em falta ----------------------------------------------------
print("\n" + "=" * 60)
print("VALORES EM FALTA (treino)")
print("=" * 60)
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
print(pd.DataFrame({"em_falta": missing, "%": missing_pct}))

# 3. Desequilíbrio de classes ---------------------------------------------
print("\n" + "=" * 60)
print("DISTRIBUIÇÃO DA CLASSE (target_class)")
print("=" * 60)
counts = train["target_class"].value_counts()
pct = (counts / len(train) * 100).round(2)
print(pd.DataFrame({"contagem": counts, "%": pct}))

# 4. Estatísticas descritivas por classe ----------------------------------
print("\n" + "=" * 60)
print("ESTATÍSTICAS DESCRITIVAS POR CLASSE")
print("=" * 60)
print(train.groupby("target_class").mean(numeric_only=True).round(3).T)

# 5. Gráfico: balanço de classes -------------------------------------------
plt.figure(figsize=(5, 4))
counts.plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Distribuição da classe (0 = não-pulsar, 1 = pulsar)")
plt.xlabel("target_class")
plt.ylabel("nº de estrelas")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_balance.png", dpi=120)
plt.close()

# 6. Matriz de correlação ---------------------------------------------------
plt.figure(figsize=(8, 6))
corr = train.drop(columns=["target_class"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlação (Pearson) entre as features")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=120)
plt.close()

# 7. Distribuição de cada feature por classe --------------------------------
features = [c for c in train.columns if c != "target_class"]
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.kdeplot(data=train, x=col, hue="target_class", fill=True, ax=axes[i], common_norm=False)
    axes[i].set_title(col, fontsize=10)
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=120)
plt.close()

print("\nGráficos guardados: class_balance.png, correlation_matrix.png, feature_distributions.png")
