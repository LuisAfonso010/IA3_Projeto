"""
Data Preparation - Dataset de Previsão de Pulsares (HTRU2)
Projeto IA III - ISTEC

Este script trata os problemas identificados na EDA:
- Valores em falta em 3 colunas (imputação pela mediana do treino)
- Normalização das features (StandardScaler ajustado só no treino)

Nota sobre o desequilíbrio de classes: fica para o próximo passo (treino do
modelo), tratado com class_weight ou SMOTE - não se aplica ainda aqui.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados -------------------------------------------------
# os CSVs devem estar na mesma pasta que este script
train = pd.read_csv("pulsar_data_train.csv")
test = pd.read_csv("pulsar_data_test.csv")

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

feature_cols = [c for c in train.columns if c != "target_class"]

# 2. Tratar valores em falta ---------------------------------------------
# Usamos a mediana (mais robusta a outliers do que a média) e ajustamos
# o imputer SÓ no treino, para depois aplicar o mesmo valor ao teste
# (evita "fuga de informação" do teste para o treino).
imputer = SimpleImputer(strategy="median")
train[feature_cols] = imputer.fit_transform(train[feature_cols])
test[feature_cols] = imputer.transform(test[feature_cols])

print("=" * 60)
print("VALORES EM FALTA DEPOIS DA IMPUTAÇÃO")
print("=" * 60)
print("Treino:", train[feature_cols].isnull().sum().sum(), "valores em falta")
print("Teste: ", test[feature_cols].isnull().sum().sum(), "valores em falta")

# 3. Normalizar as features -----------------------------------------------
# StandardScaler transforma cada coluna para média 0 e desvio-padrão 1.
# Também ajustado só no treino, pela mesma razão.
scaler = StandardScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

print("\n" + "=" * 60)
print("ESTATÍSTICAS APÓS NORMALIZAÇÃO (treino)")
print("=" * 60)
print(train[feature_cols].describe().round(3).loc[["mean", "std"]])

# 4. Guardar os dados preparados -------------------------------------------
train.to_csv("pulsar_data_train_prepared.csv", index=False)
test.to_csv("pulsar_data_test_prepared.csv", index=False)

print("\nGuardado: pulsar_data_train_prepared.csv, pulsar_data_test_prepared.csv")
