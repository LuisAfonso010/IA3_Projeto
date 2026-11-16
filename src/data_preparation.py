"""
Data Preparation - Dataset de Deteção de Fraude em Cartões de Crédito
Projeto IA III - ISTEC (CRISP-ML: Data Engineering)

Trata os problemas identificados na EDA:
- 1081 linhas duplicadas                  -> removidas
- Sem valores nulos                        -> verificação defensiva (mediana do treino)
- 'Time' (segundos) não generaliza         -> substituída por 'Hour' (hora do dia)
- 'Amount' muito assimétrico (skew ~16)    -> acrescentada 'Amount_log' (log1p)
- Dados com componente temporal            -> divisão cronológica 80% / 20% (sem shuffle)
- Escalas diferentes                       -> StandardScaler ajustado SÓ no treino

Nota sobre o desequilíbrio (0.17% de fraudes): NÃO se faz oversampling aqui; é tratado
na modelação (class_weight / amostras), para não contaminar o conjunto de teste.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (FEATURES, MODELS_DIR, RAW_FILE, TARGET, TEST_FILE, TEST_SIZE,
                        TRAIN_FILE)


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicados e linhas sem alvo/tempo; ordena cronologicamente."""
    antes = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET, "Time"])
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    print(f"[LIMPEZA CONCLUÍDA] {antes} -> {len(df)} registos ({antes - len(df)} duplicados removidos)")
    return df


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: Hour (hora do dia) e Amount_log. Funciona também para 1 transação."""
    out = df.copy()
    if "Hour" not in out.columns:
        out["Hour"] = (out["Time"] // 3600 % 24).astype(int)
    if "Amount_log" not in out.columns:
        out["Amount_log"] = np.log1p(out["Amount"].clip(lower=0))
    return out


def dividir_temporal(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Os 80% mais antigos formam o treino; os 20% mais recentes o teste (evita data leakage)."""
    corte = int(len(df) * (1 - test_size))
    treino, teste = df.iloc[:corte].copy(), df.iloc[corte:].copy()
    print(f"Treino: {len(treino)} ({int(treino[TARGET].sum())} fraudes) | "
          f"Teste: {len(teste)} ({int(teste[TARGET].sum())} fraudes)")
    return treino, teste


def tratar_nulos(treino: pd.DataFrame, teste: pd.DataFrame):
    """Imputação pela mediana do treino (defensivo; o dataset original não tem nulos)."""
    medianas = treino[FEATURES].median()
    treino[FEATURES] = treino[FEATURES].fillna(medianas)
    teste[FEATURES] = teste[FEATURES].fillna(medianas)
    return treino, teste


def ajustar_scaler(treino: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(treino[FEATURES])
    return scaler


def preparar_pipeline(guardar: bool = True):
    """Pipeline completo: raw -> limpo -> features -> split temporal -> CSVs + scaler."""
    df = pd.read_csv(RAW_FILE)
    df = limpar_dados(df)
    df = criar_features(df)
    treino, teste = dividir_temporal(df)
    treino, teste = tratar_nulos(treino, teste)
    scaler = ajustar_scaler(treino)

    colunas = FEATURES + [TARGET]
    if guardar:
        MODELS_DIR.mkdir(exist_ok=True)
        TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        treino[colunas].round(6).to_csv(TRAIN_FILE, index=False)
        teste[colunas].round(6).to_csv(TEST_FILE, index=False)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        joblib.dump(FEATURES, MODELS_DIR / "features_list.joblib")
        print(f"Guardado: {TRAIN_FILE.name}, {TEST_FILE.name}, scaler.joblib, features_list.joblib")
    return treino[colunas], teste[colunas], scaler


if __name__ == "__main__":
    tr, te, _ = preparar_pipeline()
    print(f"\nDimensões finais -> treino {tr.shape} | teste {te.shape}")
