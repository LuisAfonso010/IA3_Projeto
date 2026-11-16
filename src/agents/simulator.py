"""Simulação de situações para testar o sistema: dia normal, ataque (rajada de fraudes) e deriva de valores."""
import pandas as pd

from src.config import RANDOM_STATE, TARGET, TEST_FILE


def carregar_teste() -> pd.DataFrame:
    return pd.read_csv(TEST_FILE)


def cenario(nome: str, n: int = 3000) -> pd.DataFrame:
    """Devolve um lote de transações reais do teste, montado para cada cenário."""
    df = carregar_teste()
    fr, leg = df[df[TARGET] == 1], df[df[TARGET] == 0]
    if nome == "dia_normal":
        lote = pd.concat([leg.sample(n - 3, random_state=RANDOM_STATE), fr.sample(3, random_state=RANDOM_STATE)])
    elif nome == "ataque":
        lote = pd.concat([leg.sample(n - 40, random_state=RANDOM_STATE), fr.sample(min(40, len(fr)), random_state=RANDOM_STATE)])
    elif nome == "deriva_valores":
        lote = leg.sample(n, random_state=RANDOM_STATE).copy()
        lote["Amount"] = lote["Amount"] * 8  # valores anormalmente altos (Amount_log mantém-se -> testa robustez)
    else:
        raise ValueError(f"Cenário desconhecido: {nome}")
    return lote.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


CENARIOS = ("dia_normal", "ataque", "deriva_valores")
