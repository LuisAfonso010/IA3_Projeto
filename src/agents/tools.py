"""Ferramentas (tools) partilhadas pelos agentes. São funções puras sobre o modelo campeão."""
import json

import pandas as pd

from src.config import MANIFEST_FILE, MODELS_DIR
from src.predict import PredictorFraude

_predictor = None


def get_predictor() -> PredictorFraude:
    """Carrega o modelo campeão uma única vez (singleton)."""
    global _predictor
    if _predictor is None:
        _predictor = PredictorFraude(MODELS_DIR)
    return _predictor


def classificar_transacao(transacao: dict) -> dict:
    """Skill 'classificar_transacao': 1 transação -> previsão + probabilidade."""
    return get_predictor().prever(transacao)


def classificar_lote(df: pd.DataFrame) -> pd.DataFrame:
    """Classifica um lote e junta Amount/Hour para análise posterior."""
    res = get_predictor().prever_lote(df)
    out = res.join(df[[c for c in ("Amount", "Hour", "Class") if c in df.columns]])
    return out


def gravidade(prob: float) -> str:
    return "CRÍTICO" if prob >= 0.9 else "ALTO" if prob >= 0.7 else "MÉDIO"


def analisar_lote(res: pd.DataFrame, top_n: int = 5) -> dict:
    """Skill 'analisar_lote': estatísticas do dia e alertas ordenados por gravidade."""
    alertas = res[res["previsao"] == 1].sort_values("prob_fraude", ascending=False)
    factos = {
        "n_transacoes": int(len(res)),
        "n_alertas": int(len(alertas)),
        "taxa_alertas_%": round(len(alertas) / max(len(res), 1) * 100, 3),
        "valor_total": round(float(res["Amount"].sum()), 2),
        "valor_em_risco": round(float(alertas["Amount"].sum()), 2),
        "prob_media_alertas_%": round(float(alertas["prob_fraude"].mean() * 100), 1) if len(alertas) else 0.0,
        "hora_pico_alertas": int(alertas["Hour"].mode().iloc[0]) if len(alertas) else None,
        "alertas_por_gravidade": alertas["prob_fraude"].map(gravidade).value_counts().to_dict(),
        "top_alertas": [{"id": int(i), "prob_%": round(float(r.prob_fraude) * 100, 1), "valor": round(float(r.Amount), 2),
                         "hora": int(r.Hour), "gravidade": gravidade(r.prob_fraude)}
                        for i, r in alertas.head(top_n).iterrows()],
    }
    if "Class" in res.columns:  # só em simulação (rótulo real conhecido)
        tp = int(((res["previsao"] == 1) & (res["Class"] == 1)).sum())
        fn = int(((res["previsao"] == 0) & (res["Class"] == 1)).sum())
        factos["simulacao"] = {"fraudes_reais": tp + fn, "detetadas": tp, "falhadas": fn,
                               "falsos_positivos": int(((res["previsao"] == 1) & (res["Class"] == 0)).sum())}
    return factos


def cartao_modelo() -> dict:
    """Skill 'cartao_modelo': metadados do modelo campeão (manifesto)."""
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        m = json.load(f)
    return {"modelo": m["melhor_modelo"], "limiar": m["limiar_decisao"], "treino": m["data_avaliacao"],
            "n_modelos_comparados": len(m["resumo_modelos"])}
