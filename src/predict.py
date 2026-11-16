"""Módulo de inferência: carrega o modelo campeão (via manifesto) e classifica transações."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_preparation import criar_features


class PredictorFraude:
    """Carrega o modelo campeão e gera previsões de fraude em produção."""

    ROTULOS = {0: "Legítima", 1: "Fraude"}

    def __init__(self, pasta_models):
        self.pasta_models = Path(pasta_models)
        caminho_manifesto = self.pasta_models / "models_manifest.json"
        if not caminho_manifesto.exists():
            raise FileNotFoundError(f"Ficheiro de manifesto não encontrado em: {caminho_manifesto}")

        with open(caminho_manifesto, "r", encoding="utf-8") as f:
            self.manifesto = json.load(f)

        campeao = self.manifesto["melhor_modelo"]
        self.nome_campeao = campeao["nome"]
        self.ficheiro_campeao = campeao["ficheiro"]
        self.escalado = campeao.get("escalado", True)
        self.features = self.manifesto["features"]
        self.limiar = self.manifesto.get("limiar_decisao", 0.5)

        self.scaler = joblib.load(self.pasta_models / "scaler.joblib")
        caminho_modelo = self.pasta_models / self.ficheiro_campeao
        self.is_tf = self.ficheiro_campeao.endswith(".keras")
        if self.is_tf:
            import tensorflow as tf
            self.modelo = tf.keras.models.load_model(caminho_modelo)
        else:
            self.modelo = joblib.load(caminho_modelo)

    def _score(self, df_in: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df_in) if self.escalado else df_in
        if self.is_tf:
            return self.modelo.predict(X, verbose=0).ravel()
        return self.modelo.predict_proba(X)[:, 1]

    def prever(self, transacao: dict) -> dict:
        """Recebe {Time, V1..V28, Amount} (ou já com Hour/Amount_log) e devolve a previsão."""
        faltam = [c for c in ("V%d" % i for i in range(1, 29)) if c not in transacao]
        if faltam or "Amount" not in transacao or ("Time" not in transacao and "Hour" not in transacao):
            raise ValueError(f"Transação incompleta (faltam: {faltam or 'Amount/Time'})")
        df_in = criar_features(pd.DataFrame([transacao]))[self.features]
        p = float(self._score(df_in)[0])
        classe = int(p >= self.limiar)
        return {"modelo": self.nome_campeao, "previsao": self.ROTULOS[classe],
                "probabilidade_fraude_%": round(p * 100, 2), "limiar_%": round(self.limiar * 100, 1)}

    def prever_lote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classifica várias transações de uma vez (DataFrame com as colunas originais)."""
        df_in = criar_features(df)[self.features]
        p = self._score(df_in)
        return pd.DataFrame({"prob_fraude": p, "previsao": (p >= self.limiar).astype(int)}, index=df.index)
