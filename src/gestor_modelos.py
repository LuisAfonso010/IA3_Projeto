"""
US02 - Gestor unificado de modelos.

Carrega os classificadores guardados pelo notebook de modelação (`modelo_*.joblib`),
identifica qual é o melhor modelo global e alinha as features em inferência
pelos nomes que cada modelo guarda (12 features: elo_*, win_rate_*, forma_recente_*, h2h_*).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def prob(modelo, X: pd.DataFrame) -> np.ndarray:
    """P(equipa 1 vence). Sem predict_proba (ex.: SVM) usa-se a função de decisão numa sigmoide."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-modelo.decision_function(X)))


def slug(texto: str) -> str:
    """Mesmo slug que o notebook de modelação usa nos nomes dos ficheiros."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def _colunas_do_modelo(modelo, treino: pd.DataFrame | None) -> list[str]:
    for attr in ("feature_names_in_", "feature_name_"):
        cols = getattr(modelo, attr, None)
        if cols is not None:
            return list(cols)
    if treino is not None:
        return [c for c in treino.columns if c != "target"]
    raise ValueError("O modelo não guarda os nomes das features; indica o CSV de treino (--treino).")


class GestorModelos:
    def __init__(self, pasta: str | Path = "resultados", treino: str | Path | None = None):
        self.pasta = Path(pasta)
        ficheiro_base = self.pasta / "modelo_melhor_global.joblib"
        if not ficheiro_base.exists():
            raise FileNotFoundError(
                f"Não encontrei {ficheiro_base}. Corre a secção 7 do notebook de modelação "
                "e aponta --pasta-modelos para a pasta com os .joblib."
            )
        self.base = joblib.load(ficheiro_base)

        # Nome e accuracy de cada família, lidos do CSV do notebook (se existir)
        self.descricao: dict[str, str] = {}
        self.acc: dict[str, float] = {}
        self.nome_base, self.acc_base = type(self.base).__name__, None
        csv = self.pasta / "resultados_todos_modelos.csv"
        if csv.exists():
            res = pd.read_csv(csv)
            melhor = res.loc[res["Accuracy (teste)"].idxmax()]
            self.nome_base, self.acc_base = str(melhor["Modelo"]), float(melhor["Accuracy (teste)"])
            for _, linha in res.loc[res.groupby("Família")["Accuracy (teste)"].idxmax()].iterrows():
                self.descricao[slug(linha["Família"])] = str(linha["Modelo"])
                self.acc[slug(linha["Família"])] = float(linha["Accuracy (teste)"])

        # Todos os classificadores guardados (o melhor de cada família)
        self.modelos: dict[str, object] = {}
        for f in sorted(self.pasta.glob("modelo_*.joblib")):
            if f.name != "modelo_melhor_global.joblib":
                self.modelos[f.stem[len("modelo_"):]] = joblib.load(f)
        self.chave_base = next((k for k, m in self.modelos.items() if repr(m) == repr(self.base)), None)
        if self.chave_base is None:  # o base não coincide com nenhum ficheiro de família
            self.chave_base = "melhor_global"
            self.modelos[self.chave_base] = self.base
        self.descricao[self.chave_base] = self.nome_base
        if self.acc_base is not None:
            self.acc[self.chave_base] = self.acc_base

        df_treino = pd.read_csv(treino) if treino else None
        self.colunas = _colunas_do_modelo(self.base, df_treino)
        self.referencia = self.limites = None
        if df_treino is not None:
            feats = df_treino[self.colunas]
            self.referencia = feats.median()
            self.limites = feats.agg(["min", "max"])

    # ------------------------------------------------------------------ #
    @property
    def chaves_comite(self) -> list[str]:
        return [k for k in self.modelos if k != self.chave_base]

    def nome_de(self, chave: str) -> str:
        return self.descricao.get(chave, chave)

    def alinhar(self, jogos: pd.DataFrame) -> pd.DataFrame:
        """Seleciona e ordena as features que os modelos esperam e valida os valores."""
        em_falta = [c for c in self.colunas if c not in jogos.columns]
        if em_falta:
            raise ValueError(f"Faltam features no CSV de entrada: {em_falta}")
        X = jogos[self.colunas].apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            linhas = X.index[X.isna().any(axis=1)].tolist()
            raise ValueError(f"{len(linhas)} jogo(s) com valores em falta/não numéricos (linhas {linhas[:10]}).")
        return X

    def prob_modelo(self, chave: str, X: pd.DataFrame) -> np.ndarray:
        m = self.modelos[chave]
        cols = getattr(m, "feature_names_in_", None)
        return prob(m, X[list(cols)] if cols is not None else X)

    def prob_base(self, X: pd.DataFrame) -> np.ndarray:
        return self.prob_modelo(self.chave_base, X)

    def prob_todos(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({k: self.prob_modelo(k, X) for k in self.modelos}, index=X.index)

    def resumo(self) -> str:
        acc = f" (accuracy de teste {self.acc_base:.1%})" if self.acc_base is not None else ""
        return (f"Modelo base: {self.nome_base}{acc} | {len(self.modelos)} classificadores carregados "
                f"({len(self.chaves_comite)} no comité) | {len(self.colunas)} features")

    def verificar(self, teste_csv: str | Path) -> tuple[bool, pd.DataFrame]:
        """Avalia todos os modelos no teste e confirma que o base é o de maior accuracy."""
        df = pd.read_csv(teste_csv)
        X, y = self.alinhar(df), df["target"].to_numpy()
        linhas = []
        for chave, m in self.modelos.items():
            cols = getattr(m, "feature_names_in_", None)
            acc = float((m.predict(X[list(cols)] if cols is not None else X) == y).mean())
            linhas.append({"modelo": self.nome_de(chave), "chave": chave, "accuracy_teste": acc,
                           "base": chave == self.chave_base})
        tabela = pd.DataFrame(linhas).sort_values("accuracy_teste", ascending=False).reset_index(drop=True)
        acc_base = float(tabela.loc[tabela["base"], "accuracy_teste"].iloc[0])
        return acc_base >= tabela["accuracy_teste"].max() - 1e-12, tabela
