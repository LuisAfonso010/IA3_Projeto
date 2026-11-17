#!/usr/bin/env python3
"""
Sistema multiagente para previsão de confrontos profissionais de CS2.

Base: o melhor modelo global escolhido no notebook CS2_Modeling_Todas_Familias
(`resultados/modelo_melhor_global.joblib`). Não é preciso saber qual foi o
vencedor: o sistema carrega o ficheiro que o notebook guardou e lê o nome e a
accuracy em `resultados_todos_modelos.csv`.

Agentes (executados em sequência pelo Orquestrador):

  1. AgenteDados       valida e alinha as features com as que o modelo espera
  2. AgentePrevisao    modelo base -> P(equipa 1 vence o mapa)
  3. AgenteComite      os melhores modelos das outras famílias votam (2.ª opinião)
  4. AgenteExplicacao  que features empurram a previsão (perturbação local)
  5. AgenteSerie       converte P(mapa) em P(série) para BO1 / BO3 / BO5
  6. AgenteConfianca   junta margem + acordo do comité num nível de confiança
  7. AgenteRelator     escreve o resumo por jogo

Uso (linha de comandos):
  python cs2_multiagente.py --jogos novos_jogos.csv --pasta resultados \
      --treino cs2_matches_train.csv --formato bo3
  python cs2_multiagente.py --verificar cs2_matches_test.csv --pasta resultados

Uso (Colab / Python):
  from cs2_multiagente import prever
  saida, relatorio = prever(df_jogos, pasta='resultados',
                            treino='cs2_matches_train.csv', formato='bo3')

O CSV de entrada tem de ter as mesmas colunas de features do treino (sem
`target`). Colunas opcionais para identificação: equipa1/equipa2 (ou
team1/team2) e data.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from math import comb

import joblib
import numpy as np
import pandas as pd

FORMATOS = {"bo1": 1, "bo3": 3, "bo5": 5}
CANDIDATOS_EQUIPA1 = ["equipa1", "team1", "team_1", "team1_name", "nome_equipa1"]
CANDIDATOS_EQUIPA2 = ["equipa2", "team2", "team_2", "team2_name", "nome_equipa2"]


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def prob(modelo, X: pd.DataFrame) -> np.ndarray:
    """P(equipa 1 vence). Sem predict_proba (ex.: SVM) usa-se a função de
    decisão passada por uma sigmoide (aproximação, não calibrada)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-modelo.decision_function(X)))


def prob_serie(p, n_mapas: int):
    """P(vencer uma série à melhor de n_mapas), assumindo mapas independentes
    com a mesma probabilidade p."""
    m = n_mapas // 2 + 1
    return sum(comb(m - 1 + j, j) * p**m * (1 - p) ** j for j in range(m))


def colunas_do_modelo(modelo, treino: pd.DataFrame | None) -> list[str]:
    for attr in ("feature_names_in_", "feature_name_"):
        cols = getattr(modelo, attr, None)
        if cols is not None:
            return list(cols)
    if treino is not None:
        return [c for c in treino.columns if c != "target"]
    raise ValueError(
        "O modelo não guarda os nomes das features. Indica --treino com o "
        "cs2_matches_train.csv para eu as descobrir."
    )


def _primeira_coluna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    return next((c for c in candidatos if c in df.columns), None)


# --------------------------------------------------------------------------- #
# Contexto partilhado entre agentes
# --------------------------------------------------------------------------- #
@dataclass
class Contexto:
    jogos: pd.DataFrame
    formato: str
    modelo_base: object
    nome_base: str
    acc_base: float | None
    comite: dict
    colunas: list
    referencia: pd.Series          # valores "neutros" (mediana do treino)
    limites: pd.DataFrame | None   # min/max do treino
    nome1: pd.Series = None
    nome2: pd.Series = None
    X: pd.DataFrame = None
    saida: pd.DataFrame = None
    contribuicoes: pd.DataFrame = None
    avisos: list = field(default_factory=list)
    traco: list = field(default_factory=list)
    relatorio: str = ""
    max_relatorio: int = 20   # nº máximo de jogos no relatório de texto (o CSV leva todos)


class Agente:
    nome = "Agente"

    def executar(self, ctx: Contexto) -> None:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Agentes
# --------------------------------------------------------------------------- #
class AgenteDados(Agente):
    nome = "Dados"

    def executar(self, ctx):
        df = ctx.jogos
        em_falta = [c for c in ctx.colunas if c not in df.columns]
        if em_falta:
            raise ValueError(f"Faltam features no CSV de entrada: {em_falta}")

        X = df[ctx.colunas].apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            linhas = X.index[X.isna().any(axis=1)].tolist()
            raise ValueError(
                f"{len(linhas)} jogo(s) com valores em falta/não numéricos "
                f"(linhas {linhas[:10]}). Corrige antes de prever."
            )
        ctx.X = X

        c1 = _primeira_coluna(df, CANDIDATOS_EQUIPA1)
        c2 = _primeira_coluna(df, CANDIDATOS_EQUIPA2)
        ctx.nome1 = df[c1].astype(str) if c1 else pd.Series("Equipa 1", index=df.index)
        ctx.nome2 = df[c2].astype(str) if c2 else pd.Series("Equipa 2", index=df.index)

        # Coerência do Elo (se as três colunas existirem)
        if {"elo_team1", "elo_team2", "elo_diff"} <= set(X.columns):
            dif = X["elo_team1"] - X["elo_team2"]
            ok = np.isclose(X["elo_diff"], dif, atol=1e-6) | np.isclose(X["elo_diff"], -dif, atol=1e-6)
            if not ok.all():
                ctx.avisos.append(f"{(~ok).sum()} jogo(s) com elo_diff incoerente com elo_team1/elo_team2.")

        # Fora da distribuição de treino
        fora = pd.Series(False, index=X.index)
        if ctx.limites is not None:
            lo, hi = ctx.limites.loc["min", ctx.colunas], ctx.limites.loc["max", ctx.colunas]
            fora = ((X < lo) | (X > hi)).any(axis=1)
            if fora.any():
                ctx.avisos.append(
                    f"{fora.sum()} jogo(s) com features fora do intervalo visto no treino "
                    "(previsão menos fiável)."
                )

        ctx.saida = pd.DataFrame(index=df.index)
        ctx.saida["equipa1"] = ctx.nome1
        ctx.saida["equipa2"] = ctx.nome2
        if "data" in df.columns:
            ctx.saida["data"] = df["data"]
        ctx.saida["fora_treino"] = fora
        ctx.traco.append(f"{len(X)} jogos validados, {len(ctx.colunas)} features.")


class AgentePrevisao(Agente):
    nome = "Previsão"

    def executar(self, ctx):
        p = prob(ctx.modelo_base, ctx.X)
        ctx.saida["p_equipa1_mapa"] = p
        ctx.saida["favorita"] = np.where(p >= 0.5, ctx.nome1, ctx.nome2)
        if not hasattr(ctx.modelo_base, "predict_proba"):
            ctx.avisos.append(
                "O modelo base não devolve probabilidades; usei a função de decisão "
                "numa sigmoide (valores aproximados, não calibrados)."
            )
        ctx.traco.append(f"Modelo base: {ctx.nome_base}.")


class AgenteComite(Agente):
    nome = "Comité"

    def executar(self, ctx):
        if not ctx.comite:
            ctx.saida["acordo_comite"] = np.nan
            ctx.traco.append("Sem modelos de outras famílias; comité ignorado.")
            return
        voto_base = ctx.saida["p_equipa1_mapa"].to_numpy() >= 0.5
        acordos, probs = [], []
        for nome, m in ctx.comite.items():
            try:
                cols = getattr(m, "feature_names_in_", None)
                pm = prob(m, ctx.X[list(cols)] if cols is not None else ctx.X)
            except Exception as e:  # um modelo avariado não deve parar o sistema
                ctx.avisos.append(f"Comité: '{nome}' falhou ({type(e).__name__}); ignorado.")
                continue
            probs.append(pm)
            acordos.append((pm >= 0.5) == voto_base)
        if not acordos:
            ctx.saida["acordo_comite"] = np.nan
            return
        ctx.saida["acordo_comite"] = np.mean(acordos, axis=0)
        ctx.saida["n_comite"] = len(acordos)
        ctx.saida["p_comite_media"] = np.mean(probs, axis=0)
        ctx.traco.append(f"{len(acordos)} modelos de outras famílias consultados.")


class AgenteExplicacao(Agente):
    """Explicação local por perturbação: troca cada feature pelo valor neutro
    (mediana do treino) e mede quanto muda P(equipa 1 vence)."""

    nome = "Explicação"

    def executar(self, ctx):
        base = ctx.saida["p_equipa1_mapa"].to_numpy()
        contrib = {}
        for col in ctx.colunas:
            Xc = ctx.X.copy()
            Xc[col] = ctx.referencia[col]
            contrib[col] = base - prob(ctx.modelo_base, Xc)
        ctx.contribuicoes = pd.DataFrame(contrib, index=ctx.X.index)
        ctx.traco.append("Contribuições por feature calculadas (efeito individual vs. valor neutro).")


class AgenteSerie(Agente):
    nome = "Série"

    def executar(self, ctx):
        n = FORMATOS[ctx.formato]
        p = ctx.saida["p_equipa1_mapa"].to_numpy()
        ctx.saida["p_equipa1_serie"] = prob_serie(p, n)
        ctx.saida["formato"] = ctx.formato.upper()
        if n > 1:
            ctx.avisos.append(
                f"{ctx.formato.upper()}: assumi mapas independentes com a mesma probabilidade; "
                "o modelo não conhece o mapa jogado nem o veto."
            )
        ctx.traco.append(f"Probabilidade de série calculada ({ctx.formato.upper()}).")


class AgenteConfianca(Agente):
    nome = "Confiança"

    def __init__(self, margem_alta=0.15, margem_baixa=0.05, acordo_alto=0.75, acordo_baixo=0.5):
        self.margem_alta, self.margem_baixa = margem_alta, margem_baixa
        self.acordo_alto, self.acordo_baixo = acordo_alto, acordo_baixo

    def executar(self, ctx):
        margem = (ctx.saida["p_equipa1_mapa"] - 0.5).abs()
        acordo = ctx.saida["acordo_comite"].fillna(1.0)  # sem comité: não penaliza
        nivel = np.where(
            (margem >= self.margem_alta) & (acordo >= self.acordo_alto), "Alta",
            np.where((margem < self.margem_baixa) | (acordo < self.acordo_baixo), "Baixa", "Média"),
        )
        # Jogo fora do intervalo de treino: desce um nível
        desce = {"Alta": "Média", "Média": "Baixa", "Baixa": "Baixa"}
        nivel = np.where(ctx.saida["fora_treino"], [desce[n] for n in nivel], nivel)
        ctx.saida["margem"] = margem
        ctx.saida["confianca"] = nivel
        ctx.traco.append(
            f"Confiança: margem ≥{self.margem_alta:.0%} e acordo ≥{self.acordo_alto:.0%} = Alta; "
            f"margem <{self.margem_baixa:.0%} ou acordo <{self.acordo_baixo:.0%} = Baixa."
        )


class AgenteRelator(Agente):
    nome = "Relator"

    def executar(self, ctx):
        s = ctx.saida
        linhas = []
        cab = f"Modelo base: {ctx.nome_base}"
        if ctx.acc_base is not None:
            cab += f" (accuracy de teste no notebook: {ctx.acc_base:.1%})"
        linhas += [cab]
        if "target" in ctx.jogos.columns:
            real = ctx.jogos["target"].to_numpy()
            acc = float(((s["p_equipa1_mapa"].to_numpy() >= 0.5) == (real == 1)).mean())
            linhas.append(f"Accuracy do sistema neste conjunto ({len(s)} jogos, com target): {acc:.1%}")
        linhas.append("")
        resumos = []
        for i, (idx, r) in enumerate(s.iterrows(), 1):
            p1 = r["p_equipa1_mapa"]
            fav_mapa = p1 if p1 >= 0.5 else 1 - p1
            ps = r["p_equipa1_serie"]
            fav_serie = ps if p1 >= 0.5 else 1 - ps
            top = ctx.contribuicoes.loc[idx].sort_values(key=np.abs, ascending=False).head(3)
            fatores = ", ".join(f"{c} ({v * 100:+.1f}pp)" for c, v in top.items())
            acordo = r["acordo_comite"]
            comite = (f"{int(round(acordo * r['n_comite']))}/{int(r['n_comite'])} modelos concordam"
                      if pd.notna(acordo) else "sem comité")
            resumo = (f"{r['favorita']} favorita — {fav_mapa:.1%} por mapa, "
                      f"{fav_serie:.1%} em {r['formato']} | confiança {r['confianca']}")
            if r["fora_treino"]:
                resumo += " (features fora do intervalo de treino)"
            resumos.append(resumo)
            if i <= ctx.max_relatorio:
                linhas += [
                    f"#{i} {r['equipa1']} vs {r['equipa2']}",
                    f"   {resumo}",
                    f"   Comité: {comite} | Fatores (efeito na P da equipa 1): {fatores}",
                    "",
                ]
        if len(s) > ctx.max_relatorio:
            linhas += [f"... mais {len(s) - ctx.max_relatorio} jogos no CSV de saída.", ""]
        s["resumo"] = resumos
        if ctx.avisos:
            linhas.append("Avisos:")
            linhas += [f" - {a}" for a in dict.fromkeys(ctx.avisos)]
        ctx.relatorio = "\n".join(linhas)


# --------------------------------------------------------------------------- #
# Orquestrador
# --------------------------------------------------------------------------- #
class Orquestrador:
    def __init__(self, agentes: list[Agente] | None = None):
        self.agentes = agentes or [
            AgenteDados(), AgentePrevisao(), AgenteComite(), AgenteExplicacao(),
            AgenteSerie(), AgenteConfianca(), AgenteRelator(),
        ]

    def correr(self, ctx: Contexto, verbose: bool = False) -> Contexto:
        for ag in self.agentes:
            n0 = len(ctx.traco)
            ag.executar(ctx)
            if verbose:
                for t in ctx.traco[n0:]:
                    print(f"[{ag.nome}] {t}")
        return ctx


# --------------------------------------------------------------------------- #
# Carregamento e API principal
# --------------------------------------------------------------------------- #
def carregar_sistema(pasta: str, treino: str | None = None):
    caminho_base = os.path.join(pasta, "modelo_melhor_global.joblib")
    if not os.path.exists(caminho_base):
        raise FileNotFoundError(
            f"Não encontrei {caminho_base}. Corre a secção 7 do notebook de modelação."
        )
    base = joblib.load(caminho_base)

    nome_base, acc_base = type(base).__name__, None
    csv_res = os.path.join(pasta, "resultados_todos_modelos.csv")
    if os.path.exists(csv_res):
        res = pd.read_csv(csv_res)
        melhor = res.loc[res["Accuracy (teste)"].idxmax()]
        nome_base, acc_base = str(melhor["Modelo"]), float(melhor["Accuracy (teste)"])

    comite = {}
    for f in sorted(glob.glob(os.path.join(pasta, "modelo_*.joblib"))):
        if os.path.basename(f) == "modelo_melhor_global.joblib":
            continue
        m = joblib.load(f)
        if repr(m) == repr(base):  # é o próprio modelo base (melhor da sua família)
            continue
        comite[os.path.basename(f)[len("modelo_"):-len(".joblib")]] = m

    df_treino = pd.read_csv(treino) if treino else None
    colunas = colunas_do_modelo(base, df_treino)
    referencia = limites = None
    if df_treino is not None:
        feats = df_treino[colunas]
        referencia = feats.median()
        limites = feats.agg(["min", "max"])
    return base, nome_base, acc_base, comite, colunas, referencia, limites


def prever(jogos: pd.DataFrame, pasta: str = "resultados", treino: str | None = None,
           formato: str = "bo1", verbose: bool = False, max_relatorio: int = 20):
    """Corre o sistema multiagente. Devolve (DataFrame com previsões, relatório)."""
    formato = formato.lower()
    if formato not in FORMATOS:
        raise ValueError(f"Formato inválido '{formato}'. Usa um de {list(FORMATOS)}.")
    base, nome, acc, comite, colunas, referencia, limites = carregar_sistema(pasta, treino)

    avisos = []
    if referencia is None:
        missing = [c for c in colunas if c not in jogos.columns]
        if missing:
            raise ValueError(f"Faltam features no CSV de entrada: {missing}")
        referencia = jogos[colunas].apply(pd.to_numeric, errors="coerce").median()
        avisos.append("Sem --treino: o valor neutro da explicação é a mediana dos jogos fornecidos "
                      "e não foi possível verificar se os dados estão fora do intervalo de treino.")

    ctx = Contexto(jogos=jogos.reset_index(drop=True), formato=formato, modelo_base=base,
                   nome_base=nome, acc_base=acc, comite=comite, colunas=colunas,
                   referencia=referencia, limites=limites, avisos=avisos,
                   max_relatorio=max_relatorio)
    ctx = Orquestrador().correr(ctx, verbose=verbose)
    return ctx.saida, ctx.relatorio


def verificar_modelo_base(pasta: str, teste: str) -> bool:
    """Avalia TODOS os modelos guardados no conjunto de teste e confirma que o modelo
    base (modelo_melhor_global.joblib) é o de maior accuracy. Devolve True se for."""
    df = pd.read_csv(teste)
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    base, nome_base, acc_csv, *_ = carregar_sistema(pasta)
    linhas = []
    for f in sorted(glob.glob(os.path.join(pasta, "modelo_*.joblib"))):
        m = joblib.load(f)
        cols = getattr(m, "feature_names_in_", None)
        acc = float((m.predict(X[list(cols)] if cols is not None else X) == y).mean())
        linhas.append((os.path.basename(f), acc, repr(m) == repr(base)))
    linhas.sort(key=lambda t: -t[1])
    print(f"{'ficheiro':<42}{'accuracy teste':>15}")
    for nome, acc, e_base in linhas:
        print(f"{nome:<42}{acc:>14.2%}  {'<- modelo base' if e_base else ''}")
    acc_base = next(a for _, a, b in linhas if b) if any(b for *_, b in linhas) else None
    ok_topo = acc_base is not None and acc_base >= linhas[0][1] - 1e-12
    print(f"\nModelo base declarado: {nome_base}")
    if acc_base is not None:
        print(f"Accuracy recalculada do modelo base: {acc_base:.2%}"
              + (f" | no CSV do notebook: {acc_csv:.2%}" if acc_csv is not None else ""))
        if acc_csv is not None and abs(acc_base - acc_csv) > 2e-3:
            print("AVISO: a accuracy recalculada não bate com a do CSV (modelos/dados de execuções diferentes?).")
    print("VERIFICAÇÃO:", "OK — o modelo base é o de maior accuracy no teste." if ok_topo
          else "FALHOU — existe um modelo com accuracy superior à do modelo base.")
    return ok_topo


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sistema multiagente de previsão CS2")
    ap.add_argument("--jogos", help="CSV com as features dos jogos a prever")
    ap.add_argument("--verificar", metavar="TESTE_CSV",
                    help="confirma que o modelo base é o melhor (usa cs2_matches_test.csv) e termina")
    ap.add_argument("--pasta", default="resultados", help="pasta com os .joblib do notebook")
    ap.add_argument("--treino", help="cs2_matches_train.csv (valores neutros e limites)")
    ap.add_argument("--formato", default="bo1", choices=list(FORMATOS))
    ap.add_argument("--saida", default="previsoes.csv")
    ap.add_argument("--max-jogos", type=int, default=20, help="jogos mostrados no relatório (o CSV leva todos)")
    ap.add_argument("-v", "--verbose", action="store_true", help="mostra o rasto dos agentes")
    a = ap.parse_args(argv)

    if a.verificar:
        return 0 if verificar_modelo_base(a.pasta, a.verificar) else 1
    if not a.jogos:
        ap.error("indica --jogos (ou --verificar)")

    saida, relatorio = prever(pd.read_csv(a.jogos), a.pasta, a.treino, a.formato, a.verbose, a.max_jogos)
    saida.round(4).to_csv(a.saida, index=False)
    print(relatorio)
    print(f"\nPrevisões guardadas em {a.saida}")


if __name__ == "__main__":
    sys.exit(main())
