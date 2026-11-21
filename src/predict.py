"""
US01 - Núcleo de previsão (partilhado pelos modos `hf` e `crewai`).

Dado um DataFrame de jogos, devolve uma `FichaJogo` por jogo com: previsão do modelo base,
opinião do comité (restantes classificadores), explicação por perturbação, probabilidade de
série (BO1/BO3/BO5) e nível de confiança. Tudo numérico e determinístico: os agentes de
linguagem só escrevem texto POR CIMA destes números.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import comb

import numpy as np
import pandas as pd

FORMATOS = {"bo1": 1, "bo3": 3, "bo5": 5}
CANDIDATOS_EQUIPA1 = ["equipa1", "team1", "team_1", "team1_name", "nome_equipa1"]
CANDIDATOS_EQUIPA2 = ["equipa2", "team2", "team_2", "team2_name", "nome_equipa2"]
LIMIARES = dict(margem_alta=0.15, margem_baixa=0.05, acordo_alto=0.75, acordo_baixo=0.5)


def prob_serie(p, n_mapas: int):
    """P(vencer uma série à melhor de n_mapas), assumindo mapas independentes com o mesmo p."""
    m = n_mapas // 2 + 1
    return sum(comb(m - 1 + j, j) * p**m * (1 - p) ** j for j in range(m))


@dataclass
class FichaJogo:
    indice: int
    equipa1: str
    equipa2: str
    formato: str
    modelo_base: str
    acc_base: float | None
    p_equipa1_mapa: float
    p_equipa1_serie: float
    favorita: str
    p_fav_mapa: float
    p_fav_serie: float
    margem: float
    confianca: str
    n_comite: int
    n_concordam: int
    votos: dict = field(default_factory=dict)      # modelo -> P(equipa 1 vence o mapa)
    fatores: list = field(default_factory=list)    # [(feature, efeito em pp na P da equipa 1)]
    features: dict = field(default_factory=dict)
    fora_treino: bool = False


def nivel_confianca(margem: float, acordo: float, fora_treino: bool) -> str:
    L = LIMIARES
    if margem >= L["margem_alta"] and acordo >= L["acordo_alto"]:
        nivel = "Alta"
    elif margem < L["margem_baixa"] or acordo < L["acordo_baixo"]:
        nivel = "Baixa"
    else:
        nivel = "Média"
    if fora_treino:  # jogo fora do intervalo de treino: desce um nível
        nivel = {"Alta": "Média", "Média": "Baixa", "Baixa": "Baixa"}[nivel]
    return nivel


def analisar_jogos(gestor, jogos: pd.DataFrame, formato: str = "bo1", n_fatores: int = 3):
    """Devolve (lista de FichaJogo, lista de avisos)."""
    formato = formato.lower()
    if formato not in FORMATOS:
        raise ValueError(f"Formato inválido '{formato}'. Usa um de {list(FORMATOS)}.")
    jogos = jogos.reset_index(drop=True)
    X = gestor.alinhar(jogos)
    avisos: list[str] = []
    n_mapas = FORMATOS[formato]

    c1 = next((c for c in CANDIDATOS_EQUIPA1 if c in jogos.columns), None)
    c2 = next((c for c in CANDIDATOS_EQUIPA2 if c in jogos.columns), None)
    nome1 = jogos[c1].astype(str) if c1 else pd.Series("Equipa 1", index=jogos.index)
    nome2 = jogos[c2].astype(str) if c2 else pd.Series("Equipa 2", index=jogos.index)

    p = gestor.prob_base(X)
    p_serie = prob_serie(p, n_mapas)

    # Comité: restantes classificadores
    votos = {}
    for chave in gestor.chaves_comite:
        try:
            votos[chave] = gestor.prob_modelo(chave, X)
        except Exception as e:  # um modelo avariado não pára o sistema
            avisos.append(f"Comité: '{chave}' falhou ({type(e).__name__}); ignorado.")

    # Explicação por perturbação (efeito de cada feature vs. valor neutro)
    ref = gestor.referencia
    if ref is None:
        ref = X.median()
        avisos.append("Sem --treino: o valor neutro da explicação é a mediana dos jogos fornecidos "
                      "e não se deteta se os jogos estão fora do intervalo de treino.")
    contrib = pd.DataFrame({c: p - gestor.prob_base(X.assign(**{c: ref[c]})) for c in gestor.colunas})

    fora = pd.Series(False, index=X.index)
    if gestor.limites is not None:
        lo, hi = gestor.limites.loc["min"], gestor.limites.loc["max"]
        fora = ((X < lo) | (X > hi)).any(axis=1)
        if fora.any():
            avisos.append(f"{int(fora.sum())} jogo(s) com features fora do intervalo visto no treino.")
    if n_mapas > 1:
        avisos.append(f"{formato.upper()}: mapas assumidos independentes e com a mesma probabilidade "
                      "(o modelo não conhece o mapa jogado nem o veto).")
    if not hasattr(gestor.base, "predict_proba"):
        avisos.append("O modelo base não devolve probabilidades; valores aproximados (sigmoide da função de decisão).")

    fichas = []
    for i in X.index:
        pi, ps = float(p[i]), float(p_serie[i])
        fav_e1 = pi >= 0.5
        voto_base = pi >= 0.5
        vot = {k: float(v[i]) for k, v in votos.items()}
        n_conc = sum((v >= 0.5) == voto_base for v in vot.values())
        acordo = n_conc / len(vot) if vot else 1.0
        margem = abs(pi - 0.5)
        top = contrib.loc[i].sort_values(key=np.abs, ascending=False).head(n_fatores)
        fichas.append(FichaJogo(
            indice=int(i), equipa1=nome1[i], equipa2=nome2[i], formato=formato.upper(),
            modelo_base=gestor.nome_base, acc_base=gestor.acc_base,
            p_equipa1_mapa=pi, p_equipa1_serie=ps,
            favorita=nome1[i] if fav_e1 else nome2[i],
            p_fav_mapa=pi if fav_e1 else 1 - pi, p_fav_serie=ps if fav_e1 else 1 - ps,
            margem=margem, confianca=nivel_confianca(margem, acordo, bool(fora[i])),
            n_comite=len(vot), n_concordam=int(n_conc), votos=vot,
            fatores=[(c, float(v) * 100) for c, v in top.items()],
            features={c: float(X.loc[i, c]) for c in gestor.colunas}, fora_treino=bool(fora[i]),
        ))
    return fichas, avisos


def fichas_para_dataframe(fichas: list[FichaJogo]) -> pd.DataFrame:
    return pd.DataFrame([{
        "indice": f.indice, "equipa1": f.equipa1, "equipa2": f.equipa2, "formato": f.formato,
        "p_equipa1_mapa": round(f.p_equipa1_mapa, 4), "p_equipa1_serie": round(f.p_equipa1_serie, 4),
        "favorita": f.favorita, "confianca": f.confianca, "margem": round(f.margem, 4),
        "n_comite": f.n_comite, "n_concordam": f.n_concordam, "fora_treino": f.fora_treino,
    } for f in fichas])


# ---------------------------------------------------------------------- #
# Texto determinístico e verificação de números
# ---------------------------------------------------------------------- #
def frases_fatores(f: FichaJogo) -> list[str]:
    """Fatores escritos em relação à favorita (evita confusões de sinal)."""
    outra = f.equipa2 if f.favorita == f.equipa1 else f.equipa1
    frases = []
    for feat, delta in f.fatores:
        a_favor_fav = delta if f.favorita == f.equipa1 else -delta
        quem = f.favorita if a_favor_fav >= 0 else outra
        frases.append(f"{feat} favorece {quem} ({abs(delta):.1f} pp)")
    return frases


def factos_ficha(f: FichaJogo) -> str:
    """Bloco de factos que serve de base ao texto (e ao modelo de linguagem)."""
    acc = f" (accuracy de teste {f.acc_base * 100:.1f}%)" if f.acc_base is not None else ""
    comite = (f"{f.n_concordam} de {f.n_comite} modelos do comité concordam com o favorito"
              if f.n_comite else "sem comité")
    linhas = [
        f"Jogo: {f.equipa1} vs {f.equipa2} (formato {f.formato})",
        f"Modelo base: {f.modelo_base}{acc}",
        f"Favorita: {f.favorita}",
        f"Probabilidade da favorita por mapa: {f.p_fav_mapa * 100:.1f}%",
        f"Probabilidade da favorita na série {f.formato}: {f.p_fav_serie * 100:.1f}%",
        f"Confiança: {f.confianca} (margem de {f.margem * 100:.1f} pp; {comite})",
        "Fatores que mais pesaram: " + "; ".join(frases_fatores(f)),
    ]
    if f.fora_treino:
        linhas.append("Atenção: alguns valores deste jogo estão fora do intervalo visto no treino.")
    return "\n".join(linhas)


def texto_ficha(f: FichaJogo) -> str:
    """Relatório determinístico (também serve de alternativa quando não há modelo de linguagem)."""
    return factos_ficha(f)


def _numeros(t: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", t)]


def numeros_nao_suportados(texto: str, factos: str, tol: float = 0.51) -> list[float]:
    """Números do texto que não aparecem nos factos (tolerância de arredondamento às unidades)."""
    ok = _numeros(factos)
    return [n for n in _numeros(texto) if n not in (1, 2, 3) and not any(abs(n - o) <= tol for o in ok)]
