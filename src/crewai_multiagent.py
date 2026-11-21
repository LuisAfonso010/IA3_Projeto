"""
US04 - Multiagente com CrewAI.

Os classificadores são expostos como ferramentas (`@tool`) que os agentes chamam por iniciativa
própria:
  - Analista Preditivo de CS2: decide que ferramentas usar (previsão do modelo base, comité,
    fatores, série) para analisar um jogo.
  - Relator Técnico: escreve o relatório em português com o que o analista apurou.

O `crewai` só é importado em `executar_crewai`, para o resto do projeto funcionar sem ele.
As funções `f_*` são Python simples (testáveis sem CrewAI); `construir_ferramentas` aplica-lhes @tool.

LLM: qualquer string LiteLLM, p.ex. `ollama/qwen2.5:1.5b` (padrão, via Ollama local) ou outro
fornecedor com a respetiva chave de API. Modelos de 1.5B chamam ferramentas de forma pouco fiável.
"""
from __future__ import annotations

import functools
import json
import os

import pandas as pd

from .predict import FORMATOS, analisar_jogos, fichas_para_dataframe, numeros_nao_suportados, prob_serie

LLM_PADRAO = "ollama/qwen2.5:1.5b"
_ESTADO: dict = {"gestor": None, "X": None, "fichas": [], "formato": "bo1", "chamadas": [], "saidas": []}


def configurar(gestor, jogos: pd.DataFrame, formato: str = "bo1") -> list:
    """Prepara o estado partilhado pelas ferramentas. Devolve as fichas (para guardar o CSV)."""
    jogos = jogos.reset_index(drop=True)
    fichas, _ = analisar_jogos(gestor, jogos, formato)
    _ESTADO.update(gestor=gestor, X=gestor.alinhar(jogos), fichas=fichas, formato=formato,
                   chamadas=[], saidas=[])
    return fichas


def _com_registo(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        saida = fn(*a, **k)
        _ESTADO["chamadas"].append(fn.__name__)
        _ESTADO["saidas"].append(saida)
        return saida
    return wrapper


def _json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def _ficha(indice: int):
    if _ESTADO["gestor"] is None:
        raise RuntimeError("Chama configurar(gestor, jogos) antes de usar as ferramentas.")
    n = len(_ESTADO["fichas"])
    if not 0 <= int(indice) < n:
        raise IndexError(f"índice {indice} fora do intervalo (0 a {n - 1})")
    return _ESTADO["fichas"][int(indice)]


# --------------------------- ferramentas (Python simples) --------------------------- #
def f_listar_modelos() -> str:
    """Lista os classificadores disponíveis, qual é o melhor modelo global e a accuracy de teste de cada um."""
    g = _ESTADO["gestor"]
    return _json({"melhor_modelo": g.nome_base, "chave_melhor": g.chave_base, "modelos": [
        {"chave": k, "nome": g.nome_de(k),
         "accuracy_teste_pct": round(g.acc[k] * 100, 1) if k in g.acc else None} for k in g.modelos]})


def f_dados_jogo(indice: int) -> str:
    """Devolve as equipas, o formato e os valores das 12 features de um jogo, dado o seu índice (começa em 0)."""
    f = _ficha(indice)
    return _json({"indice": f.indice, "equipa1": f.equipa1, "equipa2": f.equipa2, "formato": f.formato,
                  "features": {k: round(v, 3) for k, v in f.features.items()},
                  "fora_do_intervalo_de_treino": f.fora_treino})


def f_prever_jogo(indice: int, modelo: str = "melhor") -> str:
    """Probabilidade de a equipa 1 vencer o mapa segundo um classificador. 'modelo' é 'melhor' (modelo base) ou uma chave de f_listar_modelos."""
    g = _ESTADO["gestor"]
    f = _ficha(indice)
    chave = g.chave_base if str(modelo).lower() in ("melhor", "base", "melhor_global") else str(modelo)
    if chave not in g.modelos:
        return _json({"erro": f"modelo '{modelo}' não existe", "modelos_validos": list(g.modelos)})
    p = float(g.prob_modelo(chave, _ESTADO["X"].iloc[[f.indice]])[0])
    return _json({"indice": f.indice, "modelo": g.nome_de(chave), "p_equipa1_vence_mapa_pct": round(p * 100, 1),
                  "p_equipa2_vence_mapa_pct": round((1 - p) * 100, 1),
                  "favorita": f.equipa1 if p >= 0.5 else f.equipa2})


def f_votos_comite(indice: int) -> str:
    """Opinião dos restantes classificadores (comité) sobre um jogo: voto de cada um e quantos concordam com o modelo base."""
    f = _ficha(indice)
    g = _ESTADO["gestor"]
    return _json({"indice": f.indice, "favorita_modelo_base": f.favorita,
                  "n_concordam": f.n_concordam, "n_comite": f.n_comite,
                  "votos": [{"modelo": g.nome_de(k), "p_equipa1_vence_mapa_pct": round(v * 100, 1)}
                            for k, v in f.votos.items()]})


def f_fatores_jogo(indice: int) -> str:
    """Features que mais empurram a previsão do modelo base num jogo (efeito em pontos percentuais na probabilidade da equipa 1)."""
    f = _ficha(indice)
    return _json({"indice": f.indice, "equipa1": f.equipa1, "equipa2": f.equipa2,
                  "efeito_na_prob_equipa1_pp": [{"feature": c, "efeito_pp": round(v, 1)} for c, v in f.fatores]})


def f_probabilidade_serie(p_mapa_pct: float, formato: str = "bo3") -> str:
    """Converte a probabilidade de vencer um mapa (em %, 0 a 100) na probabilidade de vencer a série. formato: bo1, bo3 ou bo5."""
    formato = str(formato).lower()
    if formato not in FORMATOS or not 0 <= float(p_mapa_pct) <= 100:
        return _json({"erro": "usa formato bo1/bo3/bo5 e p_mapa_pct entre 0 e 100"})
    ps = float(prob_serie(float(p_mapa_pct) / 100, FORMATOS[formato]))
    return _json({"formato": formato.upper(), "p_mapa_pct": round(float(p_mapa_pct), 1),
                  "p_serie_pct": round(ps * 100, 1)})


FERRAMENTAS_BASE = [("Listar modelos", f_listar_modelos), ("Dados do jogo", f_dados_jogo),
                    ("Prever jogo", f_prever_jogo), ("Votos do comité", f_votos_comite),
                    ("Fatores do jogo", f_fatores_jogo), ("Probabilidade de série", f_probabilidade_serie)]


def construir_ferramentas():
    """Aplica @tool do CrewAI às funções (com registo das chamadas feitas pelos agentes)."""
    from crewai.tools import tool
    return [tool(nome)(_com_registo(fn)) for nome, fn in FERRAMENTAS_BASE]


# --------------------------------- crew --------------------------------- #
def _criar_llm(llm: str):
    from crewai import LLM
    kw = {"temperature": 0.0}
    if llm.startswith("ollama/"):
        kw["base_url"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return LLM(model=llm, **kw)


def executar_crewai(gestor, jogos, formato="bo1", llm=LLM_PADRAO, verbose=False):
    """Corre uma crew (Analista -> Relator) por jogo. Devolve dict com relatórios e previsões."""
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as e:
        raise RuntimeError("Falta instalar o CrewAI: pip install crewai") from e

    fichas = configurar(gestor, jogos, formato)
    modelo_llm = _criar_llm(llm)
    ferramentas = construir_ferramentas()

    analista = Agent(
        role="Analista Preditivo de CS2",
        goal="Analisar jogos de CS2 usando os classificadores treinados e reportar números exatos.",
        backstory="Cientista de dados de esports. Só afirma o que as ferramentas devolvem e nunca inventa números.",
        tools=ferramentas, llm=modelo_llm, allow_delegation=False, verbose=verbose)
    relator = Agent(
        role="Relator Técnico",
        goal="Escrever relatórios curtos e claros em português europeu com base na análise recebida.",
        backstory="Redator técnico. Usa apenas os números que o analista forneceu.",
        llm=modelo_llm, allow_delegation=False, verbose=verbose)

    relatorios = []
    for f in fichas:
        _ESTADO["chamadas"], _ESTADO["saidas"] = [], []
        t1 = Task(
            description=(f"Analisa o jogo com índice {f.indice} (formato {f.formato}). Tens de usar as ferramentas: "
                         "'Dados do jogo', 'Prever jogo' (modelo 'melhor'), 'Votos do comité' e 'Fatores do jogo'; "
                         "se o formato não for BO1, usa também 'Probabilidade de série'. "
                         "Resume: favorita, probabilidade por mapa e na série, quantos modelos concordam, fatores principais."),
            expected_output="Resumo factual com todos os números devolvidos pelas ferramentas.",
            agent=analista)
        t2 = Task(
            description=("Escreve um relatório de 3 parágrafos curtos (máx. 120 palavras) em português europeu: "
                         "previsão, fatores principais, fiabilidade e uma limitação. Usa apenas números da análise."),
            expected_output="Relatório técnico de 3 parágrafos.", agent=relator, context=[t1])
        crew = Crew(agents=[analista, relator], tasks=[t1, t2], process=Process.sequential, verbose=verbose)
        titulo = f"#{f.indice + 1} {f.equipa1} vs {f.equipa2}"
        try:
            res = crew.kickoff()
            texto = str(getattr(res, "raw", res)).strip()
            usadas = list(_ESTADO["chamadas"])
            avisos = []
            if not usadas:
                avisos.append("O analista NÃO chamou nenhuma ferramenta: os números podem ter sido inventados.")
            mal = numeros_nao_suportados(texto, "\n".join(_ESTADO["saidas"]))
            if usadas and mal:
                avisos.append(f"Números no texto que nenhuma ferramenta devolveu: {mal}")
            relatorios.append({"indice": f.indice, "titulo": titulo, "texto": texto,
                               "ferramentas": usadas, "avisos": avisos})
        except Exception as e:  # um jogo falhado não deve parar os restantes
            relatorios.append({"indice": f.indice, "titulo": titulo, "texto": "",
                               "ferramentas": list(_ESTADO["chamadas"]),
                               "avisos": [f"Falhou: {type(e).__name__}: {e}"]})
    return {"fichas": fichas, "relatorios": relatorios, "previsoes": fichas_para_dataframe(fichas)}
