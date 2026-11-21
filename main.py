#!/usr/bin/env python3
"""
CS2 - sistema multiagente de previsão (US01: CLI).

  python main.py --modo hf      --jogos cs2_matches_test.csv --treino cs2_matches_train.csv
  python main.py --modo crewai  --jogos cs2_matches_test.csv --treino cs2_matches_train.csv --llm ollama/qwen2.5:1.5b
  python main.py --modo todos   --jogos cs2_matches_test.csv --treino cs2_matches_train.csv
  python main.py --verificar cs2_matches_test.csv          # confirma que o modelo base é o melhor

No Colab:  from main import main;  main(["--modo", "hf", "--jogos", "...", "--treino", "..."])
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _utf8():
    """Consola do Windows: evita erros com acentos e símbolos."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _escrever_relatorio(caminho: Path, titulo: str, resultado: dict) -> None:
    linhas = [titulo, "=" * len(titulo), ""]
    for r in resultado["relatorios"]:
        linhas += [r["titulo"], f"(origem: {r.get('origem', 'crewai')})" if "origem" in r else
                   f"(ferramentas usadas: {', '.join(r['ferramentas']) or 'nenhuma'})", "", r["texto"] or "(sem texto)"]
        linhas += [f"AVISO: {a}" for a in r.get("avisos", [])]
        linhas.append("")
    caminho.write_text("\n".join(linhas), encoding="utf-8")


def _mostrar(resultado: dict, titulo: str) -> None:
    print(f"\n===== {titulo} =====")
    for r in resultado["relatorios"]:
        print(f"\n{r['titulo']}  [{r.get('origem', 'crewai')}]\n{r['texto']}")
        for a in r.get("avisos", []):
            print(f"  AVISO: {a}")


def main(argv=None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(description="CS2 - sistema multiagente de previsão")
    ap.add_argument("--modo", choices=["hf", "crewai", "todos"], default="hf")
    ap.add_argument("--jogos", help="CSV com as 12 features dos jogos a analisar")
    ap.add_argument("--pasta-modelos", default="resultados", help="pasta com os .joblib do notebook")
    ap.add_argument("--treino", help="cs2_matches_train.csv (valores neutros e limites de treino)")
    ap.add_argument("--formato", default="bo1", choices=["bo1", "bo3", "bo5"])
    ap.add_argument("--max-jogos", type=int, default=5, help="nº de jogos a analisar (LLMs são lentos)")
    ap.add_argument("--qwen-dir", help="pasta do Qwen2.5-1.5B-Instruct (modo hf)")
    ap.add_argument("--sem-llm", action="store_true", help="modo hf sem modelo de linguagem (texto determinístico)")
    ap.add_argument("--llm", default=None, help="LLM do CrewAI, string LiteLLM (padrão ollama/qwen2.5:1.5b)")
    ap.add_argument("--saida", default="saida", help="pasta onde guardar CSV e relatórios")
    ap.add_argument("--verificar", metavar="TESTE_CSV", help="confirma que o modelo base é o melhor e termina")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    import pandas as pd
    from src.gestor_modelos import GestorModelos

    try:
        gestor = GestorModelos(a.pasta_modelos, a.treino)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERRO: {e}")
        return 2
    print(gestor.resumo())

    if a.verificar:
        ok, tabela = gestor.verificar(a.verificar)
        print(tabela.assign(accuracy_teste=(tabela.accuracy_teste * 100).round(2))
              .drop(columns="chave").to_string(index=False))
        print("\nVERIFICAÇÃO:", "OK - o modelo base é o de maior accuracy no teste." if ok
              else "FALHOU - existe um modelo com accuracy superior à do modelo base.")
        return 0 if ok else 1
    if not a.jogos:
        ap.error("indica --jogos (ou --verificar)")

    jogos = pd.read_csv(a.jogos)
    total = len(jogos)
    jogos = jogos.head(a.max_jogos)
    print(f"A analisar {len(jogos)} de {total} jogos (--max-jogos para mudar).")
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)

    estado: dict[str, str] = {}
    previsoes = None

    if a.modo in ("hf", "todos"):
        try:
            from src.agentes_hf import executar_hf
            r = executar_hf(gestor, jogos, a.formato, a.qwen_dir, a.sem_llm, a.verbose)
            previsoes = r["previsoes"]
            _mostrar(r, "Multiagente Hugging Face")
            _escrever_relatorio(saida / "relatorio_hf.txt", "Relatório - Hugging Face", r)
            for aviso in dict.fromkeys(r["avisos"]):
                print(f"AVISO: {aviso}")
            estado["hf"] = "OK" if (r["usou_llm"] or a.sem_llm) else "OK (sem Qwen: texto determinístico)"
        except Exception as e:
            estado["hf"] = f"FALHOU ({type(e).__name__}: {e})"

    if a.modo in ("crewai", "todos"):
        try:
            from src.crewai_multiagent import LLM_PADRAO, executar_crewai
            r = executar_crewai(gestor, jogos, a.formato, a.llm or LLM_PADRAO, a.verbose)
            previsoes = r["previsoes"] if previsoes is None else previsoes
            _mostrar(r, "Multiagente CrewAI")
            _escrever_relatorio(saida / "relatorio_crewai.txt", "Relatório - CrewAI", r)
            falhas = [x for x in r["relatorios"] if not x["texto"]]
            estado["crewai"] = "OK" if not falhas else f"parcial ({len(falhas)} jogo(s) falharam)"
        except Exception as e:
            estado["crewai"] = f"FALHOU ({type(e).__name__}: {e})"

    if previsoes is not None:
        previsoes.to_csv(saida / "previsoes.csv", index=False)
    print("\n--- Resumo ---")
    for modo, st in estado.items():
        print(f"  {modo}: {st}")
    print(f"Ficheiros em: {saida.resolve()}")
    return 0 if all(v.startswith("OK") for v in estado.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
