"""Versão em Python do notebook Multiagent_System.ipynb: corre os 3 cenários e mostra o resultado."""
import argparse
import sys

from src.agents.agents import Orquestrador
from src.agents.llm import get_backend
from src.agents.simulator import CENARIOS, cenario


def main():
    if hasattr(sys.stdout, "reconfigure"):  # consola Windows (cp1252) não imprime emojis
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Sistema multiagente de deteção de fraude")
    ap.add_argument("--llm", default="auto", choices=["auto", "transformers", "offline"], help="backend do relator")
    ap.add_argument("--saida", default="terminal,ficheiro", help="destinos dos relatórios: terminal,ficheiro")
    ap.add_argument("--cenario", default="todos", choices=["todos", *CENARIOS])
    args = ap.parse_args()

    backend = get_backend("auto" if args.llm == "auto" else args.llm if args.llm == "transformers" else "offline")
    orq = Orquestrador(backend=backend)
    print("Skills carregadas:", ", ".join(orq.skills))
    for nome in (CENARIOS if args.cenario == "todos" else [args.cenario]):
        print("\n" + "#" * 70 + f"\nCENÁRIO: {nome}\n" + "#" * 70)
        r = orq.processar_lote(cenario(nome), saida=tuple(args.saida.split(",")))
    print("\nComando /dashboard ->\n" + orq.responder("/dashboard", r["factos"]))


if __name__ == "__main__":
    main()
