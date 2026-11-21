"""Saída dos relatórios dos agentes: terminal e/ou ficheiro (reports/<data>/<n>_<agente>.md)."""
from datetime import date, datetime

from src.config import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"


def emitir(agente: str, texto: str, destinos=("terminal", "ficheiro"), ordem: int = 0):
    corpo = f"## Relatório do agente {agente} ({datetime.now():%H:%M:%S})\n\n{texto}\n"
    if "terminal" in destinos:
        print("\n" + "=" * 70 + "\n" + corpo)
    if "ficheiro" in destinos:
        pasta = REPORTS / date.today().isoformat()
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / f"{ordem}_{agente.lower()}.md").write_text(corpo, encoding="utf-8")
