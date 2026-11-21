"""Registo de skills: associa cada pasta skills/<nome>/SKILL.md à função Python que a implementa."""
import re
from pathlib import Path

from src.config import PROJECT_ROOT
from src.agents import tools

SKILLS_DIR = PROJECT_ROOT / "skills"
IMPLEMENTACOES = {
    "classificar_transacao": tools.classificar_transacao,
    "analisar_lote": tools.analisar_lote,
    "cartao_modelo": tools.cartao_modelo,
}


def carregar_skills() -> dict:
    """Lê o front-matter (name, description) de cada SKILL.md."""
    skills = {}
    for md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        txt = md.read_text(encoding="utf-8")
        nome = re.search(r"^name:\s*(.+)$", txt, re.M).group(1).strip()
        desc = re.search(r"^description:\s*(.+)$", txt, re.M).group(1).strip()
        skills[nome] = {"descricao": desc, "ficheiro": str(md.relative_to(PROJECT_ROOT)),
                        "funcao": IMPLEMENTACOES.get(nome)}
    return skills
