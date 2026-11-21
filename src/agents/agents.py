"""Sistema multiagente (versão 1: Python + Transformers). Padrão supervisor -> agentes especialistas."""
import json
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.config import PROJECT_ROOT
from src.agents import tools
from src.agents.llm import LLMBackend, get_backend
from src.agents.output import emitir
from src.agents.skills import carregar_skills
from src.agents.telegram_bot import TelegramNotifier

REPORTS = PROJECT_ROOT / "reports"


@dataclass
class Agente:
    nome: str
    papel: str
    skills: list
    log: list = field(default_factory=list)

    def registar(self, msg: str):
        self.log.append(msg)


class AgenteClassificador(Agente):
    """Usa o modelo campeão (skill classificar_transacao) sobre transações ou lotes."""

    def executar(self, df: pd.DataFrame) -> pd.DataFrame:
        res = tools.classificar_lote(df)
        self.registar(f"classificou {len(res)} transações ({int(res['previsao'].sum())} alertas)")
        return res


class AgenteAnalista(Agente):
    """Transforma as previsões em factos verificáveis (skill analisar_lote)."""

    def executar(self, res: pd.DataFrame) -> dict:
        factos = tools.analisar_lote(res)
        self.registar(f"gerou factos: {factos['n_alertas']} alertas, {factos['valor_em_risco']} EUR em risco")
        return factos


class AgenteRelator(Agente):
    """Redige o relatório com um LLM (skill relatorio_diario) e aplica o guardrail de números."""

    def __init__(self, *a, backend: LLMBackend, **k):
        super().__init__(*a, **k)
        self.backend = backend

    def executar(self, factos: dict, modelo: dict) -> str:
        g = factos.get("alertas_por_gravidade", {})
        linhas = [f"- Transações analisadas: {factos['n_transacoes']}",
                  f"- Alertas de fraude: {factos['n_alertas']} ({factos['taxa_alertas_%']}% das transações)",
                  f"- Valor total movimentado: {factos['valor_total']} euros",
                  f"- Valor em risco nos alertas: {factos['valor_em_risco']} euros",
                  f"- Probabilidade média dos alertas: {factos['prob_media_alertas_%']}%",
                  f"- Hora com mais alertas: {factos['hora_pico_alertas']}h",
                  "- Alertas por gravidade: " + ", ".join(f"{k} {v}" for k, v in g.items())]
        prompt = ("Escreve 3 frases curtas em português europeu a resumir estes factos. "
                  "Copia os números EXATAMENTE como estão, a moeda é sempre euros (nunca R$ nem dólares), "
                  "não faças contas nem inventes valores.\n" + "\n".join(linhas)
                  + "\nJSON: " + json.dumps({k: v for k, v in factos.items() if k != "top_alertas"}, ensure_ascii=False))
        narrativa = self.backend.gerar(prompt)
        tabela = "\n".join(f"| {k} | {v} |" for k, v in factos.items() if k not in ("top_alertas", "simulacao"))
        top = "\n".join(f"- #{a['id']}: {a['prob_%']}% | {a['valor']} EUR | {a['hora']}h | {a['gravidade']}" for a in factos["top_alertas"])
        md = (f"# Relatório diário de fraude - {date.today().isoformat()}\n\n"
              f"*Modelo: {modelo['modelo']['nome']} | Gerado por: {self.backend.nome}*\n\n"
              f"## Resumo\n{narrativa}\n\n## Factos (verificados, não gerados pelo LLM)\n| Métrica | Valor |\n|---|---|\n{tabela}\n\n"
              f"## Top alertas\n{top or '- sem alertas'}\n")
        self.registar(f"relatório gerado com {self.backend.nome}")
        return md


class AgenteNotificador(Agente):
    """Envia alertas e dashboard por Telegram (skill alerta_telegram)."""

    def __init__(self, *a, notificador: TelegramNotifier, **k):
        super().__init__(*a, **k)
        self.tg = notificador

    def executar(self, factos: dict, modelo: dict, relatorio: str = "") -> dict:
        n = self.tg.notificar_alertas(factos)
        self.tg.enviar(self.tg.formatar_dashboard(factos, modelo))
        if relatorio:
            self.tg.enviar(relatorio[:4000])  # limite do Telegram: 4096 caracteres por mensagem
        self.registar(f"enviou {n} alertas + dashboard{' + relatório' if relatorio else ''} ({'simulação' if self.tg.simulacao else 'Telegram real'})")
        return {"alertas_enviados": n, "simulacao": self.tg.simulacao}


class Orquestrador:
    """Supervisor: recebe o lote, delega pela ordem Classificador -> Analista -> Relator -> Notificador."""

    def __init__(self, backend: LLMBackend | None = None, notificador: TelegramNotifier | None = None):
        self.skills = carregar_skills()
        self.classificador = AgenteClassificador("Classificador", "Aplica o modelo de ML", ["classificar_transacao"])
        self.analista = AgenteAnalista("Analista", "Calcula factos e alertas", ["analisar_lote"])
        self.relator = AgenteRelator("Relator", "Escreve o relatório (LLM)", ["relatorio_diario"], backend=backend or get_backend())
        self.notificador = AgenteNotificador("Notificador", "Envia alertas Telegram", ["alerta_telegram"],
                                             notificador=notificador or TelegramNotifier())

    def processar_lote(self, df: pd.DataFrame, guardar: bool = True, saida=()) -> dict:
        """saida: destinos dos relatórios de cada agente, ex.: ("terminal", "ficheiro")."""
        modelo = tools.cartao_modelo()
        res = self.classificador.executar(df)
        emitir("Classificador", f"- {self.classificador.log[-1]}\n- Modelo: {modelo['modelo']['nome']} (limiar {modelo['limiar']})", saida, 1)
        factos = self.analista.executar(res)
        tabela = "\n".join(f"| {k} | {v} |" for k, v in factos.items() if k not in ("top_alertas", "simulacao"))
        top = "\n".join(f"- #{a['id']}: {a['prob_%']}% | {a['valor']} EUR | {a['hora']}h | {a['gravidade']}" for a in factos["top_alertas"])
        emitir("Analista", f"| Métrica | Valor |\n|---|---|\n{tabela}\n\nTop alertas:\n{top or '- sem alertas'}", saida, 2)
        relatorio = self.relator.executar(factos, modelo)
        emitir("Relator", relatorio, saida, 3)
        envio = self.notificador.executar(factos, modelo, relatorio)
        emitir("Notificador", f"- {self.notificador.log[-1]}", saida, 4)
        if guardar:
            REPORTS.mkdir(exist_ok=True)
            (REPORTS / f"relatorio_{date.today().isoformat()}.md").write_text(relatorio, encoding="utf-8")
        trace = {a.nome: a.log[-1] for a in (self.classificador, self.analista, self.relator, self.notificador)}
        return {"factos": factos, "relatorio": relatorio, "telegram": envio, "trace": trace}

    def responder(self, comando: str, factos: dict) -> str:
        return self.notificador.tg.responder_comando(comando, factos, tools.cartao_modelo())
