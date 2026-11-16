"""Notificador Telegram: alertas, dashboard diário e comandos. Modo simulação sem token."""
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

from src.config import PROJECT_ROOT

OUTBOX = PROJECT_ROOT / "reports" / "telegram_outbox.jsonl"


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.simulacao = not (self.token and self.chat_id)

    def enviar(self, texto: str) -> dict:
        registo = {"quando": datetime.now().isoformat(timespec="seconds"), "texto": texto, "simulacao": self.simulacao}
        if self.simulacao:
            OUTBOX.parent.mkdir(exist_ok=True)
            with open(OUTBOX, "a", encoding="utf-8") as f:
                f.write(json.dumps(registo, ensure_ascii=False) + "\n")
            return registo
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        dados = urllib.parse.urlencode({"chat_id": self.chat_id, "text": texto}).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=dados), timeout=15) as r:
            registo["resposta"] = json.loads(r.read())["ok"]
        return registo

    # ---- formatação das mensagens -------------------------------------------------
    @staticmethod
    def formatar_alerta(a: dict) -> str:
        return (f"🚨 ALERTA {a['gravidade']} de fraude\nTransação #{a['id']} | {a['valor']} EUR | {a['hora']}h\n"
                f"Probabilidade de fraude: {a['prob_%']}%")

    @staticmethod
    def formatar_dashboard(f: dict, modelo: dict) -> str:
        linhas = ["📊 Dashboard diário de fraude",
                  f"Transações: {f['n_transacoes']} | Alertas: {f['n_alertas']} ({f['taxa_alertas_%']}%)",
                  f"Valor em risco: {f['valor_em_risco']} EUR de {f['valor_total']} EUR",
                  f"Hora de pico: {f['hora_pico_alertas']}h | Gravidade: {f['alertas_por_gravidade']}",
                  f"Modelo: {modelo['modelo']['nome']} (PR-AUC {modelo['modelo']['pr_auc']}%)"]
        if "simulacao" in f:
            s = f["simulacao"]
            linhas.append(f"Simulação: {s['detetadas']}/{s['fraudes_reais']} fraudes detetadas, {s['falsos_positivos']} falsos positivos")
        return "\n".join(linhas)

    def notificar_alertas(self, factos: dict, minimo=("ALTO", "CRÍTICO")) -> int:
        n = 0
        for a in factos["top_alertas"]:
            if a["gravidade"] in minimo:
                self.enviar(self.formatar_alerta(a)); n += 1
        return n

    def responder_comando(self, comando: str, factos: dict, modelo: dict) -> str:
        c = comando.strip().split()[0].lower()
        if c == "/dashboard":
            return self.formatar_dashboard(factos, modelo)
        if c == "/status":
            return f"✅ Sistema ativo | {factos['n_alertas']} alertas no último lote"
        if c == "/modelo":
            m = modelo["modelo"]
            return f"🤖 {m['nome']} | PR-AUC {m['pr_auc']}% | Recall {m['recall_fraude']}% | limiar {modelo['limiar']}"
        return "Comandos: /status /dashboard /modelo"
