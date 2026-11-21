"""API HTTP mínima (stdlib) para integrar o sistema com n8n: POST /predict, GET /dashboard, GET /health."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.agents import tools
from src.agents.agents import Orquestrador
from src.agents.llm import TemplateBackend
from src.agents.simulator import cenario


class Handler(BaseHTTPRequestHandler):
    orq = None

    def _json(self, codigo: int, corpo: dict):
        dados = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados))); self.end_headers(); self.wfile.write(dados)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok", "modelo": tools.cartao_modelo()["modelo"]["nome"]})
        if self.path.startswith("/dashboard"):
            nome = self.path.split("cenario=")[-1] if "cenario=" in self.path else "dia_normal"
            r = Handler.orq.processar_lote(cenario(nome, 2000), guardar=False)
            return self._json(200, {"factos": r["factos"], "relatorio": r["relatorio"]})
        self._json(404, {"erro": "rota desconhecida"})

    def do_POST(self):
        if self.path != "/predict":
            return self._json(404, {"erro": "rota desconhecida"})
        try:
            corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._json(200, tools.classificar_transacao(corpo))
        except (ValueError, KeyError) as e:
            self._json(400, {"erro": str(e)})

    def log_message(self, *a):  # silencia o log por pedido
        pass


def servir(porta: int = 8000):
    Handler.orq = Orquestrador(backend=TemplateBackend())
    print(f"API de fraude em http://localhost:{porta} (/health, /predict, /dashboard)")
    HTTPServer(("0.0.0.0", porta), Handler).serve_forever()


if __name__ == "__main__":
    servir()
