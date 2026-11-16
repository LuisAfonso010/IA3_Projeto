"""Testes do sistema multiagente (modo offline, sem LLM nem Telegram reais)."""
import pytest

from src.agents import telegram_bot, tools
from src.agents.agents import Orquestrador
from src.agents.llm import TemplateBackend
from src.agents.simulator import cenario
from src.agents.skills import carregar_skills


@pytest.fixture(scope="module")
def orq():
    return Orquestrador(backend=TemplateBackend(), notificador=telegram_bot.TelegramNotifier(token=None, chat_id=None))


def test_skills_carregadas_com_descricao():
    s = carregar_skills()
    assert {"classificar_transacao", "analisar_lote", "relatorio_diario", "alerta_telegram"} <= set(s)
    assert all(v["descricao"] for v in s.values())


def test_ataque_gera_mais_alertas_que_dia_normal(orq, tmp_path, monkeypatch):
    monkeypatch.setattr(telegram_bot, "OUTBOX", tmp_path / "out.jsonl")
    normal = orq.processar_lote(cenario("dia_normal", 1500), guardar=False)["factos"]["n_alertas"]
    ataque = orq.processar_lote(cenario("ataque", 1500), guardar=False)["factos"]["n_alertas"]
    assert ataque > normal


def test_relatorio_contem_factos_verificados(orq, tmp_path, monkeypatch):
    monkeypatch.setattr(telegram_bot, "OUTBOX", tmp_path / "out.jsonl")
    r = orq.processar_lote(cenario("ataque", 1500), guardar=False)
    assert str(r["factos"]["n_transacoes"]) in r["relatorio"] and "Factos (verificados" in r["relatorio"]


def test_telegram_simulacao_grava_outbox(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram_bot, "OUTBOX", tmp_path / "out.jsonl")
    n = telegram_bot.TelegramNotifier()
    assert n.simulacao and n.enviar("teste")["simulacao"] and (tmp_path / "out.jsonl").exists()


def test_comando_desconhecido_devolve_ajuda(orq):
    assert "Comandos" in orq.responder("/xyz", {"n_alertas": 0})


def test_gravidade_limiares():
    assert (tools.gravidade(0.95), tools.gravidade(0.75), tools.gravidade(0.55)) == ("CRÍTICO", "ALTO", "MÉDIO")
