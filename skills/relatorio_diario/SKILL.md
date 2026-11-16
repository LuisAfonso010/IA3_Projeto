---
name: relatorio_diario
description: Gera o relatório diário de fraude em português a partir dos factos do lote, usando um LLM (Transformers) com guardrail de números.
---
# Skill: relatorio_diario

**Quando usar:** quando o utilizador pedir o relatório/dashboard do dia ou às 20:00 pelo agendador.

**Passos**
1. Receber os factos de `analisar_lote`.
2. Pedir ao LLM um texto curto, usando só os números do JSON (prompt em `src/agents/llm.py`).
3. **Guardrail:** anexar sempre a tabela de factos determinística, para que os números corretos prevaleçam sobre o texto do LLM.
4. Gravar em `reports/relatorio_<data>.md`.

**Fallback:** sem LLM disponível, usa o `TemplateBackend` (regras), mantendo o formato.
