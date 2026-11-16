---
name: alerta_telegram
description: Envia por Telegram alertas de fraude (CRÍTICO/ALTO) e o dashboard diário; em modo simulação grava a mensagem em reports/telegram_outbox.jsonl.
---
# Skill: alerta_telegram

**Variáveis de ambiente:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (sem elas o bot funciona em modo simulação).

**Mensagens**
- *Alerta:* só gravidade ALTO/CRÍTICO, com id, valor, hora e probabilidade.
- *Dashboard diário:* resumo com nº de transações, alertas, valor em risco, hora de pico e estado do modelo.
- *Comandos:* `/status`, `/dashboard`, `/modelo`.

**Regra de segurança:** nunca incluir dados pessoais nem o vetor de features completo na mensagem.
