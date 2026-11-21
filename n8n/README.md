# Versão n8n do sistema multiagente
1. `python -m src.api_server` (expõe `/predict`, `/dashboard`, `/health` em `127.0.0.1:8000`; os URLs do workflow usam `127.0.0.1` porque o Node resolve `localhost` para IPv6 e a API só escuta em IPv4).
2. No n8n: *Import from file* -> `fraude_workflow.json`; criar a credencial *Telegram API* (token do bot) e associá-la aos 2 nós Telegram; definir a variável `TELEGRAM_CHAT_ID` (e `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`, para os nós lerem `$env`).
3. Dois fluxos: dashboard diário às 20h e webhook `POST /webhook/transacao` que classifica e alerta se for fraude (se não for, responde com a classificação).

Testado com n8n 2.48.7 e Telegram real (webhook com uma fraude e uma legítima, e agendamento).
