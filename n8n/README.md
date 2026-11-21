# Versão n8n do sistema multiagente
1. `python -m src.api_server` (expõe `/predict`, `/dashboard`, `/health` em `localhost:8000`).
2. No n8n: *Import from file* -> `fraude_workflow.json`; definir as credenciais Telegram e a variável `TELEGRAM_CHAT_ID`.
3. Dois fluxos: dashboard diário às 20h e webhook `POST /webhook/transacao` que classifica e alerta se for fraude.
