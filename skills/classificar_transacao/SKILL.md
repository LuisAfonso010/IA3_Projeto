---
name: classificar_transacao
description: Classifica uma transação de cartão (V1..V28, Amount, Time/Hour) como Fraude ou Legítima com o modelo campeão e devolve a probabilidade.
---
# Skill: classificar_transacao

**Quando usar:** sempre que o utilizador ou outro agente pedir a análise de uma transação concreta.

**Entrada:** dicionário com `V1..V28`, `Amount` e `Time` (ou `Hour`).
**Saída:** `{modelo, previsao, probabilidade_fraude_%, limiar_%}`.

**Passos**
1. Validar que a transação tem todas as colunas (a ferramenta levanta `ValueError` se faltar alguma).
2. Chamar `src.agents.tools.classificar_transacao`.
3. Se `probabilidade_fraude_%` ≥ limiar, devolver gravidade (MÉDIO ≥50%, ALTO ≥70%, CRÍTICO ≥90%).

**Limites:** o modelo foi treinado com dados de 2013 (cartões europeus); não usar para decisões automáticas sem revisão humana.
