---
name: analisar_lote
description: Analisa um lote de transações classificadas e devolve estatísticas do dia, valor em risco, hora de pico e alertas ordenados por gravidade.
---
# Skill: analisar_lote

**Quando usar:** no fim de cada dia (ou janela de tempo) para preparar o dashboard e o relatório.

**Entrada:** DataFrame com `prob_fraude`, `previsao`, `Amount`, `Hour` (+ `Class` só em simulação).
**Saída:** dicionário de factos (`n_transacoes`, `n_alertas`, `valor_em_risco`, `top_alertas`, ...).

**Regras:** nunca estimar valores; todos os números vêm do lote. Em simulação, calcular também fraudes detetadas/falhadas e falsos positivos.
