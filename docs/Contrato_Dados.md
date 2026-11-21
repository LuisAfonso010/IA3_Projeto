# Contrato de Dados — Deteção de Fraude em Cartões de Crédito

Projeto individual (José Pires). Os valores abaixo foram medidos nos ficheiros do projeto; o resumo repete o `DATA_CONTRACT` de `src/config.py`.

## 1. Dados de origem (`data/raw/creditcard.csv`)

| Item | Especificação |
|---|---|
| Formato | CSV, separador `,`, cabeçalho na 1.ª linha, UTF-8 |
| Granularidade | 1 linha = 1 transação de cartão de crédito |
| Dimensão | 284.807 linhas × 31 colunas |
| Valores nulos | 0 |
| Componente temporal | **Sim**: `Time` = segundos desde a 1.ª transação (0 a 172.792 s, ≈ 48 h); ficheiro ordenado por `Time` |
| Frequência de amostragem | Irregular (1 registo por transação, sem frequência fixa) |
| Variável alvo | `Class`: 0 = legítima (284.315), 1 = fraude (492) |

### Atributos

| Coluna | Tipo | Unidade / domínio | Papel |
|---|---|---|---|
| Time | numérico | segundos desde a 1.ª transação; 0 a 172.792 | Tempo |
| V1 | float | PCA, anonimizada | Feature |
| V2 | float | PCA, anonimizada | Feature |
| V3 | float | PCA, anonimizada | Feature |
| V4 | float | PCA, anonimizada | Feature |
| V5 | float | PCA, anonimizada | Feature |
| V6 | float | PCA, anonimizada | Feature |
| V7 | float | PCA, anonimizada | Feature |
| V8 | float | PCA, anonimizada | Feature |
| V9 | float | PCA, anonimizada | Feature |
| V10 | float | PCA, anonimizada | Feature |
| V11 | float | PCA, anonimizada | Feature |
| V12 | float | PCA, anonimizada | Feature |
| V13 | float | PCA, anonimizada | Feature |
| V14 | float | PCA, anonimizada | Feature |
| V15 | float | PCA, anonimizada | Feature |
| V16 | float | PCA, anonimizada | Feature |
| V17 | float | PCA, anonimizada | Feature |
| V18 | float | PCA, anonimizada | Feature |
| V19 | float | PCA, anonimizada | Feature |
| V20 | float | PCA, anonimizada | Feature |
| V21 | float | PCA, anonimizada | Feature |
| V22 | float | PCA, anonimizada | Feature |
| V23 | float | PCA, anonimizada | Feature |
| V24 | float | PCA, anonimizada | Feature |
| V25 | float | PCA, anonimizada | Feature |
| V26 | float | PCA, anonimizada | Feature |
| V27 | float | PCA, anonimizada | Feature |
| V28 | float | PCA, anonimizada | Feature |
| Amount | float | euros (EUR); 0 a 25.691,16 | Feature |
| Class | int | {0, 1} | Alvo |

`V1`…`V28` são componentes PCA, anonimizadas por confidencialidade (sem unidade).

## 2. Dados processados (`data/processed/`)

Gerados por `src/data_preparation.py`:

- remoção de 1.081 duplicados (ficam 283.726 registos);
- `Time` substituída por `Hour` (hora do dia = `Time // 3600 % 24`, inteiro 0–23);
- criação de `Amount_log` (`log1p(Amount)`);
- divisão cronológica 80 % / 20 %, sem baralhar (componente temporal).

| Ficheiro | Linhas | Fraudes | Nulos |
|---|---|---|---|
| creditcard_train_prepared.csv | 226.980 | 399 | 0 |
| creditcard_test_prepared.csv | 56.746 | 74 | 0 |

Colunas (32): `V1`…`V28`, `Amount` (float, EUR), `Amount_log` (float, log1p de EUR), `Hour` (int, 0–23) e `Class` (int, 0/1, alvo). As 31 features do modelo são todas menos `Class` (`FEATURES` em `src/config.py`).
