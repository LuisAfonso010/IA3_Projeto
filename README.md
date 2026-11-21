# Deteção de Fraude em Cartões de Crédito

Componente individual de **Inteligência Artificial III** (ISTEC, Robótica e Inteligência Artificial, professor Flávio Vaz).

- **Autor:** José Pires
- **Dataset:** [Credit Card Fraud Detection (Kaggle/ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284.807 transações, 492 fraudes (0,17%)
- **Metodologia:** CRISP-ML (implementação) e SCRUM (gestão, Sprints 0 a 2 em `docs/`)

## Estrutura

```text
├── data/
│   ├── processed/      # treino (80% mais antigo) e teste (20% mais recente) já preparados
│   └── raw/            # colocar aqui o creditcard.csv (não versionado, ~150 MB)
├── docs/               # Data Understanding, Sprints 0-2, Product Backlog, Documentacao_Codigo, diagramas do multiagente e gráficos
├── models/             # 7 modelos scikit-learn (.joblib), 1 MLP TensorFlow (.keras), scaler, models_manifest.json e llm_config.json
├── notebooks/          # Data_Understanding, Data_Preparation, Modeling, Evaluation, Deployment, Multiagent_System
├── src/                # config, eda_creditcard, data_preparation, train_model, tune_model, evaluate_models, predict, api_server e agents/ (multiagente)
├── tests/              # test_predict.py e test_agents.py
├── skills/             # 4 skills dos agentes (SKILL.md)
├── n8n/                # fluxo n8n (fraude_workflow.json)
├── reports/            # relatórios gerados pelos agentes
├── run_system.py       # corre o sistema multiagente
├── main.py             # classifica 1 fraude e 1 transação legítima reais com todos os modelos
└── requirements.txt
```

## Preparação dos dados

Removeram-se 1.081 duplicados (283.726 registos). A coluna `Time` foi substituída por `Hour` e criou-se `Amount_log`. A divisão é cronológica: treino com 226.980 transações (399 fraudes) e teste com 56.746 (74 fraudes). O `StandardScaler` é ajustado só no treino. O desequilíbrio é tratado na modelação com pesos de classe, sem oversampling.

## Modelos e resultados

Sete modelos scikit-learn (Logistic Regression, SVM, Random Forest, Decision Tree, Gradient Boosting, Naive Bayes e KNN) e um MLP em TensorFlow, otimizados com GridSearchCV (métrica Average Precision). SVM, KNN e Gradient Boosting são otimizados numa amostra do treino por limitações de CPU. Todos são avaliados no teste completo (valores da classe fraude, em %):

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest (campeão)** | 98,15 | 71,62 | 82,81 | 98,62 | 81,41 |
| MLP (TensorFlow) | 11,23 | 85,14 | 19,84 | 95,49 | 79,40 |
| KNN | 76,71 | 75,68 | 76,19 | 93,15 | 79,33 |
| Logistic Regression | 2,68 | 91,89 | 5,20 | 98,71 | 75,92 |
| Decision Tree | 11,11 | 77,03 | 19,42 | 88,41 | 70,99 |
| SVM | 52,43 | 72,97 | 61,02 | 96,70 | 69,06 |
| Gradient Boosting | 55,21 | 71,62 | 62,35 | 98,42 | 38,34 |
| Naive Bayes | 5,03 | 79,73 | 9,46 | 96,79 | 6,19 |

O campeão é escolhido pelo maior PR-AUC no teste (limiar de decisão 0,5) e fica registado em `models/models_manifest.json`.

## Como executar

```bash
python -m venv venv
venv\Scripts\activate        # Linux/MacOS: source venv/bin/activate
pip install -r requirements.txt
# colocar creditcard.csv em data/raw/
python -m src.data_preparation
python -m src.tune_model
python -m src.evaluate_models
python main.py
pytest tests/
```

`Deployment.ipynb` faz o mesmo que `main.py`: usa `PredictorFraude` (modelo campeão) e `PredictorTodos` (todos os modelos guardados em `models/`).

## Sistema multiagente (LLM)

Agentes: Classificador → Analista → Relator (LLM) → Notificador (Telegram), coordenados por um Orquestrador (`src/agents/`). Skills em `skills/`.

O relatório do Relator sai no terminal, em `reports/relatorio_<data>.md` e no Telegram (o Notificador envia também os alertas e o dashboard).

```bash
python run_system.py --llm offline --cenario ataque        # sem LLM nem Telegram (modo simulação)
python run_system.py --llm transformers --cenario ataque   # LLM local Qwen2.5-0.5B em models/llm/ (não versionado)
```

Variáveis num `.env` (não versionado): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Sem token, as mensagens vão para `reports/telegram_outbox.jsonl`. Testes: `pytest tests/test_agents.py`.
