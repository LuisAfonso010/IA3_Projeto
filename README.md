# IA3_Projeto — Deteção de Fraude em Cartões de Crédito

Projeto desenvolvido no âmbito da unidade curricular de **Inteligência Artificial 3 (IA3)**, focado na deteção de transações fraudulentas (dataset *Credit Card Fraud Detection*, Kaggle/ULB) e na construção de modelos de Machine Learning e Deep Learning, seguindo **CRISP-ML** e **SCRUM**.

---

## 👤 Autor

* **Nome:** José Pires
* **Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (`creditcard.csv`, 284 807 transações, 492 fraudes = 0.17%)

---

## 📁 Estrutura do Projeto

```text
IA3_Projeto-Creditcard/
├── data/
│   ├── processed/
│   │   ├── creditcard_train_prepared.csv    # 80% mais antigo (treino), limpo + features
│   │   └── creditcard_test_prepared.csv     # 20% mais recente (teste)
│   └── raw/
│       └── creditcard.csv                   # Dados brutos (não versionado: 150 MB)
├── docs/
│   ├── Data Understanding.pdf               # Documentação da fase de entendimento
│   ├── Sprint 0.docx / Sprint 1.docx / Sprint 2.docx
│   ├── class_balance.png                    # Desequilíbrio de classes
│   ├── correlation_matrix.png               # Pearson vs Spearman
│   ├── feature_distributions.png            # KDE por classe
│   ├── pr_curves.png                        # Curvas Precision-Recall (avaliação)
│   └── confusion_matrix_champion.png
├── models/
│   ├── logistic_regression_opt.joblib   svm_opt.joblib   random_forest_opt.joblib
│   ├── decision_tree_opt.joblib         knn_opt.joblib   naive_bayes_opt.joblib
│   ├── gradient_boosting_opt.joblib     tensorflow_mlp_opt.keras
│   ├── scaler.joblib / features_list.joblib
│   └── models_manifest.json                 # Metadados + modelo campeão
├── notebooks/
│   ├── Data_Understanding.ipynb             # Business & Data Understanding (EDA, Canvas, contrato de dados)
│   ├── Data_Preparation.ipynb               # Limpeza, features, split temporal, scaler
│   ├── Modeling.ipynb                       # Baseline + GridSearchCV + TensorFlow
│   ├── Multiagent_System.ipynb              # Sistema multiagente + cenários
│   └── Evaluation.ipynb                     # Avaliação comparativa + manifesto + deployment simulado
├── src/
│   ├── __init__.py
│   ├── config.py                            # Caminhos, constantes, contrato de dados
│   ├── eda_creditcard.py                    # Funções de EDA
│   ├── data_preparation.py                  # Pipeline de preparação
│   ├── train_model.py                       # Baseline (LR + RF) e métricas
│   ├── tune_model.py                        # GridSearchCV (7 modelos) + MLP TensorFlow
│   ├── evaluate_models.py                   # Avaliação + manifesto
│   ├── predict.py                           # Módulo de inferência (PredictorFraude)
│   ├── api_server.py                        # API HTTP (/health, /predict, /dashboard) para o n8n
│   └── agents/                              # Sistema multiagente: tools, llm, skills, agents, telegram_bot, simulator, crewai_version
├── skills/                                  # SKILL.md dos agentes (classificar, analisar, relatório, alerta)
├── n8n/                                     # Workflow n8n (fraude_workflow.json) + README
├── reports/                                 # Relatórios gerados pelo sistema
├── tests/
│   ├── test_predict.py
│   └── test_agents.py
├── .gitignore
├── Dataset Pessoal
├── Ideia de Dataset
├── main.py                                  # Demo: classifica 1 fraude e 1 legítima reais
├── README.md
└── requirements.txt
```

---

## ⚙️ Modelos Treinados

Decision Tree, Gradient Boosting, KNN, Logistic Regression, Naive Bayes, Random Forest, SVM (scikit-learn) e **MLP (TensorFlow/Keras)**.
SVM, KNN e Gradient Boosting são otimizados numa amostra (todas as fraudes + N legítimas) por limitações de CPU; todos são avaliados no teste completo.

---

## 🚀 Instalação e Execução

```bash
python -m venv venv
venv\Scripts\activate        # Linux/MacOS: source venv/bin/activate
pip install -r requirements.txt
# colocar creditcard.csv em data/raw/
python -m src.data_preparation      # ou correr os notebooks pela ordem
python -m src.tune_model
python -m src.evaluate_models
python main.py
pytest tests/
```

---

## 📊 Metodologia do Trabalho

1. **`Data_Understanding.ipynb`**: ML Canvas, contrato de dados, nulos/duplicados, skewness/kurtosis, Pearson vs Spearman, análise temporal.
2. **`Data_Preparation.ipynb`**: remoção de duplicados, `Hour`/`Amount_log`, divisão cronológica 80/20, StandardScaler só no treino.
3. **`Modeling.ipynb`**: baseline, GridSearchCV (`TimeSeriesSplit`, métrica Average Precision) e MLP TensorFlow.
4. **`Evaluation.ipynb`**: Precision/Recall/F1 da fraude, ROC-AUC, PR-AUC, MCC; escolha do campeão e `models_manifest.json`.

Gestão por **SCRUM** (Sprints em `docs/`).


---

## 🤖 Sistema multiagente

`python run_system.py --llm transformers --cenario ataque` (cenários: `dia_normal`, `ataque`, `deriva_valores`). Windows: duplo clique em `Correr_Sistema.bat`. O LLM local (Qwen2.5-0.5B-Instruct) fica em `models/llm/` (não versionado, 1 GB; descarregar do Hugging Face). Sem internet/torch o relator usa um modo offline (`--llm offline`); sem token Telegram as mensagens vão para `reports/telegram_outbox.jsonl`. Variáveis num ficheiro `.env` (não versionado): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Validado: LLM real e Telegram real no cenário de ataque. Versão CrewAI em `src/agents/crewai_version.py` (requer um LLM configurado); fluxo n8n em `n8n/`.

## 📄 Documentação
`docs/Relatorio_IA3_Creditcard.docx` (relatório), `docs/Documentacao_Codigo.docx` (código explicado), diagramas `docs/diagrama_*.png`, `docs/Product_Backlog.md`. Os Sprints contêm campos `[Preencher]` e Daily Scrums simulados: substituir pelo registo real. Testes: `pytest tests`.
