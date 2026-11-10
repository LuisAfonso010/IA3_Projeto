# ⚽ Sistema Preditivo de Resultados de Futebol (CRISP-ML)

Este repositório contém o sistema automatizado para previsão do resultado de partidas de futebol ($H, D, A$) desenvolvido segundo o ciclo de vida **CRISP-ML** e organizado sob a metodologia ágil **Scrum**.

---

## 📌 Visão Geral do Projeto

O objetivo principal é prever a probabilidade dos três resultados possíveis numa partida de futebol (Vitória Casa, Empate, Vitória Fora) utilizando métricas históricas móveis (*rolling features* dos últimos 5 jogos) e diferenças de desempenho entre as equipas, evitando qualquer contaminação temporal de dados (*data leakage*).

---

## 🗂️ Estrutura do Repositório

```text
Projeto_Futebol/
├── data/
│   └── dataset_futebol_preparado.csv    # Dataset processado com features móveis
├── models/
│   ├── label_encoder.joblib             # Codificador das classes (H, D, A)
│   ├── scaler.joblib                    # Normalizador StandardScaler
│   ├── features_list.joblib             # Lista oficial de variáveis preditivas
│   ├── svm_opt.joblib                   # Modelo campeão otimizado
│   ├── ... (outros artefactos .joblib / .keras)
│   └── models_manifest.json             # Manifesto dinamico com metadados do modelo campeão
├── notebooks/
│   ├── 1_Data_Understanding.ipynb       # Sprint 0: Análise exploratória e diagnóstico de qualidade
│   ├── 2_Data_Preparation.ipynb         # Sprint 1: Limpeza, ordenação cronológica e rolling features
│   ├── 3_Modeling.ipynb                 # Sprint 2: Treino e GridSearchCV de 8 modelos (ML + DL)
│   └── 4_Evaluation.ipynb               # Sprint 2: Avaliação comparativa e seleção do modelo campeão
├── src/
│   ├── __init__.py
│   └── predict.py                       # Classe PredictorFutebol para inferência em produção
├── main.py                              # Script principal executável
├── requirements.txt                     # Dependências do projeto
└── README.md                            # Documentação geral
```

---

## 🚀 Sprints & Metodologia CRISP-ML

### 🔹 Sprint 0 — Data Understanding (`1_Data_Understanding.ipynb`)
- Conexão e auditoria inicial dos dados brutos.
- Diagnóstico de nulos, duplicados e inconsistências.
- Análise descritiva univariada (*Skewness* e *Kurtosis*) e multivariada (matrizes de Pearson e Spearman).

### 🔹 Sprint 1 — Data Preparation (`2_Data_Preparation.ipynb`)
- Tratamento de datas e ordenação rigorosamente cronológica.
- Remoção de variáveis colaterais que provocam *data leakage*.
- Criação de variáveis de forma móvel nos últimos 5 jogos (`Home_Forma_L5`, `Diff_Forma_L5`, `Diff_GM_L5`, `Diff_GS_L5`).
- Exportação do dataset preparado (`dataset_futebol_preparado.csv`).

### 🔹 Sprint 2 — Modeling & Evaluation (`3_Modeling.ipynb` e `4_Evaluation.ipynb`)
- Divisão temporal do dataset (`TimeSeriesSplit`) em 80% Treino e 20% Teste.
- Treino e otimização de 7 algoritmos de Machine Learning (Regressão Logística, SVM, Random Forest, Decision Tree, Naive Bayes, KNN, Gradient Boosting) via `GridSearchCV` focados no **F1-Score Macro**.
- Treino de uma Rede Neuronal MLP em **TensorFlow/Keras** com *Dropout* e *Early Stopping*.
- Seleção do modelo campeão (SVM com F1-Score Macro de 44.34%) e exportação automática do manifesto `models_manifest.json`.

---

## ⚙️ Instalação e Execução

### 1. Clonar o repositório e preparar o ambiente
```bash
git clone https://github.com/teu-usuario/projeto-futebol.git
cd projeto-futebol
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Executar o Script Principal (`main.py`)
Para rodar a pipeline de inferência com o modelo campeão a partir do manifesto:
```bash
python main.py
```

---

## 🏆 Modelo Campeão e Resultados

| Modelo | Tipo | Accuracy (%) | F1-Score Macro (%) | Precision Macro (%) | Recall Macro (%) | F1-Score Empate 'D' (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM** | **Machine Learning** | **46.82** | **44.34** | **44.51** | **44.25** | **27.96** |
| Logistic Regression | Machine Learning | 46.95 | 44.11 | 44.07 | 44.18 | 26.47 |
| Random Forest | Machine Learning | 45.01 | 43.02 | 43.36 | 42.91 | 27.44 |
| TensorFlow Neural Net | Deep Learning | 49.81 | 35.60 | 34.12 | 41.57 | 0.00 |

> **Nota:** Em previsões *pre-match* de futebol a 3 opções ($H, D, A$), valores de exatidão na casa dos 45%–50% representam o limite prático da literatura devido à estocasticidade do desporto. O modelo selecionado prioriza o equilíbrio no **F1-Score Macro** para ser capaz de prever empates e vitórias fora.