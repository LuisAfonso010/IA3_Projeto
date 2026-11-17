# Deteção de Pulsares (HTRU2) com Machine Learning, Deep Learning e Sistemas Multiagente

Projeto desenvolvido no âmbito da unidade curricular de **Inteligência Artificial III** (ISTEC), seguindo as metodologias **SCRUM** (gestão do projeto) e **CRISP-ML** (implementação técnica).

## Objetivo do Projeto

Identificar sinais de rádio provenientes de estrelas de neutrões (**pulsares**) no dataset **HTRU2**, superando o forte desequilíbrio de classes (apenas **9,2%** de pulsares reais) através de:
1. Pipeline completa de preparação de dados e modelação supervisionada (**8 algoritmos Scikit-Learn + 1 Rede Neuronal em TensorFlow/Keras**);
2. Integração dos modelos preditivos com **dois Sistemas Multiagente** distintos (**Hugging Face Transformers** e **CrewAI**) para análise cruzada e geração automática de boletins científicos.

## Estrutura do Repositório

- data/raw/ : Dataset original HTRU2
- data/processed/ : Dados tratados (imputação mediana + StandardScaler)
- docs/ : Gráficos EDA, Documentação Why e relatórios gerados pelos agentes
- models/ : Modelos treinados (.joblib e .keras) prontos para inferência
- notebooks/pulsar_pipeline_pronto.ipynb : Notebook integrador modularizado
- src/eda_pulsar.py : Análise exploratória e geração de gráficos
- src/data_preparation.py : Limpeza, tratamento de outliers e normalização
- src/train_model.py : Treino e comparação de 8 modelos Scikit-Learn + 1 TensorFlow
- src/tune_model.py : Otimização de hiperparâmetros
- src/agent_huggingface.py : Sistema Multiagente v1 (Transformers / Qwen2.5)
- src/agent_crewai.py : Sistema Multiagente v2 (Orquestração CrewAI)
- main.py : Pipeline Python end-to-end totalmente funcional

## Resultados dos Modelos (Validação Estratificada)

| Modelo | Biblioteca | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MLP (Rede Neuronal)** | Scikit-Learn | **0.9519** | 0.8571 | **0.9021** | 0.9759 |
| **Random Forest** | Scikit-Learn | 0.9189 | 0.8831 | 0.9007 | 0.9779 |
| **Gradient Boosting** | Scikit-Learn | 0.9128 | 0.8615 | 0.8864 | **0.9801** |
| **KNN (k=7)** | Scikit-Learn | 0.9403 | 0.8182 | 0.8750 | 0.9512 |
| **SVM (RBF)** | Scikit-Learn | 0.8038 | **0.9048** | 0.8513 | 0.9770 |
| **Rede Neuronal Densa** | TensorFlow | 0.7828 | **0.9048** | 0.8394 | 0.9766 |
| **Árvore de Decisão** | Scikit-Learn | 0.7473 | 0.8961 | 0.8150 | 0.9447 |
| **Regressão Logística** | Scikit-Learn | 0.7213 | 0.8961 | 0.7992 | 0.9699 |
| **Naive Bayes** | Scikit-Learn | 0.6378 | 0.8615 | 0.7330 | 0.9573 |

## Gestão SCRUM (Sprints)

| Sprint | Fase CRISP-ML | Entregáveis Principais |
| :--- | :--- | :--- |
| **Sprint 1** | *Business & Data Understanding* | Validação do dataset HTRU2, Contrato de Dados e estruturação do repositório. |
| **Sprint 2** | *Data Preparation* | Análise exploratória, imputação pela mediana, tratamento de outliers e StandardScaler. |
| **Sprint 3** | *Modeling & Evaluation* | Treino/validação de 8 modelos Scikit-Learn + 1 rede TensorFlow e exportação para models/. |
| **Sprint 4** | *Deployment & Multiagent Systems* | Implementação dos agentes (Hugging Face e CrewAI), notebook modularizado, main.py e documentação. |

## Como Executar

Executar no terminal: python main.py
