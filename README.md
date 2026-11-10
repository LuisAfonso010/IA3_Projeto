# IA3_Projeto — Previsão e Análise de Dados de Futebol

Projeto desenvolvido no âmbito da unidade curricular de **Inteligência Artificial 3 (IA3)**, focado no processamento de dados históricos de futebol e na construção de modelos de Machine Learning para a previsão de resultados.

---

## 👤 Autor

* **Nome / GitHub:** [RodrigoLima2025128](https://github.com RodrigoLima2025128)

---

## 📁 Estrutura do Projeto

A organização dos ficheiros segue a metodologia padrão de projetos de Data Science e Machine Learning (CRISP-DM):

```text
IA3_Projeto/
├── data/
│   ├── processed/
│   │   └── dataset_futebol_preparado.csv    # Dataset final limpo e transformado
│   └── raw/
│       └── Dataset/                        # Dados brutos de 22 épocas (2000-01 a 2021-2022)
├── docs/
│   ├── Data Understanding.pdf               # Documentação detalhada da fase de entendimento
│   ├── Sprint 0.docx                        # Documentação da Sprint 0
│   ├── Sprint 1.docx                        # Documentação da Sprint 1
│   └── Sprint 2.docx                        # Documentação da Sprint 2
├── models/
│   ├── decision_tree_opt.joblib             # Modelo Decision Tree otimizado
│   ├── gradient_boosting_opt.joblib         # Modelo Gradient Boosting otimizado
│   ├── knn_opt.joblib                       # Modelo K-Nearest Neighbors otimizado
│   ├── logistic_regression_opt.joblib       # Modelo Regressão Logística otimizado
│   ├── naive_bayes_opt.joblib               # Modelo Naive Bayes otimizado
│   ├── random_forest_opt.joblib             # Modelo Random Forest otimizado
│   ├── svm_opt.joblib                       # Modelo Support Vector Machine otimizado
│   ├── tensorflow_mlp_opt.keras             # Redes Neuronais (MLP com TensorFlow)
│   └── models_manifest.json                 # Metadados e desempenho dos modelos
├── notebooks/
│   ├── Data_Understanding.ipynb             # Análise exploratória dos dados (EDA)
│   ├── Data_Preparation.ipynb               # Limpeza, seleção e engenharia de atributos
│   ├── Modeling.ipynb                       # Treino e afinação de hiperparâmetros
│   └── Evaluation.ipynb                     # Avaliação comparativa de desempenho
├── src/
│   ├── __init__.py
│   └── predict.py                           # Módulo de inferência e previsão
├── .gitignore                               # Ficheiro de exclusão do Git
├── Dataset Pessoal                          # Ficheiro complementar de suporte
├── Ideia de Dataset                         # Notas sobre o desenvolvimento do dataset
├── main.py                                  # Script principal de execução da aplicação
├── README.md                                # Documentação do projeto
└── requirements.txt                         # Dependências do projeto em Python
```

---

## ⚙️ Modelos Treinados

O projeto inclui a implementação, treino e afinação de vários algoritmos de aprendizagem automática:

* **Decision Tree Classifier** (`decision_tree_opt.joblib`)
* **Gradient Boosting Classifier** (`gradient_boosting_opt.joblib`)
* **K-Nearest Neighbors (KNN)** (`knn_opt.joblib`)
* **Logistic Regression** (`logistic_regression_opt.joblib`)
* **Naive Bayes** (`naive_bayes_opt.joblib`)
* **Random Forest Classifier** (`random_forest_opt.joblib`)
* **Support Vector Machine (SVM)** (`svm_opt.joblib`)
* **Multi-Layer Perceptron (TensorFlow/Keras)** (`tensorflow_mlp_opt.keras`)

---

## 🚀 Instalação e Execução

### 1. Clonar o Repositório

```bash
git clone https://github.com/RodrigoLima2025128/IA3_Projeto.git
cd IA3_Projeto
```

### 2. Criar e Ativar Ambiente Virtual (Opcional, mas Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o Projeto

Para executar o pipeline principal de previsão:

```bash
python main.py
```

---

## 📊 Metodologia do Trabalho

O fluxo de trabalho está estruturado em 4 notebooks sequenciais presentes na pasta `notebooks/`:

1. **`Data_Understanding.ipynb`**: Análise estatística e visual dos dados históricos de 22 épocas de futebol (2000/2001 a 2021/2022).
2. **`Data_Preparation.ipynb`**: Tratamento de valores omissos, normalização, tratamento de *outliers* e *feature engineering*.
3. **`Modeling.ipynb`**: Construção e treino dos algoritmos de classificação com pesquisa de hiperparâmetros.
4. **`Evaluation.ipynb`**: Medição de métricas (Accuracy, Precision, Recall, F1-Score, ROC-AUC) para seleção do melhor modelo de previsão.

---

## 📄 Licença e Documentação

Toda a documentação adicional sobre os *sprints* de desenvolvimento e o entendimento dos dados encontra-se armazenada na pasta `docs/`.