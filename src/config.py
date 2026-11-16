"""Configuração central do projeto (caminhos, constantes e lista de features)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_FILE = DATA_RAW / "creditcard.csv"
TRAIN_FILE = DATA_PROCESSED / "creditcard_train_prepared.csv"
TEST_FILE = DATA_PROCESSED / "creditcard_test_prepared.csv"
MANIFEST_FILE = MODELS_DIR / "models_manifest.json"

TARGET = "Class"                      # 0 = transação legítima, 1 = fraude
RANDOM_STATE = 42
TEST_SIZE = 0.20                      # divisão temporal: 80% mais antigo / 20% mais recente

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_FEATURES + ["Amount", "Amount_log", "Hour"]

# Modelos pesados (SVM, KNN, Gradient Boosting) são otimizados numa amostra:
# todas as fraudes + N transações legítimas (trade-off tempo/qualidade).
SAMPLE_LEGIT = {"SVM": 15_000, "KNN": 40_000, "Gradient Boosting": 40_000}
CV_SPLITS = 3

# Contrato de dados (formato, nomes, tipos, unidades, frequência, componente temporal)
DATA_CONTRACT = {
    "formato": "CSV (separador ',', cabeçalho na 1ª linha, codificação UTF-8)",
    "granularidade": "1 linha = 1 transação de cartão de crédito",
    "componente_temporal": "Sim - 'Time' = segundos desde a 1ª transação (~48h, 2 dias de set/2013)",
    "frequencia_amostragem": "Irregular (evento: cada transação); sem frequência fixa",
    "unidades": {"Time": "segundos", "Amount": "euros (EUR)"},
    "colunas": {"Time": "int/float", **{c: "float" for c in PCA_FEATURES}, "Amount": "float", "Class": "int {0,1}"},
    "nota": "V1..V28 são componentes PCA (anonimizadas por confidencialidade); sem valores nulos.",
}
