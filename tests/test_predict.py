"""Testes básicos de dados e inferência (executar com: pytest tests/)."""
import pandas as pd
import pytest

from src.config import FEATURES, MODELS_DIR, TEST_FILE, TRAIN_FILE
from src.data_preparation import criar_features, dividir_temporal
from src.predict import PredictorFraude, PredictorTodos


def test_criar_features_uma_transacao():
    df = criar_features(pd.DataFrame([{"Time": 7200, "Amount": 0.0}]))
    assert df.loc[0, "Hour"] == 2 and df.loc[0, "Amount_log"] == 0.0


def test_divisao_temporal_nao_mistura_futuro():
    df = pd.DataFrame({"Time": range(100), "Class": [0] * 100})
    treino, teste = dividir_temporal(df, 0.2)
    assert treino["Time"].max() < teste["Time"].min()


def test_dados_preparados_sem_nulos_e_com_features():
    for ficheiro in (TRAIN_FILE, TEST_FILE):
        d = pd.read_csv(ficheiro)
        assert set(FEATURES) <= set(d.columns) and d.isnull().sum().sum() == 0


@pytest.fixture(scope="module")
def predictor():
    return PredictorFraude(MODELS_DIR)


def test_previsao_devolve_probabilidade_valida(predictor):
    linha = pd.read_csv(TEST_FILE).iloc[0].drop("Class").to_dict()
    r = predictor.prever(linha)
    assert r["previsao"] in {"Legítima", "Fraude"} and 0 <= r["probabilidade_fraude_%"] <= 100


def test_transacao_incompleta_levanta_erro(predictor):
    with pytest.raises(ValueError):
        predictor.prever({"Amount": 10})


@pytest.fixture(scope="module")
def predictor_todos():
    return PredictorTodos(MODELS_DIR)


def test_todos_os_modelos_devolvem_probabilidade_valida(predictor_todos):
    linha = pd.read_csv(TEST_FILE).iloc[0].drop("Class").to_dict()
    tabela = predictor_todos.prever(linha)
    assert len(tabela) == len(predictor_todos.modelos) == 8
    assert tabela["probabilidade_fraude_%"].between(0, 100).all()


def test_todos_os_modelos_transacao_incompleta_levanta_erro(predictor_todos):
    with pytest.raises(ValueError):
        predictor_todos.prever({"Amount": 10})


def test_dois_melhores_por_f1_e_roc_auc():
    from src.evaluate_models import dois_melhores
    df = pd.DataFrame({"Modelo": ["A", "B", "C"], "F1 Fraude (%)": [50.0, 80.0, 70.0], "ROC-AUC (%)": [99.0, 90.0, 95.0]})
    r = dois_melhores(df)
    assert list(r["F1 Fraude (%)"]["Modelo"]) == ["B", "C"] and list(r["ROC-AUC (%)"]["Modelo"]) == ["A", "C"]
