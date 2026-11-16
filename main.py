import os

import pandas as pd

from src.config import MODELS_DIR, TEST_FILE
from src.predict import PredictorFraude


def main():
    print("=" * 70)
    print("💳 SISTEMA DE DETEÇÃO DE FRAUDE EM CARTÕES DE CRÉDITO (CRISP-ML)")
    print("=" * 70 + "\n")

    if not os.path.exists(MODELS_DIR):
        print(f"❌ Pasta de modelos não encontrada: {MODELS_DIR}")
        return

    try:
        predictor = PredictorFraude(MODELS_DIR)
        print(f"✅ Modelo Campeão Carregado: {predictor.nome_campeao}")
        print(f"📋 Ficheiro Utilizado:       {predictor.ficheiro_campeao}\n")
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo campeão: {e}")
        return

    # Exemplos de teste: 1 fraude real e 1 legítima real do conjunto de teste (já com Hour/Amount_log)
    teste = pd.read_csv(TEST_FILE)
    exemplos = {"Transação FRAUDULENTA (real)": teste[teste["Class"] == 1].iloc[0],
                "Transação LEGÍTIMA (real)": teste[teste["Class"] == 0].iloc[0]}

    for titulo, linha in exemplos.items():
        transacao = linha.drop("Class").to_dict()
        resultado = predictor.prever(transacao)
        print("=" * 50)
        print(f"🎯 {titulo}")
        print("=" * 50)
        print(f"🤖 Modelo Ativo:        {resultado['modelo']}")
        print(f"🏆 Previsão Principal:  {resultado['previsao']}")
        print(f"📈 Prob. de fraude:     {resultado['probabilidade_fraude_%']}% (limiar {resultado['limiar_%']}%)")
        print(f"✔️  Classe real:         {'Fraude' if linha['Class'] == 1 else 'Legítima'}\n")


if __name__ == "__main__":
    main()
