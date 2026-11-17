"""
Pipeline Principal End-to-End - Previsão de Pulsares (HTRU2)
Projeto IA III - ISTEC

Executa todo o fluxo CRISP-ML de forma modular:
1. Treino e avaliação dos modelos (Scikit-Learn + TensorFlow) -> guarda em 'models/'
2. Execução do Sistema Multiagente v1 (Hugging Face Transformers)
3. Execução do Sistema Multiagente v2 (CrewAI)
"""

import sys
from pathlib import Path

# Adicionar pasta 'src' ao path para importar os módulos
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC_DIR))

from train_model import train_and_evaluate_all
from agent_huggingface import run_huggingface_multiagent
from agent_crewai import run_crewai_multiagent


def main():
    print("\n>>> FASE 1: TREINO E AVALIAÇÃO DE MODELOS (ML + DL) <<<")
    train_and_evaluate_all()

    print("\n>>> FASE 2: SISTEMA MULTIAGENTE V1 (HUGGING FACE) <<<")
    run_huggingface_multiagent(sample_size=100)

    print("\n>>> FASE 3: SISTEMA MULTIAGENTE V2 (CREWAI) <<<")
    run_crewai_multiagent(sample_size=100)

    print("\n>>> PIPELINE CONCLUÍDA COM SUCESSO! <<<")


if __name__ == "__main__":
    main()