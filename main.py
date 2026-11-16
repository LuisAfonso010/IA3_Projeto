"""
Ficheiro: main.py
Descrição: Versão em Python puro (totalmente funcional e equivalente ao notebook
           'notebooks/Deployment_Agents.ipynb') que carrega todos os classificadores
           e agentes generativos guardados em 'models/' e executa o sistema multiagente.
Projeto: IA3_Projeto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garantir que a raiz do projeto está no PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hf_multiagent import GestorModelos, SistemaMultiagenteHF, preparar_agente_generativo


def executar_pipeline_completo(modo: str = "hf", crewai_model: str = "ollama/qwen2.5:3b") -> None:
    print("=" * 75)
    print("IA3_Projeto — Pipeline de Deployment: Classificadores + Sistema Multiagente")
    print("=" * 75)

    # 1. Carregar todos os modelos treinados (7 Scikit-Learn + 1 TensorFlow) de 'models/'
    gestor = GestorModelos()
    modelos = gestor.carregar_todos_classificadores()
    print(f"\n[1/4] Modelos carregados de '{gestor.models_dir}': {len(modelos)} classificadores")
    for nome, tipo in gestor.tipos_modelos.items():
        print(f"      - {nome} ({tipo})")

    # 2. Obter uma amostra real do dataset preparado para demonstração
    amostra_df = gestor.obter_amostra_dataset(indice=-1)
    print(f"\n[2/4] Amostra de teste extraída ({amostra_df.shape[1]} atributos).")

    # 3. Inferência em todos os modelos treinados
    print("\n[3/4] Resultados de inferência em todos os classificadores guardados:")
    todos_resultados = gestor.prever_todos_modelos(amostra_df)
    for res in todos_resultados:
        print(
            f"      * {res['model_name']:<26} -> Previsão: {res['prediction']} | Probs: {res['probabilities']}"
        )

    # 4. Execução do Sistema Multiagente Generativo
    print(f"\n[4/4] A executar Sistema Multiagente (modo='{modo}')...")

    if modo in ("hf", "todos"):
        preparar_agente_generativo()
        sistema_hf = SistemaMultiagenteHF(gestor=gestor)
        saida_hf = sistema_hf.executar(amostra_df)
        print("\n--- RELATÓRIO GERADO PELO SISTEMA MULTIAGENTE (HUGGING FACE) ---")
        print(saida_hf["relatorio"])

    if modo in ("crewai", "todos"):
        from src.crewai_multiagent import executar_crewai

        print("\n--- RELATÓRIO GERADO PELO SISTEMA MULTIAGENTE (CREWAI) ---")
        saida_crew = executar_crewai(amostra_df, gestor=gestor, model_uri=crewai_model)
        print(saida_crew)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa os modelos e agentes de IA3_Projeto.")
    parser.add_argument(
        "--modo",
        choices=["hf", "crewai", "todos"],
        default="hf",
        help="Define qual implementação multiagente executar: 'hf' (Hugging Face), 'crewai' ou 'todos'.",
    )
    parser.add_argument(
        "--crewai-model",
        default="ollama/qwen2.5:3b",
        help="URI do modelo LLM usado pelo CrewAI (ex.: 'ollama/qwen2.5:3b').",
    )
    args = parser.parse_args()
    executar_pipeline_completo(modo=args.modo, crewai_model=args.crewai_model)
