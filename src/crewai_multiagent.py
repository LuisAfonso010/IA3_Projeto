"""
Módulo: crewai_multiagent.py
Descrição: Implementação alternativa do Sistema Multiagente recorrendo à biblioteca
           estado da arte CrewAI, integrando os classificadores guardados em 'models/'
           como ferramentas (@tool) invocadas autonomamente pelos agentes.
Projeto: IA3_Projeto
"""

from __future__ import annotations

import json
import pandas as pd
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

from src.hf_multiagent import GestorModelos


def criar_sistema_crewai(
    gestor: GestorModelos | None = None,
    model_uri: str = "ollama/qwen2.5:3b",
) -> Crew:
    """
    Constrói e devolve uma instância CrewAI configurada com dois agentes especializados:
      1. Analista de Modelos Preditivos (consulta os classificadores em 'models/')
      2. Redator de Relatórios Técnicos (elabora o relatório final em Português)

    Parâmetros:
        gestor: Instância de GestorModelos com os classificadores carregados.
        model_uri: Identificador do LLM para o CrewAI (ex.: 'ollama/qwen2.5:3b' local
                   ou 'huggingface/Qwen/Qwen2.5-3B-Instruct' com variável HF_TOKEN).
    """
    gestor_modelos = gestor or GestorModelos()
    gestor_modelos.carregar_todos_classificadores()

    llm = LLM(model=model_uri, temperature=0.2)

    @tool("Consultar Melhor Classificador de Futebol")
    def consultar_melhor_classificador(dados_json: str) -> str:
        """
        Recebe uma string JSON com os atributos de um jogo de futebol e devolve
        a previsão e as probabilidades do melhor modelo classificador guardado em 'models/'.
        """
        dados = json.loads(dados_json)
        resultado = gestor_modelos.prever_melhor_modelo(dados)
        return json.dumps(resultado, ensure_ascii=False)

    @tool("Comparar Todos os Modelos Treinados")
    def comparar_todos_modelos(dados_json: str) -> str:
        """
        Executa todos os 8 modelos treinados (Scikit-Learn e TensorFlow) sobre os dados JSON
        fornecidos e devolve uma lista comparativa das previsões.
        """
        dados = json.loads(dados_json)
        resultados = gestor_modelos.prever_todos_modelos(dados)
        return json.dumps(resultados, ensure_ascii=False)

    agente_analista = Agent(
        role="Analista Sénior de Machine Learning Desportivo",
        goal=(
            "Consultar o melhor modelo classificador e verificar o consenso entre todos os "
            "modelos treinados em 'models/' para os dados fornecidos."
        ),
        backstory=(
            "Especialista em ciência de dados e validação estatística de modelos de "
            "classificação (Scikit-Learn e TensorFlow) aplicados à previsão de resultados de futebol."
        ),
        tools=[consultar_melhor_classificador, comparar_todos_modelos],
        llm=llm,
        verbose=True,
    )

    agente_relator = Agent(
        role="Especialista em Comunicação Técnica e Relatórios de IA",
        goal="Redigir relatórios claros, rigorosos e acionáveis em Português de Portugal.",
        backstory=(
            "Consultor analítico encarregado de traduzir probabilidades estatísticas e métricas "
            "de modelos preditivos em relatórios executivos estruturados."
        ),
        llm=llm,
        verbose=True,
    )

    tarefa_analise = Task(
        description=(
            "Analisa o seguinte registo de jogo em formato JSON: {dados_jogo_json}\n"
            "1. Usa a ferramenta 'Consultar Melhor Classificador de Futebol' para obter a previsão principal.\n"
            "2. Usa a ferramenta 'Comparar Todos os Modelos Treinados' para avaliar o grau de concordância "
            "entre os modelos Scikit-Learn e TensorFlow."
        ),
        expected_output=(
            "Dicionário/resumo técnico contendo o melhor modelo, classe prevista, probabilidades "
            "e o grau de concordância entre os 8 classificadores."
        ),
        agent=agente_analista,
    )

    tarefa_relatorio = Task(
        description=(
            "Com base exclusivamente na análise produzida pela tarefa anterior, escreve um relatório "
            "técnico em Português de Portugal estruturado em:\n"
            "1. Resumo da Previsão\n"
            "2. Análise de Probabilidades e Consenso entre Modelos\n"
            "3. Recomendações e Limitações"
        ),
        expected_output="Relatório final completo formatado em Markdown.",
        agent=agente_relator,
        context=[tarefa_analise],
    )

    return Crew(
        agents=[agente_analista, agente_relator],
        tasks=[tarefa_analise, tarefa_relatorio],
        process=Process.sequential,
        verbose=True,
    )


def executar_crewai(
    dados_entrada: dict | pd.DataFrame,
    gestor: GestorModelos | None = None,
    model_uri: str = "ollama/qwen2.5:3b",
) -> str:
    """Função auxiliar de alto nível para executar o fluxo CrewAI num único passo."""
    if isinstance(dados_entrada, pd.DataFrame):
        dados_dict = dados_entrada.iloc[0].to_dict()
    else:
        dados_dict = dados_entrada

    crew = criar_sistema_crewai(gestor=gestor, model_uri=model_uri)
    resultado = crew.kickoff(inputs={"dados_jogo_json": json.dumps(dados_dict, ensure_ascii=False)})
    return str(resultado)
