"""Versão 2 do sistema multiagente com CrewAI (biblioteca estado da arte). Mesmas tools/skills da versão 1.

Necessita de um LLM configurado para o CrewAI (ex.: Ollama local: LLM_CREWAI="ollama/qwen2.5:3b",
ou OPENAI_API_KEY). A construção da Crew não chama o LLM; só crew.kickoff() o faz.
"""
import json
import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from src.agents import tools as T
from src.agents.simulator import cenario


@tool("classificar_lote_simulado")
def classificar_lote_simulado(nome_cenario: str) -> str:
    """Classifica um lote simulado (dia_normal, ataque ou deriva_valores) e devolve os factos em JSON."""
    res = T.classificar_lote(cenario(nome_cenario, 2000))
    return json.dumps(T.analisar_lote(res), ensure_ascii=False)


@tool("cartao_do_modelo")
def cartao_do_modelo() -> str:
    """Devolve os metadados do modelo campeão de deteção de fraude."""
    return json.dumps(T.cartao_modelo(), ensure_ascii=False)


def construir_crew(llm: str | None = None) -> Crew:
    llm = llm or os.environ.get("LLM_CREWAI", "ollama/qwen2.5:3b")
    analista = Agent(role="Analista de fraude", goal="Classificar lotes de transações e identificar alertas",
                     backstory="Especialista em deteção de fraude em cartões que só usa números das ferramentas.",
                     tools=[classificar_lote_simulado, cartao_do_modelo], llm=llm, allow_delegation=False)
    relator = Agent(role="Relator", goal="Escrever um relatório diário claro em português",
                    backstory="Redator técnico; nunca inventa números.", llm=llm, allow_delegation=False)
    t1 = Task(description="Classifica o cenário '{cenario}' e devolve os factos do lote.",
              expected_output="JSON com factos do lote e modelo usado.", agent=analista)
    t2 = Task(description="Com os factos anteriores, escreve o relatório diário com 2 recomendações.",
              expected_output="Relatório em português (markdown).", agent=relator, context=[t1])
    return Crew(agents=[analista, relator], tasks=[t1, t2], process=Process.sequential)


if __name__ == "__main__":
    print(construir_crew().kickoff(inputs={"cenario": "ataque"}))
