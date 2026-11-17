"""
Sistema Multiagente (Versão 2: CrewAI Framework)
Projeto IA III - ISTEC

Orquestração de uma equipa (Crew) de 3 agentes especializados em sequência:
1. Engenheiro de Deteção (Tool Caller): Executa os modelos guardados em 'models/'
   (MLP Scikit-Learn + TensorFlow) sobre novos sinais de rádio.
2. Analista de Astrofísica: Interpreta as métricas estatísticas dos sinais candidatos.
3. Redator Executivo: Compila o relatório final de observação para o observatório.
"""

import os
# Evitar pedidos de telemetria ou chave externa da OpenAI
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

from typing import Any, List, Optional
from pathlib import Path
from crewai import Agent, Task, Crew, Process, BaseLLM
from transformers import pipeline

from agent_huggingface import PulsarClassifierAgent, get_paths
import pandas as pd


class LocalHuggingFaceLLM(BaseLLM):
    """Adaptador leve para ligar modelos locais Hugging Face diretamente ao CrewAI."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model=model_id, temperature=0.1)
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            torch_dtype="auto",
        )

    def call(
        self,
        messages: str | List[dict[str, str]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        output = self.pipe(messages, max_new_tokens=300, do_sample=False)
        return output[0]["generated_text"][-1]["content"]


def run_crewai_multiagent(sample_size: int = 100):
    paths = get_paths()

    # 1. Executar a inferência nos modelos guardados em 'models/'
    test_df = pd.read_csv(paths["test_data"])
    if "target_class" in test_df.columns:
        test_df = test_df.drop(columns=["target_class"])
    sample_df = test_df.head(sample_size).copy()

    detector = PulsarClassifierAgent(paths["models"])
    results_df = detector.analyze_signals(sample_df)

    consensus = results_df[results_df["Consenso_Pulsar"] == 1]
    top_3 = results_df.sort_values(by="Prob_Sklearn_MLP", ascending=False).head(3)

    summary_context = (
        f"Sinais analisados: {len(results_df)} | "
        f"Pulsares confirmados por consenso (MLP + TensorFlow): {len(consensus)}\n"
        "Top 3 Candidatos:\n"
    )
    for idx, row in top_3.iterrows():
        summary_context += (
            f"- Sinal #{idx}: Prob_MLP={row['Prob_Sklearn_MLP']:.2%}, "
            f"Prob_TF={row['Prob_TensorFlow_DL']:.2%}, "
            f"Mean_IP={row.iloc[0]:.2f}, Excess_Kurtosis_IP={row.iloc[2]:.2f}\n"
        )

    # 2. Inicializar o LLM local para a Crew
    local_llm = LocalHuggingFaceLLM()

    # 3. Definir os Agentes da Crew
    analyst_agent = Agent(
        role="Analista de Sinais de Rádio (HTRU2)",
        goal="Avaliar a consistência entre o modelo Scikit-Learn e o TensorFlow nos sinais candidatos.",
        backstory="Especialista em processamento de sinais astronómicos e validação cruzada de modelos de IA.",
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )

    writer_agent = Agent(
        role="Coordenador Científico de Observatório",
        goal="Redigir um boletim técnico claro em Português para agendamento de telescópio.",
        backstory="Astrofísico responsável por transformar métricas técnicas em decisões de observação.",
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )

    # 4. Definir as Tarefas Sequenciais
    task_analysis = Task(
        description=(
            f"Analisa os seguintes dados extraídos pelos modelos de classificação:\n{summary_context}\n"
            "Identifica quais os sinais com maior grau de certeza e explica brevemente o porquê."
        ),
        expected_output="Lista técnica comentada dos melhores candidatos a pulsar em Português.",
        agent=analyst_agent,
    )

    task_report = Task(
        description=(
            "Com base na análise técnica anterior, escreve um boletim executivo curto em Português "
            "com título, resumo dos resultados e recomendação final de observação."
        ),
        expected_output="Boletim científico final estruturado em Markdown.",
        agent=writer_agent,
        context=[task_analysis],
    )

    # 5. Criar e executar a Crew
    pulsar_crew = Crew(
        agents=[analyst_agent, writer_agent],
        tasks=[task_analysis, task_report],
        process=Process.sequential,
        verbose=True,
    )

    final_output = pulsar_crew.kickoff()

    print("\n" + "=" * 60)
    print("BOLETIM GERADO PELA CREW (CREWAI)")
    print("=" * 60)
    print(final_output)

    report_path = paths["docs"] / "relatorio_multiagente_crewai.md"
    report_path.write_text(str(final_output), encoding="utf-8")
    print(f"\nRelatório CrewAI guardado em: {report_path}")
    return final_output


if __name__ == "__main__":
    run_crewai_multiagent()