"""
Sistema Multiagente (Versão 1: Hugging Face Transformers)
Projeto IA III - ISTEC

Arquitetura Multiagente:
1. Agente Detetor (ML/DL): Carrega os modelos treinados da pasta 'models/'
   (MLP Scikit-Learn + Rede Neuronal TensorFlow) e avalia sinais candidatos.
2. Agente Relator (LLM Hugging Face): Recebe o consenso dos classificadores
   e gera automaticamente um relatório técnico-científico sobre os pulsares detetados.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from transformers import pipeline


def get_paths():
    base_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
    if base_dir.name == "notebooks":
        base_dir = base_dir.parent
    return {
        "models": base_dir / "models",
        "test_data": base_dir / "data" / "processed" / "pulsar_data_test_prepared.csv",
        "docs": base_dir / "docs",
    }


class PulsarClassifierAgent:
    """Agente 1: Especialista em Classificação (carrega modelos da pasta 'models/')."""

    def __init__(self, models_dir: Path):
        self.sk_model = joblib.load(models_dir / "best_sklearn_model.joblib")
        self.tf_model = tf.keras.models.load_model(models_dir / "tensorflow_pulsar_model.keras")
        self.feature_names = joblib.load(models_dir / "feature_names.joblib")

    def analyze_signals(self, df_signals: pd.DataFrame) -> pd.DataFrame:
        X = df_signals[self.feature_names].copy()

        # Previsões do melhor modelo Scikit-Learn (MLP)
        sk_probs = self.sk_model.predict_proba(X)[:, 1]
        sk_preds = (sk_probs >= 0.5).astype(int)

        # Previsões do modelo Deep Learning (TensorFlow)
        tf_probs = self.tf_model.predict(X, verbose=0).ravel()
        tf_preds = (tf_probs >= 0.5).astype(int)

        report_df = df_signals.copy()
        report_df["Prob_Sklearn_MLP"] = np.round(sk_probs, 4)
        report_df["Prob_TensorFlow_DL"] = np.round(tf_probs, 4)
        report_df["Consenso_Pulsar"] = ((sk_preds == 1) & (tf_preds == 1)).astype(int)
        return report_df


class AstroReportAgent:
    """Agente 2: Agente Generativo (Hugging Face Transformers) que redige relatórios."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"A carregar modelo generativo Hugging Face ({model_id})...")
        self.generator = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            torch_dtype="auto",
        )

    def generate_report(self, summary_stats: dict, top_candidates_text: str) -> str:
        prompt = (
            "És um astrofísico especialista em estrelas de neutrões e pulsares (dataset HTRU2).\n"
            "Com base nos resultados produzidos pelos modelos de Machine Learning (MLP Scikit-Learn) "
            "e Deep Learning (TensorFlow), escreve um relatório técnico claro e conciso em Português "
            "com: (1) Resumo da deteção, (2) Análise dos principais candidatos a Pulsar, e "
            "(3) Recomendação de prioridade para observação por radiotelescópio.\n\n"
            f"ESTATÍSTICAS DO LOTE ANALISADO:\n"
            f"- Total de sinais analisados: {summary_stats['total']}\n"
            f"- Pulsares confirmados por consenso (ML + DL): {summary_stats['consensus_pulsars']}\n"
            f"- Candidatos apenas no TensorFlow (alta sensibilidade): {summary_stats['tf_only']}\n\n"
            f"PRINCIPAIS CANDIDATOS DETETADOS:\n{top_candidates_text}\n\n"
            "RELATÓRIO TÉCNICO:"
        )

        messages = [{"role": "user", "content": prompt}]
        output = self.generator(messages, max_new_tokens=350, do_sample=False)
        return output[0]["generated_text"][-1]["content"]


def run_huggingface_multiagent(sample_size: int = 100):
    paths = get_paths()

    # 1. Carregar amostra de sinais de teste não rotulados
    test_df = pd.read_csv(paths["test_data"])
    if "target_class" in test_df.columns:
        test_df = test_df.drop(columns=["target_class"])
    sample_df = test_df.head(sample_size).copy()

    # 2. Executar Agente 1 (Classificação)
    detector = PulsarClassifierAgent(paths["models"])
    results_df = detector.analyze_signals(sample_df)

    consensus = results_df[results_df["Consenso_Pulsar"] == 1]
    tf_only = results_df[(results_df["Prob_TensorFlow_DL"] >= 0.5) & (results_df["Consenso_Pulsar"] == 0)]

    stats = {
        "total": len(results_df),
        "consensus_pulsars": len(consensus),
        "tf_only": len(tf_only),
    }

    top_3 = results_df.sort_values(by="Prob_Sklearn_MLP", ascending=False).head(3)
    candidates_str = ""
    for idx, row in top_3.iterrows():
        candidates_str += (
            f"- Sinal #{idx}: Prob. MLP={row['Prob_Sklearn_MLP']:.2%}, "
            f"Prob. TensorFlow={row['Prob_TensorFlow_DL']:.2%}, "
            f"Mean_IP={row.iloc[0]:.2f}, Excess_Kurtosis_IP={row.iloc[2]:.2f}\n"
        )

    # 3. Executar Agente 2 (Geração de Relatório via LLM)
    reporter = AstroReportAgent()
    report_text = reporter.generate_report(stats, candidates_str)

    print("\n" + "=" * 60)
    print("RELATÓRIO GERADO PELO SISTEMA MULTIAGENTE (HUGGING FACE)")
    print("=" * 60)
    print(report_text)

    # Guardar relatório gerado em docs/
    report_path = paths["docs"] / "relatorio_multiagente_hf.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nRelatório guardado em: {report_path}")
    return results_df, report_text


if __name__ == "__main__":
    run_huggingface_multiagent()