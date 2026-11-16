"""
Módulo: hf_multiagent.py
Descrição: Sistema Multiagente Generativo baseado na biblioteca Transformers (Hugging Face)
           e carregamento unificado de todos os modelos treinados (classificadores, scaler,
           encoder e agente generativo) guardados no diretório 'models/'.
           Totalmente compatível com Google Colab (Google Drive) e execução local.
Projeto: IA3_Projeto
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Modelo LLM leve e eficiente em Português (executa tanto em CPU como em GPU)
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def obter_raiz_projeto() -> Path:
    """
    Resolve automaticamente a raiz do projeto:
      1. Diretório atual ou pasta pai (execução local ou em 'notebooks/')
      2. Caminhos no Google Drive quando executado no Google Colab
    """
    atual = Path.cwd().resolve()
    for candidato in (atual, atual.parent):
        if (candidato / "models").exists():
            return candidato

    colab_candidatos = [
        Path("/content/drive/MyDrive/Projeto_Futebol"),
        Path("/content/drive/MyDrive/IA3_Projeto"),
        Path("/content/IA3_Projeto"),
    ]
    for candidato in colab_candidatos:
        if (candidato / "models").exists():
            return candidato

    # Pesquisa até 2 níveis dentro de /content/drive/MyDrive se existir
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        for models_dir in mydrive.glob("*/models"):
            return models_dir.parent
        for models_dir in mydrive.glob("*/*/models"):
            return models_dir.parent

    return atual


class GestorModelos:
    """
    Carrega todos os artefactos guardados em 'models/' (modelos Scikit-Learn .joblib,
    modelo TensorFlow .keras, StandardScaler, LabelEncoder e Agente Generativo HF)
    e executa inferência unificada.
    """

    def __init__(self, models_dir: Path | None = None):
        raiz = obter_raiz_projeto()
        self.raiz = raiz
        self.models_dir = Path(models_dir) if models_dir else (raiz / "models")
        self.data_dir = raiz / "data" / "processed"
        self.manifest = self._carregar_manifesto()
        self.modelos: dict[str, Any] = {}
        self.tipos_modelos: dict[str, str] = {}
        self.scaler: Any = None
        self.label_encoder: Any = None

    def _carregar_manifesto(self) -> dict:
        manifest_path = self.models_dir / "models_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {}

    def carregar_todos_classificadores(self) -> dict[str, Any]:
        """
        Carrega todos os ficheiros .joblib e .keras em 'models/', detetando
        automaticamente classificadores, StandardScaler e LabelEncoder.
        """
        self.modelos.clear()
        self.tipos_modelos.clear()
        self.scaler = None
        self.label_encoder = None

        # 1. Ficheiros .joblib (classificadores Scikit-Learn, scaler e encoder)
        for caminho in sorted(self.models_dir.glob("*.joblib")):
            nome = caminho.stem
            obj = joblib.load(caminho)

            if hasattr(obj, "predict"):
                self.modelos[nome] = obj
                self.tipos_modelos[nome] = "scikit-learn"
            elif hasattr(obj, "transform") and not hasattr(obj, "inverse_transform"):
                self.scaler = obj
            elif hasattr(obj, "inverse_transform"):
                self.label_encoder = obj

        # 2. Modelo Deep Learning TensorFlow / Keras (.keras)
        for caminho in sorted(self.models_dir.glob("*.keras")):
            nome = caminho.stem
            try:
                import tensorflow as tf

                self.modelos[nome] = tf.keras.models.load_model(caminho)
                self.tipos_modelos[nome] = "tensorflow"
            except ImportError as exc:
                raise ImportError(
                    "TensorFlow não está instalado. Instala com 'pip install tensorflow'."
                ) from exc

        return self.modelos

    def obter_nome_melhor_modelo(self) -> str:
        """Identifica o melhor modelo a partir do models_manifest.json ou usa fallback."""
        for chave in ("best_model", "melhor_modelo", "selected_model", "champion_model"):
            if chave in self.manifest and self.manifest[chave]:
                nome = Path(str(self.manifest[chave])).stem
                if nome in self.modelos:
                    return nome

        for candidato in ("gradient_boosting_opt", "random_forest_opt", "tensorflow_mlp_opt"):
            if candidato in self.modelos:
                return candidato

        if self.modelos:
            return next(iter(self.modelos.keys()))

        raise FileNotFoundError(f"Nenhum modelo encontrado em {self.models_dir}")

    def obter_amostra_dataset(self, indice: int = -1) -> pd.DataFrame:
        """
        Extrai uma amostra real do dataset preparado para testes de inferência,
        removendo automaticamente a coluna alvo (FTR) e mantendo os atributos numéricos.
        """
        csv_path = self.data_dir / "dataset_futebol_preparado.csv"
        if not csv_path.exists():
            # Fallback caso o CSV esteja diretamente em data/
            alt_csv = self.raiz / "data" / "dataset_futebol_preparado.csv"
            if alt_csv.exists():
                csv_path = alt_csv
            else:
                raise FileNotFoundError(f"Dataset preparado não encontrado em {csv_path}")

        df = pd.read_csv(csv_path)
        possiveis_alvos = [
            self.manifest.get("target_column"),
            self.manifest.get("target"),
            "FTR",
            "target",
            "resultado",
            "Result",
        ]
        colunas_remover = [c for c in possiveis_alvos if c and c in df.columns]
        X = df.drop(columns=colunas_remover)

        features_manifesto = self.manifest.get("features") or self.manifest.get("feature_names")
        if features_manifesto:
            colunas_validas = [c for c in features_manifesto if c in X.columns]
            if colunas_validas:
                X = X[colunas_validas]

        return X.iloc[[indice]].copy()

    def _preparar_entrada(self, dados_entrada: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(dados_entrada, pd.DataFrame):
            return dados_entrada.copy()
        return pd.DataFrame([dados_entrada])

    def _adicionar_features_diferenca(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstrói as 3 variáveis de engenharia de features criadas no treino
        (Diff_Forma_L5, Diff_GM_L5, Diff_GS_L5) a partir das 6 variáveis base.
        """
        X = df.copy()
        cols_lower = {c.lower(): c for c in X.columns}

        pares_conhecidos = [
            (
                "Diff_Forma_L5",
                ("forma_casa", "home_forma", "forma_h", "form_h", "pts_h"),
                ("forma_fora", "away_forma", "forma_a", "form_a", "pts_a"),
            ),
            (
                "Diff_GM_L5",
                ("gm_casa", "home_gm", "gm_h", "golos_marcados_casa", "gs_for_h"),
                ("gm_fora", "away_gm", "gm_a", "golos_marcados_fora", "gs_for_a"),
            ),
            (
                "Diff_GS_L5",
                ("gs_casa", "home_gs", "gs_h", "golos_sofridos_casa", "gs_ag_h"),
                ("gs_fora", "away_gs", "gs_a", "golos_sofridos_fora", "gs_ag_a"),
            ),
        ]

        criadas = 0
        for nome_diff, prefixos_casa, prefixos_fora in pares_conhecidos:
            if nome_diff in X.columns:
                criadas += 1
                continue
            col_c = next(
                (orig for low, orig in cols_lower.items() if any(p in low for p in prefixos_casa)),
                None,
            )
            col_f = next(
                (orig for low, orig in cols_lower.items() if any(p in low for p in prefixos_fora)),
                None,
            )
            if col_c and col_f:
                X[nome_diff] = (X[col_c] - X[col_f]).abs()
                criadas += 1

        # Fallback por pares adjacentes (0-1, 2-3, 4-5) caso os nomes das colunas sejam diferentes
        if criadas < 3 and X.shape[1] >= 6:
            nomes_fallback = ["Diff_Forma_L5", "Diff_GM_L5", "Diff_GS_L5"]
            cols_base = list(X.columns[:6])
            for idx_par, nome_diff in enumerate(nomes_fallback):
                if nome_diff not in X.columns:
                    c1 = cols_base[idx_par * 2]
                    c2 = cols_base[idx_par * 2 + 1]
                    X[nome_diff] = (X[c1] - X[c2]).abs()

        return X

    def _alinhar_features_para_modelo(
        self, modelo: Any, tipo: str, X_raw: pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        """
        Garante que o número de colunas e o escalonamento (StandardScaler) coincidem
        exatamente com o que cada modelo espera (ex.: 9 features após engenharia de features).
        """
        X_num = X_raw.select_dtypes(include=[np.number]).copy()
        n_esperado = getattr(modelo, "n_features_in_", None)
        if n_esperado is None and tipo == "tensorflow" and hasattr(modelo, "input_shape"):
            n_esperado = modelo.input_shape[-1]

        n_scaler = getattr(self.scaler, "n_features_in_", None) if self.scaler is not None else None
        alvo_dim = n_esperado or n_scaler

        # Se o modelo ou scaler espera 9 features e temos menos (ex.: 6 base), gerar as 3 Diff_*
        if alvo_dim is not None and X_num.shape[1] < alvo_dim:
            X_exp = self._adicionar_features_diferenca(X_num)
        else:
            X_exp = X_num

        # Garantir dimensão exata mesmo que alguma coluna esteja ausente
        if alvo_dim is not None and X_exp.shape[1] < alvo_dim:
            for i in range(X_exp.shape[1], alvo_dim):
                X_exp[f"feature_pad_{i}"] = 0.0
        elif alvo_dim is not None and X_exp.shape[1] > alvo_dim:
            X_exp = X_exp.iloc[:, :alvo_dim]

        # Caso 1: Modelo scikit-learn com nomes de colunas registados
        if hasattr(modelo, "feature_names_in_"):
            cols_modelo = list(modelo.feature_names_in_)
            return X_exp.reindex(columns=cols_modelo, fill_value=0.0)

        # Caso 2: Aplicar StandardScaler se existir e tiver dimensão compatível
        if self.scaler is not None and not hasattr(modelo, "steps"):
            if n_scaler is None or n_scaler == X_exp.shape[1]:
                if hasattr(self.scaler, "feature_names_in_"):
                    X_for_scaler = X_exp.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)
                    return self.scaler.transform(X_for_scaler)
                return self.scaler.transform(X_exp.to_numpy(dtype=np.float64))

        return X_exp.to_numpy(dtype=np.float64)

    def _descodificar_classe(self, valor: Any) -> str:
        """Converte índices numéricos (0, 1, 2) de volta para ('A', 'D', 'H') se houver LabelEncoder."""
        if self.label_encoder is not None:
            try:
                if isinstance(valor, (int, np.integer)) or str(valor).isdigit():
                    return str(self.label_encoder.inverse_transform([int(valor)])[0])
            except Exception:
                pass
        return str(valor)

    def _obter_labels_classes(self, modelo: Any, n_classes: int) -> list[str]:
        if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            return [str(c) for c in self.label_encoder.classes_]
        if self.manifest.get("classes"):
            return [str(c) for c in self.manifest["classes"]]
        if hasattr(modelo, "classes_"):
            return [self._descodificar_classe(c) for c in modelo.classes_]
        if n_classes == 3:
            return ["A", "D", "H"]
        return [f"Classe_{i}" for i in range(n_classes)]

    def prever_com_modelo(self, nome_modelo: str, dados_entrada: dict | pd.DataFrame) -> dict:
        """Executa previsão num modelo específico (scikit-learn ou TensorFlow)."""
        if not self.modelos:
            self.carregar_todos_classificadores()

        if nome_modelo not in self.modelos:
            raise KeyError(
                f"Modelo '{nome_modelo}' não encontrado. Disponíveis: {list(self.modelos.keys())}"
            )

        modelo = self.modelos[nome_modelo]
        tipo = self.tipos_modelos[nome_modelo]
        X_raw = self._preparar_entrada(dados_entrada)
        X_proc = self._alinhar_features_para_modelo(modelo, tipo, X_raw)

        if tipo == "scikit-learn":
            pred_raw = modelo.predict(X_proc)[0]
            pred_val = pred_raw.item() if hasattr(pred_raw, "item") else pred_raw
            previsao = self._descodificar_classe(pred_val)
            probabilidades = {}
            if hasattr(modelo, "predict_proba"):
                probs = modelo.predict_proba(X_proc)[0]
                labels = self._obter_labels_classes(modelo, len(probs))
                probabilidades = {str(lbl): round(float(p), 4) for lbl, p in zip(labels, probs)}
        else:
            arr_in = np.asarray(X_proc, dtype=np.float32)
            probs = modelo.predict(arr_in, verbose=0)[0]
            if np.ndim(probs) == 0 or len(probs) == 1:
                p1 = float(np.ravel(probs)[0])
                probs_arr = [1.0 - p1, p1]
            else:
                probs_arr = [float(p) for p in probs]
            labels = self._obter_labels_classes(modelo, len(probs_arr))
            idx_max = int(np.argmax(probs_arr))
            previsao = labels[idx_max]
            probabilidades = {str(lbl): round(p, 4) for lbl, p in zip(labels, probs_arr)}

        return {
            "model_name": nome_modelo,
            "framework": tipo,
            "prediction": previsao,
            "probabilities": probabilidades,
        }

    def prever_todos_modelos(self, dados_entrada: dict | pd.DataFrame) -> list[dict]:
        """Invoca todos os classificadores treinados guardados em 'models/'."""
        if not self.modelos:
            self.carregar_todos_classificadores()

        resultados = []
        for nome_modelo in self.modelos:
            res = self.prever_com_modelo(nome_modelo, dados_entrada)
            resultados.append(res)
        return resultados

    def prever_melhor_modelo(self, dados_entrada: dict | pd.DataFrame) -> dict:
        """
        Tenta delegar primeiro para 'src.predict' (caso exista uma função compatível),
        caso contrário executa diretamente o melhor modelo de 'models/'.
        """
        try:
            mod_predict = importlib.import_module("src.predict")
            for fn_name in ("prever_jogo", "predict", "prever", "make_prediction"):
                if hasattr(mod_predict, fn_name):
                    fn = getattr(mod_predict, fn_name)
                    saida = fn(dados_entrada)
                    if isinstance(saida, dict) and "prediction" in saida:
                        return saida
        except Exception:
            pass

        if not self.modelos:
            self.carregar_todos_classificadores()
        melhor = self.obter_nome_melhor_modelo()
        return self.prever_com_modelo(melhor, dados_entrada)


def preparar_agente_generativo(
    model_id: str = DEFAULT_HF_MODEL,
    destino: Path | None = None,
) -> Path:
    """
    Descarrega o LLM da Hugging Face e guarda os pesos, tokenizer e configuração
    localmente em 'models/generative_agent/' para que todos os modelos (classificadores
    e generativos) fiquem centralizados no diretório 'models/'.
    """
    raiz = obter_raiz_projeto()
    pasta_destino = Path(destino) if destino else (raiz / "models" / "generative_agent")
    pasta_destino.mkdir(parents=True, exist_ok=True)
    config_path = pasta_destino / "agent_config.json"

    if not (pasta_destino / "config.json").exists():
        print(f"[HF] A descarregar e guardar modelo generativo '{model_id}' em '{pasta_destino}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        tokenizer.save_pretrained(pasta_destino)
        model.save_pretrained(pasta_destino)

    metadata = {
        "base_model": model_id,
        "local_dir": str(pasta_destino),
        "agents": ["AgenteAnalistaPreditivo", "AgenteRelatorGenerativo"],
        "temperature": 0.3,
        "max_new_tokens": 512,
    }
    config_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return pasta_destino


class AgenteAnalistaPreditivo:
    """
    Agente 1 (Analista): Interage com o melhor modelo classificador guardado em 'models/',
    executa a inferência e compara opcionalmente com os restantes modelos treinados.
    """

    def __init__(self, gestor: GestorModelos | None = None):
        self.gestor = gestor or GestorModelos()
        self.gestor.carregar_todos_classificadores()

    def analisar(self, dados_entrada: dict | pd.DataFrame, incluir_comparacao: bool = True) -> dict:
        resultado_principal = self.gestor.prever_melhor_modelo(dados_entrada)
        comparacao = self.gestor.prever_todos_modelos(dados_entrada) if incluir_comparacao else []

        return {
            "melhor_modelo": resultado_principal["model_name"],
            "framework": resultado_principal.get("framework", "desconhecido"),
            "previsao": resultado_principal["prediction"],
            "probabilidades": resultado_principal.get("probabilities", {}),
            "metricas_manifesto": self.gestor.manifest.get("best_model_metrics", {}),
            "comparacao_todos_modelos": comparacao,
        }


class AgenteRelatorGenerativo:
    """
    Agente 2 (Relator): Carrega o modelo generativo guardado em 'models/generative_agent/'
    via Hugging Face Transformers e redige um relatório técnico fundamentado.
    """

    def __init__(self, agent_dir: Path | None = None):
        raiz = obter_raiz_projeto()
        self.agent_dir = Path(agent_dir) if agent_dir else (raiz / "models" / "generative_agent")

        if not (self.agent_dir / "config.json").exists():
            preparar_agente_generativo(destino=self.agent_dir)

        self.config = json.loads((self.agent_dir / "agent_config.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.agent_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.agent_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def redigir_relatorio(self, analise: dict) -> str:
        resumo_comparacao = [
            f"{m['model_name']} ({m['framework']}): {m['prediction']} {m['probabilities']}"
            for m in analise.get("comparacao_todos_modelos", [])
        ]

        mensagens = [
            {
                "role": "system",
                "content": (
                    "És um analista sénior de Ciência de Dados aplicada ao Desporto. "
                    "Elabora relatórios técnicos rigorosos em Português de Portugal, "
                    "baseando-te exclusivamente nas previsões e probabilidades fornecidas "
                    "(onde FTR: H = Vitória Casa, D = Empate, A = Vitória Fora)."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Redige um relatório técnico de apoio à decisão sobre esta previsão de futebol:\n"
                    f"- Melhor modelo selecionado: {analise['melhor_modelo']} ({analise['framework']})\n"
                    f"- Previsão principal (FTR): {analise['previsao']}\n"
                    f"- Probabilidades: {json.dumps(analise['probabilidades'], ensure_ascii=False)}\n"
                    f"- Métricas registadas no manifesto: {json.dumps(analise['metricas_manifesto'], ensure_ascii=False)}\n"
                    "- Consenso entre os modelos treinados:\n  "
                    + "\n  ".join(resumo_comparacao)
                    + "\n\nOrganiza o relatório em 3 secções claras:\n"
                    "1. Resumo da Previsão\n"
                    "2. Análise de Confiança e Consenso entre Modelos\n"
                    "3. Recomendações e Limitações Técnicas"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            mensagens, tokenize=False, add_generation_prompt=True
        )
        saida = self.generator(
            prompt,
            max_new_tokens=self.config.get("max_new_tokens", 512),
            temperature=self.config.get("temperature", 0.3),
            do_sample=True,
            return_full_text=False,
        )
        return saida[0]["generated_text"].strip()


class SistemaMultiagenteHF:
    """Orquestrador completo do sistema multiagente utilizando Hugging Face Transformers."""

    def __init__(self, gestor: GestorModelos | None = None):
        self.gestor = gestor or GestorModelos()
        self.analista = AgenteAnalistaPreditivo(gestor=self.gestor)
        self.relator = AgenteRelatorGenerativo()

    def executar(self, dados_entrada: dict | pd.DataFrame) -> dict:
        analise = self.analista.analisar(dados_entrada, incluir_comparacao=True)
        relatorio = self.relator.redigir_relatorio(analise)
        return {
            "analise": analise,
            "relatorio": relatorio,
        }
