"""Backends de LLM. Transformers (Hugging Face) quando disponível; fallback offline determinístico."""
import json
import os

from src.config import MODELS_DIR

with open(MODELS_DIR / "llm_config.json", encoding="utf-8") as _f:
    LLM_CONFIG = json.load(_f)
_LOCAL = MODELS_DIR / "llm" / "Qwen2.5-0.5B-Instruct"
# Prefere a cópia local (sem internet); senão usa o id do Hugging Face.
_DEFAULT = str(_LOCAL) if (_LOCAL / "model.safetensors").exists() else LLM_CONFIG["model_id"]
DEFAULT_MODEL = os.environ.get("LLM_MODEL_ID", _DEFAULT)

SYSTEM_PROMPT = ("És um analista de fraude de um banco. Escreves relatórios curtos em português europeu, "
                 "usando APENAS os números do JSON fornecido. Não inventes valores. Termina com 2 recomendações.")


class LLMBackend:
    nome = "base"

    def gerar(self, prompt: str, system: str = SYSTEM_PROMPT, max_new_tokens: int = 350) -> str:
        raise NotImplementedError


class TransformersBackend(LLMBackend):
    """LLM local via Hugging Face Transformers (necessita de internet/cache do modelo e de torch)."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # import tardio
        self.nome = f"transformers:{model_id}"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def gerar(self, prompt, system=SYSTEM_PROMPT, max_new_tokens=350):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        # return_dict=True: compatível com transformers 4.x e 5.x (devolve input_ids + attention_mask)
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"]
        out = self.model.generate(input_ids=ids, attention_mask=enc.get("attention_mask"),
                                  max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


class TemplateBackend(LLMBackend):
    """Modo offline: gera o texto a partir dos factos com regras (sem LLM). Mantém o sistema testável."""
    nome = "template-offline"

    def gerar(self, prompt, system=SYSTEM_PROMPT, max_new_tokens=350):
        try:
            f = json.loads(prompt[prompt.index("{"):])
        except Exception:
            return "Sem dados suficientes para gerar o relatório."
        n, a = f["n_transacoes"], f["n_alertas"]
        txt = (f"Foram analisadas {n} transações e geraram-se {a} alertas de fraude ({f['taxa_alertas_%']}%), "
               f"com {f['valor_em_risco']} EUR em risco num total de {f['valor_total']} EUR. ")
        if a:
            txt += f"Os alertas concentram-se por volta das {f['hora_pico_alertas']}h; gravidade: {f['alertas_por_gravidade']}. "
        sim = f.get("simulacao")
        if sim:
            txt += (f"Na simulação, detetaram-se {sim['detetadas']} de {sim['fraudes_reais']} fraudes reais "
                    f"({sim['falsos_positivos']} falsos positivos). ")
        rec = ["Rever manualmente os alertas CRÍTICOS e ALTOS antes de bloquear cartões.",
               "Monitorizar a taxa de alertas: um aumento súbito pode indicar um ataque ou deriva dos dados."]
        return txt + "Recomendações: 1) " + rec[0] + " 2) " + rec[1]


def get_backend(preferido: str = "auto") -> LLMBackend:
    """auto: tenta Transformers; se falhar (sem rede/torch), usa o modo offline."""
    if preferido in ("auto", "transformers"):
        try:
            return TransformersBackend()
        except Exception as e:  # noqa: BLE001
            if preferido == "transformers":
                raise
            print(f"[LLM] Transformers indisponível ({type(e).__name__}); a usar modo offline.")
    return TemplateBackend()
