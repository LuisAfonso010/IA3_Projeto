"""
US03 - Multiagente com Hugging Face.

AgenteAnalistaPreditivo  -> corre o gestor de modelos e produz as fichas numéricas (sem LLM).
AgenteRelatorGenerativo  -> o Qwen2.5-1.5B-Instruct (guardado localmente) escreve o relatório
                            técnico em português a partir dessas fichas.

Um modelo de 1.5B inventa números com facilidade, por isso o relatório gerado é verificado:
se contiver números que não estão nos factos, é rejeitado e usa-se o texto determinístico.
"""
from __future__ import annotations

import os
from pathlib import Path

from .predict import (analisar_jogos, factos_ficha, fichas_para_dataframe,
                      numeros_nao_suportados, texto_ficha)

QWEN_DIR_PADRAO = "modelos_hf/Qwen2.5-1.5B-Instruct"

SISTEMA = (
    "És um analista de dados de CS2 (Counter-Strike 2). Escreves relatórios técnicos curtos em "
    "português europeu. Usa APENAS os factos fornecidos: não inventes números, equipas, mapas, "
    "jogadores nem resultados passados. Copia as percentagens exatamente como estão nos factos."
)
INSTRUCAO = (
    "Escreve o relatório em 3 parágrafos curtos (máximo 120 palavras no total):\n"
    "1) a previsão e a probabilidade; 2) os fatores que mais pesaram; "
    "3) a fiabilidade (confiança, comité) e uma limitação."
)


class AgenteAnalistaPreditivo:
    nome = "Analista Preditivo"

    def __init__(self, gestor, formato: str = "bo1"):
        self.gestor, self.formato = gestor, formato

    def analisar(self, jogos):
        return analisar_jogos(self.gestor, jogos, self.formato)


class AgenteRelatorGenerativo:
    nome = "Relator Generativo"

    def __init__(self, qwen_dir: str | None = None, usar_llm: bool = True, max_new_tokens: int = 320):
        self.qwen_dir = qwen_dir or os.environ.get("CS2_QWEN_DIR") or QWEN_DIR_PADRAO
        self.usar_llm, self.max_new_tokens = usar_llm, max_new_tokens
        self._pipe = None

    def preparar(self) -> None:
        """Carrega o Qwen local. Levanta RuntimeError com instruções se não for possível."""
        if not self.usar_llm or self._pipe is not None:
            return
        pasta = Path(self.qwen_dir)
        if not (pasta / "config.json").exists():
            raise RuntimeError(
                f"Não encontrei o Qwen2.5-1.5B-Instruct em '{pasta}'.\n"
                "Guarda-o localmente com:\n"
                "  python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='{self.qwen_dir}')\"\n"
                "ou indica outra pasta com --qwen-dir (ou usa --sem-llm)."
            )
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError("Falta instalar: pip install transformers torch accelerate") from e
        kw, device = {}, -1
        try:
            import torch
            if torch.cuda.is_available():
                device, kw["torch_dtype"] = 0, "auto"
        except ImportError:
            pass
        self._pipe = pipeline("text-generation", model=str(pasta), tokenizer=str(pasta), device=device, **kw)

    def escrever(self, ficha) -> dict:
        factos = factos_ficha(ficha)
        if not self.usar_llm or self._pipe is None:
            return {"texto": texto_ficha(ficha), "origem": "template"}
        msgs = [{"role": "system", "content": SISTEMA},
                {"role": "user", "content": f"Factos:\n{factos}\n\n{INSTRUCAO}"}]
        saida = self._pipe(msgs, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False)
        gerado = saida[0]["generated_text"]
        if isinstance(gerado, list):  # formato de chat: lista de mensagens
            gerado = gerado[-1]["content"]
        gerado = gerado.strip()
        mal = numeros_nao_suportados(gerado, factos)
        if not gerado or mal:
            return {"texto": texto_ficha(ficha), "origem": "template (texto do Qwen rejeitado)",
                    "rejeitado": gerado, "numeros_invalidos": mal}
        return {"texto": gerado, "origem": "qwen"}


def executar_hf(gestor, jogos, formato="bo1", qwen_dir=None, sem_llm=False, verbose=False):
    """Corre os dois agentes. Devolve dict com fichas, relatórios, avisos e DataFrame de previsões."""
    analista = AgenteAnalistaPreditivo(gestor, formato)
    relator = AgenteRelatorGenerativo(qwen_dir, usar_llm=not sem_llm)

    fichas, avisos = analista.analisar(jogos)
    if verbose:
        print(f"[{analista.nome}] {len(fichas)} jogos analisados.")
    try:
        relator.preparar()
        if verbose and relator.usar_llm:
            print(f"[{relator.nome}] Qwen carregado de '{relator.qwen_dir}'.")
    except RuntimeError as e:
        avisos.append(f"Sem modelo de linguagem ({str(e).splitlines()[0]}) — usei o texto determinístico.")
        print(f"AVISO: {e}\n       A usar o texto determinístico.\n")
        relator.usar_llm = False

    relatorios = []
    for f in fichas:
        r = relator.escrever(f)
        r["indice"] = f.indice
        r["titulo"] = f"#{f.indice + 1} {f.equipa1} vs {f.equipa2}"
        relatorios.append(r)
    return {"fichas": fichas, "relatorios": relatorios, "avisos": avisos, "usou_llm": relator.usar_llm,
            "previsoes": fichas_para_dataframe(fichas)}
