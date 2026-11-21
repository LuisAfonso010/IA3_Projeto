# CS2 - Sistema multiagente de previsão

Previsão do vencedor de mapas de CS2 com as **12 features** do dataset (`elo_*`, `win_rate_*`,
`forma_recente_*`, `h2h_*`), construída sobre o melhor modelo do notebook de modelação.

```
main.py                     CLI (--modo hf | crewai | todos | --verificar)
src/gestor_modelos.py       GestorModelos: carrega os .joblib, identifica o melhor, alinha as features
src/predict.py              núcleo numérico: previsão, comité, explicação, série BO1/BO3/BO5, confiança
src/agentes_hf.py           AgenteAnalistaPreditivo + AgenteRelatorGenerativo (Qwen2.5-1.5B-Instruct local)
src/crewai_multiagent.py    CrewAI: classificadores expostos como @tool
```

## Preparação
1. Corre o notebook de modelação (secção 7). Fica uma pasta com `modelo_*.joblib` e `resultados_todos_modelos.csv`.
2. `pip install -r requirements.txt` (usa a mesma versão de scikit-learn do treino).
3. Modo `hf` com Qwen: guarda o modelo localmente (uma vez):
   `python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='modelos_hf/Qwen2.5-1.5B-Instruct')"`
4. Modo `crewai`: precisa de um LLM com chamada de ferramentas. Padrão: Ollama local (`ollama pull qwen2.5:1.5b`);
   para outro fornecedor usa `--llm <string LiteLLM>` e a respetiva chave de API.

## Utilização
```
python main.py --verificar cs2_matches_test.csv --pasta-modelos resultados
python main.py --modo hf     --jogos cs2_matches_test.csv --treino cs2_matches_train.csv --pasta-modelos resultados --formato bo3
python main.py --modo hf     --sem-llm ...        # texto determinístico, sem Qwen
python main.py --modo crewai --llm ollama/qwen2.5:1.5b ...
python main.py --modo todos  ...
```
Os resultados ficam em `saida/` (`previsoes.csv`, `relatorio_hf.txt`, `relatorio_crewai.txt`).
Por defeito analisa 5 jogos (`--max-jogos`), porque os modelos de linguagem são lentos.

Colab: `from main import main; main(["--modo","hf","--jogos","...","--treino","..."])`

## Notas
- Os números vêm sempre dos classificadores. O Qwen só escreve texto; se inventar números, o texto é rejeitado
  e usa-se o texto determinístico. No CrewAI avisa-se se o agente não usou ferramentas ou citou números que nenhuma devolveu.
- BO3/BO5 assumem mapas independentes com a mesma probabilidade (o modelo não conhece o mapa nem o veto).
- Os limiares de confiança (`LIMIARES` em `src/predict.py`) são escolhas de partida, não calibradas por si.
