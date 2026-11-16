"""
EDA / Data Understanding - Dataset de Deteção de Fraude em Cartões de Crédito
Projeto IA III - ISTEC (CRISP-ML: Business and Data Understanding)

Funções auxiliares usadas pelo notebook Data_Understanding.ipynb e utilizáveis
também como script (python -m src.eda_creditcard):
- Contrato de dados (validação de formato, nomes e tipos)
- Estrutura e qualidade (nulos, duplicados)
- Estatística descritiva + Skewness / Kurtosis
- Variável alvo (desequilíbrio de classes)
- Correlação Pearson vs Spearman
- Distribuição das features por classe
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_CONTRACT, DOCS_DIR, RANDOM_STATE, RAW_FILE, TARGET

sns.set_theme(style="whitegrid")


def carregar_dados(caminho=RAW_FILE) -> pd.DataFrame:
    """Carrega o CSV bruto e limpa espaços nos nomes das colunas."""
    df = pd.read_csv(caminho)
    df.columns = df.columns.str.strip()
    print(f"[SUCESSO] {df.shape[0]} registos | {df.shape[1]} colunas carregadas de '{caminho}'")
    return df


def validar_contrato_dados(df: pd.DataFrame) -> bool:
    """Verifica se o dataset cumpre o contrato de dados (colunas e tipos esperados)."""
    ok = True
    for col, tipo in DATA_CONTRACT["colunas"].items():
        if col not in df.columns:
            print(f"[FALHA] coluna em falta: {col}")
            ok = False
        elif not pd.api.types.is_numeric_dtype(df[col]):
            print(f"[FALHA] coluna {col} deveria ser {tipo}, é {df[col].dtype}")
            ok = False
    extra = set(df.columns) - set(DATA_CONTRACT["colunas"])
    if extra:
        print(f"[AVISO] colunas fora do contrato: {sorted(extra)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        print("[FALHA] Class tem valores fora de {0,1}")
        ok = False
    print("Contrato de dados:", "CUMPRIDO" if ok else "VIOLADO")
    return ok


def resumo_estrutura(df: pd.DataFrame, nome: str = "Dataset") -> None:
    print("=" * 70)
    print(f"1. RESUMO DA ESTRUTURA DOS DADOS: {nome}")
    print("=" * 70)
    print(df.head())
    print("\n--- Contagem dos Tipos de Dados ---")
    print(df.dtypes.value_counts())
    print("\n--- Detalhe das Colunas e Tipos (.info()) ---")
    df.info()


def analisar_qualidade_dados(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("2. ANÁLISE DE QUALIDADE DOS DADOS (NULOS E DUPLICADOS)")
    print("=" * 70)
    duplicados = int(df.duplicated().sum())
    print(f"Linhas duplicadas encontradas: {duplicados} ({duplicados / len(df) * 100:.2f}%)")
    nulos = df.isnull().sum()
    relatorio = pd.DataFrame({"Valores_Nulos": nulos, "Percentagem_%": nulos / len(df) * 100})
    relatorio = relatorio.query("Valores_Nulos > 0")
    if relatorio.empty:
        print("Não foram encontrados valores nulos.")
    else:
        print(relatorio.sort_values("Valores_Nulos", ascending=False))
    return relatorio


def estatisticas_descritivas_completas(df: pd.DataFrame) -> pd.DataFrame:
    """Média, desvio, min, mediana, max + Skewness e Kurtosis (excesso)."""
    print("=" * 70)
    print("3. ESTATÍSTICAS DESCRITIVAS E EDA AVANÇADA")
    print("=" * 70)
    num = df.select_dtypes(include=[np.number])
    resumo = num.describe().T[["mean", "std", "min", "50%", "max"]]
    resumo["skewness"] = num.skew()
    resumo["kurtosis"] = num.kurt()
    print(resumo.round(3))
    return resumo


def analisar_variavel_alvo(df: pd.DataFrame, guardar: bool = True) -> pd.Series:
    print("=" * 70)
    print(f"4. ANÁLISE DA VARIÁVEL ALVO: '{TARGET}' (0 = legítima, 1 = fraude)")
    print("=" * 70)
    contagem = df[TARGET].value_counts().sort_index()
    pct = (contagem / len(df) * 100).round(4)
    print(pd.DataFrame({"contagem": contagem, "%": pct}))
    print(f"Rácio de desequilíbrio: 1 fraude para cada {contagem[0] / contagem[1]:.0f} legítimas")

    plt.figure(figsize=(5, 4))
    contagem.plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Distribuição da classe (0 = legítima, 1 = fraude)")
    plt.xlabel("Class")
    plt.ylabel("nº de transações")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if guardar:
        plt.savefig(DOCS_DIR / "class_balance.png", dpi=120)
    return contagem


def analisar_correlacoes(df: pd.DataFrame, guardar: bool = True) -> pd.DataFrame:
    """Pearson (linear) vs Spearman (monotónica) entre as features e a classe."""
    print("=" * 70)
    print("5. ANÁLISE MULTIVARIADA: CORRELAÇÃO (PEARSON VS SPEARMAN)")
    print("=" * 70)
    amostra = df.sample(min(len(df), 60_000), random_state=RANDOM_STATE)  # Spearman é pesado
    pearson = df.corr(method="pearson")
    spearman = amostra.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(pearson, cmap="coolwarm", center=0, ax=axes[0], cbar=False)
    axes[0].set_title("Correlação de Pearson (Linear)")
    sns.heatmap(spearman, cmap="coolwarm", center=0, ax=axes[1], cbar=False)
    axes[1].set_title("Correlação de Spearman (Monotónica, amostra 60k)")
    plt.tight_layout()
    if guardar:
        plt.savefig(DOCS_DIR / "correlation_matrix.png", dpi=110)

    com_alvo = pd.DataFrame({"pearson": pearson[TARGET], "spearman": spearman[TARGET]}).drop(TARGET)
    com_alvo["abs_pearson"] = com_alvo["pearson"].abs()
    top = com_alvo.sort_values("abs_pearson", ascending=False).head(10)
    print("\nTop 10 features mais correlacionadas com a classe (Pearson):")
    print(top.round(3))
    return com_alvo


def distribuicoes_por_classe(df: pd.DataFrame, n_features: int = 8, guardar: bool = True) -> list:
    """KDE das features mais discriminativas, por classe."""
    corr = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    top = list(corr.head(n_features).index)
    amostra = df.sample(min(len(df), 60_000), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    for ax, col in zip(axes.flatten(), top):
        sns.kdeplot(data=amostra, x=col, hue=TARGET, fill=True, ax=ax, common_norm=False, warn_singular=False)
        ax.set_title(col, fontsize=10)
    plt.tight_layout()
    if guardar:
        plt.savefig(DOCS_DIR / "feature_distributions.png", dpi=120)
    return top


def analise_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Fraudes por hora do dia (componente temporal do contrato de dados)."""
    d = df.assign(Hour=(df["Time"] // 3600 % 24).astype(int))
    por_hora = d.groupby("Hour")[TARGET].agg(["count", "sum", "mean"])
    por_hora.columns = ["transacoes", "fraudes", "taxa_fraude"]
    print(por_hora.round(4))
    return por_hora


if __name__ == "__main__":
    dados = carregar_dados()
    validar_contrato_dados(dados)
    resumo_estrutura(dados)
    analisar_qualidade_dados(dados)
    estatisticas_descritivas_completas(dados)
    analisar_variavel_alvo(dados)
    analisar_correlacoes(dados)
    distribuicoes_por_classe(dados)
    analise_temporal(dados)
    print("\nGráficos guardados em docs/: class_balance.png, correlation_matrix.png, feature_distributions.png")
