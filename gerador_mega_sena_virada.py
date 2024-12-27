import pandas as pd
import numpy as np
import random
import math
from datetime import datetime
import matplotlib.pyplot as plt

# ========================================
#           CONSTANTES
# ========================================
NUM_MIN = 1
NUM_MAX = 60


# ========================================
#   FUNÇÕES PARA ANÁLISE E TRATAMENTO
# ========================================
def carregar_dados(csv_file: str) -> pd.DataFrame:
    """
    Lê o arquivo CSV e faz o pré-processamento dos dados.
    Espera-se que o CSV contenha as colunas:
    [CONCURSO, DATA_DO_SORTEIO, BOLA1, BOLA2, BOLA3, BOLA4, BOLA5, BOLA6, GANHADORES_6_ACERTOS].
    """
    df = pd.read_csv(csv_file, encoding='utf-8', sep=',')

    # Ajuste do tipo de data
    df['DATA_DO_SORTEIO'] = pd.to_datetime(df['DATA_DO_SORTEIO'], format='%d/%m/%Y', errors='coerce')

    # Garante que as bolas sorteadas sejam inteiros
    for col in ['BOLA1', 'BOLA2', 'BOLA3', 'BOLA4', 'BOLA5', 'BOLA6']:
        df[col] = df[col].astype(int)

    # Remove linhas com datas inválidas
    df = df.dropna(subset=['DATA_DO_SORTEIO'])

    # Ordena pelo número do concurso
    df = df.sort_values(by='CONCURSO')
    df.reset_index(drop=True, inplace=True)

    return df


def filtrar_mega_da_virada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra apenas os concursos da Mega da Virada (31/12 a partir de 2008).
    """
    df_virada = df[
        (df['DATA_DO_SORTEIO'].dt.month == 12) &
        (df['DATA_DO_SORTEIO'].dt.day == 31) &
        (df['DATA_DO_SORTEIO'].dt.year >= 2008)
        ]
    df_virada.reset_index(drop=True, inplace=True)
    return df_virada


def calcular_frequencia_numeros(df: pd.DataFrame) -> pd.Series:
    """
    Calcula quantas vezes cada número (1 a 60) apareceu nos sorteios.
    Retorna um Series onde o índice é o número e o valor é a frequência.
    """
    numeros = []
    for i in range(len(df)):
        bolas = df.iloc[i, df.columns.get_loc('BOLA1'):df.columns.get_loc('BOLA6') + 1].values
        numeros.extend(bolas)

    serie_frequencia = pd.Series(numeros).value_counts().sort_index()
    # Garante que todos os números de 1 a 60 apareçam (mesmo que zero)
    serie_frequencia = serie_frequencia.reindex(range(NUM_MIN, NUM_MAX + 1), fill_value=0)

    return serie_frequencia


def calcular_atraso_numeros(df: pd.DataFrame) -> pd.Series:
    """
    Calcula o 'atraso' de cada número (1 a 60) = quantos concursos se passaram
    desde a última aparição do número.
    """
    ultimo_concurso = df['CONCURSO'].max()
    ultimo_concurso_num = {n: -1 for n in range(NUM_MIN, NUM_MAX + 1)}

    for i in range(len(df)):
        concurso = df.iloc[i, df.columns.get_loc('CONCURSO')]
        bolas = df.iloc[i, df.columns.get_loc('BOLA1'):df.columns.get_loc('BOLA6') + 1].values
        for num in bolas:
            ultimo_concurso_num[num] = concurso

    atrasos = {}
    for n in range(NUM_MIN, NUM_MAX + 1):
        if ultimo_concurso_num[n] == -1:
            atraso = ultimo_concurso
        else:
            atraso = ultimo_concurso - ultimo_concurso_num[n]
        atrasos[n] = atraso

    serie_atraso = pd.Series(atrasos).sort_index()
    return serie_atraso


def proporcao_impares_pares(df: pd.DataFrame) -> float:
    """
    Retorna a proporção média de números ímpares no histórico.
    Ex.: se 50% dos números são ímpares, retorna 0.5.
    """
    total_numeros = 0
    total_impares = 0

    for i in range(len(df)):
        bolas = df.iloc[i, df.columns.get_loc('BOLA1'):df.columns.get_loc('BOLA6') + 1].values
        total_numeros += len(bolas)
        total_impares += sum(1 for x in bolas if x % 2 != 0)

    if total_numeros == 0:
        return 0
    return total_impares / total_numeros


def ja_saiu_no_historico(combinacao, df: pd.DataFrame) -> bool:
    """
    Verifica se a combinação já apareceu no histórico (mesmas 6 dezenas).
    Retorna True se já apareceu, False caso contrário.
    """
    c_set = set(combinacao)
    for i in range(len(df)):
        bolas = df.iloc[i, df.columns.get_loc('BOLA1'):df.columns.get_loc('BOLA6') + 1].values
        if c_set == set(bolas):
            return True
    return False


def tem_mais_de_3_consecutivos(combinacao) -> bool:
    """
    Verifica se a combinação possui uma sequência de mais de 3 números consecutivos.
    """
    c_ord = sorted(combinacao)
    consecutivos = 1
    for i in range(1, len(c_ord)):
        if c_ord[i] == c_ord[i - 1] + 1:
            consecutivos += 1
            if consecutivos > 3:
                return True
        else:
            consecutivos = 1
    return False


# ========================================
#     MÉTODO PARA GARANTIR PROPORÇÃO
# ========================================
def gerar_combinacao_proporcional(
        qtd_numeros: int,
        foco: str,
        top_freq: list,
        top_atra: list
) -> list:
    """
    Gera uma lista de números (sem outras validações) com base em proporções:
      - foco = 'frequentes':  80% frequentes,  20% atrasados
      - foco = 'atrasados':   80% atrasados,   20% frequentes
      - foco = 'equilibrado': 50% frequentes,  50% atrasados

    Se houver fração, arredondar para cima (math.ceil) favorecendo:
      - foco='frequentes': arredondar para cima na parte frequente
      - foco='atrasados':  arredondar para cima na parte atrasada
      - foco='equilibrado': arredondar para cima na parte frequente

    Exemplo: se qtd_numeros=11 e foco='frequentes', 80% = 8.8 => 9 freq e 2 atras.

    Retorna a lista final, podendo ter duplicados se um mesmo número estiver
    em top_freq e top_atra. Ajustamos logo abaixo (remover duplicados).
    """
    import math

    if foco == 'frequentes':
        n_freq_float = qtd_numeros * 0.8
        n_freq = math.ceil(n_freq_float)
        n_atra = qtd_numeros - n_freq
    elif foco == 'atrasados':
        n_atra_float = qtd_numeros * 0.8
        n_atra = math.ceil(n_atra_float)
        n_freq = qtd_numeros - n_atra
    else:
        # equilibrado => 50% cada, arredondando para cima em frequentes
        n_freq_float = qtd_numeros * 0.5
        n_freq = math.ceil(n_freq_float)
        n_atra = qtd_numeros - n_freq

    # Seleciona aleatoriamente n_freq em top_freq
    if len(top_freq) < n_freq:
        escolhidos_freq = top_freq[:]
    else:
        escolhidos_freq = random.sample(top_freq, n_freq)

    # Seleciona aleatoriamente n_atra em top_atra
    if len(top_atra) < n_atra:
        escolhidos_atra = top_atra[:]
    else:
        escolhidos_atra = random.sample(top_atra, n_atra)

    # Unir e remover duplicados
    comb_set = set(escolhidos_freq + escolhidos_atra)

    # Se removemos duplicados e ficamos com menos que qtd_numeros,
    # podemos preencher com números fora desses "top" (1..60) - comb_set
    while len(comb_set) < qtd_numeros:
        resto = list(set(range(NUM_MIN, NUM_MAX + 1)) - comb_set)
        if not resto:
            break
        comb_set.add(random.choice(resto))

    # Se ainda estiver maior, faz um corte
    if len(comb_set) > qtd_numeros:
        comb_list = random.sample(list(comb_set), qtd_numeros)
    else:
        comb_list = list(comb_set)

    random.shuffle(comb_list)
    return comb_list


# ========================================
#    FUNÇÃO PRINCIPAL DE GERAÇÃO
# ========================================
def gerar_aposta(
        df: pd.DataFrame,
        freq: pd.Series,
        atraso: pd.Series,
        qtd_numeros: int,
        prop_impares: float,
        foco='equilibrado'
) -> list:
    """
    Gera uma aposta final que:
      1) Respeita a proporção de freq/atras de acordo com o foco:
         * frequentes => 80% freq, 20% atras
         * atrasados => 80% atras, 20% freq
         * equilibrado => 50%/50%
      2) Mantém a proporção histórica de ímpares (±20%),
      3) Evita mais de 3 consecutivos,
      4) Evita repetições exatas do histórico.

    Retorna uma lista de 'qtd_numeros' números que atende os critérios.
    """
    # Prepara "top" 30 de cada critério, ajustável
    top_freq = freq.sort_values(ascending=False).index.tolist()[:30]
    top_atra = atraso.sort_values(ascending=False).index.tolist()[:30]

    while True:
        # 1. Gera uma combinação respeitando a proporção exata
        combinacao = gerar_combinacao_proporcional(qtd_numeros, foco, top_freq, top_atra)
        combinacao.sort()

        # 2. Verifica a proporção de ímpares
        impares = sum(1 for x in combinacao if x % 2 != 0)
        prop_aposta_impares = impares / qtd_numeros
        if not (prop_impares * 0.8 <= prop_aposta_impares <= prop_impares * 1.2):
            continue

        # 3. Verifica se já saiu no histórico
        if ja_saiu_no_historico(combinacao, df):
            continue

        # 4. Verifica se tem mais de 3 consecutivos
        if tem_mais_de_3_consecutivos(combinacao):
            continue

        # Se passou em tudo, retorna
        return combinacao


# ========================================
#       FUNÇÕES DE VISUALIZAÇÃO
# ========================================
def plot_frequencia(freq: pd.Series):
    """
    Mostra um gráfico de barras com a frequência de cada número.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(freq.index, freq.values, color='blue')
    plt.title('Frequência dos Números Sorteados')
    plt.xlabel('Números')
    plt.ylabel('Frequência')
    plt.xticks(range(NUM_MIN, NUM_MAX + 1), rotation=90)
    plt.show()


def plot_atraso(atraso: pd.Series):
    """
    Mostra um gráfico de barras com o atraso de cada número.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(atraso.index, atraso.values, color='red')
    plt.title('Atraso dos Números (distância desde a última aparição)')
    plt.xlabel('Números')
    plt.ylabel('Atraso (concursos)')
    plt.xticks(range(NUM_MIN, NUM_MAX + 1), rotation=90)
    plt.show()


# ========================================
#        FUNÇÕES DE EXPORTAÇÃO
# ========================================
def salvar_apostas_csv(apostas: list, nome_arquivo='apostas_geradas.csv'):
    """
    Salva a lista de apostas em um arquivo CSV.
    Cada linha do CSV conterá a combinação gerada.
    """
    df_apostas = pd.DataFrame(apostas)
    df_apostas.to_csv(nome_arquivo, index=False, header=False)
    print(f"Apostas salvas em: {nome_arquivo}")


# ========================================
#              MAIN (Exemplo)
# ========================================
def main():
    """
    Função principal que executa:
      1. Leitura do CSV.
      2. Perguntas ao usuário (Mega da Virada ou Geral, quantidade de números, etc.).
      3. Análises (frequência, atraso, etc.).
      4. Geração de aposta.
      5. Visualização e exportação (opcionais).
    """
    # -------------------------------------------------------------
    # 1. Leitura do arquivo CSV
    # -------------------------------------------------------------
    # csv_file = input("Digite o caminho do arquivo CSV com os dados históricos: ")
    csv_file = '/home/daniel/Downloads/mega_sena.csv'
    df_completo = carregar_dados(csv_file)

    # -------------------------------------------------------------
    # 2. Escolha de Mega da Virada ou Geral
    # -------------------------------------------------------------
    opcao = input("Deseja gerar números com base na (1) Mega da Virada ou (2) Mega-Sena Geral? [1/2]: ")
    if opcao == '1':
        df_filtrado = filtrar_mega_da_virada(df_completo)
        if df_filtrado.empty:
            print("Não foram encontrados sorteios da Mega da Virada no CSV informado.")
            return
    else:
        df_filtrado = df_completo

    # -------------------------------------------------------------
    # 3. Solicitar quantidade de números na aposta
    # -------------------------------------------------------------
    while True:
        qtd_numeros = input("Quantos números deseja na aposta? (entre 6 e 15): ")
        try:
            qtd_numeros = int(qtd_numeros)
            if 6 <= qtd_numeros <= 15:
                break
            else:
                print("Por favor, informe um valor entre 6 e 15.")
        except:
            print("Valor inválido. Tente novamente.")

    # -------------------------------------------------------------
    # 4. Análise Estatística
    # -------------------------------------------------------------
    freq_numeros = calcular_frequencia_numeros(df_filtrado)
    atraso_numeros = calcular_atraso_numeros(df_filtrado)
    prop_impares_hist = proporcao_impares_pares(df_filtrado)

    # Visualizações (opcionais)
    plotar = input("Deseja visualizar gráficos de frequência e atraso? (s/n): ")
    if plotar.lower() == 's':
        plot_frequencia(freq_numeros)
        plot_atraso(atraso_numeros)

    # -------------------------------------------------------------
    # 5. Perguntar o foco: frequentes, atrasados ou equilibrado
    # -------------------------------------------------------------
    foco_opcao = input("Deseja focar em números (1) Frequentes, (2) Atrasados, ou (3) Equilibrado? [1/2/3]: ")
    if foco_opcao == '1':
        foco_aposta = 'frequentes'
    elif foco_opcao == '2':
        foco_aposta = 'atrasados'
    else:
        foco_aposta = 'equilibrado'

    # -------------------------------------------------------------
    # 6. Geração de Aposta (chama gerar_aposta com proporções)
    # -------------------------------------------------------------
    aposta_gerada = gerar_aposta(
        df=df_filtrado,
        freq=freq_numeros,
        atraso=atraso_numeros,
        qtd_numeros=qtd_numeros,
        prop_impares=prop_impares_hist,
        foco=foco_aposta
    )

    print("\nAposta Gerada:", aposta_gerada)

    # -------------------------------------------------------------
    # 7. Exportar resultados (opcional)
    # -------------------------------------------------------------
    exportar = input("Deseja salvar a aposta gerada em um arquivo CSV? (s/n): ")
    if exportar.lower() == 's':
        salvar_apostas_csv([aposta_gerada])  # Passa uma lista de listas

    print("Processo concluído!")


if __name__ == '__main__':
    main()
