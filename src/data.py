import pandas as pd

def getDadosGitHub():       
    arquivos_config = {
        2010: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2010.CSV",
        2011: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2011.CSV",
        2012: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2012.CSV",
        2013: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2013.CSV",
        2014: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2014.CSV",
        2015: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2015.CSV",
        2016: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2016.CSV",
        2017: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2017.CSV",
        2018: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2018.CSV",
        2019: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2019.CSV",
        2020: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2020.CSV",
        2021: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2021.CSV",
        2022: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2022.CSV",
        2023: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2023.CSV",
        2024: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2024.CSV",        
        2025: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2025.CSV",
        2026: "https://github.com/jcbdoliveira/ProfessorMauricio/raw/refs/heads/main/EXPORTA_2026.CSV"
    }
    
    mesesNomes = ["Janeiro","Fevereiro","Marco","Abril","Maio","Junho",
            "Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
    
    mesesNumeros = {
                "Janeiro":1,"Fevereiro":2,"Marco":3,"Abril":4,"Maio":5,"Junho":6,
                "Julho":7,"Agosto":8,"Setembro":9,"Outubro":10,"Novembro":11,"Dezembro":12
    }

    contasDesejadas = ["3110000","3140000","4000000","5000000","6111000"]

    destino = f"data/raw/Dados.csv" 

    dfs = []
    for ano, url in arquivos_config.items():    
        print(f"Baixando arquivo de dados brutos para o ano {ano}...")
        df = pd.read_csv(url, sep=";", decimal=",")
        df["Ano"] = ano# adiciona coluna com ano do arquivo                
        
        # Filtra pelos códigos desejados
        df = df[df["Conta"].astype(str).isin(contasDesejadas)]
        dfs.append(df)

    # Junta todos os DataFrames
    dfTotal = pd.concat(dfs, ignore_index=True)
            
    dfAuxiliar = dfTotal.melt(
                        id_vars=["Grupo","Codigo","Conta","Descricao",
                        "Saldo Anterior","Total de Debitos","Total de Creditos","Saldo Atual","Perc", "Ano"],
                        value_vars=mesesNomes,
                        var_name="Mes",
                        value_name="Valor"
    )

    dfAuxiliar["Mes_Num"] = dfAuxiliar["Mes"].map(mesesNumeros)
    dfAuxiliar.to_csv(destino, sep=";", index=False)

    return destino

def agrupar_valores_por_tipo(dfWork):
    
    registro_temporal = {}

    for _, row in dfWork.iterrows():
        ano = int(row["Ano"])
        mes_num = int(row["Mes_Num"])
        mes_nome = str(row["Mes"])
        conta = int(row["Conta"])
        valor = row["Valor_Limpo"]
        
        chave = (ano, mes_num)
        if chave not in registro_temporal:
            registro_temporal[chave] = {
                "Ano": ano,
                "Mes_Num": mes_num,
                "Mes_Nome": mes_nome,
                "Receita_Bruta": 0.0,
                "Custos_Variaveis": 0.0,
                "Despesas_Operacionais": 0.0
            }
        
        # Atribui o valor na coluna correta conforme o código contábil
        if conta == 3110000:
            registro_temporal[chave]["Receita_Bruta"] += valor        
        elif conta == 5000000:
            registro_temporal[chave]["Receita_Bruta"] += 0
        elif conta == 6111000:
            registro_temporal[chave]["Receita_Bruta"] += 0              
        elif conta == 3140000:
            registro_temporal[chave]["Custos_Variaveis"] += valor
        elif conta == 4000000:
            registro_temporal[chave]["Despesas_Operacionais"] += valor

    # Converte o dicionário agrupado no DataFrame de treinamento final
    dfResult = pd.DataFrame(registro_temporal.values())
    dfResult = dfResult.sort_values(by=["Ano", "Mes_Num"]).reset_index(drop=True)

    #Formata os valores financeiros para duas casas decimais
    colunas_financeiras = ["Receita_Bruta", "Custos_Variaveis", "Despesas_Operacionais"]
    dfResult[colunas_financeiras] = dfResult[colunas_financeiras].round(2)

    # Adiciona uma coluna de índice temporal para facilitar a análise e o modelo de previsão
    dfResult["Indice_Novo"] = range(1, len(dfResult) + 1)
    
    destino = f"data/raw/Database.csv" 
    dfResult.to_csv(destino, sep=";", index=False)
    return destino

def tratar_valores(val):
    if pd.isna(val) or str(val).strip() == "" or str(val).strip() == ",00":
        return 0.0 # Retornar 0.0 para valores nulos ou vazios

    val_str = str(val).strip()        
    val_str = val_str.replace(",", ".")

    try:
        #Converte para número e o abs() transforma o negativo contábil em positivo
        return abs(float(val_str))
    except (ValueError, TypeError):
        return 0.0

def remover_colunas_inuteis(dfWork):
    # 1. Faz uma cópia de segurança para não afetar o dado original
    df = dfWork.copy()    
    # 2. Lista de colunas para remover (ruídos contábeis)
    colunas_para_remover = [
        "Grupo", "Codigo", "Descricao", "Saldo Anterior", 
        "Total de Debitos", "Total de Creditos", "Saldo Atual", "Perc", "Valor"
    ]    
    # Remove as colunas inúteis (ignora caso alguma já não exista)
    return df.drop(columns=colunas_para_remover, errors="ignore")

def getData():    
    print(f"01. Obter dados brutos para treinamento...")        
    dfBruto = pd.read_csv(getDadosGitHub(), sep=";", decimal=",")

    print(f"02. Tratando dados brutos para treinamento...")    
    dfTratado = dfBruto.copy()
    dfTratado["Valor_Limpo"] = dfBruto["Valor"].apply(tratar_valores)

    print(f"03. Removendo colunas inuteis...")  
    dfLimpo = remover_colunas_inuteis(dfTratado)

    print(f"04. Agrupando dados por mes, ano e conta...")  
    retorno = agrupar_valores_por_tipo(dfTratado)
    return retorno




