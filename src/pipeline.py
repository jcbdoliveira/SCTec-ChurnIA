import os
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import src.data as leitor
import src.report as relatorio

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats.mstats import winsorize

# Configuração de estilo para os gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def executar_FASE_1(caminho_csv):
    print("\nFASE 1: Análise Exploratória de Dados (EDA)")
    
    df = pd.read_csv(caminho_csv, sep=';')
    
    print(f"Dimensões: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    print("\nTipos de dados:")
    print(df.dtypes)
    print("\nResumo estatístico descritivo:")
    print(df.describe())
    
    # ==========================================
    # Removendo as linhas zeradas, mês 07 até 12, para usar na previsão
    df_historico = df[(df['Receita_Bruta'] > 0) | (df['Custos_Variaveis'] > 0)].copy()
    # ==========================================

    # Gráfico 1: Histograma da Receita Bruta
    plt.figure()
    sns.histplot(df_historico['Receita_Bruta'], kde=True, color='skyblue')
    plt.title('Distribuição da Receita bruta')
    plt.xlabel('Receita Bruta (R$)')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig('outputs/figures/01.Distribuicao_receita.png')
    plt.close()
    
    # Gráfico 2: Dispersão entre Custos Variáveis e Receita Bruta sem usra transformação logaritma
    plt.figure()
    sns.scatterplot(data=df_historico, x='Custos_Variaveis', y='Receita_Bruta', hue='Ano', palette='viridis', s=80)
    plt.title('Relação entre Custos e Receita bruta')
    plt.xlabel('Custos Variáveis (R$)')
    plt.ylabel('Receita Bruta (R$)')
    plt.tight_layout()
    plt.savefig('outputs/figures/02.Dispersao_custos_e_receita.png')
    plt.close()
    
    #transformação logaritma para melhorar a visualização dos dados
    df_historico['Receita_Bruta'] = np.log(df_historico['Receita_Bruta'])    
    df_historico['Custos_Variaveis'] = np.log(df_historico['Custos_Variaveis'])
    df_historico['Despesas_Operacionais'] = np.log(df_historico['Despesas_Operacionais'])

    # Gráfico 2: Dispersão entre Custos Variáveis e Receita Bruta com transformação logaritma
    plt.figure()
    sns.scatterplot(data=df_historico, x='Custos_Variaveis', y='Receita_Bruta', hue='Ano', palette='viridis', s=80)
    plt.title('Relação entre Custos e Receita bruta (LOG)')
    plt.xlabel('Custos Variáveis (R$)')
    plt.ylabel('Receita Bruta (R$)')
    plt.tight_layout()
    plt.savefig('outputs/figures/02.Dispersao_custos_e_receitaLOG.png')
    plt.close()
    
    # Gráfico 3: Mapa de calor de correlação de Pearson
    plt.figure()
    colunas_corr = ['Ano', 'Mes_Num', 'Receita_Bruta', 'Custos_Variaveis', 'Despesas_Operacionais']
    corr_matrix = df_historico[colunas_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('outputs/figures/03.Matriz_de_correlacao.png')
    plt.close()

    #Percebi que a receita bruta tem muitos valores baixos e poucos muito altos, o que deixa a distribuição desigual.
    #No gráfico de dispersão, notei que custos e receita crescem juntos, mas não de forma totalmente linear. 
    #A correlação confirma que essas duas variáveis estão quase sempre ligadas e juntas. 
    #Também percebi que, com o passar dos anos, tanto custos quanto receitas aumentam. 
    #Isso mostra que o tempo influencia bastante nos resultados.
    return df

def executar_FASE_2(df):
    print("\nFASE 2: Tratamento e limpeza (DATA PREP)")    
    duplicadas = df.duplicated().sum()
    print(f"Linhas duplicadas encontradas: {duplicadas}")
    if duplicadas > 0:
        df = df.drop_duplicates()
        print("Linhas duplicadas removidas.")
        
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    #No conjunto de dados criei uma coluna Indice_Novo, como ela consigo identificar apartir de qual
    #linha começa os meses que estão com receita zerada. 
    #Esses meses são do mês 07 até 12 de 2026 e serão usados predição.
    df_previsao = df[df['Indice_Novo'] >= 79].copy()
    df_modelagem = df[df['Indice_Novo'] < 79].copy()
    
    plt.figure(figsize=(12, 5))
    df_modelagem[['Receita_Bruta', 'Custos_Variaveis', 'Despesas_Operacionais']].boxplot()
    plt.title('Análise de Outliers (Variáveis Financeiras)')
    plt.ylabel('Valor (R$)')
    plt.tight_layout()
    plt.savefig('outputs/figures/04.Boxplot_com_outliers.png')
    plt.close()

    df_modelagem['Receita_Bruta'] = winsorize(df_modelagem['Receita_Bruta'], limits=[0.05, 0.05])
    df_modelagem['Custos_Variaveis'] = winsorize(df_modelagem['Custos_Variaveis'], limits=[0.05, 0.05])
    df_modelagem['Despesas_Operacionais'] = winsorize(df_modelagem['Despesas_Operacionais'], limits=[0.05, 0.05])

    plt.figure(figsize=(12, 5))
    df_modelagem[['Receita_Bruta', 'Custos_Variaveis', 'Despesas_Operacionais']].boxplot()
    plt.title('Análise de Outliers (Variáveis Financeiras)')
    plt.ylabel('Valor (R$)')
    plt.tight_layout()
    plt.savefig('outputs/figures/04.Boxplot_sem_outliers.png')
    plt.close()

    print("\nUsei a técnica de winsorization para reduzir o impacto dos valores extremos nas variáveis financeiras.")
    print("Essa escolha foi feita porque havia outliers, que na minha opinião, poderiam distorcer as análises e ")
    print("comprometer a precisão dos modelos.")

    # Salva dados limpos intermediários
    df_modelagem.to_csv("data/processed/Database_limpo.csv", index=False, sep=';')    
    return df_modelagem, df_previsao

def executar_FASE_3(df_modelagem, df_previsao, margem_minima):    
    print("FASE 3: Feature engineering (Coluna calculada: churn)") 
    # Coluna custos + despesas   
    df_modelagem['Total_Custos_Despesas'] = df_modelagem['Custos_Variaveis'] + df_modelagem['Despesas_Operacionais']
    df_previsao['Total_Custos_Despesas'] = df_previsao['Custos_Variaveis'] + df_previsao['Despesas_Operacionais']
    
    # Coluna margem operacional: receita - (custso + despesas)
    df_modelagem['Margem_Operacional'] = (df_modelagem['Receita_Bruta'] - df_modelagem['Total_Custos_Despesas']) / df_modelagem['Receita_Bruta']
    
    # Criação da variável alvo 'churn' (Alerta financeiro de margem estreita)
    # 1 se (Custos + Despesas) > Receita * margem minima, senão 0
    df_modelagem['churn'] = (df_modelagem['Total_Custos_Despesas'] > (df_modelagem['Receita_Bruta'] * margem_minima)).astype(int)
    
    # Exibe a distribuição das classes de Churn
    churn_counts = df_modelagem['churn'].value_counts()
    print(f"Distribuição da classe Alvo 'churn' (com risco de {margem_minima*100}% da receita):")
    for val, count in churn_counts.items():
        porc = (count / len(df_modelagem)) * 100
        print(f"  Classe {val} (Churn = {'Sim' if val==1 else 'Não'}): {count} meses ({porc:.2f}%)")
        
    # Gráfico Comparativo: Receitas vs Custos+Despesas
    plt.figure()
    plt.bar(df_modelagem['Ano'] - 0.2, df_modelagem['Receita_Bruta'], width=0.4, label='Receita Bruta', color='green')
    plt.bar(df_modelagem['Ano'] + 0.2, df_modelagem['Total_Custos_Despesas'], width=0.4, label='Custos + Despesas', color='red')
    plt.title('Comparativo de Receitas vs Gastos Operacionais ao Longo do Tempo')
    plt.xlabel('Índice Mensal (Indice_Novo)')
    plt.ylabel('Valor (R$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/05.Receitas_vs_gastos.png')
    plt.close()
    
    # Salva o dataset final preparado para a modelagem
    df_modelagem.to_csv("data/final/Database_final.csv", index=False, sep=';')    
    return df_modelagem, df_previsao

def executar_FASE_4(df_modelagem, test_size, random_state):
    print("\nFASE 4: Preparação para modelagem")
    
    # Seleção de Features relevantes para a previsão de Churn
    features = ['Ano', 'Mes_Num', 'Indice_Novo', 'Receita_Bruta', 'Custos_Variaveis', 'Despesas_Operacionais']
    target = 'churn'
    
    X = df_modelagem[features]
    y = df_modelagem[target]
    
    # Divisão Amostral (Split) em Treino e Teste
    #split_index = int(len(X) * (1 - test_size))
    # Divide manualmente
    #X_train, X_test = X[:split_index], X[split_index:]
    #y_train, y_test = y[:split_index], y[split_index:]

    # Divisão Amostral (Split) em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True
    )
    
    print(f"Dimensões dados treino: {X_train.shape[0]} linhas e {X_train.shape[1]} colunas.")    
    print(f"Dimensões dados teste: {X_test.shape[0]} linhas e {X_test.shape[1]} colunas.")
    
    # Escalonamento Seguro (StandardScaler)  
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Escalonamento (StandardScaler) executado com sucesso.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

def executar_FASE_5(X_train, X_test, y_train, y_test, features):
    print("\nFASE 5: Modelagem, validação e diagnóstico de Overfitting")
    
    # 1. Treinamento da Regressão Linear
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Predições contínuas (probabilidade/grau de risco)
    y_pred_train_continuous = model_lr.predict(X_train)
    y_pred_test_continuous = model_lr.predict(X_test)
    
    # Convertendo para classes discretas (0 ou 1) usando padrão de 0.5
    y_pred_train_class = (y_pred_train_continuous >= 0.5).astype(int)
    y_pred_test_class = (y_pred_test_continuous >= 0.5).astype(int)
    
    # Coeficientes do Modelo de Regressão Linear
    print("\nCoeficientes obtidos pela Regressão Linear:")
    for feat, coef in zip(features, model_lr.coef_):
        print(f"  {feat}: {coef:.6f}")
    print(f"  Intercept (Viés): {model_lr.intercept_:.6f}")
    
    # 2. Treinamento da Regressão Logística
    model_log = LogisticRegression(random_state=42)
    model_log.fit(X_train, y_train)
    
    y_pred_test_log = model_log.predict(X_test)
    y_prob_test_log = model_log.predict_proba(X_test)[:, 1]
    
    print("\nModelos de Regressão Linear e Regressão Logística treinados com sucesso.")
    return model_lr, model_log, y_pred_test_continuous, y_pred_test_class, y_pred_test_log, y_prob_test_log, y_pred_train_continuous

def executar_FASE_6(model_lr, model_log, X_test, y_test, y_pred_cont, y_pred_class, 
                    y_pred_log, y_prob_log, scaler, features, X_train_scaled, 
                    y_train, y_pred_train_continuous, df_previsao):    
    print("\nFASE 6: Avaliação, interpretação e versionamento do modelo")    
    
    # Teste
    mae_test = mean_absolute_error(y_test, y_pred_cont)
    mse_test = mean_squared_error(y_test, y_pred_cont)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_cont)

    # Treino
    mae_train = mean_absolute_error(y_train, y_pred_train_continuous)
    mse_train = mean_squared_error(y_train, y_pred_train_continuous)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train_continuous)

    print("\nMétricas de Regressão Linear:")
    print("Teste:")
    print(f"  MAE: {mae_test:.4f}")
    print(f"  MSE: {mse_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  R²: {r2_test:.4f}")

    print("Treino:")
    print(f"  MAE: {mae_train:.4f}")
    print(f"  MSE: {mse_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")
    print(f"  R²: {r2_train:.4f}")

    try:
        acc = accuracy_score(y_test, y_pred_class)
        prec = precision_score(y_test, y_pred_class, zero_division=0)
        rec = recall_score(y_test, y_pred_class, zero_division=0)        
    except Exception as e:
        acc, prec, rec = 0, 0, 0
        print(f"Erro ao calcular métricas de classificação: {e}")
        
    print("\nMétricas de Classificação (Regressão Linear):")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  Precisão: {prec:.4f}")
    print(f"  Recall (Sensibilidade): {rec:.4f}")   

    acc_log = accuracy_score(y_test, y_pred_log)
    prec_log = precision_score(y_test, y_pred_log, zero_division=0)
    rec_log = recall_score(y_test, y_pred_log, zero_division=0)
    
    print("\nMétricas de Classificação (Regressão Logística):")
    print(f"  Acurácia: {acc_log:.4f}")
    print(f"  Precisão: {prec_log:.4f}")
    print(f"  Recall (Sensibilidade): {rec_log:.4f}")

    print("\nModelo campeão escolhido: >>>Regressão Linear<<<")

    df_plot = pd.DataFrame({
        "y_real": y_test,
        "y_pred_cont": y_pred_cont
    })

    plt.figure(figsize=(8,6))
    jitter = np.random.normal(0, 0.02, size=len(df_plot))
    plt.scatter(df_plot['y_real'] + jitter, 
                df_plot['y_pred_cont'], 
                color='blue', alpha=0.6, label='Previsto')
    plt.title('Dispersão: Valores Reais vs Previstos (Regressão Linear)')
    plt.xlabel('Valores Reais (0 ou 1)')
    plt.ylabel('Probabilidade Prevista')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/06.Dispersao_previstos.png')
    plt.close()

    residuos = y_test - y_pred_cont
    plt.figure(figsize=(8,6))
    sns.histplot(residuos, bins=15, kde=True, color='purple', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--', label='Zero')
    plt.title('Distribuição dos Resíduos (Erro = Real - Previsto)')
    plt.xlabel('Resíduo')
    plt.ylabel('Frequência')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/07.Distribuicao_residuos.png')
    plt.close()

    # Salvar modelos em models/v1/
    with open('models/v1/modelo_regressao_v1.pkl', 'wb') as f:
        pickle.dump(model_lr, f)
        
    with open('models/v1/modelo_logistica_v1.pkl', 'wb') as f:
        pickle.dump(model_log, f)
        
    with open('models/v1/escalonador_v1.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("\nModelos salvos com sucesso na pasta 'models/v1/'.")
    
    # Salvar métricas em formato JSON
    metricas_json = {
        "data_treinamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_utilizadas": features,
        "modelo_base_regressao_linear": {
            "MAE": float(mae_train),
            "MSE": float(mse_train),
            "RMSE": float(rmse_train),
            "R2": float(r2_train),
            "Acuracia": float(acc),
            "Precisao": float(prec),
            "Recall": float(rec)
        },
        "modelo_diferencial_regressao_logistica": {
            "Acuracia": float(acc_log),
            "Precisao": float(prec_log),
            "Recall": float(rec_log),
        }
    }
    
    with open('models/v1/metricas_v1.json', 'write' if False else 'w') as f:
        json.dump(metricas_json, f, indent=4)

    print("\n=== VEREDITO DE NEGÓCIOS ===")
    print("O modelo de Regressão Linear serve como um bom modelo para calcular o Churn.")
    print(f"Com um erro absoluto médio (MAE) de {mae_train:.4f} na probabilidade de Churn,")
    print("o modelo fornece um escore de probabilidade para tomada de decisão.")
    print("Dessa forma, os gestores financeiros da empresa podem usar este modelo ML")
    print("para tomar decisões proativas para evitar a erosão de margem, Churn.")
    
    return metricas_json

def prever_meses_futuros(model_lr, df_previsao, scaler, features, tipo, cenario='base'):
    df_projeção = df_previsao.copy()
    
    # Valores médios históricos
    receita_media = 350000.0
    custos_medios = 180000.0
    despesas_medias = 36000.0
    
    # Ajustes por cenário
    if cenario == 'cenario1':  # aumento de receita e custos 10%
        cenario = 'aumento de receita e custos 10%'
        receita_media *= 1.10       
        custos_medios *= 1.10 
    elif cenario == 'cenario2':  # aumento somente de custos 20%
        cenario = 'aumento somente de custos 20%'
        custos_medios *= 1.20
    elif cenario == 'cenario3':  # aumento das despesas 50%
        cenario = 'aumento das despesas 50%'
        despesas_medias *= 1.50
    
    # Meses de maior aperto (sazonalidade)
    for idx, row in df_projeção.iterrows():
        mes = row['Mes_Num']
        multiplicador_mes = 1.0
        if mes in [11, 12]:
            multiplicador_mes = 1.25
        elif mes in [8, 9]:
            multiplicador_mes = 0.85
            
        df_projeção.at[idx, 'Receita_Bruta'] = receita_media * multiplicador_mes
        df_projeção.at[idx, 'Custos_Variaveis'] = custos_medios * (1.10 if mes >= 10 else 1.0)
        df_projeção.at[idx, 'Despesas_Operacionais'] = despesas_medias * multiplicador_mes
    
    # Preparar features para o modelo
    X_futuro = df_projeção[features]
    X_futuro_scaled = scaler.transform(X_futuro)
    
    # Predição
    predicoes_risco = model_lr.predict(X_futuro_scaled)
    predicoes_risco = np.clip(predicoes_risco, 0, 1)
    
    df_projeção['Risco_Churn_Estimado'] = predicoes_risco
    df_projeção['Classificacao_Alerta'] = ['ALTO RISCO' if p >= 0.5 else 'BAIXO RISCO' for p in predicoes_risco]
    
    print(f"\nPrevisões de Risco de Churn para o 2º Semestre de 2026 ({cenario}):")

    colunas_exibir = ['Ano', 'Mes_Nome', 'Classificacao_Alerta']
    resultadoStr = df_projeção[colunas_exibir].tail(6).to_string(index=False)
    print(df_projeção[colunas_exibir].tail(6).to_string(index=False))
    
    if tipo == 'treino':
        nome_arquivo = f"data/final/previsoes_futuras_2026_{cenario}.csv"
        df_projeção.to_csv(nome_arquivo, index=False, sep=';')
        print(f"\nPrevisões salvas em '{nome_arquivo}'.")
        
        plt.figure()
        sns.barplot(data=df_projeção.tail(6), x='Mes_Nome', y='Risco_Churn_Estimado', hue='Mes_Nome', palette='Reds_r')
        plt.axhline(0.5, color='red', linestyle='--', label='Linha de Alerta de Churn (50%)')
        plt.title(f'Risco de Churn Estimado - {cenario} (2º Semestre de 2026)')
        plt.xlabel('Mês')
        plt.ylabel('Probabilidade / Risco Estimado')
        plt.ylim([0, 1.1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'outputs/figures/previsao_risco_futuro_{cenario}.png')
        plt.close()
        print(f"Gráfico salvo em 'outputs/figures/previsao_risco_futuro_{cenario}.png'.")

    return df_projeção, resultadoStr

def treinar():   

    caminho_dados = leitor.getData()
    if not os.path.exists(caminho_dados):
        print(f"Erro: O arquivo de dados {caminho_dados} não foi localizado!")
        return
        
    df = executar_FASE_1(caminho_dados)    
    df_modelagem, df_previsao = executar_FASE_2(df)    
    df_modelagem, df_previsao = executar_FASE_3(df_modelagem, df_previsao, 0.75)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = executar_FASE_4(df_modelagem, 0.30, 42)
    
    model_lr, model_log, y_pred_cont, y_pred_class, y_pred_log, y_prob_log, y_pred_train_continuous = executar_FASE_5(
        X_train_scaled, X_test_scaled, y_train, y_test, features
    )

    with open('models/v1/features_v1.pkl', 'wb') as f:
        pickle.dump(features, f)

    with open('models/v1/df_previsao_v1.pkl', 'wb') as f:
        pickle.dump(df_previsao, f)

    metricas = executar_FASE_6(
        model_lr, model_log, X_test_scaled, y_test, 
        y_pred_cont, y_pred_class, y_pred_log, y_prob_log, 
        scaler, features, X_train_scaled, y_train, y_pred_train_continuous, df_previsao
    )
    
    prever_meses_futuros(model_lr, df_previsao, scaler, features, 'treino')
    
    print("\n")
    print("==================================================================")
    print("TREINAMENTO REALIZADO COM SUCESSO")
    print(f"Modelo gerado em: {datetime.now()}")    
    print("SCTec - Projeto avaliativo - IA preditiva [T2]")
    print("===================================================================")

def prever():   
    try:
        # Carregar features
        with open('models/v1/features_v1.pkl', 'rb') as f:
            features = pickle.load(f)

        # Carregar df_previsao
        with open('models/v1/df_previsao_v1.pkl', 'rb') as f:
            df_previsao = pickle.load(f)

        # Carregar model_lr
        with open('models/v1/modelo_regressao_v1.pkl', 'rb') as f:
            model_lr = pickle.load(f)

        # Carregar scaler
        with open('models/v1/escalonador_v1.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as erro:
        print(f"Erro ao ler os arquivos .pkl: {erro}")
        print("Execute primeiro o treinamento com --treinar para gerar os arquivos .pkl.")
        return
        
    # Cenários de previsão
    cenarios = {
        "Base: Normal": prever_meses_futuros(model_lr, df_previsao, scaler, features, 'visao', 'base')[0][["Ano","Mes_Nome","Classificacao_Alerta"]].tail(6),
        "Cenário 1: 10% +Receita 10% +Custo": prever_meses_futuros(model_lr, df_previsao, scaler, features, 'visao', 'cenario1')[0][["Ano","Mes_Nome","Classificacao_Alerta"]].tail(6),
        "Cenário 2: 20% +Custo": prever_meses_futuros(model_lr, df_previsao, scaler, features, 'visao', 'cenario2')[0][["Ano","Mes_Nome","Classificacao_Alerta"]].tail(6),
        "Cenário 3: 50% +Despesas": prever_meses_futuros(model_lr, df_previsao, scaler, features, 'visao', 'cenario3')[0][["Ano","Mes_Nome","Classificacao_Alerta"]].tail(6),
    }

    relatorio.generate(cenarios)
    relatorio.open()
    
    arquivo = 'models/v1/df_previsao_v1.pkl'
    modificacao = os.path.getmtime(arquivo)
    data_modificacao = datetime.fromtimestamp(modificacao)
    
    print("\n")
    print("==================================================================")
    print("PREVISÃO REALIZADA COM SUCESSO")
    print(f"Modelo treinado em: {data_modificacao}")    
    print("SCTec - Projeto avaliativo - IA preditiva [T2]")
    print("==================================================================")
