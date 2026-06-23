import pandas as pd
import os

def clean_and_prepare_data():
    """
    Realiza a leitura do arquivo bruto unificado pela etapa de ingestão,
    aplica o tratamento de valores ausentes (incluindo a correção de zeros 
    artificiais no colesterol), executa a binarização do target, codifica 
    as variáveis categóricas via One-Hot Encoding, normaliza os dados numéricos 
    e persiste o resultado estruturado pronto para a modelagem.

    Retorno:
        None (Gera e salva o arquivo limpo diretamente em disco).
    """
    # Define os caminhos de diretórios conforme a arquitetura modular estabelecida
    raw_file = "data/raw/heart_disease_raw.csv"
    processed_file = "data/processed/heart_disease_cleaned.csv"
    
    # Verifica a existência do arquivo bruto de entrada para garantir a segurança do fluxo
    if not os.path.exists(raw_file):
        print(f"\n[ERRO]: Arquivo bruto não encontrado em: {raw_file}")
        print("Recomenda-se executar o script 'src/ingestion/ingestion.py' previamente.")
        return

    # Carrega a base de dados bruta unificada na memória
    tabela_final = pd.read_csv(raw_file)
    print(f"-> Base bruta carregada com sucesso. Registros iniciais: {len(tabela_final)}")
    
    # Elimina os registros corrompidos que apresentam todas as colunas nulas
    tabela_final = tabela_final.dropna(how='all')
    
    # Substitui os valores iguais a zero na coluna de colesterol por valores nulos (pd.NA),
    # visto que a análise exploratória prévia identificou tais registros como dados omissos
    tabela_final['chol'] = tabela_final['chol'].replace(0, pd.NA)
    
    # Percorre as colunas do conjunto de dados para preencher os valores ausentes 
    # utilizando a mediana calculada de cada atributo correspondente
    for col in tabela_final.columns:
        if tabela_final[col].isnull().any():
            mediana_val = tabela_final[col].median()
            tabela_final[col] = tabela_final[col].fillna(mediana_val)
    
    # Mapeia a coluna alvo de sua escala original de gravidade (0 a 4) para uma 
    # representação estritamente binária (0 indica ausência e 1 indica presença de risco)
    tabela_final['target'] = tabela_final['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Define a lista de variáveis categóricas nominais sujeitas ao processo de codificação
    colunas_categoricas = ['cp', 'restecg', 'slope', 'thal']
    
    # Converte os tipos das colunas categóricas para string a fim de assegurar o correto 
    # mapeamento binário e evitar distorções ordinais lineares nos modelos preditivos
    for col in colunas_categoricas:
        tabela_final[col] = tabela_final[col].astype(int).astype(str)
        
    # Aplica a técnica de One-Hot Encoding com a remoção da primeira coluna para mitigar a multicolinearidade
    tabela_final = pd.get_dummies(tabela_final, columns=colunas_categoricas, drop_first=True)
    
    # Define os atributos contínuos que necessitam de ajuste de escala proporcional
    colunas_continuas = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Aplica a normalização Min-Max para reescalar as variáveis numéricas dentro do intervalo contido entre 0 e 1
    for col in colunas_continuas:
        min_val = tabela_final[col].min()
        max_val = tabela_final[col].max()
        if max_val - min_val != 0:
            tabela_final[col] = (tabela_final[col] - min_val) / (max_val - min_val)
    
    # Cria o diretório de destino na pasta 'processed' caso este ainda não exista no ambiente
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    
    # Converte as variáveis booleanas resultantes do get_dummies para o formato numérico binário (0 e 1)
    colunas_bool = tabela_final.select_dtypes(include='bool').columns
    tabela_final[colunas_bool] = tabela_final[colunas_bool].astype(int)
    
    # Exporta o conjunto de dados processado em formato CSV livre de índices sequenciais artificiais
    tabela_final.to_csv(processed_file, index=False)
    print(f"\nSucesso! Os dados foram limpos, normalizados e salvos em: {processed_file}")
    print(f"Colunas finais contidas no dataset processado: {list(tabela_final.columns)}")

if __name__ == "__main__":
    # Inicia a execução do fluxo de pré-processamento de dados
    clean_and_prepare_data()