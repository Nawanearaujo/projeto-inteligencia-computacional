import pandas as pd
import os

def clean_and_prepare_data():
    """
    Função responsável por ler os arquivos originais (.data),
    tratar valores ausentes e unificar tudo em um único arquivo CSV.
    """
    # 1. Definindo os caminhos (onde os dados estão e para onde vão)
    raw_folder = "data/raw/heart+disease"
    processed_file = "data/processed/heart_disease_cleaned.csv"
    
    # 2. Nomes das colunas 
    colunas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # 3. Lista dos arquivos de diferentes hospitais que serão unidos
    arquivos = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data"
    ]
    
    lista_de_tabelas = []
    
    # 4. Lendo cada arquivo e preparando para junção
    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(raw_folder, nome_arquivo)
        
        # Verifica se o arquivo existe na pasta antes de tentar ler
        if os.path.exists(caminho_completo):
            # O 'na_values="?" faz com que o Python coloque as ? como valores nulos
            df_temp = pd.read_csv(caminho_completo, names=colunas, na_values='?')
            lista_de_tabelas.append(df_temp)
            print(f"-> Lido com sucesso: {nome_arquivo}")
        else:
            print(f"-> Aviso: Arquivo não encontrado - {caminho_completo}")
    
    # 5. Unindo todas as tabelas em uma única tabela 
    tabela_final = pd.concat(lista_de_tabelas, ignore_index=True)

    # Preenche os valores numéricos ausentes (NaN) com a mediana de cada coluna
    for col in tabela_final.columns:
        if col != 'target':
            mediana = tabela_final[col].median()
            tabela_final[col] = tabela_final[col].fillna(mediana)
    
    # Qualquer valor > 0 na coluna 'target' será considerado como 1 (risco cardíaco)
    tabela_final['target'] = tabela_final['target'].apply(lambda x: 1 if x > 0 else 0)

    tabela_final = tabela_final.dropna(subset=['target'])
    
    # 7. Salva o resultado na pasta 'processed'
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    tabela_final.to_csv(processed_file, index=False)
    print(f"-> Base limpa e unificada salva em: {processed_file}")
    
    return processed_file

if __name__ == "__main__":
    clean_and_prepare_data()