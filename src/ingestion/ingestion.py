import pandas as pd
import os

def ingest_and_combine_heart_data():
    """
    Realiza a leitura dos arquivos de dados originais provenientes das quatro 
    instituições hospitalares do consórcio UCI, unifica os registros em uma 
    estrutura de dados bruta centralizada e persiste o arquivo resultante no 
    diretório de dados brutos (raw).

    Retorno:
        None (Gera e salva o arquivo unificado diretamente em disco).
    """
    # 1. Definição dos caminhos de origem e de destino
    # Estabelece os mapeamentos de diretórios para localização da fonte e exportação do artefato
    raw_folder = "data/raw/heart+disease"
    output_raw_file = "data/raw/heart_disease_raw.csv"
    
    # 2. Atributos padrão da base de dados
    # Define os identificadores canônicos das colunas mapeadas conforme o repositório UCI
    colunas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # 3. Listagem das fontes de dados hospitalares
    # Agrupa os nomes dos arquivos oficiais que compõem o ecossistema federado do estudo
    arquivos = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data"
    ]
    
    lista_de_tabelas = []
    
    try:
        # 4. Varredura e carregamento individual das fontes hospitalares
        # Percorre a lista de arquivos para ler e estruturar cada base de dados na memória
        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(raw_folder, nome_arquivo)
            
            if os.path.exists(caminho_completo):
                # Converte o caractere de interrogação '?' em valores nulos reconhecidos pelo ecossistema do Pandas (NaN),
                # garantindo o isolamento da estrutura para o tratamento posterior no pré-processamento
                df_temp = pd.read_csv(caminho_completo, names=colunas, na_values='?')
                lista_de_tabelas.append(df_temp)
                print(f"-> Ingestão realizada: {nome_arquivo}")
            else:
                print(f"-> Aviso: Arquivo não encontrado durante a ingestão - {caminho_completo}")
        
        # Interrompe o fluxo caso nenhuma fonte de dados tenha sido carregada no ambiente
        if not lista_de_tabelas:
            print("\n[ERRO]: Nenhum arquivo de hospital foi encontrado para ingestão.")
            return

        # 5. Consolidação das matrizes de dados
        # Concatena os DataFrames carregados de maneira sequencial em uma única matriz estruturada
        df_bruto_unificado = pd.concat(lista_de_tabelas, ignore_index=True)
        
        # 6. Criação do diretório de destino e persistência física do arquivo
        # Garante a existência do diretório 'raw' no disco e exporta a base bruta intocada
        os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)
        df_bruto_unificado.to_csv(output_raw_file, index=False)
        
        print(f"\nIngestão finalizada! Base bruta total unificada com {len(df_bruto_unificado)} registros.")
        print(f"Salvo em: {output_raw_file}")

    except Exception as e:
        print(f"\n[ERRO] Falha crítica durante o processo de ingestão: {e}")

if __name__ == "__main__":
    # Inicia a execução do fluxo automatizado de ingestão e consolidação de dados
    ingest_and_combine_heart_data()