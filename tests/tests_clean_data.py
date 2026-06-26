import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.preprocessing.clean_data import clean_and_prepare_data


def test_clean_and_prepare_data_Teste_existencia_Arquivo():
    """
    Garante que a função trata o erro corretamente e exibe um aviso
    caso o arquivo bruto original não seja encontrado (Requisito 6).
    """
    # Simula não existência do arquivo bruto
    with patch('os.path.exists', return_value=False):
        # Verifica comportamento do sistema capturando o print
        with patch('builtins.print') as mock_print:
            clean_and_prepare_data()
            mock_print.assert_any_call("\n[ERRO]: Arquivo bruto não encontrado em: data/raw/heart_disease_raw.csv")


# Teste Unitário
def test_clean_and_prepare_data_sucess(tmp_path):
    """
    Cria uma base bruta simulada contendo imperfeições para checar se 
    a limpeza, binarização e normalização ocorrem com sucesso.
    """
    # Salva o diretório real do projeto
    diretorio_atual = os.getcwd()
    
    try:
        # Criação de caminhos temporários dentro do isolamento do tmp_path
        pasta_dados = tmp_path / "data" / "raw"
        pasta_dados.mkdir(parents=True)
        arquivo_entrada = pasta_dados / "heart_disease_raw.csv"

        # Criação de um DataFrame bruto com os problemas que a função trata (Simulação de dataframe):
        # - Zeros -> virar mediana
        # - Target variando de 0 a 3 -> virar binário
        # - Colunas categóricas numéricas prontas para One-Hot Encoding
        dados_brutos = pd.DataFrame({
            'age': [50.0, 60.0, 70.0],
            'trestbps': [120.0, 130.0, 140.0],
            'chol': [0.0, 200.0, 300.0],  # Esse 0 deve sumir e virar a mediana (250)
            'thalach': [150.0, 160.0, 170.0],
            'oldpeak': [1.0, 2.0, 3.0],
            'cp': [1, 2, 3],              # Categórica
            'restecg': [0, 1, 0],         # Categórica
            'slope': [1, 2, 1],           # Categórica
            'thal': [2, 3, 2],            # Categórica
            'target': [0, 2, 3]           # Será binarizado para 0, 1, 1
        })
        dados_brutos.to_csv(arquivo_entrada, index=False)

        # Altera temporariamente a raiz de execução para a pasta temporária
        os.chdir(tmp_path)
        
        # Executa a função de pré-processamento
        clean_and_prepare_data()
        
        # GARANTIAS / Asserts:
        # Verifica se o arquivo final limpo foi gerado na pasta processed
        assert os.path.exists("data/processed/heart_disease_cleaned.csv")
        
        df_limpo = pd.read_csv("data/processed/heart_disease_cleaned.csv")
        
        # Verifica binarização do target: só pode conter 0 e 1 (o 2 e 3 sumiram)
        assert set(df_limpo['target'].unique()).issubset({0, 1})
        assert df_limpo['target'].iloc[1] == 1
        
        #  Verifica normalização Min-Max: valores da idade devem estar entre 0 e 1
        assert df_limpo['age'].min() == 0.0
        assert df_limpo['age'].max() == 1.0

        #  Verifica se o zero do colesterol sumiu e não causou erro (foi imputado)
        assert float(df_limpo['chol'].iloc[0]) > 0.0

    finally:
        # Restaura obrigatoriamente o diretório original para não afetar outros testes
        os.chdir(diretorio_atual)


# Teste de Validação de Entrada
def test_clean_and_prepare_data_erro_colunas_faltantes(tmp_path):
    """
    Verifica a resiliência do código criando um arquivo que não possui 
    as colunas esperadas pelo processamento, forçando um erro controlado.
    """
    diretorio_atual = os.getcwd()
    
    try:
        pasta_dados = tmp_path / "data" / "raw"
        pasta_dados.mkdir(parents=True)
        arquivo_entrada = pasta_dados / "heart_disease_raw.csv"

        # Tabela inválida sem a coluna 'chol' ou 'target' necessária no pipeline
        dados_invalidos = pd.DataFrame({
            'coluna_aleatoria': [1, 2, 3]
        })
        dados_invalidos.to_csv(arquivo_entrada, index=False)

        os.chdir(tmp_path)

        # Como a função tenta acessar colunas específicas, ela deve quebrar com KeyError
        with pytest.raises(KeyError):
            clean_and_prepare_data()
            
    finally:
        os.chdir(diretorio_atual)