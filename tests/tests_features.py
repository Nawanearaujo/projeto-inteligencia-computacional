import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.features.build_features import select_best_features


def test_select_best_features_ExistenciaArquivo():
    """
    Garante que a função trata o erro corretamente e exibe um aviso
    caso o arquivo de dados limpos não seja encontrado (Requisito 6).
    """
    # Simula não existência do arquivo
    with patch('os.path.exists', return_value=False):
        # Verifica comportamento do sistema
        with patch('builtins.print') as mock_print:
            select_best_features()
            mock_print.assert_any_call("\n[ERRO]: Dados processados não encontrados em: data/processed/heart_disease_cleaned.csv")


# Teste Unitário
def test_select_best_features_sucess(tmp_path):
    """
    Cria dados fictícios para simular o comportamento da Random Forest
    e verifica se o arquivo otimizado final é gerado (Requisito 6).
    """
    # Salva o diretório real do projeto
    diretorio_atual = os.getcwd()
    
    try:
        # Criação de caminhos temporários dentro do isolamento do tmp_path
        pasta_dados = tmp_path / "data" / "processed"
        pasta_dados.mkdir(parents=True)
        
        arquivo_entrada = pasta_dados / "heart_disease_cleaned.csv"

        # Criação de um DataFrame falso: 3 colunas normais e o alvo (target)
        # A 'coluna_boa' vai ser idêntica ao target para a Random Forest dar muita importância a ela
        dados_falsos = pd.DataFrame({
            'coluna_boa': [1, 0, 1, 0, 1, 0],
            'coluna_ruim': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'target': [1, 0, 1, 0, 1, 0]
        })
        dados_falsos.to_csv(arquivo_entrada, index=False)

        # Altera temporariamente a raiz de execução para a pasta temporária
        os.chdir(tmp_path)
        
        # Executa a função do ecossistema
        select_best_features()
        
        # GARANTIAS (Asserts):
        # Verifica se o arquivo de saída foi realmente criado no caminho esperado pela função
        assert os.path.exists("data/processed/heart_disease_features.csv")
        
        # Verifica se o arquivo gerado contém a coluna target?
        df_resultado = pd.read_csv("data/processed/heart_disease_features.csv")
        assert 'target' in df_resultado.columns

    finally:
        # Restaura obrigatoriamente o diretório original para não afetar outros testes
        os.chdir(diretorio_atual)


# TESTE DE VERIFICAÇÃO DE ENTRADA (FALHA CONTROLADA)
def test_select_best_features_erro_se_nao_houver_coluna_target(tmp_path):
    """
    Verifica a entrada de dados: se a tabela não tiver a coluna 'target',
    o código deve levantar um erro de chave (KeyError).
    """
    # Salva o diretório real do projeto
    diretorio_atual = os.getcwd()
    
    try:
        # Criação de caminhos temporários dentro do isolamento do tmp_path
        pasta_dados = tmp_path / "data" / "processed"
        pasta_dados.mkdir(parents=True)
        arquivo_entrada = pasta_dados / "heart_disease_cleaned.csv"

        # Dados sem a coluna obrigatória 'target'
        dados_invalidos = pd.DataFrame({
            'idade': [30, 40, 50],
            'pressao': [12, 13, 14]
        })
        dados_invalidos.to_csv(arquivo_entrada, index=False)

        # Altera temporariamente a raiz de execução para a pasta temporária
        os.chdir(tmp_path)

        # Esperamos que o pytest capture um KeyError porque 'target' não existe para ser dropado
        with pytest.raises(KeyError):
            select_best_features()
            
    finally:
        # Restaura obrigatoriamente o diretório original para não afetar outros testes
        os.chdir(diretorio_atual)