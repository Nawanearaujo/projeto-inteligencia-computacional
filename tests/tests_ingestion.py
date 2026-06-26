import pytest
import os
import pandas as pd
from unittest.mock import patch
from src.ingestion.ingestion import ingest_and_combine_heart_data

def test_ingest_no_files():
    """
    Garante que a função trata o erro corretamente e exibe um aviso
    caso nenhum arquivo hospitalar seja encontrado na pasta de origem.
    """
    # Simula a inexistência dos arquivos
    with patch('os.path.exists', return_value=False):
        with patch('builtins.print') as mock_print:
            ingest_and_combine_heart_data()
            # Valida se a mensagem de erro esperada foi impressa
            mock_print.assert_any_call("\n[ERRO]: Nenhum arquivo de hospital foi encontrado para ingestão.")


# Teste Unitário
def test_ingest_success(tmp_path):
    """
    Cria hospitais fictícios com dados contendo '?' e verifica se a função
    consegue unificar os registros e converter os caracteres em nulos.
    """
    orig_dir = os.getcwd()
    
    try:
        # Montagem da estrutura de pastas exigida pelo código dentro do tmp_path
        raw_src = tmp_path / "data" / "raw" / "heart+disease"
        raw_src.mkdir(parents=True)
        
        # Simulação de bases de dois hospitais
        hosp1_df = pd.DataFrame({'age': [50, 60], 'target': [0, 1]})
        hosp2_df = pd.DataFrame({'age': ['?', 45], 'target': [1, 0]})  # Contém o caractere '?'
        
        hosp1_df.to_csv(raw_src / "processed.cleveland.data", index=False, header=False)
        hosp2_df.to_csv(raw_src / "processed.hungarian.data", index=False, header=False)
        
        # Mudança temporaria para a raiz simulada
        os.chdir(tmp_path)
        
        # Execução da função testada
        ingest_and_combine_heart_data()
        
        # GARANTIAS
        
        # Verifica se o rquivo unificado final foi gerado fisicamente no disco
        assert os.path.exists("data/raw/heart_disease_raw.csv")
        
        # Verifica se a junção das linhas ocorreu de forma correta (2 de cada = 4)
        res_df = pd.read_csv("data/raw/heart_disease_raw.csv")
        assert len(res_df) == 4
        
        # Verificação se "?" foi substituído por "NaN"
        assert pd.isna(res_df['age'].iloc[2])

    # Voltar para raiz original
    finally:
        os.chdir(orig_dir)


def test_ingest_critical_error():
    """
    Simula uma falha de sistema (exceção crítica) ao tentar ler arquivos
    para verificar se a função absorve o erro de forma segura sem quebrar o sistema.
    """
    # Teste de disparo de falha de sistema intencional
    with patch('os.path.exists', side_effect=RuntimeError("Falha de hardware simulada")):
        with patch('builtins.print') as mock_print:
            ingest_and_combine_heart_data()
            # Garante que o bloco try/except capturou o erro e avisou no terminal
            mock_print.assert_any_call("\n[ERRO] Falha crítica durante o processo de ingestão: Falha de hardware simulada")