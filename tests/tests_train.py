import pytest
import os
import pandas as pd
import numpy as np
import joblib
from src.models.train import (
    preparar_e_dividir_dados,
    treinar_regressao_logistica,
    treinar_random_forest,
    salvar_modelo_campeao
)


@pytest.fixture #Possibilita o reaproveitamento dos dados criados
def dummy_data():
    """
    Cria uma base de dados fictícia balanceada para alimentar os testes de treino.
    """
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Alvo balanceado
    })


def test_split_data(dummy_data):
    """
    Garante que a divisão de dados mantém a proporção de 80/20 estabelecida.
    """
    X_train, X_test, Y_train, Y_test = preparar_e_dividir_dados(dummy_data)
    
    # Verifica se real constituiusse a escala 8x2 (80% treino, 20% teste)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(Y_train) == 8
    assert len(Y_test) == 2


def test_train_lr(dummy_data):
    """
    Valida se o treino da Regressão Logística gera um modelo válido e funcional.
    """
    X_train, _, Y_train, _ = preparar_e_dividir_dados(dummy_data)
    model = treinar_regressao_logistica(X_train, Y_train)
    
    # Garante que o modelo foi ajustado e possui o método de predição
    assert hasattr(model, "predict")
    assert model.predict(X_train[:1]) in [0, 1]


def test_train_rf(dummy_data):
    """
    Valida se o treino do Random Forest gera um modelo válido e funcional.
    """
    X_train, _, Y_train, _ = preparar_e_dividir_dados(dummy_data)
    model = treinar_random_forest(X_train, Y_train)
    
    # Garante que as árvores de decisão foram criadas corretamente
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) == 100


def test_save_champion(tmp_path, dummy_data):
    """
    Verifica se o modelo campeão é corretamente persistido e salvo em disco.
    """
    X_train, _, Y_train, _ = preparar_e_dividir_dados(dummy_data)
    model = treinar_regressao_logistica(X_train, Y_train)
    
    # Cria uma pasta temporária isolada para testar o salvamento em disco
    test_model_dir = tmp_path / "models"
    filename = "best_model.joblib"
    
    salvar_modelo_campeao(model, filename, pasta_destino=str(test_model_dir))
    
    # Garantias: O arquivo físico existe e pode ser recarregado com sucesso?
    full_path = test_model_dir / filename
    assert os.path.exists(full_path)
    
    loaded_model = joblib.load(full_path)
    assert hasattr(loaded_model, "predict")


def test_model_accuracy_threshold(dummy_data):
    """
    Garante que o modelo treinado possui uma capacidade mínima de aprendizado
    e não está chutando os resultados aleatoriamente (Acurácia > 50%).
    """
    X_train, X_test, Y_train, Y_test = preparar_e_dividir_dados(dummy_data)
    model = treinar_random_forest(X_train, Y_train)
    
    previsoes = model.predict(X_test)
    acuracia = accuracy_score(Y_test, previsoes)
    
    # O modelo precisa alcançar acurácia mínima para ser considerado inteligente em um cenário controlado
    assert acuracia >= 0.5


def test_model_prediction_output(tmp_path, dummy_data):
    """
    Simula uma consulta médica real: carrega o modelo campeão salvo e garante
    que ele consegue receber dados de um novo paciente e prever o risco (0 ou 1).
    """
    X_train, _, Y_train, _ = preparar_e_dividir_dados(dummy_data)
    model = treinar_regressao_logistica(X_train, Y_train)
    
    # Salva temporariamente
    test_dir = tmp_path / "models"
    filename = "final_model.joblib"
    salvar_modelo_campeao(model, filename, pasta_destino=str(test_dir))
    
    # Carrega o cérebro da IA de volta
    modelo_carregado = joblib.load(test_dir / filename)
    
    # Cria os exames de 1 paciente novo (sem a coluna target)
    paciente_novo = pd.DataFrame([{
        'feature1': 0.75,
        'feature2': 0.22
    }])
    
    # A IA deve prever se o paciente é 0 (saudável) ou 1 (risco)
    predicao = modelo_carregado.predict(paciente_novo)
    assert predicao[0] in [0, 1] 