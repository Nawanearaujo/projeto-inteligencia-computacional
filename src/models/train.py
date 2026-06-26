from pathlib import Path

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #função que separa o dataset em teste e treino
from sklearn.linear_model import LogisticRegression #utilizar a regressão logística do Sckit-Learn
from sklearn.ensemble import RandomForestClassifier #utilizar o Random Forest do Sckit-Learn
import joblib #Bibloetaca para salvar o melhor algoritmo
from sklearn.metrics import accuracy_score

# Dividir os dados em treino (80%) e teste (20%)
def preparar_e_dividir_dados(dados_pacientes: pd.DataFrame):

  #target == 0 => Saudável; target == 1 => Risco cardíaco
  coluna_alvo='target'

  # Refere-se ao valor de X (sintomas e exames)
  caracteristicas = dados_pacientes.drop(columns=[coluna_alvo])

  # Refere-se à resposta final que se quer prever (a coluna target)
  rotulos = dados_pacientes[coluna_alvo]

  X_treino, X_teste, Y_treino, Y_teste = train_test_split(
      caracteristicas,
      rotulos,
      test_size=0.2,
      random_state=42, # garante que a divisão seja sempre igual
      stratify=rotulos
  )

  return X_treino, X_teste, Y_treino, Y_teste

def treinar_regressao_logistica(X_treino, Y_treino) -> LogisticRegression:
  modelo_logistica = LogisticRegression(max_iter=1000, random_state=42)
  modelo_logistica.fit(X_treino, Y_treino)

  return modelo_logistica

def treinar_random_forest(X_treino, Y_treino) -> RandomForestClassifier:
  modelo_forest = RandomForestClassifier(n_estimators=100, random_state=42)
  modelo_forest.fit(X_treino, Y_treino)

  return modelo_forest

def salvar_modelo_treinado(modelo, nome_arquivo: str, pasta_destino: str = 'models_saved'):

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    caminho_completo = os.path.join(pasta_destino, nome_arquivo)

    # salva o modelo treinado em disco
    joblib.dump(modelo, caminho_completo)

# Bloco de execução principal
if __name__ == "__main__":
    # Caminho do arquivo de features gerado na etapa anterior
    caminho_features = os.path.join("data", "processed", "heart_disease_features.csv")
    try:
        # Verifica se o caminho do arquivo de features existe antes de prosseguir com o treinamento
        if not os.path.exists(caminho_features):
            raise FileNotFoundError(
                f"O arquivo '{caminho_features}' não foi encontrado.\n"
                f"Certifique-se de rodar o script de Engenharia de Features primeiro "
            )
        
        # Carrega o dataset de features para iniciar o processo de treinamento
        dados_pacientes = pd.read_csv(caminho_features)
        print("\nPrimeiras linhas do dataset processado carregado com sucesso:")
        print(dados_pacientes.head())

        X_treino, X_teste, Y_treino, Y_teste = preparar_e_dividir_dados(dados_pacientes)
        print("\nTreinando os modelos candidatos...")
        modelo_rl = treinar_regressao_logistica(X_treino, Y_treino)
        modelo_rf = treinar_random_forest(X_treino, Y_treino)

        # Testando os modelos
        previsoes_rl = modelo_rl.predict(X_teste)
        previsoes_rf = modelo_rf.predict(X_teste)

        # Compara o Y esperado com o resultado obtido
        acuracia_rl = accuracy_score(Y_teste, previsoes_rl)
        acuracia_rf = accuracy_score(Y_teste, previsoes_rf)

        salvar_modelo_treinado(modelo_rl, "modelo_regressao_logistica.joblib")
        salvar_modelo_treinado(modelo_rf, "modelo_random_forest.joblib")

        print(f"\nAcurácia - Regressão Logística: {acuracia_rl:.4f}")
        print(f"\nAcurácia - Random Forest: {acuracia_rf:.4f}")

        if acuracia_rl >= acuracia_rf:
            print("\n-> Vencedor: Regressão Logística!")
            #salvar_modelo_campeao(modelo_rl, "modelo_regressao_logistica.joblib")
        else:
            print("\n-> Vencedor: Random Forest!")
            #salvar_modelo_campeao(modelo_rf, "modelo_random_forest.joblib")


        print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")

    except FileNotFoundError as erro_arquivo:
        print(f"\n[AVISO]: {erro_arquivo}")