from pathlib import Path

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #função que separa o dataset em teste e treino
from sklearn.linear_model import LogisticRegression #utilizar a regressão logística do Sckit-Learn
from sklearn.ensemble import RandomForestClassifier #utilizar o Random Forest do Sckit-Learn
import joblib #Bibloetaca para salvar o melhor algoritmo
from sklearn.metrics import accuracy_score
import src.preprocessing.clean_data as clean_data

dados_pacientes = pd.read_csv(clean_data.clean_and_prepare_data())
dados_pacientes = dados_pacientes.drop_duplicates()
print(f"\nRestaram {dados_pacientes.shape[0]} linhas reais.")
print(dados_pacientes.head())

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

def salvar_modelo_campeao(modelo, nome_arquivo: str, pasta_destino: str = 'models/saved'):

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    caminho_completo = os.path.join(pasta_destino, nome_arquivo)

    # salva o modelo campeão em disco
    joblib.dump(modelo, caminho_completo)

# Bloco de execução principal
if __name__ == "__main__":
    try:
        X_treino, X_teste, Y_treino, Y_teste = preparar_e_dividir_dados(dados_pacientes)
        print("\nTreinando os modelos candidatos...")
        modelo_rl = treinar_regressao_logistica(X_treino, Y_treino)
        modelo_rf = treinar_random_forest(X_treino, Y_treino)

        # TEstando os modelos
        previsoes_rl = modelo_rl.predict(X_teste)
        previsoes_rf = modelo_rf.predict(X_teste)

        # Compara o Y esperado com o resultado obtido
        acuracia_rl = accuracy_score(Y_teste, previsoes_rl)
        acuracia_rf = accuracy_score(Y_teste, previsoes_rf)

        print(f"\nAcurácia Preliminar - Regressão Logística: {acuracia_rl:.4f}")
        print(f"\nAcurácia Preliminar - Random Forest: {acuracia_rf:.4f}")

        if acuracia_rl >= acuracia_rf:
            print("\n-> Vencedor: Regressão Logística!")
            salvar_modelo_campeao(modelo_rl, "modelo_regressao_logistica.joblib")
        else:
            print("\n-> Vencedor: Random Forest!")
            salvar_modelo_campeao(modelo_rf, "modelo_random_forest.joblib")


        print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")

    except FileNotFoundError as erro_arquivo:
        print(f"\n[AVISO]: {erro_arquivo}")