import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

def select_best_features():
    """
    Realiza a leitura dos dados pré-processados, aplica algoritmos de 
    seleção de recursos baseados em árvores de decisão (Random Forest) 
    para mensurar a relevância de cada atributo frente ao alvo cardíaco, 
    filtra as variáveis mais importantes e exporta o conjunto otimizado.

    Retorno:
        None (Persiste o dataset reduzido no diretório correspondente).
    """
    # Define os caminhos de entrada e de destino dos dados mapeados
    input_file = "data/processed/heart_disease_cleaned.csv"
    output_file = "data/processed/heart_disease_features.csv"
    
    # Valida a existência do artefato produzido na etapa de pré-processamento
    if not os.path.exists(input_file):
        print(f"\n[ERRO]: Dados processados não encontrados em: {input_file}")
        print("Recomenda-se executar o script 'src/preprocessing/clean_data.py' previamente.")
        return

    # Efetua o carregamento da base limpa e normalizada
    df = pd.read_csv(input_file)
    print(f"-> Matriz de dados carregada. Atributos originais: {df.shape[1] - 1}")

    # Separa a matriz de atributos (X) do vetor de rótulos alvo (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # Instancia o classificador Random Forest para extração de relevância estatística
    # Define uma semente aleatória (random_state) para garantir a reprodutibilidade
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Captura os coeficientes de importância atribuídos a cada coluna do conjunto
    importancias = model.feature_importances_
    
    # Cria um DataFrame auxiliar para fins de ordenação e exibição dos resultados
    df_importancia = pd.DataFrame({
        'Atributo': X.columns,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    print("\n=== Ranking de Importância das Variáveis para o Coração ===")
    print(df_importancia.to_string(index=False))

    # Define um limiar de corte estatístico para exclusão de ruídos ou variáveis irrelevantes
    # Atributos com contribuição inferior a 0.015% (0.015) são desconsiderados
    limiar_corte = 0.015
    atributos_selecionados = df_importancia[df_importancia['Importancia'] >= limiar_corte]['Atributo'].tolist()
    
    print(f"\n-> Limiar de corte definido em: {limiar_corte}")
    print(f"-> Quantidade de atributos mantidos: {len(atributos_selecionados)}")
    print(f"-> Atributos descartados: {list(set(X.columns) - set(atributos_selecionados))}")

    # Reconfigura a base de dados mantendo apenas as colunas filtradas e o target
    colunas_finais = atributos_selecionados + ['target']
    df_otimizado = df[colunas_finais]

    # Garante a criação da estrutura de pastas para salvar o novo arquivo
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Exporta os dados otimizados que alimentarão a etapa de treinamento de modelos
    df_otimizado.to_csv(output_file, index=False)
    print(f"\nSucesso! O conjunto de atributos otimizados foi persistido em: {output_file}")

if __name__ == "__main__":
    # Inicia a rotina automatizada de seleção de recursos do ecossistema
    select_best_features()