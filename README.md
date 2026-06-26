# Projeto Inteligência Computacional: Previsão de Doenças Cardíacas

Este projeto foi desenvolvido como requisito avaliativo para a disciplina de Inteligência Computacional do curso de Bacharelado em Sistemas de Informação no Instituto Federal de Alagoas (IFAL) - Campus Arapiraca. O foco é aplicar técnicas de Machine Learning para a área da saúde, utilizando boas práticas de Engenharia de Software e governança de dados.

Professor: Edvonaldo Horácio dos Santos  
Equipe: Annne Karolyne Caetano, Julio Cesar dos Santos Oliveira, Marcos Felype Silva e Nawane Araújo dos Santos

---

## 1. Explicação do Problema

As doenças cardiovasculares são uma das principais causas de mortalidade no mundo. O diagnóstico rápido e preciso é vital para o tratamento eficaz. O problema abordado neste projeto é de Classificação Supervisionada: prever de forma automatizada se um paciente possui ou não doença cardíaca com base em seu histórico clínico e exames.

Para isso, utilizamos o Heart Disease Dataset da UCI Machine Learning Repository, unificando 920 registros reais de pacientes de quatro instituições médicas (Cleveland, Hungria, Suíça e V.A. Medical Center).

---

## 2. Estrutura do Projeto

O repositório segue um padrão modular para separar dados, código-fonte e documentação:

projeto-inteligencia-computacional/
├── data/
│ ├── raw/ # Dados brutos originais da UCI (.data) e arquivo unificado
│ └── processed/ # Dados limpos, pré-processados e finalizados (.csv)
├── notebooks/ # Notebooks para Análise Exploratória (EDA)
├── src/
│ ├── ingestion/ # Scripts de coleta e consolidação inicial de dados
│ ├── preprocessing/ # Rotinas de limpeza avançada e tratamento de nulos
│ ├── features/ # Engenharia de features e transformações matemáticas
│ ├── models/ # Treinamento de algoritmos de Machine Learning
│ └── evaluation/ # Métricas de validação dos modelos
├── metrics/ # Gráficos e resultados de avaliação salvos
├── models_saved/ # Modelos treinados persistidos (joblib/pickle)
├── requirements.txt # Dependências do projeto
└── README.md # Documentação principal

---

## 3. Configuração do Ambiente e Execução

### 3.1 Pré-requisitos e Bibliotecas

- Python 3.10 ou superior.
- Gerenciador de versão: Git.
- Bibliotecas principais de manipulação: `pandas`, `numpy`, `os`.
- Bibliotecas principais de Machine Learning e Visualização: `scikit-learn`, `matplotlib`, `seaborn`.
  \*Biblioteca para execução de testes: `pytest`

### 3.2 Como Executar (Comandos do Git e do Ambiente Virtual)

Clonando o repositório para a máquina local:

```powershell
git clone <URL_DO_REPOSITORIO>
cd projeto-inteligencia-computacional
```

Criar e ativar o ambiente virtual para isolar as dependências:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Instalar as bibliotecas necessárias:

```powershell
pip install -r requirements.txt
```

Executar o script de ingestão para gerar a base bruta unificada:

```powershell
python src/ingestion/ingestion.py
```

---

## 4. Etapas do Desenvolvimento

### 4.1 Coleta e Ingestão de Dados (`src/ingestion/`)

Os dados brutos originais são fragmentados por hospitais (Cleveland, Hungria, Suíça e V.A. Medical Center). O script de ingestão consolida essas quatro bases em um único arquivo, garantindo a imutabilidade da fonte original. As seguintes ações são executadas nesta etapa inicial:
*Padronização de Colunas: Aplicação do dicionário de dados oficial com os 14 atributos padrão da literatura médica (age, sex, cp, trestbps, etc.).
*Mapeamento de Nulos: Conversão do caractere `?` (dado faltante na base original) para a estrutura `NaN` do Pandas.
\*Consolidação: Concatenação das bases resultando no arquivo unificado `heart_disease_raw.csv` alocado na pasta `data/raw/`.

### 4.2 Análise Exploratória de Dados - EDA (`notebooks/`)

Esta etapa compreende o diagnóstico visual e a investigação estatística inicial do comportamento das variáveis contidas na base consolidada (heart_disease_raw.csv) antes de qualquer transformação algorítmica. Utilizando bibliotecas como matplotlib e seaborn, os seguintes diagnósticos foram implementados para orientar as estratégias de tratamento:

Mapeamento de Distribuição e Anomalias: Construção de histogramas acoplados à estimativa de densidade de kernel (KDE) para analisar o comportamento estatístico da idade (age) e do colesterol (chol). Esta visualização permitiu identificar uma grave anomalia clínica: um pico artificial de registros com colesterol igual a zero, caracterizando dados omissos mascarados.

Estudo Oclusivo por Categoria: Aplicação de diagramas de caixa (boxplots) para confrontar as distribuições dos quartis de colesterol frente ao rótulo binário da variável alvo (target), avaliando visualmente a separabilidade dos dados.

Mensuração de Correlação Linear: Geração de um mapa de calor (heatmap) baseado no coeficiente de Pearson abrangendo os atributos numéricos centrais. O diagrama foi configurado com anotações explícitas dos coeficientes para avaliar a força e a direção das associações lineares diretas com o risco cardíaco.

### 4.3 Pré-processamento (`src/preprocessing/`)

Esta etapa descreve a engenharia de higienização automatizada desenvolvida no script clean_and_prepare_data(). O objetivo é mitigar as inconsistências detectadas na EDA e converter o arranjo bruto em uma matriz matemática perfeitamente estruturada na pasta data/processed/heart_disease_cleaned.csv. O pipeline executa sequencialmente as seguintes operações:

Correção de Anomalias e Imputação: Remoção automática de linhas inteiramente nulas e substituição cirúrgica dos valores artificiais de colesterol igual a zero por pd.NA. Posteriormente, o script aplica um laço de repetição que preenche todos os dados ausentes utilizando a mediana populacional calculada de cada atributo.

Binarização do Alvo (Target): Mapeamento da variável de saída — originalmente distribuída em uma escala de gravidade clínica de 0 a 4 — para uma representação estritamente binária ($0$ para ausência e $1$ para presença de risco cardíaco), adequando o problema para classificadores supervisionados.

Codificação Categórica Neutra: Conversão das colunas nominais qualitativas (cp, restecg, slope, thal) em strings para aplicação da técnica de One-Hot Encoding via pd.get_dummies(). O parâmetro drop_first=True é acionado para remover a primeira coluna derivada de cada categoria, eliminando redundâncias matemáticas (multicolinearidade).

Normalização Min-Max: Ajuste de escala proporcional para os atributos contínuos (age, trestbps, chol, thalach, oldpeak), comprimindo seus valores estritamente dentro do intervalo estável de $[0.0, 1.0]$ para otimizar a velocidade de convergência dos futuros modelos. As variáveis booleanas geradas são salvas explicitamente como inteiros binários ($0$ ou $1$).

### 4.4 Engenharia de Features (`src/features/`)

Esta etapa compreende o isolamento estatístico e a redução de dimensionalidade da base de dados limpa, executados de forma automatizada pelo script select_best_features(). O foco é garantir a parcimônia do modelo preditivo, alimentando a próxima camada apenas com os recursos de alto valor informativo, persistidos em heart_disease_features.csv. A rotina estrutura-se em:

Extração de Relevância Nativa: Instanciação e ajuste de um classificador Random Forest composto por 100 estimadores (árvores de decisão) acoplado a uma semente de inicialização fixa (random_state=42) para assegurar a total reprodutibilidade do experimento. O algoritmo calcula matematicamente a importância de cada recurso por meio do critério de Redução da Impureza de Gini (Mean Decrease in Impurity - MDI).

Aplicação de Limiar de Corte Otimizado: Ordenação decrescente e filtragem dos atributos a partir de um limiar de corte estatístico estabelecido criteriosamente em 0.015. Esta configuração permite descartar componentes puramente ruidosos ou de baixa variância informativa (como restecg_2 e slope_3).

Equilíbrio Clínico-Matemático: Redução da dimensão original de 18 atributos para 16 variáveis altamente preditivas. Este limiar específico garante o resgate automático de colunas vitais consagradas pela literatura médica da UCI (como o número de vasos coloridos ca e anomalias de cintilografia thal), blindando o conjunto contra o sobreajuste (overfitting).

### 4.5 Modelagem e Treinamento (`src/models/`)

Esta etapa envolve o treinamento dos modelos escolhidos. Para este objetivo, foram escolhidos os modelos Regressão Logística e Random Forest. A escolha se deu a partir da relação entre as variáveis de entrada (como colesterol, idade, pressão) e o risco cardíaco. A finalidade é investigar o comportamento do algoritmo com modelos lineares e não lineares. 

No caso da Regressão Logística, a relação é linear, ou seja, o aumento de variáveis como o colesterol e a pressão tem o mesmo impacto para o risco cardíaco, independentemente do contexto. Por exemplo: caso se constate que o aumento de uma determinada taxa do colesterol de um paciente aumenta o risco cardíaco em 5%, a Regressão Logística assume que esse aumento de 5% vale tanto para um jovem de 20 anos ativo quanto para um idoso de 75 anos sedentário. 

Já na Random Forest, a relação é não linear. Este algoritmo trabalha com árvores de decisão, que realizam análises mais complexas entre as variáveis. Na prática, se assume que um paciente pode até ter um colesterol alto, mas o risco cardíaco só aparecerá se ele tiver uma idade avançada e um tipo de dor no peito específica.


### 4.6 Avaliação, Resultados e Discussão (`src/evaluation/`)

Inicialmente, utilizou-se uma base pequena de Cleveland (303 linhas). Com ela, a Regressão Logística tinha vencido, com 80,33% de acurácia contra 75,41% da Random Forest. Isso se justifica porque a Random Forest não tinha dados suficientes e estava sofrendo de Overfitting (decorando a base). 

No entanto, após a ingestão das bases de dados dos hospitais de Cleveland, Hungria, Suíça e VA Long Beach, o pré-processamento correto e a engenharia de features, o cenário mudou. Agora o dataset aumentou para 920 pacientes, permitindo a geração correta das árvores de decisão definidas no código. Já com o processo de engenharia de features, foi possível tornar binárias colunas-chave como dor no peito. 

Esse tipo de abordagem é excelente para a Random Forest, ao contrário da Regressão Logística, que sofre com as múltiplas análises matemáticas necessárias. Além disso, há colunas como ca (número de vasos sanguíneos principais coloridos por fluoroscopia, que varia de 0 a 3) em que a quantidade de vasos sanguíneos não é linear, situação que otimiza o algoritmo de Random Forest. No fim, a Random Forest teve 82,07% de acurácia, contra 79,35% da Regressão Logística.

## 5. Testes

Com o objetivo de garantir confiabilidade e reprodutibilidade do projeto, implementou-se testes automáticos com a aplicação do framework `pytest`.

Os testes criados validam desde a entrada dos dados brutos até a exportação do modelo treinado. Dessa forma, cada arquivo test faz referência a um das camadas anteriores (features, ingestion, models e preprocessing), testanto cenários de falhas e sucesso de suas funções.

### Pré-requisitos

Antes de executar os testes, certifique-se de que o ambiente virtual (`.venv`) está ativado e com as dependências instaladas (como mencionado anteriormente). Se necessário, instale os pacotes de teste executando:

````bash
pip install pytest pandas scikit-learn joblib ```
````

---

### Como executar:

Certifique-se de que seu terminal está posicionado na raiz da pasta do projeto (projeto-inteligencia-computacional) e execute o comando abaixo:

```bash
python -m pytest
```

Para uma execução detalhada que mostra o nome de cada teste que foi aprovado, utilize o modo verboso:

```bash
python -m pytest -v
```

### Estrutura e Escopo dos Testes

Os testes estão localizados na pasta tests/ e cobrem os seguintes cenários:
test_ingestion.py: Garante que os dados dos 4 hospitais estão sendo carregados corretamente e unificados sem perda de integridade.

#### test_clean_data.py:

Valida o pipeline de limpeza, verificando o tratamento de valores nulos e se a binarização da variável target ocorreu com sucesso.

#### test_build_features.py:

Verifica a seleção de variáveis importantes utilizando o algoritmo Random Forest, garantindo que o formato final dos dados esteja correto.

#### test_train.py:

Avalia o treinamento dos modelos e certifica que os arquivos finais .joblib são gerados e salvos corretamente na pasta de artefatos.
