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
│   ├── raw/                 # Dados brutos originais da UCI (.data) e arquivo unificado
│   └── processed/           # Dados limpos, pré-processados e finalizados (.csv)
├── notebooks/               # Notebooks para Análise Exploratória (EDA)
├── src/
│   ├── ingestion/           # Scripts de coleta e consolidação inicial de dados
│   ├── preprocessing/       # Rotinas de limpeza avançada e tratamento de nulos
│   ├── features/            # Engenharia de features e transformações matemáticas
│   ├── models/              # Treinamento de algoritmos de Machine Learning
│   └── evaluation/          # Métricas de validação dos modelos
├── metrics/                 # Gráficos e resultados de avaliação salvos
├── models_saved/            # Modelos treinados persistidos (joblib/pickle)
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação principal

---

## 3. Configuração do Ambiente e Execução

### 3.1 Pré-requisitos e Bibliotecas
* Python 3.10 ou superior.
* Gerenciador de versão: Git.
* Bibliotecas principais de manipulação: `pandas`, `numpy`, `os`.
* Bibliotecas principais de Machine Learning e Visualização: `scikit-learn`, `matplotlib`, `seaborn`.
*Biblioteca para execução de testes: `pytest`

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
*Consolidação: Concatenação das bases resultando no arquivo unificado `heart_disease_raw.csv` alocado na pasta `data/raw/`.

### 4.2 Análise Exploratória de Dados - EDA (`notebooks/`)

### 4.3 ré-processamento (`src/preprocessing/`)

### 4.4 Engenharia de Features (`src/features/`)

### 4.5 Modelagem e Treinamento (`src/models/`)

### 4.6 Avaliação, Resultados e Discussão (`src/evaluation/`)

## 5. Tests

Com o objetivo de garantir confiabilidade e reprodutibilidade do projeto, implementou-se testes automáticos com a aplicação do framework `pytest`.

Os testes criados validam desde a entrada dos dados brutos até a exportação do modelo treinado. Dessa forma, cada arquivo test faz referência a um das camadas anteriores (features, ingestion, models e preprocessing), testanto cenários de falhas e sucesso de suas funções.

### Pré-requisitos

Antes de executar os testes, certifique-se de que o ambiente virtual (`.venv`) está ativado e com as dependências instaladas (como mencionado anteriormente). Se necessário, instale os pacotes de teste executando:

```bash
pip install pytest pandas scikit-learn joblib ```
```
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