import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuração de estilo dos gráficos
# Define o tema visual padrão e as dimensões globais para as ilustrações gráficas
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# 2. Carregamento do conjunto de dados
# Realiza a leitura da base de dados bruta para a geração dos diagnósticos visuais
df = pd.read_csv("data/raw/heart_disease_raw.csv")

# ==========================================
# DIAGRAMA 1: HISTOGRAMAS E DISTRIBUIÇÕES
# ==========================================
# Configura uma grade de subtramas para analisar o comportamento estatístico da idade e do colesterol
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gera o histograma acoplado à estimativa de densidade de kernel (KDE) para o atributo Idade
sns.histplot(data=df, x='age', kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribuição da Idade dos Pacientes')
axes[0].set_xlabel('Idade')
axes[0].set_ylabel('Frequência')

# Construções visuais análogas aplicadas para mapear a dispersão e assimetria do Colesterol
sns.histplot(data=df, x='chol', kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribuição do Colesterol')
axes[1].set_xlabel('Colesterol (mg/dl)')
axes[1].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

# ==========================================
# DIAGRAMA 2: DISTRIBUIÇÃO POR CATEGORIA (Boxplot)
# ==========================================
# Constrói o diagrama de caixa para confrontar os quartis de colesterol sob a ótica da variável alvo
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='target', y='chol', palette='Set2')
plt.title('Distribuição do Colesterol por Presença de Risco Cardíaco')
plt.xlabel('Risco Cardíaco (0 = Não, 1 = Sim)')
plt.ylabel('Colesterol')
plt.show()

# ==========================================
# DIAGRAMA 3: MATRIZ DE CORRELAÇÃO (Heatmap)
# ==========================================
# Investiga a força e a direção das associações lineares existentes entre os atributos numéricos
plt.figure(figsize=(12, 10))

# Filtra as variáveis preditivas de maior relevância a fim de mitigar o excesso de informações na matriz
colunas_principais = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'target']
matriz_corr = df[colunas_principais].corr()

# Renderiza o mapa de calor contendo os coeficientes de correlação de Pearson anotados de forma explícita
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis Principais')
plt.show()