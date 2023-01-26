#!/usr/bin/env python
# coding: utf-8

# ### Objetivo

# Este notebook possui uma **análise descritiva** da base de Câncer de Mama (Quimioterapia) junto com as bases agregadas por municípios usando o script `src/handle_database.py`. Este script agrega os dados de Estabelecimentos, Equipes, Recursos Físicos e Recursos Humanos. Em seguida, nós juntamos a base de Câncer de Mama com essas bases agregadas. A ideia principal é avaliar o contexto no qual o paciente está inserido. Por exemplo, uma mulher de 40 anos que mora em Araraquara e não teve diagnóstico tardio possui quais recursos a disposição em sua cidade? Essa e outras perguntas iremos responder neste notebook.
# 
# No final do notebook, vamos gerar uma base pronta para ser utilizada em uma **análise preditiva**, onde usaremos aplicaremos um modelo de Regressão Logística (que não ficou bom) e um modelo LightGBM (usado como referência na apresentação).
# 
# Ambas as análises anteriores nos deram insights para construir uma **análise prescritiva** a qual será organizada no ppt em `report/`.

# ### Libs

# In[1]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[3]:


sns.set()


# In[4]:


palette = sns.color_palette("ch:2.5,-.1,dark=.1")


# ### Gera amostra aleatória

# A quantidade de dados de Câncer de Mama possui quase 8 milhões de registros. Para simplificar a análise e o tempo de execução das células desde notebook, vamos selecionar uma amostra aleatória de 400 mil registros dessa base e comparar a proporção de estadiamento dessa amostra.
# 
# ***Obs: As próximas 5 células precisam ser executadas apenas 1 vez. Após isso, basta usar o sample `Mama_Quimioterapia_SIA-SUS_sample.csv`

# In[5]:


df_qui_total = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/mama/Mama_Quimioterapia_SIA-SUS.csv',
    encoding='ISO-8859-1',
    index_col=0,
    low_memory=True
)


# In[7]:


df_qui_total['tardio_base_original'] = df_qui_total['AQ_ESTADI'].apply(lambda x: 1 if x in [3, 4] else 0)


# In[8]:


print(f"Proporção de diagnóstico tardio para câncer de Mama: {100 * df_qui_total.tardio_base_original.mean()}%")


# In[9]:


del df_qui_total['tardio_base_original']


# In[9]:


df_qui_total.sample(400000).to_csv(
    '../data/Banco_Datathon/Banco_Datathon/mama/Mama_Quimioterapia_SIA-SUS_sample.csv'
)


# ### Carrega dataset do Câncer de Mama (Quimioterapia)

# In[10]:


df_qui = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/mama/Mama_Quimioterapia_SIA-SUS_sample.csv',
    encoding='ISO-8859-1',
    index_col=0,
    low_memory=True
)


# ### Carrega bases agregadas (Recursos Humanos + Recursos Físicos + Estabelecimentos)

# In[11]:


df_rh = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/recursos_humanos.csv',
    index_col=0
)

df_rf = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/recursos_fisicos.csv',
    index_col=0
)

df_est = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/estabelecimentos.csv',
    index_col=0
)

df_equipes = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/equipes.csv',
    index_col=0
)


# In[12]:


print(df_qui.shape)
df_qui.head()


# In[13]:


print(df_rh.shape)
print(df_rf.shape)
print(df_est.shape)
print(df_equipes.shape)


# ### Nulos

# In[14]:


null_count = df_qui.isna().sum()
null_percentage = round(100 * (null_count / len(df_qui)), 3)
null_tbl = pd.DataFrame(
    data=[null_count, null_percentage, df_qui.dtypes],
    index=['null_count', 'null_percentage', 'types']).T.sort_values(ascending=False, by='null_count'
)


# In[15]:


null_tbl


# Vemos que existem algumas variáveis que possuem muitos valores nulos. Etnia por exemplo possui 99% de nulos. Vamos eliminar as variáveis que possuem >0% de nulos, com exceção da `AP_UFNACIO` (<1% de nulos).

# In[16]:


features_to_drop = null_tbl.loc[null_tbl.null_percentage > 0.0015].index.tolist()


# In[17]:


features_to_drop


# In[18]:


df_qui.drop(columns=features_to_drop, inplace=True)


# ### Construção da variável resposta (Estadiamento)

# In[19]:


df_qui.AQ_ESTADI.value_counts()


# A variável resposta pode variar de 0 a 4, onde 0, 1 e 2 corresponde aos estágios menos avançados da doença e 3, 4 corresponde aos estágios mais avançados da doença. Vamos criar uma coluna binária contendo esta informação

# In[20]:


df_qui['tardio'] = df_qui['AQ_ESTADI'].apply(lambda x: 1 if x in [3, 4] else 0)


# In[21]:


print(f"Proporção de diagnóstico tardio para câncer de Mama: {100 * df_qui.tardio.mean()}%")


# A proporção de estadiamento para essa amostragem (~400 Mil) ficou muito próxima da proporção de estadiamento para a base total (~8 M). Existem outras técnicas mais sofisticadas de sampling, porém vamos seguir com essa abordagem simples para este problema.

# ### Filtro de variáveis

# Vamos olhar para cada uma das variáveis a fim de manter apenas as variáveis que possam ser preditoras de diagnótico tardio ou nos ajudar a encontrar recomendações acionáveis para a Abrale.

# In[22]:


print(df_qui.shape)
df_qui.head()


# In[23]:


features_to_remove = [
    'AQ_ESTADI', # Já foi criada a variável resposta com base nessa coluna
    'AQ_LINFIN', # Indica que já houve diagnóstico tardio
    'AQ_ESQU_P1', # Indica que já houve diagnóstico tardio
    'AQ_TOTMPL', # Indica que já houve diagnóstico tardio
    'AQ_TOTMAU', # Indica que já houve diagnóstico tardio
    'AP_MOTSAI', # Indica que já houve diagnóstico tardio
    'AP_ENCERR', # Indica que já houve diagnóstico tardio
    'AP_PERMAN',# Indica que já houve diagnóstico tardio    
    'AP_GESTAO', # Variável de identificação
    'AP_CODUNI', # Variável de identificação
    'AP_AUTORIZ', # Variável de identificação
    'AP_CIDSEC', # Variável de identificação
    'AP_PRIPAL', # Variável de identificação
    'AP_CNSPCN', # Variável de identificação
    'AP_COIDADE', # Variável de identificação
    'AP_CEPPCN', # Variável de identificação
    'AP_CODEMI', # Variável de identificação
    'AP_CEPPCN', # Variável de identificação
    'AP_TRANSF', # As duas classes não apresentam diferença estatisticamente significativa
    'AP_TPATEN', # Todas as linhas são iguais
    'AP_CIDPRI', # Todas as linhas são iguais
    'AP_MVM', # Data
    'AP_CMP', # Data
    'AP_DTINIC', # Data
    'AP_DTFIM', # Data
    'AQ_DTINTR', # Data
    'AP_OBITO', # Indica que já houve diagnóstico tardio
    'AP_CONDIC' # Indica que já houve diagnóstico tardio
]


# In[24]:


df_qui.drop(columns=features_to_remove, inplace=True)


# In[25]:


print(df_qui.shape)
df_qui.head()


# | CAMPO     	| DESCRIÇÃO                                                               	|
# |-----------	|-------------------------------------------------------------------------	|
# | AP_UFMUN  	| Código da Unidade da Federação + Código do Município do Estabelecimento 	|
# | AP_MUNPCN 	| Código da UF + Código do Município de Residência do paciente            	|

# Os dois campos acima podem ser usados para vincularmos as bases de Municípios (Recursos Humanos + Recursos Físicos + Estabelecimentos). Como o dataset de Mama apresenta uma linha por **paciente**, vamos usar o campo `AP_MUNPCN` para realizar este vínculo. Dessa forma, teremos mais dados sobre a infraestrutura na qual o paciente está inserido. Em seguida, iremos testar a influência do contexto no diagnóstico tardio dos pacientes.

# ### Agregação da base de Câncer de Mama com Recursos Humanos + Recursos Físicos + Estabelecimentos + Equipes

# In[26]:


df_rh.rename(columns={'cod_mun': 'AP_MUNPCN'}, inplace=True)
df_rf.rename(columns={'cod_mun': 'AP_MUNPCN'}, inplace=True)
df_est.rename(columns={'cod_mun': 'AP_MUNPCN'}, inplace=True)
df_equipes.rename(columns={'cod_mun': 'AP_MUNPCN'}, inplace=True)


# In[27]:


df = pd.merge(df_qui, df_rh, how='left', on='AP_MUNPCN')
df = pd.merge(df, df_rf, how='left', on='AP_MUNPCN', suffixes=('', '_rf'))
df = pd.merge(df, df_est, how='left', on='AP_MUNPCN', suffixes=('', '_est'))
df = pd.merge(df, df_equipes, how='left', on='AP_MUNPCN', suffixes=('', '_equipes'))
df.drop(columns=['nome_mun_rf', 'nome_mun_est', 'nome_mun_equipes'], inplace=True)


# In[28]:


print(df.shape)
df.head()


# Este dataset será o dataset a ser analisado nas próximas células deste notebook.

# ### Análise Exploratória

# ### Função para plot de distribuições de variáveis contínuas

# In[29]:


def plot_distplot(df, feature, title, target, target_names, target_colors, log10=False):
    df = df.copy()
    feature_name = feature
    positive_target = df[target] == 1
    negative_target = df[target] == 0

    if log10:
        feature_name = f'log_{feature}'
        df[feature] = df[feature].replace(0, 1)
        df[feature] = df[feature].astype(float)
        df[feature_name] = np.log10(df[feature])

    sns.distplot(df.loc[positive_target, feature_name], hist=False, label='1', color=target_colors[0])
    sns.distplot(df.loc[negative_target, feature_name], hist=False, label='0', color=target_colors[1])
    _ = plt.legend(target_names)
    _ = plt.title(title)


# #### `AP_SEXO`

# In[30]:


sns.catplot(
    y='tardio', 
    x='AP_SEXO', 
    kind='bar', 
    data=df, 
    height=6,
    palette=palette
)

plt.title('Proporção de diagnóstico tardio vs Sexo');


# Vemos que o diagnóstico tardio de câncer de Mama é mais frequente nos homens do que nas mulheres. Os homens representam menos de 1% dos casos de Câncer de Mama dessa amostragem de dados.

# In[31]:


df.AP_SEXO.value_counts(normalize=True) * 100


# #### `AP_UFDIF`

# In[32]:


g = sns.catplot(
    y='tardio', 
    x='AP_UFDIF', 
    kind='bar', 
    data=df, 
    height=6,
    color='Greens_r',
    palette=palette
)
plt.title('Proporção de diagnóstico tardio vs UF de residência do paciente é diferente da UF de localização do estabelecimento');


# In[33]:


df.AP_UFDIF.value_counts(normalize=True) * 100


# Quanto a UF de residência do paciente, vemos que a proporção de diagnóstico tardio é maior. Este caso ocorre para cerca de 2% do dataset.

# #### `AP_MNDIF`

# In[34]:


sns.catplot(
    y='tardio', 
    x='AP_MNDIF', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Indica se o município de residência do paciente é diferente do município de localização do estabelecimento (N = não, S = sim)');


# Dado que o intervalo de confiança está muito alto para a classe 0, não vamos usar essa variável pois há apresenta diferença estatisticamente relevante entre as duas distribuições.

# #### `AP_TIPPRE`

# In[35]:


df.AP_TIPPRE.unique()


# In[36]:


sns.catplot(
    y='tardio', 
    x='AP_TIPPRE', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1.5,
    color='Greens_r',
    palette=palette
)

plt.title('Tipo de Prestador');


# #### `AP_MN_IND`

# In[37]:


sns.catplot(
    y='tardio', 
    x='AP_MN_IND', 
    kind='bar', 
    data=df, 
    height=5.5,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Estabelecimento Mantido / Individual');


# #### `AP_RACACOR`

# In[38]:


sns.catplot(
    y='tardio', 
    x='AP_RACACOR', 
    kind='bar', 
    data=df, 
    height=5.5,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Raça/Cor do paciente: 01 - Branca, 02 - Preta, 03 - Parda, 04 - Amarela, 05 - Indígena, 99 - Sem informação');


# In[39]:


df.AP_RACACOR.value_counts(normalize=True) * 100


# Indígena representa uma parcela bem pequena do dataset, por isso o intervalo de confiança ficou bastante alto comparado com as outras etnias.

# #### `AP_UFNACIO`

# In[40]:


df.AP_UFNACIO.value_counts().head(10)


# In[41]:


sns.catplot(
    y='tardio', 
    x='AP_UFNACIO', 
    kind='bar', 
    data=df, 
    height=6.5,
    aspect=2.5,
    color='Greens_r',
    palette=palette
)

plt.title('Nacionalidade do paciente');


# Essa variável parece não diferenciar diagnóstico tardio de diagnóstico não tardio. Portanto, não será usada no modelo.

# #### `AP_TPAPAC`

# In[42]:


sns.catplot(
    y='tardio', 
    x='AP_TPAPAC', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Indica se a APAC é 1 – inicial, 2 – continuidade, 3 – única');


# Essa variável parece não diferenciar diagnóstico tardio de diagnóstico não tardio. Portanto, não será usada no modelo.

# #### `AP_CATEND`

# In[43]:


sns.catplot(
    y='tardio', 
    x='AP_CATEND', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Caráter do Atendimento');


# Essa variável parece não diferenciar diagnóstico tardio de diagnóstico não tardio. Portanto, não será usada no modelo.

# #### `AP_VL_AP`

# In[45]:


plt.figure(figsize=(10,5))

plot_distplot(
    df=df,
    feature='AP_VL_AP',
    title='Valor Total da APAC Aprovado',
    target='tardio',
    target_names=['tardio', 'nao-tardio'],
    log10=True,
    target_colors=['darkgreen', 'lightgreen']
)


# Parece não haver diferença relevante considerando o valor da APAC.

# #### `AP_TPUPS`

# In[46]:


sns.catplot(
    y='tardio', 
    x='AP_TPUPS', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Tipo de Estabelecimento');


# #### `AP_ALTA`

# In[47]:


sns.catplot(
    y='tardio', 
    x='AP_ALTA', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Tipo de Estabelecimento');


# #### `AQ_TRANTE`

# In[48]:


sns.catplot(
    y='tardio', 
    x='AQ_TRANTE', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Tratamentos anteriores (S = Sim; N = Não)');


# #### `AQ_CONTTR`

# In[49]:


sns.catplot(
    y='tardio', 
    x='AQ_CONTTR', 
    kind='bar', 
    data=df, 
    height=6,
    aspect=1,
    color='Greens_r',
    palette=palette
)

plt.title('Continuidade do tratamento (S = Sim; N = Não)');


# ### Deteção de tumor ao longo dos anos (Valor absoluto)

# In[50]:


df[['AQ_DTIDEN', 'tardio']].head()


# In[51]:


df.AQ_DTIDEN = df.AQ_DTIDEN.astype(str)


# In[52]:


df['AQ_DTIDEN_ANO'] = df.AQ_DTIDEN.apply(lambda x: x[:4])


# In[53]:


df.AQ_DTIDEN_ANO = df.AQ_DTIDEN_ANO.astype(int)


# In[54]:


tbl_deteccao = df.loc[(df['AQ_DTIDEN_ANO'] > 2010) & (df['AQ_DTIDEN_ANO'] < 2019)][['AQ_DTIDEN_ANO', 'tardio']]


# In[55]:


tbl_deteccao = tbl_deteccao[['AQ_DTIDEN_ANO', 'tardio']].groupby(['AQ_DTIDEN_ANO']).sum().reset_index()


# In[56]:


tbl_deteccao['tardio_percentage'] = round(100 * (tbl_deteccao['tardio'] / tbl_deteccao.tardio.sum()), 2)


# In[57]:


tbl_deteccao


# In[58]:


plt.figure(figsize=(10,5))

ax = sns.lineplot(
    x="AQ_DTIDEN_ANO", 
    y="tardio", 
    data=tbl_deteccao,
    color='darkgreen'
)
ax.set_title("Evolução de detecção de estadiamento tardio ao longo dos anos")
ax.set_xlabel("Data de identificação patológica do caso")
ax.set_ylabel("Total de estadiamento tardio");


# ### Deteção de tumor ao longo dos anos (%)

# In[59]:


plt.figure(figsize=(10,5))

ax = sns.lineplot(
    x="AQ_DTIDEN_ANO", 
    y="tardio_percentage", 
    data=tbl_deteccao,
    color='darkgreen'
)
ax.set_title("Evolução de detecção de estadiamento tardio ao longo dos anos para o Câncer de Mama")
ax.set_xlabel("Ano de identificação patológica do caso")
ax.set_ylabel("% de estadiamento tardio");


# ### Idade vs Estadiamento

# In[60]:


df[['AP_NUIDADE', 'tardio']].head()


# In[61]:


plt.figure(figsize=(10,5))

plot_distplot(
    df=df,
    feature='AP_NUIDADE',
    title='Número da Idade',
    target='tardio',
    target_names=['tardio', 'nao-tardio'],
    log10=False,
    target_colors=['darkgreen', 'lightgreen']
)


# ### Função para comparar a evolução média dos recursos ao longo dos anos

# In[62]:


def gerar_tabela_recursos(recurso):
    nao_tardio = df.filter(regex=f'^{recurso}|tardio')[df['tardio'] == 0]
    tardio = df.filter(regex=f'^{recurso}|tardio')[df['tardio'] == 1]

    nao_tardio = nao_tardio.filter(regex=f'^{recurso}').mean()
    tardio = tardio.filter(regex=f'^{recurso}').mean()

    num = pd.concat([nao_tardio, tardio], axis=1).reset_index()

    num.rename(columns={'index': 'ano', 0: 'nao_tardio', 1: 'tardio'}, inplace=True)

    num['ano'] = num.ano.apply(lambda x: x[-4:])

    return num


# Este gráfico considera a média de recursos para todos os pacientes ao longo dos anos para casos de estadiamento tardio e não tardio. Dessa forma, conseguimos comparar as tendências de ambos os casos.

# ### Recursos Humanos - Evolução do número de enfermeiros por estadiamento

# In[63]:


num_enfermeiros = gerar_tabela_recursos('rh_enfermeiros')


# In[64]:


num_enfermeiros


# In[65]:


plt.figure(figsize=(10,5))

ax1 = sns.lineplot(x="ano", y="tardio", data=num_enfermeiros, color='darkgreen', label='Diagnóstico Tardio')
ax2 = sns.lineplot(x="ano", y="nao_tardio", data=num_enfermeiros, color='lightgreen', label='Diagnóstico não Tardio')
ax1.set_title("Evolução do número de enfermeiros nos municípios")
ax1.set_xlabel("Ano")
ax1.set_ylabel("Número de enfermeiros médio");


# ### Recursos Humanos - Evolução do número de médicos por estadiamento

# In[66]:


num_medicos = gerar_tabela_recursos('rh_medicos')


# In[67]:


num_medicos


# In[68]:


plt.figure(figsize=(10,5))

ax1 = sns.lineplot(x="ano", y="tardio", data=num_medicos, color='darkgreen', label='Diagnóstico Tardio')
ax2 = sns.lineplot(x="ano", y="nao_tardio", data=num_medicos, color='lightgreen', label='Diagnóstico não Tardio')
ax1.set_title("Evolução do número de médicos nos municípios")
ax1.set_xlabel("Ano")
ax1.set_ylabel("Número de médicos médio");


# ### Recursos Físicos - Evolução do número de leitos por Estadiamento

# In[69]:


num_leitos = gerar_tabela_recursos('rf_leitos')


# In[70]:


plt.figure(figsize=(10,5))

ax1 = sns.lineplot(x="ano", y="tardio", data=num_leitos, color='darkgreen', label='Diagnóstico Tardio')
ax2 = sns.lineplot(x="ano", y="nao_tardio", data=num_leitos, color='lightgreen', label='Diagnóstico não Tardio')
ax1.set_title("Evolução do número de leitos nos municípios")
ax1.set_xlabel("Ano")
ax1.set_ylabel("Número de leitos médio");


# ### Recursos Físicos - Evolução do número de mamógrafos por estadiamento

# In[71]:


num_mamografos = gerar_tabela_recursos('rf_mamografos')


# In[72]:


plt.figure(figsize=(10,5))

ax1 = sns.lineplot(x="ano", y="tardio", data=num_mamografos, color='darkgreen', label='Diagnóstico Tardio')
ax2 = sns.lineplot(x="ano", y="nao_tardio", data=num_mamografos, color='lightgreen', label='Diagnóstico não Tardio')
ax1.set_title("Evolução do número de mamógrafos")
ax1.set_xlabel("Ano")
ax1.set_ylabel("Número de mamógrafos médio");


# ### Recursos Físicos - Evolução do número de Raio-X por estadiamento

# In[73]:


num_raio_x = gerar_tabela_recursos('rf_raios_x')


# In[74]:


num_raio_x


# In[75]:


plt.figure(figsize=(10,5))

ax1 = sns.lineplot(x="ano", y="tardio", data=num_raio_x, color='darkgreen', label='Diagnóstico Tardio')
ax2 = sns.lineplot(x="ano", y="nao_tardio", data=num_raio_x, color='lightgreen', label='Diagnóstico não Tardio')
ax1.set_title("Evolução do número de Raio-X nos municípios")
ax1.set_xlabel("Ano")
ax1.set_ylabel("Número de Raio-X médio");


# ### Estabelecimentos por estadiamento

# In[76]:


est = df.filter(regex='(^est_|hosp_|ubs_|dianose|hop_).*2018|tardio').groupby(['tardio']).mean()


# In[77]:


est = est.T.reset_index()


# In[78]:


est = est.melt(id_vars=['index'])


# In[79]:


est.rename(columns={'index': 'Tipo de Estabelecimento', 'tardio': 'Diagnóstico Tardio', 'value': 'Média'}, inplace=True)


# In[80]:


ax = sns.catplot(
    x="Diagnóstico Tardio", 
    y="Média", 
    hue="Tipo de Estabelecimento", 
    kind="bar", 
    height=4.5,
    aspect=2,
    data=est,
    palette=palette
)

plt.title('Quantidade de estabelecimentos por estadiamento em 2018');


# ### Recursos físicos por estadiamento

# In[81]:


rf = df.filter(regex='^rh_.*2018|tardio').groupby(['tardio']).mean()
rf = rf.T.reset_index()
rf = rf.melt(id_vars=['index'])
rf.rename(columns={'index': 'Recursos Humanos', 'tardio': 'Diagnóstico Tardio', 'value': 'Média'}, inplace=True)


# In[82]:


ax = sns.catplot(
    x="Diagnóstico Tardio", 
    y="Média", 
    hue="Recursos Humanos", 
    kind="bar", 
    height=4.5,
    aspect=2,
    data=rf,
    palette=palette
)

plt.title('Recursos humanos por estadiamento em 2018');


# ### Equipes por estadiamento 

# In[83]:


eq = df.filter(regex='^equipes_.*2018|tardio').groupby(['tardio']).mean()
eq = eq.T.reset_index()
eq = eq.melt(id_vars=['index'])
eq.rename(columns={'index': 'Equipes', 'tardio': 'Diagnóstico Tardio', 'value': 'Média'}, inplace=True)


# In[84]:


ax = sns.catplot(
    x="Diagnóstico Tardio", 
    y="Média", 
    hue="Equipes", 
    kind="bar", 
    height=4.5,
    aspect=2,
    data=eq,
    palette=palette
)

plt.title('Equipes por estadiamento em 2018');


# ### Recursos Físicos por estadiamento (Tirando o número de Leitos)

# In[85]:


rf = df.filter(regex='^rf_.*2018|tardio').groupby(['tardio']).mean()
rf.drop(columns='rf_leitos_2018', inplace=True)
rf = rf.T.reset_index()
rf = rf.melt(id_vars=['index'])
rf.rename(columns={'index': 'Recursos Físicos', 'tardio': 'Diagnóstico Tardio', 'value': 'Média'}, inplace=True)


# In[86]:


ax = sns.catplot(
    x="Diagnóstico Tardio", 
    y="Média", 
    hue="Recursos Físicos", 
    kind="bar", 
    height=7,
    aspect=2,
    data=rf,
    palette=palette
)

plt.title('Recursos físicos por estadiamento em 2018');


# In[87]:


df.head()


# ### Função para gerar Score de RF, RH, Estabelecimentos e Equipes

# O Score é basicamente uma normalização usando o MinMaxScaler do `sklearn`. Ou seja, o município que tiver a maior quantidade daquele recurso recebe a nota 1 e o que tiver a menor quantidade daquele recurso recebe a nota 0.

# In[88]:


def gerar_score(df, tipo):
    df = df.filter(regex=f'^{tipo}|tardio')
    scaler = MinMaxScaler()
    scaler.fit(df[df.columns.tolist()])
    df[df.columns.tolist()] = scaler.transform(df[df.columns.tolist()])
    df['nota_' + tipo] = df.mean(axis=1)
    
    return df


# ### Score de Recursos Físicos

# In[89]:


score_rf = gerar_score(df, 'rf_')


# In[90]:


plt.figure(figsize=(12,6))

plot_distplot(
    df=score_rf,
    feature='nota_rf_',
    title='Nota de Recursos Físicos',
    target='tardio',
    target_names=['tardio', 'nao-tardio'],
    log10=False,
    target_colors=['darkgreen', 'lightgreen']
)


# ### Score de Recursos Humanos

# In[91]:


score_rh = gerar_score(df, 'rh_')


# In[92]:


plt.figure(figsize=(12,6))

plot_distplot(
    df=score_rh,
    feature='nota_rh_',
    title='Score de Recursos Humanos',
    target='tardio',
    target_names=['tardio', 'nao-tardio'],
    log10=False,
    target_colors=['darkgreen', 'lightgreen']
)


# ### Score de Estabelecimentos

# In[93]:


score_est = gerar_score(df, 'est_|hosp_|hop_|ubs_|diagnose_|tardio_')


# In[94]:


plt.figure(figsize=(12,6))

plot_distplot(
    df=score_est,
    feature='nota_est_|hosp_|hop_|ubs_|diagnose_|tardio_',
    title='Score de Recursos Estabelecimentos',
    target='tardio',
    target_names=['tardio', 'nao-tardio'],
    log10=False,
    target_colors=['darkgreen', 'lightgreen']
)


# ### Score de Equipes

# In[95]:


score_equipes = gerar_score(df, 'equipes_')


# In[96]:


plt.figure(figsize=(12,6))

plot_distplot(
    df=score_equipes,
    feature='nota_equipes_',
    title='Score de Equipes',
    target='tardio',
    target_names=['tardio', 'nao-tardio'],
    log10=False,
    target_colors=['darkgreen', 'lightgreen']
)


# ### Quais tipos de estabelecimentos são mais importantes para o estadiamento?

# | Prefixo            	| Estabelecimento                                                      	|
# |--------------------	|----------------------------------------------------------------------	|
# | est_cli_amb_esp_    	| Estabelecimentos - Clínicas Ambulatórios Especializados              	|
# | hosp_esp_           	| Estabelecimentos - Hospital Especializado                            	|
# | hop_geral_          	| Estabelecimentos - Hospital Geral                                    	|
# | ubs_                	| Estabelecimentos - Unidade Básica de Saúde                           	|
# | diagnose_e_terapia_ 	| Estabelecimentos - Unidade de Serviço de Apoio ao Diagnose e Terapia 	|

# In[97]:


df.filter(regex='(^est|hosp|hop|ubs|diagnose|tardio).*2017|tardio').head()


# In[98]:


est_score = score_est.filter(regex='(^est|hosp|hop|ubs|diagnose|tardio).*2017|tardio').groupby(['tardio']).mean()
est_score = est_score.T.reset_index()
est_score = est_score.melt(id_vars=['index'])
est_score.rename(columns={'index': 'Tipo de Estabelecimento', 'tardio': 'Diagnóstico Tardio', 'value': 'Média'}, inplace=True)


# In[99]:


ax = sns.catplot(
    x="Diagnóstico Tardio", 
    y="Média", 
    hue="Tipo de Estabelecimento", 
    kind="bar", 
    height=4.5,
    aspect=2,
    data=est_score,
    palette=palette
)

plt.title('Quantidade de estabelecimentos por estadiamento em 2018');


# ### Preparação das variáveis para modelagem

# In[100]:


print(df.shape)
df.head()


# Seleciona variáveis de infraestrutura (Recursos físicos, equipes, estabelecimentos, recursos humanos) na região a qual o paciente mora.

# In[101]:


infra_features = df.filter(regex='^rf_|equipes_|rh_|est_|hosp_|ubs_|dianose|hop_').columns.tolist()


# Seleciona outras características que foram consideradas mais importantes nas análises das células anteriores

# In[105]:


mama_features = ['AP_TPUPS', 'AP_TIPPRE', 'AP_MN_IND', 'AP_NUIDADE', 
                 'AP_SEXO', 'AP_RACACOR', 'AP_UFDIF', 'AQ_TRANTE', 'AQ_CONTTR']


# In[106]:


features_for_the_model = infra_features + mama_features


# In[107]:


df_model = df[features_for_the_model]


# In[108]:


df_model.head()


# ### Salva base Agregada (Mama + Recursos Humanos + Recursos Físicos + Estabelecimentos + Equipes)

# In[ ]:


df_model.to_csv('../data/Banco_Datathon/Banco_Datathon/processed/mama.csv')

