#!/usr/bin/env python
# coding: utf-8

# ### Esse notebook contém o resultado do processamento do script `src/handle_database.py`. Ou seja, a agregação das bases de Estabelecimentos, Recursos Físicos, Recursos Humanos e Equipes.

# In[1]:


import pandas as pd


# In[11]:


df_est = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/estabelecimentos.csv',
    index_col=0
)

df_rf = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/recursos_fisicos.csv',
    index_col=0
)

df_rh = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/recursos_humanos.csv',
    index_col=0
)

df_equipes = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/equipes.csv',
    index_col=0
)


# In[8]:


print(df_est.shape)
df_est.head()


# In[9]:


print(df_rf.shape)
df_rf.head()


# In[10]:


print(df_rh.shape)
df_rh.head()b


# In[12]:


print(df_equipes.shape)
df_equipes.head()

