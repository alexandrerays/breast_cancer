#!/usr/bin/env python
# coding: utf-8

# ### Libs

# In[1]:


import shap
import joblib
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn import metrics


# In[2]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# ### Variáveis categóricas

# In[3]:


def dtype_category():
    return {
        'AP_TPUPS': 'category',
        'AP_TIPPRE': 'category',
        'AP_MN_IND': 'category',
        'AP_SEXO': 'category',
        'AP_RACACOR': 'category',
        'AP_UFDIF': 'bool',
        'AQ_TRANTE': 'category',
        'AQ_CONTTR': 'category'
    }


# ### Carrega datasets

# In[4]:


df_train = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/train_v4.csv', 
    index_col=0,
    dtype=dtype_category()
)
df_test = pd.read_csv(
    '../data/Banco_Datathon/Banco_Datathon/processed/test_v4.csv', 
    index_col=0,
    dtype=dtype_category()
)
model = joblib.load(f"../model/model_lgbm_v4.pkl")


# ### Modelo no formato pkl

# In[5]:


model


# In[6]:


print(df_train.shape)
df_train.head()


# ### Separa variáveis preditoras das variável resposta

# In[7]:


def split_between_X_and_y(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    
    return X, y


# In[8]:


X_train, y_train = split_between_X_and_y(df_train, target='tardio')
X_test, y_test = split_between_X_and_y(df_test, target='tardio')


# ### Shap - Interpretando o modelo

# In[26]:


explainer = shap.TreeExplainer(model)


# In[29]:


shap_values = explainer.shap_values(X_train, tree_limit=1200)


# In[30]:


shap.summary_plot(shap_values, features=X_train, plot_type='bar', max_display=15, color='#117A65')


# In[31]:


shap.summary_plot(shap_values, X_train, plot_type='dot', show=True)


# - `AP_NUIDADE` : A idade é a variável mais importante | Quanto maior a idade, menos diagnóstico tardio como visto aqui e em `notebooks/eda_mama`
# - `AQ_CONTTR` : Continuidade do tratamento (S = Sim; N = Não) | Se sim, menos diagnóstico tardio como visto aqui e em `notebooks/eda_mama`
# - `AP_MN_IND` : Estabelecimento Mantido / Individual | Se Individual, menos diagnóstico tardio como visto em `notebooks/eda_mama`
# - `AQ_TRANTE` : Tratamentos anteriores (S = Sim; N = Não) | Se sim, menos diagnóstico tardio como visto em `notebooks/eda_mama`
# - `ubs_max_2014` : Número máximo de UBSs no município de residência do paciente em 2014 | Quanto maior, menos diagnóstico tardio

# ### Fontes

# - Implementação do modelo LightGBM. Este modelo é baseado é um Gradient Boosting baseado em árvores de decisão. Fonte: https://lightgbm.readthedocs.io/en/latest/index.html
# 
# - Interpretabilidade usando o Framework SHAP, baseado no Shapley Value da Teoria dos Jogos cooperativa. Fonte: https://arxiv.org/abs/1905.04610
